## ReKep deploys in ManiSkill guidance

1. pip install pinocchio in anaconda environment (your_conda_env/lib/python3.10/site-packages/sapien/warpper/pinocchio_model.py {If you did'n install pinocchio, this script will use pre-build pinocchio dynamic lib and we cannot edit it or need to re-build pinocchio})

2. Change compute_inverse_kinematics function in your_conda_env/lib/python3.10/site-packages/sapien/warpper/pinocchio_model.py

   ```python
   def compute_inverse_kinematics(
               self,
               link_index,
               pose,
               initial_qpos=None,
               active_qmask=None,
               eps=1e-4,
               max_iterations=1000,
               dt=0.1,
               damp=1e-6,
               iter_flag=False
           ):
               """
               Compute inverse kinematics with CLIK algorithm.
               Details see https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
               Also see bug fix from https://github.com/stack-of-tasks/pinocchio/pull/1963/files
               Args:
                   link_index: index of the link
                   pose: target pose of the link in articulation base frame
                   initial_qpos: initial qpos to start CLIK
                   active_qmask: dof sized integer array, 1 to indicate active joints and 0 for inactive joints, default to all 1s
                   max_iterations: number of iterations steps
                   dt: iteration step "speed"
                   damp: iteration step "damping"
               Returns:
                   result: qpos from IK
                   success: whether IK is successful
                   error: se3 norm error
               """
               assert 0 <= link_index < len(self.link_id_to_frame_index)
               if initial_qpos is None:
                   q = pinocchio.neutral(self.model)
               else:
                   q = self.q_s2p(initial_qpos)
   
               if active_qmask is None:
                   mask = np.ones(self.model.nv)
               else:
                   mask = np.array(active_qmask)[self.index_p2s]
   
               mask = np.diag(mask)
   
               frame = int(self.link_id_to_frame_index[link_index])
               joint = self.model.frames[frame].parent
   
               T = pose.to_transformation_matrix()
               l2w = pinocchio.SE3()
               l2w.translation[:] = T[:3, 3]
               l2w.rotation[:] = T[:3, :3]
   
               l2j = self.model.frames[frame].placement
               oMdes = l2w * l2j.inverse()
   
               best_error = 1e10
               best_q = np.array(q)
               
               iter = -1
   
               for i in range(max_iterations):
                   pinocchio.forwardKinematics(self.model, self.data, q)
                   iMd = self.data.oMi[joint].actInv(oMdes)
                   err = pinocchio.log6(iMd).vector
                   err_norm = np.linalg.norm(err)
                   if err_norm < best_error:
                       best_error = err_norm
                       best_q = q
   
                   if err_norm < eps:
                       iter = i + 1
                       success = True
                       break
   
                   J = pinocchio.computeJointJacobian(self.model, self.data, q, joint)
                   Jlog = pinocchio.Jlog6(iMd.inverse())
                   J = -Jlog @ J
                   J = J @ mask
   
                   JJt = J @ J.T
                   JJt[np.diag_indices_from(JJt)] += damp
   
                   v = -(J.T @ np.linalg.solve(JJt, err))
                   q = pinocchio.integrate(self.model, q, v * dt)
               else:
                   success = False
               if iter_flag:
                   return self.q_p2s(best_q), success, best_error, iter
               else:
                   return self.q_p2s(best_q), success, best_error
   ```

3. Change compute_ik function in you_conda_env/lib/python3.10/site-packages/mani_skill/agents/controllers/utils/kinematics.py (line 124)

   ```python
   def compute_ik(
           self,
           target_pose: Pose,
           q0: torch.Tensor,
           pos_only: bool = False,
           action=None,
           use_delta_ik_solver: bool = False,
           iter_flag=False
       ):
           """Given a target pose, via inverse kinematics compute the target joint positions that will achieve the target pose
   
           Args:
               target_pose (Pose): target pose of the end effector in the world frame. note this is not relative to the robot base frame!
               q0 (torch.Tensor): initial joint positions of every active joint in the articulation
               pos_only (bool): if True, only the position of the end link is considered in the IK computation
               action (torch.Tensor): delta action to be applied to the articulation. Used for fast delta IK solutions on the GPU.
               use_delta_ik_solver (bool): If true, returns the target joint positions that correspond with a delta IK solution. This is specifically
                   used for GPU simulation to determine which GPU IK algorithm to use.
           """
           if self.use_gpu_ik:
               q0 = q0[:, self.active_ancestor_joint_idxs]
               if not use_delta_ik_solver:
                   tf = pk.Transform3d(
                       pos=target_pose.p,
                       rot=target_pose.q,
                       device=self.device,
                   )
                   self.pik.initial_config = q0  # shape (num_retries, active_ancestor_dof)
                   result = self.pik.solve(
                       tf
                   )  # produce solutions in shape (B, num_retries/initial_configs, active_ancestor_dof)
                   # TODO return mask for invalid solutions. CPU returns None at the moment
                   return result.solutions[:, 0, :]
               else:
                   jacobian = self.pk_chain.jacobian(q0)
                   # code commented out below is the fast kinematics method
                   # jacobian = (
                   #     self.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                   #         self.articulation.get_qpos()[:, self.active_ancestor_joint_idxs]
                   #     )
                   #     .view(-1, len(self.active_ancestor_joints), 6)
                   #     .permute(0, 2, 1)
                   # )
                   # jacobian = jacobian[:, :, self.qmask]
                   if pos_only:
                       jacobian = jacobian[:, 0:3]
   
                   # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
                   delta_joint_pos = torch.linalg.pinv(jacobian) @ action.unsqueeze(-1)
                   return q0 + delta_joint_pos.squeeze(-1)
           else:
               if iter_flag:
                   result, success, error, iter = self.pmodel.compute_inverse_kinematics(
                       self.end_link_idx,
                       target_pose.sp,
                       initial_qpos=q0.cpu().numpy()[0],
                       active_qmask=self.qmask,
                       max_iterations=100,
                       iter_flag=iter_flag
                   )
               else:
                   result, success, error = self.pmodel.compute_inverse_kinematics(
                       self.end_link_idx,
                       target_pose.sp,
                       initial_qpos=q0.cpu().numpy()[0],
                       active_qmask=self.qmask,
                       max_iterations=100,
                   )
               if success:
                   if not iter_flag:
                       return common.to_tensor(
                           [result[self.active_ancestor_joint_idxs]], device=self.device
                       )
                   else:
                       return [common.to_tensor(
                           [result[self.active_ancestor_joint_idxs]], device=self.device
                       ), iter, error]
               else:
                   return None
   ```

4. Add solve function in you_conda_env/lib/python3.10/site-packages/mani_skill/agents/controllers/pd_ee_pose.py

```
def solve(self, action: Array, iter_flag=False):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base
        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == PDEEPosControllerConfig
        result = self.kinematics.compute_ik(
            self._target_pose,
            self.articulation.get_qpos(),
            pos_only=pos_only,
            action=action,
            use_delta_ik_solver=self.config.use_delta and not self.config.use_target,
            iter_flag=iter_flag
        )

        iter = None
        error = None
        if iter_flag:
            if result is not None:
                self._target_qpos = result[0]
                iter = result[1]
                error = result[2]
            else:
                self._target_qpos = None
        else:
            self._target_qpos = None

        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)
        
        return iter, error, self._target_qpos
```



## ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

#### [[Project Page]](https://rekep-robot.github.io/) [[Paper]](https://rekep-robot.github.io/rekep.pdf) [[Video]](https://youtu.be/2S8YhBdLdww)

[Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Chen Wang](https://www.chenwangjeremy.net/)<sup>1*</sup>, [Yunzhu Li](https://yunzhuli.github.io/)<sup>2*</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup> (\* indicates equal contributions)

<sup>1</sup>Stanford University, <sup>3</sup>Columbia University

<img  src="media/pen-in-holder-disturbances.gif" width="550">

This is the official demo code for [ReKep](https://rekep-robot.github.io/) implemented in [OmniGibson](https://behavior.stanford.edu/omnigibson/index.html). ReKep is a method that uses large vision models and vision-language models in a hierarchical optimization framework to generate closed-loop trajectories for manipulation tasks.

**Note: This codebase currently does not contain the perception pipeline used in our real-world experiments, including keypoint tracking, mask tracking, and SDF reconstruction. Instead, it directly accesses this information from the simulation. If you are interested in deploying the code on a real robot, please refer to [this section](#real-world-deployment) below and the paper [Appendix](https://rekep-robot.github.io/rekep.pdf#page=20.09), where further details are provided.**

## Setup Instructions

Note that this codebase is best run with a display. For running in headless mode, refer to the [instructions in OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html).

- Install [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html). This code is tested on [this commit](https://github.com/StanfordVL/OmniGibson/tree/cc0316a0574018a3cb2956fcbff3be75c07cdf0f).

NOTE: If you encounter the warning `We did not find Isaac Sim under ~/.local/share/ov/pkg.` when running `./scripts/setup.sh` for OmniGibson, first ensure that you have installed Isaac Sim. Assuming Isaac Sim is installed in the default directory, then provide the following path `/home/[USERNAME]/.local/share/ov/pkg/isaac-sim-2023.1.1` (replace `[USERNAME]` with your username).

- Install ReKep in the same conda environment:
```Shell
conda activate omnigibson
cd ..
git clone https://github.com/huangwl18/ReKep.git
cd ReKep
pip install -r requirements.txt
```

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key and set it up as an environment variable:
```Shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## Running Demo

We provide a demo "pen-in-holder" task that illustrates the core idea in ReKep. Below we provide several options to run the demo.

Notes:
- An additional `--visualize` flag may be added to visualize every solution from optimization, but since the pipeline needs to repeatedly solves optimization problems, the visualization is blocking and needs to be closed every time in order to continue (by pressing "ESC").
- Due to challenges of determinism of the physics simulator, different runs with the same random seed may produce different results. It is possible that the robot may fail at the provided task, especially when external disturbances are applied. In this case, we recommend running the demo again.

### Demo with Cached Query

We recommend starting with the cached VLM query.

```Shell
python main.py --use_cached_query [--visualize]
```

A video will be saved to `./videos` by default.

<img  src="media/pen-in-holder.gif" width="480">

### Demo with External Disturbances

Since ReKep acts as a closed-loop policy, it is robust to disturbances with automatic failure recovery both within stages and across stages. To demonstrate this in simulation, we apply the following disturbances for the "pen-in-holder" task:

- Move the pen when robot is trying to grasp the pen

- Take the pen out of the gripper when robot is trying to reorient the pen

- Move the holder when robot is trying to drop the pen into the holder

<img  src="media/pen-in-holder-disturbances.gif" width="480">

Note that since the disturbances are pre-defined, we recommend running with the cached query.

```Shell
python main.py --use_cached_query --apply_disturbance [--visualize]
```
### Demo with Live Query

The following script can be run to query VLM for a new sequence of ReKep constraints and executes them on the robot:

```Shell
python main.py [--visualize]
```

## Setup New Environments
Leveraging the diverse objects and scenes provided by [BEHAVIOR-1K](https://behavior.stanford.edu/) in [OmniGibson](https://behavior.stanford.edu/omnigibson/index.html), new tasks and scenes can be easily configured. To change the objects, you may check out the available objects as part of the BEHAVIOR assets on this [page](https://behavior.stanford.edu/knowledgebase/objects/index.html) (click on each object instance to view its visualization). After identifying the objects, we recommend making a copy of the JSON scene file `./configs/og_scene_file_pen.json` and edit the `state` and `objects_info` accordingly. Remember that the scene file need to be supplied to the `Main` class at initialization. Additional [scenes](https://behavior.stanford.edu/knowledgebase/scenes/index.html) and [robots](https://behavior.stanford.edu/omnigibson/getting_started/examples.html#robots) provided by BEHAVIOR-1K may also be possible, but they are currently untested.

## Real-World Deployment
To deploy ReKep in the real world, most changes should only be needed inside `environment.py`. Specifically, all of the "exposed functions" need to be changed for the real world environment. The following components need to be implemented:

- **Robot Controller**: Our real-world implementation uses the joint impedance controller from [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control) for our Franka setup. Specifically, when `execute_action` in `environment.py` receives a target end-effector pose, we first calculate IK to obtain the target joint positions and send the command to the low-level controller.
- **Keypoint Tracker**: Keypoints need to be tracked in order to perform closed-loop replanning, and this typically needs to be achieved using RGD-D cameras. Our real-world implementation uses similarity matching of [DINOv2](https://github.com/facebookresearch/dinov2) features calculated from multiple RGB-D cameras to track the keypoints (details may be found in the [paper](https://rekep-robot.github.io/rekep.pdf) appendix). Alternatively, we also recommend trying out specialized point trackers, such as [\[1\]](https://github.com/henry123-boy/SpaTracker), [\[2\]](https://github.com/google-deepmind/tapnet), [\[3\]](https://github.com/facebookresearch/co-tracker), and [\[4\]](https://github.com/aharley/pips2).
- **SDF Reconstruction**: In order to avoid collision with irrelevant objects or the table, an SDF voxel grid of the environment needs to be provided to the solvers. Additionally, the SDF should ignore robot arm and any grasped objects. Our real-world implementation uses [nvblox_torch](https://github.com/NVlabs/nvblox_torch) for ESDF reconstruction, [cuRobo](https://github.com/NVlabs/curobo) for segmenting robot arm, and [Cutie](https://github.com/hkchengrex/Cutie) for object mask tracking.
- **(Optional) Consistency Cost**: If closed-loop replanning is desired, we find it helpful to include a consistency cost in the solver to encourage the new solution to be close to the previous one (more details can be found in the [paper](https://rekep-robot.github.io/rekep.pdf) appendix).
- **(Optional) Grasp Metric or Grasp Detector**: We include a cost that encourages top-down grasp pose in this codebase, in addition to the collision avoidance cost and the ReKep constraint (for identifying grasp keypoint), which collectively identify the 6 DoF grasp pose. Alternatively, other grasp metrics can be included, such as force-closure. Our real-world implementation instead uses grasp detectors [AnyGrasp](https://github.com/graspnet/anygrasp_sdk), which is implemented as a special routine because it is too slow to be used as an optimizable cost.

Since there are several components in the pipeline, running them sequentially in the real world may be too slow. As a result, we recommend running the following compute-intensive components in separate processes in addition to the main process that runs `main.py`: `subgoal_solver`, `path_solver`, `keypoint_tracker`, `sdf_reconstruction`, `mask_tracker`, and `grasp_detector` (if used).

## Known Limitations
- **Prompt Tuning**: Since ReKep relies on VLMs to generate code-based constraints to solve for the behaviors of the robot, it is sensitive to the specific VLM used and the prompts given to the VLM. Although visual prompting is used, typically we find that the prompts do not necessarily need to contain image-text examples or code examples, and pure-text high-level instructions can go a long way with the latest VLM such as `GPT-4o`. As a result, when starting with a new domain and if you observe that the default prompt is failing, we recommend the following steps: 1) pick a few representative tasks in the domain for validation purposes, 2) procedurally update the prompt with high-level text examples and instructions, and 3) test the prompt by checking the text output and return to step 2 if needed.

- **Performance Tuning**: For clarity purpose, the entire pipeline is run sequentially. The latency introduced by the simulator and the solvers gets compounded. If this is a concern, we recommend running compute-intensive components, such as the simulator, the `subgoal_solver`, and the `path_solver`, in separate processes, but concurrency needs to be handled with care. More discussion can be found in the "Real-World Deployment" section. To tune the solver, the `objective` function typically takes the majority of time, and among the different costs, the reachability cost by the IK solver is typically most expensive to compute. Depending on the task, you may reduce `sampling_maxfun` and `maxiter` in `configs/config.yaml` or disable the reachability cost. 

- **Task-Space Planning**: Since the current pipeline performs planning in the task space (i.e., solving for end-effector poses) instead of the joint space, it occasionally may produce actions kinematically challenging for robots to achieve, especially for tasks that require 6 DoF motions.

## Troubleshooting

For issues related to OmniGibson, please raise a issue [here](https://github.com/StanfordVL/OmniGibson/issues). You are also welcome to join the [Discord](https://discord.com/invite/bccR5vGFEx) channel for timely support.

For other issues related to the code in this repo, feel free to raise an issue in this repo and we will try to address it when available.
