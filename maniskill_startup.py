import argparse
import numpy as np

# from transforms3d.quaternions import mat2quat, quat2mat
from scipy.spatial.transform import Rotation as R
import torch
import json

import os
from os.path import join
import sys

sys.path.append(os.getcwd())

from maniskill_env import *
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer

# print(mani_env.get_collision_points)


class Main:
    def __init__(self, env, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")

        self.output_dir = output_dir
        self.config = global_config["main"]
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed(self.config["seed"])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config["keypoint_proposer"])
        self.constraint_generator = ConstraintGenerator(
            global_config["constraint_generator"]
        )
        # initialize environment
        self.env = env
        ik_solver = self.env.ik_solver
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(
            global_config["subgoal_solver"], ik_solver, self.env.reset_joint_pos
        )
        self.path_solver = PathSolver(
            global_config["path_solver"], ik_solver, self.env.reset_joint_pos
        )
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config["visualizer"], self.env)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config["vlm_camera"]]["rgb"]
        points = cam_obs[self.config["vlm_camera"]]["points"]
        mask = cam_obs[self.config["vlm_camera"]]["seg"]
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(
                rgb, points, mask
            )

            print(
                f"{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}"
            )
            print(self.visualize)
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {
                "init_keypoint_positions": keypoints,
                "num_keypoints": len(keypoints),
            }
            rekep_program_dir = self.constraint_generator.generate(
                projected_img, instruction, metadata
            )
            print(f"{bcolors.HEADER}Constraints generated{bcolors.ENDC}")
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, "metadata.json"), "r") as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {
            stage: False for stage in range(1, self.program_info["num_stages"] + 1)
        }
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info["init_keypoint_positions"])
        # load constraints
        self.constraint_fns = dict()
        for stage in range(
            1, self.program_info["num_stages"] + 1
        ):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ["subgoal", "path"]:
                load_path = os.path.join(
                    rekep_program_dir, f"stage{stage}_{constraint_type}_constraints.txt"
                )
                get_grasping_cost_fn = get_callable_grasping_cost_fn(
                    self.env
                )  # special grasping function for VLM to call
                stage_dict[constraint_type] = (
                    load_functions_from_txt(load_path, get_grasping_cost_fn)
                    if os.path.exists(load_path)
                    else []
                )
            self.constraint_fns[stage] = stage_dict

        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(
            self.program_info["num_keypoints"] + 1, dtype=bool
        )
        self.keypoint_movable_mask[0] = (
            True  # first keypoint is always the ee, so it's movable
        )

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        # breakpoint()
        while not self.env.done:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate(
                [[self.env.get_ee_pos()], scene_keypoints], axis=0
            )  # first keypoint is always the ee
            self.curr_ee_pose = np.concatenate(self.env.get_ee_pose(), axis=0)
            self.curr_joint_pos = self.env.get_arm_joint_positions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config["sdf_voxel_size"])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]["path"]
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config["constraint_tolerance"]:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]["path"]
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config["constraint_tolerance"]:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:
                        break
                print(
                    f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}"
                )
                self._update_stage(new_stage)
            else:
                # apply disturbance
                # self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                # if self.last_sim_step_counter == self.env.step_counter:
                #     print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(
                    next_subgoal, from_scratch=self.first_iter
                )
                self.first_iter = False
                self.action_queue = next_path.tolist()
                # self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while (
                    len(self.action_queue) > 0
                    and count < self.config["action_steps_per_iter"]
                    and not self.env.done
                ):
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    # print(next_action)
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info["num_stages"]:
                        # self.env.sleep(2.0)
                        print("I have finish the task")
                        # save_path = self.env.save_video()
                        # print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]["subgoal"]
        path_constraints = self.constraint_fns[self.stage]["path"]
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array(
                [-self.config["grasp_depth"] / 2.0, 0, 0]
            )
        debug_dict["stage"] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]["path"]
        path, debug_dict = self.path_solver.solve(
            self.curr_ee_pose,
            next_subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        print(path)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate(
            [
                self.curr_ee_pose.reshape(1, -1),
                path,
            ],
            axis=0,
        )
        num_steps = get_linear_interpolation_steps(
            full_control_points[0],
            full_control_points[-1],
            self.config["interpolate_pos_step_size"],
            self.config["interpolate_rot_step_size"],
        )
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info["grasp_keypoints"][self.stage - 1] != -1
        self.is_release_stage = (
            self.program_info["release_keypoints"][self.stage - 1] != -1
        )
        # can only be grasp stage or release stage or none
        assert (
            self.is_grasp_stage + self.is_release_stage <= 1
        ), "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = False

    def _update_keypoint_movable_mask(self):
        for i in range(
            1, len(self.keypoint_movable_mask)
        ):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        ee_tran, ee_quat = self.env.get_ee_pose()
        ee_tran += T.quat2mat(ee_quat) @ np.array([self.config["grasp_depth"], 0, 0])
        grasp_action = np.concatenate([ee_tran, ee_quat, np.array([0])])
        self.env.apply_action(grasp_action)

    def _execute_release_action(self):
        self.env.open_gripper()


def build_task(task_name, instruction):
    task = {
        "scene_file": None,
        "instruction": instruction,
        "rekep_program_dir": f"./vlm_query/{task_name}",
        "disturbance_seq": None,
    }
    return task


def exec_task(mani_env, task):
    scene_file = task["scene_file"]
    instruction = task["instruction"]

    main = Main(mani_env, visualize=args.visualize)
    main.perform_task(
        instruction,
        rekep_program_dir=task["rekep_program_dir"] if args.use_cached_query else None,
        disturbance_seq=task.get("disturbance_seq", None)
        if args.apply_disturbance
        else None,
    )


import datetime

experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = "./evaluation/" + experiment_date
interactive_path = (
    "/home/lr-2002/project/reasoning_manipulation/ManiSkill/env_ins_objects.pkl"
)
import pickle as pkl

error_list = []
# processed_count = 0
total_count = 0
NUM_EPISODE = 2

with open(interactive_path, "rb") as f:
    env_dict = pkl.load(f)
    total_count = len(env_dict)
    # breakpoint()
    print(f"Loaded {total_count} environments from {interactive_path}")
primitive_list_pkl = (
    "/home/lr-2002/project/reasoning_manipulation/ManiSkill/list_primitive_task.pkl"
)

with open(primitive_list_pkl, "rb") as f:
    primitive_list = pkl.load(f)
# try:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pen", help="task to perform")
    parser.add_argument(
        "--use_cached_query",
        action="store_true",
        help="instead of querying the VLM, use the cached query",
    )
    parser.add_argument(
        "--apply_disturbance",
        action="store_true",
        help="apply disturbance to test the robustness",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)',
    )
    parser.add_argument(
        "--run_pri", action="store_true", default=False, help="run primitive ? "
    )
    parser.add_argument(
        "--run_inter", action="store_true", default=False, help="run interacitve?"
    )
    parser.add_argument(
        "-f", "--filter", type=str, default='', help="run interacitve?"
    )
    args = parser.parse_args()
    # breakpoint()
    final_list = []
    if args.run_pri:
        final_list.append(primitive_list)
    if args.run_inter:
        final_list.append(env_dict)
    for task_set in final_list:
        for task_name, info in task_set.items():
            if not(args.filter.lower() in task_name.lower()):
                continue
            # breakpoint()

            global_config = get_config(config_path="./configs/config.yaml")
            mani_env = ManiSkill_Env(task_name=task_name, config=global_config, control_mode="pd_ee_pose", output_dir=output_dir , model_class='rekep_base')
            mani_env.reset()
            if isinstance(info, dict):
                info = info["ins"]

            try:
                exec_task(mani_env, build_task(task_name, info))
                try: 
                    mani_env.close()
                except Exception as e :
                     pass
            except KeyboardInterrupt:
                print(
                    f"[Interrupted] Task '{task_name}' was interrupted by user (Ctrl+C). Skipping..."
                )
                break
            except Exception as e:
                print(
                    f"[Error] Exception occurred while executing task '{task_name}': {e}"
                )
                import traceback
                traceback.print_exc()
                try: 
                    mani_env.close()
                except Exception as e :
                     pass
                continue
