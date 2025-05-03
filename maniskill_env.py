from mani_skill.utils import sapien_utils
import sapien.wrapper.pinocchio_model
import gymnasium as gym
import cv2
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import sapien.core as sapien
from mani_skill.utils import sapien_utils
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import transform_utils as T
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import trimesh
import time
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_to_matrix
)
import torch
import os
from os.path import join
import sys
sys.path.append(os.getcwd())
from gymnasium.wrappers import TimeLimit
from mani_skill.utils.wrappers import VLARecorderWrapper
from utils import *
import tqdm
CLOSE = 0
OPEN = 1

class TqdmWrapper(gym.Wrapper):
    """
    为 ManiSkill 环境添加 tqdm 进度条的包装器。

    Args:
        env: 要包装的 Gymnasium 环境。
        max_episode_steps: 每个 episode 的最大步数。
        desc: 进度条的描述。
    """

    def __init__(self, env, max_episode_steps, desc="Episode Progress"):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.desc = desc
        self.cnt = 0
        self.done = False 
        self.pbar = None

    def reset(self, **kwargs):
        # 重置环境
        obs, info = self.env.reset(**kwargs)
        self.cnt = 0
        self.not_done = False 
        # 如果有旧的进度条，先关闭
        if self.pbar is not None:
            self.pbar.close()
        
        # 初始化新的 tqdm 进度条
        self.pbar = tqdm.tqdm(total=self.max_episode_steps, desc=self.desc)

        return obs, info

    def step(self, action):
        # 执行 step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cnt += 1
        # 更新进度条
        if self.pbar is not None:
            self.pbar.update(1)

        # # 如果 episode 结束，关闭进度条
        if terminated or truncated:
            # pdb.set_trace()
            self.done = True
            self.close()

        return obs, reward, terminated, truncated, info

    def close(self):
        # 关闭进度条（如果存在）
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        # 关闭环境
        super().close()

class ManiSkill_Env():
    def __init__(self, config=None, verbose=False, task_name="Tabletop-Pick-Apple-v1", obs_mode="rgb+depth+segmentation", dt=1/60, control_mode='pd_joint_pos', output_dir='test', model_class="rekep_default"):

        self.env = gym.make(
            task_name, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode=obs_mode, # there is also "state_dict", "rgbd", ...
            control_mode=control_mode, # there is also "", ...
            render_mode="rgb_array",
            camera_width=1024,  # Camera resolution width
            camera_height=1024,  # Camera resolution height
        )
        term_steps = 5000
        self.env = TimeLimit(self.env, max_episode_steps=term_steps)
        self.env = VLARecorderWrapper(
            self.env,
            output_dir=output_dir,
            model_class=model_class,
            model_path="None",
            save_trajectory=False,
        )

        self.env = TqdmWrapper(self.env, max_episode_steps=term_steps)
        debug = False
        vis = False
        if control_mode == 'pd_joint_pos':
            self.planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=debug,
                vis=vis,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_acc_limits=0.5,
                joint_vel_limits=0.5,
            )
        else:
            self.planner = None
        
        self.control_mode = control_mode
        self.dt = dt
        self.interp_num = 2

        # get semantic id2name and name2id maps, get robot id on the same time
        segmentation_id_map = self.env.unwrapped.segmentation_id_map
        self.id2name = {}
        for key in segmentation_id_map.keys():
            self.id2name[key] = segmentation_id_map[key].name
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.robot_mask_ids = []
        for key in self.name2id.keys():
            if key.split('_')[0] == 'panda':
                self.robot_mask_ids.append(self.name2id[key])
        
        self.base_position = np.squeeze(self.env.unwrapped.agent.robot.pose.p.numpy())

        self.default_quat = self.get_ee_quat()

        self.frame = 'world'
        self.camera_list = ['base_camera', 'base_front_camera','base_up_front_camera', 'front_camera']

        # rekep part
        self.config = config

        if config is not None:
            self.bounds_min = np.array(self.config['main']['bounds_min'])
            self.bounds_max = np.array(self.config['main']['bounds_max'])
            self.interpolate_pos_step_size = self.config['main']['interpolate_pos_step_size']
            self.interpolate_rot_step_size = self.config['main']['interpolate_rot_step_size']

        self.verbose = verbose

        self.ik_solver = self.env.agent.controller.controllers['arm']
        
    @property
    def workspace_bounds_max(self):
        return self.bounds_max

    @property
    def workspace_bounds_min(self):
        return self.bounds_min
    
    @property
    def base_transformation(self):
        pos = np.squeeze(self.env.unwrapped.agent.robot.pose.p.numpy())
        quat = np.squeeze(self.env.unwrapped.agent.robot.pose.q.numpy())
        return np.concatenate([pos, quat])

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def transform_goal_to_wrt_base(self, goal_pose):
        base_pose = self.base_transformation
        base_tf = np.eye(4)
        base_tf[0:3, 3] = base_pose[:3]
        base_tf[0:3, 0:3] = quat2mat(base_pose[3:])
        goal_tf = np.eye(4)
        goal_tf[0:3, 3] = goal_pose[:3]
        goal_tf[0:3, 0:3] = quat2mat(goal_pose[3:])
        goal_tf = np.linalg.inv(base_tf).dot(goal_tf)
        new_goal_pose = np.zeros(7)
        new_goal_pose[:3] = goal_tf[0:3, 3]
        new_goal_pose[3:] = mat2quat(goal_tf[0:3, 0:3])
        return new_goal_pose
    
    def world_to_robot(self, object_pos, object_quat, robot_pos, robot_quat):
        # Step 1: Create transform matrices
        obj_R = R.from_quat(object_quat).as_matrix()    # 3x3 rotation matrix
        robot_R = R.from_quat(robot_quat).as_matrix()

        # Step 2: Build homogeneous matrices
        obj_T = np.eye(4)
        obj_T[:3, :3] = obj_R
        obj_T[:3, 3] = object_pos

        robot_T = np.eye(4)
        robot_T[:3, :3] = robot_R
        robot_T[:3, 3] = robot_pos

        # Step 3: Invert robot pose
        robot_T_inv = np.linalg.inv(robot_T)

        # Step 4: Transform object pose into robot frame
        obj_in_robot_T = np.matmul(robot_T_inv, obj_T)

        # Step 5: Extract translation and rotation
        trans = obj_in_robot_T[:3, 3]
        rot = R.from_matrix(obj_in_robot_T[:3, :3]).as_quat()

        return trans, rot


    def update_cameras_view(self):
        obs = self.env.unwrapped.render_all()
        obs = np.squeeze(obs.numpy())
        # print(f"\033[91m end-effector position is {self.get_ee_pos()}  \033[0m")
        bgr_obs_show = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow('obs', bgr_obs_show)
        cv2.waitKey(int(self.dt * 1000))

    def load_task(self, task_name, obs_mode="rgb+depth+segmentation"):
        self.env = gym.make(
            task_name, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode=obs_mode, # there is also "state_dict", "rgbd", ...
            control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
            render_mode="rgb_array",
        )
    
    def set_env_resolution(self, set_env_resolution):
        pass

    def reset(self):
        self.env.reset()
        # self.name_id_dict = self.env.unwrapped.get_all_name_id_dict()
        self.init_qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7] # get each motor pos of initialize pose
        self.init_qvel = np.squeeze(self.env.agent.robot.get_qvel().numpy())[:7]
        self.init_pose, self.init_quat = self.get_ee_pose()
        for i in range(1, 8):
            self.env.step(self.get_tcp_pose())
        obs = self.env.unwrapped.get_obs()
        self.latest_obs = obs
        self.latest_action = np.concatenate([self.init_pose, self.init_quat, np.array([1])])
        self.update_cameras_view()
    
    # @property
    # def name_id_dict(self):
    #     return self.name_id_dict

    @property
    def reset_joint_pos(self):
        return self.init_qpos

    def reset_to_default_pose(self):
        ee_action = self.latest_action[-1:]
        action = np.concatenate([self.init_pose, self.init_quat, ee_action])
        self.apply_action(action)

    def _reset_task_variables(self):
        self.init_qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7] # get each motor pos of initialize pose, contain finger pos
        self.init_qvel = np.squeeze(self.env.agent.robot.get_qvel().numpy())[:7]
        self.init_pose, self.init_quat = self.get_ee_pose()

        for i in range(1, 8):
            self.env.step(self.get_tcp_pose())
        obs = self.env.unwrapped.get_obs()
        self.latest_obs = obs
        self.latest_action = np.concatenate([self.init_pose, self.init_quat, np.array([1])])
        self.update_cameras_view()

    def apply_action(self, action, ignore_arm=False, ignore_ee=False):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply (xyz + wxyz)

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.control_mode == 'pd_joint_pos':
            if not ignore_ee:
                ee_action = action[7]
                ee_action = ee_action > 0.5
                qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
                if ee_action == 0:
                    # step_action = np.concatenate([qpos, np.array([0])])
                    if self.planner is not None:
                        _, reward, _, _, info = self.planner.close_gripper()
                    else:
                        pass
                    # self.env.step(step_action)
                    self.latest_action[-1] = 0
                else:
                    # step_action = np.concatenate([qpos, np.array([1])])
                    if self.planner is not None:
                        _, reward, _, _, info = self.planner.open_gripper()
                    else:
                        pass
                    # self.env.step(step_action)
                    self.latest_action[-1] = 1
                self.update_cameras_view()
            else:
                ee_action = self.latest_action[-1]

            if not ignore_arm:
                arm_quat = action[3:7]
                arm_pos = action[:3]
                pose = sapien.Pose(p=arm_pos, q=arm_quat)
                result = self.planner.move_to_pose_with_screw((pose), dry_run=True)
                if result == -1:
                    print(f'result = -1, cannot move to target pose by IK planner')
                else:
                    for pos in result['position']:
                        cur_pos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
                        end_pos = pos
                        diff_pos = end_pos - cur_pos
                        for i in range(self.interp_num):
                            pos = cur_pos + diff_pos * (i + 1) / self.interp_num
                            pos = pos.tolist()
                            pos.append(ee_action)
                            pos = np.array(pos)
                            self.env.step(pos)
                            self.latest_action = pos
                            self.update_cameras_view()
        elif self.control_mode == 'pd_ee_pose':
            if len(action) == 7:
                action = np.concatenate([action, self.latest_action[-1:]])
                ignore_ee = True
            action[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
            step_action = np.zeros(8)
            if ignore_ee:
                step_action[-1] = self.latest_action[-1]
            else:
                step_action[-1] = action[-1]
                self.latest_action[-1] = action[-1]
            if ignore_arm:
                step_action[:-1] = self.latest_action[:-1]
            else:
                step_action[:-1] = action[:-1]
                self.latest_action[:-1] = action[:-1]
            
            step_space = np.zeros(7)
            pose_wrt_base = self.transform_goal_to_wrt_base(step_action[:7])
            step_space[:3] = pose_wrt_base[:3]
            step_space[3:6] = self.quat2rpy(pose_wrt_base[3:7])
            step_space[-1] = step_action[-1]
            for i in range(5):
                self.env.step(step_space)
                self.update_cameras_view()

    def quat2rpy(self, quat):
        matrix = quaternion_to_matrix(torch.tensor(quat))
        rpy = matrix_to_euler_angles(matrix, 'XYZ').numpy()
        return rpy

    def open_gripper(self):
        action = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        self.apply_action(action, ignore_arm=True)
    
    def close_gripper(self):
        if self.control_mode == 'pd_joint_pos':
            action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        elif self.control_mode == 'pd_ee_pose':
            action = np.array([0, 0, 0, 0, 0, 0, 0, -1])
        self.apply_action(action, ignore_arm=True)

    def move_to_pose(self, pose):
        action = pose
        self.apply_action(action, ignore_ee=True)

    def set_gripper_state(self, state):
        action = np.array([0, 0, 0, 0, 0, 0, 0, state])
        self.apply_action(action, ignore_arm=True)
        
    def get_ee_pose(self):
        tcp_pose = self.get_tcp_pose()
        position, quat = tcp_pose[:3], tcp_pose[3:]
        return position, quat
    def get_tcp_pose(self):
        obs = self.env.unwrapped.get_obs()
        tcp_pose = np.squeeze(obs['extra']['tcp_pose'].numpy())
        return tcp_pose
    def get_ee_pos(self):
        position, quat = self.get_ee_pose()
        return position

    def get_ee_quat(self):
        position, quat = self.get_ee_pose()
        return quat
    
    def get_last_gripper_action(self):
        return self.latest_action[-1]





    ''' Start ReKep'''

    def get_cam_obs(self):
        self.last_cam_obs = dict()
        for cam in self.camera_list:
            points, rgb, points_mask, depth = self.get_3d_point_cloud(cam, use_depth=True)
            self.last_cam_obs[cam] = {'rgb': rgb, 'depth': depth, 'points': points, 'seg': points_mask}
        return self.last_cam_obs
    
    def register_keypoints(self, keypoints):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.keypoints = keypoints
        self._keypoint_registry = dict()
        self._keypoint2object = dict()
        exclude_names = ['wall', 'floor', 'ceiling', 'table', 'fetch', 'robot', 'ground','panda', 'goal']
        for idx, keypoint in enumerate(keypoints):
            closest_distance = np.inf
            for obj in self.env.scene.actors:
                entity = self.env.scene.actors[obj]
                if any([name in obj.lower() for name in exclude_names]):
                    continue
                # a list of object mesh in world frame
                # print(type(entity))
                # print(f"\033[91m {entity} \033[0m")
                print(entity.name)
                collision_meshs = (entity).get_collision_meshes()
                for mesh in collision_meshs:
                    points_world = mesh.sample(1000)
                    dists = np.linalg.norm(points_world - keypoint, axis=1)
                    point = points_world[np.argmin(dists)]
                    distance = np.linalg.norm(point - keypoint)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_point = point
                        closest_obj = obj
                        cloest_entity = entity
                        link_name = None
            for obj in self.env.scene.articulations:
                if any([name in obj.lower() for name in exclude_names]):
                    continue
                entity = self.env.scene.articulations[obj]
                for link in entity.links:
                    if len(link.meshes) != 0:
                        colli_mesh = link.meshes[link.name]
                    else:
                        try:
                            colli_mesh = link.generate_mesh(filter=lambda _, render_shape: True, mesh_name=link.name)
                        except Exception as e:
                            print('skipping link ', link.name)
                            continue
                    for mesh in colli_mesh:
                        points_world = mesh.sample(1000)
                        dists = np.linalg.norm(points_world - keypoint, axis=1)
                        point = points_world[np.argmin(dists)]
                        distance = np.linalg.norm(point - keypoint)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_point = point
                            closest_obj = obj
                            cloest_entity = link
                            link_name = link.name
            self._keypoint_registry[idx] = (closest_obj, cloest_entity.pose, link_name)
            self._keypoint2object[idx] = closest_obj
            # overwrite the keypoint with the closest point
            self.keypoints[idx] = closest_point

    def get_keypoint_positions(self):
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3)
        Given the registered keypoints, this function returns their current positions in the world frame.
        """
        assert hasattr(self, '_keypoint_registry') and self._keypoint_registry is not None, "Keypoints have not been registered yet."
        keypoint_positions = []
        for idx, (closest_obj, pose, link_name) in self._keypoint_registry.items():
            tran = pose.p.cpu().numpy()
            quat = pose.q.cpu().numpy()
            rot = T.quat2mat(quat)
            init_pose = T.pose2mat((tran, quat))
            centering_transform = T.pose_inv(init_pose)
            keypoint_centered = np.dot(centering_transform, np.append(self.keypoints[idx], 1))[:3]
            if closest_obj in self.env.scene.articulations:
                entity = self.env.scene.articulations[closest_obj]
                for link in entity.links:
                    if link.name == link_name:
                        curr_pose = link.pose
            else:
                entity = self.env.scene.actors[closest_obj]
                curr_pose = entity.pose
            
            tran = curr_pose.p.cpu().numpy()
            quat = curr_pose.q.cpu().numpy()
            rot = T.quat2mat(quat)
            curr_pose = T.pose2mat((tran, quat))

            keypoint = np.dot(curr_pose, np.append(keypoint_centered, 1))[:3]
            keypoint_positions.append(keypoint)
        return np.array(keypoint_positions)
    
    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint2object[keypoint_idx]

    def is_grasping(self, candidate_obj):
        print("I am in is grasping function")
        if candidate_obj not in self.env.scene.actors:
            # assert(f'{candidate_obj} is not a rigid object and is not contained by scene.actors, please recheck it')
            return False
        else:
            entity = self.env.scene.actors[candidate_obj]
            if self.env.agent.is_grasping(entity):
                return True
            else:
                return False

    def get_gripper_null_action(self):
        return self.latest_action[-1]

    def get_collision_points(self, noise=True):
        """
        Get the points of the gripper and any object in hand.
        """
        # add gripper collision points
        collision_points = []
        for art in self.env.scene.articulations:
            if art == 'panda' or art == "panda_wristcam":
                entity = self.env.scene.articulations[art]
                for link in entity.links:

                    if 'link8' in link.name :
                        continue
                    try:
                        if len(link.meshes) != 0:
                            colli_mesh = link.meshes[link.name]
                        else:

                            try:
                                colli_mesh = link.generate_mesh(filter=lambda _, render_shape: True, mesh_name=link.name)
                            except Exception as e:
                                print('skipping link ', link.name)
                                continue
                            # colli_mesh = link.generate_mesh(filter=lambda _, render_shape: True, mesh_name=link.name)
                        for mesh in colli_mesh:
                            points_world = mesh.sample(1000)
                            collision_points.append(points_world)
                    except Exception as e:
                        print('exception', e)
                        continue 
        for obj in self.env.scene.actors:
            entity = self.env.scene.actors[obj]
            if self.env.agent.is_grasping(entity):
                collision_meshs = (entity).get_collision_meshes()
                collision_points.append(collision_meshs.sample(1000))

        
        collision_points = np.concatenate(collision_points, axis=0)
        return collision_points
    
    def get_arm_joint_positions(self):
        return np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
    
    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        start = time.time()
        exclude_names = ['wall', 'floor', 'ceiling', 'table', 'ground', 'goal']
        if exclude_robot:
            exclude_names += ['fetch', 'robot', 'panda']
        if exclude_obj_in_hand:
            assert self.config['env']['robot']['robot_config']['grasping_mode'] in ['assisted', 'sticky'], "Currently only supported for assisted or sticky grasping"
            # in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
            in_hand_obj = None
            for obj in self.env.scene.actors:
                entity = self.env.scene.actors[obj]
                if self.env.agent.is_grasping(entity):
                    in_hand_obj = obj
            if in_hand_obj is not None:
                exclude_names.append(in_hand_obj.lower())
                #exclude_names.append(in_hand_obj.name.lower()) # raw code
        trimesh_objects = []
        
        for obj in self.env.scene.actors:
                entity = self.env.scene.actors[obj]
                if any([name in obj.lower() for name in exclude_names]):
                    continue
                # breakpoint()
                print(entity.name)
                collision_meshs = (entity).get_collision_meshes()
                trimesh_objects = trimesh_objects + collision_meshs
                # trimesh_objects.append(collision_meshs)
                
        for obj in self.env.scene.articulations:
                if any([name in obj.lower() for name in exclude_names]):
                    continue
                trimesh_objects = []
                entity = self.env.scene.articulations[obj]
                for link in entity.links:
                    if len(link.meshes) != 0:
                        colli_mesh = link.meshes[link.name]
                    else:
                        try:
                            colli_mesh = link.generate_mesh(filter=lambda _, render_shape: True, mesh_name=link.name)
                        except Exception as e:
                            print('skipping link ', link.name)
                            continue
                        # colli_mesh = link.generate_mesh(filter=lambda _, render_shape: True, mesh_name=link.name)
                    trimesh_objects = trimesh_objects + colli_mesh
                    # trimesh_objects.append(colli_mesh)
                
        # chain trimesh objects
        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        vertex_positions = scene_mesh.vertices
        triangle_indices = scene_mesh.faces
        vertex_positions = o3d.core.Tensor(vertex_positions, dtype=o3d.core.Dtype.Float32)
        triangle_indices = o3d.core.Tensor(triangle_indices, dtype=o3d.core.Dtype.UInt32)
        _ = scene.add_triangles(vertex_positions, triangle_indices)  # we do not need the geometry ID for mesh
        # create a grid
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape
        grid = np.mgrid[self.bounds_min[0]:self.bounds_max[0]:steps[0],
                        self.bounds_min[1]:self.bounds_max[1]:steps[1],
                        self.bounds_min[2]:self.bounds_max[2]:steps[2]]
        grid = grid.reshape(3, -1).T
        # compute SDF
        sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
        # convert back to np array
        sdf_voxels = sdf_voxels.cpu().numpy()
        # open3d has flipped sign from our convention
        sdf_voxels = -sdf_voxels
        sdf_voxels = sdf_voxels.reshape(shape)
        print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] SDF voxels computed in {time.time() - start:.4f} seconds{bcolors.ENDC}')
        return sdf_voxels
    
    def execute_action(
            self,
            action,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            if precise:
                pos_threshold = 0.03
                rot_threshold = 3.0
            else:
                pos_threshold = 0.10
                rot_threshold = 5.0
            action = np.array(action).copy()
            assert action.shape == (8,)
            target_pose = action[:7]
            gripper_action = action[7]

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            current_pose = np.concatenate(self.get_ee_pose(), axis=0)
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            intermediate_pos_threshold = 0.10
            intermediate_rot_threshold = 5.0
            for pose in pose_seq[:-1]:
                self.apply_action(pose, False, True)
                # self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
            # move to the final pose with required precision
            pose = pose_seq[-1]
            for iter in range(20):
                self.apply_action(pose, False, True)
            # self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action 
            # ======================================
            if gripper_action == self.get_gripper_open_action():
                self.open_gripper()
            elif gripper_action == self.get_gripper_close_action():
                self.close_gripper()
            elif gripper_action == self.get_gripper_null_action():
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            
            return pos_error, rot_error
    def get_gripper_close_action(self):
        return CLOSE
    
    def get_gripper_open_action(self):
        return OPEN
    
    def compute_target_delta_ee(self, target_pose):
        ee_tran, ee_quat = self.get_ee_pose()
        pos_error = np.abs(ee_tran - target_pose[:3])
        rot_error = self.angle_between_quats(ee_quat, target_pose[3:])
        return pos_error, rot_error

    def angle_between_quats(self, q1, q2):
        """Angle between two quaternions"""
        return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

    # input action is [translation, quaternion] shape is [7, ]
    def solve_ik(self, action):
        ee_pos = action[:3]
        ee_quat = action[3:7]
        matrix = quaternion_to_matrix(torch.tensor(ee_quat))
        ee_rpy = matrix_to_euler_angles(matrix, 'XYZ').numpy()

        action = np.concatenate([ee_pos, ee_rpy]).reshape((1, -1))

        iter, error, target_qpos = self.ik_solver.solve(action, iter_flag=True)

    ''' End ReKep '''










    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        points, rgbs, point_masks = [], [], []
        for camera in self.camera_list:
            point, rgb, point_mask = self.get_3d_point_cloud(camera)
            points.append(point)
            rgbs.append(rgb)
            point_masks.append(point_mask)        
        return points, rgbs, point_masks
    '''
    Description: 
        Get 3D point cloud from RGBD cameras from simulator

    Returns:
        points:
            3D point cloud
        points_colors:
            3D point cloud's color
        points_mask:
            3D point cloud's semantic id
    '''
    def get_3d_point_cloud(self, camera_name, use_depth=False):
        obs = self.env.unwrapped.get_obs()
        depth = np.squeeze(obs['sensor_data'][camera_name]['depth'].cpu().numpy()) / 1000.0
        rgb = np.squeeze(obs['sensor_data'][camera_name]['rgb'].cpu().numpy())
        param = obs['sensor_param'][camera_name]
        mask = np.squeeze(obs['sensor_data'][camera_name]['segmentation'].cpu().numpy())

        intrinsic_matrix = np.squeeze(param['intrinsic_cv'].cpu().numpy())

        # points_mask_colors = self.depth_to_pc(intrinsic_matrix, depth, rgb, mask)
        points_colors = self.get_color_points(rgb, depth, param, mask)

        points = points_colors[:, :3]
        points_mask = mask.reshape((-1, 1))
        points_colors = points_colors[:, 3:]
        if not use_depth:
            return points, rgb, points_mask
        else:
            return points.reshape(rgb.shape), rgb, mask, depth
        # n*3, h*w*3, n*3, h*w*1
    
    '''
    Description:
        Use camera intrisic matrix and depth map to project 2D image into 3D point cloud
    Args:
        rgb: 2D RGB image. If rgb is not None, return colored point cloud.
        mask: 2D mask. If mask is not None, return masked point cloud.
    Returns:
        points: 1-3 cols are 3D point cloud, 4 col is mask, 5-7 cols are rgb.
    '''
    def depth_to_pc(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth

        
        points = np.vstack((x.flatten(), y.flatten(), z.flatten(), mask.flatten())).T

        flatten_rgb = rgb.reshape((-1, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    def get_color_points(self, rgb, depth, cam_param, mask=None):
        intrinsic_matrix = np.squeeze(cam_param['intrinsic_cv'].numpy())

        color_points = self.depth_to_point_cloud(intrinsic_matrix, depth, rgb, mask)
        if self.frame == 'base_front':
            return color_points
        cam2world_gl = np.squeeze(cam_param['cam2world_gl'].numpy())
        color_points[:, :3] = self.transform_camera_to_world(color_points[:, :3], cam2world_gl)
        return color_points

    def depth_to_point_cloud(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        flatten_rgb = rgb.reshape((H*W, 3))
        points = np.hstack((points, flatten_rgb))
        return points
    '''
    Description: 
        transform point cloud from camera coordinicate to world coordinate
    '''
    def transform_camera_to_world(self, points, extri_mat):
        R = extri_mat[:3, :3]
        t = extri_mat[:3, 3]
        pcd_world = (R @ points.T).T - t
        rotation_y_180 = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])
        pcd_world = (rotation_y_180 @ pcd_world.T).T
        return pcd_world
    
    def o3d_to_grasp(self, rot):
        theta = np.pi / 2
        rotate_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        rotate_y = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        rotate_z = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        unit_mat = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        maniskill_down = rotate_x @ rotate_x @ unit_mat
        o3d_down = rotate_z @ rotate_y @ unit_mat
        transfer_mat = maniskill_down @ o3d_down.T
        
        return maniskill_down @ rot
    def close(self):
        self.env.close()

    @property
    def done(self):
        return self.env.done

