import argparse
import numpy as np
# from transforms3d.quaternions import mat2quat, quat2mat 
from scipy.spatial.transform import Rotation as R
import torch
import json
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_to_matrix
)

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


mani_env = ManiSkill_Env(control_mode='pd_ee_pose')
mani_env.reset()

# print("Observation space", mani_env.env.observation_space)
# print("Action space", mani_env.env.action_space)

tran, quat = mani_env.get_ee_pose()
obs = mani_env.env.get_obs()
matrix = quaternion_to_matrix(torch.tensor(quat))
rpy = matrix_to_euler_angles(matrix, 'XYZ').numpy()
# print(np.concatenate(mani_env.get_ee_pose()))
robot_pose = mani_env.base_transformation
# print(np.concatenate(mani_env.get_ee_pose()))
# print(robot_pose)
# new_grasp_pose = mani_env.transform_goal_to_wrt_base(np.concatenate(mani_env.get_ee_pose()))

# print(new_grasp_pose)
# print(np.concatenate([base_tran, base_quat]))
# # action = np.concatenate([tran, rpy, np.array([0])])
# action = np.concatenate([base_tran, base_quat, np.array([1])])
# action = np.concatenate([new_grasp_pose, np.array([1])])
pose = np.concatenate(mani_env.get_ee_pose())
action = np.concatenate([pose, np.array([1])])
for i in range(5):
    # mani_env.env.step(action)
    mani_env.apply_action(action)
    mani_env.update_cameras_view()
    print(np.concatenate(mani_env.get_ee_pose()))
mani_env.close_gripper()
for i in range(5):
    mani_env.update_cameras_view()

# action = np.concatenate([tran, quat, np.array([1])])

# for i in range(10):
#     tran += np.array([0.01, 0.01, 0])
#     action = np.concatenate([tran, quat, np.array([1])])
#     mani_env.apply_action(action)

