import torch
import time
from abc import ABC
from easydict import EasyDict
import numpy as np
from src.hcrl.hcrl_policy_utils import Adam
from collections import namedtuple
from typing import Optional, Tuple
# from torch.distributions import Independent, Normal, Categorical
from torch.distributions import Independent, Normal
import math 

from torch.distributions.categorical import Categorical

import os
import gym
import csv
import wandb
import gym_hybrid
from datetime import datetime
from tqdm import tqdm 
from src.hcrl.hcrl_model import RunningMeanStd

from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd


gae_data = namedtuple('gae_data', ['value', 'next_value', 'reward', 'done', 'traj_flag'])
ppo_policy_data = namedtuple('ppo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight'])
ppo_data = namedtuple(
    'ppo_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'adv', 'return_', 'weight']
)
ppo_policy_loss = namedtuple('ppo_policy_loss', ['policy_loss', 'entropy_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])
ppo_loss = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])


def position_vehicle_vertex(width_vehicle, length_vehicle, vehicle_pos):
    '''
    Compute the vertex of the ego vehicle from the vehicle position
    Input :
        width_vehicle: the width of the ego vehicle
        length_vehicle: the length of the ego vehicle
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir]
    Return:
        vehicle_vert: the vertex position of the ego vehicle; vehicle_vert = [[vehicle_leftlow, vehicle_rightlow,
                                                                               vehicle_righttop, vehicle_lefttop]]
    '''
    x = vehicle_pos[0]
    y = vehicle_pos[1]
    dir = vehicle_pos[2]
    # The four vertices of the vehicle
    vehicle_vert = []
    vehicle_leftlow = [x - length_vehicle/2*math.cos(dir) + width_vehicle/2*math.sin(dir),
                       y - length_vehicle/2*math.sin(dir) - width_vehicle/2*math.cos(dir)]
    vehicle_rightlow = [x + length_vehicle/2*math.cos(dir) + width_vehicle/2*math.sin(dir),
                        y + length_vehicle/2*math.sin(dir) - width_vehicle/2*math.cos(dir)]
    vehicle_righttop = [x + length_vehicle/2*math.cos(dir) - width_vehicle/2*math.sin(dir),
                        y + length_vehicle/2*math.sin(dir) + width_vehicle/2*math.cos(dir)]
    vehicle_lefttop = [x - length_vehicle/2*math.cos(dir) - width_vehicle/2*math.sin(dir),
                       y - length_vehicle/2*math.sin(dir) + width_vehicle/2*math.cos(dir)]
    # vehicle_vert.append([vehicle_leftlow, vehicle_rightlow, vehicle_righttop, vehicle_lefttop])
    vehicle_vert.append([vehicle_leftlow, vehicle_rightlow, vehicle_righttop, vehicle_lefttop])

    return vehicle_vert

def generate_discrete_points(start_point, heading_angle, radius, arc_length, interval, turn_direction=None):
    """
    Generate discrete points at fixed intervals on an arc or a straight line.

    :param start_point: Coordinates of the starting point (x, y)
    :param heading_angle: Heading angle (in degrees, with 0 degrees being the positive direction of the x - axis)
    :param radius: Radius of the arc or the straight line
    :param arc_length: Arc length
    :param interval: Fixed interval between discrete points
    :param turn_direction: Direction. 0 indicates a right turn, 1 indicates going straight, and 2 indicates a left turn
    :return: A list of coordinates of discrete points [(x1, y1), (x2, y2), ...]
    """
    points = []
    x, y = start_point
    # heading_rad = np.radians(heading_angle)  # Convert the heading angle to radians.
    heading_rad = heading_angle

    if turn_direction == 2 or turn_direction == 0:
        # Convert the heading angle to radians
        if turn_direction == 2:  # turn left
            center_x = x - radius * np.sin(heading_rad)
            center_y = y + radius * np.cos(heading_rad)
            arc_angle = arc_length / radius
            start_angle = np.arctan2(y - center_y, x - center_x)
            theta = np.linspace(start_angle, start_angle + arc_angle, int(abs(arc_length) // interval) + 1)
            heading_theta = np.linspace(heading_rad, heading_rad + arc_angle, int(abs(arc_length) // interval) + 1)
        else:  # turn right
            center_x = x + radius * np.sin(heading_rad)
            center_y = y - radius * np.cos(heading_rad)
            arc_angle = arc_length / radius
            start_angle = np.arctan2(y - center_y, x - center_x)
            theta = np.linspace(start_angle, start_angle - arc_angle, int(abs(arc_length) // interval) + 1)
            heading_theta = np.linspace(heading_rad, heading_rad - arc_angle, int(abs(arc_length) // interval) + 1)
        
        # Generate discrete points on the arc.
        # for angle in theta:
        #     px = center_x + radius * np.cos(angle)
        #     py = center_y + radius * np.sin(angle)
        #     points.append((px, py, angle))

        for i in range(len(theta)):
            px = center_x + radius * np.cos(theta[i])
            py = center_y + radius * np.sin(theta[i])
            points.append((px, py, heading_theta[i]))
    
    else:  # go straight
        num_points = int(abs(arc_length) // interval) + 1
        direction = 1 if arc_length >= 0 else -1
        dx = direction * interval * np.cos(heading_rad)
        dy = direction * interval * np.sin(heading_rad)
        
        for i in range(num_points):
            px = x + i * dx
            py = y + i * dy
            points.append((px, py, heading_rad))

    return points

def get_path_with_actions(start_points, instructions, interval=0.01):
    """
    Generate and draw discrete points on the path according to the instructions.

    :param start_points: A list of coordinates and headings of the initial points [(x1, y1, heading1), (x2, y2, heading2), ...]
    :param instructions: A list of instructions for each point [(turn_direction1, arc_length1), ...]
    :param interval: Interval between discrete points
    """
    all_points = []  # It is used to store all the discrete points

    for (x, y, heading), instruction_set in zip(start_points, instructions):
        direction, arc_length = instruction_set
        radius = 5  # fixed radius
        points = generate_discrete_points((x, y), heading, radius, arc_length, interval, turn_direction=direction)
        all_points.extend(points)  # Record discrete points
    # Draw all the discrete points
    all_points = np.array(all_points)
    return all_points

# Calculate the vehicle contour
def vehicle_outline(x, y, theta, vehicle):
    corners = [
        (x + vehicle.length / 2 * math.cos(theta) - vehicle.width / 2 * math.sin(theta),
         y + vehicle.length / 2 * math.sin(theta) + vehicle.width / 2 * math.cos(theta)),
        (x + vehicle.length / 2 * math.cos(theta) + vehicle.width / 2 * math.sin(theta),
         y + vehicle.length / 2 * math.sin(theta) - vehicle.width / 2 * math.cos(theta)),
        (x - vehicle.length / 2 * math.cos(theta) + vehicle.width / 2 * math.sin(theta),
         y - vehicle.length / 2 * math.sin(theta) - vehicle.width / 2 * math.cos(theta)),
        (x - vehicle.length / 2 * math.cos(theta) - vehicle.width / 2 * math.sin(theta),
         y - vehicle.length / 2 * math.sin(theta) + vehicle.width / 2 * math.cos(theta)),
    ]
    return Polygon(corners)

# Path Visualization
def visualize_path(
        parking_polygon, 
        obstacles, 
        path, 
        vehicle,
        park_spaces,
        lane_mark,
    ):
    plt.figure(figsize=(10, 10))

    # Draw the parking lot
    x, y = parking_polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Parking Area')

    # Draw the obstacles
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        plt.fill(x, y, 'r', alpha=0.5, label='Obstacle')

    # Draw the path.
    # for state in path:
    #     vehicle_poly = vehicle_outline(state[0], state[1], state[2], vehicle)
    #     x, y = vehicle_poly.exterior.xy
    #     plt.plot(x, y, 'b-', alpha=0.5)

    '''
    绘制车辆路径
    '''
    # data_dir = './eval_log_3/obs_act_rew_20241115_234904.csv' 
    # data = pd.read_csv(data_dir)    # Read the CSV file         
    # traj_x = data['pos_x'].tolist()
    # traj_y = data['pos_y'].tolist()
    # traj_dir = data['pos_dir'].tolist()
    # action_type = data['action_type'].tolist()
    # action_args = data['action_args'].tolist()
    traj_x, traj_y, traj_dir, action_type, action_args = [], [], [], [], []
    for state in path:
        traj_x.append(state[0])
        traj_y.append(state[1])
        traj_dir.append(state[2])
        action_type.append(state[17])
        action_args.append(state[18])

    # Calculate the central point of the vehicle's trajectory
    start_points_traj = []
    actions = []
    for i in range(len(traj_x)-1):
        start_points_traj.append((traj_x[i], traj_y[i], traj_dir[i]))
        actions.append((action_type[i], action_args[i]))

    all_points = get_path_with_actions(start_points_traj, actions, interval=0.2)
    plt.scatter(all_points[:, 0], all_points[:, 1], s= 5)


    # Calculate the vehicle vertices of the vehicle trajectory
    path_vehicle_verts = []
    for i in range(len(all_points)):
        point_vehicle_vert = position_vehicle_vertex(
            width_vehicle = vehicle.width,
            length_vehicle = vehicle.length,
            vehicle_pos = all_points[i]
        )
        path_vehicle_verts.append(point_vehicle_vert[0])


    # Draw the bounding box of the vehicle trajectory
    for i in range(len(path_vehicle_verts)):
        for j in range(len(path_vehicle_verts[i])):
            if j == len(path_vehicle_verts[i])-1:
                x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][0][0]]
                y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][0][1]]
            else:
                x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][j+1][0]]
                y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][j+1][1]]
            # plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=0.5)
            plt.plot(x, y, 'b-', alpha=0.5)


    # Draw the initialized vehicle
    vehicle_pos = [traj_x[0], traj_y[0], traj_dir[0]]
    vehicle_vert = position_vehicle_vertex(
        width_vehicle = vehicle.width, 
        length_vehicle = vehicle.length, 
        vehicle_pos = vehicle_pos 
    )
    for i in range(len(vehicle_vert[0])):
        # The last edge
        if i == len(vehicle_vert[0])-1:
            x = [vehicle_vert[0][i][0], vehicle_vert[0][0][0]]
            y = [vehicle_vert[0][i][1], vehicle_vert[0][0][1]]
        # The last edge
        else:
            x = [vehicle_vert[0][i][0], vehicle_vert[0][i+1][0]]
            y = [vehicle_vert[0][i][1], vehicle_vert[0][i+1][1]]
        # plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
        plt.plot(x, y, 'b-', alpha=0.5)
        # Draw the central position of the vehicle
    # plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
    # Draw the vehicle heading.
    # plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

    # #  Draw the final position of the vehicle.
    final_vehicle_pos = [traj_x[-1], traj_y[-1], traj_dir[-1]]
    final_vehicle_vert = position_vehicle_vertex(
        width_vehicle = vehicle.width, 
        length_vehicle = vehicle.length, 
        vehicle_pos = final_vehicle_pos
    )
    for i in range(len(final_vehicle_vert[0])):
        # The last edge
        if i == len(final_vehicle_vert[0])-1:
            x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][0][0]]
            y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][0][1]]
        # The last edge
        else:
            x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][i+1][0]]
            y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][i+1][1]]
        # plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
        plt.plot(x, y, 'b-', alpha=0.5)
    # Draw the central position of the vehicle
    # plt.plot(final_vehicle_pos[0], final_vehicle_pos[1], marker='o', c='k',markersize=10)  
    # Draw the vehicle heading
    # plot_arrow(final_vehicle_pos[0], final_vehicle_pos[1], final_vehicle_pos[2], length=2.0, color='green')  

    # Draw the storage location line.
    for park_space in park_spaces:
        x, y = park_space.exterior.xy
        plt.plot(x, y, color='gray', alpha=0.8)

    # print('lane_mark')
    # Draw the lane line.
    x = [lane_mark[0][0][0],lane_mark[0][1][0]]
    y = [lane_mark[0][0][1],lane_mark[0][1][1]]
    plt.plot(x, y, label='linear', linestyle='--', color='gray', linewidth=2.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=24)  # Main scale
    # Add mathematical formulas
    # plt.text(1, 11, r"$Cs = 9.31;$", fontsize=24, color="k")
    # plt.text(1.5, 11.7, "C$_s$ = 6.81; N$_g$ = 2", fontsize=24, color="k")
    # plt.text(3, 10.5, "T$_c$ = 0.07", fontsize=24, color="k")
    # plt.title("HCRL", fontsize=26)
    # save_dir = './view_trajectory/fig_perpendicular_v28'
    plt.text(1.5, 11.7, "C$_s$ = 6.81; N$_g$ = 2", fontsize=24, color="k")
    plt.text(3, 10.5, "T$_c$ = 0.07", fontsize=24, color="k")
    plt.title("HCRL", fontsize=26)
    save_dir = './view_trajectory/fig_perpendicular_v28'
    plt.grid()
    plt.savefig(save_dir)
    plt.savefig(save_dir+'.pdf')

# Vehicle class
class Vehicle:
    def __init__(self, length, width, wheel_base, turning_radius):
        self.length = length  # Vehicle length
        self.width = width    # Vehicle width
        self.wheel_base = wheel_base  # Wheelbase
        self.turning_radius = turning_radius  # Minimum turning radius

def record_result_eval(algorithm, env_id, eval_result):
    '''
    Record the result of the evaluation.
    '''
    print('record result')
    # set the directory and file name
    # env_dir = env_id+'_result'
    # result_dir = os.path.join(env_dir, algorithm)
    result_dir= os.path.join('Eval_result', algorithm, env_id)
    result_file = 'eval_result.csv'

    # Create the directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Full file path for the result of evaluation.
    result_eval_filepath = os.path.join(result_dir, result_file)
    
    # Write data to a CSV file
    with open(result_eval_filepath, 'w', newline='') as file:
        writer_result = csv.writer(file)
        writer_result.writerow([
            'success_ratio_mean',
            'ep_ret_list_mean', 
            'ep_cost_list_mean', 
            'ep_time_list_mean',
            'ep_len_list_mean', 
            'ep_arc_len_list_mean', 
            'ep_shifting_numbers_list_mean', 
            'ep_min_rad_list_mean', 
            'ep_rad_numbers_list_mean', 
            'success_ratio_var', 
            'ep_ret_list_var',
            'ep_cost_list_var',
            'ep_time_list_var',
            'ep_len_list_var',
            'ep_arc_len_list_var',
            'ep_shifting_numbers_list_var',
            'ep_min_rad_list_var',
            'ep_rad_numbers_list_var' 
        ])
        writer_result.writerows(eval_result)
    
    return True


def calculate_movement_type(path_points):
    """
    Determine whether the vehicle is moving forward or backward.

    Parameters:
        path_points (list): Each path point is a list containing [x, y, theta],
        representing the x-coordinate, y-coordinate, and heading angle (unit: radians) respectively.
    Returns:
        list: The determination results of forward or backward movement, with each value being "Forward" or "Reverse".
    """
    movement_types = []
    
    for i in range(len(path_points) - 1):
        # Get two consecutive path points.
        x1, y1, theta1 = path_points[i]
        x2, y2, theta2 = path_points[i + 1]
        
        # Calculate the motion direction vector of the vehicle
        dx = x2 - x1
        dy = y2 - y1
        motion_angle = math.atan2(dy, dx)  # The angle of the direction of motion        
        
        # alculate the included angle between the heading angle and the direction of motion
        delta_angle = theta1 - motion_angle
        delta_angle = (delta_angle + math.pi) % (2 * math.pi) - math.pi  # Normalize it to the range of [-π, π].
        
        # If the included angle is close to 0, it indicates forward movement; if it is close to ±π, it indicates backward movement.
        if abs(delta_angle) < math.pi / 2:  # If the included angle is less than 90 degrees, it means moving forward.
            movement_types.append("Forward")
        else:
            movement_types.append("Reverse")
    
    return movement_types

def calculate_turning_radius(path_points):
    """
    Calculate the turning radius between consecutive path points.

    Parameters:
        path_points (list): Each path point is a list containing [x, y, theta], representing the x-coordinate, y-coordinate, and heading angle (in radians) respectively.
    Returns:
        list: The turning radius between every two path points, with the unit being the same length unit as the input.
    """
    turning_radii = []

    for i in range(len(path_points) - 1):
        # Extract two consecutive path points.
        x1, y1, theta1 = path_points[i]
        x2, y2, theta2 = path_points[i + 1]

        # Calculate the chord length between two points.
        chord_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the change in the heading angle (delta theta)
        delta_theta = abs(theta2 - theta1)
        
        # Ensure that delta_theta is within the range of [0, pi].
        delta_theta = min(delta_theta, 2 * math.pi - delta_theta)

        # The turning radius R = chord_length / (2 * sin(delta_theta / 2))
        if delta_theta == 0:  # Avoid the situation of division by zero, which indicates driving in a straight line and the radius is infinite.
            radius = float('inf')
        else:
            radius = chord_length / (2 * math.sin(delta_theta / 2))

        turning_radii.append(radius)

    return turning_radii

def gae(value, next_value, reward, done, gamma: float = 0.99, lambda_: float = 0.95) -> torch.FloatTensor:

    
    done = done.float()
    traj_flag = done
    if len(value.shape) == len(reward.shape) + 1:  # for some marl case: value(T, B, A), reward(T, B)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        traj_flag = traj_flag.unsqueeze(-1)

    next_value *= (1 - done)
    delta = reward + gamma * next_value - value
    factor = gamma * lambda_ * (1 - traj_flag)
    adv = torch.zeros_like(value)
    gae_item = torch.zeros_like(value[0])

    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item
    return adv

class HCRLPolicy_Safe(ABC):
    
    def __init__(
            self,
            env_id, 
            buf,
            model,
            device,
            ppo_param_init = True, 
            learning_rate: float = 3e-4,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            gamma = 0.99, 
            gae_lambda = 0.95, 
            recompute_adv = True,
            value_weight = 0.5,
            entropy_weight = 0.5,
            # clip_ratio = 0.05,
            clip_ratio = 0.2,
            adv_norm = True,
            value_norm = True,
            wandb_flag = False,
            env = None,
            share_encoder = True,
            batch_size = 64,
            save_freq = 10,
            run_name = 'not_wandb'
    )-> None:
        self._model = model.to(device)
        self._env_id = env_id
        self._buf = buf
        self._device = device
        self._ppo_param_init = ppo_param_init
        self._learning_rate = learning_rate
        self._grad_clip_type = grad_clip_type
        self._grad_clip_value = grad_clip_value
        
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._recompute_adv = recompute_adv
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight
        self._clip_ratio = clip_ratio
        self._adv_norm = adv_norm
        self._value_norm = value_norm
        self._wandb_flag = wandb_flag
        self._env = env
        self._share_encoder = share_encoder
        self._save_freq = save_freq
        self.batch_size = batch_size
        self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._running_mean_cost_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._reward_update = 0
        self._cost_update = 0
        self._reward_cost_update = 0
        self._init_cost_update = True
        self._run_name = run_name


        # Init the model of the HPPO network
        if self._ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                if isinstance(m, torch.nn.Linear):
                    # orthogonal initialization
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)

            for m in self._model.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        if self._share_encoder:
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )
        else:
            self._actor_optimizer = Adam(
                self._model.actor.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )
            self._critic_optimizer = Adam(
                self._model.critic.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )
            self._critic_cost_optimizer = Adam(
                self._model.critic_cost.parameters(),
                lr=self._learning_rate,
                grad_clip_type=self._grad_clip_type,
                clip_value=self._grad_clip_value
            )


    def update(self, 
               data, 
               train_iters=10,
               epoch_step = 1,
               best_performance = float('-inf'), 
               current_reward = float('-inf'),
               run_name = 'not_wandb',
               max_epoch_step = 5000,
               cost_limit = 5,
               limit_delta = 10,
        )-> None:
        """
        Overview:
            Given training data, implement network update for one iteration and update related variables.
            Learner's API for serial entry.
            Also called in ``start`` for each iteration's training.
        Arguments:
            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.

        .. note::

            ``_policy`` must be set before calling this method.

            ``_policy.forward`` method contains: forward, backward, grad sync(if in multi-gpu mode) and
            parameter update.

            ``before_iter`` and ``after_iter`` hooks are called at the beginning and ending.
        """
        
        # train_iters = 10
        value_weight = 0.5
        # entropy_weight = 0.2
        entropy_weight = 0.5
        clip_ratio = 0.2
        
        batch_iters = 10

        # batch = data.get()
        batch = data.get()


        for epoch in range(train_iters):
            # share the embedding network
            if self._share_encoder:   
                with torch.no_grad():
                    # value = self._model.compute_critic(batch['obs'])['value']
                    # next_value = self._model.compute_critic(batch['next_obs'])['value']
                    # #value norm
                    # value *= self._running_mean_std.std
                    # next_value *= self._running_mean_std.std
                    # batch['adv']=gae(value,next_value,batch['reward'],batch['done'])
                    
                    # unnormalized_returns = value + batch['adv']
                    
                    # batch['value'] = value / self._running_mean_std.std
                    # batch['ret'] = unnormalized_returns / self._running_mean_std.std
                    # self._running_mean_std.update(unnormalized_returns.cpu().numpy())

                    '''
                    The value, ret and adv of the cost.
                    '''
                    value_output = self._model.compute_critic(batch['obs'])
                    next_value_output = self._model.compute_critic(batch['next_obs'])
                    # The value of the reward
                    value = value_output['value']
                    next_value = next_value_output['value']
                    # The value of the cost
                    value_cost = value_output['value_cost']
                    next_value_cost = next_value_output['value_cost']
                    
                    # value norm of the reward
                    value *= self._running_mean_std.std
                    next_value *= self._running_mean_std.std
                    batch['adv']=gae(value,next_value,batch['reward'],batch['done'])                    
                    unnormalized_returns = value + batch['adv']                    
                    batch['value'] = value / self._running_mean_std.std
                    batch['ret'] = unnormalized_returns / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_returns.cpu().numpy())

                    # value norm of the cost
                    value_cost *= self._running_mean_cost_std.std
                    next_value_cost *= self._running_mean_cost_std.std
                    batch['adv_cost'] = gae(value_cost, next_value_cost, batch['cost'], batch['done'])
                    unnormalized_returns_cost = value_cost + batch['adv_cost']
                    batch['value_cost'] = value_cost / self._running_mean_cost_std.std
                    batch['ret_cost'] = unnormalized_returns_cost / self._running_mean_cost_std.std
                    self._running_mean_cost_std.update(unnormalized_returns_cost.cpu().numpy())

                
                for i in range(batch_iters):
                    batch_train = dict(
                        obs=batch['obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        next_obs=batch['next_obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        discrete_act=batch['discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        parameter_act=batch['parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        reward=batch['reward'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret=batch['ret'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv=batch['adv'][i*self.batch_size:(i+1)*self.batch_size,],
                        value=batch['value'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_discrete_act=batch['logp_discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_parameter_act=batch['logp_parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        done=batch['done'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_type=batch['logit_action_type'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argsmu=batch['logit_action_argsmu'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argssigma=batch['logit_action_argssigma'][i*self.batch_size:(i+1)*self.batch_size,],
                        # The sample of the cost
                        cost = batch['cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret_cost = batch['ret_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv_cost = batch['adv_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        value_cost = batch['value_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                    )
                    output = self._model.compute_actor_critic(batch_train['obs'])
                    # The adv of the reward
                    adv = batch_train['adv']
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    # The adv of the cost
                    adv_cost = batch_train['adv_cost']
                    adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-8)

                    # The final adv of the hppo+crpo
                    adv_final = adv 
                    
                    #discrete loss
                    discrete_weight = torch.ones_like(adv_final)
                    dist_discrete_new = Categorical(logits=output['logit']['action_type'])
                    dist_discrete_old = Categorical(logits=batch_train['logit_action_type'])
                    logp_discrete_new = dist_discrete_new.log_prob(batch_train['discrete_act'])
                    logp_discrete_old = dist_discrete_old.log_prob(batch_train['discrete_act'])
                    dist_discrete_new_entropy = dist_discrete_new.entropy()
                    if dist_discrete_new_entropy.shape != discrete_weight.shape:
                        dist_discrete_new_entropy = dist_discrete_new.entropy().mean(dim=1)
                    discrete_entropy_loss = (dist_discrete_new_entropy*discrete_weight).mean()
                    discrete_ratio = torch.exp(logp_discrete_new-logp_discrete_old)
                    if discrete_ratio.shape !=adv_final.shape:
                        discrete_ratio = discrete_ratio.mean(dim=1)
                    discrete_surr1 = discrete_ratio * adv_final
                    discrete_surr2 = discrete_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_final
                    discrete_policy_loss = (-torch.min(discrete_surr1, discrete_surr2) * discrete_weight).mean()
                    with torch.no_grad():
                        dis_approx_kl = (logp_discrete_old - logp_discrete_new).mean().item()
                        dis_clipped = discrete_ratio.gt(1 + clip_ratio) | discrete_ratio.lt(1 - clip_ratio)
                        dis_clipfrac = torch.as_tensor(dis_clipped).float().mean().item()
                        
                    #continuous loss
                    args_weight = torch.ones_like(adv_final)
                    dist_args_new = Independent(Normal(output['logit']['action_args']['mu'], output['logit']['action_args']['sigma']), 1)
                    dist_args_old = Independent(Normal(batch_train['logit_action_argsmu'], batch_train['logit_action_argssigma']), 1)
                    logp_args_new = dist_args_new.log_prob(batch_train['parameter_act'])
                    logp_args_old = dist_args_old.log_prob(batch_train['parameter_act'])
                    args_entropy_loss = (dist_args_new.entropy() * args_weight).mean()
                    args_ratio = torch.exp(logp_args_new - logp_args_old)
                    args_surr1 = args_ratio * adv_final
                    args_surr2 = args_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_final
                    args_policy_loss = (-torch.min(args_surr1, args_surr2) * args_weight).mean()
                    with torch.no_grad():
                        args_approx_kl = (logp_args_old - logp_args_new).mean().item()
                        args_clipped = args_ratio.gt(1 + clip_ratio) | args_ratio.lt(1 - clip_ratio)
                        args_clipfrac = torch.as_tensor(args_clipped).float().mean().item()
                        
                    # value loss of the reward
                    value_clip = batch_train['value'] + (output['value'] - batch_train['value']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret'] - output['value']).pow(2)
                    v2 = (batch_train['ret'] - value_clip).pow(2)
                    value_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()

                    # value loss of the cost
                    value_cost_clip = batch_train['value_cost'] + (output['value_cost'] - batch_train['value_cost']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret_cost'] - output['value_cost']).pow(2)
                    v2 = (batch_train['ret_cost'] - value_cost_clip).pow(2)
                    value_cost_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()
                    
                    total_loss = discrete_policy_loss + args_policy_loss + value_weight*value_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss) + value_weight*value_cost_loss
                    
                    self._optimizer.zero_grad()
                    total_loss.backward()
                    self._optimizer.step()
                    
                    if self._wandb_flag:
                        wandb.log({'record/discrete_policy_loss': discrete_policy_loss.item(), 
                                'record/args_policy_loss': args_policy_loss.item(), 
                                'record/value_loss':value_loss.item(),
                                'record/discrete_entropy_loss': discrete_entropy_loss.item(),
                                'record/args_entropy_loss': args_entropy_loss.item(),
                                'record/dis_approx_kl:': dis_approx_kl,
                                'record/dis_clipfrac:': dis_clipfrac,
                                'record/args_approx_kl:': args_approx_kl,
                                'record/args_clipfrac:': args_clipfrac,                                   
                                })
                
            else:
                with torch.no_grad():
                    value_output = self._model.compute_critic(batch['obs'])
                    next_value_output = self._model.compute_critic(batch['next_obs'])
                    # The value of the reward
                    value = value_output['value']
                    next_value = next_value_output['value']
                    # The value of the cost
                    value_cost = value_output['value_cost']
                    next_value_cost = next_value_output['value_cost']
                    
                    # value norm of the reward
                    value *= self._running_mean_std.std
                    next_value *= self._running_mean_std.std
                    batch['adv']=gae(value,next_value,batch['reward'],batch['done'])                    
                    unnormalized_returns = value + batch['adv']                    
                    batch['value'] = value / self._running_mean_std.std
                    batch['ret'] = unnormalized_returns / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_returns.cpu().numpy())

                    # value norm of the cost
                    value_cost *= self._running_mean_cost_std.std
                    next_value_cost *= self._running_mean_cost_std.std
                    batch['adv_cost'] = gae(value_cost, next_value_cost, batch['cost'], batch['done'])
                    unnormalized_returns_cost = value_cost + batch['adv_cost']
                    batch['value_cost'] = value_cost / self._running_mean_cost_std.std
                    batch['ret_cost'] = unnormalized_returns_cost / self._running_mean_cost_std.std
                    self._running_mean_cost_std.update(unnormalized_returns_cost.cpu().numpy())

                
                for i in range(batch_iters):
                    batch_train = dict(
                        obs=batch['obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        next_obs=batch['next_obs'][i*self.batch_size:(i+1)*self.batch_size,],
                        discrete_act=batch['discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        parameter_act=batch['parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        reward=batch['reward'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret=batch['ret'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv=batch['adv'][i*self.batch_size:(i+1)*self.batch_size,],
                        value=batch['value'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_discrete_act=batch['logp_discrete_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        logp_parameter_act=batch['logp_parameter_act'][i*self.batch_size:(i+1)*self.batch_size,],
                        done=batch['done'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_type=batch['logit_action_type'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argsmu=batch['logit_action_argsmu'][i*self.batch_size:(i+1)*self.batch_size,],
                        logit_action_argssigma=batch['logit_action_argssigma'][i*self.batch_size:(i+1)*self.batch_size,],
                        # The sample of the cost
                        cost = batch['cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        ret_cost = batch['ret_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        adv_cost = batch['adv_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        value_cost = batch['value_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                        ep_cost = batch['epret_cost'][i*self.batch_size:(i+1)*self.batch_size,],
                    )

                    output = self._model.compute_actor_critic(batch_train['obs'])
                    adv = batch_train['adv']
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    adv_cost = batch_train['adv_cost']
                    adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-8)
                
                    # if batch_train['ep_cost'].mean() > (cost_limit + limit_delta) and epoch_step >= 0.3*max_epoch_step and (self._reward_cost_update / (self._cost_update+1) == 4):
                    #     adv_final = - adv_cost
                    #     self._cost_update += 1
                    # else:
                    #     adv_final = adv
                    #     self._reward_update += 1
                    #     if self._cost_update > 0:
                    #         self._reward_cost_update +=1
                    # print('self._env_id:', self._env_id)
                    # quit()

                    if self._env_id == 'Perpendicular_safe-v0':
                        if batch_train['ep_cost'].mean()*10 > (cost_limit + limit_delta) and epoch_step >= 0.5*max_epoch_step:
                        # if False:
                            if self._init_cost_update:
                                adv_final = - adv_cost
                                self._cost_update += 1
                                self._init_cost_update = False
                            else:
                                if self._reward_cost_update / (self._cost_update) >= 2:
                                    adv_final = - adv_cost
                                    self._cost_update += 1
                                else:
                                    adv_final = adv
                                    self._reward_update += 1
                                    if self._cost_update > 0:
                                        self._reward_cost_update +=1
                        else:
                            adv_final = adv
                            self._reward_update += 1
                            if self._cost_update > 0:
                                self._reward_cost_update +=1
                    else:
                        if batch_train['ep_cost'].mean()*15 > (cost_limit + limit_delta) and epoch_step >= 0.6*max_epoch_step:
                            if self._init_cost_update:
                                adv_final = - adv_cost
                                self._cost_update += 1
                                self._init_cost_update = False
                            else:
                                if self._reward_cost_update / (self._cost_update) == 2:
                                    adv_final = - adv_cost
                                    self._cost_update += 1
                                else:
                                    adv_final = adv
                                    self._reward_update += 1
                                    if self._cost_update > 0:
                                        self._reward_cost_update +=1
                        else:
                            adv_final = adv
                            self._reward_update += 1
                            if self._cost_update > 0:
                                self._reward_cost_update +=1

                    # print('cost_mean:', batch_train['ep_cost'].mean())
                    # print('self._init_cost_update:', self._init_cost_update)
                    # print('self._reward_cost_update:', self._reward_cost_update)
                    # print('self._reward_update:', self._reward_update)
                    # print('self._cost_update:', self._cost_update)
                    # print('cost_limit:', cost_limit + limit_delta)
                    # print('epoch_step:', epoch_step)
                    # print('init_update:', 0.001*max_epoch_step)
                                                
                    
                    #discrete loss
                    discrete_weight = torch.ones_like(adv_final)
                    dist_discrete_new = Categorical(logits=output['logit']['action_type'])
                    dist_discrete_old = Categorical(logits=batch_train['logit_action_type'])
                    logp_discrete_new = dist_discrete_new.log_prob(batch_train['discrete_act'])
                    logp_discrete_old = dist_discrete_old.log_prob(batch_train['discrete_act'])
                    dist_discrete_new_entropy = dist_discrete_new.entropy()
                    if dist_discrete_new_entropy.shape != discrete_weight.shape:
                        dist_discrete_new_entropy = dist_discrete_new.entropy().mean(dim=1)
                    discrete_entropy_loss = (dist_discrete_new_entropy*discrete_weight).mean()
                    discrete_ratio = torch.exp(logp_discrete_new-logp_discrete_old)
                    if discrete_ratio.shape != adv_final.shape:
                        discrete_ratio = discrete_ratio.mean(dim=1)
                    discrete_surr1 = discrete_ratio * adv_final
                    discrete_surr2 = discrete_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_final
                    discrete_policy_loss = (-torch.min(discrete_surr1, discrete_surr2) * discrete_weight).mean()
                    with torch.no_grad():
                        dis_approx_kl = (logp_discrete_old - logp_discrete_new).mean().item()
                        dis_clipped = discrete_ratio.gt(1 + clip_ratio) | discrete_ratio.lt(1 - clip_ratio)
                        dis_clipfrac = torch.as_tensor(dis_clipped).float().mean().item()
                        
                    #continuous loss
                    args_weight = torch.ones_like(adv_final)
                    dist_args_new = Independent(Normal(output['logit']['action_args']['mu'], output['logit']['action_args']['sigma']), 1)
                    dist_args_old = Independent(Normal(batch_train['logit_action_argsmu'], batch_train['logit_action_argssigma']), 1)
                    logp_args_new = dist_args_new.log_prob(batch_train['parameter_act'])
                    logp_args_old = dist_args_old.log_prob(batch_train['parameter_act'])
                    args_entropy_loss = (dist_args_new.entropy() * args_weight).mean()
                    args_ratio = torch.exp(logp_args_new - logp_args_old)
                    args_surr1 = args_ratio * adv_final
                    args_surr2 = args_ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv_final
                    args_policy_loss = (-torch.min(args_surr1, args_surr2) * args_weight).mean()
                    with torch.no_grad():
                        args_approx_kl = (logp_args_old - logp_args_new).mean().item()
                        args_clipped = args_ratio.gt(1 + clip_ratio) | args_ratio.lt(1 - clip_ratio)
                        args_clipfrac = torch.as_tensor(args_clipped).float().mean().item()
                        
                    # value loss of the reward
                    value_clip = batch_train['value'] + (output['value'] - batch_train['value']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret'] - output['value']).pow(2)
                    v2 = (batch_train['ret'] - value_clip).pow(2)
                    value_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()

                    # value loss of the cost
                    value_cost_clip = batch_train['value_cost'] + (output['value_cost'] - batch_train['value_cost']).clamp(-clip_ratio, clip_ratio)
                    v1 = (batch_train['ret_cost'] - output['value_cost']).pow(2)
                    v2 = (batch_train['ret_cost'] - value_cost_clip).pow(2)
                    value_cost_loss = 0.5 * (torch.max(v1, v2) * args_weight).mean()
                    
                    
                    # total_loss = discrete_policy_loss + args_policy_loss + value_weight*value_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss)

                    actor_loss = discrete_policy_loss + args_policy_loss - entropy_weight*(discrete_entropy_loss+args_entropy_loss)

                    critic_loss = value_loss

                    critic_cost_loss = value_cost_loss

                    # backward for the actor network
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self._actor_optimizer.step()

                    # backward for the value critic netwrok
                    self._critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self._critic_optimizer.step()

                    # backward for the cost value critic network
                    self._critic_cost_optimizer.zero_grad()
                    critic_cost_loss.backward()
                    self._critic_cost_optimizer.step()
                    
                    if self._wandb_flag:
                        wandb.log({'record/discrete_policy_loss': discrete_policy_loss.item(), 
                                'record/args_policy_loss': args_policy_loss.item(), 
                                'record/value_loss':value_loss.item(),
                                'record/discrete_entropy_loss': discrete_entropy_loss.item(),
                                'record/args_entropy_loss': args_entropy_loss.item(),
                                'record/dis_approx_kl:': dis_approx_kl,
                                'record/dis_clipfrac:': dis_clipfrac,
                                'record/args_approx_kl:': args_approx_kl,
                                'record/args_clipfrac:': args_clipfrac,
                                'record/value_cost_loss': value_cost_loss.item(),
                                'update/reward_update': self._reward_update,
                                'update/cost_update': self._cost_update,
                                })
 

    def rollout(self, steps_per_epoch)-> None:
        '''
        Overview:
            Roll out to collect the sample and store to the buffer.
        Arguments:
            - env_id: the environment id. 
        '''
        # local_steps_per_epoch = 1000
        
        # env = gym.make(self._env_id)
        env = self._env

        # Prepare for interaction with environment
        start_time = time.time()
        obs, ep_ret, ep_len, ep_cost = env.reset(), 0, 0, 0

        trajectory_obs = []
        trajectory_act_rew = []
        trajectory_obs_act_rew = []

        for t in range(steps_per_epoch):
            
                trajectory_obs = list(tuple(obs))
                # Get the discrete and parameters action, 
                with torch.no_grad():
                    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                    # get the logit of the action throught HPPO network
                    action_value = self._model.compute_actor_critic(state)

                    logit = action_value['logit']
                    value = action_value['value']
                    value_cost = action_value['value_cost']
                    # discrete action
                    action_type_logit = logit['action_type']
                    prob = torch.softmax(action_type_logit, dim=-1)   # This 
                    pi_action = Categorical(prob)
                    action_type = pi_action.sample()                    
                    log_prob_action_type = pi_action.log_prob(action_type)

                    # continuous action
                    mu, sigma = logit['action_args']['mu'], logit['action_args']['sigma']
                    dist = Independent(Normal(mu, sigma), 1)
                    action_args = dist.sample()
                    # print('action_args:', action_args)
                    log_prob_action_args = dist.log_prob(action_args)

                    action = (int(action_type.cpu().numpy()), action_args.cpu().float().numpy().flatten())


                # interaction with the environment
                # next_obs, reward, done, info = env.step(action)
                next_obs, reward, cost, done, info = env.step(action)

                # record the action and reward for the trajectory.
                trajectory_act = [action[0], action[1][0]]
                trajectory_rew = [
                    info['reward_target'],      
                    info['reward_crash'],
                    info['reward_distance'],
                    info['reward_direction'],
                    info['reward']
                ]
                trajectory_act_rew.append(tuple(trajectory_act + trajectory_rew))
                trajectory_obs_act_rew.append(tuple(trajectory_obs + trajectory_act + trajectory_rew))

                ep_ret += reward
                ep_cost += cost
                ep_len += 1

                # Store the sample to the buffer.
                self._buf.store(
                    obs = obs,
                    next_obs = next_obs,
                    discrete_act = action_type.cpu(),
                    parameter_act = action_args.cpu(),
                    rew = reward,
                    val = value,
                    logp_discrete_act = log_prob_action_type.cpu(),
                    logp_parameter_act = log_prob_action_args.cpu(),
                    done = done,
                    logit_action_type = logit['action_type'].cpu(),
                    logit_action_argsmu = logit['action_args']['mu'].cpu(),
                    logit_action_argssigma = logit['action_args']['sigma'].cpu(),
                    val_cost = value_cost,
                    cost = cost
                )

                if self._wandb_flag:
                    wandb.log({'reward/rew': reward, 
                                'reward/value':value,
                                'cost/value': value_cost,
                              })

                # Update the obs
                obs = next_obs

                # The stop condition for each epoch
                epoch_ended = t==steps_per_epoch-1

                # The trajectory or epoch is stop
                if done or epoch_ended:
                    if reward!=0 and cost!=0:
                        value = reward
                        value_cost = cost
                    elif reward!=0 and cost==0:
                        value = reward
                        with torch.no_grad():
                            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                            # get the logit of the action throught HPPO network
                            action_value = self._model.compute_actor_critic(state)  
                        value_cost = action_value['value_cost'].cpu().float().numpy()
                    elif reward == 0 and cost!=0:
                        value_cost = cost
                        with torch.no_grad():
                            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                            # get the logit of the action throught HPPO network
                            action_value = self._model.compute_actor_critic(state)  
                        value = action_value['value'].cpu().float().numpy()                  
                    else:
                        with torch.no_grad():
                            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                            # get the logit of the action throught HPPO network
                            action_value = self._model.compute_actor_critic(state)  
                        value = action_value['value'].cpu().float().numpy()
                        value_cost = action_value['value_cost'].cpu().float().numpy()                 

                    self._buf.finish_path(value, value_cost)

                    if self._wandb_flag:
                        wandb.log({'ep_ret': ep_ret,
                                   'ep_cost': ep_cost, 
                                   'ep_len':ep_len})
                        
                    ''' 
                    Record the trajectory data.
                    '''
                    # set the directory and file name
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_time_dir = f"{current_time}"

                    # Define the directory and file name
                    directory = 'log_3'
                    obs_act_rew_filename = 'obs_act_rew_'+current_time_dir+'.csv'

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Full file path for the observation，action and reward trajectory
                    obs_act_rew_filepath = os.path.join(directory, obs_act_rew_filename)

                    # if ep_ret > 0 and len(trajectory_obs_act_rew)>1:
                    if info['reward_target'] > 0 and len(trajectory_obs_act_rew)>1:
                        trajectory_final_obs = list(tuple(next_obs))
                        trajectory_obs_act_rew.append(tuple(trajectory_final_obs + trajectory_act + trajectory_rew))
                        # Write data to a CSV file
                        with open(obs_act_rew_filepath, 'w', newline='') as file:
                            writer = csv.writer(file)
                            # Write the header
                            writer.writerow(['pos_x', 'pos_y', 'pos_dir', 
                                            'target_x', 'target_y', 'target_dir',
                                            'bias_x', 'bias_y', 'bias_dir',
                                            'vert_dis1', 'vert_dis2', 'vert_dis3', 'vert_dis4',
                                            'action_type', 'action_args',
                                            'reward_target', 'reward_crash', 'reward_distance',
                                            'reward_direction', 'reward'])
                            writer.writerows(trajectory_obs_act_rew)


                    obs, ep_ret, ep_len, ep_cost = env.reset(), 0, 0, 0


    def evaluate(
            self, 
            eval_epoch=5, 
            epsilon_rad=1e-2,
            algorithm = 'Hppo_safe',
            env_id = 'Perpendicular',
            run_name = 'not_wandb',
        ):
        '''
        Eval the model 
        '''
        print('=====evaluate======')
        success_ratio = 0
        env = self._env
        mean_ep_ret, mean_ep_len, mean_ep_cost = 0, 0, 0
        mean_arc_len, mean_rad_numbers = 0, 0
        mean_min_rad, mean_shifting_numbers = 5, 0 
        

        obs, ep_ret, ep_cost, ep_len = env.reset(eval_stage=True), 0, 0, 0
        
        ep_arc_len = 0
        rad_numbers = 0
        shifting_numbers = 0

        eval_trajectory_act_rew = []
        eval_trajectory_obs_act_rew = []
        eval_trajectory_pos = []

        eval_epoch_step = eval_epoch
        
        # The start time for epoch evaluation
        start_time = time.time()

        # for t in range(eval_epoch):
        while eval_epoch_step > 0:                
            trajectory_obs = list(tuple(obs))
            # Get the discrete and parameters action, 
            with torch.no_grad():
                state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                # get the logit of the action throught HPPO network
                action_value = self._model.compute_actor(state)

                logit = action_value['logit']
                # discrete action
                action_type_logit = logit['action_type']
                prob = torch.softmax(action_type_logit, dim=-1)        # This
                action_type = torch.argmax(prob, dim=1, keepdim=True) 
                # print('action_type:', action_type)

                # continuous action
                mu, _ = logit['action_args']['mu'], logit['action_args']['sigma']
                # print('mu:', mu)
                action_args = mu


                action = (int(action_type.cpu().numpy()), action_args.cpu().float().numpy().flatten())
                # arc_len = action_args.cpu().float().numpy().flatten()

            # interaction with the environment
            # next_obs, reward, done, info = env.step(action)
            next_obs, reward, cost, done, info = env.step(action)
            

            # record the action and reward for the trajectory.
            trajectory_act = [action[0], action[1][0]]
            trajectory_rew = [
                info['reward_target'],      
                info['reward_crash'],
                info['reward_distance'],
                info['reward_direction'],
                info['reward']
            ]
            eval_trajectory_act_rew.append(tuple(trajectory_act + trajectory_rew))
            eval_trajectory_obs_act_rew.append(tuple(trajectory_obs + trajectory_act + trajectory_rew))
            eval_trajectory_pos.append(tuple(trajectory_obs)[0:3])
           

            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            ep_arc_len += abs(action_args.cpu().float().numpy().flatten()[0])

            # Update the obs
            obs = next_obs

            # The trajectory or epoch is stop
            if done :
                ''' 
                Record the trajectory data.
                '''
                # Calculate the turning radius of the vehicle.
                # rad_numbers = 0
                radii = calculate_turning_radius(eval_trajectory_pos)
                min_rad = min(radii)
                for h in range(len(radii)-1):
                    if abs(radii[h] - radii[h+1]) < epsilon_rad:
                        rad_numbers += 1 
                mean_rad_numbers += rad_numbers
                mean_min_rad += min_rad

                # Determine whether the vehicle is moving forward or backward.
                # shifting_numbers = 0
                movement_types = calculate_movement_type(eval_trajectory_pos)
                for k in range(len(movement_types)-1):
                    if movement_types[k] !=  movement_types[k+1]:
                        shifting_numbers +=1
                mean_shifting_numbers += shifting_numbers

                # set the directory and file name
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_time_dir = f"{current_time}"

                # Define the directory and file name
                # directory = self._env_id + '_eval'
                trajectory_dir = os.path.join('Trajectory_result', algorithm, env_id, run_name)
                obs_act_rew_filename = 'obs_act_rew_'+current_time_dir+'.csv'

                # Create the directory if it doesn't exist
                if not os.path.exists(trajectory_dir):
                    os.makedirs(trajectory_dir)

                # Full file path for the observation，action and reward trajectory
                obs_act_rew_filepath = os.path.join(trajectory_dir, obs_act_rew_filename)

                # if ep_ret > 0 and len(eval_trajectory_obs_act_rew)>1:
                if info['reward_target'] > 0 and len(eval_trajectory_obs_act_rew)>1:
                    trajectory_final_obs = list(tuple(next_obs))
                    eval_trajectory_obs_act_rew.append(tuple(trajectory_final_obs + trajectory_act + trajectory_rew))
                    # Write data to a CSV file
                    with open(obs_act_rew_filepath, 'w', newline='') as file:
                        writer = csv.writer(file)
                        # Write the header
                        writer.writerow(['pos_x', 'pos_y', 'pos_dir', 
                                        'target_x', 'target_y', 'target_dir',
                                        'bias_x', 'bias_y', 'bias_dir',
                                        'ray_dis1', 'ray_dis2', 'ray_dis3', 'ray_dis4',
                                        'ray_dis5', 'ray_dis6', 'ray_dis7', 'ray_dis8',
                                        'action_type', 'action_args',
                                        'reward_target', 'reward_crash', 'reward_distance',
                                        'reward_direction', 'reward'])
                        writer.writerows(eval_trajectory_obs_act_rew)

                if info['reward_target'] > 0:
                    print('ep_ret:', ep_ret)
                    print('ep_len:', ep_len)
                
                mean_ep_ret += ep_ret
                mean_ep_cost += ep_cost
                mean_ep_len += ep_len
                mean_arc_len += ep_arc_len

                if info['reward_target']> 0:
                    success_ratio += 1

                # if self._wandb_flag:
                #     wandb.log({'eval_ep_ret': ep_ret, 'eval_ep_len': ep_len, 'eval_ep_cost':ep_cost})                

                obs, ep_ret, ep_len, ep_cost = env.reset(eval_stage = True), 0, 0, 0

                ep_arc_len = 0
                rad_numbers = 0
                shifting_numbers = 0

                eval_trajectory_act_rew = []
                eval_trajectory_obs_act_rew = []
                eval_epoch_step -= 1

        end_time = time.time()
        current_time = end_time - start_time

        if self._wandb_flag:
            wandb.log({'eval_mean_ep_ret': mean_ep_ret/eval_epoch, 
                       'eval_mean_ep_len':mean_ep_len/eval_epoch, 
                       'eval_mean_ep_cost':mean_ep_cost/eval_epoch,
                       'eval_mean_time':current_time/eval_epoch,
                       'eval_mean_arc_len': mean_arc_len/eval_epoch,
                       'eval_mean_rad_numbers': mean_rad_numbers/eval_epoch,
                       'eval_mean_shifting_numbers': mean_shifting_numbers/eval_epoch,
                       'eval_mean_min_rad': mean_min_rad/eval_epoch,
                       'eval_success_ratio': success_ratio/eval_epoch,
                       })
            # wandb.log({'eval_success_ratio': success_ratio/eval_epoch})
        
        eval_info = {}
        eval_info['eval_mean_ep_ret'] = mean_ep_ret/eval_epoch
        eval_info['eval_mean_ep_len'] = mean_ep_ret/eval_epoch
        eval_info['eval_mean_ep_cost'] = mean_ep_cost/eval_epoch
        eval_info['eval_success_ratio'] = success_ratio/eval_epoch       

        return eval_info
    

    def evaluate_policy(self, eval_epoch=5, epsilon_rad=1e-2, seed=0, env_id='Perpendicular'):
        '''
        Eval the model 
        '''
        eval_epoch_step = eval_epoch
        env = self._env
        success_ratio = 0
        mean_ep_ret, mean_ep_len, mean_ep_cost = 0, 0, 0
        mean_arc_len = 0
        mean_rad_numbers = 0
        mean_shifting_numbers = 0
        mean_min_rad = 0

        start_time = time.time()
        ep_ret_list = []
        ep_cost_list = []
        ep_len_list = []
        ep_time_list = []
        ep_arc_len_list = []
        ep_shifting_numbers_list = []
        ep_min_rad_list = []
        ep_rad_numbers_list = []
        
        for j in tqdm(range(eval_epoch), desc='Test Loop'):
            ep_start_time = time.time()
            obs, ep_ret, ep_cost, ep_len = env.reset(eval_stage=True), 0, 0, 0            
            done = False

            eval_trajectory_pos = []
            eval_trajectory_act_rew = []
            eval_trajectory_obs_act_rew = []
            ep_arc_len = 0
            rad_numbers = 0
            shifting_numbers = 0
            min_rad = float('inf')

            while not done:
                trajectory_obs = list(tuple(obs))
                # Get the discrete and parameters action, 
                with torch.no_grad():
                    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
                    # get the logit of the action throught HPPO network
                    action_value = self._model.compute_actor(state)

                    logit = action_value['logit']
                    # discrete action
                    action_type_logit = logit['action_type']
                    prob = torch.softmax(action_type_logit, dim=-1)        # This
                    action_type = torch.argmax(prob, dim=1, keepdim=True) 

                    # continuous action
                    mu, _ = logit['action_args']['mu'], logit['action_args']['sigma']
                    action_args = mu


                    action = (int(action_type.cpu().numpy()), action_args.cpu().float().numpy().flatten())

                # Step with environment
                next_obs, reward, cost, done, info = env.step(action)


                # record the action and reward for the trajectory.
                trajectory_act = [action[0], action[1][0]]
                trajectory_rew = [
                    info['reward_target'], 
                    info['reward_crash'],
                    info['reward_distance'],
                    info['reward_direction'],
                    info['reward']
                ]
                eval_trajectory_act_rew.append(tuple(trajectory_act + trajectory_rew))
                eval_trajectory_obs_act_rew.append(tuple(trajectory_obs + trajectory_act + trajectory_rew))
                eval_trajectory_pos.append(tuple(trajectory_obs)[0:3])

                # trajectory_pos.append(tuple(next_obs[0:3]))
                ep_ret += reward
                ep_cost += cost
                ep_len += 1
                ep_arc_len += abs(action_args.cpu().float().numpy().flatten()[0])
                # print('action_args.cpu().float().numpy().flatten()_type:', action_args.cpu().float().numpy().flatten()[0])
                # quit()

                # Update the obs
                obs = next_obs

            ep_end_time = time.time()
            ep_time = ep_end_time - ep_start_time
            # Record the success rate
            if info['reward_target'] > 0:
                success_ratio += 1
                  

            # Record the reward and cost values.
            mean_ep_ret += ep_ret
            mean_ep_cost += ep_cost
            mean_ep_len += ep_len
            mean_arc_len += ep_arc_len

            '''
            Calculate the turning radius, the minimum radius, and the number of gear shifts.
            '''
            # Record the reward and cost values.
            # rad_numbers = 0
            radii = calculate_turning_radius(eval_trajectory_pos)
            min_rad = min(radii)
            for h in range(len(radii)-1):
                if abs(radii[h] - radii[h+1]) < epsilon_rad:
                    rad_numbers += 1 
            mean_rad_numbers += rad_numbers
            mean_min_rad += min_rad

            # Determine whether the vehicle is moving forward or backward.
            # shifting_numbers = 0
            movement_types = calculate_movement_type(eval_trajectory_pos)
            for k in range(len(movement_types)-1):
                if movement_types[k] !=  movement_types[k+1]:
                    shifting_numbers +=1
            mean_shifting_numbers += shifting_numbers

            ep_ret_list.append(ep_ret)
            ep_cost_list.append(ep_cost)
            ep_time_list.append(ep_time)
            ep_len_list.append(ep_len)
            ep_arc_len_list.append(ep_arc_len)
            ep_shifting_numbers_list.append(shifting_numbers)
            ep_min_rad_list.append(min_rad)
            ep_rad_numbers_list.append(rad_numbers)


            '''
            Record the trajectory locally.
            '''
            # set the directory and file name
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_time_dir = f"{current_time}"

            # Define the directory and file name
            directory = self._env_id + '_eval'
            obs_act_rew_filename = 'obs_act_rew_'+current_time_dir+'.csv'

            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Full file path for the observation，action and reward trajectory
            obs_act_rew_filepath = os.path.join(directory, obs_act_rew_filename)

            # if ep_ret > 0 and len(eval_trajectory_obs_act_rew)>1:
            if info['reward_target'] > 0 and len(eval_trajectory_obs_act_rew)>1:
                trajectory_final_obs = list(tuple(next_obs))
                eval_trajectory_obs_act_rew.append(tuple(trajectory_final_obs + trajectory_act + trajectory_rew))
                # Write data to a CSV file
                with open(obs_act_rew_filepath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    # Write the header
                    writer.writerow(['pos_x', 'pos_y', 'pos_dir', 
                                    'target_x', 'target_y', 'target_dir',
                                    'bias_x', 'bias_y', 'bias_dir',
                                    'ray_dis1', 'ray_dis2', 'ray_dis3', 'ray_dis4',
                                    'ray_dis5', 'ray_dis6', 'ray_dis7', 'ray_dis8',
                                    'action_type', 'action_args',
                                    'reward_target', 'reward_crash', 'reward_distance',
                                    'reward_direction', 'reward'])
                    writer.writerows(eval_trajectory_obs_act_rew)

            # '''
            # Trajectory visualization
            # '''
            if j == 20:
                # Trajectory information
                path = eval_trajectory_pos
                path_act = eval_trajectory_obs_act_rew

                # Parking lot configuration information
                env_set = env.get_env_set()
                start_pos = env_set['vehicle_pos']
                goal_pos = env_set['target_pos']
                obstacles_vehicle = env_set['obstacles']
                park_vertex = env_set['park_edge']
                vehicle_config = env_set['vehicle_config']
                park_spaces = env_set['park_spaces']
                lane_mark = env_set['lane_mark']

                # Vehicle parameters
                vehicle = Vehicle(
                    length = vehicle_config.vehicle_length, 
                    width = vehicle_config.vehicle_width, 
                    wheel_base = vehicle_config.wheelbase, 
                    turning_radius = 5.0
                )

                if path:
                    visualize_path(park_vertex, obstacles_vehicle, path_act, vehicle, park_spaces, lane_mark)
                else:
                    print("No path found (it may be due to timeout or the path is not feasible).")
                quit()


        eval_result = []
        end_time = time.time()
        current_time = end_time - start_time

        print('======================')
        print('eval_mean_ep_ret:', mean_ep_ret/eval_epoch)
        print('eval_mean_ep_len:', mean_ep_len/eval_epoch)
        print('eval_mean_ep_cost:', mean_ep_cost/eval_epoch)
        print('eval_success_ratio:', success_ratio/eval_epoch)
        print('eval_mean_time:', current_time/eval_epoch)
        print('eval_mean_arc_len:', mean_arc_len/eval_epoch)

        ep_ret_list_mean, ep_ret_list_var = np.mean(ep_ret_list), np.std(ep_ret_list, ddof=1)
        ep_cost_list_mean, ep_cost_list_var = np.mean(ep_cost_list), np.std(ep_cost_list, ddof=1)
        ep_time_list_mean, ep_time_list_var = np.mean(ep_time_list), np.std(ep_time_list, ddof=1)
        ep_len_list_mean, ep_len_list_var = np.mean(ep_len_list), np.std(ep_len_list, ddof=1)
        ep_arc_len_list_mean, ep_arc_len_list_var = np.mean(ep_arc_len_list), np.std(ep_arc_len_list, ddof=1)
        ep_shifting_numbers_list_mean, ep_shifting_numbers_list_var = np.mean(ep_shifting_numbers_list), np.std(ep_shifting_numbers_list, ddof=1)
        ep_min_rad_list_mean, ep_min_rad_list_var = np.mean(ep_min_rad_list), np.std(ep_min_rad_list, ddof=1)
        ep_rad_numbers_list_mean, ep_rad_numbers_list_var = np.mean(ep_rad_numbers_list), np.std(ep_rad_numbers_list, ddof=1)

        print('ep_ret_list_mean, ep_ret_list_var:', ep_ret_list_mean, ep_ret_list_var)
        print('ep_cost_list_mean, ep_cost_list_var:', ep_cost_list_mean, ep_cost_list_var)
        print('ep_time_list_mean, ep_time_list_var:', ep_time_list_mean, ep_time_list_var)
        print('ep_len_list_mean, ep_len_list_var:', ep_len_list_mean, ep_len_list_var)
        print('ep_arc_len_list_mean, ep_arc_len_list_var:', ep_arc_len_list_mean, ep_arc_len_list_var)
        print('ep_shifting_numbers_list_mean, ep_shifting_numbers_list_var:', ep_shifting_numbers_list_mean, ep_shifting_numbers_list_var)
        print('ep_min_rad_list_mean, ep_min_rad_list_var:', ep_min_rad_list_mean, ep_min_rad_list_var)
        print('ep_rad_numbers_list_mean, ep_rad_numbers_list_var:', ep_rad_numbers_list_mean, ep_rad_numbers_list_var)

        eval_result.append(
            (# mean
            success_ratio/eval_epoch,
            ep_ret_list_mean, 
            ep_cost_list_mean, 
            ep_time_list_mean,
            ep_len_list_mean, 
            ep_arc_len_list_mean, 
            ep_shifting_numbers_list_mean, 
            ep_min_rad_list_mean, 
            ep_rad_numbers_list_mean, 
            # std
            success_ratio/eval_epoch, 
            ep_ret_list_var,
            ep_cost_list_var,
            ep_time_list_var,
            ep_len_list_var,
            ep_arc_len_list_var,
            ep_shifting_numbers_list_var,
            ep_min_rad_list_var,
            ep_rad_numbers_list_var )
            )
        
        record_flag = record_result_eval(
            algorithm = 'HPPO_'+ str(seed),
            env_id = env_id,
            eval_result = eval_result
        )
        
        eval_info = {}
        eval_info['eval_mean_ep_ret'] = mean_ep_ret/eval_epoch
        eval_info['eval_mean_ep_len'] = mean_ep_ret/eval_epoch
        eval_info['eval_mean_ep_cost'] = mean_ep_cost/eval_epoch
        eval_info['eval_success_ratio'] = success_ratio/eval_epoch       


        return eval_info

    
