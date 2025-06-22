from collections import namedtuple
from typing import Optional
from typing import Tuple

import sys

import gym
import gym_hybrid
import numpy as np
import cv2
import os
from gym import spaces
from gym.utils import seeding

from shapely.geometry import Point, Polygon
import math
import matplotlib.pyplot as plt
import pandas as pd
# import random
from easydict import EasyDict
from shapely.geometry import Point, Polygon, LineString

from .parking_environment_safe_core import plot_arrow, calculate_overlap_area
from .parking_environment_safe_core import distance_point_to_segment, point_to_polygon_distance
from .parking_environment_safe_core import ceter_to_rear, rear_to_ceter
from .parking_environment_safe_core import is_point_in_rectangle, position_vehicle_vertex
from .parking_environment_safe_core import env_stop_condition, compute_env_reward
from .parking_environment_safe_core import compute_vehicle_pos, compute_vehicle_pos_hybrid_com
from .parking_environment_safe_core import Perpendicular_parking, Perpendicular_parking_hybrid, Perpendicular_parking_hybrid_boundary
from .parking_environment_safe_core import Perpendicular_parking_hybrid_boundary
from .parking_environment_safe_core import generate_init_vehicle, get_init_vehicle_pos
from .parking_environment_safe_core import generate_target_parking,  get_target_parking_pos
from .parking_environment_safe_core import get_init_vehicle_area, vehicle_wall_obstacle, vehicle_wall_obstacle_noise
from .parking_environment_safe_core import ego_vehicle_vertex_midpoints, lidar_detection_distance
from .parking_environment_safe_core import Parallel_parking_hybrid_boundary, get_target_parking_pos_parallel
from .parking_environment_safe_core import vehicle_wall_obstacle_parallel, get_init_vehicle_area_parallel, vehicle_wall_obstacle_parallel_noise
from .parking_environment_safe_core import get_init_vehicle_pos_parallel, compute_env_reward_parallel


import warnings
warnings.filterwarnings("ignore")
from .agents import BaseAgent, MovingAgent, SlidingAgent

ACTION_Ratio = 1
# Discrete Action
ACTION ={'rightfront':0,
         'straightfront':1,
         'leftfront':2,
         'leftrear':3,
         'straightrear':4,
         'rightrear':5}

ACTION_COM ={'right':0,
            'straight':1,
            'left':2}

ACTION = EasyDict(ACTION)
ACTION_COM = EasyDict(ACTION_COM)
# The minimum radius
RADIUS = 5

def generate_discrete_points(start_point, heading_angle, radius, arc_length, interval, turn_direction=None):
    """
    生成弧线或直线上以固定间隔的离散点
    :param start_point: 起点坐标 (x, y)
    :param heading_angle: 航向角度(以度为单位,0 度为 x 轴正方向）
    :param radius: 圆弧或直线的半径
    :param arc_length: 弧长
    :param interval: 离散点的固定间隔
    :param turn_direction: 方向 0 表示右转,1 表示直行,2 表示左转
    :return: 离散点的坐标列表 [(x1, y1), (x2, y2), ...]
    """
    points = []
    x, y = start_point
    # heading_rad = np.radians(heading_angle)  # 将航向角转换为弧度
    heading_rad = heading_angle

    if turn_direction == 2 or turn_direction == 0:
        # 计算圆心位置
        if turn_direction == 2:  # 左转
            center_x = x - radius * np.sin(heading_rad)
            center_y = y + radius * np.cos(heading_rad)
            arc_angle = arc_length / radius
            start_angle = np.arctan2(y - center_y, x - center_x)
            theta = np.linspace(start_angle, start_angle + arc_angle, int(abs(arc_length) // interval) + 1)
            heading_theta = np.linspace(heading_rad, heading_rad + arc_angle, int(abs(arc_length) // interval) + 1)
        else:  # 右转
            center_x = x + radius * np.sin(heading_rad)
            center_y = y - radius * np.cos(heading_rad)
            arc_angle = arc_length / radius
            start_angle = np.arctan2(y - center_y, x - center_x)
            theta = np.linspace(start_angle, start_angle - arc_angle, int(abs(arc_length) // interval) + 1)
            heading_theta = np.linspace(heading_rad, heading_rad - arc_angle, int(abs(arc_length) // interval) + 1)
        
        # 生成弧线上的离散点
        # for angle in theta:
        #     px = center_x + radius * np.cos(angle)
        #     py = center_y + radius * np.sin(angle)
        #     points.append((px, py, angle))

        for i in range(len(theta)):
            px = center_x + radius * np.cos(theta[i])
            py = center_y + radius * np.sin(theta[i])
            points.append((px, py, heading_theta[i]))
    
    else:  # 直行
        num_points = int(abs(arc_length) // interval) + 1
        direction = 1 if arc_length >= 0 else -1
        dx = direction * interval * np.cos(heading_rad)
        dy = direction * interval * np.sin(heading_rad)
        
        for i in range(num_points):
            px = x + i * dx
            py = y + i * dy
            points.append((px, py, heading_rad))

    return points

def get_path_with_actions(start_points, instructions, interval=0.1):
    """
    根据指令生成并绘制路径上的离散点
    :param start_points: 初始点坐标和航向的列表 [(x1, y1, heading1), (x2, y2, heading2), ...]
    :param instructions: 每个点的指令列表 [(turn_direction1, arc_length1), ...]
    :param interval: 离散点间隔
    """
    all_points = []  # 用于存储所有离散点

    for (x, y, heading), instruction_set in zip(start_points, instructions):
        direction, arc_length = instruction_set
        radius = 5  # 固定半径
        points = generate_discrete_points((x, y), heading, radius, arc_length, interval, turn_direction=direction)
        all_points.extend(points)  # 记录离散点
    # 绘制所有离散点
    all_points = np.array(all_points)
    return all_points

class Action:
    """"
    Action class to store and standardize the action for the environment.
    """

    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters * ACTION_Ratio

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        # if len(self.parameters) == 2:
        #     return self.parameters[self.id]
        # else:
        #     if self.parameters[0] < 0:
        #         return 0 
        #     elif self.parameter[0]>2:
        #         return 2
        #     else:
        #         return self.parameters[0]
            
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]

class PerpendicularParkEnv_Safe(gym.Env):
    '''
        Create the RL environment for perpendicular parking environment
    '''
    def __init__(self):
        '''
        Init the environment
        '''
        super(PerpendicularParkEnv_Safe, self).__init__()
        self.width_park = 2.5    # the width of the parking spaces
        self.length_park = 5.3   # the legth of the parking spaces
        self.width_lane = 3.5    # the width of the lane
        self.width_vehicle = 2.0     # 车辆宽度
        self.length_vehicle = 4.3    # 车辆长度
        # self.width_vehicle = 1.8     # 车辆宽度
        # self.length_vehicle = 4.0   # 车辆长度
        self.parking_num = 4         # 停车位
        self.wheelbase = 2.7         # 轴距
        
        # self.action_dim = 6       # 动作维度
        self.action_dim = 3       # 动作维度
        self.state_dim = 17        # 观测状态维度
        # self.max_steps = 2000     # 环境最大步数
        # self.max_steps = 100
        self.max_steps = 50
        # self.max_steps = 30
        self.current_step = 0     # 初始化当前步数
        self.park_bounary = 0.5   # 停车场边界宽度

        # 车辆参数信息
        vehicle_config = {
            'vehicle_width': 2.0,
            'vehicle_length': 4.3,
            # 'vehicle_width': 1.8,
            # 'vehicle_length': 4.0,
            'wheelbase': 2.7
        }

        # 停车场信息
        parking_config = {
            'park_width': 2.5,
            'park_length': 5.3,
            'lane_width': 3.5,
            'parking_num': 4
        }
        self.vehicle_config = EasyDict(vehicle_config)
        self.parking_config = EasyDict(parking_config)

        self.ego_vehicle_area =  self.vehicle_config.vehicle_width* self.vehicle_config.vehicle_length

        # 初始化静态的库位信息
        '''
        self.parking_spaces: 停车位的库位信息
        self.lane_marks: 停车场车道线信息
        '''
        
        # self.parking_spaces, self.lane_marks, self.park_vertex = Perpendicular_parking_hybrid(
        #     width_park = self.width_park,
        #     length_park = self.length_park,
        #     width_lane = self.width_lane,
        #     parking_num = 4
        # )

        # 初始化停车场库位+
        self.parking_spaces, self.lane_marks, self.park_vertex = Perpendicular_parking_hybrid_boundary(
            width_park = self.width_park,
            length_park = self.length_park,
            width_lane = self.width_lane,
            parking_num = 4,
            boundary_width = self.park_bounary
        )
        
        # # 库位障碍停车场周围墙壁
        # self.park_vehicle_polygons, self.park_vertex_polygon = vehicle_wall_obstacle(
        #     parking_spaces = self.parking_spaces,
        #     park_vertex = self.park_vertex,
        #     vehicle_config = self.vehicle_config,
        # )

        # print('self.parking_spaces:', self.parking_spaces)
        # print('self.parking_spaces_len:', len(self.parking_spaces))
        # quit() 

        # # 可视化库位
        # self.render_parking(
        #     parking_spaces = self.parking_spaces, 
        #     lane_marks = self.lane_marks, 
        #     park_vertex = self.park_vertex
        # )
        
        # self.render_parking_trajectory(
        #     parking_spaces = self.parking_spaces,
        #     lane_marks = self.lane_marks,
        #     park_vertex = self.park_vertex
        #     # vehicle_area = self.vehicle_area
        # )

        # 初始化动作空间和状态空间
        # parameters_min = 0
        parameters_min = -1
        # parameters_max = 2
        parameters_max = 1
        self.action_space = spaces.Tuple((spaces.Discrete(self.action_dim), 
                                          spaces.Box(low=parameters_min, high=parameters_max)))
        self.observation_space = spaces.Box(low=-20, high=20, 
                                            shape=(self.state_dim,), dtype=np.float32)
        # print('init perpendicular parking environment')

    def seed(self, seed = 0):
        '''
        Set the seed for the environment
        '''
        np.random.seed(seed)
        return True
    
    def reset(self, train_stage=1, eval_stage=False):
        '''
        Reset the environment
        '''
        # reset the target parking spaces
        '''
        self.target_parking_id: 目标库位id
        self.target_pos: 目标库位中心位置及航向
        ''' 
        # 是否需要重置环境标志位        
        reset_flag = True 
        '''
        1. 外部初始化
        2. 或者初始化的环境中车辆位置与障碍物有重叠则表明不合理需要重置环境
        '''
        while reset_flag:
            # 随机产生目标库位
            # self.target_parking_id = random.randint(1,2)
            self.target_parking_id = 1
            self.target_pos, self.target_area = get_target_parking_pos(
                target_parking_id = self.target_parking_id,
                parking_spaces = self.parking_spaces
            )

            # 库位障碍停车场周围墙壁
            self.park_vehicle_polygons, self.park_vertex_polygon, self.park_spaces_polygons = vehicle_wall_obstacle(
                parking_spaces = self.parking_spaces,
                park_vertex = self.park_vertex,
                vehicle_config = self.vehicle_config,
                target_parking_id = self.target_parking_id,
            )

            # # 库位障碍停车场周围墙壁
            # self.park_vehicle_polygons, self.park_vertex_polygon, self.park_spaces_polygons = vehicle_wall_obstacle_noise(
            #     parking_spaces = self.parking_spaces,
            #     park_vertex = self.park_vertex,
            #     vehicle_config = self.vehicle_config,
            #     target_parking_id = self.target_parking_id,
            # )

            # reset the position of the ego vehicle 
            '''
            self.vehicle_pos: 主车的中心位置坐标及航向
            self.vehicle_vert: 主车四个顶点位置
            self.vehicle_pos_r: 主车后轴中心位置坐标及航向
            '''
            # area_id = random.randint(1, 2)
            if eval_stage == True:
                area_id = np.random.randint(3, 4)
            else:
                area_id = np.random.randint(1, 4)
            # area_id = 2
            # area_id = 1
            # area_id = np.random.randint(1, 3)
            # area_id = np.random.randint(2, 3)
            # area_id = np.random.randint(1, 2)
            # area_id = np.random.randint(1, 4)
        
            # 初始化车辆区域
            vehicle_area = get_init_vehicle_area(
                target_parking_id = self.target_parking_id,
                area_id = area_id,
                parking_config = self.parking_config,
                vehicle_config = self.vehicle_config,
                park_boundary = self.park_bounary
            )

            # vehicle_area1 = get_init_vehicle_area(
            #     target_parking_id = self.target_parking_id,
            #     area_id = 2,
            #     parking_config = self.parking_config,
            #     vehicle_config = self.vehicle_config,
            #     park_boundary = self.park_bounary
            # )

            # vehicle_area2 = get_init_vehicle_area(
            #     target_parking_id = self.target_parking_id,
            #     area_id = 3,
            #     parking_config = self.parking_config,
            #     vehicle_config = self.vehicle_config,
            #     park_boundary = self.park_bounary
            # )

            # self.render_parking_area(
            #     parking_spaces = self.parking_spaces,
            #     lane_marks = self.lane_marks,
            #     park_vertex = self.park_vertex,
            #     vehicle_area = vehicle_area,
            #     vehicle_area1 = vehicle_area1,
            #     vehicle_area2 = vehicle_area2
            # )

            # 获取车辆位置及航向
            self.vehicle_pos = get_init_vehicle_pos(
                vehicle_area = vehicle_area,
                vehicle_config = self.vehicle_config,
                area_id = area_id,
            )

            # 计算车辆初试位置到目标位置的距离
            self.init_ego_target_dis = math.sqrt(
                (self.vehicle_pos[0]-self.target_pos[0])**2+
                (self.vehicle_pos[1]-self.target_pos[1])**2
            ) 

            # 车辆顶点坐标
            self.vehicle_vert = position_vehicle_vertex(
                width_vehicle = self.width_vehicle,
                length_vehicle = self.length_vehicle,
                vehicle_pos = self.vehicle_pos
            )

            # 根据车辆中心和四个顶点获取polygons形式的四个顶点和四条边的中点
            self.vehicle_vert_polygons, self.vehicle_midpoints_polygons = ego_vehicle_vertex_midpoints(
                vehicle_vert = self.vehicle_vert
            )

            # 射线起点集合，四个顶点和四条边的中点
            self.vehicle_ray_start_points = self.vehicle_vert_polygons + self.vehicle_midpoints_polygons

            # 计算伪激光雷达到障碍物的距离
            self.obs_min_dis = lidar_detection_distance(
                vehicle_ray_start_points = self.vehicle_ray_start_points,
                park_vehicle_polygons = self.park_vehicle_polygons,
                park_vertex_polygon = self.park_vertex_polygon,
                vehicle_pos = self.vehicle_pos
            )

            # 根据车辆中心位置获取车辆后轴坐标
            self.vehicle_pos_r = ceter_to_rear(
                ceter_point = self.vehicle_pos, 
                wheelbase = self.wheelbase
            )

            need_reset = any(x < 0 for x in self.obs_min_dis)

            if need_reset:
                reset_flag = True
            else:
                reset_flag = False

        # self.render_parking_vehicle(
        #     parking_spaces = self.parking_spaces,
        #     lane_marks = self.lane_marks,
        #     vehicle_pos = self.vehicle_pos,
        #     vehicle_vert = self.vehicle_vert,
        #     target_parking_id = self.target_parking_id,
        #     target_pos = self.target_pos,
        #     vehicle_area = vehicle_area,
        #     park_vertex = self.park_vertex,
        #     park_vehicle_polygons = self.park_vehicle_polygons,
        #     vehicle_ray_start_points = self.vehicle_ray_start_points,
        #     obs_min_dis = self.obs_min_dis,
        # )

        # self.render_park_trajectory(
        #     parking_spaces = self.parking_spaces,
        #     lane_marks = self.lane_marks,
        #     vehicle_pos = self.vehicle_pos,
        #     vehicle_vert = self.vehicle_vert,
        #     target_parking_id = self.target_parking_id,
        #     target_pos = self.target_pos,
        #     vehicle_area = vehicle_area,
        #     park_vertex = self.park_vertex,
        #     park_vehicle_polygons = self.park_vehicle_polygons,
        #     vehicle_ray_start_points = self.vehicle_ray_start_points,
        #     obs_min_dis = self.obs_min_dis,
        # )
        # quit()

        # # 计算车辆四个顶点到停车场边界的距离
        # vert_mindis = []
        # for i in range(len(self.vehicle_vert[0])):
        #     px = self.vehicle_vert[0][i][0]
        #     py = self.vehicle_vert[0][i][1]
        #     # 判定车辆定点是否已经出界
        #     # Internal_points = is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex)
        #     if is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex):
        #         min_distance = point_to_polygon_distance(px, py, self.park_vertex)
        #     else:
        #         min_distance = - point_to_polygon_distance(px, py, self.park_vertex)
        #     vert_mindis.append(min_distance)
            
        # print('vert_mindis:', vert_mindis)
        # 主车与目标库位的位置和航向偏差
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the state after reset the environment
        # self.state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos)
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis).tolist()
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis)
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + self.obs_min_dis)
        self.current_step = 0
 
        return state
    
    def step(self, raw_action):
        '''
        Interact with the environment
        '''
        # action = action.tolist()
        self.current_step +=1     # 更新当前步数
        action = Action(*raw_action)
        '''
        计算下一个状态
        '''
        # print('self.vehicle_pos_r:', self.vehicle_pos_r)
        # Compute the position of the ego vehicle
        # self.vehicle_pos_r = compute_vehicle_pos_hybrid(
        #     vehicle_pos = self.vehicle_pos_r,
        #     action = action
        # )

        # # 计算后轴位置的转换
        # self.vehicle_pos_r = compute_vehicle_pos_hybrid_com(
        #     vehicle_pos = self.vehicle_pos_r,
        #     action = action
        # )

        # # 由车辆后轴计算车辆质心坐标
        # self.vehicle_pos = rear_to_ceter(
        #     rear_point = self.vehicle_pos_r,
        #     wheelbase = self.wheelbase
        # )

        self.vehicle_pos = compute_vehicle_pos_hybrid_com(
            vehicle_pos = self.vehicle_pos,
            action = action
        )

        # Transform the vertex of the ego vehicle
        self.vehicle_vert = position_vehicle_vertex(
            width_vehicle = self.width_vehicle,
            length_vehicle = self.length_vehicle,
            vehicle_pos = self.vehicle_pos
        )

        # 根据车辆中心和四个顶点获取polygons形式的四个顶点和四条边的中点
        self.vehicle_vert_polygons, self.vehicle_midpoints_polygons = ego_vehicle_vertex_midpoints(
            vehicle_vert = self.vehicle_vert
        )

        # 射线起点集合，四个顶点和四条边的中点
        self.vehicle_ray_start_points = self.vehicle_vert_polygons + self.vehicle_midpoints_polygons

        # 计算伪激光雷达到障碍物的距离
        self.obs_min_dis = lidar_detection_distance(
            vehicle_ray_start_points = self.vehicle_ray_start_points,
            park_vehicle_polygons = self.park_vehicle_polygons,
            park_vertex_polygon = self.park_vertex_polygon,
            vehicle_pos = self.vehicle_pos
        )


        # # 计算车辆四个顶点到停车场边界的距离
        # vert_mindis = []
        # for i in range(len(self.vehicle_vert[0])):
        #     px = self.vehicle_vert[0][i][0]
        #     py = self.vehicle_vert[0][i][1]
        #     # min_distance = point_to_polygon_distance(px, py, self.park_vertex)
        #     if is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex):
        #         min_distance = point_to_polygon_distance(px, py, self.park_vertex)
        #     else:
        #         min_distance = - point_to_polygon_distance(px, py, self.park_vertex)
        #     vert_mindis.append(min_distance)
        
        # The bias of the position and directory between the ego vehicle and the target parking space
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the new state
        # self.state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos)
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis).tolist()
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis)
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + self.obs_min_dis)

        '''
        计算环境反馈的奖励、终止条件
        '''
        # Compute the reward and done
        info = compute_env_reward(
            vehicle_pos = self.vehicle_pos,
            target_pos = self.target_pos,
            parking_spaces = self.parking_spaces,
            vehicle_vertex = self.vehicle_vert,
            park_vertex = self.park_vertex,
            target_area = self.target_area,
            ego_vehicle_area = self.ego_vehicle_area,
            init_ego_target_dis = self.init_ego_target_dis,
            current_step = self.current_step,
            max_step = self.max_steps,
            obs_min_dis = self.obs_min_dis,
        )

        '''
        判定是否超过最大步数
        '''
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        '''
        设置环境done
        '''
        info['truncated'] = truncated
        done = info['truncated'] or info['terminated']

        '''
        环境终止输出最终步数
        '''
        # if done:
        #     print('finished step:', self.current_step)

        # return self.state, info['reward'], done, info
        return state, info['reward'], info['cost'], done, info 

    def close(self):
        '''
        Close the environment
        '''
        return True    
    
    def get_env_set(self):
        '''
        Get the parameters of the current environment
        
        '''
        env_set = {}
        env_set['vehicle_pos'] = self.vehicle_pos
        env_set['target_pos'] = self.target_pos
        env_set['obstacles'] = self.park_vehicle_polygons
        env_set['park_edge'] = self.park_vertex_polygon
        env_set['vehicle_config'] = self.vehicle_config
        env_set['park_spaces'] = self.park_spaces_polygons
        env_set['lane_mark'] = self.lane_marks
        env_set['ray_start_points'] = self.vehicle_ray_start_points
        env_set['vehicle_vert'] = self.vehicle_vert
        env_set['vehicle_pos'] = self.vehicle_pos
        env_set['obs_min_dis'] = self.obs_min_dis

        return env_set

    def render_parking(self,parking_spaces,lane_marks,park_vertex):
        '''
        Render the parking scenarios
        '''
        save_dir = './fig/init_parking_v01'

        # 绘制库位线
        plt.figure(figsize=(10,17.5))
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)
        
        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=3.0)

        # 绘制停车场边界
        for i in range(len(park_vertex)):
            if i == len(park_vertex)-1:
                x = [park_vertex[i][0],park_vertex[0][0]]
                y = [park_vertex[i][1],park_vertex[0][1]]
            else:
                x = [park_vertex[i][0],park_vertex[i+1][0]]
                y = [park_vertex[i][1],park_vertex[i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=4.0)

     
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')

    def render_parking_area(self,parking_spaces,lane_marks,park_vertex,vehicle_area, vehicle_area1, vehicle_area2):
        '''
        Render the parking scenarios
        '''
        save_dir = './fig/init_parking_area_v01'

        # 绘制库位线
        plt.figure(figsize=(10,17.5))
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)
        
        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=3.0)

        # 绘制车辆区域
        for i in range(len(vehicle_area[0])):
            # 最后一条边
            if i == len(vehicle_area[0])-1:
                x = [vehicle_area[0][i][0], vehicle_area[0][0][0]]
                y = [vehicle_area[0][i][1], vehicle_area[0][0][1]]
            # 其他边
            else:
                x = [vehicle_area[0][i][0], vehicle_area[0][i+1][0]]
                y = [vehicle_area[0][i][1], vehicle_area[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-.', color='black', linewidth=3.0)
            # 车辆区域补充颜色
            x = [vehicle_area[0][0][0], vehicle_area[0][1][0]]
            y1 = [vehicle_area[0][0][1], vehicle_area[0][1][1]]
            y2 = [vehicle_area[0][2][1], vehicle_area[0][3][1]]
            plt.fill_between(x,y1,y2,color='grey')


        # 绘制车辆区域
        for i in range(len(vehicle_area1[0])):
            # 最后一条边
            if i == len(vehicle_area1[0])-1:
                x = [vehicle_area1[0][i][0], vehicle_area1[0][0][0]]
                y = [vehicle_area1[0][i][1], vehicle_area1[0][0][1]]
            # 其他边
            else:
                x = [vehicle_area1[0][i][0], vehicle_area1[0][i+1][0]]
                y = [vehicle_area1[0][i][1], vehicle_area1[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-.', color='black', linewidth=3.0)
            # 车辆区域补充颜色
            # x = [vehicle_area1[0][0][0], vehicle_area1[0][1][0]]
            # y1 = [vehicle_area1[0][0][1], vehicle_area1[0][1][1]]
            # y2 = [vehicle_area1[0][2][1], vehicle_area1[0][3][1]]
            # plt.fill_between(x,y1,y2,color='grey')


        # 绘制车辆区域
        for i in range(len(vehicle_area2[0])):
            # 最后一条边
            if i == len(vehicle_area2[0])-1:
                x = [vehicle_area2[0][i][0], vehicle_area2[0][0][0]]
                y = [vehicle_area2[0][i][1], vehicle_area2[0][0][1]]
            # 其他边
            else:
                x = [vehicle_area2[0][i][0], vehicle_area2[0][i+1][0]]
                y = [vehicle_area2[0][i][1], vehicle_area2[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-.', color='black', linewidth=3.0)
            # 车辆区域补充颜色
            x = [vehicle_area2[0][0][0], vehicle_area2[0][1][0]]
            y1 = [vehicle_area2[0][0][1], vehicle_area2[0][1][1]]
            y2 = [vehicle_area2[0][2][1], vehicle_area2[0][3][1]]
            plt.fill_between(x,y1,y2,color='grey')

        # 绘制停车场边界
        for i in range(len(park_vertex)):
            if i == len(park_vertex)-1:
                x = [park_vertex[i][0],park_vertex[0][0]]
                y = [park_vertex[i][1],park_vertex[0][1]]
            else:
                x = [park_vertex[i][0],park_vertex[i+1][0]]
                y = [park_vertex[i][1],park_vertex[i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=4.0)

     
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')

    def render_parking_vehicle(
            self,
            parking_spaces,
            lane_marks,
            vehicle_pos, 
            vehicle_vert, 
            target_parking_id,
            target_pos,
            vehicle_area,
            park_vertex,
            park_vehicle_polygons,
            vehicle_ray_start_points,
            obs_min_dis,
    ):
        '''
        Render the init parking scenarios and vehicle.
        '''
        save_dir = './fig/parking_area_vehicle_v04'
        plt.figure(figsize=(10,12.5))

        # 车辆中心坐标
        vehicle_center = Point(vehicle_pos[0], vehicle_pos[1])
        ray_length = 10

        # 绘制停车场边界
        park_vertex_polygons = Polygon(park_vertex)
        park_x, park_y = park_vertex_polygons.exterior.xy
        plt.fill(park_x, park_y, label='parking lot', color='lightgrey')
        
        # 绘制停车区域
        parking_spaces_polygons = []
        for i in range(len(parking_spaces)):
            parking_polygon = Polygon(
                [(parking_spaces[i][0][0], parking_spaces[i][0][1]),
                (parking_spaces[i][1][0], parking_spaces[i][1][1]),
                (parking_spaces[i][2][0], parking_spaces[i][2][1]),
                (parking_spaces[i][3][0], parking_spaces[i][3][1])]
            )
            parking_spaces_polygons.append(parking_polygon)

        for parking_spaces_polygon in parking_spaces_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                parking_area_x, parking_area_y = parking_spaces_polygon.exterior.xy
                plt.fill(parking_area_x, parking_area_y, alpha = 0.5, label='parking area', color ='lightblue')

        # 绘制停车库位线
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=1.0)


        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=2.0)

        # 绘制已停车的车辆
        for park_vehicle_polygon in park_vehicle_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                park_vehicle_x, park_vehicle_y = park_vehicle_polygon.exterior.xy
                plt.fill(park_vehicle_x, park_vehicle_y, alpha = 0.5, label='park vehicle', color ='salmon')

        # 绘制车辆初试位置及边框
        vehicle_vert_polygon = Polygon(
            [(vehicle_vert[0][0][0], vehicle_vert[0][0][1]),
            (vehicle_vert[0][1][0], vehicle_vert[0][1][1]),
            (vehicle_vert[0][2][0], vehicle_vert[0][2][1]),
            (vehicle_vert[0][3][0], vehicle_vert[0][3][1])]
        )
        vehicle_x, vehicle_y = vehicle_vert_polygon.exterior.xy
        plt.fill(vehicle_x, vehicle_y, label='parking lot', color='lightgreen')
        # 绘制车辆中心位置
        plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # 绘制车辆航向
        plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # 绘制伪激光雷达的线
        id = 0
        for start_point in vehicle_ray_start_points:

            # 计算从车辆中心到起点的角度
            angle = np.arctan2(start_point.y - vehicle_center.y, start_point.x - vehicle_center.x)

            # 计算射线终点
            end_x = start_point.x + obs_min_dis[id]*np.cos(angle)
            end_y = start_point.y + obs_min_dis[id]*np.sin(angle)

            # 绘制射线
            plt.plot([start_point.x, end_x], [start_point.y, end_y], color='blue', alpha=0.5)
            id += 1

        plt.grid()
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')

    def render_park_trajectory(
            self,
            parking_spaces,
            lane_marks,
            vehicle_pos, 
            vehicle_vert, 
            target_parking_id,
            target_pos,
            vehicle_area,
            park_vertex,
            park_vehicle_polygons,
            vehicle_ray_start_points,
            obs_min_dis,
    ):
        '''
        Render the parking scenarios and parking trajectory.
        '''
        save_dir = './perpendicular_fig/eval_tra/trajectory_20241115_234904'

        plt.figure(figsize=(10,12.5))

        # # 车辆中心坐标
        # vehicle_center = Point(vehicle_pos[0], vehicle_pos[1])

        # 绘制停车场边界
        park_vertex_polygons = Polygon(park_vertex)
        park_x, park_y = park_vertex_polygons.exterior.xy
        plt.fill(park_x, park_y, label='parking lot', color='lightgrey')
        
        # 绘制停车区域
        parking_spaces_polygons = []
        for i in range(len(parking_spaces)):
            parking_polygon = Polygon(
                [(parking_spaces[i][0][0], parking_spaces[i][0][1]),
                (parking_spaces[i][1][0], parking_spaces[i][1][1]),
                (parking_spaces[i][2][0], parking_spaces[i][2][1]),
                (parking_spaces[i][3][0], parking_spaces[i][3][1])]
            )
            parking_spaces_polygons.append(parking_polygon)

        for parking_spaces_polygon in parking_spaces_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                parking_area_x, parking_area_y = parking_spaces_polygon.exterior.xy
                plt.fill(parking_area_x, parking_area_y, alpha = 0.5, label='parking area', color ='lightblue')

        # 绘制停车库位线
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=1.0)


        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=2.0)

        # 绘制已停车的车辆
        for park_vehicle_polygon in park_vehicle_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                park_vehicle_x, park_vehicle_y = park_vehicle_polygon.exterior.xy
                plt.fill(park_vehicle_x, park_vehicle_y, alpha = 0.5, label='park vehicle', color ='salmon')

        # # 绘制车辆初试位置及边框
        # vehicle_vert_polygon = Polygon(
        #     [(vehicle_vert[0][0][0], vehicle_vert[0][0][1]),
        #     (vehicle_vert[0][1][0], vehicle_vert[0][1][1]),
        #     (vehicle_vert[0][2][0], vehicle_vert[0][2][1]),
        #     (vehicle_vert[0][3][0], vehicle_vert[0][3][1])]
        # )
        # vehicle_x, vehicle_y = vehicle_vert_polygon.exterior.xy
        # plt.fill(vehicle_x, vehicle_y, label='parking lot', color='lightgreen')
        # # 绘制车辆中心位置
        # plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # # 绘制车辆航向
        # plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # 绘制泊车轨迹
        data_dir = './eval_log_3/obs_act_rew_20241115_234904.csv' 
        data = pd.read_csv(data_dir)    # 读取CSV文件        
        traj_x = data['pos_x'].tolist()
        traj_y = data['pos_y'].tolist()
        traj_dir = data['pos_dir'].tolist()
        action_type = data['action_type'].tolist()
        action_args = data['action_args'].tolist()
        # print('traj_x:', traj_x)
        # quit()

        # 计算车辆轨迹的中心点
        start_points_traj = []
        actions = []
        for i in range(len(traj_x)-1):
            start_points_traj.append((traj_x[i], traj_y[i], traj_dir[i]))
            actions.append((action_type[i], action_args[i]))

        all_points = get_path_with_actions(start_points_traj, actions)
        plt.scatter(all_points[:, 0], all_points[:, 1], s= 5)


        # 计算车辆轨迹的车辆顶点
        path_vehicle_verts = []
        for i in range(len(all_points)):
            point_vehicle_vert = position_vehicle_vertex(
                width_vehicle = self.vehicle_config.vehicle_width,
                length_vehicle = self.vehicle_config.vehicle_length,
                vehicle_pos = all_points[i]
            )
            path_vehicle_verts.append(point_vehicle_vert[0])


        # 绘制车辆轨迹的边框
        for i in range(len(path_vehicle_verts)):
            for j in range(len(path_vehicle_verts[i])):
                if j == len(path_vehicle_verts[i])-1:
                    x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][0][0]]
                    y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][0][1]]
                else:
                    x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][j+1][0]]
                    y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=0.5)


        # 绘制初始化车辆
        vehicle_pos = [traj_x[0], traj_y[0], traj_dir[0]]
        vehicle_vert = position_vehicle_vertex(width_vehicle = self.width_vehicle, 
                                                 length_vehicle = self.length_vehicle, 
                                                 vehicle_pos = vehicle_pos )
        for i in range(len(vehicle_vert[0])):
            # 最后一条边
            if i == len(vehicle_vert[0])-1:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][0][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][i+1][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
            # 绘制车辆中心位置
        plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # 绘制车辆航向
        plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # # 绘制最终车辆位置
        final_vehicle_pos = [traj_x[-1], traj_y[-1], traj_dir[-1]]
        final_vehicle_vert = position_vehicle_vertex(width_vehicle = self.width_vehicle, 
                                                     length_vehicle = self.length_vehicle, 
                                                     vehicle_pos = final_vehicle_pos)
        for i in range(len(final_vehicle_vert[0])):
            # 最后一条边
            if i == len(final_vehicle_vert[0])-1:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][0][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][i+1][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
        # 绘制车辆中心位置
        plt.plot(final_vehicle_pos[0], final_vehicle_pos[1], marker='o', c='k',markersize=10)  
        # 绘制车辆航向
        plot_arrow(final_vehicle_pos[0], final_vehicle_pos[1], final_vehicle_pos[2], length=2.0, color='green')  

        plt.grid()
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')
        # quit()

    def render_init_parking(self,parking_spaces, lane_marks, vehicle_pos, vehicle_vert, target_parking_id, target_pos, vehicle_area):
        '''
        Render the init parking scenarios
        '''
        save_dir = './fig/parking_vehicle_pos_02'

        # 绘制库位线
        plt.figure(figsize=(10,17.5))
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)
        
        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=3.0)

        # 绘制车辆区域
        for i in range(len(vehicle_area[0])):
            # 最后一条边
            if i == len(vehicle_area[0])-1:
                x = [vehicle_area[0][i][0], vehicle_area[0][0][0]]
                y = [vehicle_area[0][i][1], vehicle_area[0][0][1]]
            # 最后一条边
            else:
                x = [vehicle_area[0][i][0], vehicle_area[0][i+1][0]]
                y = [vehicle_area[0][i][1], vehicle_area[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-.', color='black', linewidth=3.0)
            # 车辆区域补充颜色
            x = [vehicle_area[0][0][0], vehicle_area[0][1][0]]
            y1 = [vehicle_area[0][0][1], vehicle_area[0][1][1]]
            y2 = [vehicle_area[0][2][1], vehicle_area[0][3][1]]
            plt.fill_between(x,y1,y2,color='grey')  

        # 绘制初试化车辆
        for i in range(len(vehicle_vert[0])):
            # 最后一条边
            if i == len(vehicle_vert[0])-1:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][0][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][i+1][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)
            # 车辆补充颜色
            # x = [vehicle_vert[0][0][0], vehicle_vert[0][1][0]]
            # y1 = [vehicle_vert[0][0][1], vehicle_vert[0][1][1]]
            # y2 = [vehicle_vert[0][2][1], vehicle_vert[0][3][1]]
            # plt.fill_between(x,y1,y2,color='turquoise')
            # 绘制车辆中心位置
            plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)


        # 绘制目标车位
        for j in range(len(parking_spaces[target_parking_id])):
            if j == len(parking_spaces[target_parking_id])-1:
                x = [parking_spaces[target_parking_id][j][0],parking_spaces[target_parking_id][0][0]]
                y = [parking_spaces[target_parking_id][j][1],parking_spaces[target_parking_id][0][1]]
            else:
                x = [parking_spaces[target_parking_id][j][0],parking_spaces[target_parking_id][j+1][0]]
                y = [parking_spaces[target_parking_id][j][1],parking_spaces[target_parking_id][j+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)
            # 目标库位填充颜色
            x = [parking_spaces[target_parking_id][0][0], parking_spaces[target_parking_id][1][0]]
            y1 = [parking_spaces[target_parking_id][0][1], parking_spaces[target_parking_id][1][1]]
            y2 = [parking_spaces[target_parking_id][2][1], parking_spaces[target_parking_id][3][1]]
            plt.fill_between(x,y1,y2,color='limegreen')
            # 绘制库位中心位置
            plt.plot(target_pos[0], target_pos[1], marker='*', c='k', markersize=15) 

        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')

        return True
        
    def render_parking_trajectory(self,parking_spaces,lane_marks, park_vertex):
        '''
        Render the parking scenarios
        '''
        save_dir = './fig_3/eval_fig_tra/trajectory_20241019_121240'

        # 绘制库位线
        plt.figure(figsize=(10,17.5))
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=3.0)

        # 绘制停车场边界
        for i in range(len(park_vertex)):
            if i == len(park_vertex)-1:
                x = [park_vertex[i][0],park_vertex[0][0]]
                y = [park_vertex[i][1],park_vertex[0][1]]
            else:
                x = [park_vertex[i][0],park_vertex[i+1][0]]
                y = [park_vertex[i][1],park_vertex[i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=4.0)
        
        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=3.0)

        # 绘制泊车轨迹
        data_dir = './eval_log_3/obs_act_rew_20241019_121240.csv' 
        data = pd.read_csv(data_dir)    # 读取CSV文件        
        traj_x = data['pos_x'].tolist()
        traj_y = data['pos_y'].tolist()
        traj_dir = data['pos_dir'].tolist()
        plt.scatter(traj_x, traj_y)    # 绘制离散点图


        # 绘制初始化车辆
        vehicle_pos = [traj_x[0], traj_y[0], traj_dir[0]]
        vehicle_vert = position_vehicle_vertex(width_vehicle = self.width_vehicle, 
                                                 length_vehicle = self.length_vehicle, 
                                                 vehicle_pos = vehicle_pos )
        for i in range(len(vehicle_vert[0])):
            # 最后一条边
            if i == len(vehicle_vert[0])-1:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][0][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][i+1][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
            # 绘制车辆中心位置
        plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # 绘制车辆航向
        plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # # 绘制最终车辆位置
        final_vehicle_pos = [traj_x[-1], traj_y[-1], traj_dir[-1]]
        final_vehicle_vert = position_vehicle_vertex(width_vehicle = self.width_vehicle, 
                                                     length_vehicle = self.length_vehicle, 
                                                     vehicle_pos = final_vehicle_pos)
        for i in range(len(final_vehicle_vert[0])):
            # 最后一条边
            if i == len(final_vehicle_vert[0])-1:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][0][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][i+1][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
        # 绘制车辆中心位置
        plt.plot(final_vehicle_pos[0], final_vehicle_pos[1], marker='o', c='k',markersize=10)  
        # 绘制车辆航向
        plot_arrow(final_vehicle_pos[0], final_vehicle_pos[1], final_vehicle_pos[2], length=2.0, color='green')  
             
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')

    
class ParallelParkEnv_Safe(gym.Env):
    '''
        Create the RL environment for parallel parking environment
    '''
    def __init__(self):
        '''
        Init the environment
        '''
        super(ParallelParkEnv_Safe, self).__init__()
       
        self.action_dim = 3       # 离散动作维度
        self.state_dim = 17       # 观测状态维度
        self.max_steps = 60       # 环境最大步数
        self.current_step = 0     # 初始化当前步数
        self.park_bounary = 0.5   # 停车场边界宽度

        # 车辆参数信息
        vehicle_config = {
            'vehicle_width': 2.0,     # 车辆宽度
            'vehicle_length': 4.3,    # 车辆长度
            'wheelbase': 2.7          # 轴距
        }

        # 停车场信息
        parking_config = {
            'park_width': 2.5,      # 车位宽度
            # 'park_length': 5.3,     # 车位长度
            'park_length': 6.2,     # 车位长度
            'lane_width': 3.5,      # 车道宽度
            'parking_num': 3,       # 单边停车位数
            'parking_boundary': 0.5, # 停车场边界间隙
        }
        self.vehicle_config = EasyDict(vehicle_config)
        self.parking_config = EasyDict(parking_config)

        self.ego_vehicle_area =  self.vehicle_config.vehicle_width* self.vehicle_config.vehicle_length

        # 初始化停车场库位       
        self.parking_spaces, self.lane_marks, self.park_vertex = Parallel_parking_hybrid_boundary(
            vehicle_config = self.vehicle_config,
            parking_config = self.parking_config,
        )


        # 初始化动作空间和状态空间
        parameters_min = -1  # 连续参数的最小值
        parameters_max = 1   # 连续参数的最大值
        # 动作空间
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.action_dim), 
            spaces.Box(low=parameters_min, high=parameters_max)
        ))
        # 观测空间
        self.observation_space = spaces.Box(
            low=-20, high=20, 
            shape=(self.state_dim,), dtype=np.float32
        )

    def seed(self, seed = 0):
        '''
        Set the seed for the environment
        '''
        np.random.seed(seed)
        return True
    

    def reset(self, train_stage=1, eval_stage=False):
        '''
        Reset the environment
        '''
        # reset the target parking spaces
        '''
        self.target_parking_id: 目标库位id
        self.target_pos: 目标库位中心位置及航向
        ''' 
        # 是否需要重置环境标志位        
        reset_flag = True 
        '''
        1. 外部初始化
        2. 或者初始化的环境中车辆位置与障碍物有重叠则表明不合理需要重置环境
        '''
        while reset_flag:
            # 随机产生目标库位
            # self.target_parking_id = random.randint(1,2)
            self.target_parking_id = 1
            self.target_pos, self.target_area = get_target_parking_pos_parallel(
                target_parking_id = self.target_parking_id,
                parking_spaces = self.parking_spaces
            )

            # # 库位障碍及停车场周围墙壁
            # self.park_vehicle_polygons, self.park_vertex_polygon, self.park_spaces_polygons = vehicle_wall_obstacle_parallel(
            #     parking_spaces = self.parking_spaces,
            #     park_vertex = self.park_vertex,
            #     vehicle_config = self.vehicle_config,
            #     target_parking_id = self.target_parking_id,
            # )

            # 库位障碍及停车场周围墙壁
            self.park_vehicle_polygons, self.park_vertex_polygon, self.park_spaces_polygons = vehicle_wall_obstacle_parallel_noise(
                parking_spaces = self.parking_spaces,
                park_vertex = self.park_vertex,
                vehicle_config = self.vehicle_config,
                target_parking_id = self.target_parking_id,
            )

            # reset the position of the ego vehicle 
            '''
            self.vehicle_pos: 主车的中心位置坐标及航向
            self.vehicle_vert: 主车四个顶点位置
            self.vehicle_pos_r: 主车后轴中心位置坐标及航向
            '''
            # area_id = random.randint(1, 2)
            # if eval_stage == True:
            #     area_id = np.random.randint(2, 3)
            # else:
            #     area_id = np.random.randint(2, 3)

            if eval_stage == True:
                area_id = np.random.randint(3, 4)
            else:
                area_id = np.random.randint(1, 4)
        
            # 初始化车辆区域
            vehicle_area = get_init_vehicle_area_parallel(
                target_parking_id = self.target_parking_id,
                area_id = area_id,
                parking_config = self.parking_config,
                vehicle_config = self.vehicle_config,
                park_boundary = self.park_bounary
            )

            # 获取车辆位置及航向
            self.vehicle_pos = get_init_vehicle_pos_parallel(
                vehicle_area = vehicle_area,
                vehicle_config = self.vehicle_config,
                area_id = area_id,
            )

            # 计算车辆初试位置到目标位置的距离
            self.init_ego_target_dis = math.sqrt(
                (self.vehicle_pos[0]-self.target_pos[0])**2+
                (self.vehicle_pos[1]-self.target_pos[1])**2
            )    

            # 车辆顶点坐标
            self.vehicle_vert = position_vehicle_vertex(
                width_vehicle = self.vehicle_config.vehicle_width,
                length_vehicle = self.vehicle_config.vehicle_length,
                vehicle_pos = self.vehicle_pos
            )

            # 根据车辆中心和四个顶点获取polygons形式的四个顶点和四条边的中点
            self.vehicle_vert_polygons, self.vehicle_midpoints_polygons = ego_vehicle_vertex_midpoints(
                vehicle_vert = self.vehicle_vert
            )

            # 射线起点集合，四个顶点和四条边的中点
            self.vehicle_ray_start_points = self.vehicle_vert_polygons + self.vehicle_midpoints_polygons

            # 计算伪激光雷达到障碍物的距离
            self.obs_min_dis = lidar_detection_distance(
                vehicle_ray_start_points = self.vehicle_ray_start_points,
                park_vehicle_polygons = self.park_vehicle_polygons,
                park_vertex_polygon = self.park_vertex_polygon,
                vehicle_pos = self.vehicle_pos
            )

            # # 根据车辆中心位置获取车辆后轴坐标
            # self.vehicle_pos_r = ceter_to_rear(
            #     ceter_point = self.vehicle_pos, 
            #     wheelbase = self.vehicle_config.wheelbase
            # )

            need_reset = any(x < 0 for x in self.obs_min_dis)

            if need_reset:
                reset_flag = True
            else:
                reset_flag = False

        # # 绘制初始化车辆位置
        # self.render_parking_vehicle(
        #     parking_spaces = self.parking_spaces,
        #     lane_marks = self.lane_marks,
        #     vehicle_pos = self.vehicle_pos,
        #     vehicle_vert = self.vehicle_vert,
        #     target_parking_id = self.target_parking_id,
        #     target_pos = self.target_pos,
        #     vehicle_area = vehicle_area,
        #     park_vertex = self.park_vertex,
        #     park_vehicle_polygons = self.park_vehicle_polygons,
        #     vehicle_ray_start_points = self.vehicle_ray_start_points,
        #     obs_min_dis = self.obs_min_dis,
        # )

        # # # 绘制轨迹信息
        # self.render_park_trajectory(
        #     parking_spaces = self.parking_spaces,
        #     lane_marks = self.lane_marks,
        #     vehicle_pos = self.vehicle_pos,
        #     vehicle_vert = self.vehicle_vert,
        #     target_parking_id = self.target_parking_id,
        #     target_pos = self.target_pos,
        #     vehicle_area = vehicle_area,
        #     park_vertex = self.park_vertex,
        #     park_vehicle_polygons = self.park_vehicle_polygons,
        #     vehicle_ray_start_points = self.vehicle_ray_start_points,
        #     obs_min_dis = self.obs_min_dis,
        # )
        

        # 主车与目标库位的位置和航向偏差
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the state after reset the environment
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + self.obs_min_dis)
        self.current_step = 0
 
        return state
    
    def step(self, raw_action):
        '''
        Interact with the environment
        '''
        # action = action.tolist()
        self.current_step +=1     # 更新当前步数
        action = Action(*raw_action)
        '''
        计算下一个状态
        '''
        self.vehicle_pos = compute_vehicle_pos_hybrid_com(
            vehicle_pos = self.vehicle_pos,
            action = action
        )

        # Transform the vertex of the ego vehicle
        self.vehicle_vert = position_vehicle_vertex(
            width_vehicle = self.vehicle_config.vehicle_width,
            length_vehicle = self.vehicle_config.vehicle_length,
            vehicle_pos = self.vehicle_pos
        )

        # 根据车辆中心和四个顶点获取polygons形式的四个顶点和四条边的中点
        self.vehicle_vert_polygons, self.vehicle_midpoints_polygons = ego_vehicle_vertex_midpoints(
            vehicle_vert = self.vehicle_vert
        )

        # 射线起点集合，四个顶点和四条边的中点
        self.vehicle_ray_start_points = self.vehicle_vert_polygons + self.vehicle_midpoints_polygons

        # 计算伪激光雷达到障碍物的距离
        self.obs_min_dis = lidar_detection_distance(
            vehicle_ray_start_points = self.vehicle_ray_start_points,
            park_vehicle_polygons = self.park_vehicle_polygons,
            park_vertex_polygon = self.park_vertex_polygon,
            vehicle_pos = self.vehicle_pos
        )
       
        # The bias of the position and directory between the ego vehicle and the target parking space
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the new state
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + self.obs_min_dis)

        '''
        计算环境反馈的奖励、终止条件
        '''
        # Compute the reward and done
        info = compute_env_reward_parallel(
            vehicle_pos = self.vehicle_pos,
            target_pos = self.target_pos,
            parking_spaces = self.parking_spaces,
            vehicle_vertex = self.vehicle_vert,
            park_vertex = self.park_vertex,
            target_area = self.target_area,
            ego_vehicle_area = self.ego_vehicle_area,
            init_ego_target_dis = self.init_ego_target_dis,
            current_step = self.current_step,
            max_step = self.max_steps,
            obs_min_dis = self.obs_min_dis,
        )

        '''
        判定是否超过最大步数
        '''
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        '''
        设置环境done
        '''
        info['truncated'] = truncated
        done = info['truncated'] or info['terminated']

        '''
        环境终止输出最终步数
        '''
        # if done:
        #     print('finished step:', self.current_step)

        # return self.state, info['reward'], done, info
        return state, info['reward'], info['cost'], done, info 
    

    def close(self):
        '''
        Close the environment
        '''
        return True
    
    
    # def get_env_set(self):
    #     '''
    #     Get the parameters of the current environment        
    #     '''
    #     env_set = {}
    #     env_set['vehicle_pos'] = self.vehicle_pos
    #     env_set['target_pos'] = self.target_pos
    #     env_set['obstacles'] = self.park_vehicle_polygons
    #     env_set['park_edge'] = self.park_vertex_polygon
    #     env_set['vehicle_config'] = self.vehicle_config
    #     env_set['park_spaces'] = self.park_spaces_polygons
    #     env_set['lane_mark'] = self.lane_marks

    #     return env_set
    
    def get_env_set(self):
        '''
        Get the parameters of the current environment
        
        '''
        env_set = {}
        env_set['vehicle_pos'] = self.vehicle_pos
        env_set['target_pos'] = self.target_pos
        env_set['obstacles'] = self.park_vehicle_polygons
        env_set['park_edge'] = self.park_vertex_polygon
        env_set['vehicle_config'] = self.vehicle_config
        env_set['park_spaces'] = self.park_spaces_polygons
        env_set['lane_mark'] = self.lane_marks
        env_set['ray_start_points'] = self.vehicle_ray_start_points
        env_set['vehicle_vert'] = self.vehicle_vert
        env_set['vehicle_pos'] = self.vehicle_pos
        env_set['obs_min_dis'] = self.obs_min_dis

        return env_set
    
    
    def render_parking_vehicle(
            self,
            parking_spaces,
            lane_marks,
            vehicle_pos, 
            vehicle_vert, 
            target_parking_id,
            target_pos,
            vehicle_area,
            park_vertex,
            park_vehicle_polygons,
            vehicle_ray_start_points,
            obs_min_dis,
    ):
        '''
        Render the init parking scenarios and vehicle.
        '''
        save_dir = './fig/parallel_parking_area_vehicle_v06'
        plt.figure(figsize=(10,12.5))

        # 车辆中心坐标
        vehicle_center = Point(vehicle_pos[0], vehicle_pos[1])
        ray_length = 10

        # 绘制停车场边界
        park_vertex_polygons = Polygon(park_vertex)
        park_x, park_y = park_vertex_polygons.exterior.xy
        plt.fill(park_x, park_y, label='parking lot', color='lightgrey')
        
        # 绘制停车区域
        parking_spaces_polygons = []
        for i in range(len(parking_spaces)):
            parking_polygon = Polygon(
                [(parking_spaces[i][0][0], parking_spaces[i][0][1]),
                (parking_spaces[i][1][0], parking_spaces[i][1][1]),
                (parking_spaces[i][2][0], parking_spaces[i][2][1]),
                (parking_spaces[i][3][0], parking_spaces[i][3][1])]
            )
            parking_spaces_polygons.append(parking_polygon)

        for parking_spaces_polygon in parking_spaces_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                parking_area_x, parking_area_y = parking_spaces_polygon.exterior.xy
                plt.fill(parking_area_x, parking_area_y, alpha = 0.5, label='parking area', color ='lightblue')

        # 绘制停车库位线
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=1.0)


        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=2.0)

        # 绘制已停车的车辆
        for park_vehicle_polygon in park_vehicle_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                park_vehicle_x, park_vehicle_y = park_vehicle_polygon.exterior.xy
                plt.fill(park_vehicle_x, park_vehicle_y, alpha = 0.5, label='park vehicle', color ='salmon')

        # 绘制车辆初试位置及边框
        vehicle_vert_polygon = Polygon(
            [(vehicle_vert[0][0][0], vehicle_vert[0][0][1]),
            (vehicle_vert[0][1][0], vehicle_vert[0][1][1]),
            (vehicle_vert[0][2][0], vehicle_vert[0][2][1]),
            (vehicle_vert[0][3][0], vehicle_vert[0][3][1])]
        )
        vehicle_x, vehicle_y = vehicle_vert_polygon.exterior.xy
        plt.fill(vehicle_x, vehicle_y, label='parking lot', color='lightgreen')
        # 绘制车辆中心位置
        plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # 绘制车辆航向
        plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # 绘制伪激光雷达的线
        id = 0
        for start_point in vehicle_ray_start_points:

            # 计算从车辆中心到起点的角度
            angle = np.arctan2(start_point.y - vehicle_center.y, start_point.x - vehicle_center.x)

            # 计算射线终点
            end_x = start_point.x + obs_min_dis[id]*np.cos(angle)
            end_y = start_point.y + obs_min_dis[id]*np.sin(angle)

            # 绘制射线
            plt.plot([start_point.x, end_x], [start_point.y, end_y], color='blue', alpha=0.5)
            id += 1

        plt.grid()
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')
        quit()

    def render_park_trajectory(
            self,
            parking_spaces,
            lane_marks,
            vehicle_pos, 
            vehicle_vert, 
            target_parking_id,
            target_pos,
            vehicle_area,
            park_vertex,
            park_vehicle_polygons,
            vehicle_ray_start_points,
            obs_min_dis,
    ):
        '''
        Render the parking scenarios and parking trajectory.
        '''
        save_dir = './parallel/eval_tra/trajectory_20241125_065228'

        plt.figure(figsize=(10,12.5))

        # # 车辆中心坐标
        # vehicle_center = Point(vehicle_pos[0], vehicle_pos[1])

        # 绘制停车场边界
        park_vertex_polygons = Polygon(park_vertex)
        park_x, park_y = park_vertex_polygons.exterior.xy
        plt.fill(park_x, park_y, label='parking lot', color='lightgrey')
        
        # 绘制停车区域
        parking_spaces_polygons = []
        for i in range(len(parking_spaces)):
            parking_polygon = Polygon(
                [(parking_spaces[i][0][0], parking_spaces[i][0][1]),
                (parking_spaces[i][1][0], parking_spaces[i][1][1]),
                (parking_spaces[i][2][0], parking_spaces[i][2][1]),
                (parking_spaces[i][3][0], parking_spaces[i][3][1])]
            )
            parking_spaces_polygons.append(parking_polygon)

        for parking_spaces_polygon in parking_spaces_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                parking_area_x, parking_area_y = parking_spaces_polygon.exterior.xy
                plt.fill(parking_area_x, parking_area_y, alpha = 0.5, label='parking area', color ='lightblue')

        # 绘制停车库位线
        for i in range(len(parking_spaces)):
            for j in range(len(parking_spaces[i])):
                if j == len(parking_spaces[i])-1:
                    x = [parking_spaces[i][j][0],parking_spaces[i][0][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][0][1]]
                else:
                    x = [parking_spaces[i][j][0],parking_spaces[i][j+1][0]]
                    y = [parking_spaces[i][j][1],parking_spaces[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='black', linewidth=1.0)


        # 绘制车道线
        x = [lane_marks[0][0][0],lane_marks[0][1][0]]
        y = [lane_marks[0][0][1],lane_marks[0][1][1]]
        plt.plot(x, y, label='linear', linestyle='--', color='black', linewidth=2.0)

        # 绘制已停车的车辆
        for park_vehicle_polygon in park_vehicle_polygons:
            if isinstance(parking_spaces_polygon, Polygon):
                park_vehicle_x, park_vehicle_y = park_vehicle_polygon.exterior.xy
                plt.fill(park_vehicle_x, park_vehicle_y, alpha = 0.5, label='park vehicle', color ='salmon')

        # # 绘制车辆初试位置及边框
        # vehicle_vert_polygon = Polygon(
        #     [(vehicle_vert[0][0][0], vehicle_vert[0][0][1]),
        #     (vehicle_vert[0][1][0], vehicle_vert[0][1][1]),
        #     (vehicle_vert[0][2][0], vehicle_vert[0][2][1]),
        #     (vehicle_vert[0][3][0], vehicle_vert[0][3][1])]
        # )
        # vehicle_x, vehicle_y = vehicle_vert_polygon.exterior.xy
        # plt.fill(vehicle_x, vehicle_y, label='parking lot', color='lightgreen')
        # # 绘制车辆中心位置
        # plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # # 绘制车辆航向
        # plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # 绘制泊车轨迹
        data_dir = './Parallel_safe-v0_eval/obs_act_rew_20241125_065228.csv' 
        data = pd.read_csv(data_dir)    # 读取CSV文件        
        traj_x = data['pos_x'].tolist()
        traj_y = data['pos_y'].tolist()
        traj_dir = data['pos_dir'].tolist()
        action_type = data['action_type'].tolist()
        action_args = data['action_args'].tolist()

        start_points_traj = []
        actions = []
        for i in range(len(traj_x)-1):
            start_points_traj.append((traj_x[i], traj_y[i], traj_dir[i]))
            actions.append((action_type[i], action_args[i]))

        all_points = get_path_with_actions(start_points_traj, actions)
        plt.scatter(all_points[:, 0], all_points[:, 1], s= 5)


        path_vehicle_verts = []
        for i in range(len(all_points)):
            point_vehicle_vert = position_vehicle_vertex(
                width_vehicle = self.vehicle_config.vehicle_width,
                length_vehicle = self.vehicle_config.vehicle_length,
                vehicle_pos = all_points[i]
            )

            path_vehicle_verts.append(point_vehicle_vert[0])

        # print('path_vehicle_verts:', len(path_vehicle_verts))
        print('path_vehicle_verts:', len(path_vehicle_verts[0]))

        for i in range(len(path_vehicle_verts)):
            for j in range(len(path_vehicle_verts[i])):
                if j == len(path_vehicle_verts[i])-1:
                    x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][0][0]]
                    y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][0][1]]
                else:
                    x = [path_vehicle_verts[i][j][0],path_vehicle_verts[i][j+1][0]]
                    y = [path_vehicle_verts[i][j][1],path_vehicle_verts[i][j+1][1]]
                plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=0.5)


        # 绘制初始化车辆
        vehicle_pos = [traj_x[0], traj_y[0], traj_dir[0]]
        vehicle_vert = position_vehicle_vertex(width_vehicle = self.vehicle_config.vehicle_width, 
                                                 length_vehicle = self.vehicle_config.vehicle_length, 
                                                 vehicle_pos = vehicle_pos )
        for i in range(len(vehicle_vert[0])):
            # 最后一条边
            if i == len(vehicle_vert[0])-1:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][0][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [vehicle_vert[0][i][0], vehicle_vert[0][i+1][0]]
                y = [vehicle_vert[0][i][1], vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
            # 绘制车辆中心位置
        plt.plot(vehicle_pos[0], vehicle_pos[1], marker='o', c='k',markersize=10)
        # 绘制车辆航向
        plot_arrow(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2], length=2.0, color='green')

        # # 绘制最终车辆位置
        final_vehicle_pos = [traj_x[-1], traj_y[-1], traj_dir[-1]]
        final_vehicle_vert = position_vehicle_vertex(width_vehicle = self.vehicle_config.vehicle_width, 
                                                     length_vehicle = self.vehicle_config.vehicle_length, 
                                                     vehicle_pos = final_vehicle_pos)
        for i in range(len(final_vehicle_vert[0])):
            # 最后一条边
            if i == len(final_vehicle_vert[0])-1:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][0][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][0][1]]
            # 最后一条边
            else:
                x = [final_vehicle_vert[0][i][0], final_vehicle_vert[0][i+1][0]]
                y = [final_vehicle_vert[0][i][1], final_vehicle_vert[0][i+1][1]]
            plt.plot(x, y, label='linear', linestyle='-', color='green', linewidth=3.0)
        # 绘制车辆中心位置
        plt.plot(final_vehicle_pos[0], final_vehicle_pos[1], marker='o', c='k',markersize=10)  
        # 绘制车辆航向
        plot_arrow(final_vehicle_pos[0], final_vehicle_pos[1], final_vehicle_pos[2], length=2.0, color='green')  

        plt.grid()
        plt.axis('off')
        plt.savefig(save_dir)
        plt.savefig(save_dir+'.pdf')
        quit()


class DiagonalParkEnv(gym.Env):
    '''
        Create the RL environment for diagonal parking environment
    '''