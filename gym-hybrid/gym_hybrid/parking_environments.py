from collections import namedtuple
from typing import Optional
from typing import Tuple

import sys

import gym
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

# from ding.utils import get_rank

import warnings
warnings.filterwarnings("ignore")

# gym.logger.set_level(40)  # noqa

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

def plot_arrow(x, y, heading, length=1.0, color='r'):
    """
    根据起点坐标 (x, y) 和航向绘制箭头

    参数:
    - x, y: 起点坐标
    - heading: 航向，单位为弧度
    - length: 箭头的长度 (默认是 1.0)
    - color: 箭头的颜色 (默认是红色)
    """
    # 计算箭头的 dx 和 dy
    dx = length * np.cos(heading)
    dy = length * np.sin(heading)
    
    # 绘制箭头
    plt.arrow(x, y, dx, dy, head_width=0.15, head_length=0.3, linewidth=3.0, fc=color, ec=color)


def calculate_overlap_area(quad1, quad2):
    """
    计算两个四边形的重叠区域面积。
    
    参数:
    quad1 (list): 第一个四边形的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    quad2 (list): 第二个四边形的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    返回:
    float: 重叠区域的面积
    """
    # 创建两个 Polygon 对象
    polygon1 = Polygon(quad1)
    polygon2 = Polygon(quad2)
    
    # 计算交集面积
    intersection_area = polygon1.intersection(polygon2).area
    
    return intersection_area

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """
    计算点 (px, py) 到线段 ((x1, y1), (x2, y2)) 的最短距离
    """
    # 线段的向量
    dx = x2 - x1
    dy = y2 - y1

    # 线段退化为一个点的情况
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # 计算投影点 t，t 表示点在线段上的投影位置（0 <= t <= 1 表示投影在线段上）
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    if t < 0:
        # 投影在线段外，在起点之前
        nearest_x, nearest_y = x1, y1
    elif t > 1:
        # 投影在线段外，在终点之后
        nearest_x, nearest_y = x2, y2
    else:
        # 投影在线段上
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

    # 计算点到最近点的距离
    return math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

def point_to_polygon_distance(px, py, polygon):
    """
    计算点 (px, py) 到多边形的最短距离
    polygon 是多边形的顶点列表，例如 [(x1, y1), (x2, y2), ...]
    """
    min_distance = float('inf')

    # 遍历多边形的每一条边
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]  # 循环遍历，连接最后一条边
        distance = distance_point_to_segment(px, py, x1, y1, x2, y2)
        min_distance = min(min_distance, distance)

    return min_distance

def ceter_to_rear(ceter_point, wheelbase):
    '''
    Transform the vehicle center position to the rear axle center position.
    Input:
        center_point: the ceter point of the ego vehicle.
        length_vehicle: the length of the ego vehicle.
    Return:
        rear_point: the rear axle ceter point of the ego vehicle.    
    '''
    rear_point_x = ceter_point[0] - wheelbase/2*math.cos(ceter_point[2])
    rear_point_y = ceter_point[1] - wheelbase/2*math.sin(ceter_point[2])
    rear_point_dir = ceter_point[2]

    rear_point = [rear_point_x, rear_point_y, rear_point_dir]
    return rear_point

def rear_to_ceter(rear_point, wheelbase):
    '''
    Transform the rear axle center position to the center position of the ego vehicle.
    Input:
        rear_point: the real axle ceter point of the ego vehicle.
        length_vehicle: the length of the ego vehicle.
    Return: 
        ceter_point: the ceter position of the ego vehicle.
    '''
    ceter_point_x = rear_point[0] + wheelbase/2*math.cos(rear_point[2])
    ceter_point_y = rear_point[1] + wheelbase/2*math.sin(rear_point[2])
    ceter_point_dir = rear_point[2]
    
    ceter_point = [ceter_point_x, ceter_point_y, ceter_point_dir]
    return ceter_point
 
def is_point_in_rectangle(p, rect):
    '''
    Compute whether a point is within a rectangular area.
    Inupt: 
        p: the position of the point.
        rect: the position of the rectangular. rect = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    Return:
        flag: the flag of within the rectangular area.
    '''
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    p1, p2, p3, p4 = rect

    within_flag = (cross(p1, p2, p) >= 0 and 
                  cross(p2, p3, p) >= 0 and 
                  cross(p3, p4, p) >= 0 and 
                  cross(p4, p1, p) >= 0)
    
    # return (cross(p1, p2, p) >= 0 and cross(p2, p3, p) >= 0 and 
    #         cross(p3, p4, p) >= 0 and cross(p4, p1, p) >= 0)
    return within_flag

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
    # 车辆的四个顶点
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

def env_stop_condition(parking_spaces, vehicle_vertex, ego_target_dis, ego_target_dir, bias_x, bias_y, park_vertex):
    '''
    Compute the stop condition
    Input:
        parking_spaces: the position of the parking spaces; parking_spaces = [[parking_position1],
                                                                               [parking_position2],
                                                                               ...].
        vehicle_vertex: the vertex position of the ego vehicle; vehicle_vertex = [[leftlow, rightlow, 
                                                                                   righttop, lefttop]]
        ego_target_dis: the distance from ego vehicle to target parking.
        ego_target_dir: the dirtory error of the ego vehicle and target parking.
    Return:
        Done:         
    '''
    reward_target = 0
    reward_crash = 0
    terminated = False
    truncated = False
    info = {}
    lf = (4.3-2.7)/2
    lr = (4.3-2.7)/2

    '''
    成功停车奖励和终止标志位
    '''
    # Successful parking termination trajectory
    # if ego_target_dis < 0.3 and abs(ego_target_dir)<15/180*math.pi:
    if abs(bias_x) < 0.15 and abs(bias_y)<0.35 and abs(ego_target_dir)<10/180*math.pi:
        reward_target = 10
        terminated = True

    '''
    车辆超出停车场的停车区的奖励和终止标志位
    '''
    # Get the parking spaces area
    # parking_points = []
    # for i in range(len(parking_spaces)):
    #     for j in range(len(parking_spaces[0])):
    #         parking_points.append(tuple(parking_spaces[i][j]))
    # parking_polygon = Polygon(parking_points)
    # min_parking_x, min_parking_y, max_parking_x, max_parking_y = parking_polygon.bounds    
    # parking_vertex_areas = [(min_parking_x,min_parking_y),
    #                         (max_parking_x,min_parking_y),
    #                         (max_parking_x,max_parking_y),
    #                         (min_parking_x,max_parking_y)]
    
    # print('parking_vertex_areas:', parking_vertex_areas)
    # print('parking_vertex_areas_type:', type(parking_vertex_areas))
    # print('park_vertex:', park_vertex)
    # print('park_vertex_type:', type(park_vertex))
    # quit()

    parking_vertex_areas = park_vertex
    
    # Compute the vertex of ego vehicle wether within the parking spaces area.
    # 车辆顶点超过停车区域则终止轨迹，并设置负奖励
    for i in range(len(vehicle_vertex[0])):
        vehicle_vertex_tuple = tuple(vehicle_vertex[0][i])
        if not is_point_in_rectangle(vehicle_vertex_tuple, parking_vertex_areas):
            # print('vehicle_vertex_tuple:', vehicle_vertex_tuple)
            # print('parking_vertex_areas:', parking_vertex_areas)
            reward_crash = -10
            terminated = True
            break

    info = {'terminated': terminated,
            'reward_target': reward_target,
            'reward_crash': reward_crash}

    return info

def compute_env_reward(vehicle_pos, 
                       target_pos, 
                       parking_spaces, 
                       vehicle_vertex, 
                       park_vertex,
                       target_area,
                       ego_vehicle_area,
                       init_ego_target_dis,
                       current_step,
                       max_step):
    '''
    Compute the reward based on the state
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir].
        target_pos: the position of the target; target_pos = [target_x, target_y, target_dir].
    Return:
        Reward: the reward of current state.
    '''
    # alpha_dis = 0.20
    # alpha_dis = 0.10
    # alpha_dir = 0.20

    # alpha_dis = 0.0
    # alpha_dir = 0.5
    # alpha_target = 20
    # alpha_crash = 2

    # alpha_dis = 1
    # alpha_dir = 1
    # alpha_target = 4
    # alpha_crash = 1

    # alpha_dis = 1
    # alpha_dir = 1
    # alpha_target = 4
    # alpha_crash = 1
    # alpha_overlap = 0.5

    # alpha_dis = 0.5
    # alpha_dir = 0.5
    # alpha_target = 4
    # alpha_crash = 1
    # alpha_overlap = 1 
    # alpha_time = 0.1

    alpha_dis = 1
    alpha_dir = 1
    alpha_target = 4
    alpha_crash = 0.5
    alpha_overlap = 0
    alpha_time = 0

    '''
    计算路由惩罚
    '''
    time_cost = math.tanh(current_step/(10*max_step))
    reward_time = -time_cost

    '''
    计算重叠区域的奖励
    '''
    # 转换车辆区域的坐标信息
    vehicle_area = []
    for i in range(len(vehicle_vertex[0])):
        vehicle_area.append(tuple(vehicle_vertex[0][i]))
    
    overlap_area = calculate_overlap_area(vehicle_area, target_area)
    reward_overlap = overlap_area / ego_vehicle_area

    '''
    计算主车中心与目标库位中心位置的欧式距离
    '''
    # Compute the maxmize distance of the ego vehicle and target parking
    max_distance = math.sqrt(math.pow(10,2)+math.pow(17.6,2))
    min_distance = 1.0
    # Compute the distance of the ego vehicle and target parking
    bias_x = target_pos[0] - vehicle_pos[0]
    bias_y = target_pos[1] - vehicle_pos[1] 
    ego_target_dis = math.sqrt(math.pow(target_pos[0] - vehicle_pos[0], 2) +
                               math.pow(target_pos[1] - vehicle_pos[1], 2))
    # Normalized the distance of the ego vehicle and target parking
    ego_target_dis_normalize = ego_target_dis/max_distance
    # ego_target_dis_normalize = (ego_target_dis - init_ego_target_dis)/max(init_ego_target_dis, min_distance)
    
    '''
    计算主车航向与库位航向的偏差
    '''
    # # Compute the direction of the ego vehicle and target parking 
    # ego_target_dir = abs(vehicle_pos[2] - target_pos[2]) % math.pi
    # # Normalized the direction of the ego vehicle and target parking
    # ego_target_dir_normalize = ego_target_dir / math.pi

    # Compute the direction of the ego vehicle and target parking 
    # # ego_target_dir = (vehicle_pos[2] - target_pos[2]) % (2*math.pi)
    ego_target_dir = vehicle_pos[2] - target_pos[2]
    # # Normalized the direction of the ego vehicle and target parking
    # ego_target_dir_normalize = abs(ego_target_dir) / (2*math.pi)

    # math.pi
    math_pi = 3.1415926  
    if ego_target_dir > 0:
        ego_target_dir = ego_target_dir % math_pi
    else:
        ego_target_dir = ego_target_dir % math_pi - math_pi

    ego_target_dir_normalize = abs(ego_target_dir)/math_pi
  

    '''
    计算距离和航向偏差的奖励
    '''
    # # The reward of the distance bias
    # reward_distance = 1 - ego_target_dis_normalize
    # # The reward of the directory bias
    # reward_direction = 1 - ego_target_dir_normalize

    # The reward of the distance bias
    reward_distance =  - ego_target_dis_normalize
    # The reward of the directory bias
    reward_direction =  - ego_target_dir_normalize

    '''
    计算成功停车或异常终止条件及奖励
    '''    
    info = env_stop_condition(
        parking_spaces = parking_spaces, 
        vehicle_vertex = vehicle_vertex, 
        ego_target_dis = ego_target_dis, 
        ego_target_dir = ego_target_dir,
        bias_x = bias_x,
        bias_y = bias_y,
        park_vertex = park_vertex
    )

    info['reward_distance'] = reward_distance
    info['reward_direction'] = reward_direction
    info['reward_overlap'] = reward_overlap
    info['reward_time'] = reward_time

    reward = alpha_dis * info['reward_distance'] + \
             alpha_dir*info['reward_direction'] + \
             alpha_crash*info['reward_crash'] + \
             alpha_target*info['reward_target'] + \
             alpha_overlap*info['reward_overlap']+\
             alpha_time*info['reward_time']
    
    info['reward'] = reward
    # if info['reward']> 100:
    #     print('reward:', info['reward'])
    #     # quit()

    return info

def compute_vehicle_pos(vehicle_pos, action):
    '''
    compute the position of the ego vehicle via the current position and action
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir].
        dir_flg: the flag of the ego vehicle forward or backing
        action: the action of the ego vehicle; action = [velocity_rear, wheel_dir]
    Return:
        next_vehicle_pos: the position of the ego vehicle after excute the action.
    '''
    # Compute the position and dirtory of the vehicle.       
    time_interval = 0.1      # The time interval between each step, and the unit is second.
    wheelbase_length = 2.7   # The vehicle's wheelbase, and the unit is meter.

    # Compute the delta of the position and dirtory
    delta_pos_x = action[0]*time_interval*math.cos(vehicle_pos[2])
    delta_pos_y = action[0]*time_interval*math.sin(vehicle_pos[2])
    delta_pos_dir = action[0]*time_interval*math.tan(action[1])/wheelbase_length
    
    # Compute the new position of the ego vehicle
    pos_x = vehicle_pos[0] + delta_pos_x
    pos_y = vehicle_pos[1] + delta_pos_y
    pos_dir = vehicle_pos[2] + delta_pos_dir

    # Get the new postion of the ego vehicle
    next_vehicle_pos = [pos_x, pos_y, pos_dir]

    return next_vehicle_pos

def compute_vehicle_pos_hybrid(vehicle_pos, action):
    '''
    compute the position of the ego vehicle via the current position and action
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir]
        action: the class for hybrid action
    Return: 
        next_vehicle_pos: the position of the ego vehicle after excute the action.
    '''
    # The current position of the ego vehicle
    pos_x = vehicle_pos[0]
    pos_y = vehicle_pos[1]
    pos_dir = vehicle_pos[2]

    # next_vehicle_pos = []
    # The alpha of the path planning
    if (isinstance(action.id, list) != True) and (isinstance(action.id, np.ndarray) != True):
        action.id = [action.id]

    alpha = action.parameters[0]/RADIUS

    if action.id[0] == ACTION.rightfront:         # 右前
        next_pos_x = pos_x + 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir - alpha/2)
        next_pos_y = pos_y + 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir - alpha/2)
        next_pos_dir = pos_dir - alpha
    elif action.id[0] == ACTION.straightfront:    # 正前
        next_pos_x = pos_x + action.parameters[0]*math.cos(pos_dir)
        next_pos_y = pos_y + action.parameters[0]*math.sin(pos_dir)
        next_pos_dir = pos_dir
    elif action.id[0] == ACTION.leftfront:        # 左前
        next_pos_x = pos_x + 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir + alpha/2)
        next_pos_y = pos_y + 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir + alpha/2)
        next_pos_dir = pos_dir + alpha
    elif action.id[0] == ACTION.leftrear:         # 左后
        next_pos_x = pos_x - 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir - alpha/2)
        next_pos_y = pos_y - 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir - alpha/2)
        next_pos_dir = pos_dir - alpha
    elif action.id[0] == ACTION.straightrear:     # 正后
        next_pos_x = pos_x - action.parameters[0]*math.cos(pos_dir)
        next_pos_y = pos_y - action.parameters[0]*math.sin(pos_dir)
        next_pos_dir = pos_dir
    elif action.id[0] == ACTION.rightrear:        # 右后
        next_pos_x = pos_x - 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir + alpha/2)
        next_pos_y = pos_y - 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir + alpha/2)
        next_pos_dir = pos_dir + alpha

    # Get the new position of the ego vehicle
    next_vehicle_pos = [next_pos_x, next_pos_y, next_pos_dir]
    # print('next_vehicle_pos:', next_vehicle_pos)

    return next_vehicle_pos

def compute_vehicle_pos_hybrid_com(vehicle_pos, action):
    '''
    compute the position of the ego vehicle via the current position and action
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir]
        action: the class for hybrid action
    Return: 
        next_vehicle_pos: the position of the ego vehicle after excute the action.
    '''
    # The current position of the ego vehicle
    pos_x = vehicle_pos[0]
    pos_y = vehicle_pos[1]
    pos_dir = vehicle_pos[2]

    # next_vehicle_pos = []
    # The alpha of the path planning
    if (isinstance(action.id, list) != True) and (isinstance(action.id, np.ndarray) != True):
        action.id = [action.id]

    alpha = action.parameters[0]/RADIUS

    if action.id[0] == ACTION_COM.right:
        next_pos_x = pos_x + 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir - alpha/2)
        next_pos_y = pos_y + 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir - alpha/2)
        next_pos_dir = pos_dir - alpha
    elif action.id[0] == ACTION_COM.straight:
        next_pos_x = pos_x + action.parameters[0]*math.cos(pos_dir)
        next_pos_y = pos_y + action.parameters[0]*math.sin(pos_dir)
        next_pos_dir = pos_dir
    elif action.id[0] == ACTION_COM.left:
        next_pos_x = pos_x + 2*RADIUS*math.sin(alpha/2)*math.cos(pos_dir + alpha/2)
        next_pos_y = pos_y + 2*RADIUS*math.sin(alpha/2)*math.sin(pos_dir + alpha/2)
        next_pos_dir = pos_dir + alpha    

    # Get the new position of the ego vehicle
    next_vehicle_pos = [next_pos_x, next_pos_y, next_pos_dir]
    # print('next_vehicle_pos:', next_vehicle_pos)

    return next_vehicle_pos

def Perpendicular_parking(width_park,
                          length_park,
                          width_lane,
                          parking_num):
    '''
    Set the prependicular parking scenarios
    Input:
        width_park: the width of the parking spaces
        length_park: the length of the parking spaces
        width_lane: the width of the lane
        parking_num: the numbers of the parking spaces
    Output:
        parking_spaces: the position of the parking spaces;   
                        [left_low, right_low, right_top, left_top] -> parking_spaces[i]
        lane_marks: the lines of the parking scenarios
        vehicle_area: the area of the init vehicle.       
    '''

    parking_spaces = []
    # 下方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park, 0]
        parking_rightlow = [(i+1)*width_park, 0]
        parking_righttop = [(i+1)*width_park, length_park]
        parking_lefttop = [i*width_park, length_park]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop]) 

    # 上方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park, length_park+2*width_lane]
        parking_rightlow = [(i+1)*width_park, length_park+2*width_lane]
        parking_righttop = [(i+1)*width_park, 2*length_park+2*width_lane]
        parking_lefttop = [i*width_park, 2*length_park+2*width_lane]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop])

    # 双向车道线
    lane_marks = []
    leftpoints = [0, length_park + width_lane]
    rightpoints = [parking_num*width_park, length_park + width_lane]
    lane_marks.append([leftpoints, rightpoints])

    # # 车辆初始区域
    # vehicle_area = []
    # offset_val = 0.5
    # area_leftlow = [0, length_park + width_lane/2 - offset_val]
    # area_rightlow = [(parking_num-1)*width_park, length_park + width_lane/2 - offset_val]
    # area_lefttop = [(parking_num-1)*width_park, length_park + width_lane/2 + offset_val]
    # area_righttop = [0, length_park + width_lane/2 + offset_val]
    # vehicle_area.append([area_leftlow, area_rightlow, area_lefttop, area_righttop])

    '''
    车辆初试区域
    '''
    # 车辆初始区域
    vehicle_area = []
    offset_val = 0.2
    area_leftlow = [width_lane/2 + 0.5, length_park + width_lane/2 - offset_val]
    area_rightlow = [(parking_num-1)*width_park - 0.5, length_park + width_lane/2 - offset_val]
    area_lefttop = [(parking_num-1)*width_park - 0.5, length_park + width_lane/2 + offset_val]
    area_righttop = [width_lane/2 + 0.5, length_park + width_lane/2 + offset_val]
    vehicle_area.append([area_leftlow, area_rightlow, area_lefttop, area_righttop])

    return parking_spaces, lane_marks, vehicle_area

def Perpendicular_parking_hybrid(width_park,
                                 length_park,
                                 width_lane,
                                 parking_num):
    '''
    Set the prependicular parking scenarios
    Input:
        width_park: the width of the parking spaces
        length_park: the length of the parking spaces
        width_lane: the width of the lane
        parking_num: the numbers of the parking spaces
    Output:
        parking_spaces: the position of the parking spaces;   
                        [left_low, right_low, right_top, left_top] -> parking_spaces[i]
        lane_marks: the lines of the parking scenarios
        vehicle_area: the area of the init vehicle.       
    '''

    parking_spaces = []
    # 下方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park, 0]
        parking_rightlow = [(i+1)*width_park, 0]
        parking_righttop = [(i+1)*width_park, length_park]
        parking_lefttop = [i*width_park, length_park]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop]) 

    # 上方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park, length_park+2*width_lane]
        parking_rightlow = [(i+1)*width_park, length_park+2*width_lane]
        parking_righttop = [(i+1)*width_park, 2*length_park+2*width_lane]
        parking_lefttop = [i*width_park, 2*length_park+2*width_lane]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop])

    # 双向车道线
    lane_marks = []
    leftpoints = [0, length_park + width_lane]
    rightpoints = [parking_num*width_park, length_park + width_lane]
    lane_marks.append([leftpoints, rightpoints])

    # 车库顶点
    park_leftlow = (0, 0)
    park_rightlow = (parking_num*width_park, 0)
    park_righttop = (parking_num*width_park, 2*(length_park + width_lane))
    park_lefttop = (0, 2*(length_park + width_lane))
    park_vertex = [park_leftlow, park_rightlow, park_righttop, park_lefttop]


    return parking_spaces, lane_marks, park_vertex

def Perpendicular_parking_hybrid_boundary(width_park,
                                          length_park,
                                          width_lane,
                                          parking_num,
                                          boundary_width=0.5):
    '''
    Set the prependicular parking scenarios
    Input:
        width_park: the width of the parking spaces
        length_park: the length of the parking spaces
        width_lane: the width of the lane
        parking_num: the numbers of the parking spaces
    Output:
        parking_spaces: the position of the parking spaces;   
                        [left_low, right_low, right_top, left_top] -> parking_spaces[i]
        lane_marks: the lines of the parking scenarios
        vehicle_area: the area of the init vehicle.       
    '''

    parking_spaces = []
    # 下方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park+boundary_width, 0+boundary_width]
        parking_rightlow = [(i+1)*width_park+boundary_width, 0+boundary_width]
        parking_righttop = [(i+1)*width_park+boundary_width, length_park+boundary_width]
        parking_lefttop = [i*width_park+boundary_width, length_park+boundary_width]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop]) 

    # 上方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*width_park+boundary_width, length_park+2*width_lane+boundary_width]
        parking_rightlow = [(i+1)*width_park+boundary_width, length_park+2*width_lane+boundary_width]
        parking_righttop = [(i+1)*width_park+boundary_width, 2*length_park+2*width_lane+boundary_width]
        parking_lefttop = [i*width_park+boundary_width, 2*length_park+2*width_lane+boundary_width]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop])

    # 双向车道线
    lane_marks = []
    leftpoints = [0, length_park + width_lane + boundary_width]
    rightpoints = [parking_num*width_park+2*boundary_width, length_park + width_lane + boundary_width]
    lane_marks.append([leftpoints, rightpoints])

    # 车库顶点
    park_leftlow = (0, 0)
    park_rightlow = (parking_num*width_park+2*boundary_width, 0)
    park_righttop = (parking_num*width_park+2*boundary_width, 2*(length_park + width_lane)+2*boundary_width)
    park_lefttop = (0, 2*(length_park + width_lane)+2*boundary_width)
    park_vertex = [park_leftlow, park_rightlow, park_righttop, park_lefttop]

    return parking_spaces, lane_marks, park_vertex

def generate_init_vehicle(vehicle_area, width_vehicle, length_vehicle):
    '''
    Generate the init position and dir of the vehicle
    input:
        vehicle_area: the area of the vehicle [leftlow, rightlow, righttop, lefttop]
    output:
        vehicle_pos: the init position and dir of the random vehicle
        vehicle_vert: the vertex points of the random vehicle
    '''

    # 将车辆区域坐标list转化为tuple
    vertex_points = []
    for i in range(len(vehicle_area[0])):
        vertex_points.append(tuple(vehicle_area[0][i]))

    # 在车辆区域生成坐标
    polygon = Polygon(vertex_points)
    min_x, min_y, max_x, max_y = polygon.bounds
    random_point = Point(np.random.uniform(min_x, max_x),
                         np.random.uniform(min_y, max_y))
    vehicle_pos = [random_point.x, random_point.y]
    # 给车辆坐标增加航向
    vehicle_dir = 0
    vehicle_pos.append(vehicle_dir)
    
    # 车辆的四个顶点
    vehicle_vert = []
    vehicle_leftlow = [random_point.x - length_vehicle/2, random_point.y - width_vehicle/2]
    vehicle_rightlow = [random_point.x + length_vehicle/2, random_point.y - width_vehicle/2]
    vehicle_righttop = [random_point.x + length_vehicle/2, random_point.y + width_vehicle/2]
    vehicle_lefttop = [random_point.x - length_vehicle/2, random_point.y + width_vehicle/2]
    vehicle_vert.append([vehicle_leftlow, vehicle_rightlow, vehicle_righttop, vehicle_lefttop])

    return vehicle_pos, vehicle_vert

def get_init_vehicle_pos(vehicle_area, vehicle_config, area_id):
    '''
    Generate the init position and dir of the vehicle
    input:
        vehicle_area: the area of the vehicle [leftlow, rightlow, righttop, lefttop]
    output:
        vehicle_pos: the init position and dir of the random vehicle
        vehicle_vert: the vertex points of the random vehicle
    '''
    # 初始化车辆参数
    vehicle_length = vehicle_config.vehicle_length
    vehilce_width = vehicle_config.vehicle_width

    # 将车辆区域坐标list转化为tuple
    vertex_points = []
    for i in range(len(vehicle_area[0])):
        vertex_points.append(tuple(vehicle_area[0][i]))

    # 在车辆区域生成坐标
    polygon = Polygon(vertex_points)
    min_x, min_y, max_x, max_y = polygon.bounds
    random_point = Point(np.random.uniform(min_x, max_x),
                         np.random.uniform(min_y, max_y))
    vehicle_pos = [random_point.x, random_point.y]

    # 获取车辆航向
    if area_id == 1:
        vehicle_dir = 90
    elif area_id == 2:
        vehicle_dir = 45
    elif area_id == 3:
        vehicle_dir = 0
    # 补充航向随机值
    dir_offset = np.random.uniform(-10,10)
    vehicle_dir = vehicle_dir + dir_offset
    # 将角度转化为弧度
    vehicle_dir = vehicle_dir/180*math.pi
    vehicle_pos.append(vehicle_dir)
    

    # # 将车辆区域坐标list转化为tuple
    # vertex_points = []
    # for i in range(len(vehicle_area[0])):
    #     vertex_points.append(tuple(vehicle_area[0][i]))

    # # 在车辆区域生成坐标
    # polygon = Polygon(vertex_points)
    # min_x, min_y, max_x, max_y = polygon.bounds
    # random_point = Point(random.uniform(min_x, max_x),
    #                      random.uniform(min_y, max_y))
    # vehicle_pos = [random_point.x, random_point.y]
    # # 给车辆坐标增加航向
    # vehicle_dir = 0
    # vehicle_pos.append(vehicle_dir)

    return vehicle_pos

def generate_target_parking(vehicle_pos, length_vehicle, parking_num, width_park, parking_spaces):
    '''
    Generate the target parking spaces during reset the environment.
    input:
        vehicle_pos: the init position and dir of the random vehicle.
        length_vehicle: the length of the vehicle.
        parking_num: the number of the parking spaces in the scenarios.
        width_park: the width of the parking spaces.
        parking_spaces: the multiple parking spaces of the parking spaces scenarios.
    return:
        target_parking_id: the parking spaces id of the target parking.
        target_pos: the target position of the target parking spaces.
    '''    
    # 获取目标车库位置的ID
    target_parking_id = 0

    if vehicle_pos[0] + length_vehicle/2 > (parking_num-1)*width_park:
        target_parking_id = parking_num - 1
    elif vehicle_pos[0] + length_vehicle/2 > (parking_num-2)*width_park:
        target_parking_id = parking_num - 2
    elif vehicle_pos[0] + length_vehicle/2 > (parking_num-3)*width_park:
        target_parking_id = parking_num - 3
    else:
        target_parking_id = 0

    # 获取目标车库坐标
    '''
    矩形中心坐标等于四个顶点坐标的均值
    '''
    target_x = 0
    target_y = 0
    for i in range(len(parking_spaces[target_parking_id])):
        target_x += parking_spaces[target_parking_id][i][0]
        target_y += parking_spaces[target_parking_id][i][1]

    target_x = target_x/len(parking_spaces[target_parking_id])
    target_y = target_y/len(parking_spaces[target_parking_id])

    # 目标位姿增加航向信息
    target_dir = 90/180*math.pi
    target_pos = [target_x, target_y, target_dir]
   
    return target_parking_id, target_pos

def get_target_parking_pos(target_parking_id, parking_spaces):
    '''
    Generate the target parking spaces during reset the environment.
    input:
        target_parking_id: the parking spaces id of the target parking.
        parking_spaces: the multiple parking spaces of the parking spaces scenarios.
    return:
        target_pos: the target position of the target parking spaces.
    '''    

    # 获取目标车库坐标
    '''
    矩形中心坐标等于四个顶点坐标的均值
    '''
    target_x = 0
    target_y = 0
    target_area = []   # 目标停车区域
    for i in range(len(parking_spaces[target_parking_id])):
        target_x += parking_spaces[target_parking_id][i][0]
        target_y += parking_spaces[target_parking_id][i][1]
        target_area.append(tuple(parking_spaces[target_parking_id][i]))

    target_x = target_x/len(parking_spaces[target_parking_id])
    target_y = target_y/len(parking_spaces[target_parking_id])

    # 目标位姿增加航向信息
    target_dir = 90/180*math.pi
    target_pos = [target_x, target_y, target_dir]
   
    return target_pos, target_area

def get_init_vehicle_area(target_parking_id, area_id, parking_config, vehicle_config, park_boundary):
    '''
    车辆初试区域
    '''
    park_width = parking_config.park_width
    park_length = parking_config.park_length
    lane_width = parking_config.lane_width
    
    # 车辆初始区域
    vehicle_area = []

    # if area_id == 1:
    #     x_0 = (target_parking_id + 0.5)*park_width
    #     y_0 = 0.5*park_length
    #     x_offset = 0.1
    #     y_offset = 0.5*park_length + 1/4*lane_width
    #     area_leftlow = [x_0 - x_offset, y_0]
    #     area_rightlow = [x_0 + x_offset, y_0]
    #     area_righttop = [x_0 + x_offset, y_0 + y_offset]
    #     area_lefttop = [x_0 - x_offset, y_0 + y_offset]
    #     vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    
    if area_id == 1:        # 库位正前方
        x_0 = (target_parking_id + 0.5)*park_width + park_boundary
        y_0 = 0.75*park_length + park_boundary
        x_offset = 0.1
        y_offset = 0.25*park_length + 1/2*lane_width
        area_leftlow = [x_0 - x_offset, y_0]
        area_rightlow = [x_0 + x_offset, y_0]
        area_righttop = [x_0 + x_offset, y_0 + y_offset]
        area_lefttop = [x_0 - x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    elif area_id == 2:      # 库位右前方
        x_0 = (target_parking_id + 0.5)*park_width + park_boundary
        y_0 = park_length + park_boundary
        x_1 = (target_parking_id + 0.5)*park_width + 0.5* lane_width + park_boundary
        y_1 = park_length + 0.5*lane_width + park_boundary
        x_offset = 0.2
        y_offset = 0
        area_leftlow = [x_0 - x_offset, y_0]
        area_rightlow = [x_0 + x_offset, y_0]
        area_righttop = [x_1 + x_offset, y_1]
        area_lefttop = [x_1 - x_offset, y_1]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    elif area_id == 3:      # 库位附近
        x_0 = (target_parking_id + 0.5)* park_width + park_boundary
        y_0 = park_length + 0.5*lane_width + park_boundary
        x_offset = 0.5*park_width
        y_offset = 0.2  
        # area_leftlow = [x_0 - x_offset/2, y_0 - y_offset]
        # area_rightlow = [x_0 + x_offset, y_0 - y_offset]
        # area_righttop = [x_0 + x_offset, y_0 + y_offset]
        # area_lefttop = [x_0 - x_offset/2, y_0 + y_offset]
        area_leftlow = [x_0 - x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 - x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop]) 
    else:
        print('Error: Invalid value. The area_id must be 1, 2, or 3.')
        sys.exit(1) # 中断程序      
    return vehicle_area


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


class PerpendicularParkEnv(gym.Env):
    '''
        Create the RL environment for perpendicular parking environment
    '''
    def __init__(self):
        '''
        Init the environment
        '''
        super(PerpendicularParkEnv, self).__init__()
        self.width_park = 2.5    # the width of the parking spaces
        self.length_park = 5.3   # the legth of the parking spaces
        self.width_lane = 3.5    # the width of the lane
        self.width_vehicle = 2.0     # 车辆宽度
        self.length_vehicle = 4.3    # 车辆长度
        self.parking_num = 4         # 停车位
        self.wheelbase = 2.7         # 轴距
        
        # self.action_dim = 6       # 动作维度
        self.action_dim = 3       # 动作维度
        self.state_dim = 13        # 观测状态维度
        # self.max_steps = 2000     # 环境最大步数
        self.max_steps = 100
        self.current_step = 0     # 初始化当前步数
        self.park_bounary = 0.5   # 停车场边界宽度 

        # 车辆参数信息
        vehicle_config = {
            'vehicle_width': 2.0,
            'vehicle_length': 4.3,
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

        # 初始化停车场库位
        self.parking_spaces, self.lane_marks, self.park_vertex = Perpendicular_parking_hybrid_boundary(
            width_park = self.width_park,
            length_park = self.length_park,
            width_lane = self.width_lane,
            parking_num = 4,
            boundary_width = self.park_bounary
        )

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
        # 随机产生目标库位
        # self.target_parking_id = random.randint(1,2)
        self.target_parking_id = 1
        self.target_pos, self.target_area = get_target_parking_pos(
            target_parking_id = self.target_parking_id,
            parking_spaces = self.parking_spaces
        )

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

        # 根据车辆中心位置获取车辆后轴坐标
        self.vehicle_pos_r = ceter_to_rear(
            ceter_point = self.vehicle_pos, 
            wheelbase = self.wheelbase
        )

        # 计算车辆四个顶点到停车场边界的距离
        vert_mindis = []
        for i in range(len(self.vehicle_vert[0])):
            px = self.vehicle_vert[0][i][0]
            py = self.vehicle_vert[0][i][1]
            # 判定车辆定点是否已经出界
            # Internal_points = is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex)
            if is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex):
                min_distance = point_to_polygon_distance(px, py, self.park_vertex)
            else:
                min_distance = - point_to_polygon_distance(px, py, self.park_vertex)
            vert_mindis.append(min_distance)
            
        # print('vert_mindis:', vert_mindis)
        # 主车与目标库位的位置和航向偏差
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the state after reset the environment
        # self.state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos)
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis).tolist()
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis)
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

        # 计算车辆四个顶点到停车场边界的距离
        vert_mindis = []
        for i in range(len(self.vehicle_vert[0])):
            px = self.vehicle_vert[0][i][0]
            py = self.vehicle_vert[0][i][1]
            # min_distance = point_to_polygon_distance(px, py, self.park_vertex)
            if is_point_in_rectangle(self.vehicle_vert[0][i], self.park_vertex):
                min_distance = point_to_polygon_distance(px, py, self.park_vertex)
            else:
                min_distance = - point_to_polygon_distance(px, py, self.park_vertex)
            vert_mindis.append(min_distance)
        
        # The bias of the position and directory between the ego vehicle and the target parking space
        self.bias_pos = [self.vehicle_pos[i] - self.target_pos[i] for i in range(len(self.vehicle_pos))]

        # Get the new state
        # self.state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos)
        # state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis).tolist()
        state = np.array(self.vehicle_pos + self.target_pos + self.bias_pos + vert_mindis)

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
            max_step = self.max_steps
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
        return state, info['reward'], done, info 

    def close(self):
        '''
        Close the environment
        '''
        return True

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

    
class ParallelParkEnv(gym.Env):
    '''
        Create the RL environment for parallel parking environment
    '''


class DiagonalParkEnv(gym.Env):
    '''
        Create the RL environment for diagonal parking environment
    '''