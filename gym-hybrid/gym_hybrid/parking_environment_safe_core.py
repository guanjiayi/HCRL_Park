from collections import namedtuple
from typing import Optional
from typing import Tuple

import sys

# import gym
import numpy as np
import cv2
import os
# from gym import spaces
# from gym.utils import seeding

from shapely.geometry import Point, Polygon
import math
import matplotlib.pyplot as plt
import pandas as pd
# import random
from easydict import EasyDict
from shapely.geometry import Point, Polygon, LineString


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


# 检查点是否在四边形的边上
def is_point_on_polygon_edge(point, polygon):
    '''
    检查点point是否在四边形polygon的四条边上
    point: 是待检测的点
    polygon: 是检测的四边形
    '''
    for i in range(len(polygon.exterior.coords) - 1):
        edge = LineString([polygon.exterior.coords[i], polygon.exterior.coords[i + 1]])
        if point.distance(edge) < 1e-9:  # 允许微小误差
            return True
    return False

# 检查点是否在圆形的边上
def is_point_on_circle_edge(point, circle):
    '''
    检查点point是否在圆形circle上
    point: 是待检测的点
    circle: 是待检测的圆形
    '''
    return point.distance(circle.exterior) < 1e-9  # 允许微小误差

def vehicle_wall_obstacle(
        parking_spaces, 
        park_vertex, 
        vehicle_config, 
        target_parking_id,
    ):
    '''
    根据库位和停车场边界确定障碍车辆和停车场边界

    参数：
        parking_spaces: 库位顶点列表
        park_vertex: 停车场顶点
    '''
    vehicle_width = vehicle_config.vehicle_width
    vehicle_length = vehicle_config.vehicle_length

    '''
    将每个库位顶点转化为车辆顶点坐标
    '''
    # 计算库位车辆的顶点
    vehicle_vertex_polygons = []
    parking_spaces_polygons = []
    id = 0

    for park_sapce in parking_spaces:
        parking_spaces_polygons.append(
             Polygon([
                (park_sapce[0][0],park_sapce[0][1]),
                (park_sapce[1][0],park_sapce[1][1]),
                (park_sapce[2][0],park_sapce[2][1]),
                (park_sapce[3][0],park_sapce[3][1]),
            ])
        )
        if id != target_parking_id:
            # 计算四边形的中心点        
            park_space_xs = [point[0] for point in park_sapce]
            park_space_ys = [point[1] for point in park_sapce]
            park_sapce_x = sum(park_space_xs)/4
            park_sapce_y = sum(park_space_ys)/4

            # 根据新长宽，计算新的顶点
            half_width = vehicle_width/2
            half_length = vehicle_length/2

            vehicle_polygon = [
                (park_sapce_x - half_width, park_sapce_y - half_length),  # 左下
                (park_sapce_x + half_width, park_sapce_y - half_length),  # 右下
                (park_sapce_x + half_width, park_sapce_y + half_length),  # 右上
                (park_sapce_x - half_width, park_sapce_y + half_length),  # 左上
            ]
            
            vehicle_vertex_polygons.append(Polygon(vehicle_polygon))
        id += 1

    '''
    将库位坐标转化为Polygon的形式    
    '''
    park_vertex_polygon = Polygon([
        (park_vertex[0][0],park_vertex[0][1]),
        (park_vertex[1][0],park_vertex[1][1]),
        (park_vertex[2][0],park_vertex[2][1]),
        (park_vertex[3][0],park_vertex[3][1]),
    ])
    # print('vehicle_vertex_polygons:',vehicle_vertex_polygons)

    return vehicle_vertex_polygons, park_vertex_polygon, parking_spaces_polygons


def vehicle_wall_obstacle_noise(
        parking_spaces, 
        park_vertex, 
        vehicle_config, 
        target_parking_id,
    ):
    '''
    根据库位和停车场边界确定障碍车辆和停车场边界

    参数：
        parking_spaces: 库位顶点列表
        park_vertex: 停车场顶点
    '''
    vehicle_width = vehicle_config.vehicle_width
    vehicle_length = vehicle_config.vehicle_length

    '''
    将每个库位顶点转化为车辆顶点坐标
    '''
    # 计算库位车辆的顶点
    vehicle_vertex_polygons = []
    parking_spaces_polygons = []
    id = 0

    for park_sapce in parking_spaces:
        parking_spaces_polygons.append(
             Polygon([
                (park_sapce[0][0],park_sapce[0][1]),
                (park_sapce[1][0],park_sapce[1][1]),
                (park_sapce[2][0],park_sapce[2][1]),
                (park_sapce[3][0],park_sapce[3][1]),
            ])
        )
        # 车库已停车辆
        if id != target_parking_id:
            # 计算四边形的中心点        
            park_space_xs = [point[0] for point in park_sapce]
            park_space_ys = [point[1] for point in park_sapce]
            park_sapce_x = sum(park_space_xs)/4
            park_sapce_y = sum(park_space_ys)/4


            # 产生位置和航向噪声
            noise_x = np.random.uniform(-0.1, 0.1, size=1)
            noise_y = np.random.uniform(-0.25, 0.25, size=1)
            noise_dir_deg = np.random.uniform(-6, 6, size=1)


            # 噪声扰动之后的航向和中心点
            cx = park_sapce_x + noise_x
            cy = park_sapce_y + noise_y
            c_dir_deg = 90 + noise_dir_deg
            c_dir_rad = np.radians(c_dir_deg)
            cx = cx.item()
            cy = cy.item()
            c_dir_rad = c_dir_rad.item()

            # 根据新长宽，计算新的顶点
            half_width = vehicle_width/2
            half_length = vehicle_length/2

            # 局部坐标系下的顶点
            local_vertices = np.array([
                [-half_length, half_width],  # Top-Left
                [half_length, half_width],   # Top-Right
                [half_length, -half_width],  # Bottom-Right
                [-half_length, -half_width]  # Bottom-Left
            ])

            # 旋转矩阵
            rotation_matrix = np.array([
                [np.cos(c_dir_rad), -np.sin(c_dir_rad)],
                [np.sin(c_dir_rad), np.cos(c_dir_rad)]
            ])

            # 全局坐标系下的顶点
            global_vertices = np.dot(local_vertices, rotation_matrix.T) + np.array([cx, cy])


            vehicle_polygon = [
                (global_vertices[0][0], global_vertices[0][1]),
                (global_vertices[1][0], global_vertices[1][1]),
                (global_vertices[2][0], global_vertices[2][1]),
                (global_vertices[3][0], global_vertices[3][1]),
            ]

            # vehicle_polygon = [
            #     (park_sapce_x - half_width, park_sapce_y - half_length),  # 左下
            #     (park_sapce_x + half_width, park_sapce_y - half_length),  # 右下
            #     (park_sapce_x + half_width, park_sapce_y + half_length),  # 右上
            #     (park_sapce_x - half_width, park_sapce_y + half_length),  # 左上
            # ]
            
            vehicle_vertex_polygons.append(Polygon(vehicle_polygon))
        id += 1

    '''
    将库位坐标转化为Polygon的形式    
    '''
    park_vertex_polygon = Polygon([
        (park_vertex[0][0],park_vertex[0][1]),
        (park_vertex[1][0],park_vertex[1][1]),
        (park_vertex[2][0],park_vertex[2][1]),
        (park_vertex[3][0],park_vertex[3][1]),
    ])
    # print('vehicle_vertex_polygons:',vehicle_vertex_polygons)

    return vehicle_vertex_polygons, park_vertex_polygon, parking_spaces_polygons


def vehicle_wall_obstacle_parallel(
        parking_spaces, 
        park_vertex, 
        vehicle_config, 
        target_parking_id,
    ):
    '''
    根据库位和停车场边界确定障碍车辆和停车场边界

    参数：
        parking_spaces: 库位顶点列表
        park_vertex: 停车场顶点
    '''
    vehicle_width = vehicle_config.vehicle_width
    vehicle_length = vehicle_config.vehicle_length

    '''
    将每个库位顶点转化为车辆顶点坐标
    '''
    # 计算库位车辆的顶点
    vehicle_vertex_polygons = []
    parking_spaces_polygons = []
    id = 0

    for park_sapce in parking_spaces:
        parking_spaces_polygons.append(
             Polygon([
                (park_sapce[0][0],park_sapce[0][1]),
                (park_sapce[1][0],park_sapce[1][1]),
                (park_sapce[2][0],park_sapce[2][1]),
                (park_sapce[3][0],park_sapce[3][1]),
            ])
        )
        if id != target_parking_id:
            # 计算四边形的中心点        
            park_space_xs = [point[0] for point in park_sapce]
            park_space_ys = [point[1] for point in park_sapce]
            park_sapce_x = sum(park_space_xs)/4
            park_sapce_y = sum(park_space_ys)/4

            # 根据新长宽，计算新的顶点
            half_width = vehicle_width/2
            half_length = vehicle_length/2

            vehicle_polygon = [
                (park_sapce_x - half_length, park_sapce_y - half_width),  # 左下
                (park_sapce_x + half_length, park_sapce_y - half_width),  # 右下
                (park_sapce_x + half_length, park_sapce_y + half_width),  # 右上
                (park_sapce_x - half_length, park_sapce_y + half_width),  # 左上
            ]
            
            vehicle_vertex_polygons.append(Polygon(vehicle_polygon))
        id += 1

    '''
    将库位坐标转化为Polygon的形式    
    '''
    park_vertex_polygon = Polygon([
        (park_vertex[0][0],park_vertex[0][1]),
        (park_vertex[1][0],park_vertex[1][1]),
        (park_vertex[2][0],park_vertex[2][1]),
        (park_vertex[3][0],park_vertex[3][1]),
    ])
    # print('vehicle_vertex_polygons:',vehicle_vertex_polygons)

    return vehicle_vertex_polygons, park_vertex_polygon, parking_spaces_polygons


def vehicle_wall_obstacle_parallel_noise(
        parking_spaces, 
        park_vertex, 
        vehicle_config, 
        target_parking_id,
    ):
    '''
    根据库位和停车场边界确定障碍车辆和停车场边界

    参数：
        parking_spaces: 库位顶点列表
        park_vertex: 停车场顶点
    '''
    vehicle_width = vehicle_config.vehicle_width
    vehicle_length = vehicle_config.vehicle_length

    '''
    将每个库位顶点转化为车辆顶点坐标
    '''
    # 计算库位车辆的顶点
    vehicle_vertex_polygons = []
    parking_spaces_polygons = []
    id = 0

    for park_sapce in parking_spaces:
        parking_spaces_polygons.append(
             Polygon([
                (park_sapce[0][0],park_sapce[0][1]),
                (park_sapce[1][0],park_sapce[1][1]),
                (park_sapce[2][0],park_sapce[2][1]),
                (park_sapce[3][0],park_sapce[3][1]),
            ])
        )
        if id != target_parking_id:
            # 计算四边形的中心点        
            park_space_xs = [point[0] for point in park_sapce]
            park_space_ys = [point[1] for point in park_sapce]
            park_sapce_x = sum(park_space_xs)/4
            park_sapce_y = sum(park_space_ys)/4

            # 产生位置和航向噪声
            noise_x = np.random.uniform(-0.1, 0.1, size=1)
            noise_y = np.random.uniform(-0.25, 0.25, size=1)
            noise_dir_deg = np.random.uniform(-6, 6, size=1)

            # 噪声扰动之后的航向和中心点
            cx = park_sapce_x + noise_x
            cy = park_sapce_y + noise_y
            c_dir_deg = noise_dir_deg
            c_dir_rad = np.radians(c_dir_deg)
            cx = cx.item()
            cy = cy.item()
            c_dir_rad = c_dir_rad.item()

            # 根据新长宽，计算新的顶点
            half_width = vehicle_width/2
            half_length = vehicle_length/2

            # 局部坐标系下的顶点
            local_vertices = np.array([
                [-half_length, half_width],  # Top-Left
                [half_length, half_width],   # Top-Right
                [half_length, -half_width],  # Bottom-Right
                [-half_length, -half_width]  # Bottom-Left
            ])

            # 旋转矩阵
            rotation_matrix = np.array([
                [np.cos(c_dir_rad), -np.sin(c_dir_rad)],
                [np.sin(c_dir_rad), np.cos(c_dir_rad)]
            ])

            # 全局坐标系下的顶点
            global_vertices = np.dot(local_vertices, rotation_matrix.T) + np.array([cx, cy])

            vehicle_polygon = [
                (global_vertices[0][0], global_vertices[0][1]),
                (global_vertices[1][0], global_vertices[1][1]),
                (global_vertices[2][0], global_vertices[2][1]),
                (global_vertices[3][0], global_vertices[3][1]),
            ]

            # vehicle_polygon = [
            #     (park_sapce_x - half_length, park_sapce_y - half_width),  # 左下
            #     (park_sapce_x + half_length, park_sapce_y - half_width),  # 右下
            #     (park_sapce_x + half_length, park_sapce_y + half_width),  # 右上
            #     (park_sapce_x - half_length, park_sapce_y + half_width),  # 左上
            # ]
            
            vehicle_vertex_polygons.append(Polygon(vehicle_polygon))
        id += 1

    '''
    将库位坐标转化为Polygon的形式    
    '''
    park_vertex_polygon = Polygon([
        (park_vertex[0][0],park_vertex[0][1]),
        (park_vertex[1][0],park_vertex[1][1]),
        (park_vertex[2][0],park_vertex[2][1]),
        (park_vertex[3][0],park_vertex[3][1]),
    ])
    # print('vehicle_vertex_polygons:',vehicle_vertex_polygons)

    return vehicle_vertex_polygons, park_vertex_polygon, parking_spaces_polygons


def ego_vehicle_vertex_midpoints(vehicle_vert):
    '''
    根据车辆的四个顶点获取车辆的polygon形式的四个顶点和四条边的中点。
    '''
    vehicle_vert = vehicle_vert[0]

    # 计算四条边的中点
    vehicle_minpoints_polygon = [
        Point((vehicle_vert[i][0]+vehicle_vert[(i+1)%4][0])/2,
              (vehicle_vert[i][1]+vehicle_vert[(i+1)%4][1])/2)
        for i in range(len(vehicle_vert))
    ]

    # 获取顶点的polygon形式
    vehicle_vertex_polygon = [
        Point(vehicle_vert[i][0],vehicle_vert[i][1])
        for i in range(len(vehicle_vert))
    ]

    return vehicle_vertex_polygon, vehicle_minpoints_polygon

def lidar_detection_distance(
        vehicle_ray_start_points,
        park_vehicle_polygons,
        park_vertex_polygon,
        vehicle_pos,
    ):
    '''
    根据车身的激光发射点、周围车辆顶点、停车场顶点等信息计算周围障碍物到主要的相对距离
    '''
    # 车辆中心坐标
    vehicle_center = Point(vehicle_pos[0], vehicle_pos[1])

    # 射线的最大长度
    ray_length = 10

    # 存储每个发生点到障碍物或墙壁交点的距离结果
    obs_dis = []
    wal_dis = []
    min_dis = []


    # 遍历每个射线的起点
    for start_point in vehicle_ray_start_points:

        # 计算从车辆中心到起点的角度
        angle = np.arctan2(start_point.y - vehicle_center.y, start_point.x - vehicle_center.x)

        # 计算射线终点
        end_x = start_point.x + ray_length*np.cos(angle)
        end_y = start_point.y + ray_length*np.sin(angle)
        ray = LineString([start_point, (end_x, end_y)])  # 创建射线
        
        # 初始化最小距离为射线最大长度
        obs_min_distance = ray_length
        wal_min_distance = ray_length

        # 检查每个障碍物的交点
        for obstacle  in park_vehicle_polygons:
            '''
            先判定射线的起点是否在障碍物内部
            1. 如果在内部则直接赋值-1
            2. 否则计算射线与障碍物有交点
            '''
            # 如果发射起点在障碍物内部则直接赋值-1
            if obstacle.contains(start_point):  
                obs_min_distance = -1
            # 如果发射起点不在障碍物内部则判定摄像是否与障碍物存在交点          
            else:   
                # 判定射线是否与障碍物存在交点
                intersection = ray.intersection(obstacle)
                # 如果射线与障碍物相交
                if not intersection.is_empty:
                    # 如果交点是一个点，计算距离
                    if intersection.geom_type == 'Point':
                        distance = start_point.distance(intersection)
                        obs_min_distance = min(obs_min_distance, distance)
                    # 如果交点是线段，计算距离到最近的点
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            distance = start_point.distance(point)
                            obs_min_distance = min(obs_min_distance, distance)
                    elif intersection.geom_type == 'LineString':
                        for point in intersection.coords:
                            point = Point(point)
                            # 判定交点知否在障碍物表面
                            if isinstance(obstacle, Polygon):
                                point_on_edge = is_point_on_polygon_edge(point, obstacle)
                            else:
                                point_on_edge = is_point_on_circle_edge(point, obstacle)                    
                            # 如果交点在障碍物表面则计算该点到射线起点的距离
                            if point_on_edge:
                                distance = start_point.distance(point)
                                obs_min_distance = min(obs_min_distance, distance)  

        # 将该起点到障碍物的最小距离添加到列表中
        obs_dis.append(obs_min_distance)

        # 检查射线与停车场墙壁交点
        for obstacle in [park_vertex_polygon]:
            '''
            先判定射线的起点是否在停车场外
            1. 如果在停车场外则直接赋值-1
            2. 否则计算射线与墙壁的交点
            '''
            # 判定发射起点是否在墙外
            if not obstacle.contains(start_point):
                wal_min_distance = -1
            else: 
                # 判定射线是否与障碍物存在交点
                intersection = ray.intersection(obstacle)
                # 如果射线与障碍物相交
                if not intersection.is_empty:
                    # 如果交点是一个点，计算距离
                    if intersection.geom_type == 'Point':
                        distance = start_point.distance(intersection)
                        wal_min_distance = min(wal_min_distance, distance)
                    # 如果交点是线段，计算距离到最近的点
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            distance = start_point.distance(point)
                            wal_min_distance = min(wal_min_distance, distance)
                    elif intersection.geom_type == 'LineString':
                        # print('obstacle:', obstacle)
                        for point in intersection.coords:
                            point = Point(point)
                            if isinstance(obstacle, Polygon):
                                point_on_edge = is_point_on_polygon_edge(point, obstacle)
                            else:
                                point_on_edge = is_point_on_circle_edge(point, obstacle)
                            
                            if point_on_edge:
                                distance = start_point.distance(point)
                                wal_min_distance = min(wal_min_distance, distance)    
        # 将该起点到墙壁的最小距离添加到列表中
        wal_dis.append(wal_min_distance)

    # 输出最小距离
    min_dis = [min(dis1, dis2) for dis1, dis2 in zip(obs_dis, wal_dis)]
    # min_dis = wal_dis 
    return min_dis


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

def env_stop_condition(
        parking_spaces, 
        vehicle_vertex, 
        ego_target_dis, 
        ego_target_dir, 
        bias_x, 
        bias_y, 
        park_vertex,
        obs_min_dis,
    ):
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
    cost_crash = 0

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

    # parking_vertex_areas = park_vertex    
    # # Compute the vertex of ego vehicle wether within the parking spaces area.
    # # 车辆顶点超过停车区域则终止轨迹，并设置负奖励
    # for i in range(len(vehicle_vertex[0])):
    #     vehicle_vertex_tuple = tuple(vehicle_vertex[0][i])
    #     if not is_point_in_rectangle(vehicle_vertex_tuple, parking_vertex_areas):
    #         # print('vehicle_vertex_tuple:', vehicle_vertex_tuple)
    #         # print('parking_vertex_areas:', parking_vertex_areas)
    #         reward_crash = -10
    #         terminated = True
    #         break
    
    # 车辆超出停车区域或者与其他车辆存在碰撞则终止轨迹，并设置负奖励
    np_obs_min_dis = np.array(obs_min_dis)
    if np.any(np_obs_min_dis<0):
        reward_crash = -10
        cost_crash = 10
        # terminated = True


    info = {'terminated': terminated,
            'reward_target': reward_target,
            'reward_crash': reward_crash,
            'cost_crash': cost_crash,}

    return info

def env_stop_condition_parallel(
        parking_spaces, 
        vehicle_vertex, 
        ego_target_dis, 
        ego_target_dir, 
        bias_x, 
        bias_y, 
        park_vertex,
        obs_min_dis,
    ):
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
    cost_crash = 0

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
    if abs(bias_x) < 0.5 and abs(bias_y)<0.1 and abs(ego_target_dir)<10/180*math.pi:
        reward_target = 10
        terminated = True

  
    # 车辆超出停车区域或者与其他车辆存在碰撞则终止轨迹，并设置负奖励
    np_obs_min_dis = np.array(obs_min_dis)
    if np.any(np_obs_min_dis<0):
        reward_crash = -10
        cost_crash = 4
        # terminated = True


    info = {'terminated': terminated,
            'reward_target': reward_target,
            'reward_crash': reward_crash,
            'cost_crash': cost_crash,}

    return info

def compute_env_reward(
        vehicle_pos, 
        target_pos, 
        parking_spaces, 
        vehicle_vertex, 
        park_vertex,
        target_area,
        ego_vehicle_area,
        init_ego_target_dis,
        current_step,
        max_step,
        obs_min_dis,
    ):
    '''
    Compute the reward based on the state
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir].
        target_pos: the position of the target; target_pos = [target_x, target_y, target_dir].
    Return:
        Reward: the reward of current state.
    '''

    alpha_dis = 0.5
    alpha_dir = 1
    alpha_target = 4
    alpha_crash = 0
    alpha_overlap = 0
    alpha_time = 0

    # 初始化奖励值
    reward_distance = 0
    reward_direction = 0
    reward_overlap = 0
    reward_time = 0

    # 设置安全距离
    safe_min_dis = 0.2
    crash_max_cost = 5
    

    '''
    计算路由成本
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
        park_vertex = park_vertex,
        obs_min_dis = obs_min_dis,
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
    # 综合奖励
    info['reward'] = reward

    '''
    计算与障碍物的距离成本
    '''
    min_dis = min(obs_min_dis)
    safe_dis_cost = 0
    if min_dis < safe_min_dis:
        safe_dis_cost = crash_max_cost - crash_max_cost / safe_min_dis* min_dis
    info['cost_dis'] = safe_dis_cost

    # 综合成本
    info['cost'] = info['cost_crash'] + info['cost_dis']

    return info

def compute_env_reward_parallel(
        vehicle_pos, 
        target_pos, 
        parking_spaces, 
        vehicle_vertex, 
        park_vertex,
        target_area,
        ego_vehicle_area,
        init_ego_target_dis,
        current_step,
        max_step,
        obs_min_dis,
    ):
    '''
    Compute the reward based on the state
    Input:
        vehicle_pos: the position of the ego vehicle; vehicle_pos = [pos_x, pos_y, pos_dir].
        target_pos: the position of the target; target_pos = [target_x, target_y, target_dir].
    Return:
        Reward: the reward of current state.
    '''

    alpha_dis = 0.5
    alpha_dir = 0.5
    alpha_target = 4
    alpha_crash = 0
    alpha_overlap = 0
    alpha_time = 0

    # 初始化奖励值
    reward_distance = 0
    reward_direction = 0
    reward_overlap = 0
    reward_time = 0

    # 设置安全距离
    safe_min_dis = 0.2
    # crash_max_cost = 5
    crash_max_cost = 2
    

    '''
    计算路由成本
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
    info = env_stop_condition_parallel(
        parking_spaces = parking_spaces, 
        vehicle_vertex = vehicle_vertex, 
        ego_target_dis = ego_target_dis, 
        ego_target_dir = ego_target_dir,
        bias_x = bias_x,
        bias_y = bias_y,
        park_vertex = park_vertex,
        obs_min_dis = obs_min_dis,
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
    # 综合奖励
    info['reward'] = reward

    '''
    计算与障碍物的距离成本
    '''
    min_dis = min(obs_min_dis)
    safe_dis_cost = 0
    if min_dis < safe_min_dis:
        safe_dis_cost = crash_max_cost - crash_max_cost / safe_min_dis* min_dis
    info['cost_dis'] = safe_dis_cost

    # 综合成本
    info['cost'] = info['cost_crash'] + info['cost_dis']

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

def Parallel_parking_hybrid_boundary(
        vehicle_config,
        parking_config,
    ):
    '''
    Set the parallel parking scenarios
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
    length_park = parking_config.park_length
    width_park = parking_config.park_width
    parking_num = parking_config.parking_num
    boundary_width = parking_config.parking_boundary
    width_lane = parking_config.lane_width


    parking_spaces = []
    # 下方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*length_park+boundary_width, 0+boundary_width]
        parking_rightlow = [(i+1)*length_park+boundary_width, 0+boundary_width]
        parking_righttop = [(i+1)*length_park+boundary_width, width_park+boundary_width]
        parking_lefttop = [i*length_park+boundary_width, width_park+boundary_width]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop]) 

    # 上方多个库位
    for i in range(parking_num):
        parking_leftlow = [i*length_park+boundary_width, width_park+2*width_lane+boundary_width]
        parking_rightlow = [(i+1)*length_park+boundary_width, width_park+2*width_lane+boundary_width]
        parking_righttop = [(i+1)*length_park+boundary_width, 2*width_park+2*width_lane+boundary_width]
        parking_lefttop = [i*length_park+boundary_width, 2*width_park+2*width_lane+boundary_width]
        parking_spaces.append([parking_leftlow, parking_rightlow, parking_righttop, parking_lefttop])

    # 双向车道线
    lane_marks = []
    leftpoints = [0, width_park + width_lane + boundary_width]
    rightpoints = [parking_num*length_park+2*boundary_width, width_park + width_lane + boundary_width]
    lane_marks.append([leftpoints, rightpoints])

    # 车库顶点
    park_leftlow = (0, 0)
    park_rightlow = (parking_num*length_park+2*boundary_width, 0)
    park_righttop = (parking_num*length_park+2*boundary_width, 2*(width_park + width_lane)+2*boundary_width)
    park_lefttop = (0, 2*(width_park + width_lane)+2*boundary_width)
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

def get_init_vehicle_pos_parallel(vehicle_area, vehicle_config, area_id):
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
    # if area_id == 1:
    #     vehicle_dir = 45
    # elif area_id == 2:
    #     vehicle_dir = 0
    vehicle_dir = 0


    # 补充航向随机值
    dir_offset = np.random.uniform(-10,10)
    vehicle_dir = vehicle_dir + dir_offset
    # 将角度转化为弧度
    vehicle_dir = vehicle_dir/180*math.pi
    vehicle_pos.append(vehicle_dir)
    
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


def get_target_parking_pos_parallel(target_parking_id, parking_spaces):
    '''
    Generate the target parking spaces during reset the environment in ParallelPark Env.
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
    target_dir = 0
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


def get_init_vehicle_area_parallel_org(target_parking_id, area_id, parking_config, vehicle_config, park_boundary):
    '''
    车辆初试区域
    '''
    park_width = parking_config.park_width
    park_length = parking_config.park_length
    lane_width = parking_config.lane_width
    
    # 车辆初始区域
    vehicle_area = []
 
    if area_id == 1:        # 倾斜入库位置（库位附近）
        x_0 = target_parking_id*park_length + park_boundary
        y_0 = park_width + park_boundary + 0.5*lane_width
        x_offset = 0.5*park_length
        y_offset = 0.2
        area_leftlow = [x_0 - 0.0*x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 1.0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 1.0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 - 0.0*x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    elif area_id == 2:      # 平行于库位位置
        x_0 = (target_parking_id+0.5)*park_length + park_boundary
        y_0 = park_width + park_boundary + 0.5*lane_width
        x_offset = 0.5*park_length
        y_offset = 0.1
        area_leftlow = [x_0 + 0.5*x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 1.0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 1.0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 + 0.5*x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    else:
        print('Error: Invalid value. The area_id must be 1, 2, or 3.')
        sys.exit(1) # 中断程序      
    return vehicle_area

def get_init_vehicle_area_parallel(target_parking_id, area_id, parking_config, vehicle_config, park_boundary):
    '''
    车辆初试区域
    '''
    park_width = parking_config.park_width
    park_length = parking_config.park_length
    lane_width = parking_config.lane_width
    
    # 车辆初始区域
    vehicle_area = []

    # print('area_id:', area_id)
    # quit()
 
    if area_id == 1:        # 倾斜入库位置（库位附近）
        x_0 = (target_parking_id+0.5)*park_length + park_boundary
        y_0 = 0.5*park_width + park_boundary 
        x_offset = 0.8
        y_offset = 0.2
        area_leftlow = [x_0 - 1.0*x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 1.0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 1.0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 - 1.0*x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    elif area_id == 2:      # 平行于库位位置
        x_0 = (target_parking_id+1.1)*park_length + park_boundary
        y_0 = park_width + park_boundary + 0.5*lane_width
        x_offset = 0.5*park_length
        y_offset = 0.1
        area_leftlow = [x_0 + 0.5*x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 1.0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 1.0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 + 0.5*x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    elif area_id == 3:      # 平行于库位位置
        x_0 = (target_parking_id+0.5)*park_length + park_boundary
        y_0 = park_width + park_boundary + 0.5*lane_width
        x_offset = 0.5*park_length
        y_offset = 0.1
        area_leftlow = [x_0 + 0.5*x_offset, y_0 - y_offset]
        area_rightlow = [x_0 + 1.0*x_offset, y_0 - y_offset]
        area_righttop = [x_0 + 1.0*x_offset, y_0 + y_offset]
        area_lefttop = [x_0 + 0.5*x_offset, y_0 + y_offset]
        vehicle_area.append([area_leftlow, area_rightlow, area_righttop, area_lefttop])
    else:
        print('Error: Invalid value. The area_id must be 1, 2, or 3.')
        sys.exit(1) # 中断程序      
    return vehicle_area