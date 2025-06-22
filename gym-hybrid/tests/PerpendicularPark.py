import time
import gym
import math
import gym_hybrid
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


# 车辆类
class Vehicle:
    def __init__(self, length, width, wheel_base, turning_radius):
        self.length = length  # 车辆长度
        self.width = width    # 车辆宽度
        self.wheel_base = wheel_base  # 轴距
        self.turning_radius = turning_radius  # 最小转弯半径

# 计算车辆轮廓
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


# 路径可视化
def visualize_path(
        parking_polygon, 
        obstacles, 
        # path, 
        vehicle,
        park_spaces,
        lane_mark,
    ):
    plt.figure(figsize=(10, 10))

    # 绘制停车场
    x, y = parking_polygon.exterior.xy
    plt.plot(x, y, 'k-', label='Parking Area')

    # 绘制障碍物
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        plt.fill(x, y, 'r', alpha=0.5, label='Obstacle')

    # # 绘制路径
    # for state in path:
    #     vehicle_poly = vehicle_outline(state[0], state[1], state[2], vehicle)
    #     x, y = vehicle_poly.exterior.xy
    #     plt.plot(x, y, 'b-', alpha=0.5)

    # 绘制库位线
    for park_space in park_spaces:
        x, y = park_space.exterior.xy
        plt.plot(x, y, color='gray', alpha=0.8)

    # print('lane_mark')
    # 绘制车道线
    x = [lane_mark[0][0][0],lane_mark[0][1][0]]
    y = [lane_mark[0][0][1],lane_mark[0][1][1]]
    plt.plot(x, y, label='linear', linestyle='--', color='gray', linewidth=2.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=24)  # 主刻度
    plt.text(3, 7.5, "C$_s$ = 11.56; N$_g$ = 2; T$_c$ = 24.89", fontsize=24, color="k")
    # plt.title("Hybrid A*", fontsize=26)
    save_dir = './fig_area/fig_perpendicular_area_v01'
    plt.grid()
    plt.savefig(save_dir)
    plt.savefig(save_dir+'.pdf')


if __name__ == '__main__':
    env = gym.make('Perpendicular-v0')
    state = env.reset()
    print('init state:', state)

    ACTION_SPACE = env.action_space[0].n
    PARAMETERS_SPACE = env.action_space[1].shape[0]
    OBSERVATION_SPACE = env.observation_space.shape[0]
    # print('env.action:',env.action_space.sample())
    # quit()

    done = False
    while not done:
        # print('action:',env.action_space.sample() )
        # print('action[1]:',env.action_space.sample()[1][0] )
        # quit()
        env_set = env.get_env_set()
        start_pos = env_set['vehicle_pos']
        goal_pos = env_set['target_pos']
        obstacles_vehicle = env_set['obstacles']
        park_vertex = env_set['park_edge']
        vehicle_config = env_set['vehicle_config']
        park_spaces = env_set['park_spaces']
        lane_mark = env_set['lane_mark']

        # 定义停车场区域
        parking_polygon = park_vertex

        # 定义障碍物
        obstacles_list = obstacles_vehicle

                # 车辆参数
        vehicle = Vehicle(
            length = vehicle_config.vehicle_length, 
            width = vehicle_config.vehicle_width, 
            wheel_base = vehicle_config.wheelbase, 
            turning_radius = 5.0
        )

        visualize_path(parking_polygon, obstacles_list, vehicle, park_spaces, lane_mark)

        state, reward, done, info = env.step(env.action_space.sample())
        # print('state:', state)
        # print('reward:', reward)
        # print('done:', done)
        # print('action:',env.action_space.sample() )
        # quit()
        # print(f'State: {state} Reward: {reward} Done: {done}')
        time.sleep(0.1)