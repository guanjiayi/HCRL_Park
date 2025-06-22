from gym.envs.registration import register
from gym_hybrid.environments import MovingEnv
from gym_hybrid.environments import SlidingEnv

from .parking_environment_core import *

from .parking_environments import PerpendicularParkEnv
from .parking_environments import ParallelParkEnv
from .parking_environments import DiagonalParkEnv

from .parking_environments_safe import PerpendicularParkEnv_Safe
from .parking_environments_safe import ParallelParkEnv_Safe

from .parking_environments_safe_negative import PerpendicularParkEnv_Safe_Negative
from .parking_environments_safe_negative import ParallelParkEnv_Safe_Negative

register(
    id='Moving-v0',
    entry_point='gym_hybrid:MovingEnv',
)
register(
    id='Sliding-v0',
    entry_point='gym_hybrid:SlidingEnv',
)

# 定义垂直位泊车环境ID及入口
register(
    id='Perpendicular-v0',
    entry_point='gym_hybrid:PerpendicularParkEnv'
)

# 定义平行位泊车环境ID及入口
register(
    id='Parallel-v0',
    entry_point='gym_hybrid:ParallelParkEnv'
)

# 定义斜列泊车环境ID及入口
register(
    id='Diagonal-v0',
    entry_point='gym_hybrid:DiagonalParkEnv'
)

# 定义安全垂直泊车位环境ID及入口
register(
    id='Perpendicular_safe-v0',
    entry_point='gym_hybrid:PerpendicularParkEnv_Safe'
)

# 定义安全平行泊车位环境ID及入口
register(
    id='Parallel_safe-v0',
    entry_point='gym_hybrid:ParallelParkEnv_Safe'
)

# 定义安全垂直泊车位环境ID及入口 负奖励
register(
    id='Perpendicular_safe_negative-v0',
    entry_point='gym_hybrid:PerpendicularParkEnv_Safe_Negative'
)

# 定义安全平行泊车位环境ID及入口
register(
    id='Parallel_safe_negative-v0',
    entry_point='gym_hybrid:ParallelParkEnv_Safe_Negative'
)