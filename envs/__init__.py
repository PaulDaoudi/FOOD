from gym.envs.registration import register
from pybullet_envs import *


register(
    id='MinitaurLinearBulletEnv-v0',
    entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
    max_episode_steps=1000,
    reward_threshold=15.0,
    kwargs={'accurate_motor_model_enabled': False, 'pd_control_enabled': True}
    )

register(
    id='heavy_HalfCheetah-v2',
    entry_point='envs.heavy_mujoco:HalfCheetahModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
