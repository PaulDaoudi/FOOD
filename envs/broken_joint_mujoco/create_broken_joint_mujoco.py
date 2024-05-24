from gym.wrappers.clip_action import ClipAction
from gym.envs.registration import EnvSpec
from agents.envs.env_utils import TimeLimit
import gym
from pybullet_envs import *
from typing import Optional


def get_broken_joint_mujoco(
    env_name: str,
    mode: str,
    broken_joint: Optional[int] = 0,
    max_episode_steps: int = 1000
) -> gym.Env:
    """
    Creates an environment with a broken joint.

    Args:
        env_name (str): The name of the environment (e.g., 'ant', 'cheetah', 'walker', 'hopper', 'Minitaur').
        mode (str): Mode of the environment. Options are 'source' or 'target'.
        broken_joint (Optional[int]): The index of the broken joint. Default is 0.
        max_episode_steps (int): The maximum number of steps per episode. Default is 1000.

    Returns:
        gym.Env: The created environment with the specified broken joint.
    """
    from agents.envs.darc_envs import BrokenJoint
    import agents.envs.reacher_7dof.reacher_7dof as reacher_7dof
    from gym.envs.mujoco import ant_v3, half_cheetah_v3, walker2d_v3, hopper_v3

    # New Gym versions
    if env_name == "ant":
        env = ant_v3.AntEnv()
        spec_name = 'Ant-v3'
    elif env_name == "cheetah":
        env = half_cheetah_v3.HalfCheetahEnv()
        spec_name = 'HalfCheetah-v3'
    elif env_name == "walker":
        env = walker2d.Walker2dEnv()
        spec_name = 'Walker2d-v2'
    elif env_name == "hopper":
        env = hopper.HopperEnv()
        spec_name = 'Hopper-v2'

    # Minitaur
    elif env_name == 'Minitaur':
        env = gym.make('MinitaurBulletEnv-v0').env

    else:
        raise NotImplementedError

    # Create env
    if mode == "source":
        env = BrokenJoint(env, broken_joint=None)
    else:
        assert mode == "target"
        env = BrokenJoint(env, broken_joint=broken_joint)
    env = TimeLimit(ClipAction(env), max_episode_steps=max_episode_steps)
    if 'Minitaur' not in env_name:
        env.unwrapped.spec = EnvSpec(spec_name)
    return env
