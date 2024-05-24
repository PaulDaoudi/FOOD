from gym.wrappers.clip_action import ClipAction
from agents.envs.env_utils import TimeLimit
from typing import Tuple


def get_highly_crippled_mujoco(
    mode: str,
    env_name: str,
    broken_joints: Tuple[int, ...] = (0, 1),
    max_episode_steps: int = 1000
) -> gym.Env:
    """
    Creates a MuJoCo environment with specific joints broken.

    Args:
        mode (str): The mode of the environment. Options are 'sim' or 'real'.
        env_name (str): The name of the environment (e.g., 'ant', 'cheetah', 'walker', 'reacher_7dof').
        broken_joints (Tuple[int, ...]): Indices of joints to be broken in 'real' mode. Default is (0, 1).
        max_episode_steps (int): The maximum number of steps per episode. Default is 1000.

    Returns:
        gym.Env: The created environment with specified broken joints.
    """  
    from agents.envs.darc_envs import BrokenJoints
    from gym.envs.mujoco import ant_v3, half_cheetah_v3, walker2d_v3, hopper_v3

    # new versions
    if env_name == "ant":
        env = ant_v3.AntEnv()
    elif env_name == "cheetah":
        env = half_cheetah_v3.HalfCheetahEnv()
    elif env_name == "walker":
        env = walker2d_v3.Walker2dEnv()
    elif env_name == "walker":
        env = hopper_v3.HopperEnv()
    else:
        raise NotImplementedError
    if mode == "sim":
        env = BrokenJoints(env, broken_joints=None)
    else:
        assert mode == "real"
        env = BrokenJoints(env, broken_joints=broken_joints)
    env = TimeLimit(ClipAction(env), max_episode_steps=max_episode_steps)
    return env
