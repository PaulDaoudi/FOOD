from gym.wrappers.clip_action import ClipAction
from envs.env_utils import TimeLimit
from typing import Optional


def get_foot_friction_mujoco(
    env_name: str,
    mode: str,
    max_episode_steps: int = 1000,
    foot_fric: float = 1.0
) -> gym.Env:
    """
    Creates a MuJoCo environment with modified foot friction.

    Args:
        env_name (str): The name of the environment (e.g., 'HalfCheetah-v2', 'ant').
        mode (str): Mode of the environment. Options are 'source' or 'target'.
        max_episode_steps (int): The maximum number of steps per episode. Default is 1000.
        foot_fric (float): The foot friction coefficient to set in 'target' mode. Default is 1.0.

    Returns:
        gym.Env: The created environment with modified foot friction.
    """
    assert mode in ['source', 'target']
    from gym.envs.mujoco import half_cheetah

    # Create env
    if 'HalfCheetah-v2' in env_name:
        idx = [5, 8]
        env = half_cheetah.HalfCheetahEnv()
    elif 'ant' in env_name:
        idx = [4, 7, 10, 13]
        env = ant_v3.AntEnv()
    else:
        raise NotImplementedError

    # Modify friction
    if mode == 'target':
        env.model.geom_friction[:][idx, 0] = foot_fric

    # Wrap env
    env = TimeLimit(ClipAction(env), max_episode_steps=max_episode_steps)

    return env
