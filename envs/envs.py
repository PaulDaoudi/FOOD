import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from envs.get_env import get_env
from typing import Union, List, Any

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(
    env_id: str,
    seed: int,
    rank: int,
    log_dir: str,
    allow_early_resets: bool,
    mode: str = 'target',
    orig_cwd: str = './',
    source_max_episode_steps: int = 200,
    target_max_episode_steps: int = 300,
    friction_multiplier: float = 1,
    gravity: float = 10,
    noisy_env: bool = False,
    noise_std: float = 0.0
):
    """
    Creates a gym environment with the specified settings.

    Args:
        env_id (str): The ID of the gym environment.
        seed (int): Random seed.
        rank (int): The rank of the environment (for vectorized environments).
        log_dir (str): Directory to save monitor logs.
        allow_early_resets (bool): Whether to allow early resets in Monitor.
        mode (str, optional): Mode of the environment ('source' or 'target'). Defaults to 'target'.
        orig_cwd (str, optional): Original current working directory. Defaults to './'.
        source_max_episode_steps (int, optional): Max episode steps for source environment. Defaults to 200.
        target_max_episode_steps (int, optional): Max episode steps for target environment. Defaults to 300.
        friction_multiplier (float, optional): Friction multiplier. Defaults to 1.
        gravity (float, optional): Gravity value. Defaults to 10.
        noisy_env (bool, optional): Whether to add noise to the environment. Defaults to False.
        noise_std (float, optional): Standard deviation of the noise. Defaults to 0.0.

    Returns:
        function: A thunk that creates the environment.
    """

    def _thunk():
        env = get_env(env_id,
                      mode=mode,
                      source_max_episode_steps=source_max_episode_steps,
                      target_max_episode_steps=target_max_episode_steps,
                      orig_cwd=orig_cwd,
                      friction_multiplier=friction_multiplier,
                      gravity=gravity,
                      noisy_env=noisy_env,
                      noise_std=noise_std,
                      )

        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    seed: int,
    target_num_processes: int,
    num_processes: int,
    gamma: float,
    log_dir: str,
    device: torch.device,
    allow_early_resets: bool,
    num_frame_stack: Optional[int] = None,
    mode: str = 'target',
    orig_cwd: str = './',
    source_max_episode_steps: int = 200,
    target_max_episode_steps: int = 300,
    normalize_obs: bool = False,
    added_stiffness: float = 0,
    dof_damping: float = 0,
    friction_multiplier: float = 1,
    gravity: float = 10,
    noisy_env: bool = False,
    noise_std: float = 0.0
) -> VecEnvWrapper:
    """
    Creates a vectorized environment with the specified settings.

    Args:
        env_name (str): The name of the environment.
        seed (int): Random seed.
        target_num_processes (int): Number of target processes.
        num_processes (int): Number of processes.
        gamma (float): Discount factor.
        log_dir (str): Directory to save monitor logs.
        device (torch.device): The device to run the environment on.
        allow_early_resets (bool): Whether to allow early resets in Monitor.
        num_frame_stack (Optional[int], optional): Number of frames to stack. Defaults to None.
        mode (str, optional): Mode of the environment ('target' or 'source'). Defaults to 'target'.
        orig_cwd (str, optional): Original current working directory. Defaults to './'.
        source_max_episode_steps (int, optional): Max episode steps for source environment. Defaults to 200.
        target_max_episode_steps (int, optional): Max episode steps for target environment. Defaults to 300.
        normalize_obs (bool, optional): Whether to normalize observations. Defaults to False.
        added_stiffness (float, optional): Added stiffness. Defaults to 0.
        dof_damping (float, optional): Degree of freedom damping. Defaults to 0.
        friction_multiplier (float, optional): Friction multiplier. Defaults to 1.
        gravity (float, optional): Gravity value. Defaults to 10.
        noisy_env (bool, optional): Whether to add noise to the environment. Defaults to False.
        noise_std (float, optional): Standard deviation of the noise. Defaults to 0.0.

    Returns:
        VecEnvWrapper: A vectorized environment.
    """
    if mode in ['target', 'evaluation']:
        glob_num_process = target_num_processses
    else:
        glob_num_process = num_processes

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, mode=mode, orig_cwd=orig_cwd,
                 source_max_episode_steps=source_max_episode_steps, target_max_episode_steps=target_max_episode_steps,
                 added_stiffness=added_stiffness,
                 dof_damping=dof_damping,
                 friction_multiplier=friction_multiplier,
                 gravity=gravity,
                 noisy_env=noisy_env,
                 noise_std=noise_std,
                 )
        for i in range(glob_num_process)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # To normalize obs or not
    if normalize_obs:
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, norm_reward=False)
            else:
                envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    """
    Gym wrapper to handle time limit masking.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def step(self, action: np.ndarray):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    """
    Gym wrapper to mask the goal in observations.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def observation(self, observation: np.ndarray):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    """
    Base class for transposing observations.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env: Optional[gym.Env] = None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env: Optional[gym.Env] = None, op: List = [2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob: np.ndarray):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv: Any, device: torch.device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions: Union[np.ndarray, torch.Tensor]):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    """
    VecNormalize wrapper to normalize observations.

    Args:
        venv (VecEnvWrapper): The vectorized environment to wrap.
        gamma (float, optional): Discount factor for normalization.
    """

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    """
    VecEnvWrapper to stack frames for the environment.

    Args:
        venv (VecEnvWrapper): The vectorized environment to wrap.
        nstack (int): Number of frames to stack.
        device (torch.device, optional): The device to run the environment on.
    """

    def __init__(self, venv: Any, nstack: int, device: Optional[torch.device] = None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
