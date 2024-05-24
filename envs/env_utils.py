import gym
from gym import ActionWrapper
import numpy as np
from gym.spaces import Box
from typing import Optional, SupportsFloat, Tuple


class NoisyRealEnv(gym.ActionWrapper):
    """
    Action wrapper to add noise to the actions of an environment.

    Args:
        env (gym.Env): The environment to wrap.
        noise_value (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, env: gym.Env, noise_value: float = 0.1):
        super(NoisyRealEnv, self).__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_shape = self.action_space.shape[0]
        self.noise_value = noise_value

    def action(self, action: np.ndarray) -> np.ndarray:
        action += np.random.normal(loc=0,
                                   scale=self.noise_value,
                                   size=self.action_shape)
        return action.clip(self.low, self.high)


class NoisyAction(ActionWrapper):
    """
    Action wrapper to add noise to the actions of an environment.

    Args:
        env (gym.Env): The environment to wrap.
        std_noise (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, env: gym.Env, std_noise: float = 1.0):
        assert isinstance(env.action_space, Box)
        super().__init__(env)
        self.std_noise = std_noise

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Adds noise to the action.
        """
        noisy_action = action + np.random.randn(action.shape[0],) * self.std_noise
        return noisy_action


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """
    Action wrapper to normalize the action space of an environment to [-1, 1].

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super(NormalizedEnv, self).__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action.
        """
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action.
        """
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class NoisyActionTimeLimit(gym.Wrapper):
    """
    Wrapper to add noise to actions and apply a time limit to an environment.

    Args:
        env (gym.Env): The environment to wrap.
        eps (float): The probability threshold to change the action.
        max_episode_steps (Optional[int]): The maximum number of steps per episode.
    """

    def __init__(
        self, 
        env: gym.Env, 
        eps: float = 0.3, 
        max_episode_steps: Optional[int] = None
    ):
        assert isinstance(env.action_space, Box)
        super().__init__(env)
        self.eps = eps
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Traditional step function.
        """
        # change action if required
        if np.random.randn() > self.eps:
            # get direction
            direction = self.env.env.env._get_obs()[2]
            if direction > -np.pi / 2 and direction < np.pi / 2:
                action = np.array([0, -0.25])
                # Check if the direction is between pi/2 and pi, or between -pi/2 and -pi
            elif (direction > np.pi / 2 and direction <= np.pi) or (direction < -np.pi / 2 and direction >= -np.pi):
                action = np.array([0, 0.25])

        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class TimeLimit(gym.Wrapper):
    """
    Wrapper to apply a time limit to an environment.

    Args:
        env (gym.Env): The environment to wrap.
        max_episode_steps (Optional[int]): The maximum number of steps per episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Traditional step function.
        """
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ForcedTimeLimit(gym.Wrapper):
    """
    Wrapper to apply a forced time limit to an environment, where the episode is always marked as done at the time limit.

    Args:
        env (gym.Env): The environment to wrap.
        max_episode_steps (Optional[int]): The maximum number of steps per episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Modify traditional step function.
        """
        observation, reward, done, info = self.env.step(action)
        done = False
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

"""
Utility functions used for classic control environments.
"""

from typing import Optional, SupportsFloat, Tuple


def verify_number_and_cast(x: SupportsFloat) -> float:
    """
    Verify parameter is a single number and cast to a float.
    """
    try:
        x = float(x)
    except (ValueError, TypeError):
        raise ValueError(f"An option ({x}) could not be converted to a float.")
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.
    
    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.
    
    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high

