import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from collections import deque
from typing import Optional, Tuple, List


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self, 
        g: float = 10.0, 
        wind: float = 0, 
        early_done: bool = False
    ):
        """
        Initialize the Pendulum environment.

        Args:
            g (float): Gravity constant.
            wind (float): Wind force applied to the pendulum.
            early_done (bool): Whether to enable early termination.
        """
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.wind = 0
        self.early_done = early_done
        self.viewer = None
        self.history = deque(maxlen=40)

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the random number generator.

        Args:
            seed (Optional[int]): The seed value.

        Returns:
            List[int]: A list containing the seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, s: Tuple[float, float], u: float) -> float:
        """
        Calculate the reward for a given state and action.

        Args:
            s (Tuple[float, float]): The state (theta, thetadot).
            u (float): The action (torque).

        Returns:
            float: The reward.
        """
        th, thdot = s
        return - angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform one step in the environment.

        Args:
            u (np.ndarray): The action (torque).

        Returns:
            Tuple[np.ndarray, float, bool, dict]: Observation, reward, done flag, and additional info.
        """
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        self.history.append(abs(costs))

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt + self.wind
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        done = False
        if self.early_done:
            if len(self.history) >= 39 and np.sum(self.history) <= 2.5:
                done = True

        return self._get_obs(), -costs, done, {}

    def reset(self, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.

        Args:
            init_state (Optional[np.ndarray]): Optional initial state.

        Returns:
            np.ndarray: The initial observation.
        """
        self.history = deque(maxlen=40)
        high = np.array([np.pi / 4, 0.5])
        if init_state is not None:
            self.state = init_state
        else:
            self.state = self.np_random.uniform(low=-high, high=high) - np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            np.ndarray: The current observation.
        """
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode (str): The render mode ("human" or "rgb_array").

        Returns:
            Optional[np.ndarray]: The rendered frame (if mode is "rgb_array").
        """
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(self.l, 0.2*self.l)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self) -> None:
        """
        Close the environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].

    Args:
        x (float): The angle to normalize.

    Returns:
        float: The normalized angle.
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
