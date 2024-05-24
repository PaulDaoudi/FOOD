import torch
from agents.algo_utils.global_imitation.i2l.misc.utils import obs_batch_normalize
from typing import Tuple, Any


class DarcRolloutStorage(object):
    """
    Same as traditional storage but much bigger, because it needs to store all values.

    Args:
        num_steps (int): Number of steps to store.
        num_processes (int): Number of processes.
        obs_shape (Tuple[int, ...]): Shape of the observations.
        action_space (Any): Action space.
        ob_rms (Any): Running mean and standard deviation of observations.
    """

    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: Tuple[int, ...],
        action_space: Any,
        ob_rms: Any
    ) -> None:

        self.raw_obs = torch.zeros(num_steps, num_processes, *obs_shape)
        self.normalized_obs = torch.zeros(num_steps, num_processes, *obs_shape)
        self.raw_next_obs = torch.zeros(num_steps, num_processes, *obs_shape)
        self.normalized_next_obs = torch.zeros(num_steps, num_processes, *obs_shape)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.num_steps = num_steps
        self.step = 0

        # In case
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Normalization
        self.ob_rms = ob_rms

    def to(self, device: torch.device) -> None:
        """
        Moves the storage tensors to the specified device.

        Args:
            device (torch.device): The device to move the tensors to.
        """
        self.raw_obs = self.raw_obs.to(device)
        self.normalized_obs = self.normalized_obs.to(device)
        self.raw_next_obs = self.raw_next_obs.to(device)
        self.normalized_next_obs = self.normalized_next_obs.to(device)
        self.actions = self.actions.to(device)

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor
    ) -> None:
        """
        Inserts a new step into the storage.

        Args:
            obs (torch.Tensor): The observations.
            actions (torch.Tensor): The actions.
            next_obs (torch.Tensor): The next observations.
        """
        self.raw_obs[self.step].copy_(obs)
        self.raw_next_obs[self.step].copy_(next_obs)

        self.normalized_obs[self.step].copy_(obs_batch_normalize(obs, update_rms=False, rms_obj=self.ob_rms))
        self.normalized_next_obs[self.step].copy_(obs_batch_normalize(next_obs, update_rms=False, rms_obj=self.ob_rms))

        self.actions[self.step].copy_(actions)
        self.step = (self.step + 1) % self.num_steps

    def flatten_all(self) -> None:
        """
        Flattens all stored tensors.
        """
        self.flatten_raw_obs = self.raw_obs.view(-1, self.raw_obs.shape[-1])
        self.flatten_raw_next_obs = self.raw_next_obs.view(-1, self.raw_next_obs.shape[-1])
        self.flatten_normalized_obs = self.normalized_obs.view(-1, self.normalized_obs.shape[-1])
        self.flatten_normalized_next_obs = self.normalized_next_obs.view(-1, self.normalized_next_obs.shape[-1])
        self.flatten_actions = self.actions.view(-1, self.actions.shape[-1])
