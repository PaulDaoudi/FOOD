import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from agents.algo_utils.global_imitation.utils import obs_batch_normalize, RunningMeanStd
import torch.nn.functional as F
from typing import Tuple, List, Generator, Optional, Any


def _flatten_helper(T: int, N: int, _tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens a tensor for batch processing.

    Args:
        T (int): Number of time steps.
        N (int): Number of processes.
        _tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.
    """
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    """
    Storage for rollouts used in the Off-Dynamics algorithms.
    """

    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: Tuple[int, ...],
        action_space: Any,
        recurrent_hidden_state_size: int
    ) -> None:
        """
        Initializes the class.
        """

        # Initializes data (normalized or not)
        self.raw_obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.normalized_obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.rewards_imit = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_imit_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns_imit = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.pretanh_actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
            self.pretanh_actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

        # In case
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # For normalization
        self.ob_rms = RunningMeanStd(shape=obs_shape)

    def to(self, device: torch.device) -> None:
        """
        Moves the storage to the specified device.

        Args:
            device (torch.device): The device to move the storage to.
        """
        self.raw_obs = self.raw_obs.to(device)
        self.normalized_obs = self.normalized_obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.rewards_imit = self.rewards_imit.to(device)
        self.value_preds = self.value_preds.to(device)
        self.value_imit_preds = self.value_imit_preds.to(device)
        self.returns = self.returns.to(device)
        self.returns_imit = self.returns_imit.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs: torch.Tensor,
        recurrent_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        pretanh_actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        value_imit_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor
    ) -> None:
        """
        Inserts a new step into the storage.

        Args:
            obs (torch.Tensor): Observations.
            recurrent_hidden_states (torch.Tensor): Recurrent hidden states.
            actions (torch.Tensor): Actions.
            pretanh_actions (torch.Tensor): Pre-tanh actions.
            action_log_probs (torch.Tensor): Action log probabilities.
            value_preds (torch.Tensor): Value predictions.
            value_imit_preds (torch.Tensor): Imitation value predictions.
            rewards (torch.Tensor): Rewards.
            masks (torch.Tensor): Masks.
            bad_masks (torch.Tensor): Bad masks.
        """
        self.raw_obs[self.step + 1].copy_(obs)
        self.normalized_obs[self.step + 1].copy_(obs_batch_normalize(obs, update_rms=True, rms_obj=self.ob_rms))
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.pretanh_actions[self.step].copy_(pretanh_actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.value_imit_preds[self.step].copy_(value_imit_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self, rollout: Optional[RolloutStorage] = None) -> None:
        """
        Updates the storage after each step.

        Args:
            rollout (Optional[RolloutStorage]): Optional rollout storage to copy from.
        """
        if rollout is not None:
            for i in range(self.raw_obs.shape[1]):
                self.raw_obs[0, i].copy_(rollout.raw_obs[-1].reshape(-1))
                self.normalized_obs[0, i].copy_(rollout.normalized_obs[-1].reshape(-1))
                self.recurrent_hidden_states[0, i].copy_(rollout.recurrent_hidden_states[-1].reshape(-1))
                self.masks[0, i].copy_(rollout.masks[-1].reshape(-1))
                self.bad_masks[0, i].copy_(rollout.bad_masks[-1].reshape(-1))
        else:
            self.raw_obs[0].copy_(self.raw_obs[-1])
            self.normalized_obs[0].copy_(self.normalized_obs[-1])
            self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
            self.masks[0].copy_(self.masks[-1])
            self.bad_masks[0].copy_(self.bad_masks[-1])

    def set_rollout(self, rollout: RolloutStorage) -> None:
        """
        Sets the rollout storage to the given rollout.

        Args:
            rollout (RolloutStorage): The rollout storage to set.
        """
        for i in range(self.raw_obs.shape[1]):
            self.raw_obs[0, i].copy_(rollout.raw_obs[0].reshape(-1))
            self.normalized_obs[0, i].copy_(rollout.normalized_obs[0].reshape(-1))
            self.recurrent_hidden_states[0, i].copy_(rollout.recurrent_hidden_states[0].reshape(-1))
            self.masks[0, i].copy_(rollout.masks[0].reshape(-1))
            self.bad_masks[0, i].copy_(rollout.bad_masks[0].reshape(-1))

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True
    ) -> None:
        """
        Computes the returns.

        Args:
            next_value (torch.Tensor): The next value.
            use_gae (bool): Whether to use Generalized Advantage Estimation (GAE).
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda.
            use_proper_time_limits (bool): Whether to use proper time limits.
        """
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * \
                            self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] *
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                            - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def update_darc_rewards(
        self, 
        d_sa: Any, 
        d_sas: Any, 
        use_darc_normalization: bool = True
    ) -> None:
        """
        Updates the DARC rewards.

        Args:
            d_sa (Any): State-action discriminator.
            d_sas (Any): State-action-state discriminator.
            use_darc_normalization (bool): Whether to use DARC normalization.
        """
        # careful, we consider big arrays, so we do it "nb_division" times to avoid RAM saturation
        nb_division = 15
        if use_darc_normalization:
            all_obs = self.normalized_obs[:-1].view(-1, *self.normalized_obs.size()[2:])
            all_next_obs = self.normalized_obs[1:].view(-1, *self.normalized_obs.size()[2:]).clone()
        else:
            all_obs = self.raw_obs[:-1].view(-1, *self.raw_obs.size()[2:])
            all_next_obs = self.raw_obs[1:].view(-1, *self.raw_obs.size()[2:]).clone()
        all_actions = self.actions.view(-1, self.actions.size(-1))

        # Split the input batch into nb_division smaller tensors
        input_splits_obs = torch.split(all_obs, all_obs.shape[0] // nb_division, dim=0)
        input_splits_actions = torch.split(all_actions, all_actions.shape[0] // nb_division, dim=0)
        input_splits_next_obs = torch.split(all_next_obs, all_next_obs.shape[0] // nb_division, dim=0)

        # Create a list to store the intermediate outputs
        sa_prob_list, sas_prob_list = [], []

        # Loop through the input splits and perform forward passes
        for i in range(len(input_splits_obs)):

            # Get the input split for the current forward pass
            input_split_obs = input_splits_obs[i]
            input_split_actions = input_splits_actions[i]
            input_split_next_obs = input_splits_next_obs[i]

            # Perform the forward pass on the input split
            # sa probs
            sa_logits_split = d_sa(input_split_obs, input_split_actions)
            sa_prob_split = F.softmax(sa_logits_split, dim=1)

            # sas probs
            adv_logits_split = d_sas(input_split_obs, input_split_actions, input_split_next_obs)
            sas_prob_split = F.softmax(adv_logits_split + sa_logits_split, dim=1)

            # Append the output split to the list of outputs
            sa_prob_list.append(sa_prob_split)
            sas_prob_list.append(sas_prob_split)

        # Concatenate the output splits along the batch dimension
        sa_prob = torch.cat(sa_prob_list, dim=0)
        sas_prob = torch.cat(sas_prob_list, dim=0)

#         # sa probs
#         sa_logits = d_sa(self.obs[:-1].view(-1, * self.obs.size()[2:]),
#                          self.actions.view(-1, self.actions.size(-1)))
#         sa_prob = F.softmax(sa_logits, dim=1)
#
#         # sas probs
#         adv_logits = d_sas(self.obs[:-1].view(-1, * self.obs.size()[2:]),
#                            self.actions.view(-1, self.actions.size(-1)),
#                            self.obs[1:].view(-1, * self.obs.size()[2:]))
#         sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        # get bonuses
        with torch.no_grad():
            # clipped pM/pM^
            log_ratio = torch.log(sas_prob[:, 0]) \
                        - torch.log(sas_prob[:, 1]) \
                        - torch.log(sa_prob[:, 0]) \
                        + torch.log(sa_prob[:, 1])

        # get new rewards
        rews = self.rewards.view(-1, 1)

        self.rewards = (rews + log_ratio.view(rews.shape)).view(self.num_steps, self.num_processes, 1)

    def compute_both_returns(
        self,
        next_value: torch.Tensor,
        next_imit_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True
    ) -> None:
        """
        Computes the returns for both the RL  and Imitation value functions.

        Args:
            next_value (torch.Tensor): The next value.
            next_imit_value (torch.Tensor): The next imitation value.
            use_gae (bool): Whether to use Generalized Advantage Estimation (GAE).
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda.
            use_proper_time_limits (bool): Whether to use proper time limits.
        """
        if use_proper_time_limits:
            if use_gae:
                # get both last values
                self.value_preds[-1] = next_value
                self.value_imit_preds[-1] = next_imit_value
                gae = 0
                gae_imit = 0
                for step in reversed(range(self.rewards.size(0))):
                    # classic values
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * \
                            self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]

                    # imit values
                    delta_imit = self.rewards_imit[step] + gamma * self.value_imit_preds[step + 1] * \
                                 self.masks[step + 1] - self.value_imit_preds[step]
                    gae_imit = delta_imit + gamma * gae_lambda * self.masks[step + 1] * gae_imit
                    gae_imit = gae_imit * self.bad_masks[step + 1]
                    self.returns_imit[step] = gae_imit + self.value_imit_preds[step]
            else:
                # both values
                self.returns[-1] = next_value
                self.returns_imit[-1] = next_imit_value
                for step in reversed(range(self.rewards.size(0))):
                    # classic returns
                    self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]

                    # imit returns
                    self.returns_imit[step] = (self.returns_imit[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards_imit[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_imit_preds[step]
        else:
            if use_gae:
                # both values
                self.value_preds[-1] = next_value
                self.value_imit_preds[-1] = next_imit_value
                gae = 0
                gae_imit = 0
                for step in reversed(range(self.rewards.size(0))):
                    # classic values
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                            - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]

                    # imit values
                    delta_imit = self.rewards_imit[step] + gamma * self.value_imit_preds[step + 1] * self.masks[step + 1] \
                            - self.value_imit_preds[step]
                    gae_imit = delta_imit + gamma * gae_lambda * self.masks[step + 1] * gae_imit
                    self.returns_imit[step] = gae_imit + self.value_imit_preds[step]
            else:
                # both values
                self.returns[-1] = next_value
                self.returns_imit[-1] = next_imit_value
                for step in reversed(range(self.rewards.size(0))):
                    # classic return
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

                    # imit return
                    self.returns_imit[step] = self.returns_imit[step + 1] * \
                                         gamma * self.masks[step + 1] + self.rewards_imit[step]

    def feed_forward_generator(
        self,
        fetch_normalized: bool,
        advantages: torch.Tensor,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
        add_next_obs: bool = False
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates mini-batches for feed-forward updates.

        Args:
            fetch_normalized (bool): Whether to fetch normalized observations.
            advantages (torch.Tensor): Advantages.
            num_mini_batch (Optional[int]): Number of mini-batches.
            mini_batch_size (Optional[int]): Size of each mini-batch.
            add_next_obs (bool): Whether to add next observations to the mini-batch.

        Yields:
            Generator[Tuple[torch.Tensor, ...], None, None]: A generator of mini-batches.
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if fetch_normalized:
                obs_batch = self.normalized_obs[:-1].view(-1, *self.normalized_obs.size()[2:])[indices]
            else:
                obs_batch = self.raw_obs[:-1].view(-1, *self.raw_obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            pretanh_actions_batch = self.pretanh_actions.view(-1, self.pretanh_actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            if add_next_obs:
                if fetch_normalized:
                    next_obs_batch = self.normalized_obs[1:].view(-1, * self.normalized_obs.size()[2:])[indices]
                else:
                    next_obs_batch = self.raw_obs[1:].view(-1, * self.raw_obs.size()[2:])[indices]
                yield obs_batch, \
                      next_obs_batch, \
                      recurrent_hidden_states_batch, \
                      actions_batch, \
                      pretanh_actions_batch, \
                      value_preds_batch, \
                      return_batch, \
                      masks_batch, \
                      old_action_log_probs_batch, \
                      adv_targ
            else:

                yield obs_batch, \
                      recurrent_hidden_states_batch, \
                      actions_batch, \
                      pretanh_actions_batch, \
                      value_preds_batch, \
                      return_batch, \
                      masks_batch, \
                      old_action_log_probs_batch, \
                      adv_targ

    def both_values_feed_forward_generator(
        self,
        fetch_normalized: bool,
        advantages: torch.Tensor,
        advantages_imit: torch.Tensor,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates mini-batches for feed-forward updates considering both RL and Imitation value functions.

        Args:
            fetch_normalized (bool): Whether to fetch normalized observations.
            advantages (torch.Tensor): Advantages.
            advantages_imit (torch.Tensor): Imitation advantages.
            num_mini_batch (Optional[int]): Number of mini-batches.
            mini_batch_size (Optional[int]): Size of each mini-batch.

        Yields:
            Generator[Tuple[torch.Tensor, ...], None, None]: A generator of mini-batches.
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if fetch_normalized:
                obs_batch = self.normalized_obs[:-1].view(-1, *self.normalized_obs.size()[2:])[indices]
            else:
                obs_batch = self.raw_obs[:-1].view(-1, *self.raw_obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            pretanh_actions_batch = self.pretanh_actions.view(-1, self.pretanh_actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            value_imit_preds_batch = self.value_imit_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            return_imit_batch = self.returns_imit[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
                adv_imit_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]
                adv_imit_targ = advantages_imit.view(-1, 1)[indices]

            yield obs_batch, \
                  recurrent_hidden_states_batch, \
                  actions_batch, \
                  pretanh_actions_batch, \
                  value_preds_batch, \
                  value_imit_preds_batch, \
                  return_batch, \
                  return_imit_batch, \
                  masks_batch, \
                  old_action_log_probs_batch, \
                  adv_targ, \
                  adv_imit_targ

    def recurrent_generator(
        self,
        advantages: torch.Tensor,
        num_mini_batch: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates mini-batches for recurrent updates.

        Args:
            advantages (torch.Tensor): Advantages.
            num_mini_batch (int): Number of mini-batches.

        Yields:
            Generator[Tuple[torch.Tensor, ...], None, None]: A generator of mini-batches.
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
