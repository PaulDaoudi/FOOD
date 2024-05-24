import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional
from agents.common.model_2_values import Policy2Values
from agents.common.storage import RolloutStorage


class PPO:
    """
    Traditional Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        actor_critic: Policy2Values,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        mini_batch_size: Optional[int] = None
    ) -> None:
        """
        Initializes the PPO algorithm with the specified parameters.
        """
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def get_actor_critic_losses(
        self,
        obs_batch: torch.Tensor,
        recurrent_hidden_states_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        pretanh_actions_batch: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        old_action_log_probs_batch: torch.Tensor,
        adv_targ: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the losses for the actor-critic model.

        Args:
            obs_batch (torch.Tensor): Batch of observations.
            recurrent_hidden_states_batch (torch.Tensor): Batch of recurrent hidden states.
            actions_batch (torch.Tensor): Batch of actions.
            pretanh_actions_batch (torch.Tensor): Batch of pre-tanh actions.
            value_preds_batch (torch.Tensor): Batch of value predictions.
            return_batch (torch.Tensor): Batch of returns.
            masks_batch (torch.Tensor): Batch of masks.
            old_action_log_probs_batch (torch.Tensor): Batch of old action log probabilities.
            adv_targ (torch.Tensor): Batch of advantage targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Value loss, action loss, and distribution entropy.
        """
        # Reshape to do in a single forward pass for all steps
        values, values_imit, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(
            obs_batch,
            recurrent_hidden_states_batch,
            masks_batch,
            actions_batch,
            pretanh_actions_batch
        )

        # Get losses
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                                      (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        return value_loss, action_loss, dist_entropy

    def update(self, source_rollouts: RolloutStorage) -> Dict[str, float]:
        """
        Updates the actor-critic model using the provided rollouts.

        Args:
            source_rollouts (RolloutStorage): Rollouts containing the data (from the source environment) for updating the model.

        Returns:
            Dict[str, float]: Dictionary containing important metrics.
        """

        # Source advantages
        source_advantages = source_rollouts.returns[:-1] - source_rollouts.value_preds[:-1]
        source_advantages = (source_advantages - source_advantages.mean()) / (source_advantages.std() + 1e-5)

        # Logs
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        counts = 0

        for e in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                raise NotImplementedError
                # data_generator = rollouts.recurrent_generator(
                #     advantages, self.num_mini_batch)
            else:
                data_generator = source_rollouts.feed_forward_generator(
                    fetch_normalized=True,
                    advantages=source_advantages,
                    num_mini_batch=self.num_mini_batch,
                    mini_batch_size=self.mini_batch_size
                )

            for sample in data_generator:

                counts += 1

                # Get data
                source_obs_batch, \
                source_recurrent_hidden_states_batch, \
                source_actions_batch, \
                source_pretanh_actions_batch, \
                source_value_preds_batch, \
                source_return_batch, \
                source_masks_batch, \
                source_old_action_log_probs_batch, \
                source_adv_targ = sample

                value_loss, action_loss, dist_entropy = \
                    self.get_actor_critic_losses(source_obs_batch,
                                                 source_recurrent_hidden_states_batch,
                                                 source_actions_batch,
                                                 source_pretanh_actions_batch,
                                                 source_value_preds_batch,
                                                 source_return_batch,
                                                 source_masks_batch,
                                                 source_old_action_log_probs_batch,
                                                 source_adv_targ)

                # Get losses
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # Logs
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        metrics = {
            'value_loss_epoch': value_loss_epoch,
            'action_loss_epoch': action_loss_epoch,
            'dist_entropy_epoch': dist_entropy_epoch,
        }

        return metrics
