import torch
import torch.nn as nn
import torch.optim as optim


class PPO:

    def __init__(self,
                 actor_critic,
                 d_sa,
                 d_sas,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 clip_dynamics_ratio_min = 1e-5,
                 clip_dynamics_ratio_max = 1,
                 sa_prob_clip = 0.0,
                 **kwargs):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # darc parameters
        self.clip_dynamics_ratio_min = clip_dynamics_ratio_min
        self.clip_dynamics_ratio_max = clip_dynamics_ratio_max
        self.sa_prob_clip = sa_prob_clip
        self.d_sa = d_sa
        self.d_sas = d_sas


    def get_actor_critic_losses(self,
                                obs_batch,
                                recurrent_hidden_states_batch,
                                actions_batch,
                                value_preds_batch,
                                return_batch,
                                masks_batch,
                                old_action_log_probs_batch,
                                adv_targ
                                ):
        # Reshape to do in a single forward pass for all steps
        values, values_wass, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            obs_batch,
            recurrent_hidden_states_batch,
            masks_batch,
            actions_batch
        )

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

    def update(self, sim_rollouts, *args, **kwargs):
        """
        # To Do: Log values as well.
        """

        # sim advantages
        sim_advantages = sim_rollouts.returns[:-1] - sim_rollouts.value_preds[:-1]
        sim_advantages = (sim_advantages - sim_advantages.mean()) / (sim_advantages.std() + 1e-5)

        # for logging
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
                data_generator = sim_rollouts.feed_forward_generator(
                    sim_advantages, self.num_mini_batch
                )

            for sample in data_generator:

                counts += 1

                sim_obs_batch, \
                sim_recurrent_hidden_states_batch, \
                sim_actions_batch, \
                sim_value_preds_batch, \
                sim_return_batch, \
                sim_masks_batch, \
                sim_old_action_log_probs_batch, \
                sim_adv_targ = sample

                value_loss, action_loss, dist_entropy = \
                    self.get_actor_critic_losses(sim_obs_batch,
                                                 sim_recurrent_hidden_states_batch,
                                                 sim_actions_batch,
                                                 sim_value_preds_batch,
                                                 sim_return_batch,
                                                 sim_masks_batch,
                                                 sim_old_action_log_probs_batch,
                                                 sim_adv_targ)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

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
