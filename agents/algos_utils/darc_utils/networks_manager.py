import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.common.storage import RolloutStorage
from agents.algo_utils.global_imitation.utils import obs_batch_normalize, RunningMeanStd
from agents.algo_utils.global_imitation.global_imitation_utils import get_dataset, expert_data_loader
from agents.algo_utils.darc_utils.darc_classifiers import ConcatDiscriminator
from typing import Any, Dict, Callable


class NetworksManager:
    """
    Class that handles DARC classifiers, their training, and the reward modification.

    Args:
        cfg (Dict): Configuration object.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollout storage for the source environments.
        action_eval_fn (Callable): Function to evaluate actions.
        device (torch.device): Device to run the networks on.
        noise_std_discriminator (float, optional): Standard deviation for discriminator noise. Default is 0.1.
    """
    def __init__(
        self,
        cfg: Dict,
        source_envs: Any,
        source_rollouts: RolloutStorage,
        action_eval_fn: Callable,
        device: torch.device,
        noise_std_discriminator: float = 0.1
    ) -> None:

        obs_dim = source_envs.observation_space.shape[0]
        acs_dim = source_envs.action_space.shape[0]
        self.device = device

        # DARC discriminators and optimizers
        d_sa, d_sas = self.create_darc_classifiers(source_envs, cfg, device)
        self.d_sa = d_sa
        self.d_sas = d_sas
        self.d_sa_optimizer = optim.Adam(self.d_sa.parameters(), lr=cfg['lr'], eps=cfg['eps'])
        self.d_sas_optimizer = optim.Adam(self.d_sas.parameters(), lr=cfg['lr'], eps=cfg['eps'])

        # (infinite) Generator to loop over expert states and actions
        src_expert_data_filename = os.path.join(cfg['orig_cwd'],
                                                cfg['experts_dir'],
                                                'trajs_{}.pickle'.format(cfg['dir_env']))
        self.expert_dataset = get_dataset(src_expert_data_filename,
                                          cfg['num_expert_trajs'],
                                          1,
                                          add_actions=False,
                                          add_all=True)
        self.optim_params = {k: cfg[k] for k in ['expert_batch_size',
                                                 'gail_grad_steps',
                                                 'gail_batch_size',
                                                 'darc_grad_steps']}

        self.source_rollouts = source_rollouts
        self.action_eval_fn = action_eval_fn

        # Normalization
        self.input_rms = RunningMeanStd(shape=2*obs_dim + acs_dim)
        self.noise_std_discriminator = noise_std_discriminator

    def create_darc_classifiers(
        self, 
        envs: Any, 
        cfg: Dict, 
        device: torch.device
    ) -> Tuple[ConcatDiscriminator, ConcatDiscriminator]:
        """
        Creates DARC classifiers for state-action and state-action-state pairs.

        Args:
            envs (Any): The environments.
            cfg (Dict): Configuration object.
            device (torch.device): Device to run the classifiers on.

        Returns:
            Tuple[ConcatDiscriminator, ConcatDiscriminator]: The state-action and state-action-state classifiers.
        """
        num_state = envs.observation_space.shape[0]
        if envs.action_space.__class__.__name__ == 'Discrete':
            num_action = 1
        else:
            num_action = envs.action_space.shape[0]
        d_sa = ConcatDiscriminator(num_state + num_action, cfg['classifier_hidden'], 2, device,
                                   dropout=cfg['dis_dropout']).float().to(device)
        d_sas = ConcatDiscriminator(2 * num_state + num_action, cfg['classifier_hidden'], 2, device,
                                    dropout=cfg['dis_dropout']).float().to(device)
        return d_sa, d_sas

    def update_darc_classifiers(
        self, 
        expert_loader: Any, 
        rollouts: RolloutStorage
    ) -> Dict[str, float]:
        """
        Updates the DARC classifiers using expert and policy data.

        Args:
            expert_loader (Any): Data loader for expert data.
            rollouts (RolloutStorage): Rollout storage containing policy data.

        Returns:
            Dict[str, float]: Dictionary of loss metrics for the DARC classifiers.
        """
        policy_data_generator = rollouts.feed_forward_generator(fetch_normalized=False,
                                                                advantages=None,
                                                                mini_batch_size=expert_loader.batch_size,
                                                                add_next_obs=True)

        d_sa_loss_epoch = 0
        d_sas_loss_epoch = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):

            n += 1

            # Get elements
            expert_state, expert_action, _, expert_next_state = expert_batch
            policy_state, policy_action, policy_next_state = policy_batch[0], policy_batch[3], policy_batch[1]

            # Get normalized elements
            policy_sas = torch.cat([policy_state, policy_action, policy_next_state], dim=1)
            expert_sas = torch.cat([expert_state, expert_action, expert_next_state], dim=1)
            normalized_sas = obs_batch_normalize(torch.cat([policy_sas, expert_sas], dim=0),
                                                 update_rms=True,
                                                 rms_obj=self.input_rms)
            policy_sas, expert_sas = torch.split(normalized_sas, [policy_state.size(0), expert_state.size(0)], dim=0)
            policy_state, policy_action, policy_next_state = torch.split(policy_sas, [policy_state.size(1),
                                                                                      policy_action.size(1),
                                                                                      policy_next_state.size(1)], dim=1)
            expert_state, expert_action, expert_next_state = torch.split(expert_sas, [expert_state.size(1),
                                                                                      expert_action.size(1),
                                                                                      expert_next_state.size(1)], dim=1)

            # Input noise: prevents overfitting (main DARC hyper-param)
            if self.noise_std_discriminator > 0:
                expert_state += torch.randn(expert_state.shape, device=self.device) * self.noise_std_discriminator
                expert_action += torch.randn(expert_action.shape, device=self.device) * self.noise_std_discriminator
                expert_next_state += torch.randn(expert_next_state.shape, device=self.device) * self.noise_std_discriminator
                policy_state += torch.randn(policy_state.shape, device=self.device) * self.noise_std_discriminator
                policy_action += torch.randn(policy_action.shape, device=self.device) * self.noise_std_discriminator
                policy_next_state += torch.randn(policy_next_state.shape, device=self.device) * self.noise_std_discriminator

            # Targets
            target_sa_logits = self.d_sa(expert_state, expert_action)
            target_sa_prob = F.softmax(target_sa_logits, dim=1)
            source_sa_logits = self.d_sa(policy_state, policy_action)
            source_sa_prob = F.softmax(source_sa_logits, dim=1)

            target_adv_logits = self.d_sas(expert_state, expert_action, expert_next_state)
            target_sas_prob = F.softmax(target_adv_logits + target_sa_logits, dim=1)
            source_adv_logits = self.d_sas(policy_state, policy_action, expert_next_state)
            source_sas_prob = F.softmax(source_adv_logits + source_sa_logits, dim=1)

            # Losses
            dsa_loss = (- torch.log(target_sa_prob[:, 0]) - torch.log(source_sa_prob[:, 1])).mean()
            dsas_loss = (- torch.log(target_sas_prob[:, 0]) - torch.log(source_sas_prob[:, 1])).mean()

            # Optimize discriminator(s,a) and discriminator(s,a,s')
            self.d_sa_optimizer.zero_grad()
            dsa_loss.backward(retain_graph=True)

            self.d_sas_optimizer.zero_grad()
            dsas_loss.backward()

            self.d_sa_optimizer.step()
            self.d_sas_optimizer.step()

            # Los
            d_sa_loss_epoch += dsa_loss.detach().item()
            d_sas_loss_epoch += dsas_loss.detach().item()

        d_sa_loss_epoch /= n
        d_sas_loss_epoch /= n

        metrics = {
            'd_sa_loss_epoch': d_sa_loss_epoch,
            'd_sas_loss_epoch': d_sas_loss_epoch,
        }

        return metrics

    def update(self, iter_num: int) -> Dict[str, float]:
        """
        Updates the networks and rewards for a given iteration.

        Args:
            iter_num (int): The iteration number.

        Returns:
            Dict[str, float]: Dictionary of metrics for the updates.
        """
        # Get data loader
        self.expert_gen = expert_data_loader(self.expert_dataset, True, self.optim_params['gail_batch_size'])

        # Update rewards with values from the discriminator
        self.source_rollouts.update_darc_rewards(self.d_sa, self.d_sas)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        for _ in range(self.optim_params['darc_grad_steps']):
            metrics = self.update_darc_classifiers(self.expert_gen, self.source_rollouts, sb3=self.sb3)

        return metrics
