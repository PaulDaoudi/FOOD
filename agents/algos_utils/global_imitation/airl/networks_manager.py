import os
from agents.algo_utils.global_imitation.global_imitation_utils import get_dataset, expert_data_loader
from agents.algo_utils.global_imitation.airl.discriminator_model import AIRLDiscriminator
from agents.common.storage import RolloutStorage
from typing import Callable, Dict


class NetworksManager:
    """
    Class that handles the AIRL discriminator, the Wasserstein critic, and the priority buffers.

    Args:
        cfg (dict): Configuration dictionary containing various settings.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollouts from the source environments.
        action_eval_fn (Callable): Function to evaluate actions.
        device (torch.device): Device to use for computations.
    """

    def __init__(
        self, 
        cfg: Dict, 
        source_envs: Any, 
        source_rollouts: RolloutStorage, 
        action_eval_fn: Callable, 
        device: torch.device
    ) -> None:

        obs_dim = source_envs.observation_space.shape[0]
        acs_dim = source_envs.action_space.shape[0]

        # Discriminator
        self.discriminator = AIRLDiscriminator(obs_dim, acs_dim, hidden_dim=64, device=device)

        # (infinite) Generator to loop over expert states and actions
        src_expert_data_filename = os.path.join(cfg['orig_cwd'],
                                                cfg['experts_dir'],
                                                'trajs_{}.pickle'.format(cfg['env_name']))
        self.expert_dataset = get_dataset(src_expert_data_filename,
                                          cfg['num_expert_trajs'],
                                          1,
                                          add_actions=True)

        self.airl_grad_steps = cfg['airl_grad_steps']
        self.optim_params = {k: cfg[k] for k in ['expert_batch_size', 'airl_grad_steps', 'airl_batch_size']}
        self.source_rollouts = source_rollouts
        self.action_eval_fn = action_eval_fn

    def update(self, iter_num: int) -> Dict[str, float]:
        """
        Updates the networks by training the AIRL discriminator and updating the rewards.

        Args:
            iter_num (int): Iteration number.

        Returns:
            Dict[str, float]: Dictionary containing the discriminator loss.
        """
        # Get data loader
        self.expert_gen = expert_data_loader(self.expert_dataset, True, self.optim_params['airl_batch_size'])

        # Update rewards with values from the discriminator
        self.discriminator.predict_batch_rewards(self.source_rollouts)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        for _ in range(self.optim_params['gail_graairl_grad_stepsd_steps']):
            discriminator_loss = self.discriminator.update(self.action_eval_fn,
                                                           self.expert_gen,
                                                           self.source_rollouts,
                                                           num_grad_steps=1)

        return {
            'discriminator_loss': discriminator_loss
        }
