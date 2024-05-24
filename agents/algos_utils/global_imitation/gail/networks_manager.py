import os
from agents.algo_utils.global_imitation.global_imitation_utils import get_dataset, expert_data_loader
from agents.algo_utils.global_imitation.gail.discriminator_model import GAILDiscriminator
from agents.common.storage import RolloutStorage
from typing import Callable, Dict, Any


class NetworksManager:
    """
    Class that handles the GAIL discriminator, its associated critic, and the priority buffers.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollouts containing policy data.
        action_eval_fn (Callable): Function to evaluate actions.
        device (torch.device): Device to use for computations.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        source_envs: Any,
        source_rollouts: RolloutStorage,
        action_eval_fn: Callable,
        device: torch.device = None
    ) -> None:

        obs_dim = source_envs.observation_space.shape[0]
        acs_dim = source_envs.action_space.shape[0]
        self.imitation_input = cfg['imitation_input']

        # Discriminator
        self.discriminator = GAILDiscriminator(obs_dim,
                                               acs_dim,
                                               hidden_dim=64,
                                               device=device,
                                               type_gail=cfg['type_gail'],
                                               imitation_input=self.imitation_input)

        # (infinite) Generator to loop over expert states and actions
        src_expert_data_filename = os.path.join(cfg['orig_cwd'],
                                                cfg['experts_dir'],
                                                'trajs_{}.pickle'.format(cfg['dir_env']))
        self.expert_dataset = get_dataset(src_expert_data_filename,
                                          cfg['num_expert_trajs'],
                                          1,
                                          to_add='all')
        self.optim_params = {k: cfg[k] for k in ['expert_batch_size', 'gail_grad_steps', 'gail_batch_size']}
        self.source_rollouts = source_rollouts
        self.action_eval_fn = action_eval_fn

    def update(self, iter_num: int) -> Dict[str, float]:
        """
        Updates the discriminator and the rewards using expert and policy data.

        Args:
            iter_num (int): Iteration number (unused in this implementation).

        Returns:
            Dict[str, float]: Dictionary containing the discriminator loss.
        """
        # Get data loader
        self.expert_gen = expert_data_loader(self.expert_dataset, True, self.optim_params['gail_batch_size'])

        # Update rewards with values from the discriminator
        self.discriminator.predict_batch_rewards(self.source_rollouts)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        for _ in range(self.optim_params['gail_grad_steps']):
            discriminator_loss = self.discriminator.update(self.expert_gen,
                                                           self.source_rollouts)

        return {
            'discriminator_loss': discriminator_loss
        }
