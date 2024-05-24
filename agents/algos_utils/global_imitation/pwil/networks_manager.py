import os
import ot
import torch
import numpy as np
from agents.algo_utils.global_imitation.pwil.transport_utils import vectorize, get_trajs_from_rollout, \
    get_rollout_from_file, get_subsampled_trajs
from agents.algo_utils.global_imitation.global_imitation_utils import get_dataset, expert_data_loader


class NetworksManager:
    """
    Class that handles the optimal transport, the Wasserstein critic, and the priority buffers.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        source_envs (gym.Env): Source environments.
        source_rollouts (RolloutStorage): Rollouts containing policy data.
        action_eval_fn (Callable): Function to evaluate actions.
        device (torch.device): Device to use for computations.
        type_wass_reward (str): Type of Wasserstein reward. Defaults to 'traditional' (other option is PWIL).
        recurrent_hidden_state_size (int, optional): Size of the recurrent hidden state.
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        source_envs: Any,
        source_rollouts: RolloutStorage,
        action_eval_fn: Callable,
        device: torch.device,
        type_wass_reward: str = 'traditional',
        recurrent_hidden_state_size: int = None
    ) -> None:

        obs_dim = source_envs.observation_space.shape[0]
        acs_dim = source_envs.action_space.shape[0]
        self.imitation_input = cfg['imitation_input']

        # (infinite) Generator to loop over expert states and actions
        src_expert_data_filename = os.path.join(cfg['orig_cwd'],
                                                cfg['experts_dir'],
                                                'trajs_{}.pickle'.format(cfg['env_name']))
        self.target_rollouts = get_rollout_from_file(src_expert_data_filename,
                                                     action_space=source_envs.action_space,
                                                     recurrent_hidden_state_size=recurrent_hidden_state_size)
        self.source_rollouts = source_rollouts
        self.action_eval_fn = action_eval_fn
        self.type_wass_reward = type_wass_reward

    def predict_batch_rewards(
        self, 
        source_rollouts: RolloutStorage, 
        target_rollouts: RolloutStorage
    ) -> None:
        """
        Predicts rewards for a batch of source and target rollouts using optimal transport.

        Args:
            source_rollouts (RolloutStorage): Source rollouts containing policy data.
            target_rollouts (RolloutStorage): Target rollouts containing expert data.
        """
        time_horizon = source_rollouts.raw_obs.shape[1]

        # Scaling factor
        dim_obs = source_rollouts.raw_obs.shape[-1]
        dim_act = source_rollouts.actions.shape[-1]
        scaling = 5 * time_horizon / np.sqrt(dim_obs + dim_act)

        # Get target trajectory (or trajectories)
        target_trajs = get_trajs_from_rollout(target_rollouts)
        subsampled_target_trajs = get_subsampled_trajs(target_trajs, subsampling=2)
        target_vec_trajs = vectorize(subsampled_target_trajs, imitation_input=self.imitation_input)
        target_weights = 1. / len(target_vec_trajs) * torch.ones(len(target_vec_trajs))

        # Get source trajectories
        source_trajs = get_trajs_from_rollout(source_rollouts)
        # subsampled_source_trajs = get_subsampled_trajs(source_trajs, subsampling=2)
        source_vec_trajs = vectorize(source_trajs, imitation_input=self.imitation_input)
        source_weights = 1. / len(source_vec_trajs) * torch.ones(len(source_vec_trajs))

        # Get distance
        cost_matrix = ot.dist(source_vec_trajs, target_vec_trajs, metric='euclidean')
        _, logs = ot.emd2(source_weights, target_weights, cost_matrix, return_matrix=True)
        ot_couplings = logs['G']

        # Get rewards
        source_len = source_trajs[0]['observations'].shape[0]
        source_processes = len(source_trajs)
        rewards = - (cost_matrix * ot_couplings).sum(dim=1)
        if self.type_wass_reward == 'pwil':
            rewards = 5 * torch.exp(- scaling * rewards)
        rewards = rewards.view((source_processes, source_len, 1)).permute(1, 0, 2)
        source_rollouts.rewards_imit.copy_(rewards)

    def update(self, iter_num: int) -> Dict[str, float]:
        """
        Updates the rewards using the predicted rewards from the optimal transport.

        Args:
            iter_num (int): Iteration number (unused in this implementation).

        Returns:
            Dict[str, float]: Dictionary containing update metrics.
        """
        # Update rewards with values from the discriminator
        self.predict_batch_rewards(self.source_rollouts, self.target_rollouts, imit_only=self.imit_only)
        return {}
