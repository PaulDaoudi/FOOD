from algos.algo_utils.darc_utils.networks_manager import NetworksManager
from agents.common.model_2_values import Policy2Values
from agents.common.storage import RolloutStorage
from agents.common.sampler import StepSampler
from typing import Any


def create_darc_networks_manager(
    cfg: Any,
    source_envs: Any,
    source_rollouts: RolloutStorage,
    actor_critic: Policy2Values,
    source_sampler: StepSampler,
    device: torch.device,
    noise_std_discriminator: float = 0.1
) -> NetworksManager:
    """
    Creates DARC networks manager.

    Args:
        cfg (Any): Configuration object.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollout storage for the source environments.
        actor_critic (Policy2Values): Actor-critic model.
        source_sampler (StepSampler): Sampler for the source environments.
        device (torch.device): Device to run the networks on.
        noise_std_discriminator (float, optional): Standard deviation for discriminator noise. Default is 0.1.

    Returns:
        NetworksManager: created DARC networks manager.
    """
    networks_manager = NetworksManager(cfg,
                                       source_envs,
                                       source_rollouts,
                                       actor_critic.evaluate_actions,
                                       device=device,
                                       noise_std_discriminator=noise_std_discriminator)
    return networks_manager
