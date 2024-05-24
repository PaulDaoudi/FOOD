from agents.algo_utils.global_imitation.create_networks_manager import create_imit_networks_manager
from agents.algo_utils.darc_utils.create_networks_manager import create_darc_networks_manager
from agents.common.storage import RolloutStorage
from typing import Any, Dict, Optional


def create_global_networks_manager(
    cfg: Any,
    source_envs: Any,
    source_rollouts: RolloutStorage,
    actor_critic: Any,
    source_sampler: Optional[Any] = None,
    device: str = 'cpu'
) -> Any:
    """
    Creates a global networks manager based on the agent type specified in the configuration.

    Args:
        cfg (Any): Configuration object.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollout storage for the source environments.
        actor_critic (Any): Actor-critic model.
        source_sampler (Any, optional): Sampler for the source environments. Defaults to None.
        device (str, optional): Device to run the networks on. Defaults to 'cpu'.

    Returns:
        Any: The created networks manager.

    Raises:
        NotImplementedError: If the agent type specified in the configuration is not implemented.
    """

    if 'Imit' in cfg['agent']:
        networks_manager = create_imit_networks_manager(cfg,
                                                        source_envs,
                                                        source_rollouts,
                                                        actor_critic,
                                                        source_sampler,
                                                        device)
    elif 'DARC' in cfg['agent']:
        networks_manager = create_darc_networks_manager(cfg,
                                                        source_envs,
                                                        source_rollouts,
                                                        actor_critic,
                                                        source_sampler,
                                                        device,
                                                        noise_std_discriminator=cfg['noise_std_discriminator'])
    else:
        raise NotImplementedError
    return networks_manager
