from agents.algo_utils.global_imitation.airl.networks_manager import NetworksManager as AIRLNetworksManager
from agents.algo_utils.global_imitation.gail.networks_manager import NetworksManager as GAILNetworksManager
from agents.algo_utils.global_imitation.pwil.networks_manager import NetworksManager as WassNetworksManager
from agents.common.storage import RolloutStorage
from agents.common.model_2_values import Policy2Values
from typing import Any, Dict


dict_managers = {
    'GAIL': GAILNetworksManager,
    'AIRL': AIRLNetworksManager,
    'Wass': WassNetworksManager
}


def create_imit_networks_manager(
    cfg: Any,
    source_envs: Any,
    source_rollouts: RolloutStorage,
    actor_critic: Policy2Values,
    device: str = 'cpu'
) -> Any:
    """
    Creates an imitation networks manager based on the imitation type specified in the configuration.

    Args:
        cfg (Any): Configuration object.
        source_envs (Any): Source environments.
        source_rollouts (RolloutStorage): Rollout storage for the source environments.
        actor_critic (Policy2Values): Actor-critic model.
        device (str, optional): Device to run the networks on. Defaults to 'cpu'.

    Returns:
        Any: The created networks manager.
    """
    networks_manager = dict_managers[cfg['type_imit']](cfg,
                                                       source_envs,
                                                       source_rollouts,
                                                       actor_critic.evaluate_actions,
                                                       device=device)
    return networks_manager
