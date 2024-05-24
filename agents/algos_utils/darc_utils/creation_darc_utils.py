from agents.algo_utils.darc_utils.darc_classifiers import ConcatDiscriminator
from typing import Tuple, Any
import torch
import gym


def create_darc_classifiers(
    real_envs: gym.Env,
    cfg: Any,
    device: torch.device
) -> Tuple[ConcatDiscriminator, ConcatDiscriminator]:
    """
    Creates DARC classifiers for state-action and state-action-state pairs.

    Args:
        real_envs (gym.Env): The real environments.
        cfg (Any): Configuration object.
        device (torch.device): Device to run the classifiers on.

    Returns:
        Tuple[ConcatDiscriminator, ConcatDiscriminator]: The state-action and state-action-state classifiers.
    """
    num_state = real_envs.observation_space.shape[0]
    if real_envs.action_space.__class__.__name__ == 'Discrete':
        num_action = 1
    else:
        num_action = real_envs.action_space.shape[0]
    d_sa = ConcatDiscriminator(
        num_state + num_action, 
        cfg['classifier_hidden'], 
        2, 
        device,
        dropout=cfg['dis_dropout']
    ).float().to(device)
    d_sas = ConcatDiscriminator(
        2 * num_state + num_action, 
        cfg['classifier_hidden'], 
        2, 
        device,
        dropout=cfg['dis_dropout']
    ).float().to(device)
    return d_sa, d_sas
