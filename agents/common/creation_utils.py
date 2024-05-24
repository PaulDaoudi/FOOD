from typing import Any, Dict, Tuple
from agents import algo
from agents.common.model_2_values import Policy2Values
from agents.common.sampler import StepSampler


def get_actor_critic(
    agent_name: str,
    observation_space_shape: Tuple[int, ...],
    action_space: Any,
    base_kwargs: Dict[str, Any],
    gaussian_kwargs: Dict[str, Any]
) -> Policy2Values:
    """
    Returns a Policy2Values object initialized with the given parameters.

    Args:
        agent_name (str): Name of the agent.
        observation_space_shape (Tuple[int, ...]): Shape of the observation space.
        action_space (Any): The action space.
        base_kwargs (Dict[str, Any]): Base keyword arguments for the policy.
        gaussian_kwargs (Dict[str, Any]): Gaussian keyword arguments for the policy.

    Returns:
        Policy2Values: An instance of the Policy2Values class.
    """
    return Policy2Values(
        observation_space_shape,
        action_space,
        base_kwargs=base_kwargs,
        gaussian_kwargs=gaussian_kwargs
        )


def create_ppo_based_agent(
    actor_critic: Policy2Values,
    cfg: Dict[str, Any],
    source_sampler: StepSampler
) -> Any:
    """
    Creates an Off-Dynamics agent based on PPO.

    Args:
        actor_critic (Policy2Values): The actor-critic model.
        cfg (Dict[str, Any]): Configuration dictionary.
        source_sampler (StepSampler): The sampler used for collecting experiences.

    Returns:
        Any: An instance of a PPO-based agent.
    """
    if cfg['agent'] in ['PPO', 'PPOANE', 'PPODARC']:
        agent = algo.PPO(
            actor_critic,
            cfg['clip_param'],
            cfg['ppo_epoch'],
            cfg['num_mini_batch'],
            cfg['value_loss_coef'],
            cfg['entropy_coef'],
            lr=cfg['lr'],
            eps=cfg['eps'],
            max_grad_norm=cfg['max_grad_norm'],
            mini_batch_size=cfg['mini_batch_size'])
    elif 'DARC' in cfg['agent']:
        agent = algo.PPODARC(
            actor_critic,
            cfg['clip_param'],
            cfg['ppo_epoch'],
            cfg['num_mini_batch'],
            cfg['value_loss_coef'],
            cfg['entropy_coef'],
            lr=cfg['lr'],
            eps=cfg['eps'],
            max_grad_norm=cfg['max_grad_norm'])
    elif 'Imit' in cfg['agent']:
        agent = algo.PPOImit(actor_critic,
                             cfg['clip_param'],
                             cfg['ppo_epoch'],
                             cfg['num_mini_batch'],
                             cfg['value_loss_coef'],
                             cfg['entropy_coef'],
                             lr=cfg['lr'],
                             eps=cfg['eps'],
                             max_grad_norm=cfg['max_grad_norm'],
                             imit_coef=cfg['imit_coef'])
    else:
        raise NotImplementedError
    return agent


def create_agent(
    actor_critic: Policy2Values,
    cfg: Dict[str, Any],
    source_sampler: StepSampler
) -> Any:
    """
    Creates an agent based on the given configuration.

    Args:
        actor_critic (Policy2Values): The actor-critic model.
        cfg (Dict[str, Any]): Configuration dictionary.
        source_sampler (Any): The sampler used for collecting experiences.

    Returns:
        Any: An instance of an agent.
    """
    if 'PPO' in cfg['agent']:
        agent = create_ppo_based_agent(actor_critic,
                                       cfg,
                                       source_sampler)
    else:
        raise NotImplementedError

    return agent
