import os
from typing import Dict, Any, Union


def get_global_env_name(env_name: str) -> str:
    """
    Returns the global environment name based on the given environment name.

    Args:
        env_name (str): The environment name.

    Returns:
        str: The global environment name.
    """
    # Pendulum
    if 'pendulum' in env_name:
        return 'pendulum'

    # V3 versions (initial experiments)
    elif 'cheetah' in env_name:
        return 'cheetah'
    elif 'ant' in env_name:
        return 'ant'
    elif 'hopper' in env_name:
        return 'hopper'
    elif 'walker' in env_name:
        return 'walker'

    # Gym versions (used to extract environments from GARAT)
    elif env_name == 'heavy_HalfCheetah-v2':
        return 'HalfCheetah-v2'
    elif 'Minitaur' in env_name:
        return 'Minitaur'

    # Othervise
    else:
        return env_name


def get_env_kwargs(env_name: str, config: Dict[str, Any]) -> Dict[str, Union[str, float]]:
    """
    Returns the environment keyword arguments based on the given environment name and configuration.

    Args:
        env_name (str): The environment name.
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Union[str, float]]: The environment keyword arguments.
    """
    kwargs = {}
    if 'gravity' in env_name:
        kwargs['gravity'] = config['gravity']
    elif 'heavy' in env_name:
        kwargs['mass'] = config['mass']
    elif 'friction' in env_name:
        kwargs['friction_multiplier'] = config['friction_multiplier']
    return kwargs


def get_imitation_input(agent: str) -> str:
    """
    Returns the imitation input type based on the given agent name.

    Args:
        agent (str): The agent name.

    Returns:
        str: The imitation input type.
    """
    if 'GAIL-SAS' in agent:
        imitation_input = 'obs_act_obs'
    elif 'GAIL-S' in agent:
        imitation_input = 'obs'
    elif 'GAIL' in agent:
        imitation_input = 'obs_act'
    elif 'GAIfO' in agent:
        imitation_input = 'obs'
    elif 'AIRL' in agent:
        imitation_input = 'obs_act'
    elif 'I2L' in agent:
        imitation_input = 'obs_act'
    elif 'Wass-SAS' in agent:
        imitation_input = 'obs_act_obs'
    elif 'Wass-S' in agent:
        imitation_input = 'obs'
    elif 'Wass' in agent:
        imitation_input = 'obs_act'
    else:
        raise NotImplementedError
    return imitation_input


def get_type_imit(config: Dict[str, Any]) -> str:
    """
    Returns the type of imitation based on the given configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        str: The type of imitation.
    """
    if 'GAIL' in config['agent']:
        type_imit = 'GAIL'
    elif 'GAIfO' in config['agent']:
        type_imit = 'GAIfO'
    elif 'AIRL' in config['agent']:
        type_imit = 'AIRL'
    elif 'I2L' in config['agent']:
        type_imit = 'I2L'
    elif 'Wass' in config['agent']:
        type_imit = 'Wass'
    else:
        raise NotImplementedError
    return type_imit


def remove_grid_search(config: Dict[str, Any]) -> None:
    """
    Removes grid search specific settings from the given configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        None
    """
    config['noise_std_discriminator'] = 0
    config['imitation_input'] = 'obs'
    config['type_gail'] = ['ikostrikov']
    config['type_wass_reward'] = 'classic'
    config['imit_coef'] = 1


def update_config(config: Dict[str, Any]) -> None:
    """
    Updates the given configuration based on various conditions.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        None
    """
    # Remove DARC hyperparams
    if 'DARC' not in config['agent']:
        config['noise_std_discriminator'] = config['noise_std_discriminator'][0]

    # Check Imitation hyper-parameters
    if 'Imit' in config['agent']:
        # Add info_to_transport
        config['imitation_input'] = get_imitation_input(config['agent'])

        # Type imitation
        config['type_imit'] = get_type_imit(config)
        if config['type_imit'] == 'GAIL':
            config['type_gail'] = ['ikostrikov']

    # Remove this if not Wass imititation (PWIL)
    if 'Wass' not in config['agent']:
        config['type_wass_reward'] = config['type_wass_reward'][0]

    # Remove if not ANE
    if 'ANE' not in config['agent']:
        config['ane_noise_std'] = [config['ane_noise_std'][0]]

    # Check
    if config['type_training'] in ['use_only_sim', 'use_only_real']:
        # Cannot bias learning if the agent considers one environment
        if 'Imit' in config['agent'] or 'DARC' in config['agent']:
            raise ValueError('Wrong agent!!')
        # Classic
        if config['type_training'] in ['use_only_sim']:
            config['initialize_real_policy'] = False
