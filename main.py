from typing import Dict
import os
from omegaconf import DictConfig
from ray import tune
import numpy as np
import torch
from envs.envs import make_vec_envs
from agents.common.utils import set_random_seed, update_linear_schedule, Timer
from agents.common.storage import RolloutStorage
from agents.common.sampler import StepSampler
from agents.common.creation_utils import create_agent, get_actor_critic
from agents.common.names import get_env_kwargs
from agents.algo_utils.global_imitation.utils import obs_batch_normalize
from agents.algo_utils.create_networks_manager import create_global_networks_manager
from evaluation import evaluate


def main(cfg: Dict):
    """
    Training and evaluation process of the RL agents in the Off-Dynamics setting.

    Args:
        cfg (Dict): A configuration object containing all necessary settings and hyperparameters for the training process.

    Returns:
        None
    """

    # For easier object manipulation
    cfg = DictConfig(cfg)

    # Set seed
    set_random_seed(cfg['repeat_run'] + 10)

    # Get global hyperparams
    num_run = cfg['repeat_run']
    device = torch.device(cfg['device'])
    cfg['log_dir'] = os.getcwd()

    # Get kwargs to create environments
    env_kwargs = get_env_kwargs(cfg['env_name'], cfg)

    # If ANE, add noise
    noisy_env = False
    noise_std = 0.0
    if 'ANE' in cfg['agent']:
        noisy_env = True
        noise_std = cfg['ane_noise_std']

    # Create envs
    target_envs, source_envs = [make_vec_envs(cfg['env_name'],
                                              cfg['repeat_run'],
                                              cfg['target_num_processes'],
                                              cfg['source_num_processes'],
                                              cfg['gamma'],
                                              cfg['log_dir'],
                                              device,
                                              True,  # False,
                                              mode=mode,
                                              orig_cwd=cfg['orig_cwd'],
                                              source_max_episode_steps=cfg['source_max_traj_length'],
                                              target_max_episode_steps=cfg['target_max_traj_length'],
                                              noisy_env=noisy_env,
                                              noise_std=noise_std,
                                              **env_kwargs
                                              )
                           for mode in ['target', 'sourve']]

    if cfg['type_training'] == 'use_only_target':
        source_envs = make_vec_envs(cfg['env_name'],
                                 cfg['repeat_run'],
                                 cfg['source_num_processes'],
                                 cfg['source_num_processes'],
                                 cfg['gamma'],
                                 cfg['log_dir'],
                                 device,
                                 True,  # False,
                                 mode='target',
                                 orig_cwd=cfg['orig_cwd'],
                                 source_max_episode_steps=cfg['source_max_traj_length'],
                                 target_max_episode_steps=cfg['target_max_traj_length'],
                                 **env_kwargs
                                 )

    actor_critic = get_actor_critic(cfg['agent'],
                                    target_envs.observation_space.shape,
                                    target_envs.action_space,
                                    base_kwargs={
                                        'hidden_size': cfg['hidden_size'],
                                        'recurrent': cfg['recurrent_policy']
                                    },
                                    gaussian_kwargs={
                                        'tanh_squash': cfg['tanh_squash'],
                                        'logstd_init': cfg['policy_logstd_init']
                                    })
    actor_critic.to(device)

    # Rollouts and Samplers
    source_rollouts = RolloutStorage(cfg['source_num_steps'],
                                  cfg['source_num_processes'],
                                  source_envs.observation_space.shape,
                                  source_envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

    source_sampler = StepSampler(
        source_envs,
        cfg['source_num_steps'],
        source_rollouts,
        device=device,
    )

    # Get optimal policy from source
    if cfg['initialize_target_policy']:
        actor_critic.load_state_dict(torch.load(os.path.join(
            cfg['orig_cwd'],
            cfg['path_optimal_source_policy'])))
        ob_rms = torch.load(os.path.join(
            cfg['orig_cwd'],
            cfg['path_optimal_ob_rms']))
        source_rollouts.ob_rms = ob_rms
        source_rollouts.normalized_obs[0] = obs_batch_normalize(source_rollouts.normalized_obs[0], update_rms=True, rms_obj=source_rollouts.ob_rms)

    # Build the required elements for the considered agent
    if 'DARC' in cfg['agent'] or 'Imit' in cfg['agent']:
        networks_manager = create_global_networks_manager(cfg,
                                                          source_envs,
                                                          source_rollouts,
                                                          actor_critic,
                                                          source_sampler,
                                                          device,
                                                          imit_only=imit_only)

    # Create agents
    agent = create_agent(actor_critic,
                         cfg,
                         source_sampler,
                         source_envs,
                         target_envs)

    # Log everything there
    metrics = {}

    # Get number of n_epochs
    for epoch in range(cfg['n_epochs']):

        # Sample trajectories from sourceulator with current policy (done with n_processes)
        with Timer() as source_timer:
            source_trajs = source_sampler.sample(actor_critic)

        # Update bias to account for dynamics discrepancies
        if 'DARC' in cfg['agent'] or 'Imit' in cfg['agent']:
            # Update discriminators and rewards
            metrics_to_update = networks_manager.update(epoch)
            metrics.update(metrics_to_update)

        # Decrease learning rate linearly
        if cfg['use_linear_lr_decay']:
            utils.update_linear_schedule(
                agent.optimizer,
                epoch,
                cfg['n_epochs'],
                cfg['lr'])

        # Compute returns
        with torch.no_grad():
            source_next_value, source_next_value_imit = actor_critic.get_values(
                source_rollouts.normalized_obs[-1],
                source_rollouts.recurrent_hidden_states[-1],
                source_rollouts.masks[-1])
            source_next_value = source_next_value.detach()
            source_next_value_imit = source_next_value_imit.detach()
        source_rollouts.compute_both_returns(source_next_value,
                                             source_next_value_imit,
                                             cfg['use_gae'],
                                             cfg['gamma'],
                                             cfg['gae_lambda'],
                                             cfg['use_proper_time_limits'])

        # Update agent
        with Timer() as update_timer:
            metrics_to_update = agent.update(source_rollouts)
            metrics.update(metrics_to_update)
        source_rollouts.after_update()

        # Save agent if necessary
        if epoch == cfg['n_epochs'] - 1:
            save_path = os.path.join(cfg['orig_cwd'],
                                     'agents',
                                     'envs',
                                     cfg['env_name'],
                                     'models',
                                     cfg['agent'])
            os.makedirs(save_path, exist_ok=True)
            imit_coef = cfg['imit_coef']
            noise_std_discriminator = cfg['noise_std_discriminator']
            end_ = f'imit_coef_{imit_coef}_noise_discr_{noise_std_discriminator}' \
                   f'_decay_{cfg["use_linear_lr_decay"]}_squashed_{cfg["tanh_squash"]}'

            torch.save(actor_critic.state_dict(),
                       os.path.join(save_path, f'actor_critic_epoch_{epoch}_trial_{num_run}_{end_}'))
            torch.save(source_rollouts.ob_rms,
                       os.path.join(save_path, f'ob_rms_epoch_{epoch}_trial_{num_run}_{end_}'))

        # Tune report metrics
        if epoch % cfg['log_interval'] == 0 or epoch == cfg['n_epochs'] - 1:
            
            # Get true reward on target environment
            eval_episode_rewards = evaluate(actor_critic, cfg, device, source_rollouts.ob_rms)
            metrics['mean_avg_return'] = np.mean(eval_episode_rewards)

            if len(source_sampler.episode_rewards) > 1:
                metrics['source_mean_avg_return'] = np.mean(source_sampler.episode_rewards[-1])
                metrics['source_average_return'] = source_sampler.episode_rewards[-1]
            else:
                raise ValueError('Should not happen')

            # Time
            metrics['epoch'] = epoch
            metrics['update_time'] = update_timer()
            metrics['epoch_time'] = source_timer() + update_timer()

            # Report metrics
            tune.report(**metrics)
