from typing import List, Optional, Dict
import torch
from envs.envs import make_vec_envs
from agents.algo_utils.global_imitation.utils import obs_batch_normalize
from agents.common.names import get_env_kwargs
from agents.common.model_2_values import Policy2Values


def evaluate(
        actor_critic: Policy2Values,
        cfg: Dict,
        device: torch.device,
        ob_rms: Optional[torch.Tensor] = None,
        eval_source: bool = False,
        deterministic: bool = True,
        n_eval: int = 5
    )
    """
    Evaluates the performance of the given actor-critic model on the specified environment (whether source or target).

    Args:
        actor_critic (Policy2Values): The actor-critic model to be evaluated.
        cfg (Dict): Configuration object containing environment and evaluation settings.
        device (torch.device): The device (CPU/GPU) to run the evaluation on.
        ob_rms (Optional[torch.Tensor]): Optional observation normalization statistics.
        eval_source (bool): Whether to evaluate on the source environment. Defaults to False.
        deterministic (bool): Whether to use deterministic actions. Defaults to True.
        n_eval (int): Number of evaluation episodes. Defaults to 5.

    Returns:
        List[float]: A list of episode rewards obtained during evaluation.
    """

    env_kwargs = get_env_kwargs(cfg['env_name'], cfg)

    mode = 'target'
    if eval_source:
        mode = 'source'

    eval_envs = make_vec_envs(cfg['env_name'],
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
                              **env_kwargs
                              )

    num_processes = cfg['target_num_processes']

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < n_eval:
        with torch.no_grad():
            # Normalize obs
            normalized_obs = obs_batch_normalize(obs, update_rms=False, rms_obj=ob_rms)

            # Get action
            _, _, action, _, _, _, eval_recurrent_hidden_states, _ = actor_critic.act(
                normalized_obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return eval_episode_rewards
