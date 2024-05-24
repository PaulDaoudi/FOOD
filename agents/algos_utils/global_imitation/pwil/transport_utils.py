import torch
import random
from typing import List, Dict, Any, Optional
from agents.common.storage import RolloutStorage


def vectorize(
    trajs_init: List[Dict[str, torch.Tensor]], 
    imitation_input: str = 'obs'
) -> torch.Tensor:
    """
    Vectorizes trajectories by concatenating the specified inputs.

    Args:
        trajs_init (List[Dict[str, torch.Tensor]]): List of trajectories.
        imitation_input (str): Specifies which parts of the trajectories to concatenate. Options are 'obs', 'obs_act', or 'obs_act_obs'.

    Returns:
        torch.Tensor: A single tensor containing the vectorized trajectories.
    """
    if imitation_input == 'obs':
        trajs = [t['observations'] for t in trajs_init]
    elif imitation_input == 'obs_act':
        trajs = [torch.concat([t['observations'], t['actions']], dim=-1) for t in trajs_init]
    else:
        trajs = [torch.concat([t['observations'], t['actions'], t['next_observations']], dim=-1)
                 for t in trajs_init]
    return torch.vstack(trajs)


def get_subsampled_trajs(
    trajs: List[Dict[str, torch.Tensor]], 
    subsampling: int = 20
) -> List[Dict[str, torch.Tensor]]:
    """
    Subsamples trajectories by taking every nth step.

    Args:
        trajs (List[Dict[str, torch.Tensor]]): List of trajectories to subsample.
        subsampling (int): Subsampling interval.

    Returns:
        List[Dict[str, torch.Tensor]]: List of subsampled trajectories.
    """
    if subsampling == 1:
        return trajs
    subsampled_trajs = []
    for traj in trajs:
        random_offset = random.randint(0, subsampling - 1)
        sub_traj = {
            'observations': traj['observations'][random_offset::subsampling],
            'actions': traj['actions'][random_offset::subsampling],
            'next_observations': traj['next_observations'][random_offset::subsampling]
        }
        subsampled_trajs.append(sub_traj)
    return subsampled_trajs


def get_trajs_from_rollout(
    rollout: RolloutStorage
) -> List[Dict[str, torch.Tensor]]:
    """
    Extracts trajectories from rollout storage.

    Args:
        rollout (RolloutStorage): Rollout storage containing the trajectories.

    Returns:
        List[Dict[str, torch.Tensor]]: List of trajectories.
    """
    trajs = []
    for nb in range(rollout.num_processes):
        traj = {}
        traj['observations'] = []
        traj['actions'] = []
        traj['next_observations'] = []

        all_obs = rollout.raw_obs[:, nb, :]
        all_actions = rollout.actions[:, nb, :]

        for i in range(all_actions.shape[0]):
            traj['observations'].append(all_obs[i])
            traj['actions'].append(all_actions[i])
            traj['next_observations'].append(all_obs[i+1])

        traj['observations'] = torch.vstack(traj['observations']).float()
        traj['actions'] = torch.vstack(traj['actions']).float()
        traj['next_observations'] = torch.vstack(traj['next_observations']).float()

        trajs.append(traj)

    return trajs


def get_rollout_from_file(
    filename: str, 
    action_space: Optional[Any] = None, 
    recurrent_hidden_state_size: 
    int = None
) -> RolloutStorage:
    """
    Loads rollout data from a file and converts it into RolloutStorage format.

    Args:
        filename (str): Path to the file containing the rollout data.
        action_space (Any): Action space of the environment.
        recurrent_hidden_state_size (int, optional): Size of the recurrent hidden state.

    Returns:
        RolloutStorage: Rollout storage containing the loaded data.
    """    
    import pickle
    import numpy as np
    from agents.common.storage import RolloutStorage

    # Retrieve data
    with open(filename, 'rb') as fd:
        data = pickle.load(fd)
    trajs = {}

    # Get states and add next states at the end
    trajs.update({'states': torch.from_numpy(data['states'].astype(np.float32))})
    trajs.update({'actions': torch.from_numpy(data['actions'].astype(np.float32))})
    trajs.update({'pretanh_actions': torch.from_numpy(data['pretanh_actions'].astype(np.float32))})
    trajs.update({'next_states': torch.from_numpy(data['next_states'].astype(np.float32))})
    # trajs.update({'rewards': torch.from_numpy(data['rewards'].astype(np.float32)).squeeze(dim=2)})
    trajs.update({'rewards': torch.from_numpy(data['rewards'].astype(np.float32))})

    # Create rollouts
    num_processes, num_steps, obs_shape = trajs['states'].shape
    obs_shape = tuple([obs_shape])
    real_rollouts = RolloutStorage(num_steps,
                                   num_processes,
                                   obs_shape,
                                   action_space,
                                   recurrent_hidden_state_size)

    # Get all states
    states = torch.from_numpy(data['states'].astype(np.float32)).view(-1,
                                                                      real_rollouts.num_processes,
                                                                      *obs_shape)
    final_next_state = torch.from_numpy(data['next_states'].astype(np.float32))[:, -1, :].view(-1,
                                                                                               real_rollouts.num_processes,
                                                                                               *obs_shape)
    all_states = torch.cat([states, final_next_state], dim=0)


    real_rollouts.raw_obs = all_states.view(-1, real_rollouts.num_processes, *obs_shape)
    real_rollouts.actions = trajs['actions'].view(-1, real_rollouts.num_processes, action_space.shape[0])  # [:, :-1, :]
    real_rollouts.pretanh_actions = trajs['pretanh_actions'].view(-1, real_rollouts.num_processes, action_space.shape[0])
    real_rollouts.rewards = trajs['rewards'].view(-1, real_rollouts.num_processes, 1)

    return real_rollouts
