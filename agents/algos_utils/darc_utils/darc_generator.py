from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Generator, Tuple, Any


def darc_feed_forward_generator(
    fetch_normalized: bool,
    darc_target_rollouts: Any,
    darc_source_rollouts: Any,
    batch_size: int = 1000,
    mini_batch_size: int = 128,
    nb_epoch: int = 0
) -> Generator[
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, torch.Tensor
    ], 
    None, 
    None
]:
    """
    Generates mini-batches for DARC feed-forward training.

    Args:
        fetch_normalized (bool): Whether to fetch normalized observations.
        darc_target_rollouts (Any): DARC rollouts for the target environment.
        darc_source_rollouts (Any): DARC rollouts for the source environment.
        batch_size (int, optional): Size of each batch. Defaults to 1000.
        mini_batch_size (int, optional): Size of each mini-batch. Defaults to 128.
        nb_epoch (int, optional): Number of epochs. Defaults to 0.

    Yields:
        Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]: A generator of mini-batches.
    """

    # Get appropriate number of batch sizes
    target_sampler = BatchSampler(
        SubsetRandomSampler(range(batch_size)),
        mini_batch_size,
        drop_last=True)
    source_sampler = iter(BatchSampler(
        SubsetRandomSampler(range(batch_size)),
        mini_batch_size,
        drop_last=True))

    for target_indices in target_sampler:

        source_indices = next(source_sampler)

        if fetch_normalized:
            # Target sampler
            target_obs_batch = darc_target_rollouts.flatten_normalized_obs[target_indices]
            target_next_obs_batch = darc_target_rollouts.flatten_normalized_next_obs[target_indices]

            # Source sampler
            source_obs_batch = darc_source_rollouts.flatten_normalized_obs[source_indices]
            source_next_obs_batch = darc_source_rollouts.flatten_normalized_next_obs[source_indices]

        else:
            # Target sampler
            target_obs_batch = darc_target_rollouts.flatten_raw_obs[target_indices]
            target_next_obs_batch = darc_target_rollouts.flatten_raw_next_obs[target_indices]

            # Source sampler
            source_obs_batch = darc_source_rollouts.flatten_raw_obs[source_indices]
            source_next_obs_batch = darc_source_rollouts.flatten_raw_next_obs[source_indices]

        # Target sampler
        target_actions_batch = darc_target_rollouts.flatten_actions[target_indices]

        # Source sampler
        source_actions_batch = darc_source_rollouts.flatten_actions[source_indices]

        yield target_obs_batch, \
              target_next_obs_batch, \
              target_actions_batch, \
              source_obs_batch, \
              source_next_obs_batch, \
              source_actions_batch


def global_recurrent_generator():
    raise NotImplementedError
