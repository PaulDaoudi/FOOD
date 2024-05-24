import torch
import numpy as np
from typing import Any, Generator, List, Tuple, Union


def zipsame(*seqs: List[Any]) -> zip:
    """
    Zips sequences together, ensuring they are of the same length.

    Args:
        *seqs (List[Any]): Sequences to zip together.

    Returns:
        zip: Zipped sequences.
    """
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)


def expert_gen_inf(
    filename: str,
    num_trajs: int,
    batch_sz: int,
    subsample_frequency: int,
    drop_last: bool,
    to_add: str = 'obs_act'
) -> Generator:
    """
    A generator which (infinitely) loops over the expert data.

    Args:
        filename (str): Path to the expert data file.
        num_trajs (int): Number of trajectories to use.
        batch_sz (int): Batch size for data loading.
        subsample_frequency (int): Frequency for subsampling the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        to_add (str, optional): Whether to include actions in the data. Defaults to 'obs_act'.

    Yields:
        Generator: Batch of expert data.
    """
    dataset = ExpertDataset(filename, num_trajs, subsample_frequency, to_add=to_add)
    dloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_sz,
                                          shuffle=True,
                                          drop_last=drop_last)

    gen = iter(dloader)

    while True:
        try:
            *data, = next(gen)
        except StopIteration:
            # restart generator
            gen = iter(dloader)
            *data, = next(gen)
        yield data


def get_dataset(
    filename: str,
    num_trajs: int,
    subsample_frequency: int,
    to_add: str ='obs_act'
) -> ExpertDataset:
    """
    Creates an expert dataset.

    Args:
        filename (str): Path to the expert data file.
        num_trajs (int): Number of trajectories to use.
        subsample_frequency (int): Frequency for subsampling the data.
        to_add (str, optional): Whether to include actions in the data. Defaults to 'obs_act'.

    Returns:
        ExpertDataset: The created expert dataset.
    """
    dataset = ExpertDataset(filename, num_trajs, subsample_frequency, to_add=to_add)
    return dataset


def expert_data_loader(
    dataset: ExpertDataset,
    drop_last: bool,
    num_grad_steps: int
) -> torch.utils.data.DataLoader:
    """
    Creates a data loader which (infinitely) loops over the expert data.

    Args:
        dataset (ExpertDataset): The expert dataset.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_grad_steps (int): Number of gradient steps.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    dloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1000*int(dataset.num_trajs)//num_grad_steps,
                                          shuffle=True,
                                          drop_last=drop_last)
    return dloader


class ExpertDataset(torch.utils.data.Dataset):
    """
    A dataset class for expert trajectories.

    Args:
        filename (str): Path to the expert data file.
        num_trajs (int): Number of trajectories to use.
        subsample_frequency (int): Frequency for subsampling the data.
        to_add (str, optional): Data to include ('obs_act' or 'all' or 'obs'). Defaults to 'obs_act'.
    """

    def __init__(
        self,
        filename: str,
        num_trajs: int,
        subsample_frequency: int,
        to_add: str = 'obs_act'
    ) -> None:
        self.num_trajs = num_trajs

        if filename.endswith('.pickle'):
            all_trajs = ExpertDataset.process_pickle(filename)
            print('all_trajs states shape:', all_trajs['states'].shape)

        else:
            raise NotImplementedError()

        self.to_add = to_add
        self.all_trajs = all_trajs

        # select the best expert trajectory for imitation
        es, idx = torch.sort(all_trajs['rewards'].sum(dim=1).view(-1), descending=True)
        idx = idx.tolist()[:num_trajs]
        print("Loaded {} expert trajs with average returns {:.2f}".format(num_trajs, es[:num_trajs].mean().item()))

        self.trajs = {}
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajs, ), dtype=torch.int32)

        for k, v in all_trajs.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajs):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajs[k] = torch.stack(samples)
            else:
                self.trajs[k] = data // subsample_frequency

    @staticmethod
    def process_pickle(filename: str) -> dict:
        """
        Processes a pickle file containing expert data.

        Args:
            filename (str): Path to the pickle file.

        Returns:
            dict: Processed expert trajectories.
        """
        import pickle
        with open(filename, 'rb') as fd:
            data = pickle.load(fd)

        trajs = {}
        trajs.update({'states': torch.from_numpy(data['states'].astype(np.float32))})
        trajs.update({'actions': torch.from_numpy(data['actions'].astype(np.float32))})
        trajs.update({'pretanh_actions': torch.from_numpy(data['pretanh_actions'].astype(np.float32))})
        trajs.update({'next_states': torch.from_numpy(data['next_states'].astype(np.float32))})
        # trajs.update({'rewards': torch.from_numpy(data['rewards'].astype(np.float32)).squeeze(dim=2)})
        trajs.update({'rewards': torch.from_numpy(data['rewards'].astype(np.float32))})
        trajs.update({'lengths': torch.LongTensor(data['lengths'])})
        return trajs

    def __len__(self) -> int:
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: Total number of data points.
        """
        return self.trajs['lengths'].sum()

    def __getitem__(self, i: int) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Gets a data point from the dataset.

        Args:
            i (int): Index of the data point.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]: The data point.
        """
        traj_idx = 0

        # find the trajectory (traj_idx) from which to return the data
        while self.trajs['lengths'][traj_idx].item() <= i:
            i -= self.trajs['lengths'][traj_idx].item()
            traj_idx += 1

        if self.to_add == 'obs_act':
            return [
                self.trajs['states'][traj_idx][i],
                self.trajs['actions'][traj_idx][i],
                self.trajs['pretanh_actions'][traj_idx][i]
            ]
        if self.to_add == 'all':
            return [
                self.trajs['states'][traj_idx][i],
                self.trajs['actions'][traj_idx][i],
                self.trajs['pretanh_actions'][traj_idx][i],
                self.trajs['next_states'][traj_idx][i]
            ]
        else:
            return [
                self.trajs['states'][traj_idx][i]
            ]
