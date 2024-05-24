import glob
import os
import numpy as np
import random
import torch
import torch.nn as nn
import yaml
from envs.envs import VecNormalize
import time
import smtplib
import socket
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Optional, Dict, Any



# Get a render function
def get_render_func(venv: Any) -> Optional[Callable]:
    """
    Gets the render function from the environment.

    Args:
        venv: The environment.

    Returns:
        Optional[Callable]: The render function if available.
    """
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv: Any) -> VecNormalize:
    """
    Gets the VecNormalize instance from the environment.

    Args:
        venv: The environment.

    Returns:
        VecNormalize: The VecNormalize instance.

    Raises:
        NotImplementedError: If the VecNormalize instance is not found.
    """
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)
    else:
        raise NotImplementedError(f'venv is {venv}')
    return None


class AddBias(nn.Module):
    """
    A custom bias layer for KFAC implementation.

    Args:
        bias (torch.Tensor): The bias tensor.
    """

    def __init__(self, bias: torch.Tensor) -> None:
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the bias to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with added bias.
        """
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    total_num_epochs: int, 
    initial_lr: float
) -> None:
    """
    Decreases the learning rate linearly.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        total_num_epochs (int): Total number of epochs.
        initial_lr (float): Initial learning rate.
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(
    module: nn.Module, 
    weight_init: Callable, 
    bias_init: Callable, 
    gain: float = 1
) -> nn.Module:
    """
    Initializes the weights and biases of a module.

    Args:
        module (nn.Module): The module to initialize.
        weight_init (Callable): The weight initialization function.
        bias_init (Callable): The bias initialization function.
        gain (float): Gain for the weight initialization.

    Returns:
        nn.Module: The initialized module.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir: str) -> None:
    """
    Cleans up the log directory.

    Args:
        log_dir (str): The log directory.
    """
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


class Timer(object):
    """
    A simple timer class for measuring elapsed time.
    """

    def __init__(self) -> None:
        self._time = None

    def __enter__(self) -> Timer:
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self._time = time.time() - self._start_time

    def __call__(self) -> Optional[float]:
        """
        Returns the elapsed time.

        Returns:
            Optional[float]: The elapsed time.
        """
        return self._time


def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prefix_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Adds a prefix to metric keys.

    Args:
        metrics (Dict[str, Any]): The metrics.
        prefix (str): The prefix to add.

    Returns:
        Dict[str, Any]: The metrics with prefixed keys.
    """
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

