# Taken from https://github.com/t6-thu/H2O/blob/d4859b67ca3c1af4c3c1eac3c65721c41e5ea2b0/Network/Weight_net.py


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Union, Any


class Discriminator(nn.Module):
    """
    A neural network discriminator for classifying input data.

    Args:
        num_input (int): Number of input features.
        num_hidden (int): Number of hidden units.
        num_output (int, optional): Number of output units. Defaults to 2.
        device (str, optional): Device to run the network on. Defaults to "cuda".
        dropout (bool, optional): Whether to use dropout. Defaults to False.
    """

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_output: int = 2,
        device: str = "cuda",
        dropout: bool = False
    ) -> None:
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Forward pass through the discriminator network.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input data.

        Returns:
            torch.Tensor: Output of the network.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        output = 2 * torch.tanh(self.fc3(x))
        return output


class ConcatDiscriminator(Discriminator):
    """
    Concatenate inputs along a specified dimension and then pass through the MLP.

    Args:
        dim (int, optional): Dimension along which to concatenate inputs. Defaults to 1.
    """

    def __init__(self, *args: Any, dim: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through the concatenated discriminator network.

        Args:
            *inputs (torch.Tensor): Input tensors to concatenate.
            **kwargs (Any): Additional arguments.

        Returns:
            torch.Tensor: Output of the network.
        """
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)
