import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.common.distributions import Bernoulli, Categorical, DiagGaussian
from agents.common.utils import init
from typing import Tuple, Dict, Any, Union


class Flatten(nn.Module):
    """
    Flatten layer for neural networks.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flattens the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Flattened tensor.
        """
        return x.view(x.size(0), -1)


class Policy2Values(nn.Module):
    """
    Policy model with two value heads for actor-critic methods.
    One is for traditional critic, the other is for imitation.
    """

    def __init__(self,
        obs_shape: Tuple[int, ...],
        action_space: gym.Space,
        base: Optional[nn.Module] = None,
        base_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes the Policy2Values model.

        Args:
            obs_shape (Tuple[int, ...]): Shape of the observation space.
            action_space (gym.Space): The action space of the environment.
            base (Optional[nn.Module]): The base network module.
            base_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the base network.
            gaussian_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the Gaussian distribution.
        """
        super(Policy2Values, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs, **gaussian_kwargs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        # tanh squashed or not
        self.tanh_squash = gaussian_kwargs['tanh_squash']

    @property
    def is_recurrent(self) -> bool:
        """
        Returns whether the model is recurrent.

        Returns:
            bool: True if the model is recurrent, False otherwise.
        """
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self) -> int:
        """
        Returns the size of the recurrent hidden state.

        Returns:
            int: Size of the recurrent hidden state.
        """
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor) -> None:
        """
        Forward pass is not implemented.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.
        """
        raise NotImplementedError

    def act(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[DiagGaussian, Categorical, Bernoulli]]:
        """
        Computes actions based on the policy.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Tuple: Values, imitation values, actions, pre-tanh actions, action log probabilities, distribution entropy, recurrent hidden states, distribution.
        """
        value, value_imit, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            if self.tanh_squash:
                pretanh_action, action = dist.mode()
            else:
                action = pretanh_action = dist.mode()
            # deterministic should only be used for policy "evaluation",
            # at which point we should not be requiring log_prob and entropy
            action_log_probs = dist_entropy = None
        else:
            rsample = dist.rsample()
            if len(rsample) == 2:  # TanhNormal
                pretanh_action, action = rsample
            else:  # Normal
                pretanh_action = action = rsample
            action_log_probs = dist.log_probs(pretanh_action.detach(), action.detach())
            dist_entropy = dist.entropy(pretanh_action, action).mean(0)

        return value, value_imit, action, pretanh_action, action_log_probs, dist_entropy, rnn_hxs, dist

    def get_values(
        self, 
        inputs: torch.Tensor, 
        rnn_hxs: torch.Tensor, 
        $masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the values of the inputs.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Values and imitation values.
        """
        value, value_imit, _, _ = self.base(inputs, rnn_hxs, masks)
        return value, value_imit

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        action: torch.Tensor,
        pretanh_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[DiagGaussian, Categorical, Bernoulli]]:
        """
        Evaluates the actions based on the policy.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.
            action (torch.Tensor): Actions.
            pretanh_action (torch.Tensor): Pre-tanh actions.

        Returns:
            Tuple: Values, imitation values, action log probabilities, distribution entropy, recurrent hidden states, distribution.
        """
        value, value_imit, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        # if self.tanh_squash:
        action_log_probs = dist.log_probs(pretanh_action, action)
        dist_entropy = dist.entropy(pretanh_action, action).mean(0)
        # else:
        #     action_log_probs = dist.log_probs(pretanh_action, action)
        #     dist_entropy = dist.entropy().mean()

        return value, value_imit, action_log_probs, dist_entropy, rnn_hxs, dist


class NNBase(nn.Module):
    """
    Base class for neural network modules.
    """

    def __init__(
        self, 
        recurrent: bool,
        recurrent_input_size: int, 
        hidden_size: int
    ) -> None:
        """
        Initializes the NNBase class.

        Args:
            recurrent (bool): Whether the network is recurrent.
            recurrent_input_size (int): Input size for the recurrent layer.
            hidden_size (int): Hidden size.
        """
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self) -> bool:
        """
        Returns whether the network is recurrent.

        Returns:
            bool: True if the network is recurrent, False otherwise.
        """
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self) -> int:
        """
        Returns the size of the recurrent hidden state.

        Returns:
            int: Size of the recurrent hidden state.
        """
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self) -> int:
        """
        Returns the output size of the network.

        Returns:
            int: Output size.
        """
        return self._hidden_size

    def _forward_gru(
        self, 
        x: torch.Tensor, 
        hxs: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU.

        Args:
            x (torch.Tensor): Input tensor.
            hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and updated hidden states.
        """
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    """
    Convolutional neural network base class.
    """

    def __init__(
        self, 
        num_inputs: int, 
        recurrent: bool = False, 
        hidden_size: int = 512
    ) -> None:
        """
        Initializes the CNNBase class.

        Args:
            num_inputs (int): Number of input channels.
            recurrent (bool, optional): Whether the network is recurrent. Defaults to False.
            hidden_size (int, optional): Hidden size. Defaults to 512.
        """
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(
        self, 
        inputs: torch.Tensor, 
        rnn_hxs: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Critic values, actor features, and updated hidden states.
        """
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    """
    Multi-layer perceptron base class.
    """

    def __init__(
        self, 
        num_inputs: int, 
        recurrent: bool = False, 
        hidden_size: int = 64
    ) -> None:
        """
        Initializes the MLPBase class.

        Args:
            num_inputs (int): Number of input features.
            recurrent (bool, optional): Whether the network is recurrent. Defaults to False.
            hidden_size (int, optional): Hidden size. Defaults to 64.
        """
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.critic_imit = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(
        self, 
        inputs: torch.Tensor, 
        rnn_hxs: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            inputs (torch.Tensor): Input tensor.
            rnn_hxs (torch.Tensor): Recurrent hidden states.
            masks (torch.Tensor): Masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Critic values, imitation critic values, actor features, and updated hidden states.
        """
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        return self.critic(x), self.critic_imit(x), hidden_actor, rnn_hxs
