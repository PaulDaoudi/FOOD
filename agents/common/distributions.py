import torch
import torch.nn as nn
from agents.common.utils import AddBias, init
from typing import Tuple


LOG_SIG_MIN = -10
LOG_SIG_MAX = 2

"""
Modify standard PyTorch Normal to be compatible with this code.
"""
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, _, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self, *args: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian distribution with optional Tanh squashing.
    """

    def __init__(
        self, 
        num_inputs: int, 
        num_outputs: int, 
        tanh_squash: bool, 
        logstd_init: float
    ) -> None:
        """
        Initializes the DiagGaussian distribution.
        """
        super(DiagGaussian, self).__init__()

        self.tanh_squash = tanh_squash
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(float(logstd_init)*torch.ones(num_outputs))

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass to compute the distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Any: Normal distribution.
        """
        action_mean = self.fc_mean(x)
        action_logstd = self.logstd(torch.zeros_like(action_mean))
        action_logstd = torch.clamp(action_logstd, LOG_SIG_MIN, LOG_SIG_MAX)

        if not self.tanh_squash:
            return FixedNormal(action_mean, action_logstd.exp())

        return TanhNormal(action_mean, action_logstd.exp())

class TanhNormal:
    """
    Represent distribution of X where Z ~ N(mean, std) and X ~ tanh(Z).
    """

    def __init__(self, normal_mean: torch.Tensor, normal_std: torch.Tensor) -> None:
        """
        Initializes the TanhNormal distribution.

        Args:
            normal_mean (torch.Tensor): Mean of the wrapped-normal distribution.
            normal_std (torch.Tensor): Standard deviation of the wrapped-normal distribution.
        """
        self._wrapped_normal = FixedNormal(normal_mean, normal_std)

    def log_probs(self, pretanh_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probabilities.

        Args:
            pretanh_actions (torch.Tensor): Inverse hyperbolic tangent of actions.
            actions (torch.Tensor): Actions.

        Returns:
            torch.Tensor: Log probabilities.
        """
        if pretanh_actions is None:
            assert False, "Numerical calculation of inverse-tanh could be unstable. To bypass this, provide pre-tanh actions to this function."

        # [warning:] this is the "incorrect" log-prob since we don't include the -log(1-actions^2) term which
        # comes from using tanh as the flow-layer atop the gaussian. Note that for PPO, this term cancels
        # out in the importance sampling ratio
        return self._wrapped_normal.log_probs(None, pretanh_actions)

    def entropy(self, pretanh_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return the (incorrect) entropy of the normal-gaussian (pre tanh). Since we do not have a closed analytical form
        for TanhNormal, an alternative is to return a Monte-carlo estimate of -E(log\pi)
        """
        return self._wrapped_normal.entropy()

    def mode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mode of the distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mode before and after applying Tanh.
        """
        m = self._wrapped_normal.mode()
        return m, torch.tanh(m)

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample with reparameterization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sample before and after applying Tanh.
        """
        z = self._wrapped_normal.rsample()
        return z, torch.tanh(z)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample without reparameterization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sample before and after applying Tanh.
        """
        z = self._wrapped_normal.sample().detach()
        return z, torch.tanh(z)



# Categorical
class FixedCategorical(torch.distributions.Categorical):
    """
    Fixed Categorical distribution.
    """

    def sample(self) -> torch.Tensor:
        """
        Sample from the distribution.

        Returns:
            torch.Tensor: Sampled actions.
        """
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probabilities.

        Args:
            actions (torch.Tensor): Actions.

        Returns:
            torch.Tensor: Log probabilities.
        """
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self) -> torch.Tensor:
        """
        Compute the mode of the distribution.

        Returns:
            torch.Tensor: Mode of the distribution.
        """
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    """
    Fixed Categorical distribution.
    """

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """
        Initializes the Categorical distribution module.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output features.
        """
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> FixedCategorical:
        """
        Forward pass to compute the distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            FixedCategorical: Categorical distribution.
        """
        x = self.linear(x)
        return FixedCategorical(logits=x)

# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    """
    Fixed Bernoulli distribution.
    """

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probabilities.

        Args:
            actions (torch.Tensor): Actions.

        Returns:
            torch.Tensor: Log probabilities.
        """
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """
        Compute the entropy.

        Returns:
            torch.Tensor: Entropy of the distribution.
        """
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        """
        Compute the mode of the distribution.

        Returns:
            torch.Tensor: Mode of the distribution.
        """
        return torch.gt(self.probs, 0.5).float()

class Bernoulli(nn.Module):
    """
    Bernoulli distribution.
    """

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """
        Initializes the Bernoulli distribution.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output features.
        """        
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> FixedBernoulli:
        """
        Forward pass to compute the distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            FixedBernoulli: Bernoulli distribution.
        """
        x = self.linear(x)
        return FixedBernoulli(logits=x)
