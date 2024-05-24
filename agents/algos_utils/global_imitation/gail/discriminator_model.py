import torch
import numpy as np
import torch.nn as nn
from agents.algo_utils.global_imitation.utils import obs_batch_normalize, log_sum_exp, RunningMeanStd
from torch import autograd
import torch.nn.functional as F
from agents.common.storage import RolloutStorage
from typing import Callable


class GAILDiscriminator(nn.Module):
    """
    GAILDiscriminator is designed for Generative Adversarial Imitation Learning.

    Args:
        ob_dim (int): Observation dimension.
        ac_dim (int): Action dimension.
        hidden_dim (int): Hidden layer dimension.
        device (torch.device): Device to use for computations.
        type_gail (str): Type of GAIL (default: 'ikostrikov').
        imitation_input (str): Type of input for imitation ('obs', 'obs_act', or 'obs_act_obs').
    """

    def __init__(
        self, 
        ob_dim: int, 
        ac_dim: int, 
        hidden_dim: int, 
        device: torch.device, 
        type_gail: str = 'ikostrikov', 
        imitation_input: str = 'obs_act'
    ) -> None:
        super(GAILDiscriminator, self).__init__()
        self.imitation_input = imitation_input
        self.type_gail = type_gail
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        shape_ob_rms = 2 * ob_dim + ac_dim
        if self.imitation_input == 'obs':
            input_dim = ob_dim
        elif self.imitation_input == 'obs_act_obs':
            input_dim = 2 * ob_dim + ac_dim
            shape_ob_rms = input_dim
        else:
            input_dim = ob_dim + ac_dim

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1, bias=False))

        self.input_rms = RunningMeanStd(shape=shape_ob_rms)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
        self.device = device
        self.to(device)
        self.train()

    def forward(self) -> None:
        """
        Forward method is not implemented.
        """
        raise NotImplementedError()

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_action: torch.Tensor,
        expert_next_state: torch.Tensor,
        policy_state: torch.Tensor,
        policy_action: torch.Tensor,
        policy_next_state: torch.Tensor,
        lambda_: float = 10
    ) -> torch.Tensor:
        """
        Computes the gradient penalty for improved training stability.

        Args:
            expert_state (torch.Tensor): Expert states.
            expert_action (torch.Tensor): Expert actions.
            expert_next_state (torch.Tensor): Expert next states.
            policy_state (torch.Tensor): Policy states.
            policy_action (torch.Tensor): Policy actions.
            policy_next_state (torch.Tensor): Policy next states.
            lambda_ (float): Gradient penalty coefficient.

        Returns:
            torch.Tensor: Gradient penalty.
        """
        alpha = torch.rand(expert_state.size(0), 1)
        if self.imitation_input == 'obs':
            expert_data = expert_state
            policy_data = policy_state
        elif self.imitation_input == 'obs_act_obs':
            expert_data = torch.cat([expert_state, expert_action, expert_next_state], dim=1)
            policy_data = torch.cat([policy_state, policy_action, policy_next_state], dim=1)
        else:
            expert_data = torch.cat([expert_state, expert_action], dim=1)
            policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.tower(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def forward_tower(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes the input through the discriminator tower.

        Args:
            state (torch.Tensor): States.
            action (torch.Tensor): Actions.
            next_state (torch.Tensor): Next states.

        Returns:
            torch.Tensor: Discriminator output.
        """
        if self.imitation_input == 'obs':
            data = state
        elif self.imitation_input == 'obs_act_obs':
            data = torch.cat([state, action, next_state], dim=1)
        else:
            data = torch.cat([state, action], dim=1)
        return self.tower(data)

    def update(
        self, 
        expert_loader: DataLoader, 
        rollouts: RolloutStorage, 
    ) -> float:
        """
        Updates the GAILDiscriminator using policy and expert data.

        Args:
            expert_loader (DataLoader): DataLoader for expert data.
            rollouts (RolloutStorage): Rollouts containing policy data.

        Returns:
            float: Average loss value over the update steps.
        """
        # Initialize and generate data
        self.train()
        policy_data_generator = rollouts.feed_forward_generator(fetch_normalized=False,
                                                                advantages=None,
                                                                mini_batch_size=expert_loader.batch_size,
                                                                add_next_obs=True)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):

            # Get data
            expert_state, expert_action, _, expert_next_state = expert_batch
            policy_state, policy_action, policy_next_state = policy_batch[0], policy_batch[3], policy_batch[1]

            # Get normalized elements
            policy_sas = torch.cat([policy_state, policy_action, policy_next_state], dim=1)
            expert_sas = torch.cat([expert_state, expert_action, expert_next_state], dim=1)
            normalized_sas = obs_batch_normalize(torch.cat([policy_sas, expert_sas], dim=0), update_rms=True,
                                                 rms_obj=self.input_rms)
            policy_sas, expert_sas = torch.split(normalized_sas, [policy_state.size(0),
                                                                  expert_state.size(0)], dim=0)
            policy_state, policy_action, policy_next_state = torch.split(policy_sas,
                                                                         [policy_state.size(1),
                                                                          policy_action.size(1),
                                                                          policy_next_state.size(1)
                                                                          ], dim=1)
            expert_state, expert_action, expert_next_state = torch.split(expert_sas,
                                                                         [expert_state.size(1),
                                                                          expert_action.size(1),
                                                                          expert_next_state.size(1)], dim=1)

            # Get losses
            policy_d = self.forward_tower(policy_state, policy_action, policy_next_state)
            expert_d = self.forward_tower(expert_state, expert_action, expert_next_state)
            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device)
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device)
            )
            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action, expert_next_state,
                                             policy_state, policy_action, policy_next_state)

            # Logs
            loss += (gail_loss + grad_pen).item()
            n += 1

            # Training
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_batch_rewards(self, rollouts: RolloutStorage) -> None:
        """
        Predicts rewards for a batch of observations and actions.

        Args:
            rollouts (RolloutStorage): Rollouts containing policy data.
        """
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        next_obs = rollouts.raw_obs[1:].view(-1, self.ob_dim)
        sas = torch.cat([obs, acs, next_obs], dim=1)
        sas = obs_batch_normalize(sas, update_rms=False, rms_obj=self.input_rms)
        s, a, next_s = torch.split(sas, [obs.size(1), acs.size(1), next_obs.size(1)], dim=1)

        with torch.no_grad():
            self.eval()
            d = self.forward_tower(s, a, next_s)
            s = torch.sigmoid(d)
            if self.type_gail == 'original':
                rewards = -s.log()
            else:
                rewards = s.log() - (1 - s).log()
            rewards = rewards / np.sqrt(self.input_rms.var[0] + 1e-8)
            rewards = rewards.view(rollouts.num_steps, -1, 1)
            rollouts.rewards_imit.copy_(rewards)
