import numpy as np
from collections import deque
import torch
from agents.common.storage import RolloutStorage
from agents.algo_utils.global_imitation.utils import OrderedDefaultDict, FixSizeOrderedDict
from typing import List, Dict, Any, Optional


class StepSampler(object):
    """
    A sampler that collects samples from the environment for a specified number of steps.
    """

    def __init__(
        self,
        envs: Any,
        num_steps: int,
        rollouts: RolloutStorage,
        device: str = 'cpu'
    ) -> None:
        """
        Initializes the sampler class.
        """
        # Global params
        self._envs = envs
        self.num_steps = num_steps
        self.rollouts = rollouts
        self.device = device

        # Episode rewards
        self.episode_rewards = deque(maxlen=50)
        self.episode_lengths = deque(maxlen=50)

        # To initialize rollout
        obs = self.envs.reset()
        self.rollouts.raw_obs[0].copy_(obs)
        self.rollouts.normalized_obs[0].copy_(obs)
        self.rollouts.to(self.device)

        # old stuff (not sure if required anymore)
        # self._traj_steps = 0
        # self._current_observation = self.env.reset()

    def sample_n_trajs(
        self,
        actor_critic: Any,
        add_to_rollout: bool = True,
        n_trajs: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Samples n trajectories from the environment.

        Args:
            actor_critic: The actor-critic model.
            add_to_rollout: Whether to add samples to the rollout storage.
            n_trajs: Number of trajectories to sample.

        Returns:
            List[Dict[str, torch.Tensor]]: List of trajectories.
        """
        # get n trajs
        trajs = []

        for n in range(n_trajs):

            # to log
            observations = []
            actions = []
            next_observations = []
            obs = self.rollouts.obs[0]

            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, value_wass, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        self.rollouts.obs[step],
                        self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step]
                    )

                # logs
                observations.append(obs)
                actions.append(action)

                # Observe reward and next obs
                obs, reward, done, infos = self.envs.step(action)
                for info in infos:
                    if 'episode' in info.keys():
                        self.episode_rewards.append(info['episode']['r'])

                # Log next obs
                next_observations.append(obs)

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                if add_to_rollout:
                    self.rollouts.insert(obs,
                                         recurrent_hidden_states,
                                         action,
                                         action_log_prob,
                                         value,
                                         value_wass,
                                         reward,
                                         masks,
                                         bad_masks)

            # metrics to return
            metrics_to_return = dict(
                observations=torch.concat([*observations], dim=0).float(),
                actions=torch.concat([*actions], dim=0).float(),
                next_observations=torch.concat([*next_observations], dim=0).float(),
            )
            trajs.append(metrics_to_return)

        return trajs

    def sample(
        self,
        actor_critic: Any,
        add_to_rollout: bool = True,
        darc_rollout: Optional[Any] = None,
        deterministic: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Samples from the environment for the specified number of steps.

        Args:
            actor_critic: The actor-critic model.
            add_to_rollout: Whether to add samples to the rollout storage.
            darc_rollout: Optional DARC rollout storage.
            deterministic: Whether to use deterministic actions.

        Returns:
            List[Dict[str, torch.Tensor]]: List of metrics.
        """

        # to log
        internal_states = []
        raw_observations = []
        # normalized_observations = []
        actions = []
        raw_next_observations = []
        # normalized_next_observations = []
        rewards = []
        raw_obs = self.rollouts.raw_obs[0]
        # normalized_obs = self.rollouts.normalized_obs[0]

        for step in range(self.num_steps):

            # Sample actions
            with torch.no_grad():
                value, value_imit, action, pretanh_action, action_log_prob, dist_entropy, rnn_hxs, dist = actor_critic.act(
                    self.rollouts.normalized_obs[step],
                    self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step],
                    deterministic=deterministic
                )

            # logs
            raw_observations.append(raw_obs)
            # normalized_observations.append(normalized_obs)
            actions.append(action)

            # Observe reward and next obs
            raw_next_obs, reward, done, infos = self.envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    if self.type_imit == 'I2L':
                        self.num_finished_trajs += 1

            # add all to darc rollouts if required
            if darc_rollout is not None:
                darc_rollout.insert(raw_obs, action, raw_next_obs)

            # obs equals next_obs
            raw_obs = raw_next_obs.clone()

            # Log next obs
            raw_next_observations.append(raw_obs)
            rewards.append(reward)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            if add_to_rollout:
                self.rollouts.insert(raw_obs,
                                     rnn_hxs,
                                     action,
                                     pretanh_action,
                                     action_log_prob,
                                     value,
                                     value_imit,
                                     reward,
                                     masks,
                                     bad_masks)

        # metrics to return
        metrics_to_return = dict(
            internal_states=internal_states,
            observations=torch.vstack(raw_observations).float(),
            actions=torch.vstack(actions).float(),
            rewards=torch.stack(rewards).float(),
            next_observations=torch.vstack(raw_next_observations).float(),
        )

        return [ metrics_to_return ]

    @property
    def envs(self) -> Any:
        """
        Returns the environments.

        Returns:
            Any: The environments.
        """
        return self._envs
