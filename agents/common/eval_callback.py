from typing import Optional, Callable
from stable_baselines3.common.callbacks import BaseCallback
import gym
from envs import *
import os
import glob
from envs.get_env import get_env


class EvalOntargetCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback`` for evaluating the policy on the target environment.
    """

    def __init__(self,
        verbose: int = 0,
        current_trial: int = 0,
        target_policy: Optional[Any] = None,
        env: Optional[gym.Env] = None,
        eval_fn: Optional[Callable] = None,
        orig_cwd: str = './'
    ) -> None:
        """
        Intializes the Callback module.
        """
        super(EvalOntargetCallback, self).__init__(verbose)
        current_cwd = os.getcwd()
        self.path = f'{current_cwd}/output_{current_trial}.txt'
        # os.makedirs(f'{current_cwd}', exist_ok=True)
        self.env = env
        self.orig_cwd = orig_cwd
        self.eval_fn = eval_fn
        self.target_policy = target_policy
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_rollout_start(self) -> None:
        """
        This method is called before a new rollout starts. It evaluates the policy on both real and simulated environments.
        """
        test_env = get_env(self.env, mode='target', orig_cwd=self.orig_cwd)
        val = self.eval_fn(test_env,
                           self.target_policy,
                           render=False,
                           iters=14,
                           deterministic=True)

        source_env = get_env(self.env, mode='source', orig_cwd=self.orig_cwd)
        source_val = self.eval_fn(source_env,
                               self.target_policy,
                               render=False,
                               iters=14,
                               deterministic=True)

        with open(self.path, "a") as txt_file:
            print(f'env: {self.env}', file=txt_file)
            print(f'test_env: {test_env}', file=txt_file)
            print(f'source_env: {source_env}', file=txt_file)
            print(f'source val: {source_val}', file=txt_file)
            print(f'target {val} \n', file=txt_file)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
