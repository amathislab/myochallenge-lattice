import numpy as np
import torch
from gym import spaces
from typing import Generator, Dict, Optional
from stable_baselines3.common.type_aliases import (
    DictRolloutBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer


class RolloutBufferTensors(RolloutBuffer):
    def reset(self) -> None:
        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=torch.float,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=torch.float,
            device=self.device,
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.values = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.advantages = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: torch.Tensor
    ) -> None:
        """
        Compute returns on tensors, not np.array

        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_values = last_values.flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            # print(self.rewards[step].shape, next_values.shape, self.values[step].shape)
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(data))


class DictRolloutBufferTensors(RolloutBufferTensors):
    def reset(self) -> None:
        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = torch.zeros(
                (self.buffer_size, self.n_envs) + obs_input_shape,
                dtype=torch.float,
                device=self.device,
            )
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=torch.float,
            device=self.device,
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.values = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.advantages = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float, device=self.device
        )
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = obs[key].clone()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            # print(obs_.shape, self.observations[key][self.pos].shape)
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={
                key: obs[batch_inds] for (key, obs) in self.observations.items()
            },
            actions=self.actions[batch_inds],
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self.advantages[batch_inds].flatten(),
            returns=self.returns[batch_inds].flatten(),
        )
