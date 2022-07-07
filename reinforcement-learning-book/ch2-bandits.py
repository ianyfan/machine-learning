"""
Multi-armed Bandit
==================
An implementation of the bandit problem from Chapter 2 of
Reinforcement Learning: An Introduction by Sutton & Barto.

Tests a number of different policies for bandits.
Performance is optimised by using NumPy to pre-compute random samples.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from math import log, sqrt
from typing import Any, Optional

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from numpy.random import Generator, SeedSequence
from tqdm.auto import trange  # type: ignore

_FloatArrayT = np.ndarray[Any, np.dtype[np.float64]]


class Bandit:
    arm_values: _FloatArrayT
    std_dev: float
    nonstationarity: Optional[tuple[float, float]]
    rng: Generator

    _precomputed_reward_stochasticities: _FloatArrayT
    _precomputed_values: _FloatArrayT
    _precomputed_maxes: np.ndarray[Any, np.dtype[np.int64]]

    def __init__(self,
                 arms: int,
                 value: float,
                 std_dev: float,
                 nonstationarity: Optional[tuple[float, float]] = None,
                 *,
                 steps: int = 0,
                 seed: Optional[SeedSequence] = None) -> None:
        if arms != int(arms):
            raise ValueError('Invalid `arms`')

        self.arm_values = np.array([value] * arms, dtype=float)
        self.std_dev = std_dev
        self.nonstationarity = nonstationarity
        self.rng = np.random.default_rng(seed)

        self._precomputed_reward_stochasticities = self.rng.normal(
                0, std_dev, steps)
        if nonstationarity is not None:
            nonstationarities = self.rng.normal(*nonstationarity,
                                                (steps, arms))
            self._precomputed_values = np.concatenate(
                    (np.expand_dims(self.arm_values, 0),
                     nonstationarities)).cumsum(axis=0)
            self._precomputed_maxes = self._precomputed_values.max(axis=1)
        else:
            self._precomputed_values = np.array([])
            self._precomputed_maxes = np.array([])

        self.step = -1

    def act(self, arm: int) -> tuple[float, bool]:
        self.step += 1

        value = self.arm_values[arm]
        reward = value + self.get_reward_stochasticity()
        is_optimal = value == self.get_max_value()

        self.arm_values = self.get_next_arm_values()

        return reward, is_optimal

    def get_max_value(self) -> int:
        try:
            return self._precomputed_maxes[self.step]
        except IndexError:
            return self.arm_values.max()

    def get_reward_stochasticity(self) -> float:
        try:
            return self._precomputed_reward_stochasticities[self.step]
        except IndexError:
            return self.rng.normal(0, self.std_dev)

    def get_next_arm_values(self) -> _FloatArrayT:
        try:
            return self._precomputed_values[self.step + 1]
        except IndexError:
            if self.nonstationarity is not None:
                return self.arm_values + self.rng.normal(
                        *self.nonstationarity, len(self.arm_values))
            else:
                return self.arm_values


class Model(ABC):
    arms: int
    last_action: int
    action_count: list[int]
    rng: Generator
    step: int

    def __init__(self,
                 arms: int,
                 *,
                 seed: Optional[SeedSequence] = None,
                 **kwargs: Any) -> None:
        if arms != int(arms):
            raise ValueError('Invalid `arms`')

        self.arms = arms
        self.last_action = -1
        self.action_count = [0] * arms
        self.rng = np.random.default_rng(seed)
        self.step = -1

    def act(self) -> int:
        self.step += 1
        self.last_action = self._act()
        self.action_count[self.last_action] += 1
        return self.last_action

    @abstractmethod
    def _act(self) -> int:
        """Choose an action."""

    @abstractmethod
    def reward(self, reward: float) -> None:
        """Reward the model."""


class AverageModel(Model):
    epsilon: float
    update_coefficient: Optional[float]  # alpha
    estimates: list[float]  # Q

    _precomputed_actions: np.ndarray[Any, Any]

    def __init__(self,
                 arms: int,
                 *,
                 baseline: float = 0,
                 epsilon: float = 0,
                 update_coefficient: Optional[float] = None,
                 steps: int = 0,
                 seed: Optional[SeedSequence] = None,
                 **kwargs: Any) -> None:
        super().__init__(arms, seed=seed)

        if baseline != float(baseline):
            raise ValueError('Invalid `baseline`')
        if not 0 <= epsilon < 1:
            raise ValueError('Invalid `epsilon`')
        if not (update_coefficient is None or 0 <= update_coefficient < 1):
            raise ValueError('Invalid `update_coefficient`')
        if kwargs:
            raise ValueError('Got extra keyword argument')

        self.epsilon = epsilon
        self.update_coefficient = update_coefficient
        self.estimates = [baseline] * arms

        self._precomputed_actions = np.array([None] * steps)
        explore_steps = self.rng.choice(
                2, size=steps, p=[1 - self.epsilon, self.epsilon])
        self._precomputed_actions[explore_steps != 0] = self.rng.integers(
                self.arms, size=sum(explore_steps))

        self._max_arm = 0

    def _act(self) -> int:
        action = self.get_stochastic_action()
        return action if action is not None else self._max_arm

    def get_stochastic_action(self) -> Optional[int]:
        """Return action if stochastic, otherwise None."""
        try:
            return self._precomputed_actions[self.step]
        except IndexError:
            if self.rng.random() < self.epsilon:
                return self.rng.integers(self.arms)
            else:
                return None

    def reward(self, reward: float) -> None:
        update_coefficient = (self.update_coefficient
                              if self.update_coefficient is not None
                              else 1 / self.action_count[self.last_action])
        current_estimate = self.estimates[self.last_action]
        self.estimates[self.last_action] += ((reward - current_estimate)
                                             * update_coefficient)

        if self.last_action == self._max_arm:
            if reward < current_estimate:
                self._max_arm = int(np.array(self.estimates).argmax())
        elif self.estimates[self.last_action] > self.estimates[self._max_arm]:
            self._max_arm = self.last_action


class UpperConfidenceBoundModel(Model):
    uncertainty: float  # c
    update_coefficient: Optional[float]  # alpha
    estimates: list[float]  # Q

    inv_sqrt_action_count: _FloatArrayT

    def __init__(self,
                 arms: int,
                 *,
                 uncertainty: float,
                 update_coefficient: Optional[float] = None,
                 seed: Optional[SeedSequence] = None,
                 **kwargs: Any) -> None:
        super().__init__(arms)

        if kwargs:
            raise ValueError('Got extra keyword argument')

        self.uncertainty = uncertainty
        self.update_coefficient = update_coefficient
        self.estimates = [0] * arms
        self.inv_sqrt_action_count = np.array([float('inf')] * arms)

    def _act(self) -> int:
        if self.step:
            values = (np.array(self.estimates)
                      + sqrt(log(self.step + 1)) * self.inv_sqrt_action_count)
            action = int(values.argmax())
        else:
            action = 0
        self.inv_sqrt_action_count[action] = (
                self.uncertainty / sqrt(self.action_count[action] + 1))
        return action

    def reward(self, reward: float) -> None:
        update_coefficient = (self.update_coefficient
                              if self.update_coefficient is not None
                              else 1 / self.action_count[self.last_action])
        current_estimate = self.estimates[self.last_action]
        self.estimates[self.last_action] += ((reward - current_estimate)
                                             * update_coefficient)


class StochasticGradientAscentModel(Model):
    learning_rate: float  # α
    preferences: _FloatArrayT  # H
    update_coefficient: Optional[float]
    baseline: Optional[float]  # R bar

    def __init__(self,
                 arms: int,
                 *,
                 learning_rate: float = 0,
                 update_coefficient: Optional[float] = 0,  # for baseline
                 seed: Optional[SeedSequence] = None,
                 **kwargs: Any) -> None:
        super().__init__(arms, seed=seed)

        self.learning_rate = learning_rate
        self.preferences = np.array([0.] * arms)
        self.update_coefficient = update_coefficient
        self.baseline = None

        self._indicator = np.identity(arms)

    def _act(self) -> int:
        unnormed_softmax = np.exp(self.preferences)
        self._softmax = unnormed_softmax / sum(unnormed_softmax)
        return self.rng.choice(self.arms, p=list(self._softmax))

    def reward(self, reward: float) -> None:
        if self.baseline is None:
            self.baseline = reward
        self.preferences += (
                self.learning_rate
                * (reward - self.baseline)
                * (self._indicator[self.last_action] - self._softmax))

        update_coefficient = (self.update_coefficient
                              if self.update_coefficient is not None
                              else 1 / (self.step + 1))
        self.baseline += (reward - self.baseline) * update_coefficient


ModelFactory = Callable[..., Model]


class IncrementalAverager:
    def __init__(self, length: int) -> None:
        self.value = np.zeros(length)
        self.step = 0

    def update(self, values: list[Any]) -> None:
        self.step += 1
        self.value += (np.array(values) - self.value) / self.step


@dataclass
class Testbed:
    runs: int
    steps: int
    arms: int
    value: float
    std_dev: float
    nonstationarity: Optional[tuple[float, float]] = None
    seed: Optional[int] = None

    def create_bandit(self, seed: Optional[SeedSequence] = None) -> Bandit:
        return Bandit(self.arms,
                      self.value,
                      self.std_dev,
                      self.nonstationarity,
                      steps=self.steps,
                      seed=seed)

    def run(self,
            model_factory: ModelFactory,
            runs: Optional[int] = None) -> tuple[list[float], list[float]]:
        if runs is None:
            runs = self.runs

        ss = SeedSequence(self.seed)
        bandit_ss, model_ss = ss.spawn(2)
        bandit_seeds = bandit_ss.spawn(self.runs)
        model_seeds = model_ss.spawn(self.runs)

        avg_rewards = IncrementalAverager(self.steps)
        avg_optimal_actions = IncrementalAverager(self.steps)
        for i in trange(runs):
            bandit = self.create_bandit(bandit_seeds[i])
            model = model_factory(self.arms,
                                  steps=self.steps,
                                  seed=model_seeds[i])
            rewards = []
            optimal_actions = []
            for _ in range(self.steps):
                action = model.act()
                reward, is_optimal = bandit.act(action)
                model.reward(reward)
                rewards.append(reward)
                optimal_actions.append(is_optimal)
            avg_rewards.update(rewards)
            avg_optimal_actions.update(optimal_actions)
        return list(avg_rewards.value), list(avg_optimal_actions.value)


def sample_average_experiment() -> None:
    SEED = 0

    RUNS = 2000
    STEPS = 10000
    ARMS = 10
    VALUE = 0
    STD_DEV = 1
    NONSTATIONARITY = (0, 0.01)

    EPSILON = 0.1
    UPDATE_COEFFICIENT = 0.1

    testbed = Testbed(RUNS, STEPS, ARMS, VALUE, STD_DEV, NONSTATIONARITY, SEED)

    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('Average reward')
    axes[1].set_ylabel('% Optimal action')
    axes[1].set_xlabel('Steps')

    models = {'Sample average': partial(AverageModel,
                                        epsilon=EPSILON),
              'Exponential average': partial(
                  AverageModel,
                  epsilon=EPSILON,
                  update_coefficient=UPDATE_COEFFICIENT)}
    for label, model in models.items():
        rewards, optimal_actions = testbed.run(model)

        axes[0].plot(rewards, label=label)
        axes[1].plot(optimal_actions)

    figure.legend()


@dataclass
class ParameterStudy:
    name: str
    model_factory: ModelFactory
    parameter: str
    log_values: Iterable[int]
    runs: Optional[int] = None

    def __post_init__(self) -> None:
        self.log_values = [*self.log_values]


def parameter_study() -> None:
    MAX_WORKERS = 3
    SEED = 0

    RUNS = 200
    STEPS = 200000
    MEASURE_STEPS = 100000
    ARMS = 10
    VALUE = 0
    STD_DEV = 1
    NONSTATIONARITY = (0, 0.01)

    UPDATE_COEFFICIENT = 0.1

    print(f'Running parameter study with up to {MAX_WORKERS} workers.')

    testbed = Testbed(RUNS, STEPS, ARMS, VALUE, STD_DEV, NONSTATIONARITY, SEED)

    figure, axis = plt.subplots(1, 1)

    studies = [ParameterStudy('Sample average (ε)',
                              AverageModel,
                              'epsilon',
                              range(-8, -2)),
               ParameterStudy('Exponential average (ε)',
                              partial(AverageModel,
                                      update_coefficient=UPDATE_COEFFICIENT),
                              'epsilon',
                              range(-9, -3)),
               ParameterStudy('Greedy with optimistic baseline (Q₀)',
                              partial(AverageModel,
                                      update_coefficient=UPDATE_COEFFICIENT),
                              'baseline',
                              range(-3, 4)),
               ParameterStudy('Upper Confidence Bound (c)',
                              partial(UpperConfidenceBoundModel,
                                      update_coefficient=UPDATE_COEFFICIENT),
                              'uncertainty',
                              range(3, 9)),
               ParameterStudy('Gradient bandit',
                              partial(StochasticGradientAscentModel,
                                      update_coefficient=UPDATE_COEFFICIENT),
                              'learning_rate',
                              range(-5, 1),
                              runs=20)]
    for study in studies:
        results = []
        futures = {}
        with ProcessPoolExecutor(max_workers=3) as executor:
            print(f'Running parameter study: {study.name}')
            for log_value in study.log_values:
                model = partial(study.model_factory,
                                **{study.parameter: 2 ** log_value})
                futures[log_value] = executor.submit(
                        testbed.run, model, study.runs)

            for log_value, future in futures.items():
                rewards, _ = future.result()
                results.append((log_value,
                                sum(rewards[-MEASURE_STEPS:]) / MEASURE_STEPS))

        axis.plot(*zip(*results), label=study.name)

    figure.legend()


if __name__ == '__main__':
    # sample_average_experiment()
    parameter_study()
    plt.show()
