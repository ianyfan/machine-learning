"""
Mountain Car
============
An implementation of the mountain car problem from Chapter 10 of
Reinforcement Learning: An Introduction by Sutton & Barto.

This implements semi-gradient Sarsa with approximation.

"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from math import cos
from random import Random
from typing import Any, ClassVar, Iterator, NewType, Optional

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


class IncrementalAverager:
    def __init__(self, length: int) -> None:
        self.value = [0.] * length
        self.step = 0

    def update(self, values: list[float]) -> None:
        self.step += 1
        self.value = [old + (new - old) / self.step
                      for old, new in zip(self.value, values)]


@dataclass
class State:
    position: float
    velocity: float

    terminal: ClassVar['State']

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[float]:
        yield self.position
        yield self.velocity


State.terminal = State(float('nan'), float('nan'))
Action = NewType('Action', int)
Reward = NewType('Reward', float)
Feature = NewType('Feature', np.ndarray[Any, np.dtype[np.float64]])
FeatureSpace = Mapping[tuple[State, Action], Feature]


@dataclass
class MountainCarEnvironment:
    actions: ClassVar[list[Action]] = [Action(-1), Action(0), Action(1)]
    min_position: ClassVar[float] = -1.2
    max_position: ClassVar[float] = 0.5
    min_velocity: ClassVar[float] = -0.07
    max_velocity: ClassVar[float] = 0.07

    seed: InitVar[int]
    rng: Random = field(init=False)

    last_state: State = field(init=False)

    def __post_init__(self, seed: int) -> None:
        self.rng = Random(seed)

    def reset(self) -> State:
        self.last_state = State(self.rng.uniform(-0.6, -0.4), 0)
        return self.last_state

    def act(self, action: Action) -> tuple[Reward, State]:
        position = self.last_state.position
        velocity = self.last_state.velocity

        next_velocity = clamp(velocity + 0.001*action - 0.0025*cos(3*position),
                              self.min_velocity,
                              self.max_velocity)
        next_position = clamp(position + next_velocity,
                              self.min_position,
                              self.max_position)

        if next_position == self.max_position:
            self.last_state = State.terminal
            reward = 0
        else:
            if next_position == self.min_position:
                next_velocity = 0
            self.last_state = State(next_position, next_velocity)
            reward = -1
        return Reward(reward), self.last_state


class TileCoding(FeatureSpace):
    def __init__(self,
                 actions: int,
                 grid_size: int,
                 *dimensions: tuple[float, float],
                 tilings: Optional[int] = None,
                 displacement: Optional[Iterable[int]] = None) -> None:
        if tilings is None:
            tilings = 4
            while tilings < 4 * len(dimensions):
                tilings *= 2

        if displacement is None:
            displacement = [2*d + 1 for d in range(len(dimensions))]
        displacement = [d / tilings for d in displacement]

        self.grids = [(low, grid_size / (high - low), d)
                      for (low, high), d in zip(dimensions, displacement)]

        self.size = (actions
                     * grid_size ** 2
                     * max(0, len(dimensions) - 1) * (grid_size + 1) ** 2)

        self.tile_mappings: list[dict[tuple[tuple[int, ...], Action], int]]
        self.tile_mappings = [{} for _ in range(tilings)]
        self.counter = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, key: tuple[State, Action]) -> Feature:
        vector = Feature(np.zeros(self.size))
        vector[self.get_tiles(key)] = 1
        return vector

    def get_tiles(self, key: tuple[State, Action]) -> list[int]:
        state, action = key

        indices = []
        for i, tile_mapping in enumerate(self.tile_mappings):
            tile = tuple(int(multiplier * (value - low) - i * displacement)
                         for value, (low, multiplier, displacement)
                         in zip(state, self.grids))
            try:
                index = tile_mapping[tile, action]
            except KeyError:
                if self.counter == self.size:
                    raise IndexError('Exceeded tile coding size.')
                index = tile_mapping[tile, action] = self.counter
                self.counter += 1
            indices.append(index)
        return indices


@dataclass
class Sarsa:
    environment: MountainCarEnvironment
    features: FeatureSpace

    step_size: float
    discount_rate: float = 1
    epsilon: float = 0
    rng: Optional[Random] = None

    weights: Feature = field(init=False)
    last_state: State = field(init=False)
    last_action: Action = field(init=False)
    last_value: float = field(init=False)
    last_gradient: Feature = field(init=False)

    def __post_init__(self) -> None:
        self.weights = Feature(np.zeros(len(self.features)))

    def value_and_gradient(
            self, state: State, action: Action) -> tuple[float, Feature]:
        feature = self.features[state, action]
        return np.dot(self.weights, feature), feature

    def act(self, state: State) -> Action:
        if self.epsilon > 0:
            assert self.rng is not None
            raise NotImplementedError
        else:
            action, value, gradient = self.greedy_action(state)

        self.last_state = state
        self.last_action = action
        self.last_value = value
        self.last_gradient = gradient
        return action

    def greedy_action(self, state: State) -> tuple[Action, float, Feature]:
        max_action = None
        max_value = None
        max_gradient = None
        for action in self.environment.actions:
            value, gradient = self.value_and_gradient(state, action)
            if max_value is None or value > max_value:
                max_action = action
                max_value = value
                max_gradient = gradient
        assert max_action is not None
        return max_action, max_value, max_gradient

    def reward(self, reward: Reward, next_state: State) -> Optional[Action]:
        value = self.last_value
        gradient = self.last_gradient
        if next_state == State.terminal:
            multiplier = self.step_size * (reward - value)
            next_action = None
        else:
            next_action = self.act(next_state)
            next_value = self.last_value  # due to caching
            multiplier = (self.step_size *
                          (reward + self.discount_rate * next_value - value))

        self.weights = Feature(self.weights + multiplier * gradient)
        return next_action


def main() -> None:
    GRID_SIZE = 8
    STEP_SIZE = 0.5 / GRID_SIZE
    EPSILON = 0
    DISCOUNT_RATE = 1

    RUNS = 100
    EPISODES = 500

    seeds = [*range(RUNS)]
    avg_episode_lengths = IncrementalAverager(EPISODES)
    for seed in seeds:
        environment = MountainCarEnvironment(seed)
        features = TileCoding(len(environment.actions),
                              GRID_SIZE,
                              (environment.min_position,
                               environment.max_position),
                              (environment.min_velocity,
                               environment.max_velocity))
        model = Sarsa(environment, features, STEP_SIZE, DISCOUNT_RATE, EPSILON)
        episode_lengths = []
        for episode in range(EPISODES):
            step = 0.
            state = environment.reset()
            action: Optional[Action] = model.act(state)
            while state != State.terminal:
                assert action is not None
                step += 1
                reward, next_state = environment.act(action)
                next_action = model.reward(reward, next_state)
                state, action = next_state, next_action
            episode_lengths.append(step)
        avg_episode_lengths.update(episode_lengths)

    # from matplotlib import pyplot as plt
    # plt.plot(avg_episode_lengths.value)


if __name__ == '__main__':
    main()
