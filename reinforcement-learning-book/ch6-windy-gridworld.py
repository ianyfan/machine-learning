"""
Windy Gridworld
===============
An implementation of the windy gridworld problem from Chapter 6 of
Reinforcement Learning: An Introduction by Sutton & Barto.

This implements a number of different Temporal Difference models.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Any, Iterable, Mapping, Optional, TypeVar, Union


_T = TypeVar('_T')
Reward = Union[float, int]


@dataclass(frozen=True)
class Action:
    x: int
    y: int


class CardinalAction(Action, Enum):
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)


class KingAction(Action, Enum):
    DOWN = (0, -1)
    DOWN_LEFT = (-1, -1)
    DOWN_RIGHT = (-1, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    UP_LEFT = (-1, 1)
    UP_RIGHT = (1, 1)


class KingAndWaitAction(Action, Enum):
    DOWN = (0, -1)
    DOWN_LEFT = (-1, -1)
    DOWN_RIGHT = (-1, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    UP_LEFT = (-1, 1)
    UP_RIGHT = (1, 1)
    WAIT = (0, 0)


@dataclass(frozen=True)
class State:
    x: int
    y: int
    is_terminal: bool = False

    def __repr__(self) -> str:
        return f'({self.x}, {self.y})'

    @classmethod
    def terminal(cls) -> State:
        return cls(-1, -1, True)


@dataclass
class WindyGridworld:
    height: int
    wind_strengths: list[int]
    starting_state: State
    finishing_state: State
    action_factory: type[Action]
    rng: Optional[Random] = None

    action_space: dict[State, list[Action]] = field(init=False)

    last_state: State = field(init=False)

    def __post_init__(self) -> None:
        self.action_space = {}
        actions = [*self.action_factory]
        for x, wind_strength in enumerate(self.wind_strengths):
            for y in range(self.height):
                state = State(x, y)
                self.action_space[state] = actions

    def in_grid(self, state: State) -> bool:
        return (0 <= state.x < len(self.wind_strengths)
                and 0 <= state.y < self.height)

    def start(self) -> State:
        self.last_state = self.starting_state
        return self.last_state

    def act(self, action: Action) -> tuple[State, Reward]:
        assert self.last_state is not None

        x = max(0, min(len(self.wind_strengths) - 1,
                       self.last_state.x + action.x))

        wind = self.wind_strengths[self.last_state.x]
        if self.rng is not None:
            wind += self.rng.choice((-1, 0, 1))
        y = max(0, min(self.height - 1, self.last_state.y + action.y + wind))

        next_state = State(x, y)
        if next_state == self.finishing_state:
            self.last_state = State.terminal()
        else:
            self.last_state = next_state
        return self.last_state, -1


class TDModel(ABC):
    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random) -> None:
        self.rng = rng
        self.discount_rate = discount_rate

        self.action_values = {State.terminal(): {Action(0, 0): 0.}}
        self.greedy_actions = {State.terminal(): Action(0, 0)}
        for state, actions in action_space.items():
            actions = [*actions]
            self.greedy_actions[state] = actions[0]
            self.action_values[state] = {action: self.init_action_value(state,
                                                                        action)
                                         for action in actions}

        self.training = True

    def init_action_value(self, state: State, action: Action) -> Reward:
        return 0

    def start(self) -> None:
        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None

    def act(self, state: State, reward: Optional[Reward] = None) -> Action:
        """Act according to target policy."""
        if self.training:
            action = self._act_train(state, reward)
        else:
            action = self._act(state)
        self.last_state = state
        self.last_action = action
        return action

    @abstractmethod
    def _act(self, state: State) -> Action:
        """Act according to target policy."""

    @abstractmethod
    def _act_train(self, state: State, reward: Optional[Reward]) -> Action:
        """Act according to behaviour policy."""


def argmax(d: Mapping[_T, Any]) -> _T:
    argmax = None
    valmax = None
    for k, v in d.items():
        if argmax is None or v > valmax:
            argmax = k
            valmax = v

    if argmax is None:
        raise ValueError('Empty mapping')
    return argmax


class Sarsa(TDModel):
    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random,
                 epsilon: float,
                 step_size: float) -> None:
        super().__init__(action_space,
                         discount_rate,
                         rng)

        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

        assert 0 <= step_size <= 1
        self.step_size = step_size

    def _act(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice([*self.action_values[state]])
        else:
            return self.greedy_actions[state]

    def _act_train(self, state: State, reward: Optional[Reward]) -> Action:
        action = self._act(state)
        if reward is not None:
            assert self.last_state is not None and self.last_action is not None
            last_value = self.action_values[self.last_state][self.last_action]
            next_step_value = self.action_values[state][action]
            self.action_values[self.last_state][self.last_action] += (
                    self.step_size * (reward
                                      + self.discount_rate * next_step_value
                                      - last_value))
            self.greedy_actions[self.last_state] = argmax(
                    self.action_values[self.last_state])
        return action


class QLearning(TDModel):
    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random,
                 epsilon: float,
                 step_size: float) -> None:
        super().__init__(action_space,
                         discount_rate,
                         rng)

        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

        assert 0 <= step_size <= 1
        self.step_size = step_size

    def _act(self, state: State) -> Action:
        return self.greedy_actions[state]

    def _act_train(self, state: State, reward: Optional[Reward]) -> Action:
        if reward is not None:
            assert self.last_state is not None and self.last_action is not None

            greedy_action = self.greedy_actions[state]
            next_step_value = self.action_values[state][greedy_action]

            last_value = self.action_values[self.last_state][self.last_action]
            self.action_values[self.last_state][self.last_action] += (
                    self.step_size * (reward
                                      + self.discount_rate * next_step_value
                                      - last_value))

            self.greedy_actions[self.last_state] = argmax(
                    self.action_values[self.last_state])

        if self.rng.random() < self.epsilon:
            return self.rng.choice([*self.action_values[state]])
        else:
            return self._act(state)


class ExpectedSarsa(TDModel):
    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random,
                 epsilon: float,
                 step_size: float) -> None:
        super().__init__(action_space,
                         discount_rate,
                         rng)

        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

        assert 0 <= step_size <= 1
        self.step_size = step_size

    def _act(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice([*self.action_values[state]])
        else:
            return self.greedy_actions[state]

    def _act_train(self, state: State, reward: Optional[Reward]) -> Action:
        if reward is not None:
            assert self.last_state is not None and self.last_action is not None

            greedy_action = self.greedy_actions[state]
            next_step_value = (
                    self.epsilon * sum(self.action_values[state].values())
                    + ((1 - self.epsilon)
                       * self.action_values[state][greedy_action]))

            last_value = self.action_values[self.last_state][self.last_action]
            self.action_values[self.last_state][self.last_action] += (
                    self.step_size * (reward
                                      + self.discount_rate * next_step_value
                                      - last_value))

            self.greedy_actions[self.last_state] = argmax(
                    self.action_values[self.last_state])

        return self._act(state)


DISCOUNT_RATE = 0.9
EPSILON = 0.1
STEP_SIZE = 0.5


for action_factory in (CardinalAction, KingAction, KingAndWaitAction):
    print(f'Using {action_factory.__name__}')
    gridworld = WindyGridworld(height=7,
                               wind_strengths=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                               starting_state=State(0, 3),
                               finishing_state=State(7, 3),
                               action_factory=action_factory,
                               rng=None)
    model = ExpectedSarsa(gridworld.action_space,
                          DISCOUNT_RATE,
                          Random(1),
                          EPSILON,
                          STEP_SIZE)
    for _ in range(10000):
        model.start()
        reward = None
        state = gridworld.start()
        while not state.is_terminal:
            action = model.act(state, reward)
            state, reward = gridworld.act(action)

    model.training = False
    total_steps = 0
    epochs = 1000
    for epoch in range(epochs):
        state = gridworld.start()
        while not state.is_terminal:
            action = model.act(state)
            state, __ = gridworld.act(action)
            total_steps += 1

    print(f'Average steps: {total_steps / epochs}')
    print()
