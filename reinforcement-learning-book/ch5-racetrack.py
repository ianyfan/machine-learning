"""
Racetrack
=========
An implementation of the racetrack problem from Chapter 5 of
Reinforcement Learning: An Introduction by Sutton & Barto.

This implements off-policy Monte Carlo policy iteration.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Any, Optional, TypeVar, Union

_T = TypeVar('_T')
Reward = Union[float, int]


DISCOUNT_RATE = 1


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __repr__(self) -> str:
        return f'({self.x}, {self.y})'


@dataclass(frozen=True)
class State:
    position: Position
    vx: int
    vy: int

    def __repr__(self) -> str:
        return f'({self.position.x}+{self.vx}, {self.position.y}+{self.vy})'


@dataclass(frozen=True)
class Action:
    x: int
    y: int

    def __post_init__(self) -> None:
        assert self.x in (-1, 0, 1) and self.y in (-1, 0, 1)

    def __repr__(self) -> str:
        return f'+({self.x}, {self.y})'


Episode = list[tuple[State, Action, Reward]]


@dataclass(init=False)
class Racetrack:
    racetrack: list[str]
    positions: set[Position]
    starting_positions: set[Position]
    finishing_positions: set[Position]

    action_space: dict[State, list[Action]]

    trajectory: set[Position]
    last_state: Optional[State]

    noise_enabled: bool
    rng: Random

    def __init__(self, racetrack: str, rng: Random) -> None:
        self.rng = rng
        self.noise_enabled = True
        self.last_state = None
        self.trajectory = set()

        lines = racetrack.splitlines()

        height = len(lines)
        for line in reversed(lines):
            if not line or line.isspace():
                height -= 1
            else:
                break

        del lines[height:]
        indent = len(lines[0])
        for line in lines:
            line_indent = len(line) - len(line.lstrip())
            indent = min(indent, line_indent)

        self.positions = set()
        self.starting_positions = set()
        self.finishing_positions = set()
        self.racetrack = []
        for y, line in enumerate(reversed(lines)):
            line = line[indent:].rstrip()
            if line:
                self.racetrack.append(line)
                for x, char in enumerate(line):
                    position = Position(x, y)
                    if not char.isspace():
                        self.positions.add(position)
                        if char == 's':
                            self.starting_positions.add(position)
                        elif char == 'f':
                            self.finishing_positions.add(position)
                        else:
                            assert char == '.'

        allowed_velocities = [(vx, vy)
                              for vx in range(5)
                              for vy in range(5)
                              if not vx == vy == 0]

        self.action_space = {}
        for vx in range(5):
            for vy in range(5):
                actions = [Action(acc_x, acc_y)
                           for acc_x in (-1, 0, 1)
                           for acc_y in (-1, 0, 1)
                           if (vx + acc_x, vy + acc_y) in allowed_velocities]
                for position in self.positions:
                    if (position not in self.finishing_positions
                            and (not vx == vy == 0
                                 or position in self.starting_positions)):
                        state = State(position, vx, vy)
                        self.action_space[state] = actions

    def print_trajectory(self, **kwargs: Any) -> None:
        output = ''
        for y, line in enumerate(self.racetrack):
            new_line = ''
            for x, ch in enumerate(line):
                p = Position(x, y)
                new_line += '•' if p in self.trajectory else ch
            output = f'{new_line}\n{output}'
        print(output, **kwargs)

    def restart(self) -> State:
        start = self.rng.choice([*self.starting_positions])
        self.trajectory.add(start)
        self.last_state = State(start, 0, 0)
        return self.last_state

    def reset(self) -> State:
        self.trajectory = set()
        return self.restart()

    def act(self, action: Action) -> tuple[Optional[State], Reward]:
        assert self.last_state is not None

        vx = self.last_state.vx + action.x
        vy = self.last_state.vy + action.y
        assert 0 <= vx < 5 and 0 <= vy < 5 and not vx == vy == 0

        if self.noise_enabled and self.rng.random() < 0.1:
            vx = self.last_state.vx
            vy = self.last_state.vy
        old_position = self.last_state.position
        old_x = old_position.x
        old_y = old_position.y

        new_x = old_position.x + vx
        new_y = old_position.y + vy

        if not vx == vy == 0:
            t = max(vx, vy)
            for i in range(1, t + 1):
                x = old_x + vx / t * i
                y = old_y + vy / t * i
                position = Position(round(x), round(y))
                if position in self.finishing_positions:
                    self.trajectory.add(position)
                    self.last_state = None
                    return None, -1
                elif position not in self.positions:
                    return self.restart(), -1

        new_position = Position(new_x, new_y)
        self.trajectory.add(new_position)
        self.last_state = State(new_position, vx, vy)
        return self.last_state, -1


class MonteCarloModel(ABC):
    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random) -> None:
        self.rng = rng
        self.discount_rate = discount_rate

        self.greedy_actions = {}
        self.action_values = {}
        self.weights = {}
        for state, actions in action_space.items():
            actions = [*actions]
            self.greedy_actions[state] = actions[0]
            self.action_values[state] = {action: self.init_action_value(state,
                                                                        action)
                                         for action in actions}
            self.weights[state] = {action: 0. for action in actions}

    def init_action_value(self, state: State, action: Action) -> Reward:
        return 0

    @abstractmethod
    def act(self, state: State) -> Action:
        """Act according to target policy."""

    def act_train(self, state: State) -> Action:
        """Act according to behaviour policy."""
        return self.act(state)

    @abstractmethod
    def update(self, episode: Episode) -> None:
        """Update policy."""


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


class OnPolicyModel(MonteCarloModel):
    # Epsilon-greedy Monte Carlo model
    # A lower epsilon works better here

    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random,
                 epsilon: float) -> None:
        super().__init__(action_space, discount_rate, rng)

        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def act(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice([*self.action_values[state]])
        else:
            return self.greedy_actions[state]

    def update(self, episode: Episode) -> None:
        first_visits = {}
        for t, (state, action, _) in enumerate(episode):
            if (state, action) not in first_visits:
                first_visits[state, action] = t
        first_visit_steps = set(first_visits.values())

        discounted_return = 0.
        for t, (state, action, reward) in reversed([*enumerate(episode)]):
            discounted_return = self.discount_rate * discounted_return + reward
            if t in first_visit_steps:
                self.weights[state][action] += 1
                self.action_values[state][action] += (
                        (discounted_return - self.action_values[state][action])
                        / self.weights[state][action])

                self.greedy_actions[state] = argmax(self.action_values[state])


class OffPolicyModel(MonteCarloModel):
    # Also epsilon-greedy Monte Carlo model
    # A higher epsilon works better here

    def __init__(self,
                 action_space: Mapping[State, Iterable[Action]],
                 discount_rate: float,
                 rng: Random,
                 epsilon: float) -> None:
        super().__init__(action_space, discount_rate, rng)

        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def init_action_value(self, state: State, action: Action) -> Reward:
        # stops car from having bad greedy actions
        return -float('inf')

    def act(self, state: State) -> Action:
        return self.greedy_actions[state]

    def act_train(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice([*self.action_values[state]])
        else:
            return self.act(state)

    def update(self, episode: Episode) -> None:
        discounted_return = 0.
        weight = 1.
        for t, (state, action, reward) in reversed([*enumerate(episode)]):
            discounted_return = self.discount_rate * discounted_return + reward

            if self.weights[state][action]:
                self.weights[state][action] += weight
                self.action_values[state][action] += (
                        weight
                        * (discounted_return
                           - self.action_values[state][action])
                        / self.weights[state][action])
            else:
                # first time update, when value is -inf
                self.weights[state][action] = weight
                self.action_values[state][action] = discounted_return

            self.greedy_actions[state] = policy_action = argmax(
                    self.action_values[state])
            if action != policy_action:
                break

            actions_count = len(self.action_values[state])
            weight /= (1 - self.epsilon * (1 - 1 / actions_count))


def run_episode(racetrack: Racetrack,
                model: MonteCarloModel,
                train: bool) -> Episode:
    racetrack.noise_enabled = train
    episode = []
    state: Optional[State] = racetrack.reset()
    while state is not None:
        action = model.act_train(state) if train else model.act(state)
        next_state, reward = racetrack.act(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


racetrack1 = Racetrack('''
   .............f
  ..............f
  ..............f
 ...............f
................f
................f
..........
.........
.........
.........
.........
.........
.........
.........
 ........
 ........
 ........
 ........
 ........
 ........
 ........
 ........
  .......
  .......
  .......
  .......
  .......
  .......
  .......
   ......
   ......
   ssssss
''', Random(0))

racetrack2 = Racetrack('''
                ...............f
             ..................f
            ...................f
           ....................f
           ....................f
           ....................f
           ....................f
            ...................f
             ..................f
              ................
              .............
              ............
              ..........
              .........
             ..........
            ...........
           ............
          .............
         ..............
        ...............
       ................
      .................
     ..................
    ...................
   ....................
  .....................
 ......................
.......................
.......................
sssssssssssssssssssssss
''', Random(0))

racetrack = racetrack2

model = OffPolicyModel(racetrack.action_space, DISCOUNT_RATE, Random(1), 0.25)
for epoch in range(1, 1000001):
    episode = run_episode(racetrack, model, True)
    model.update(episode)

    if epoch % 1000 == 0:
        len_episodes = []
        for i in range(1000):
            episode = run_episode(racetrack, model, False)
            len_episodes.append(len(episode))
        avg_len_episodes = sum(len_episodes) / len(len_episodes)
        print(f'Epoch {epoch}: average {avg_len_episodes} steps')
