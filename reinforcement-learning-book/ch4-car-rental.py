"""
Jack's Car Rental
=================
An implementation of the car rental problem from Chapter 4 of
Reinforcement Learning: An Introduction by Sutton & Barto.

This implements tabular policy iteration using dynamic programming.

"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from math import exp
from typing import NamedTuple, NewType

# Environment parameters
MAX_CARS_AT_LOC = 20
MAX_CARS_MOVED = 5
MOVE_CAR_PRICE = 2
RENTAL_PRICE = 10
REQUESTS_AT_FIRST_LOC_MEAN = 3
REQUESTS_AT_SECOND_LOC_MEAN = 4
RETURNS_AT_FIRST_LOC_MEAN = 3
RETURNS_AT_SECOND_LOC_MEAN = 2
SECOND_LOT_PRICE = 4
SECOND_LOT_THRESHOLD = 10
FREE_SHUTTLE = True

# Model hyperparameters
ACCURACY_OF_POLICY_ESTIMATION = 0.01
DISCOUNT_RATE = 0.9


Action = NewType('Action', int)


def cost(action: Action) -> int:
    if FREE_SHUTTLE:
        return MOVE_CAR_PRICE * (action - 1 if action > 0 else -action)
    else:
        return MOVE_CAR_PRICE * abs(action)


@cache
def factorial(n: int) -> int:
    return 1 if n <= 0 else n * factorial(n - 1)


@cache
def poisson(mean: float, n: int, and_tail: bool = False) -> float:
    if and_tail:
        return 1 - sum(poisson(mean, i) for i in range(n))
    else:
        return mean ** n / factorial(n) * exp(-mean)


class Transition(NamedTuple):
    next_state: State
    reward: float
    probability: float


class State(NamedTuple):
    cars_at_first_loc: int
    cars_at_second_loc: int

    @property
    def actions(self) -> list[Action]:
        max_cars_moved_from_first_loc = min(MAX_CARS_MOVED,
                                            self.cars_at_first_loc)
        max_cars_moved_from_second_loc = min(MAX_CARS_MOVED,
                                             self.cars_at_second_loc)
        return [*range(-max_cars_moved_from_second_loc,
                       max_cars_moved_from_first_loc + 1)]

    @classmethod
    @cache
    def _rentals(cls,
                 cars_1: int,
                 cars_2: int) -> list[Transition]:
        # _1 means at first location, _2 means at second location
        transitions = {}
        for requests_1 in range(cars_1 + 1):
            requests_1_reward = RENTAL_PRICE * requests_1
            requests_1_prob = poisson(REQUESTS_AT_FIRST_LOC_MEAN,
                                      requests_1,
                                      requests_1 == cars_1)
            remaining_1 = cars_1 - requests_1
            for requests_2 in range(cars_2 + 1):
                requests_2_reward = RENTAL_PRICE * requests_2
                requests_2_prob = poisson(REQUESTS_AT_SECOND_LOC_MEAN,
                                          requests_2,
                                          requests_2 == cars_2)
                remaining_2 = cars_2 - requests_2

                for next_cars_1 in range(remaining_1, MAX_CARS_AT_LOC + 1):
                    returns_1 = next_cars_1 - remaining_1
                    returns_1_prob = poisson(RETURNS_AT_FIRST_LOC_MEAN,
                                             returns_1,
                                             next_cars_1 == MAX_CARS_AT_LOC)

                    for next_cars_2 in range(remaining_2, MAX_CARS_AT_LOC + 1):
                        returns_2 = next_cars_2 - remaining_2
                        returns_2_prob = poisson(
                                RETURNS_AT_SECOND_LOC_MEAN,
                                returns_2,
                                next_cars_2 == MAX_CARS_AT_LOC)

                        reward = requests_1_reward + requests_2_reward
                        probability = (requests_1_prob
                                       * requests_2_prob
                                       * returns_1_prob
                                       * returns_2_prob)
                        t = (next_cars_1, next_cars_2, reward)
                        try:
                            transitions[t] += probability
                        except KeyError:
                            transitions[t] = probability
        return [(State(next_cars_1, next_cars_2), reward, probability)
                for (next_cars_1, next_cars_2, reward), probability
                in transitions.items()]

    def transitions(self, action: Action) -> list[Transition]:
        if action not in self.actions:
            return []

        cars_1 = min(MAX_CARS_AT_LOC, self.cars_at_first_loc - action)
        cars_2 = min(MAX_CARS_AT_LOC, self.cars_at_second_loc + action)

        move_car_reward = -(
                cost(action)
                + SECOND_LOT_PRICE * (int(cars_1 > SECOND_LOT_THRESHOLD)
                                      + int(cars_2 > SECOND_LOT_THRESHOLD)))

        return [(next_state, reward + move_car_reward, probability)
                for next_state, reward, probability
                in self._rentals(cars_1, cars_2)]


ALL_STATES = [State(x, y)
              for x in range(MAX_CARS_AT_LOC + 1)
              for y in range(MAX_CARS_AT_LOC + 1)]


def print_policy(iteration: int, policy: Mapping[State, float]) -> None:
    output = 'π' + chr(0x2080 + iteration)
    for cars_1 in reversed(range(MAX_CARS_AT_LOC + 1)):
        output += '\n'
        for cars_2 in range(MAX_CARS_AT_LOC + 1):
            output += f'{policy[State(cars_1, cars_2)]:3}'
    print(output)


def main():
    values = {state: 0 for state in ALL_STATES}
    policy = {state: 0 for state in ALL_STATES}

    iteration = 0
    print_policy(iteration, policy)

    policy_stable = False
    while not policy_stable:
        iteration += 1

        # policy evaluation
        change = ACCURACY_OF_POLICY_ESTIMATION + 1
        while change > ACCURACY_OF_POLICY_ESTIMATION:
            change = 0
            for state, action in policy.items():
                old_value = values[state]
                values[state] = sum(probability
                                    * (reward
                                       + DISCOUNT_RATE * values[next_state])
                                    for next_state, reward, probability
                                    in state.transitions(action))
                change = max(change, abs(values[state] - old_value))

        # policy improvement
        policy_stable = True
        for state, old_action in policy.items():
            new_action = None
            max_value = -float('inf')
            for action in state.actions:
                value = sum(probability
                            * (reward
                               + DISCOUNT_RATE * values[next_state])
                            for next_state, reward, probability
                            in state.transitions(action))
                if value > max_value:
                    new_action = action
                    max_value = value

            policy[state] = new_action
            if new_action != old_action:
                policy_stable = False

        print_policy(iteration, policy)


if __name__ == '__main__':
    main()
