import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dim, act_dim):
        self.max_size = max_size
        self.size = 0
        self.i = 0
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros(max_size, dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.terminals = torch.zeros(max_size, dtype=torch.float32)

    def store(self, transition):
        state, action, reward, next_state, terminal = transition
        self.states[self.i] = state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.next_states[self.i] = next_state
        self.terminals[self.i] = terminal

        self.i = (self.i + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample(self, k):
        idx = np.random.choice(self.size, k)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.terminals[idx]
