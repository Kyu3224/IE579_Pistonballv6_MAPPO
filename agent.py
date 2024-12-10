import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Define your network layers here (same as in your original code)
        self.network = nn.Sequential(
            # (Batch, 4, 64, 64) -> (Batch, 32, 64, 64)
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            # (Batch, 32, 64, 64) -> (Batch, 32, 32, 32)
            nn.MaxPool2d(2),
            # For Non-linearity
            nn.ReLU(),
            # (Batch, 32, 32, 32) -> (Batch, 64, 32, 32)
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            # (Batch, 64, 32, 32) -> (Batch, 64, 16, 16)
            nn.MaxPool2d(2),
            nn.ReLU(),
            # (Batch, 64, 16, 16) -> (Batch, 128, 16, 16)
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            # (Batch, 128, 16, 16) -> (Batch, 128, 8, 8)
            nn.MaxPool2d(2),
            nn.ReLU(),
            # (Batch, 128, 16, 16) -> (Batch, 128 * 8 * 8)
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)
    obs = torch.tensor(obs).to(device)
    return obs


def batchify(x, device):
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x
