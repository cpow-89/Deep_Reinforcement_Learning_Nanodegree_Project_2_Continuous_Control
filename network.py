import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    """Deep Deterministic Policy Gradient Actor (Policy) Network"""

    def __init__(self, n_inputs, n_actions, fc_units=(256, 128), seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.network = self._create_network(n_inputs, n_actions, fc_units)

    def _create_network(self, n_inputs, n_actions, fc_units):
        return nn.Sequential(
            nn.Linear(n_inputs, fc_units[0]),
            nn.ReLU(),
            nn.Linear(fc_units[0], fc_units[1]),
            nn.ReLU(),
            nn.Linear(fc_units[1], n_actions),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)


class DDPGCritic(nn.Module):
    """Deep Deterministic Policy Gradient Critic (Value) Network"""

    def __init__(self, n_inputs, n_actions, fc_units=(256, 128), seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_head = self._create_state_head(n_inputs, fc_units)
        self.state_action_body = self._create_state_action_body(n_actions, fc_units)

    def _create_state_head(self, n_inputs, fc_units):
        return nn.Sequential(
            nn.Linear(n_inputs, fc_units[0]),
            nn.ReLU(),
        )

    def _create_state_action_body(self, n_actions, fc_units=(256, 128)):
        return nn.Sequential(
            nn.Linear(fc_units[0] + n_actions, fc_units[1]),
            nn.ReLU(),
            nn.Linear(fc_units[1], 1)
        )

    def forward(self, state, action):
        state_head_out = self.state_head(state)
        return self.state_action_body(torch.cat([state_head_out, action], dim=1))
