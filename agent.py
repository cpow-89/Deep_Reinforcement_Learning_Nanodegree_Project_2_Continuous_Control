import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import DDPGActor
from network import DDPGCritic


class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent"""

    def __init__(self, config, noise, replay_buffer):
        self.config = config
        self.device = torch.device(config["device"])
        self.seed = random.seed(config["seed"])

        # Actor
        self.actor = DDPGActor(config["n_inputs"], config["n_actions"]).to(self.device)
        self.actor_target = DDPGActor(config["n_inputs"], config["n_actions"]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["learning_rate_actor"])

        # Critic
        self.critic = DDPGCritic(config["n_inputs"], config["n_actions"]).to(self.device)
        self.critic_target = DDPGCritic(config["n_inputs"], config["n_actions"]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["learning_rate_critic"],
                                           weight_decay=config["l2_weight_decay"])

        # Noise process
        self.noise = noise

        # Replay memory
        self.memory = replay_buffer

    def save_experiences(self, state, action, reward, next_state, done):
        """Prepare and save experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return action

    def act_noisy(self, state):
        action = self.act(state)
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        """Update network using one sample of experience from memory"""
        if len(self.memory) > self.config["batch_size"]:
            states, actions, rewards, next_states, dones = self.memory.sample(self.config["batch_size"])
            self._update_critic(states, actions, rewards, next_states, dones)
            self._update_actor(states)
            self._update_targets()

    def _update_critic(self, states, actions, rewards, next_states, dones):
        loss = self._calc_critic_loss(states, actions, rewards, next_states, dones)
        self._update_weights(self.critic_optimizer, loss)

    def _calc_critic_loss(self, states, actions, rewards, next_states, dones):
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        q_targets = rewards + (self.config["gamma"] * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)
        return F.mse_loss(q_expected, q_targets)

    def _update_actor(self, states):
        loss = self._calc_actor_loss(states)
        self._update_weights(self.actor_optimizer, loss)

    def _calc_actor_loss(self, states):
        actions_pred = self.actor(states)
        return -self.critic(states, actions_pred).mean()

    def _update_targets(self):
        self._soft_update(self.critic, self.critic_target, self.config["tau"])
        self._soft_update(self.actor, self.actor_target, self.config["tau"])

    def _soft_update(self, local_network, target_network, tau):
        """Soft update target network parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _update_weights(self, optimizer, loss):
        """run one update step"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
