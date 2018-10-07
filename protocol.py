from tensorboardX import SummaryWriter
import os
import numpy as np
import utils


class Protocol:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(os.path.join(".", *self.config["monitor_dir"],
                                                 self.config["env_name"], utils.get_current_date_time()))
        self.episode_reward = 0
        self.episode_rewards = []
        self.step = 0
        self.actor_loss = None
        self.critic_loss = None

    def step_update(self, reward, actor_loss, critic_loss):
        self.step += 1
        self.episode_reward += reward
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss

    def episode_update(self):
        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0

    def write_statistics(self):
        self.writer.add_scalar("reward_100", np.mean(self.episode_rewards[-100:]), self.step)
        if self.actor_loss and self.critic_loss:
            self.writer.add_scalar("actor_loss", self.actor_loss, self.step)
            self.writer.add_scalar("critic_loss", self.critic_loss, self.step)

    def __str__(self):
        episode_info = "Episode: {}".format(len(self.episode_rewards))
        step_info = "Step: {}".format(self.step)
        mean_reward_info = "Mean reward over the last 100 steps: {}".format(np.mean(self.episode_rewards[-100:]))
        return " - ".join([episode_info, step_info, mean_reward_info])
