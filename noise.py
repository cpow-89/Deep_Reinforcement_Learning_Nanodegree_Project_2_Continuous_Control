import numpy as np
import copy
import random


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck Noise
       - generates random samples from a Gaussian (Normal) distribution
       - each sample affects the next one
       - that means that two consecutive samples are more likely to be closer together than further apart
    """

    def __init__(self, size, mu, theta, sigma, seed=0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)
        self.seed = random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
