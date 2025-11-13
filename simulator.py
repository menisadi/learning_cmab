import numpy as np


class Simulator:
    def __init__(self, n_arms, n_features, noise_std=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.true_thetas = np.random.randn(n_arms, n_features)

    def get_reward(self, x, arm):
        return np.dot(self.true_thetas[arm], x) + np.random.randn() * self.noise_std
