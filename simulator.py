import numpy as np


class ContinousSimulator:
    def __init__(self, n_arms, n_features, noise_std=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.true_thetas = np.random.randn(n_arms, n_features)

    def get_reward(self, x, arm):
        return np.dot(self.true_thetas[arm], x) + np.random.randn() * self.noise_std


class BinarySimulator:
    def __init__(self, n_arms, p_success):
        self.n_arms = n_arms
        self.p_success = p_success  # List of success probabilities for each arm

    def get_reward(self, arm):
        return np.random.binomial(1, self.p_success[arm])
