import numpy as np


class BinaryThompsonSampling:
    def __init__(self, n_arms: int):
        self.name = "Binary Thompson Sampling"
        self.alphas = np.ones(n_arms)  # Success count + 1 (prior)
        self.betas = np.ones(n_arms)  # Failure count + 1 (prior)

    def select_arm(self):
        sampled_ps = np.random.beta(self.alphas, self.betas)
        return np.argmax(sampled_ps)

    def update(self, arm: int, reward: int):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward
