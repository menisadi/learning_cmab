import numpy as np


class BinaryEpsilonGreedy:
    def __init__(self, epsilon: float, n_arms: int):
        self.epsilon = epsilon
        self.name = "Binary Epsilon-Greedy"
        self.success_counts = np.zeros(n_arms)
        self.trial_counts = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.success_counts))
        estimated_ps = np.divide(
            self.success_counts,
            self.trial_counts,
            out=np.zeros_like(self.success_counts),
            where=self.trial_counts != 0,
        )
        return np.argmax(estimated_ps)

    def update(self, arm: int, reward: int):
        self.trial_counts[arm] += 1
        self.success_counts[arm] += reward
