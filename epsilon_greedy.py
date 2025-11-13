import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon: float, n_arms: int, n_features: int):
        self.epsilon = epsilon
        self.name = "Epsilon-Greedy"
        self.estimated_thetas = [np.zeros(n_features) for _ in range(n_arms)]
        self.arm_counts = np.zeros(n_arms)

    def select_arm(self, x):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.estimated_thetas))
        estimated_rewards = [
            np.dot(self.estimated_thetas[a], x)
            for a in range(len(self.estimated_thetas))
        ]
        return np.argmax(estimated_rewards)

    def update(self, arm, x, reward):
        # Simple online least squares update
        self.arm_counts[arm] += 1
        lr = 1 / self.arm_counts[arm]
        self.estimated_thetas[arm] += (
            lr * (reward - np.dot(self.estimated_thetas[arm], x)) * x
        )
