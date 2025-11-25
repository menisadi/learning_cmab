import numpy as np


class UCB1:
    def __init__(self, n_arms: int, confidence: float = 2.0):
        self.name = "UCB1"
        self.n_arms = n_arms
        self.confidence = confidence
        self.counts = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

    def select_arm(self) -> int:
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm

        total_counts = np.sum(self.counts)
        average_rewards = self.rewards / self.counts
        exploration_bonus = np.sqrt(
            (self.confidence * np.log(total_counts)) / self.counts
        )
        ucb_scores = average_rewards + exploration_bonus
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.rewards[arm] += reward
