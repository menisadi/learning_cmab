import numpy as np


class LinUCB:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.name = "LinUCB"
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = np.array([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))
        self.estimated_thetas = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, x: np.ndarray) -> int:
        x = np.asarray(x)
        p_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            self.estimated_thetas[arm] = theta
            exploitation = np.dot(theta, x)
            exploration = self.alpha * np.sqrt(np.dot(x, A_inv @ x))
            p_values[arm] = exploitation + exploration

        return int(np.argmax(p_values))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
