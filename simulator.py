import numpy as np


class Simulator:
    def __init__(
        self,
        n_arms: int,
        n_features: int | None = None,
        noise_std: float = 0.1,
        reward_type: str = "continuous",
        non_contextual: bool = False,
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.reward_type = reward_type
        self.non_contextual = non_contextual

        if non_contextual:
            self.arm_means = np.random.randn(n_arms)
        else:
            assert n_features is not None, (
                "n_features must be specified for contextual bandits."
            )
            self.true_thetas = np.random.randn(n_arms, n_features)

    def get_reward(self, x, arm):
        if self.non_contextual:
            mean = self.arm_means[arm]
        else:
            mean = np.dot(self.true_thetas[arm], x)
        if self.reward_type == "binary":
            prob = 1 / (1 + np.exp(-mean))
            return np.random.binomial(1, prob)
        else:
            return mean + np.random.randn() * self.noise_std
