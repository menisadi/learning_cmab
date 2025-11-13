import numpy as np


class ThompsonSampling:
    """
    Linear contextual Thompson Sampling per arm.

    Model per arm a:
        r_t = x_t^T theta_a + noise

    Prior: theta_a ~ N(0, lambda_prior^{-1} I)
    Likelihood: noise ~ N(0, noise_std^2)

    We maintain, for each arm a:
        A_a = lambda_prior * I + sum_{t: a_t=a} (1 / noise_std^2) x_t x_t^T
        b_a = sum_{t: a_t=a} (1 / noise_std^2) r_t x_t

    Posterior:
        Sigma_a = A_a^{-1}
        mu_a    = Sigma_a b_a
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        lambda_prior: float = 1.0,
        noise_std: float = 1.0,
    ):
        self.name = "Thompson Sampling"
        self.n_arms = n_arms
        self.n_features = n_features
        self.lambda_prior = lambda_prior
        self.noise_std = noise_std

        # For each arm, maintain A_a (precision-ish) and b_a.
        # Start with A_a = lambda * I, b_a = 0
        self.A = np.array(
            [lambda_prior * np.eye(n_features) for _ in range(n_arms)]
        )  # shape: (n_arms, n_features, n_features)
        self.b = np.zeros((n_arms, n_features))  # shape: (n_arms, n_features)

        # Convenience: store the posterior means (estimated thetas)
        self.estimated_thetas = [np.zeros(n_features) for _ in range(n_arms)]

    def _posterior_params(self, arm: int):
        """
        Compute posterior mean and covariance for a given arm.
        """
        A_a = self.A[arm]
        b_a = self.b[arm]

        # Sigma_a = A_a^{-1}, mu_a = Sigma_a b_a
        Sigma_a = np.linalg.inv(A_a)
        mu_a = Sigma_a @ b_a
        return mu_a, Sigma_a

    def select_arm(self, x):
        """
        Sample theta_a from the posterior for each arm,
        then pick the arm with the highest sampled reward x^T theta_a.
        """
        sampled_rewards = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            mu_a, Sigma_a = self._posterior_params(a)
            # Sample theta_a ~ N(mu_a, Sigma_a)
            theta_sample = np.random.multivariate_normal(mu_a, Sigma_a)
            sampled_rewards[a] = np.dot(theta_sample, x)

            # Keep track of posterior mean as "estimated_theta"
            self.estimated_thetas[a] = mu_a

        return int(np.argmax(sampled_rewards))

    def update(self, arm, x, reward):
        """
        Bayesian linear regression update for the chosen arm.
        """
        x = np.asarray(x)
        # Precision weight from noise variance
        beta = 1.0 / (self.noise_std**2)

        # A_a <- A_a + beta x x^T
        self.A[arm] += beta * np.outer(x, x)
        # b_a <- b_a + beta r x
        self.b[arm] += beta * reward * x

        # Optionally, we could refresh estimated_thetas[arm] here too
        # (we already refresh in select_arm, but this keeps it always up to date)
        A_a = self.A[arm]
        b_a = self.b[arm]
        mu_a = np.linalg.solve(A_a, b_a)
        self.estimated_thetas[arm] = mu_a
