import numpy as np


class BinaryThompsonSampling:
    """
    Non contextual Thompson Sampling for binary rewards.
    Model per arm a:
        r_t ~ Bernoulli(p_a)
    Prior: p_a ~ Beta(alpha_prior, beta_prior)
    We maintain, for each arm a:
        alpha_a = alpha_prior + sum_{t: a_t=a} r_t
        beta_a  = beta_prior + sum_{t: a_t=a} (1 - r_t)
    Posterior:
        p_a ~ Beta(alpha_a, beta_a)
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ):
        self.name = "Binary Thompson Sampling"
        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # For each arm, maintain alpha_a and beta_a.
        # Start with alpha_a = alpha_prior, beta_a = beta_prior
        self.alpha = np.array([alpha_prior for _ in range(n_arms)])  # shape: (n_arms,)
        self.beta = np.array([beta_prior for _ in range(n_arms)])  # shape: (n_arms,)

    def select_arm(self):
        """
        Sample p_a from the posterior for each arm,
        and select the arm with the highest sampled p_a.
        """
        sampled_ps = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_ps)

    def update(self, arm: int, reward: int):
        """
        Update the posterior parameters for the selected arm.
        """
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
