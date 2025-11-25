# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Epsilon-Greedy Adaptation to Parameter Shift in Contextual Bandits

# %%
import numpy as np
import matplotlib.pyplot as plt
from simulator import ContinousSimulator

# from epsilon_greedy import EpsilonGreedy
from thompson_sampler import ThompsonSampling
from lin_ucb import LinUCB
from tqdm import tqdm

# %%
n_rounds = 50_000
n_arms = 4
n_features = 5
epsilon = 0.05
shift_round = 20_000
revert_round = 22_000

# %%
simulator = ContinousSimulator(n_arms, n_features, noise_std=0.1)
original_thetas = simulator.true_thetas.copy()
shifted_thetas = np.random.randn(n_arms, n_features)  # New random thetas for the shift

# estimator = EpsilonGreedy(epsilon, n_arms, n_features)
estimator = ThompsonSampling(n_arms, n_features, discount_factor=1)
# estimator = LinUCB(n_arms, n_features, alpha=1.0)

rewards_history = []
optimal_rewards = []
theta_history = []

# %%
for t in tqdm(range(n_rounds)):
    # Simulate parameter shift
    if t == shift_round:
        simulator.true_thetas = shifted_thetas.copy()
    if t == revert_round:
        simulator.true_thetas = original_thetas.copy()

    x_t = np.random.randn(n_features)
    true_rewards = [np.dot(simulator.true_thetas[a], x_t) for a in range(n_arms)]
    optimal_rewards.append(np.max(true_rewards))

    a_t = estimator.select_arm(x_t)
    r_t = simulator.get_reward(x_t, a_t)
    rewards_history.append(r_t)

    estimator.update(a_t, x_t, r_t)
    theta_history.append([theta.copy() for theta in estimator.estimated_thetas])

# %%
cumulative_reward = np.cumsum(rewards_history)
regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))
normalized_regret = regret / np.arange(1, n_rounds + 1)

# %%
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(cumulative_reward, label="Cumulative Reward")
plt.axvline(shift_round, color="orange", linestyle="--", label="Shift")
plt.axvline(revert_round, color="green", linestyle="--", label="Revert")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.title("Cumulative Reward")

plt.subplot(1, 3, 2)
plt.plot(regret, color="red", label="Cumulative Regret")
plt.axvline(shift_round, color="orange", linestyle="--")
plt.axvline(revert_round, color="green", linestyle="--")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.title("Cumulative Regret")

plt.subplot(1, 3, 3)
plt.plot(normalized_regret, color="green", label="Normalized Regret")
plt.axvline(shift_round, color="orange", linestyle="--")
plt.axvline(revert_round, color="green", linestyle="--")
plt.xlabel("Rounds")
plt.ylabel("Normalized Regret")
plt.legend()
plt.title("Normalized Regret")
plt.tight_layout()
plt.show()

# %%
# Plot estimated thetas for each arm over time (for one feature as example)
theta_history = np.array(theta_history)  # shape: (n_rounds, n_arms, n_features)
plt.figure(figsize=(12, 6))
for arm in range(n_arms):
    plt.plot(theta_history[:, arm, 0], label=f"Arm {arm} (feature 0)")
plt.axvline(shift_round, color="orange", linestyle="--", label="Shift")
plt.axvline(revert_round, color="green", linestyle="--", label="Revert")
plt.xlabel("Rounds")
plt.ylabel("Estimated Theta (feature 0)")
plt.legend()
plt.title("Estimated Theta for Feature 0 Over Time")
plt.show()
