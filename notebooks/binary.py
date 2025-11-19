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
# # Binary Epsilon-Greedy Strategy for Multi-Armed Bandits
# This notebook demonstrates the implementation and simulation of a Binary Epsilon-Greedy strategy for multi
# -armed bandit problems.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# %%
# cd /Users/meni/Code/personal/learning_cmab

# %%
from simulator import BinarySimulator
from epsilon_greedy_binary import BinaryEpsilonGreedy
from binary_thompson_sampler import BinaryThompsonSampling

# %% [markdown]
# ## Parameters

# %%
n_rounds = 5000
n_arms = 4
p_success = [0.1, 0.5, 0.7, 0.9]  # Success probabilities for each arm
# %% [markdown]
# ## Simulation
# %%
simulator = BinarySimulator(n_arms, p_success)
estimator = BinaryEpsilonGreedy(epsilon=0.05, n_arms=n_arms)
rewards_history = []
optimal_rewards = []
for t in tqdm(range(n_rounds)):
    true_rewards = [simulator.p_success[a] for a in range(n_arms)]
    optimal_rewards.append(np.max(true_rewards))

    a_t = estimator.select_arm()
    r_t = simulator.get_reward(a_t)
    rewards_history.append(r_t)

    estimator.update(a_t, r_t)

# Compute cumulative reward and regret
cumulative_reward = np.cumsum(rewards_history)
regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))

print(f"Cumulative Reward after {n_rounds} rounds: {cumulative_reward[-1]:.2f}")
print(
    f"Cumulative Optimal Reward after {n_rounds} rounds: {np.sum(optimal_rewards):.2f}"
)
print(f"Cumulative Regret after {n_rounds} rounds: {regret[-1]:.2f}")

normalized_regret = regret / np.arange(1, n_rounds + 1)
print(f"Normalized Regret after {n_rounds} rounds: {normalized_regret[-1]:.4f}")

# %% [markdown]
# ## Results Visualization

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cumulative_reward, label="Cumulative Reward")
plt.plot(np.cumsum(optimal_rewards), label="Cumulative Optimal Reward")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward vs Optimal Reward")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(regret, label="Cumulative Regret", color="red")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Over Time")
plt.legend()
plt.tight_layout()

# %%
plt.figure(figsize=(6, 4))
normalized_regret = regret / np.arange(1, n_rounds + 1)
plt.plot(normalized_regret, label="Normalized Regret", color="green")
plt.xlabel("Rounds")
plt.ylabel("Normalized Regret")
plt.title("Normalized Regret Over Time")
plt.legend()
plt.tight_layout()


# %% [markdown]
# Comparing the performance of Binary Epsilon-Greedy with Thompson Sampling


# Run both simulations
def run_simulation(estimator, n_rounds, n_arms, p_success):
    simulator = BinarySimulator(n_arms, p_success)
    rewards_history = []
    optimal_rewards = []
    for t in range(n_rounds):
        true_rewards = [simulator.p_success[a] for a in range(n_arms)]
        optimal_rewards.append(np.max(true_rewards))

        a_t = estimator.select_arm()
        r_t = simulator.get_reward(a_t)
        rewards_history.append(r_t)

        estimator.update(a_t, r_t)

    cumulative_reward = np.cumsum(rewards_history)
    regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))
    normalized_regret = regret / np.arange(1, n_rounds + 1)
    return cumulative_reward, normalized_regret


# %%
n_rounds = 5000
n_arms = 4
p_sucess = np.random.rand(n_arms)

# %%
thompson_estimator = BinaryThompsonSampling(n_arms)
epsilon_greedy_estimator = BinaryEpsilonGreedy(epsilon=0.05, n_arms=n_arms)

thompson_cum_reward, thompson_norm_regret = run_simulation(
    thompson_estimator, n_rounds, n_arms, p_sucess
)
epsilon_cum_reward, epsilon_norm_regret = run_simulation(
    epsilon_greedy_estimator, n_rounds, n_arms, p_sucess
)

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(thompson_cum_reward, label="Thompson Sampling")
plt.plot(epsilon_cum_reward, label="Epsilon-Greedy")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(thompson_norm_regret, label="Thompson Sampling")
plt.plot(epsilon_norm_regret, label="Epsilon-Greedy")
plt.xlabel("Rounds")
plt.ylabel("Normalized Regret")
plt.title("Normalized Regret Comparison")
plt.legend()
plt.tight_layout()

plt.show()
