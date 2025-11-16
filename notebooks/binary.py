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
print("\nCumulative Reward after {} rounds: {}".format(n_rounds, cumulative_reward[-1]))
print(
    "Cumulative Optimal Reward after {} rounds: {}".format(
        n_rounds, np.sum(optimal_rewards)
    )
)
print("Cumulative Regret after {} rounds: {}".format(n_rounds, regret[-1]))
normalized_regret = regret / np.arange(1, n_rounds + 1)
print("Normalized Regret after {} rounds: {}".format(n_rounds, normalized_regret[-1]))

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

# %%
