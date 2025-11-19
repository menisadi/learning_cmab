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
# # Epsilon-Greedy Strategy for Contextual Bandits

# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# %%
# cd /Users/meni/Code/work/cmab

# %%
from simulator import ContinousSimulator
from epsilon_greedy import EpsilonGreedy
from thompson_sampler import ThompsonSampling

# %% [markdown]
# ## Parameters

# %%
n_rounds = 50_000
n_arms = 4
n_features = 5
epsilon = 0.01


# %% [markdown]
# ## Simulation

# %%
simulator = ContinousSimulator(n_arms, n_features)
# estimator = EpsilonGreedy(epsilon, n_arms, n_features)
estimator = ThompsonSampling(n_arms, n_features, lambda_prior=1.0)

rewards_history = []
optimal_rewards = []

# %%
for t in tqdm(range(n_rounds)):
    x_t = np.random.randn(n_features)
    true_rewards = [np.dot(simulator.true_thetas[a], x_t) for a in range(n_arms)]
    optimal_rewards.append(np.max(true_rewards))

    a_t = estimator.select_arm(x_t)
    r_t = simulator.get_reward(x_t, a_t)
    rewards_history.append(r_t)

    estimator.update(a_t, x_t, r_t)


# %% [markdown]
# ## Results

# %%
# Compute cumulative reward and regret
cumulative_reward = np.cumsum(rewards_history)
regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))


# %% [markdown]
# ## Plots

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(cumulative_reward, label="Cumulative Reward")
plt.plot(np.cumsum(optimal_rewards), label="Cumulative Optimal Reward")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward vs Optimal Reward")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(regret, color="red", label="Cumulative Regret")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Over Time")
plt.legend()

plt.subplot(1, 3, 3)
normalized_regret = regret / np.arange(1, n_rounds + 1)
plt.plot(normalized_regret, color="green", label="Normalized Regret")
plt.xlabel("Rounds")
plt.ylabel("Normalized Regret")
plt.title("Normalized Regret Over Time")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Comparing Thompson Sampling and Epsilon-Greedy

# %%
# Reset simulator for fair comparison
simulator = ContinousSimulator(n_arms, n_features)
estimators = {
    "Epsilon-Greedy_explore": EpsilonGreedy(0.1, n_arms, n_features),
    "Epsilon-Greedy_sticky": EpsilonGreedy(0.01, n_arms, n_features),
    "Thompson Sampling": ThompsonSampling(n_arms, n_features, lambda_prior=1.0),
}
results = {}

# %%
for name, estimator in estimators.items():
    rewards_history = []
    optimal_rewards = []

    for t in tqdm(range(n_rounds), desc=f"Running {name}"):
        x_t = np.random.randn(n_features)
        true_rewards = [np.dot(simulator.true_thetas[a], x_t) for a in range(n_arms)]
        optimal_rewards.append(np.max(true_rewards))

        a_t = estimator.select_arm(x_t)
        r_t = simulator.get_reward(x_t, a_t)
        rewards_history.append(r_t)

        estimator.update(a_t, x_t, r_t)

    cumulative_reward = np.cumsum(rewards_history)
    regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))
    normalized_regret = regret / np.arange(1, n_rounds + 1)

    results[name] = {
        "cumulative_reward": cumulative_reward,
        "regret": regret,
        "normalized_regret": normalized_regret,
    }

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
for name, data in results.items():
    plt.plot(data["cumulative_reward"], label=name)
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")
plt.legend()
plt.subplot(1, 3, 2)
for name, data in results.items():
    plt.plot(data["regret"], label=name)
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Comparison")
plt.legend()
plt.subplot(1, 3, 3)
for name, data in results.items():
    plt.plot(data["normalized_regret"], label=name)
plt.xlabel("Rounds")
plt.ylabel("Normalized Regret")
plt.title("Normalized Regret Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# %%
