import numpy as np
from simulator import ContinousSimulator, BinarySimulator
from binary_tompson_sampler import BinaryThompsonSampling
from epsilon_greedy import EpsilonGreedy
from epsilon_greedy_binary import BinaryEpsilonGreedy
from thomson_sampler import ThompsonSampling


def run_continuos_simulation():
    n_rounds = 5000
    n_arms = 4
    n_features = 5
    epsilon = 0.01

    # Simulation
    simulator = ContinousSimulator(n_arms, n_features, noise_std=0.0)
    estimator = EpsilonGreedy(epsilon, n_arms, n_features)
    # estimator = ThompsonSampling(n_arms, n_features, lambda_prior=1.0, noise_std=0.1)

    rewards_history = []
    optimal_rewards = []

    for t in range(n_rounds):
        x_t = np.random.randn(n_features)
        true_rewards = [np.dot(simulator.true_thetas[a], x_t) for a in range(n_arms)]
        optimal_rewards.append(np.max(true_rewards))

        a_t = estimator.select_arm(x_t)
        r_t = simulator.get_reward(x_t, a_t)
        rewards_history.append(r_t)

        estimator.update(a_t, x_t, r_t)

    # Compute cumulative reward and regret
    cumulative_reward = np.cumsum(rewards_history)
    regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))

    print("Estimated Thetas:")
    for a in range(n_arms):
        print("Arm {}: {}".format(a, estimator.estimated_thetas[a]))

    print("\nTrue Thetas:")
    for a in range(n_arms):
        print("Arm {}: {}".format(a, simulator.true_thetas[a]))

    print(
        "\nCumulative Reward after {} rounds: {}".format(
            n_rounds, cumulative_reward[-1]
        )
    )
    print(
        "Cumulative Optimal Reward after {} rounds: {}".format(
            n_rounds, np.sum(optimal_rewards)
        )
    )
    print("Cumulative Regret after {} rounds: {}".format(n_rounds, regret[-1]))

    normalized_regret = regret / np.arange(1, n_rounds + 1)
    print(
        "Normalized Regret after {} rounds: {}".format(n_rounds, normalized_regret[-1])
    )


def run_binary_simulation():
    n_rounds = 5000
    n_arms = 4
    p_success = [0.1, 0.5, 0.7, 0.9]

    # Simulation
    simulator = BinarySimulator(n_arms, p_success)
    # estimator = BinaryThompsonSampling(n_arms)
    estimator = BinaryEpsilonGreedy(epsilon=0.05, n_arms=n_arms)

    rewards_history = []
    optimal_rewards = []

    for t in range(n_rounds):
        true_rewards = [simulator.p_success[a] for a in range(n_arms)]
        optimal_rewards.append(np.max(true_rewards))

        a_t = estimator.select_arm()
        r_t = simulator.get_reward(a_t)
        rewards_history.append(r_t)

        estimator.update(a_t, r_t)

    # Compute cumulative reward and regret
    cumulative_reward = np.cumsum(rewards_history)
    regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards_history))

    print(
        "\nCumulative Reward after {} rounds: {}".format(
            n_rounds, cumulative_reward[-1]
        )
    )
    print(
        "Cumulative Optimal Reward after {} rounds: {}".format(
            n_rounds, np.sum(optimal_rewards)
        )
    )
    print("Cumulative Regret after {} rounds: {}".format(n_rounds, regret[-1]))

    normalized_regret = regret / np.arange(1, n_rounds + 1)
    print(
        "Normalized Regret after {} rounds: {}".format(n_rounds, normalized_regret[-1])
    )


def main():
    # run_continuos_simulation()
    run_binary_simulation()


if __name__ == "__main__":
    main()
