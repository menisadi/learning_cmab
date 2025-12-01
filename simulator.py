from dataclasses import dataclass
from typing import List, Sequence, Tuple

from tqdm import tqdm
import numpy as np


class ContinousSimulator:
    def __init__(self, n_arms, n_features, noise_std=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.true_thetas = np.random.randn(n_arms, n_features)

    def get_reward(self, x, arm):
        return np.dot(self.true_thetas[arm], x) + np.random.randn() * self.noise_std


class BinarySimulator:
    def __init__(self, n_arms, p_success):
        self.n_arms = n_arms
        self.p_success = p_success  # List of success probabilities for each arm

    def get_reward(self, arm):
        return np.random.binomial(1, self.p_success[arm])


@dataclass
class BidContext:
    publisher: str
    placement: str
    device: str
    hour_bucket: int
    user_segment: str


@dataclass
class BidLog:
    context: BidContext
    features: np.ndarray
    arm: int
    floor: float
    reward: float
    bids: np.ndarray


class BidFlooringSimulator:
    """
    Simulates an online bid-flooring setting where arms are possible floor prices.

    - Context contains publisher, placement, device, hour, and user segment.
    - Reward is the winning bid if it clears the floor, otherwise 0.
    - Mini-batch updates: arm selection uses the current model snapshot for a batch
      of events, then the model is updated with the accumulated logs.
    """

    def __init__(
        self,
        floor_options: Sequence[float],
        publishers: Sequence[str] | None = None,
        placements: Sequence[str] | None = None,
        devices: Sequence[str] | None = None,
        user_segments: Sequence[str] | None = None,
        hours: Sequence[int] | None = None,
        base_bid: float = 1.5,
        bid_sigma: float = 0.7,
        bidder_range: Tuple[int, int] = (2, 8),
        seed: int | None = None,
    ):
        self.floor_options = np.asarray(floor_options, dtype=float)
        self.publishers = list(publishers or ["news", "sports", "finance"])
        self.placements = list(placements or ["hero", "sidebar", "feed"])
        self.devices = list(devices or ["mobile", "desktop", "tablet"])
        self.user_segments = list(user_segments or ["anon", "casual", "loyal"])
        self.hours = list(hours or range(24))
        self.base_bid = base_bid
        self.bid_sigma = bid_sigma
        self.bidder_range = bidder_range
        self.rng = np.random.default_rng(seed)

        self.publisher_effects = self.rng.normal(0.0, 0.3, len(self.publishers))
        self.placement_effects = self.rng.normal(0.0, 0.2, len(self.placements))
        self.device_effects = self.rng.normal(0.0, 0.1, len(self.devices))
        self.segment_effects = self.rng.normal(0.0, 0.25, len(self.user_segments))

        # Smooth diurnal curve for hours of day.
        self.hour_effects = 0.2 * np.sin(
            np.linspace(0, 2 * np.pi, num=len(self.hours), endpoint=False)
        )

        # Track indices for one-hot encoding.
        self._publisher_index = {p: i for i, p in enumerate(self.publishers)}
        self._placement_index = {p: i for i, p in enumerate(self.placements)}
        self._device_index = {d: i for i, d in enumerate(self.devices)}
        self._segment_index = {s: i for i, s in enumerate(self.user_segments)}

        # Number of features exposed to models (one-hot + 2 time encodings).
        self.n_features = (
            len(self.publishers)
            + len(self.placements)
            + len(self.devices)
            + len(self.user_segments)
            + 2
        )

    def sample_context(self) -> BidContext:
        return BidContext(
            publisher=self.rng.choice(self.publishers),
            placement=self.rng.choice(self.placements),
            device=self.rng.choice(self.devices),
            hour_bucket=int(self.rng.choice(self.hours)),
            user_segment=self.rng.choice(self.user_segments),
        )

    def encode_context(self, context: BidContext) -> np.ndarray:
        """
        One-hot encode categorical features and add cyclical hour encoding.
        """
        features: List[float] = []
        features.extend(self._one_hot(context.publisher, self.publishers))
        features.extend(self._one_hot(context.placement, self.placements))
        features.extend(self._one_hot(context.device, self.devices))
        features.extend(self._one_hot(context.user_segment, self.user_segments))

        # Cyclical time-of-day encoding to preserve proximity between 23->0.
        hour_fraction = context.hour_bucket / max(1, len(self.hours))
        features.append(np.sin(2 * np.pi * hour_fraction))
        features.append(np.cos(2 * np.pi * hour_fraction))

        return np.asarray(features, dtype=float)

    def simulate_bids(self, context: BidContext) -> np.ndarray:
        """
        Draw a set of bids conditioned on the context.
        """
        n_bidders = int(
            self.rng.integers(self.bidder_range[0], self.bidder_range[1] + 1)
        )
        base = self._latent_value(context)
        mean_bid = max(0.05, self.base_bid * max(0.25, 1.0 + base))
        return self.rng.lognormal(
            mean=np.log(mean_bid), sigma=self.bid_sigma, size=n_bidders
        )

    def calculate_reward(self, floor: float, bids: np.ndarray) -> float:
        """
        Revenue is the winning bid when it clears the floor, otherwise 0.
        """
        cleared = bids[bids >= floor]
        if cleared.size == 0:
            return 0.0
        return float(np.max(cleared))

    def run_mini_batch(
        self, model, n_batches: int, batch_size: int, show_progress: bool = True
    ) -> Tuple[List[BidLog], np.ndarray]:
        """
        Run an online simulation with delayed (mini-batch) model updates.

        Returns:
            logs: flattened list of BidLog entries.
            batch_rewards: total revenue per processed batch.
        """
        logs: List[BidLog] = []
        batch_rewards = np.zeros(n_batches)

        for batch_idx in tqdm(range(n_batches), disable=not show_progress):
            batch_logs: List[BidLog] = []

            # Decision policy uses the snapshot of model parameters for this batch.
            for _ in tqdm(range(batch_size), disable=not show_progress, leave=False):
                context = self.sample_context()
                features = self.encode_context(context)
                arm = model.select_arm(features)
                floor = float(self.floor_options[arm])
                bids = self.simulate_bids(context)
                reward = self.calculate_reward(floor, bids)
                batch_logs.append(
                    BidLog(
                        context=context,
                        features=features,
                        arm=arm,
                        floor=floor,
                        reward=reward,
                        bids=bids,
                    )
                )

            # After collecting the mini-batch, update the model.
            for log in batch_logs:
                model.update(log.arm, log.features, log.reward)

            batch_rewards[batch_idx] = sum(log.reward for log in batch_logs)
            logs.extend(batch_logs)

        return logs, batch_rewards

    def _latent_value(self, context: BidContext) -> float:
        """
        Hidden payoff function that maps context to expected bid strength.
        """
        base = 0.0
        base += self.publisher_effects[self._publisher_index[context.publisher]]
        base += self.placement_effects[self._placement_index[context.placement]]
        base += self.device_effects[self._device_index[context.device]]
        base += self.segment_effects[self._segment_index[context.user_segment]]
        base += self.hour_effects[context.hour_bucket % len(self.hour_effects)]
        return base + self.rng.normal(0.0, 0.05)

    def _one_hot(self, key: str, values: Sequence[str]) -> List[int]:
        vec = [0] * len(values)
        vec[values.index(key)] = 1
        return vec
