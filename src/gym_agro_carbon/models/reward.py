# ./src/gym_agro_carbon/models/reward.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from gym_agro_carbon.models.context import ContextSpec, ContextEncoder


ActionId = int


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """
    Reward model configuration (V1).

    This file defines a simple, interpretable stochastic reward generator.

    Key design goals:
    - Keep biology minimal (algorithm benchmarking first).
    - Ensure rewards are bounded and stable.
    - Make the effect of soils and agent-planted trees explicit and testable.

    Outputs:
    - C: carbon sequestration proxy (>= 0)
    - Y: yield proxy (>= 0)
    - r: weighted scalar reward r = alpha*C + (1-alpha)*Y
    """

    context_spec: ContextSpec

    # Reward trade-off (policy parameter chosen by the user in V1)
    alpha: float

    # Noise (Gaussian, applied independently to C and Y)
    sigma_c: float = 0.05
    sigma_y: float = 0.05

    # Tree contribution (progressive with tau / M)
    tree_carbon_gain: float = 0.50
    tree_yield_gain: float = 0.10

    # Optional clipping to keep values in a stable range
    clip_min: float = 0.0
    clip_max: Optional[float] = None  # e.g. 2.0 if you want strict bounded rewards

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1]. Got alpha={self.alpha}.")
        if self.sigma_c < 0 or self.sigma_y < 0:
            raise ValueError("sigma_c and sigma_y must be non-negative.")
        if self.tree_carbon_gain < 0 or self.tree_yield_gain < 0:
            raise ValueError("tree gains must be non-negative.")
        if self.clip_max is not None and self.clip_max < self.clip_min:
            raise ValueError("clip_max must be >= clip_min.")


@dataclass(slots=True)
class RewardModel:
    """
    Minimal stochastic reward model for contextual bandits (V1).

    Model structure (simple and interpretable):
    - Soil impacts baseline productivity via a multiplicative scale factor.
    - Each action has a base mean for Y and C.
    - Agent-planted trees add a progressive bonus based on tau (age), capped at M.
    - Gaussian noise is added to both C and Y.

    Important:
    - Pre-existing trees are assumed to be included in soil categories (s),
      so tau only tracks agent-planted trees.
    """

    spec: RewardSpec
    encoder: ContextEncoder

    # Action labels are optional but convenient for logging
    action_labels: Optional[Dict[int, str]] = None

    # Action ids are assumed to be 0..3 (FALLOW, FALLOW_MANURE, TREE_PLANTING, IDLE)
    ACTION_FALLOW: int = 0
    ACTION_FALLOW_MANURE: int = 1
    ACTION_TREE_PLANTING: int = 2
    ACTION_IDLE: int = 3

    def __post_init__(self) -> None:
        self._validate_internal()

    def _validate_internal(self) -> None:
        # encoder/spec consistency
        if self.encoder.spec is not self.spec.context_spec:
            raise ValueError("ContextEncoder.spec must be the same object as RewardSpec.context_spec.")

        # ensure action ids are compact 0..3
        action_ids = sorted(self.action_means_y().keys())
        if action_ids != [0, 1, 2, 3]:
            raise ValueError(f"RewardModel expects action ids [0,1,2,3]. Got {action_ids}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_CY_grid(
        self,
        soil_grid: np.ndarray,
        tau_grid: np.ndarray,
        actions_grid: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (C_grid, Y_grid) for a batch of parcels.

        Inputs
        - soil_grid: (H,W) with values in {1..S}
        - tau_grid : (H,W) with values in {0..M}
        - actions_grid: (H,W) with values in {0..3}

        Returns
        - C_grid: (H,W) float32
        - Y_grid: (H,W) float32
        """
        soil_grid = np.asarray(soil_grid, dtype=np.int32)
        tau_grid = np.asarray(tau_grid, dtype=np.int32)
        actions_grid = np.asarray(actions_grid, dtype=np.int32)

        if soil_grid.shape != tau_grid.shape or soil_grid.shape != actions_grid.shape:
            raise ValueError("soil_grid, tau_grid, and actions_grid must have the same shape.")

        S = self.spec.context_spec.S
        M = self.spec.context_spec.M

        if soil_grid.min() < 1 or soil_grid.max() > S:
            raise ValueError(f"soil_grid values must be in [1..{S}].")
        if tau_grid.min() < 0 or tau_grid.max() > M:
            raise ValueError(f"tau_grid values must be in [0..{M}].")
        if actions_grid.min() < 0 or actions_grid.max() > 3:
            raise ValueError("actions_grid values must be in [0..3].")

        # 1) Baseline soil scaling in [0.8, 1.2] (simple linear mapping)
        soil_scale = self._soil_scale(soil_grid, S)

        # 2) Base action means
        mean_y = self._mean_from_actions(actions_grid, self.action_means_y()) * soil_scale
        mean_c = self._mean_from_actions(actions_grid, self.action_means_c()) * soil_scale

        # 3) Progressive tree bonus from agent-planted trees: tau -> progress in [0,1]
        progress = self._tree_progress(tau_grid, M)
        mean_c = mean_c + self.spec.tree_carbon_gain * progress
        mean_y = mean_y + self.spec.tree_yield_gain * progress

        # 4) Add Gaussian noise
        Y = mean_y + rng.normal(0.0, self.spec.sigma_y, size=mean_y.shape).astype(np.float32)
        C = mean_c + rng.normal(0.0, self.spec.sigma_c, size=mean_c.shape).astype(np.float32)

        # 5) Clip / enforce non-negative
        C = self._clip(C)
        Y = self._clip(Y)

        return C.astype(np.float32), Y.astype(np.float32)

    def reward_grid(self, C_grid: np.ndarray, Y_grid: np.ndarray) -> np.ndarray:
        """
        Compute weighted scalar reward per parcel:
            r = alpha*C + (1-alpha)*Y
        """
        alpha = self.spec.alpha
        r = alpha * C_grid + (1.0 - alpha) * Y_grid
        return r.astype(np.float32)

    def global_reward(self, r_grid: np.ndarray) -> float:
        """Sum of local rewards over the territory."""
        return float(np.sum(r_grid))

    # ------------------------------------------------------------------
    # Means for oracle 
    # ------------------------------------------------------------------
    def mu_CY(self, action: ActionId, s: int, tau: int) -> Tuple[float, float]:
        """
        Return the mean (mu_C, mu_Y) for a single context x=(s,tau) and action.

        This is useful for:
        - oracle policy
        - regret computation from true means

        Note: It mirrors the same deterministic part used in sampling.
        """
        S = self.spec.context_spec.S
        M = self.spec.context_spec.M

        if not (0 <= action <= 3):
            raise ValueError("action must be in [0..3].")
        if not (1 <= s <= S):
            raise ValueError(f"s must be in [1..{S}]. Got {s}.")
        if not (0 <= tau <= M):
            raise ValueError(f"tau must be in [0..{M}]. Got {tau}.")

        soil_scale = self._soil_scale(np.array([[s]], dtype=np.int32), S)[0, 0]
        base_y = self.action_means_y()[action] * soil_scale
        base_c = self.action_means_c()[action] * soil_scale

        progress = self._tree_progress(np.array([[tau]], dtype=np.int32), M)[0, 0]
        mu_y = float(base_y + self.spec.tree_yield_gain * progress)
        mu_c = float(base_c + self.spec.tree_carbon_gain * progress)
        return mu_c, mu_y

    def mu_reward(self, action: ActionId, s: int, tau: int) -> float:
        """Mean scalar reward for (action, context)."""
        mu_c, mu_y = self.mu_CY(action, s, tau)
        alpha = self.spec.alpha
        return float(alpha * mu_c + (1.0 - alpha) * mu_y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def action_means_y(self) -> Dict[int, float]:
        """
        Base mean yield by action (before soil scaling and tree bonus).

        You can later calibrate these values from domain knowledge.
        """
        return {
            self.ACTION_FALLOW: 0.40,
            self.ACTION_FALLOW_MANURE: 0.60,
            self.ACTION_TREE_PLANTING: 0.50,
            self.ACTION_IDLE: 0.55,
        }

    def action_means_c(self) -> Dict[int, float]:
        """
        Base mean carbon by action (before soil scaling and tree bonus).
        """
        return {
            self.ACTION_FALLOW: 0.60,
            self.ACTION_FALLOW_MANURE: 0.70,
            self.ACTION_TREE_PLANTING: 0.80,
            self.ACTION_IDLE: 0.50,
        }

    @staticmethod
    def _mean_from_actions(actions_grid: np.ndarray, table: Dict[int, float]) -> np.ndarray:
        # Fast vectorized lookup for 0..3 actions
        lut = np.array([table[0], table[1], table[2], table[3]], dtype=np.float32)
        return lut[actions_grid]

    @staticmethod
    def _soil_scale(soil_grid: np.ndarray, S: int) -> np.ndarray:
        """
        Map soil in {1..S} to a multiplicative factor in [0.8, 1.2].
        Linear mapping for V1.
        """
        if S == 1:
            return np.ones_like(soil_grid, dtype=np.float32)
        return (0.8 + 0.4 * (soil_grid.astype(np.float32) - 1.0) / (S - 1)).astype(np.float32)

    @staticmethod
    def _tree_progress(tau_grid: np.ndarray, M: int) -> np.ndarray:
        """
        Map tau in {0..M} to progress in [0,1].
        """
        if M <= 0:
            return np.zeros_like(tau_grid, dtype=np.float32)
        return np.clip(tau_grid.astype(np.float32) / float(M), 0.0, 1.0).astype(np.float32)

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if self.spec.clip_max is None:
            return np.clip(arr, self.spec.clip_min, None)
        return np.clip(arr, self.spec.clip_min, self.spec.clip_max)
