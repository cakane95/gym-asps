# ./src/gym_agro_carbon/envs/grid.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any

import numpy as np

from gym_agro_carbon.models.context import ContextEncoder, ContextSpec
from gym_agro_carbon.models.reward import RewardModel, RewardSpec

@dataclass(frozen=True, slots=True)
class EnvSpec():
    """
    Environnement specification.

    This object is intentionally simple.
    It centralizes all fixed configuration parameters and provides basic validation + convenient derived properties.

    Notes:

    - Grid size: N = H x W parcels
    - Time: T seasons (finite horizon)
    - Context: soil type s in {1..S}, agent-planted tree age tau in {0..M}
    - Actions are discrete indices in {0,1,2,3}
    """
    
    # --- Core Dimensions ---
    H: int
    W: int
    T: int

    # --- Context parameters ---
    S: int # number of soil types
    M: int # maturity age for agent-planted tree

    # --- Environmental Reward Trade-off ---
    alpha: float # in [0,1]

    # --- Reproducibility ---
    seed: Optional[int] = None

    # --- Action Mapping ---
    ACTION_FALLOW: int = 0
    ACTION_FALLOW_MANURE: int = 1
    ACTION_TREE_PLANTING: int = 2
    ACTION_IDLE: int = 3

    def __post_init__(self) -> None:
        self.validate()

    # ---------- Validation ----------
    def validate(self) -> None:
        """Validate the specification. Raises ValueError on invalid settings."""
        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"H and W must be positive. Got H={self.H}, W={self.W}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive (finite horizon). Got T={self.T}.")
        if self.S <= 0:
            raise ValueError(f"S must be positive. Got S={self.S}.")
        if self.M < 0:
            raise ValueError(f"M must be >= 0. Got M={self.M}.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1]. Got alpha={self.alpha}.")

        # Actions must be distinct and form a compact set {0..K-1} for simplicity.
        action_ids = sorted(self.action_labels().keys())
        expected = list(range(self.num_actions))
        if action_ids != expected:
            raise ValueError(
                "Action indices must be exactly {0..K-1} without gaps. "
                f"Got {action_ids}, expected {expected}."
            )

        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be None or a non-negative integer. Got {self.seed}.")
        
    # ------- Derived properties -------
    @property
    def N(self) -> int:
        """ Number of parcels in the grid."""
        return self.H * self.W
    
    @property
    def num_actions(self) -> int:
        """ Number of discrete actions (fixed to 4). """
        return 4
    
    @property
    def num_contexts(self) -> int:
        """
        Number of discrete contexts as (soil, tau).
        - soil has S values
        - tau has M+1 values ('0' means no agent-planted tree)
        """
        return self.S * (self.M + 1)
    
    @property
    def soil_values(self) -> Tuple[int,int]:
        """ Inclusive bounds for soil types. """
        return (1, self.S)
    
    @property
    def tau_values(self) -> Tuple[int,int]:
        """ Inclusive bounds for tau. """
        return (0, self.M)
    
    @property
    def action_values(self) -> Tuple[int, int]:
        """ Inclusive bounds for actions. """
        return (0, self.num_actions - 1)
    
    # ------- Helpers --------
    def action_labels(self) -> Dict[int, str]:
        """ Mapping from action id to human-readable label. """
        return {
            self.ACTION_FALLOW: "FALLOW",
            self.ACTION_FALLOW_MANURE: "FALLOW_MANURE",
            self.ACTION_TREE_PLANTING: "TREE_PLANTING",
            self.ACTION_IDLE: "IDLE",
        }
    
    def to_dict(self) -> Dict[str, object]:
        """ Serialize spec to a dict  (useful for logs and experiment tracking). """
        d = asdict(self)
        # Remove redundant fields (action ids are alreeady  in labels but can be useful to keep)
        d["N"] = self.N
        d["num_actions"] = self.num_actions
        d["num_contexts"] = self.num_contexts
        d["action_labels"] = self.action_labels()
        return d

@dataclass(slots=True)
class GridState():
    """
    Minimal internal state for the grid environment.

    Fields
    - t: current season index (starts at 0 after reset)
    - soil: (H,W) int array with values in {1,.,S}
    - tau: (H, W) int array with values in {0,.,M}, '0' means no agent-planted tree
    """
    
    t: int
    soil: np.ndarray
    tau: np.ndarray

    # ------ Constructors ---------
    @classmethod
    def init(
        cls,
        *,
        H:int,
        W: int,
        S: int,
        M: int,
        rng: np.random.Generator,
        soil: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
    ) -> "GridState":
        """
        Initialize a new GridState.

        Parameters
        - H, W: grid dimensions
        - S: number of soil types
        - M: maturity age of tree
        - rng : numpy random number generator
        - soil: optional pre-defined soil matrix (H,W)
        - tau: optional pre-defined tau matrix

        Returns
        - GridState with t=0
        """
        if H <= 0 or W <= 0:
            raise ValueError(f"H and W must be positive. Got H={H}, W={W}.")
        if S <= 0:
            raise ValueError(f"S must be positive. Got S={S}.")
        if M < 0:
            raise ValueError(f"M must be >= 0. Got M={M}.")

        if soil is None:
            # Soil types are 1..S (inclusive)
            soil = rng.integers(low=1, high=S + 1, size=(H, W), dtype=np.int32)
        else:
            soil = np.asarray(soil, dtype=np.int32)
            if soil.shape != (H, W):
                raise ValueError(f"soil must have shape {(H, W)}. Got {soil.shape}.")
            if soil.min() < 1 or soil.max() > S:
                raise ValueError(f"soil values must be in [1..{S}]. Got [{soil.min()}..{soil.max()}].")
            
        if tau is None:
            tau = np.zeros((H, W), dtype=np.int32)
        else:
            tau = np.asarray(tau, dtype=np.int32)
            if tau.shape != (H, W):
                raise ValueError(f"tau must have shape {(H, W)}. Got {tau.shape}.")
            if tau.min() < 0 or tau.max() > M:
                raise ValueError(f"tau values must be in [0..{M}]. Got [{tau.min()}..{tau.max()}].")

        return cls(t=0, soil=soil, tau=tau)
    
    # ---------- Core dynamics helpers ----------
    def tick(self) -> None:
        """Advance time by one season."""
        self.t += 1

    def age_agent_trees(self, M: int) -> None:
        """
        Age agent-planted trees by +1 season, capped at M.
        tau=0 stays 0.
        """
        if M < 0:
            raise ValueError(f"M must be >= 0. Got M={M}.")
        if M == 0:
            # Only tau=0 is allowed; everything stays 0.
            self.tau.fill(0)
            return

        mask = self.tau > 0
        if np.any(mask):
            self.tau[mask] = np.minimum(self.tau[mask] + 1, M)

    def plant_agent_trees(self, plant_mask: np.ndarray) -> None:
        """
        Plant new agent trees on cells where:
        - plant_mask is True
        - and currently tau == 0
        Sets tau to 1 on those cells.
        """
        plant_mask = np.asarray(plant_mask, dtype=bool)
        if plant_mask.shape != self.tau.shape:
            raise ValueError(f"plant_mask must have shape {self.tau.shape}. Got {plant_mask.shape}.")

        new_tree_cells = plant_mask & (self.tau == 0)
        self.tau[new_tree_cells] = 1

    # ---------- Convenience ----------
    def copy(self) -> "GridState":
        """Deep copy arrays (safe for debugging and tests)."""
        return GridState(t=int(self.t), soil=self.soil.copy(), tau=self.tau.copy())

    def is_terminal(self, T: int) -> bool:
        """Return True if the episode should terminate (t >= T)."""
        if T <= 0:
            raise ValueError(f"T must be positive. Got T={T}.")
        return self.t >= T
    
    def summary(self) -> Dict[str, Any]:
        """Quick diagnostic summary for logging/debugging."""
        tau = self.tau
        return {
            "t": int(self.t),
            "grid_shape": tuple(tau.shape),
            "num_parcels": int(tau.size),
            "num_agent_trees": int(np.sum(tau > 0)),
            "mean_tau_positive": float(tau[tau > 0].mean()) if np.any(tau > 0) else 0.0,
            "max_tau": int(tau.max()) if tau.size else 0,
            "soil_min": int(self.soil.min()) if self.soil.size else 0,
            "soil_max": int(self.soil.max()) if self.soil.size else 0,
        }

@dataclass(slots=True)
class GridObservationEncoder:
    """
    Encode the full GridState into observations consumable by learning algorithms.

    V1 choice:
    - Observation = discrete context_id per parcel
    - Shape: (H, W)
    - context_id is defined by ContextEncoder
    """

    context_encoder: ContextEncoder

    def encode(self, state: GridState) -> np.ndarray:
        """
        Encode the grid state into a matrix of context IDs.

        Parameters
        ----------
        state : GridState
            Internal environment state containing soil and tau grids.

        Returns
        -------
        obs : np.ndarray of shape (H, W)
            Discrete context_id for each parcel.
        """
        soil = state.soil
        tau = state.tau

        if soil.shape != tau.shape:
            raise ValueError(
                f"soil and tau must have the same shape. "
                f"Got soil={soil.shape}, tau={tau.shape}."
            )

        # Vectorized encoding:
        # context_id = (s - 1) * (M + 1) + tau
        M = self.context_encoder.spec.M
        context_ids = (soil - 1) * (M + 1) + tau

        return context_ids.astype(np.int32)

    def encode_single(self, s: int, tau: int) -> int:
        """
        Encode a single parcel context.
        Convenience wrapper around ContextEncoder.
        """
        return self.context_encoder.to_id(s, tau)

    def decode_single(self, context_id: int) -> tuple[int, int]:
        """
        Decode a single context_id back to (soil, tau).
        """
        return self.context_encoder.from_id(context_id)

class AgroCarbonGridEnv:
    """
    Minimal Gym-like environment for V1.

    - One step = one season
    - Batch actions: actions is (H,W) with one action per parcel
    - Context per parcel: (soil, tau)
      * soil is static (includes pre-existing vegetation effects)
      * tau tracks only agent-planted trees

    Returns:
    - obs: (H,W) matrix of context_id
    - reward: global scalar sum over parcels
    - info: contains local grids (C_grid, Y_grid, reward_grid) + diagnostics
    """
    def __init__(
        self,
        env_spec: EnvSpec,
        context_spec: ContextSpec,
        reward_model: RewardModel,
        *,
        soil: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
    ) -> None:
        # Basic consistency checks
        if context_spec.S != env_spec.S or context_spec.M != env_spec.M:
            raise ValueError(
                "EnvSpec (S,M) must match ContextSpec (S,M). "
                f"Got EnvSpec(S={env_spec.S}, M={env_spec.M}) vs "
                f"ContextSpec(S={context_spec.S}, M={context_spec.M})."
            )
        if abs(env_spec.alpha - reward_model.spec.alpha) > 1e-12:
            raise ValueError(
                "EnvSpec.alpha must match RewardSpec.alpha used by RewardModel. "
                f"Got EnvSpec.alpha={env_spec.alpha}, RewardModel.alpha={reward_model.spec.alpha}."
            )
        if reward_model.spec.context_spec is not context_spec:
            raise ValueError(
                "RewardModel must be built using the same ContextSpec instance passed to the env."
            )

        self.env_spec = env_spec
        self.context_spec = context_spec
        self.context_encoder = ContextEncoder(context_spec)
        self.obs_encoder = GridObservationEncoder(self.context_encoder)

        self.reward_model = reward_model

        self._rng = np.random.default_rng(env_spec.seed)
        self._init_soil = soil
        self._init_tau = tau

        self.state: Optional[GridState] = None

    # -------------------------
    # Gym-like API
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            if seed < 0:
                raise ValueError(f"seed must be non-negative. Got {seed}.")
            self._rng = np.random.default_rng(seed)

        self.state = GridState.init(
            H=self.env_spec.H,
            W=self.env_spec.W,
            S=self.env_spec.S,
            M=self.env_spec.M,
            rng=self._rng,
            soil=self._init_soil,
            tau=self._init_tau,
        )

        obs = self.obs_encoder.encode(self.state)
        info = {
            "t": self.state.t,
            "alpha": self.env_spec.alpha,
            "action_labels": self.env_spec.action_labels(),
            "num_agent_trees": int(np.sum(self.state.tau > 0)),
        }
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("You must call reset() before step().")

        actions = np.asarray(actions, dtype=np.int32)
        expected_shape = (self.env_spec.H, self.env_spec.W)
        if actions.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}. Got {actions.shape}.")
        if actions.min() < 0 or actions.max() >= self.env_spec.num_actions:
            raise ValueError(
                f"actions must be in [0..{self.env_spec.num_actions - 1}]. "
                f"Got [{actions.min()}..{actions.max()}]."
            )

        # --- Dynamics: age existing trees, then plant new trees
        self.state.age_agent_trees(self.env_spec.M)
        plant_mask = actions == self.env_spec.ACTION_TREE_PLANTING
        self.state.plant_agent_trees(plant_mask)

        # --- Rewards: sample C,Y then compute scalar reward
        C_grid, Y_grid = self.reward_model.sample_CY_grid(
            soil_grid=self.state.soil,
            tau_grid=self.state.tau,
            actions_grid=actions,
            rng=self._rng,
        )
        r_grid = self.reward_model.reward_grid(C_grid, Y_grid)
        R_t = self.reward_model.global_reward(r_grid)

        # --- Advance time
        self.state.tick()

        obs = self.obs_encoder.encode(self.state)
        terminated = self.state.is_terminal(self.env_spec.T)
        truncated = False

        info: Dict[str, Any] = {
            "t": self.state.t,
            "R_t": float(R_t),
            "reward_grid": r_grid,
            "C_grid": C_grid,
            "Y_grid": Y_grid,
            "num_agent_trees": int(np.sum(self.state.tau > 0)),
        }
        return obs, float(R_t), terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """
        Minimal human rendering:
        returns an RGB array based on tau (agent-planted tree age heatmap).
        """
        if mode == "none":
            return None
        if self.state is None:
            raise RuntimeError("You must call reset() before render().")

        tau = self.state.tau.astype(np.float32)
        denom = max(1, self.env_spec.M)
        gray = np.clip((tau / denom) * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # (H,W,3)
        return rgb

    def close(self) -> None:
        self.state = None