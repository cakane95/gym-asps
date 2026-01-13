# ./src/gym_agro_carbon/models/context.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Optional metadata (for logs/plots/README). The model MUST NOT depend on these names.
DEFAULT_SOIL_NAMES: Dict[int, str] = {
    1: "SoilType_1",
    2: "SoilType_2",
    3: "SoilType_3",
    4: "SoilType_4",
    5: "SoilType_5",
    6: "SoilType_6",
    7: "SoilType_7",
    8: "SoilType_8",
}


@dataclass(frozen=True, slots=True)
class ContextSpec:
    """
    Context specification for contextual bandits.

    Context x = (s, tau) where:
    - s   : soil type in {1..S}
    - tau : agent-planted tree age in {0..M}
            tau = 0 means "no agent-planted tree on the parcel"
    """

    S: int  # number of soil types
    M: int  # tree maturity age (max tau)

    soil_names: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.S <= 0:
            raise ValueError(f"S must be positive. Got S={self.S}.")
        if self.M < 0:
            raise ValueError(f"M must be >= 0. Got M={self.M}.")

        # Validate soil names if provided
        if self.soil_names is not None:
            if set(self.soil_names.keys()) != set(range(1, self.S + 1)):
                raise ValueError(
                    "soil_names keys must match {1..S}. "
                    f"Expected {list(range(1, self.S + 1))}, got {sorted(self.soil_names.keys())}."
                )

    @property
    def num_contexts(self) -> int:
        return self.S * (self.M + 1)


@dataclass(frozen=True, slots=True)
class ContextEncoder:
    """
    Discrete encoder for contexts.

    We define a bijection between:
    - (s, tau) with s in {1..S} and tau in {0..M}
    - context_id in {0..S*(M+1)-1}

    Encoding:
        context_id = (s - 1) * (M + 1) + tau
    Decoding:
        tau  = context_id % (M + 1)
        s    = (context_id // (M + 1)) + 1
    """

    spec: ContextSpec

    def to_id(self, s: int, tau: int) -> int:
        if not (1 <= s <= self.spec.S):
            raise ValueError(f"s must be in [1..{self.spec.S}]. Got {s}.")
        if not (0 <= tau <= self.spec.M):
            raise ValueError(f"tau must be in [0..{self.spec.M}]. Got {tau}.")
        return (s - 1) * (self.spec.M + 1) + tau

    def from_id(self, context_id: int) -> Tuple[int, int]:
        if not (0 <= context_id < self.spec.num_contexts):
            raise ValueError(
                f"context_id must be in [0..{self.spec.num_contexts - 1}]. Got {context_id}."
            )
        tau = context_id % (self.spec.M + 1)
        s = (context_id // (self.spec.M + 1)) + 1
        return s, tau

    def soil_name(self, s: int) -> str:
        """
        Return the soil display name (metadata only).
        If no custom names were provided, fall back to a generic label.
        """
        if not (1 <= s <= self.spec.S):
            raise ValueError(f"s must be in [1..{self.spec.S}]. Got {s}.")
        names = self.spec.soil_names
        if names is None:
            return f"SoilType_{s}"
        return names[s]
