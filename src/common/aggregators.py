"""Aggregation operators for CADEMAS-ML decision integration.

Operators follow the definitions in manuscript R1V2, Section 4:

- Linear:            A_L(R, Q) = λ R + (1-λ) Q
- Weighted geometric: A_G(R, Q) = R^λ Q^(1-λ), with A_G = 0 if R = 0 or Q = 0
- Minimum:           A_M(R, Q) = min(R, Q)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from common.config import OPERATORS

FloatArray = NDArray[np.floating]


class Aggregator(ABC):
    """Common interface for aggregation operators."""

    name: str

    @abstractmethod
    def aggregate(self, R: FloatArray, Q: FloatArray, lam: float) -> FloatArray:
        """Return prioritization scores P_i = A(R_i, Q_i)."""


class LinearAggregator(Aggregator):
    name = "linear"

    def aggregate(self, R: FloatArray, Q: FloatArray, lam: float) -> FloatArray:
        R = np.asarray(R, dtype=float)
        Q = np.asarray(Q, dtype=float)
        return lam * R + (1.0 - lam) * Q


class GeometricAggregator(Aggregator):
    name = "geometric"

    def aggregate(self, R: FloatArray, Q: FloatArray, lam: float) -> FloatArray:
        R = np.asarray(R, dtype=float)
        Q = np.asarray(Q, dtype=float)
        P = np.power(R, lam) * np.power(Q, 1.0 - lam)
        P[(R == 0.0) | (Q == 0.0)] = 0.0
        return P


class MinimumAggregator(Aggregator):
    name = "min"

    def aggregate(self, R: FloatArray, Q: FloatArray, lam: float) -> FloatArray:
        del lam  # A_M does not depend on λ
        R = np.asarray(R, dtype=float)
        Q = np.asarray(Q, dtype=float)
        return np.minimum(R, Q)


AGGREGATORS: dict[str, Aggregator] = {
    "linear": LinearAggregator(),
    "geometric": GeometricAggregator(),
    "min": MinimumAggregator(),
}


def aggregate(operator: str, R: FloatArray, Q: FloatArray, lam: float) -> FloatArray:
    """Dispatch to a registered aggregator."""
    if operator not in AGGREGATORS:
        raise ValueError(f"Unknown operator: {operator}. Expected one of {OPERATORS}.")
    return AGGREGATORS[operator].aggregate(R, Q, lam)
