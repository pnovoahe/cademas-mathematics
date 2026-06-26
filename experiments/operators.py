"""Aggregation operators for CADEMAS decision integration."""

import numpy as np


def aggregate(operator: str, R: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute prioritization scores P_i = A(R_i, Q_i).

    Operators:
    - linear:    lambda * R + (1 - lambda) * Q
    - min:       min(R, Q)
    - max:       max(R, Q)
    - geometric: R^lambda * Q^(1-lambda), with P=0 when R=0 or Q=0
    """
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if operator == "linear":
        return lam * R + (1.0 - lam) * Q

    if operator == "min":
        return np.minimum(R, Q)

    if operator == "max":
        return np.maximum(R, Q)

    if operator == "geometric":
        P = np.power(R, lam) * np.power(Q, 1.0 - lam)
        P[(R == 0) | (Q == 0)] = 0.0
        return P

    raise ValueError(f"Unknown operator: {operator}")
