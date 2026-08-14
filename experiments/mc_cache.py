"""Disk cache for aggregated Monte Carlo summaries used by plots and stats.

Stores a few hundred rows of means/CIs (kilobytes), not trial-level draws.
Invalidate automatically when the experiment fingerprint changes, or force
a refresh with ``CADEMAS_MC_REFRESH=1`` / ``--refresh``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from systemic_config import (
    INTER_LAMBDA_MAX,
    INTER_LAMBDA_MIN,
    INTER_LAMBDA_POINTS,
    INTER_Q,
    INTER_R_MU,
    INTER_R_STD,
    K,
    MC_CACHE_DIR,
    MC_SEED_BASE,
    MU_SHIFT_MAX,
    MU_SHIFT_MIN,
    MU_SHIFT_POINTS,
    N,
    N_INTER,
    N_MC_TRIALS,
    N_STD,
    N_VETO,
    NOISE_DISTRIBUTIONS,
    NOISE_LAMBDAS,
    NOISE_POPULATION_MODELS,
    NOISE_SIGMA_MAX,
    NOISE_SIGMA_MIN,
    NOISE_SIGMA_POINTS,
    OPP_LAMBDA_MAX,
    OPP_LAMBDA_MIN,
    OPP_LAMBDA_POINTS,
    PREDICTIVE_OVERCONFIDENCE_LAMBDA,
    VETO_R_STD,
)

CACHE_VERSION = 1


def refresh_requested() -> bool:
    return os.environ.get("CADEMAS_MC_REFRESH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def experiment_fingerprint(n_trials: int, seed_base: int) -> str:
    payload = {
        "version": CACHE_VERSION,
        "n": N,
        "k": K,
        "n_std": N_STD,
        "n_veto": N_VETO,
        "n_inter": N_INTER,
        "n_trials": n_trials,
        "seed_base": seed_base,
        "opp": [OPP_LAMBDA_MIN, OPP_LAMBDA_MAX, OPP_LAMBDA_POINTS],
        "overconf_lambda": PREDICTIVE_OVERCONFIDENCE_LAMBDA,
        "mu": [MU_SHIFT_MIN, MU_SHIFT_MAX, MU_SHIFT_POINTS, VETO_R_STD],
        "noise_lambdas": list(NOISE_LAMBDAS),
        "noise_populations": list(NOISE_POPULATION_MODELS),
        "noise_dists": list(NOISE_DISTRIBUTIONS),
        "noise_sigma": [NOISE_SIGMA_MIN, NOISE_SIGMA_MAX, NOISE_SIGMA_POINTS],
        "inter": [
            INTER_Q,
            INTER_R_MU,
            INTER_R_STD,
            INTER_LAMBDA_MIN,
            INTER_LAMBDA_MAX,
            INTER_LAMBDA_POINTS,
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _paths(name: str) -> tuple[Path, Path]:
    return MC_CACHE_DIR / f"{name}.csv", MC_CACHE_DIR / f"{name}.json"


def load_or_compute(
    name: str,
    compute: Callable[[], pd.DataFrame],
    *,
    n_trials: int = N_MC_TRIALS,
    seed_base: int = MC_SEED_BASE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached aggregated results, or run ``compute`` and store them."""
    MC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _paths(name)
    fingerprint = experiment_fingerprint(n_trials, seed_base)
    force = refresh or refresh_requested()

    if not force and csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            df = pd.read_csv(csv_path)
            print(f"[cache] loaded {name} ({len(df)} rows) from {csv_path}")
            return df

    df = compute()
    df.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_rows": int(len(df)),
                "n_trials": n_trials,
                "seed_base": seed_base,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[cache] wrote {name} ({len(df)} rows) -> {csv_path}")
    return df
