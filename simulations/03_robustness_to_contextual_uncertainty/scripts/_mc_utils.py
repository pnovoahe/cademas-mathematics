"""Monte Carlo cache helpers for Experiment 03 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from common.config import N_MONTE_CARLO
from common.utils import utc_now_iso, write_json


def run_monte_carlo_cached(
    *,
    results_dir: Path,
    raw_filename: str,
    meta_filename: str,
    fingerprint: str,
    experiment_name: str,
    meta_extra: dict[str, Any],
    refresh: bool,
    trial_fn: Callable[[int], list[dict]],
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / raw_filename
    meta_path = results_dir / meta_filename

    if raw_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {raw_path} ({meta.get('n_rows')} rows)")
            return pd.read_csv(raw_path)

    records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        records.extend(trial_fn(trial_idx))
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc] trial {trial_idx + 1}/{N_MONTE_CARLO}")

    raw = pd.DataFrame.from_records(records)
    raw.to_csv(raw_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": experiment_name,
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(raw)),
            **meta_extra,
        },
    )
    print(f"[mc] wrote {raw_path} ({len(raw)} rows)")
    return raw
