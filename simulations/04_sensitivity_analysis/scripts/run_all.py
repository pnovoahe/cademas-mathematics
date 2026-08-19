#!/usr/bin/env python3
"""Orchestrator for Experiment 04 — Sensitivity Analysis.

Runs all three analyses sequentially:
  1. Lambda perturbation (Analysis 1)
  2. Score uncertainty (Analysis 2)
  3. Deep validation / diagnostic report (Analysis 3)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS_DIR.parent
SIM_DIR = EXP_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

from common.utils import ensure_dirs, utc_now_iso, write_json

import run_lambda_perturbation
import run_score_uncertainty
import deep_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Force recompute Monte Carlo trials (ignore cache)")
    args = parser.parse_args()

    dirs = ensure_dirs(EXP_DIR)

    print("=" * 60)
    print("Experiment 04 — Sensitivity Analysis")
    print("=" * 60)

    print("\n[1/3] Lambda perturbation analysis...")
    agg_lam = run_lambda_perturbation.main(refresh=args.refresh)

    print("\n[2/3] Score uncertainty analysis...")
    agg_score = run_score_uncertainty.main(refresh=args.refresh)

    print("\n[3/3] Deep validation and diagnostic report...")
    deep_validation.main(agg_lam=agg_lam, agg_score=agg_score)

    write_json(
        dirs["results"] / "run_metadata.json",
        {
            "experiment": "04_sensitivity_analysis",
            "timestamp": utc_now_iso(),
            "analyses": ["lambda_perturbation", "score_uncertainty", "stability_summary"],
        },
    )
    print("\n[done] Experiment 04 complete.")
    print(f"  Diagnostic report: {dirs['results'] / 'diagnostic_report.md'}")


if __name__ == "__main__":
    main()
