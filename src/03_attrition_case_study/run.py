#!/usr/bin/env python3
"""Experiment 03 — Attrition case-study operator scoring.

Builds case-level prioritization scores for seven aggregation configurations:
- A_L at lambda in {0.10, 0.50, 0.90}
- A_G at lambda in {0.10, 0.50, 0.90}
- A_M (minimum)

Inputs:
- data/cases_attrition_100.csv (features + Case_ID)
- models/*.zip MOJO files + models/model_definitions.json
- context/context_digital_transformation.json

Output:
- results/cases_attrition_100_operator_scores.csv
- results/run_metadata.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h2o
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
SIM_DIR = EXP_DIR.parent
REPO_DIR = SIM_DIR.parent
CADEMAS_APP_DIR = REPO_DIR.parent / "cademas-app"
CADEMAS_APP_APP_DIR = CADEMAS_APP_DIR / "app"

if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(CADEMAS_APP_APP_DIR) not in sys.path:
    sys.path.insert(0, str(CADEMAS_APP_APP_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

from common.aggregators import aggregate  # noqa: E402
from common.config import agreement_operator_specs  # noqa: E402
from common.plotting import attrition_case_overview, attrition_rank_bump  # noqa: E402
from common.utils import path_for_metadata  # noqa: E402
from fuzzy_context import calculate_context_score  # noqa: E402

DATA_CSV = EXP_DIR / "data" / "cases_attrition_100.csv"
CONTEXT_JSON = EXP_DIR / "context" / "context_digital_transformation.json"
MODEL_DEFS_JSON = EXP_DIR / "models" / "model_definitions.json"
MODELS_DIR = EXP_DIR / "models"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
OUT_CSV = RESULTS_DIR / "cases_attrition_100_operator_scores.csv"
OUT_META = RESULTS_DIR / "run_metadata.json"

MODEL_SHORT = {
    "cw_StackedEnsemble_BestOfFamily_4_AutoML_1_20260118_132607.zip": "cw",
    "fin_StackedEnsemble_BestOfFamily_4_AutoML_2_20260118_150802.zip": "fin",
    "od_GBM_grid_1_AutoML_3_20260118_163705_model_26.zip": "od",
    "hr_StackedEnsemble_BestOfFamily_4_AutoML_2_20260119_90304.zip": "hr",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_cases() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, sep=";")
    if "Case_ID" not in df.columns:
        raise ValueError(f"Missing Case_ID column in {DATA_CSV}.")
    return df


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_runtime() -> None:
    missing = [path_for_metadata(p, base=EXP_DIR) for p in (DATA_CSV, CONTEXT_JSON, MODEL_DEFS_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    for model_file in MODEL_SHORT:
        if not (MODELS_DIR / model_file).exists():
            raise FileNotFoundError(f"Missing MOJO file: models/{model_file}")
    if not CADEMAS_APP_APP_DIR.exists():
        raise FileNotFoundError(
            "Expected sibling repository 'cademas-app' with app/fuzzy_context.py "
            "(place it next to this companion repository)."
        )


def _auc_weights(model_defs: dict) -> dict[str, float]:
    raw = {
        model_file: float(model_defs[model_file]["performance"]["auc"])
        for model_file in MODEL_SHORT
    }
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("Invalid AUC weights: sum must be > 0.")
    return {k: v / total for k, v in raw.items()}


def _predict_r_and_probs(df: pd.DataFrame, auc_weights: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model_df = df.drop(columns=["Case_ID"], errors="ignore")
    n = len(df)
    risk = np.zeros(n, dtype=float)
    probs: dict[str, np.ndarray] = {}

    for model_file, short_name in MODEL_SHORT.items():
        mojo = h2o.import_mojo(str(MODELS_DIR / model_file))
        hf = h2o.H2OFrame(model_df)
        try:
            pred_df = mojo.predict(hf).as_data_frame()
        finally:
            h2o.remove(hf)
        p_col = "p1" if "p1" in pred_df.columns else pred_df.columns[-1]
        p_vals = pred_df[p_col].to_numpy(dtype=float)
        probs[f"{short_name}_prob"] = p_vals
        risk += p_vals * auc_weights[model_file]
    return risk, probs


def _build_output(df: pd.DataFrame, r_vals: np.ndarray, q_vals: np.ndarray, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Case_ID": df["Case_ID"].astype(str),
            "attrition": df["attrition"].astype(str),
            "R": np.round(r_vals, 6),
            "Q": np.round(q_vals, 6),
        }
    )
    for cfg_id, operator, lam, _label in agreement_operator_specs():
        if operator == "min":
            col_name = "P_AM"
            lam_use = 0.0
        else:
            prefix = "AL" if operator == "linear" else "AG"
            col_name = f"P_{prefix}_{lam:.2f}"
            lam_use = float(lam)
        out[col_name] = np.round(aggregate(operator, r_vals, q_vals, lam_use), 6)

    # Optional audit columns used later in the manuscript.
    out["cw_prob"] = np.round(probs["cw_prob"], 6)
    out["fin_prob"] = np.round(probs["fin_prob"], 6)
    out["od_prob"] = np.round(probs["od_prob"], 6)
    out["hr_prob"] = np.round(probs["hr_prob"], 6)
    return out


def write_figures(scores: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(
        attrition_case_overview(
            scores=scores,
            path_stem=FIGURES_DIR / "attrition_case_overview",
            top_k=10,
        )
    )
    paths.extend(
        attrition_rank_bump(
            scores=scores,
            path_stem=FIGURES_DIR / "attrition_rank_bump",
            top_k=10,
        )
    )
    paths.extend(
        attrition_rank_bump(
            scores=scores,
            path_stem=FIGURES_DIR / "attrition_rank_bump_delta",
            top_k=10,
            delta_vs_baseline=True,
        )
    )
    for p in paths:
        print(f"[fig] {p}")
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Rebuild figures from the cached operator-score CSV (no H2O).",
    )
    args = parser.parse_args()

    if args.figures_only:
        if not OUT_CSV.exists():
            raise FileNotFoundError(f"Missing cached scores: {OUT_CSV}")
        scores = pd.read_csv(OUT_CSV, sep=";")
        write_figures(scores)
        return
    _check_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases_df = _load_cases()
    context_cfg = _load_json(CONTEXT_JSON)
    model_defs = _load_json(MODEL_DEFS_JSON)
    auc_w = _auc_weights(model_defs)

    # Q from fuzzy rules; final logic.op = AND (min of four high-level criteria).
    context_scores, _details = calculate_context_score(cases_df, context_cfg, aggregation=None)
    q_vals = context_scores.to_numpy(dtype=float)

    h2o.init(max_mem_size="700m", nthreads=1)
    try:
        r_vals, probs = _predict_r_and_probs(cases_df, auc_w)
    finally:
        try:
            h2o.shutdown(prompt=False)
        except Exception:
            pass

    out_df = _build_output(cases_df, r_vals, q_vals, probs)
    out_df.to_csv(OUT_CSV, sep=";", index=False)

    metadata = {
        "created_at_utc": _utc_now_iso(),
        "input_data": path_for_metadata(DATA_CSV, base=EXP_DIR),
        "input_context": path_for_metadata(CONTEXT_JSON, base=EXP_DIR),
        "input_model_definitions": path_for_metadata(MODEL_DEFS_JSON, base=EXP_DIR),
        "models_dir": path_for_metadata(MODELS_DIR, base=EXP_DIR),
        "auc_weights": auc_w,
        "n_cases": int(len(out_df)),
        "operators": [cfg_id for cfg_id, *_ in agreement_operator_specs()],
        "output_csv": path_for_metadata(OUT_CSV, base=EXP_DIR),
    }
    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ok] wrote {OUT_CSV} ({len(out_df)} rows)")
    print(f"[ok] wrote {OUT_META}")
    write_figures(out_df)


if __name__ == "__main__":
    main()
