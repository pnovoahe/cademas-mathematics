#!/usr/bin/env python3
"""Experiment 02 — Sensitivity to Gaussian noise on predictive scores R.

Q is held fixed (Experiment 01 population design). For each Monte Carlo
trial and each σ_R, R is perturbed. Metrics V / τ / Jaccard-vs-clean and
pairwise Top-K agreement use seven ranking configurations: A_L and A_G at
λ∈{0.10,0.50,0.90} plus A_M. A separate sweep records V(K) for
K∈{10,20,…,100}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SIM_DIR = EXP_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from common.aggregators import aggregate  # noqa: E402
from common.config import (  # noqa: E402
    AGREEMENT_LAMBDA_VALUES,
    CI_LEVEL,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    N_STD,
    N_VETO,
    OPERATOR_LABELS,
    OPERATORS,
    SEEDS,
    SIGMA_R_FIGURE_VALUES,
    SIGMA_R_VALUES,
    SIGMA_R_V_BY_K_VALUES,
    STD_Q_BETA,
    STD_R_BETA,
    TOP_K,
    TOP_K_SWEEP_VALUES,
    VETO_FRACTION,
    VETO_R_BETA,
    agreement_operator_specs,
)
from common.generators import (  # noqa: E402
    apply_score_uncertainty,
    apply_score_uncertainty_with_clip_mask,
    generate_population,
)
from common.metrics import (  # noqa: E402
    jaccard_top_k,
    kendall_tau_top_k,
    policy_violation_rate,
    topk_clip_veto_shares,
)
from common.plotting import (  # noqa: E402
    sensitivity_operator_agreement,
    sensitivity_r_noise_overview,
    sensitivity_v_by_k,
)
from common.utils import (  # noqa: E402
    ensure_dirs,
    experiment_fingerprint,
    summarize_trials,
    trial_seed,
    utc_now_iso,
    write_json,
)

METRIC_COLS = (
    "policy_violation_rate",
    "kendall_tau",
    "jaccard_top_k",
)
PAIR_COLS = ("kendall_tau", "jaccard_top_k")
V_BY_K_COLS = ("policy_violation_rate",)
CLIPPED_AL90_COLS = (
    "clipped_non_veto_top_k_share",
    "vetoed_top_k_share",
    "other_top_k_share",
)
LINEAR_AL90 = "linear@0.90"
LINEAR_AL90_LAM = 0.90
SPECS = agreement_operator_specs()


def _agg(op: str, R: np.ndarray, Q: np.ndarray, lam: float | None) -> np.ndarray:
    return aggregate(op, R, Q, 0.0 if lam is None else float(lam))


def _fingerprint() -> str:
    return experiment_fingerprint(
        "02_sensitivity_analysis",
        {
            "n_cases": N_CASES,
            "n_std": N_STD,
            "n_veto": N_VETO,
            "veto_fraction": VETO_FRACTION,
            "top_k": TOP_K,
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "sigma_r": list(SIGMA_R_VALUES),
            "agreement_lambdas": list(AGREEMENT_LAMBDA_VALUES),
            "agreement_configs": [s[0] for s in SPECS],
            "metric_configs": [s[0] for s in SPECS],
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
            "pairs": "lower_triangle_7configs",
        },
    )


def _v_by_k_fingerprint() -> str:
    return experiment_fingerprint(
        "02_sensitivity_analysis_v_by_k",
        {
            "n_cases": N_CASES,
            "n_std": N_STD,
            "n_veto": N_VETO,
            "veto_fraction": VETO_FRACTION,
            "top_k_sweep": list(TOP_K_SWEEP_VALUES),
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "sigma_r": list(SIGMA_R_VALUES),
            "agreement_configs": [s[0] for s in SPECS],
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
        },
    )


def _clipped_al90_fingerprint() -> str:
    return experiment_fingerprint(
        "02_sensitivity_analysis_clipped_al90",
        {
            "n_cases": N_CASES,
            "n_std": N_STD,
            "n_veto": N_VETO,
            "veto_fraction": VETO_FRACTION,
            "top_k_sweep": list(TOP_K_SWEEP_VALUES),
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "sigma_r": list(SIGMA_R_V_BY_K_VALUES),
            "config_id": LINEAR_AL90,
            "lambda": LINEAR_AL90_LAM,
            "metrics": list(CLIPPED_AL90_COLS),
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
        },
    )


def _run_trial(trial_idx: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(rng, n_std=N_STD, n_veto=N_VETO)

    p_clean_cfg: dict[str, np.ndarray] = {
        cid: _agg(op, pop.R, pop.Q, lam) for cid, op, lam, _ in SPECS
    }

    metric_rows: list[dict] = []
    pair_rows: list[dict] = []

    for sigma_r in SIGMA_R_VALUES:
        noise_rng = np.random.default_rng(
            trial_seed(trial_idx) + int(round(float(sigma_r) * 10000))
        )
        r_noisy = apply_score_uncertainty(pop.R, noise_rng, sigma_r)

        p_noisy_cfg = {
            cid: _agg(op, r_noisy, pop.Q, lam) for cid, op, lam, _ in SPECS
        }
        for cid, op, lam, _ in SPECS:
            P = p_noisy_cfg[cid]
            P_clean = p_clean_cfg[cid]
            metric_rows.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "sigma_r": float(sigma_r),
                    "config_id": cid,
                    "operator": op,
                    "lambda": np.nan if lam is None else float(lam),
                    "policy_violation_rate": policy_violation_rate(
                        P, pop.Q, TOP_K, R=r_noisy, case_id=pop.case_id
                    ),
                    "kendall_tau": kendall_tau_top_k(
                        P_clean,
                        P,
                        TOP_K,
                        R_ref=pop.R,
                        R=r_noisy,
                        case_id=pop.case_id,
                    ),
                    "jaccard_top_k": jaccard_top_k(
                        P_clean,
                        P,
                        TOP_K,
                        R_ref=pop.R,
                        R=r_noisy,
                        case_id=pop.case_id,
                    ),
                }
            )

        p_cfg = p_noisy_cfg
        # One directed pair per unordered combination → lower triangle only.
        for (cid_a, _, _, _), (cid_b, _, _, _) in itertools.combinations(SPECS, 2):
            Pa, Pb = p_cfg[cid_a], p_cfg[cid_b]
            pair_rows.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "sigma_r": float(sigma_r),
                    "operator_a": cid_a,
                    "operator_b": cid_b,
                    "kendall_tau": kendall_tau_top_k(
                        Pa, Pb, TOP_K, R_ref=r_noisy, R=r_noisy, case_id=pop.case_id
                    ),
                    "jaccard_top_k": jaccard_top_k(
                        Pa, Pb, TOP_K, R_ref=r_noisy, R=r_noisy, case_id=pop.case_id
                    ),
                }
            )

    return metric_rows, pair_rows


def _run_v_by_k_trial(trial_idx: int) -> list[dict]:
    """Policy violation rate $V$ over the Top-$K$ sweep for one Monte Carlo trial."""
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(rng, n_std=N_STD, n_veto=N_VETO)
    rows: list[dict] = []

    for sigma_r in SIGMA_R_VALUES:
        noise_rng = np.random.default_rng(
            trial_seed(trial_idx) + int(round(float(sigma_r) * 10000))
        )
        r_noisy = apply_score_uncertainty(pop.R, noise_rng, sigma_r)
        for cid, op, lam, _ in SPECS:
            P = _agg(op, r_noisy, pop.Q, lam)
            for k in TOP_K_SWEEP_VALUES:
                rows.append(
                    {
                        "trial": trial_idx,
                        "seed": trial_seed(trial_idx),
                        "sigma_r": float(sigma_r),
                        "top_k": int(k),
                        "config_id": cid,
                        "operator": op,
                        "lambda": np.nan if lam is None else float(lam),
                        "policy_violation_rate": policy_violation_rate(
                            P, pop.Q, int(k), R=r_noisy, case_id=pop.case_id
                        ),
                    }
                )
    return rows


def _run_clipped_al90_trial(trial_idx: int) -> list[dict]:
    """Top-$K$ clip/veto composition for $A_L(0.90)$ over the $K$ sweep."""
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(rng, n_std=N_STD, n_veto=N_VETO)
    rows: list[dict] = []

    for sigma_r in SIGMA_R_V_BY_K_VALUES:
        noise_rng = np.random.default_rng(
            trial_seed(trial_idx) + int(round(float(sigma_r) * 10000))
        )
        r_noisy, is_clipped = apply_score_uncertainty_with_clip_mask(
            pop.R, noise_rng, sigma_r
        )
        P = _agg("linear", r_noisy, pop.Q, LINEAR_AL90_LAM)
        for k in TOP_K_SWEEP_VALUES:
            shares = topk_clip_veto_shares(
                P,
                pop.Q,
                is_clipped,
                int(k),
                R=r_noisy,
                case_id=pop.case_id,
            )
            rows.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "sigma_r": float(sigma_r),
                    "top_k": int(k),
                    "config_id": LINEAR_AL90,
                    **shares,
                }
            )
    return rows


def run_monte_carlo(
    results_dir: Path, *, refresh: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = results_dir / "trials_metrics.csv"
    pairs_path = results_dir / "trials_pairs.csv"
    meta_path = results_dir / "run_metadata.json"
    fingerprint = _fingerprint()

    if (
        metrics_path.exists()
        and pairs_path.exists()
        and meta_path.exists()
        and not refresh
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {metrics_path}")
            return pd.read_csv(metrics_path), pd.read_csv(pairs_path)

    metric_records: list[dict] = []
    pair_records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        m_rows, p_rows = _run_trial(trial_idx)
        metric_records.extend(m_rows)
        pair_records.extend(p_rows)
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc] trial {trial_idx + 1}/{N_MONTE_CARLO}")

    metrics = pd.DataFrame.from_records(metric_records)
    pairs = pd.DataFrame.from_records(pair_records)
    metrics.to_csv(metrics_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": "02_sensitivity_analysis",
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows_metrics": int(len(metrics)),
            "n_rows_pairs": int(len(pairs)),
            "n_monte_carlo": N_MONTE_CARLO,
            "seeds": SEEDS,
            "seed_base": MC_SEED_BASE,
            "sigma_r_values": list(SIGMA_R_VALUES),
            "agreement_configs": [s[0] for s in SPECS],
            "metric_configs": [s[0] for s in SPECS],
            "ci_level": CI_LEVEL,
        },
    )
    print(f"[mc] wrote {metrics_path} ({len(metrics)} rows)")
    print(f"[mc] wrote {pairs_path} ({len(pairs)} rows)")
    return metrics, pairs


def aggregate_results(
    metrics: pd.DataFrame, pairs: pd.DataFrame, results_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_metrics = summarize_trials(
        metrics, group_cols=("sigma_r", "config_id"), metrics=METRIC_COLS
    )
    agg_pairs = summarize_trials(
        pairs,
        group_cols=("sigma_r", "operator_a", "operator_b"),
        metrics=PAIR_COLS,
    )
    # Orient pairs so matrix rows/cols follow SPECS order: store both
    # (a,b) and fill only lower triangle in the plot via config order.
    # combinations(SPECS) yields a before b in SPECS order; for lower
    # triangle we need row_index > col_index. Remap when building matrix.
    oriented: list[dict] = []
    order = {cid: i for i, (cid, *_rest) in enumerate(SPECS)}
    for _, row in agg_pairs.iterrows():
        a, b = str(row["operator_a"]), str(row["operator_b"])
        if order[a] < order[b]:
            # Put higher-index config on rows (lower triangle visually).
            a, b = b, a
        rec = row.to_dict()
        rec["operator_a"] = a
        rec["operator_b"] = b
        oriented.append(rec)
    agg_pairs = pd.DataFrame(oriented)

    agg_metrics_path = results_dir / "aggregated_metrics.csv"
    agg_pairs_path = results_dir / "aggregated_pairs.csv"
    agg_metrics.to_csv(agg_metrics_path, index=False)
    agg_pairs.to_csv(agg_pairs_path, index=False)
    print(f"[agg] wrote {agg_metrics_path}")
    print(f"[agg] wrote {agg_pairs_path}")
    return agg_metrics, agg_pairs


def run_v_by_k(results_dir: Path, *, refresh: bool) -> pd.DataFrame:
    trials_path = results_dir / "trials_v_by_k.csv"
    meta_path = results_dir / "run_metadata_v_by_k.json"
    fingerprint = _v_by_k_fingerprint()

    if trials_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {trials_path}")
            return pd.read_csv(trials_path)

    records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        records.extend(_run_v_by_k_trial(trial_idx))
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc:v(k)] trial {trial_idx + 1}/{N_MONTE_CARLO}")

    trials = pd.DataFrame.from_records(records)
    trials.to_csv(trials_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": "02_sensitivity_analysis_v_by_k",
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(trials)),
            "n_monte_carlo": N_MONTE_CARLO,
            "seeds": SEEDS,
            "seed_base": MC_SEED_BASE,
            "sigma_r_values": list(SIGMA_R_VALUES),
            "top_k_sweep": list(TOP_K_SWEEP_VALUES),
            "agreement_configs": [s[0] for s in SPECS],
            "ci_level": CI_LEVEL,
        },
    )
    print(f"[mc:v(k)] wrote {trials_path} ({len(trials)} rows)")
    return trials


def aggregate_v_by_k(trials: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    agg = summarize_trials(
        trials,
        group_cols=("sigma_r", "top_k", "config_id"),
        metrics=V_BY_K_COLS,
    )
    out_path = results_dir / "aggregated_v_by_k.csv"
    agg.to_csv(out_path, index=False)
    print(f"[agg] wrote {out_path}")
    return agg


def run_clipped_al90(results_dir: Path, *, refresh: bool) -> pd.DataFrame:
    trials_path = results_dir / "trials_clipped_al90_topk.csv"
    meta_path = results_dir / "run_metadata_clipped_al90_topk.json"
    fingerprint = _clipped_al90_fingerprint()

    if trials_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {trials_path}")
            return pd.read_csv(trials_path)

    records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        records.extend(_run_clipped_al90_trial(trial_idx))
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc:clip] trial {trial_idx + 1}/{N_MONTE_CARLO}")

    trials = pd.DataFrame.from_records(records)
    trials.to_csv(trials_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": "02_sensitivity_analysis_clipped_al90",
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(trials)),
            "n_monte_carlo": N_MONTE_CARLO,
            "seeds": SEEDS,
            "seed_base": MC_SEED_BASE,
            "sigma_r_values": list(SIGMA_R_V_BY_K_VALUES),
            "top_k_sweep": list(TOP_K_SWEEP_VALUES),
            "config_id": LINEAR_AL90,
            "ci_level": CI_LEVEL,
        },
    )
    print(f"[mc:clip] wrote {trials_path} ({len(trials)} rows)")
    return trials


def aggregate_clipped_al90(trials: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    agg = summarize_trials(
        trials,
        group_cols=("sigma_r", "top_k"),
        metrics=CLIPPED_AL90_COLS,
    )
    out_path = results_dir / "aggregated_clipped_al90_topk.csv"
    agg.to_csv(out_path, index=False)
    print(f"[agg] wrote {out_path}")
    return agg


def write_figures(
    agg_metrics: pd.DataFrame,
    agg_pairs: pd.DataFrame,
    agg_v_by_k: pd.DataFrame,
    agg_clipped_al90: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    rng = np.random.default_rng(trial_seed(0))
    pop = generate_population(rng, n_std=N_STD, n_veto=N_VETO)
    noisy_pops: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sigma_r in SIGMA_R_FIGURE_VALUES:
        noise_rng = np.random.default_rng(
            trial_seed(0) + int(round(float(sigma_r) * 10000))
        )
        r_noisy = apply_score_uncertainty(pop.R, noise_rng, sigma_r)
        noisy_pops[float(sigma_r)] = (r_noisy, pop.Q, pop.is_veto)

    paths: list[Path] = []
    paths.extend(
        sensitivity_r_noise_overview(
            noisy_pops=noisy_pops,
            agg_metrics=agg_metrics,
            path_stem=figures_dir / "sensitivity_r_noise_overview",
            sigma_values=SIGMA_R_FIGURE_VALUES,
        )
    )
    paths.extend(
        sensitivity_operator_agreement(
            agg_pairs=agg_pairs,
            path_stem=figures_dir / "sensitivity_operator_agreement",
            sigma_values=SIGMA_R_FIGURE_VALUES,
        )
    )
    fig_v = agg_v_by_k[
        agg_v_by_k["sigma_r"].apply(
            lambda s: any(np.isclose(float(s), float(v)) for v in SIGMA_R_V_BY_K_VALUES)
        )
    ]
    paths.extend(
        sensitivity_v_by_k(
            agg_v_by_k=fig_v,
            agg_clipped_al90=agg_clipped_al90,
            path_stem=figures_dir / "sensitivity_v_by_k",
            sigma_values=SIGMA_R_V_BY_K_VALUES,
            k_values=TOP_K_SWEEP_VALUES,
        )
    )
    return paths


def write_docs(
    agg_metrics: pd.DataFrame,
    agg_pairs: pd.DataFrame,
    agg_v_by_k: pd.DataFrame,
    agg_clipped_al90: pd.DataFrame,
    exp_dir: Path,
) -> None:
    lines = [
        "# Sensitivity to predictive-score noise (Experiment 02)",
        "",
        f"Population as in Experiment 01 ($N={N_CASES}$, veto fraction "
        f"{VETO_FRACTION:.2f}). Only $R$ is perturbed; $Q$ is fixed.",
        "Metric bar panels use the same seven ranking configurations as the "
        "agreement heatmaps: "
        + ", ".join(s[3] for s in SPECS) + ".",
        f"Monte Carlo: {N_MONTE_CARLO} trials. Noise grid: {list(SIGMA_R_VALUES)}.",
        f"Policy-violation sweep: $K\\in\\{{{', '.join(str(k) for k in TOP_K_SWEEP_VALUES)}\\}}$.",
        "",
    ]
    (exp_dir / "results_narrative.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (exp_dir / "captions.md").write_text(
        "\n".join(
            [
                "# Figure captions",
                "",
                "## sensitivity_r_noise_overview",
                "",
                "Sensitivity of aggregation to Gaussian noise on predictive scores "
                f"($N={N_CASES}$, $K={TOP_K}$, {N_MONTE_CARLO} Monte Carlo replications). "
                "Bar panels report Monte Carlo means of $V$, Kendall $\\tau$ "
                "(Top-$K$ union), "
                "and Jaccard Top-$K$ with $95\\%$ percentile intervals for "
                "seven ranking configurations ($A_L$ and $A_G$ at "
                f"$\\lambda\\in\\{{{', '.join(f'{x:.2g}' for x in AGREEMENT_LAMBDA_VALUES)}\\}}$, "
                "plus $A_M$). "
                "(\\textbf{a}--\\textbf{c}) Illustrative $(R',Q)$ populations. "
                "(\\textbf{d}--\\textbf{f}) Monte Carlo means versus clean rankings "
                "with $95\\%$ percentile intervals.",
                "",
                "## sensitivity_operator_agreement",
                "",
                "Lower-triangle pairwise agreement among seven ranking configurations "
                f"($A_L$ and $A_G$ at $\\lambda\\in\\{{{', '.join(f'{x:.2f}' for x in AGREEMENT_LAMBDA_VALUES)}\\}}$, "
                "plus $A_M$) under shared $R'$ noise. "
                "Kendall $\\tau$ is computed on the union of the two Top-$K$ sets; "
                "Jaccard is the Top-$K$ set overlap. Diagonal omitted.",
                "",
                "## sensitivity_v_by_k",
                "",
                "Policy violation rate $V$ as a function of the Top-$K$ selection size "
                f"($K\\in\\{{{', '.join(str(k) for k in TOP_K_SWEEP_VALUES)}\\}}$) for seven "
                "ranking configurations under four predictive-noise levels "
                f"($\\sigma_R\\in\\{{{', '.join(f'{x:.2f}' for x in SIGMA_R_V_BY_K_VALUES)}\\}}$). "
                "Lines show Monte Carlo means with shaded $95\\%$ percentile intervals. "
                "Panel~(\\textbf{e}) shows, for $A_L(0.90)$, the Top-$K$ composition "
                "at each $(\\sigma_R,K)$ as a 100\\% stacked tile "
                "(bottom to top: other, clipped non-veto, $Q$-vetoed), "
                "with $\\sigma_R$ increasing upward.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (exp_dir / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 02 — Sensitivity to predictive-score noise",
                "",
                "Gaussian noise on $R$ only; $Q$ fixed as in Experiment 01.",
                "",
                "Figures:",
                "",
                "- `sensitivity_r_noise_overview` — $(R',Q)$ scatters and $V$/τ/Jaccard bars",
                "- `sensitivity_operator_agreement` — pairwise Kendall (Top-$K$ union) / Jaccard heatmaps",
                "- `sensitivity_v_by_k` — $V(K)$ for $K=10,20,\\ldots,100$ at four $\\sigma_R$ (2×2 + panel e)",
                "",
                "## Run",
                "",
                "```bash",
                "cd src && .venv/bin/python 02_sensitivity_analysis/run.py",
                "```",
                "",
                "Use `--refresh` to ignore the Monte Carlo cache.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _ = (agg_metrics, agg_pairs, agg_v_by_k, agg_clipped_al90, OPERATOR_LABELS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip Monte Carlo; rebuild figures from cached aggregated CSVs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(EXP_DIR)
    if args.figures_only:
        agg_metrics = pd.read_csv(paths["results"] / "aggregated_metrics.csv")
        agg_pairs = pd.read_csv(paths["results"] / "aggregated_pairs.csv")
        agg_v_by_k = pd.read_csv(paths["results"] / "aggregated_v_by_k.csv")
        agg_clipped_al90 = pd.read_csv(
            paths["results"] / "aggregated_clipped_al90_topk.csv"
        )
    else:
        metrics, pairs = run_monte_carlo(paths["results"], refresh=bool(args.refresh))
        agg_metrics, agg_pairs = aggregate_results(metrics, pairs, paths["results"])
        v_trials = run_v_by_k(paths["results"], refresh=bool(args.refresh))
        agg_v_by_k = aggregate_v_by_k(v_trials, paths["results"])
        clipped_trials = run_clipped_al90(paths["results"], refresh=bool(args.refresh))
        agg_clipped_al90 = aggregate_clipped_al90(clipped_trials, paths["results"])
    fig_paths = write_figures(
        agg_metrics, agg_pairs, agg_v_by_k, agg_clipped_al90, paths["figures"]
    )
    write_docs(agg_metrics, agg_pairs, agg_v_by_k, agg_clipped_al90, EXP_DIR)
    for path in fig_paths:
        print(f"[fig] {path}")
    print("done.")


if __name__ == "__main__":
    main()
