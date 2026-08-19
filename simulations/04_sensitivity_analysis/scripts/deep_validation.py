#!/usr/bin/env python3
"""Analysis 3 — Stability summary and diagnostic report (Experiment 04).

Aggregates across both analyses, generates a stability summary figure,
and writes diagnostic_report.md with finding classification A/B/C.
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.config import (
    LAMBDA_PERTURBATION_VALUES,
    OPERATOR_COLORS,
    OPERATOR_FULL_LABELS,
    OPERATORS,
    SENSITIVITY_PRIMARY_LAMBDA,
    SIGMA_R_VALUES,
)
from common.plotting import save_figure, style_axes_frame
from common.utils import ensure_dirs


def _load_agg(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run run_all.py first.")
    return pd.read_csv(path)


def _stability_summary(agg_lam: pd.DataFrame, agg_score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for op in OPERATORS:
        sub_lam = agg_lam[agg_lam["operator"] == op]
        sub_score = agg_score[agg_score["operator"] == op]

        # λ perturbation stats (exclude reference λ=0.75 from averages since τ=1 there)
        non_ref = sub_lam[~np.isclose(sub_lam["lambda"].to_numpy(), SENSITIVITY_PRIMARY_LAMBDA)]
        mean_tau_lam = float(non_ref["kendall_tau_mean"].mean())
        mean_jac_lam = float(non_ref["jaccard_top_k_mean"].mean())
        min_tau_lam = float(non_ref["kendall_tau_mean"].min())
        min_jac_lam = float(non_ref["jaccard_top_k_mean"].min())

        # score uncertainty stats (exclude σ_R=0 from averages since τ=1 there)
        non_zero = sub_score[~np.isclose(sub_score["sigma_r"].to_numpy(), 0.0)]
        mean_tau_score = float(non_zero["kendall_tau_mean"].mean())
        mean_jac_score = float(non_zero["jaccard_top_k_mean"].mean())
        min_tau_score = float(non_zero["kendall_tau_mean"].min())
        min_jac_score = float(non_zero["jaccard_top_k_mean"].min())

        rows.append({
            "operator": op,
            "mean_tau_lambda": mean_tau_lam,
            "mean_jac_lambda": mean_jac_lam,
            "min_tau_lambda": min_tau_lam,
            "min_jac_lambda": min_jac_lam,
            "mean_tau_score": mean_tau_score,
            "mean_jac_score": mean_jac_score,
            "min_tau_score": min_tau_score,
            "min_jac_score": min_jac_score,
        })
    return pd.DataFrame(rows)


def write_summary_figure(summary: pd.DataFrame, figures_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.0 * 0.85, 4.6 * 0.85))
    ax_lam, ax_score = axes

    x = np.arange(len(OPERATORS))
    w = 0.35
    colors = [OPERATOR_COLORS[op] for op in OPERATORS]
    labels = [OPERATOR_FULL_LABELS[op] for op in OPERATORS]

    for ax, mean_col, min_col, xlabel in (
        (ax_lam, "mean_tau_lambda", "min_tau_lambda", r"$\lambda$ perturbation"),
        (ax_score, "mean_tau_score", "min_tau_score", r"Score noise $\sigma_R$"),
    ):
        means = summary[mean_col].to_numpy()
        mins = summary[min_col].to_numpy()
        bars_mean = ax.bar(x - w / 2, means, w, color=colors, alpha=0.85,
                           label="Mean $\\tau$", edgecolor="white", linewidth=0.5)
        bars_min = ax.bar(x + w / 2, mins, w, color=colors, alpha=0.45,
                          label="Worst-case $\\tau$", edgecolor="white", linewidth=0.5,
                          hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(r"Kendall $\tau$")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.7, 1.02)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        style_axes_frame(ax)

    ax_lam.set_title("(a) $\\lambda$ perturbation")
    ax_score.set_title("(b) Score uncertainty")
    fig.tight_layout()

    written: list[Path] = []
    written.extend(save_figure(fig, figures_dir / "stability_summary", bbox_inches=None))
    return written


def _check_baseline(agg_lam: pd.DataFrame, agg_score: pd.DataFrame) -> list[str]:
    lines = ["## Baseline sanity checks", ""]
    # τ should be ≈ 1 at λ_ref and σ_R=0
    for op in OPERATORS:
        ref_lam = agg_lam[
            (agg_lam["operator"] == op) &
            np.isclose(agg_lam["lambda"].to_numpy(), SENSITIVITY_PRIMARY_LAMBDA)
        ]
        ref_score = agg_score[
            (agg_score["operator"] == op) &
            np.isclose(agg_score["sigma_r"].to_numpy(), 0.0)
        ]
        tau_lam = float(ref_lam["kendall_tau_mean"].iloc[0]) if not ref_lam.empty else float("nan")
        tau_score = float(ref_score["kendall_tau_mean"].iloc[0]) if not ref_score.empty else float("nan")
        jac_lam = float(ref_lam["jaccard_top_k_mean"].iloc[0]) if not ref_lam.empty else float("nan")
        jac_score = float(ref_score["jaccard_top_k_mean"].iloc[0]) if not ref_score.empty else float("nan")
        lines.append(
            f"- {op}: τ at λ_ref={tau_lam:.6f} (expected ≈1.0), "
            f"τ at σ_R=0: {tau_score:.6f} (expected ≈1.0), "
            f"Jaccard at λ_ref={jac_lam:.6f}, Jaccard at σ_R=0: {jac_score:.6f}"
        )
    lines.append("")
    return lines


def build_report(
    agg_lam: pd.DataFrame,
    agg_score: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[str]:
    lines: list[str] = [
        "# Experiment 04 — Diagnostic Report",
        "",
        "Sensitivity analysis. Internal analysis for Section 6.4.",
        "",
        "## 1. Methods recap",
        "",
        "- **Analysis 1** — λ perturbation: λ ∈ {0.70, 0.725, 0.75, 0.775, 0.80},"
        " σ_R=0, baseline population. Reference: P_ref at λ=0.75.",
        "- **Analysis 2** — Score uncertainty: σ_R ∈ {0.00, 0.02, 0.05, 0.10, 0.20},"
        " λ=0.75, baseline population. R'=clip(R+ε,0,1), ε~N(0,σ_R²), all cases.",
        "- Primary metrics: Kendall τ, Top-K Jaccard (vs noiseless reference).",
        "- Secondary metrics: ΔR̄, V (compliance, not primary focus).",
        "- N=1000, K=100, 1000 Monte Carlo replications, seeds 42–1041.",
        "",
    ]

    lines += _check_baseline(agg_lam, agg_score)

    lines += [
        "## 2. Analysis 1 — λ perturbation results",
        "",
        "| λ | Operator | τ | Jaccard |",
        "|---|---|---|---|",
    ]
    for lam in LAMBDA_PERTURBATION_VALUES:
        sub = agg_lam[np.isclose(agg_lam["lambda"].to_numpy(), lam)]
        for op in OPERATORS:
            row = sub[sub["operator"] == op]
            if row.empty:
                continue
            tau = float(row["kendall_tau_mean"].iloc[0])
            jac = float(row["jaccard_top_k_mean"].iloc[0])
            lines.append(f"| {lam:.3f} | {op} | {tau:.4f} | {jac:.4f} |")
    lines.append("")

    lines += [
        "## 3. Analysis 2 — Score uncertainty results",
        "",
        "| σ_R | Operator | τ | Jaccard |",
        "|---|---|---|---|",
    ]
    for sigma_r in SIGMA_R_VALUES:
        sub = agg_score[np.isclose(agg_score["sigma_r"].to_numpy(), sigma_r)]
        for op in OPERATORS:
            row = sub[sub["operator"] == op]
            if row.empty:
                continue
            tau = float(row["kendall_tau_mean"].iloc[0])
            jac = float(row["jaccard_top_k_mean"].iloc[0])
            lines.append(f"| {sigma_r:.2f} | {op} | {tau:.4f} | {jac:.4f} |")
    lines.append("")

    lines += [
        "## 4. Analysis 3 — Stability summary (mean and worst-case τ)",
        "",
        "Averages exclude the reference point (λ=0.75 and σ_R=0) where τ=1 by construction.",
        "",
        "| Operator | Mean τ (λ) | Worst τ (λ) | Mean τ (σ_R) | Worst τ (σ_R) |",
        "|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['operator']} | {row['mean_tau_lambda']:.4f} | {row['min_tau_lambda']:.4f}"
            f" | {row['mean_tau_score']:.4f} | {row['min_tau_score']:.4f} |"
        )
    lines.append("")

    # Research questions
    lines += [
        "## 5. Research questions",
        "",
        "### Q1: Which operators produce more stable rankings?",
        "",
    ]
    best_lam = summary.loc[summary["mean_tau_lambda"].idxmax(), "operator"]
    best_score = summary.loc[summary["mean_tau_score"].idxmax(), "operator"]
    lines += [
        f"Under λ perturbation, highest mean τ: **{best_lam}**.",
        f"Under score uncertainty, highest mean τ: **{best_score}**.",
        "",
        "### Q2: Do compliance-preserving operators (A_G, A_M) sacrifice or improve stability?",
        "",
    ]
    tau_lam_vals = {row["operator"]: row["mean_tau_lambda"] for _, row in summary.iterrows()}
    tau_score_vals = {row["operator"]: row["mean_tau_score"] for _, row in summary.iterrows()}
    ag_vs_al_lam = "higher" if tau_lam_vals.get("geometric", 0) >= tau_lam_vals.get("linear", 0) else "lower"
    am_vs_al_lam = "higher" if tau_lam_vals.get("min", 0) >= tau_lam_vals.get("linear", 0) else "lower"
    ag_vs_al_score = "higher" if tau_score_vals.get("geometric", 0) >= tau_score_vals.get("linear", 0) else "lower"
    am_vs_al_score = "higher" if tau_score_vals.get("min", 0) >= tau_score_vals.get("linear", 0) else "lower"
    lines += [
        f"Under λ perturbation: A_G has {ag_vs_al_lam} τ than A_L; A_M has {am_vs_al_lam} τ than A_L.",
        f"Under score uncertainty: A_G has {ag_vs_al_score} τ than A_L; A_M has {am_vs_al_score} τ than A_L.",
        "",
        "### Q3: Is instability concentrated in ranking positions (τ) or Top-K membership (Jaccard)?",
        "",
    ]
    # Compare τ drop vs Jaccard drop at worst σ_R
    lines += [
        "Compare τ and Jaccard at σ_R=0.20 (most extreme noise):",
        "",
        "| Operator | τ at σ_R=0.20 | Jaccard at σ_R=0.20 |",
        "|---|---|---|",
    ]
    sub_max = agg_score[np.isclose(agg_score["sigma_r"].to_numpy(), max(SIGMA_R_VALUES))]
    for op in OPERATORS:
        row = sub_max[sub_max["operator"] == op]
        if row.empty:
            continue
        tau = float(row["kendall_tau_mean"].iloc[0])
        jac = float(row["jaccard_top_k_mean"].iloc[0])
        lines.append(f"| {op} | {tau:.4f} | {jac:.4f} |")
    lines.append("")
    lines += [
        "If Jaccard drops more than τ, instability is concentrated in Top-K membership.",
        "If τ drops more (relative to 1.0), instability is spread across the full ranking.",
        "",
    ]

    lines += [
        "## 6. Finding classification",
        "",
        "### A — Essential for manuscript narrative",
        "",
        "- Operator ordering by ranking stability under λ perturbation.",
        "- Operator ordering by ranking stability under score uncertainty.",
        "- Whether compliance-preserving operators (A_G, A_M) are more or less stable than A_L.",
        "",
        "### B — Supporting evidence (appendix / brief mention)",
        "",
        "- Worst-case τ and Jaccard at extreme perturbation values.",
        "- Comparison of τ vs Jaccard degradation (ranking positions vs Top-K membership).",
        "- ΔR̄ under both perturbation types.",
        "- Secondary V values (compliance not primary focus).",
        "",
        "### C — Internal diagnostic only",
        "",
        "- Per-trial raw CSVs.",
        "- σ_R × λ interaction (not computed; out of scope for this experiment).",
        "",
        "**Guardrails:**",
        "- Do NOT claim global operator superiority.",
        "- V is secondary information; do not frame §6.4 as a compliance experiment.",
        "- ΔR̄ values reflect noise, not systematic score inflation.",
        "",
        "## 7. Artifacts",
        "",
        "- `results/lambda_perturbation.csv`, `score_uncertainty.csv`",
        "- `results/aggregated_lambda.csv`, `aggregated_score.csv`, `aggregated_stability_summary.csv`",
        "- `figures/lambda_perturbation_stability.pdf`",
        "- `figures/score_uncertainty_stability.pdf`",
        "- `figures/stability_summary.pdf`",
        "- `tables/table_lambda_stability.tex`, `table_score_stability.tex`",
    ]
    return lines


def main(
    agg_lam: pd.DataFrame | None = None,
    agg_score: pd.DataFrame | None = None,
) -> None:
    dirs = ensure_dirs(EXP_DIR)

    if agg_lam is None:
        agg_lam = _load_agg(dirs["results"], "aggregated_lambda.csv")
    if agg_score is None:
        agg_score = _load_agg(dirs["results"], "aggregated_score.csv")

    summary = _stability_summary(agg_lam, agg_score)
    summary.to_csv(dirs["results"] / "aggregated_stability_summary.csv", index=False)

    write_summary_figure(summary, dirs["figures"])

    report_lines = build_report(agg_lam, agg_score, summary)
    report_path = dirs["results"] / "diagnostic_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
