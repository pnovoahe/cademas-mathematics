#!/usr/bin/env python3
"""Experiment 6.2 — Effects of predictive overconfidence.

Illustrates the population-level consequences of a deterministic upward
bias in R on contextually weak cases (manuscript R1V2, Sections 4.4 and
5.2). Does not rank operators as globally better or worse.

Aligned with first_round_v2/manuscriptR1V2.tex, Section 6.2.
Does not draft manuscript prose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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
    CI_LEVEL,
    DELTA_TABLE_VALUES,
    DELTA_VALUES,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATORS,
    OVERCONFIDENCE_LAMBDAS,
    OVERCONFIDENCE_PRIMARY_LAMBDA,
    Q_WEAK_THRESHOLD,
    SEEDS,
    STD_Q_BETA,
    STD_R_BETA,
    TOP_K,
    VETO_FRACTION,
    VETO_R_BETA,
    n_std_from_fraction,
    n_veto_from_fraction,
)
from common.generators import (  # noqa: E402
    apply_predictive_overconfidence,
    generate_population,
)
from common.metrics import (  # noqa: E402
    jaccard_top_k,
    kendall_tau,
    policy_violation_rate,
    top_k_indices,
)
from common.plotting import (  # noqa: E402
    square_single_panel_line_ci,
    square_two_panel_line_ci,
)
from common.utils import (  # noqa: E402
    ensure_dirs,
    experiment_fingerprint,
    format_ci,
    summarize_trials,
    trial_seed,
    utc_now_iso,
    write_json,
    write_latex_table,
)

METRICS = (
    "policy_violation_rate",
    "predictive_utility",
    "kendall_tau",
    "jaccard_top_k",
)
OPERATOR_TEX = {
    "linear": r"$A_L$ (Linear)",
    "geometric": r"$A_G$ (Geometric)",
    "min": r"$A_M$ (Minimum)",
}
OPERATOR_SHORT = {
    "linear": r"$A_L$",
    "geometric": r"$A_G$",
    "min": r"$A_M$",
}


@dataclass(frozen=True)
class RunSettings:
    veto_fraction: float
    n_veto: int
    n_std: int
    q_weak_threshold: float
    refresh: bool


def _settings_from_args(args: argparse.Namespace) -> RunSettings:
    fraction = float(VETO_FRACTION)
    return RunSettings(
        veto_fraction=fraction,
        n_veto=n_veto_from_fraction(fraction, N_CASES),
        n_std=n_std_from_fraction(fraction, N_CASES),
        q_weak_threshold=float(args.q_threshold),
        refresh=bool(args.refresh),
    )


def _fingerprint(settings: RunSettings) -> str:
    return experiment_fingerprint(
        "02_effects_of_predictive_overconfidence",
        {
            "n_cases": N_CASES,
            "veto_fraction": settings.veto_fraction,
            "n_std": settings.n_std,
            "n_veto": settings.n_veto,
            "top_k": TOP_K,
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "delta_values": list(DELTA_VALUES),
            "q_weak_threshold": settings.q_weak_threshold,
            "overconfidence_lambdas": list(OVERCONFIDENCE_LAMBDAS),
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
            "operators": list(OPERATORS),
            "metrics": list(METRICS),
        },
    )


def _run_trial(trial_idx: int, settings: RunSettings) -> list[dict]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(rng, n_std=settings.n_std, n_veto=settings.n_veto)
    records: list[dict] = []
    for lam in OVERCONFIDENCE_LAMBDAS:
        p_ref: dict[str, np.ndarray] = {
            operator: aggregate(operator, pop.R, pop.Q, lam) for operator in OPERATORS
        }
        for delta in DELTA_VALUES:
            r_prime = apply_predictive_overconfidence(
                pop.R, pop.Q, delta, q_threshold=settings.q_weak_threshold
            )
            for operator in OPERATORS:
                p_ref_op = p_ref[operator]
                p = aggregate(operator, r_prime, pop.Q, lam)
                v = policy_violation_rate(
                    p, pop.Q, TOP_K, R=r_prime, case_id=pop.case_id
                )
                idx = top_k_indices(p, TOP_K, R=r_prime, case_id=pop.case_id)
                utility = float(np.mean(pop.R[idx]))
                records.append(
                    {
                        "trial": trial_idx,
                        "seed": trial_seed(trial_idx),
                        "veto_fraction": settings.veto_fraction,
                        "q_weak_threshold": settings.q_weak_threshold,
                        "lambda": lam,
                        "delta": delta,
                        "operator": operator,
                        "policy_violation_rate": v,
                        "predictive_utility": utility,
                        "kendall_tau": kendall_tau(p_ref_op, p),
                        "jaccard_top_k": jaccard_top_k(
                            p_ref_op,
                            p,
                            TOP_K,
                            R_ref=pop.R,
                            R=r_prime,
                            case_id=pop.case_id,
                        ),
                    }
                )
    return records


def run_monte_carlo(settings: RunSettings, results_dir: Path) -> pd.DataFrame:
    raw_path = results_dir / "trials_raw.csv"
    meta_path = results_dir / "run_metadata.json"
    fingerprint = _fingerprint(settings)

    if raw_path.exists() and meta_path.exists() and not settings.refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {raw_path} ({meta.get('n_rows')} rows)")
            return pd.read_csv(raw_path)

    records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        records.extend(_run_trial(trial_idx, settings))
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc] trial {trial_idx + 1}/{N_MONTE_CARLO}")

    raw = pd.DataFrame.from_records(records)
    raw.to_csv(raw_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": "02_effects_of_predictive_overconfidence",
            "manuscript": "first_round_v2/manuscriptR1V2.tex",
            "subsection": "6.2 Effects of Predictive Overconfidence",
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(raw)),
            "n_monte_carlo": N_MONTE_CARLO,
            "seeds": SEEDS,
            "seed_base": MC_SEED_BASE,
            "n_cases": N_CASES,
            "veto_fraction": settings.veto_fraction,
            "n_std": settings.n_std,
            "n_veto": settings.n_veto,
            "top_k": TOP_K,
            "delta_values": list(DELTA_VALUES),
            "delta_table_values": list(DELTA_TABLE_VALUES),
            "q_weak_threshold": settings.q_weak_threshold,
            "overconfidence_lambdas": list(OVERCONFIDENCE_LAMBDAS),
            "primary_lambda": OVERCONFIDENCE_PRIMARY_LAMBDA,
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
            "operators": list(OPERATORS),
            "metrics": list(METRICS),
            "ci_level": CI_LEVEL,
        },
    )
    print(f"[mc] wrote {raw_path} ({len(raw)} rows)")
    return raw


def _order_operators(df: pd.DataFrame, extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    rank = {op: i for i, op in enumerate(OPERATORS)}
    out = df.copy()
    out["_op"] = out["operator"].map(rank)
    sort_cols = list(extra_cols) + ["_op"]
    out = out.sort_values(sort_cols).drop(columns="_op").reset_index(drop=True)
    return out


def _primary(df: pd.DataFrame) -> pd.DataFrame:
    out = df[np.isclose(df["lambda"].to_numpy(), OVERCONFIDENCE_PRIMARY_LAMBDA)].copy()
    return _order_operators(out, extra_cols=("delta",))


def _subset_table_deltas(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(df), dtype=bool)
    for delta in DELTA_TABLE_VALUES:
        mask |= np.isclose(df["delta"].to_numpy(), delta)
    out = df.loc[mask].copy()
    out["delta"] = out["delta"].round(2)
    return _order_operators(out, extra_cols=("delta",))


def _lookup(df: pd.DataFrame, operator: str, delta: float) -> pd.Series:
    sub = df[(df["operator"] == operator) & np.isclose(df["delta"].to_numpy(), delta)]
    if sub.empty:
        raise KeyError(f"No row for operator={operator}, delta={delta}")
    return sub.iloc[0]


def _lookup_lam(df: pd.DataFrame, operator: str, lam: float, delta: float) -> pd.Series:
    sub = df[
        (df["operator"] == operator)
        & np.isclose(df["lambda"].to_numpy(), lam)
        & np.isclose(df["delta"].to_numpy(), delta)
    ]
    if sub.empty:
        raise KeyError(f"No row for operator={operator}, lambda={lam}, delta={delta}")
    return sub.iloc[0]


def _ci(row: pd.Series, metric: str) -> str:
    return format_ci(
        float(row[f"{metric}_mean"]),
        float(row[f"{metric}_ci_low"]),
        float(row[f"{metric}_ci_high"]),
    )


def _ylim_from_ci(
    df: pd.DataFrame,
    lo_col: str,
    hi_col: str,
    *,
    floor: float,
    ceil: float,
    pad: float = 0.08,
) -> tuple[float, float]:
    lo = float(df[lo_col].min())
    hi = float(df[hi_col].max())
    span = max(hi - lo, 0.08)
    return (max(floor, lo - pad * span), min(ceil, hi + pad * span))


def _operators_separate(df: pd.DataFrame, mean_col: str, min_range: float = 0.02) -> bool:
    spreads = df.groupby("delta")[mean_col].agg(lambda s: float(s.max() - s.min()))
    return float(spreads.max()) >= min_range


def _jaccard_redundant_with_tau(df: pd.DataFrame) -> bool:
    """True if Jaccard ranks operators the same way as Kendall τ at max δ."""
    dmax = float(df["delta"].max())
    at_max = df[np.isclose(df["delta"].to_numpy(), dmax)]
    tau_order = (
        at_max.sort_values("kendall_tau_mean")["operator"].tolist()
    )
    jac_order = (
        at_max.sort_values("jaccard_top_k_mean")["operator"].tolist()
    )
    return tau_order == jac_order


def _first_increase(df: pd.DataFrame, operator: str, metric: str, *, atol: float = 0.005) -> float | None:
    sub = df[df["operator"] == operator].sort_values("delta")
    base = float(sub.iloc[0][f"{metric}_mean"])
    for _, row in sub.iloc[1:].iterrows():
        if float(row[f"{metric}_mean"]) > base + atol:
            return float(row["delta"])
    return None


def write_figures(primary_df: pd.DataFrame, figures_dir: Path) -> tuple[list[Path], bool, bool]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    util_ylim = _ylim_from_ci(
        primary_df,
        "predictive_utility_ci_low",
        "predictive_utility_ci_high",
        floor=0.0,
        ceil=1.0,
        pad=0.20,
    )
    written.extend(
        square_two_panel_line_ci(
            primary_df,
            primary_df,
            path_stem=figures_dir / "overconfidence_v_utility",
            x_col="delta",
            xlabel=r"Overconfidence level $\delta$",
            y_col_a="policy_violation_rate_mean",
            ci_low_a="policy_violation_rate_ci_low",
            ci_high_a="policy_violation_rate_ci_high",
            ylabel_a=r"Policy violation rate $V$",
            ylim_a=(-0.08, 1.0),
            y_col_b="predictive_utility_mean",
            ci_low_b="predictive_utility_ci_low",
            ci_high_b="predictive_utility_ci_high",
            ylabel_b=r"Predictive utility $\bar{R}$",
            ylim_b=util_ylim,
            xticks=DELTA_VALUES,
            yticks_a=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
    )
    plot_tau = _operators_separate(primary_df, "kendall_tau_mean")
    if plot_tau:
        tau_ylim = _ylim_from_ci(
            primary_df,
            "kendall_tau_ci_low",
            "kendall_tau_ci_high",
            floor=0.0,
            ceil=1.05,
            pad=0.25,
        )
        written.extend(
            square_single_panel_line_ci(
                primary_df,
                path_stem=figures_dir / "overconfidence_kendall_tau",
                x_col="delta",
                y_col="kendall_tau_mean",
                ci_low_col="kendall_tau_ci_low",
                ci_high_col="kendall_tau_ci_high",
                xlabel=r"Overconfidence level $\delta$",
                ylabel=r"Kendall $\tau$",
                ylim=tau_ylim,
                xticks=DELTA_VALUES,
            )
        )
    plot_jaccard = (
        _operators_separate(primary_df, "jaccard_top_k_mean")
        and not _jaccard_redundant_with_tau(primary_df)
    )
    if plot_jaccard:
        jac_ylim = _ylim_from_ci(
            primary_df,
            "jaccard_top_k_ci_low",
            "jaccard_top_k_ci_high",
            floor=0.0,
            ceil=1.05,
            pad=0.25,
        )
        written.extend(
            square_single_panel_line_ci(
                primary_df,
                path_stem=figures_dir / "overconfidence_jaccard",
                x_col="delta",
                y_col="jaccard_top_k_mean",
                ci_low_col="jaccard_top_k_ci_low",
                ci_high_col="jaccard_top_k_ci_high",
                xlabel=r"Overconfidence level $\delta$",
                ylabel=r"Jaccard Top-$K$",
                ylim=jac_ylim,
                xticks=DELTA_VALUES,
            )
        )
    return written, plot_tau, plot_jaccard


def write_tables(
    primary_df: pd.DataFrame,
    tables_dir: Path,
    settings: RunSettings,
    *,
    plot_tau: bool,
) -> list[Path]:
    table_df = _subset_table_deltas(primary_df)
    table_df["operator_tex"] = table_df["operator"].map(OPERATOR_TEX)
    table_df["delta_tex"] = table_df["delta"].map(lambda x: rf"${x:.2f}$")
    written: list[Path] = []
    path_v = tables_dir / "table_predictive_overconfidence.tex"
    write_latex_table(
        table_df,
        path_v,
        columns=(
            ("delta_tex", r"$\delta$"),
            ("operator_tex", "Operator"),
            ("policy_violation_rate_mean", r"$V$ mean"),
            ("policy_violation_rate_std", r"$V$ std"),
            ("policy_violation_rate_ci_low", r"$V$ CI low"),
            ("policy_violation_rate_ci_high", r"$V$ CI high"),
            ("predictive_utility_mean", r"$\bar{R}$ mean"),
            ("predictive_utility_ci_low", r"$\bar{R}$ CI low"),
            ("predictive_utility_ci_high", r"$\bar{R}$ CI high"),
        ),
        group_column="delta_tex",
        col_spec="llccccccc",
        caption=(
            r"Policy violation rate $V$ and predictive utility $\bar{R}$ (mean original "
            rf"$R$ of the Top-$K$ set) under predictive overconfidence at "
            rf"$\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$. Overconfidence "
            rf"$R'=\mathrm{{clip}}(R+\delta,0,1)$ is applied when "
            rf"$Q_i\le {settings.q_weak_threshold:.2f}$. Values are Monte Carlo "
            rf"means and {int(CI_LEVEL * 100)}\% percentile confidence intervals "
            rf"over {N_MONTE_CARLO} replications ($N={N_CASES}$, $K={TOP_K}$)."
        ),
        label="tab:predictive-overconfidence",
    )
    written.append(path_v)
    if plot_tau:
        path_tau = tables_dir / "table_overconfidence_kendall.tex"
        write_latex_table(
            table_df,
            path_tau,
            columns=(
                ("delta_tex", r"$\delta$"),
                ("operator_tex", "Operator"),
                ("kendall_tau_mean", r"$\tau$ mean"),
                ("kendall_tau_std", r"$\tau$ std"),
                ("kendall_tau_ci_low", r"$\tau$ CI low"),
                ("kendall_tau_ci_high", r"$\tau$ CI high"),
                ("jaccard_top_k_mean", r"Jaccard mean"),
                ("jaccard_top_k_ci_low", r"Jaccard CI low"),
                ("jaccard_top_k_ci_high", r"Jaccard CI high"),
            ),
            group_column="delta_tex",
            col_spec="llccccccc",
            caption=(
                r"Ranking stability relative to the unperturbed ranking ($\delta=0$) "
                rf"under predictive overconfidence at $\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$. "
                r"Kendall $\tau$ compares full score vectors; Jaccard compares Top-$K$ sets. "
                rf"Same Monte Carlo protocol as Table~\ref{{tab:predictive-overconfidence}}."
            ),
            label="tab:overconfidence-kendall",
        )
        written.append(path_tau)
    return written


def _metric_grid(df: pd.DataFrame, metric: str) -> str:
    header = "| $\\delta$ | " + " | ".join(OPERATOR_SHORT[op] for op in OPERATORS) + " |"
    sep = "|---|---:|---:|---:|"
    lines = [header, sep]
    for delta in DELTA_VALUES:
        cells = [f"{delta:.2f}"]
        for operator in OPERATORS:
            row = _lookup(df, operator, delta)
            cells.append(_ci(row, metric))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _peak_row(df: pd.DataFrame, operator: str, metric: str) -> pd.Series:
    sub = df[df["operator"] == operator].sort_values("delta")
    return sub.loc[sub[f"{metric}_mean"].idxmax()]


def _describe_series(df: pd.DataFrame, operator: str, metric: str) -> str:
    sub = df[df["operator"] == operator].sort_values("delta")
    start = sub.iloc[0]
    end = sub.iloc[-1]
    peak = _peak_row(df, operator, metric)
    onset = _first_increase(df, operator, metric)
    parts = [
        f"{OPERATOR_SHORT[operator]}: {_ci(start, metric)} at $\\delta=0.00$; "
        f"{_ci(end, metric)} at $\\delta={float(end['delta']):.2f}$"
    ]
    peak_delta = float(peak["delta"])
    start_mean = float(start[f"{metric}_mean"])
    end_mean = float(end[f"{metric}_mean"])
    peak_mean = float(peak[f"{metric}_mean"])
    interior = (peak_delta > float(start["delta"]) + 1e-12) and (
        peak_delta < float(end["delta"]) - 1e-12
    )
    if interior and peak_mean > max(start_mean, end_mean) + 0.005:
        parts.append(
            f"interior maximum {_ci(peak, metric)} at $\\delta={peak_delta:.2f}$"
        )
    if onset is not None:
        parts.append(f"first mean increase after $\\delta=0$ at $\\delta={onset:.2f}$")
    elif abs(end_mean - start_mean) <= 1e-12:
        parts.append(r"no change relative to $\delta=0$ on this grid")
    elif end_mean < start_mean:
        parts.append(r"mean decreases over the $\delta$ grid")
    return "; ".join(parts) + "."


def _robustness_line(agg_df: pd.DataFrame, lam: float, metric: str) -> str:
    bits = []
    for operator in OPERATORS:
        row0 = _lookup_lam(agg_df, operator, lam, 0.00)
        row1 = _lookup_lam(agg_df, operator, lam, float(DELTA_VALUES[-1]))
        bits.append(
            f"{OPERATOR_SHORT[operator]} {_ci(row0, metric)} → {_ci(row1, metric)}"
        )
    return f"$\\lambda={lam:.2f}$: " + "; ".join(bits) + "."


def write_captions(
    exp_dir: Path,
    settings: RunSettings,
    *,
    plot_tau: bool,
    plot_jaccard: bool,
) -> None:
    lines = [
        "# Captions (Experiment 02)",
        "",
        "Generated from Monte Carlo results. Not manuscript-ready prose.",
        "",
        "## Figure: overconfidence_v_utility (`fig:overconfidence-v-utility`)",
        "",
        (
            f"Policy violation rate $V$ and predictive utility $\\bar{{R}}$ "
            f"(mean original $R$ of the selected Top-$K$) versus overconfidence "
            f"level $\\delta$ at $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$ "
            f"($N={N_CASES}$, $K={TOP_K}$, $q_{{\\mathrm{{weak}}}}="
            f"{settings.q_weak_threshold:.2f}$, {N_MONTE_CARLO} Monte Carlo "
            "replications). Points are means; shaded bands are $95\\%$ percentile "
            "confidence intervals. (\\textbf{a}) $V(\\delta)$. "
            "(\\textbf{b}) $\\bar{R}(\\delta)$."
        ),
        "",
    ]
    if plot_tau:
        lines.extend(
            [
                "## Figure: overconfidence_kendall_tau (`fig:overconfidence-kendall`)",
                "",
                (
                    f"Kendall $\\tau$ between the unperturbed ranking ($\\delta=0$) "
                    f"and the ranking under overconfidence $\\delta$, at "
                    f"$\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$. Same Monte Carlo "
                    "protocol as the two-panel figure."
                ),
                "",
            ]
        )
    if plot_jaccard:
        lines.extend(
            [
                "## Figure: overconfidence_jaccard (`fig:overconfidence-jaccard`)",
                "",
                (
                    f"Jaccard overlap of Top-$K$ sets relative to $\\delta=0$ at "
                    f"$\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$. Plotted because "
                    "the qualitative pattern is not redundant with Kendall $\\tau$."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Table: table_predictive_overconfidence.tex (`tab:predictive-overconfidence`)",
            "",
            (
                f"Monte Carlo mean, standard deviation, and $95\\%$ percentile CI of "
                f"$V$ and $\\bar{{R}}$ at $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$ "
                f"for $\\delta\\in\\{{{', '.join(f'{x:.2f}' for x in DELTA_TABLE_VALUES)}\\}}$."
            ),
            "",
        ]
    )
    if plot_tau:
        lines.extend(
            [
                "## Table: table_overconfidence_kendall.tex (`tab:overconfidence-kendall`)",
                "",
                (
                    f"Kendall $\\tau$ and Jaccard Top-$K$ at the same representative "
                    f"$\\delta$ values and $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$."
                ),
                "",
            ]
        )
    (exp_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    primary_df: pd.DataFrame,
    exp_dir: Path,
    settings: RunSettings,
    *,
    plot_tau: bool,
    plot_jaccard: bool,
) -> None:
    conclusions = [
        "Observed Monte Carlo behaviour (no ranking of operators as better/worse):",
        "",
        "Policy violation rate $V$ at "
        f"$\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$:",
    ]
    for operator in OPERATORS:
        conclusions.append(f"- {_describe_series(primary_df, operator, 'policy_violation_rate')}")
    conclusions.extend(["", r"Predictive utility $\bar{R}$ (original $R$ of Top-$K$):"])
    for operator in OPERATORS:
        conclusions.append(f"- {_describe_series(primary_df, operator, 'predictive_utility')}")
    conclusions.extend(["", r"Kendall $\tau$ vs the $\delta=0$ ranking:"])
    for operator in OPERATORS:
        conclusions.append(f"- {_describe_series(primary_df, operator, 'kendall_tau')}")

    tau_fig = (
        "- `figures/overconfidence_kendall_tau.{pdf,png}` — Kendall $\\tau$ vs $\\delta$"
        if plot_tau
        else "- Kendall $\\tau$ figure omitted (operators do not separate on this grid)"
    )
    jac_fig = (
        "- `figures/overconfidence_jaccard.{pdf,png}` — Jaccard Top-$K$ vs $\\delta$"
        if plot_jaccard
        else "- Jaccard figure omitted (redundant with Kendall $\\tau$, or no separation)"
    )
    tau_table = (
        "- `tables/table_overconfidence_kendall.tex` — $\\tau$ and Jaccard at representative $\\delta$"
        if plot_tau
        else ""
    )

    text = f"""# Experiment 02 — Effects of predictive overconfidence

Manuscript reference: `first_round_v2/manuscriptR1V2.tex`, Section 6.2
(`\\subsection{{Effects of Predictive Overconfidence}}`).

This folder contains the simulation, artifacts, and an LLM-facing report.
It does **not** draft manuscript text.

## Scientific objective

This subsection is **not** a comparison of which operator is better. It
demonstrates the **population-level consequences** of artificial predictive
overconfidence under different aggregation semantics (manuscript Section 4.4,
compensation / rank-reversal thresholds):

- how a systematic upward bias in $R$ on contextually weak cases changes
  Top-$K$ policy violations;
- whether that bias also changes the true predictive quality of the selected
  set (mean **original** $R$);
- how much the ranking moves relative to the unperturbed ranking.

## Overconfidence mechanism

For each Monte Carlo population, scores of contextually weak cases are shifted

$$
R_i' = \\mathrm{{clip}}(R_i + \\delta, 0, 1)
\\qquad\\text{{iff}}\\qquad
Q_i \\le q_{{\\mathrm{{weak}}}}.
$$

$\\delta$ is deterministic (not random). Cases with $Q_i > q_{{\\mathrm{{weak}}}}$
are left unchanged. The default threshold $q_{{\\mathrm{{weak}}}}={settings.q_weak_threshold:.2f}$
includes all vetoes ($Q_i=0$) and standard cases with weak context.

The baseline population is the same generator as Experiment 01.

## Experimental design

For each Monte Carlo replication:

1. Generate a synthetic population of $N$ cases (same as Experiment 01).
2. For each $\\lambda\\in\\{{{', '.join(f'{x:.2f}' for x in OVERCONFIDENCE_LAMBDAS)}\\}}$
   and each operator, store the unperturbed ranking $P_{{\\mathrm{{ref}}}}=A(R,Q)$.
3. For each $\\delta$, form $R'$ and $P=A(R',Q)$.
4. Select Top-$K$ with ties $(-P,-R',\\mathrm{{case\\_id}})$.
5. Compute $V$ on $\\{{Q_i=0\\}}$, $\\bar{{R}}$ as the mean **original** $R$ of
   the selected set, Kendall $\\tau(P_{{\\mathrm{{ref}}}},P)$, and Jaccard Top-$K$.

Primary figures use $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$. The other
$\\lambda$ values are stored for robustness checks.

## Parameters

All defaults live in `simulations/common/config.py`.

| Parameter | Value |
|---|---|
| $N$ | {N_CASES} |
| `veto_fraction` | {settings.veto_fraction:.2f} |
| Standard cases | {settings.n_std} |
| Veto cases ($Q_i=0$) | {settings.n_veto} |
| Top-$K$ | {TOP_K} |
| $q_{{\\mathrm{{weak}}}}$ | {settings.q_weak_threshold:.2f} |
| $\\delta$ grid | {DELTA_VALUES} |
| $\\lambda$ (stored) | {OVERCONFIDENCE_LAMBDAS} |
| $\\lambda$ (primary figures) | {OVERCONFIDENCE_PRIMARY_LAMBDA:.2f} |
| $R$ (standard) | Beta{STD_R_BETA} |
| $Q$ (standard) | Beta{STD_Q_BETA} |
| $R$ (veto, adversarial) | Beta{VETO_R_BETA} |
| Monte Carlo replications | {N_MONTE_CARLO} |
| Seeds | {MC_SEED_BASE} … {MC_SEED_BASE + N_MONTE_CARLO - 1} |
| Confidence intervals | {int(CI_LEVEL * 100)}% percentiles |

## Metrics

- **Policy violation rate** $V=|\\mathcal{{T}}_K\\cap\\mathcal{{V}}|/K$, $\\mathcal{{V}}=\\{{i:Q_i=0\\}}$.
- **Predictive utility** $\\bar{{R}}$ = mean original $R_i$ of the Top-$K$ selected from $P=A(R',Q)$.
- **Kendall $\\tau$** between $P_{{\\mathrm{{ref}}}}$ and $P$ (same trial, operator, $\\lambda$).
- **Jaccard Top-$K$** between the $\\delta=0$ set and the perturbed set.

## How to run

```bash
pip install -r simulations/requirements.txt
cd simulations/02_effects_of_predictive_overconfidence
python run.py
python run.py --refresh
python run.py --q-threshold 0.25
```

Monte Carlo trials are cached in `results/trials_raw.csv` keyed by a
fingerprint of the scientific parameters. Re-running without `--refresh`
rebuilds figures, tables, and documentation from the cache when the
fingerprint matches.

## Generated outputs

- `results/trials_raw.csv` — one row per (trial, $\\lambda$, $\\delta$, operator)
- `results/aggregated.csv` — mean, std, CI95% over all stored $\\lambda$
- `results/aggregated_primary.csv` — subset $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$
- `results/run_metadata.json` — seeds, threshold, fingerprint
- `figures/overconfidence_v_utility.{{pdf,png}}` — primary two-panel figure
{tau_fig}
{jac_fig}
- `tables/table_predictive_overconfidence.tex` — $V$ and $\\bar{{R}}$
{tau_table}
- `captions.md`
- `results_narrative.md`

## Main conclusions from the results

{chr(10).join(conclusions)}
"""
    (exp_dir / "README.md").write_text(text, encoding="utf-8")


def write_narrative(
    agg_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    exp_dir: Path,
    settings: RunSettings,
    *,
    plot_tau: bool,
    plot_jaccard: bool,
) -> None:
    dmax = float(DELTA_VALUES[-1])
    paper_bits = [
        "Primary paper figure: `figures/overconfidence_v_utility.pdf` (two square",
        "panels: (a) $V$ vs $\\delta$, (b) $\\bar{R}$ vs $\\delta$) at",
        f"$\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$.",
    ]
    if plot_tau:
        paper_bits.append(
            "Optional paper figure: `figures/overconfidence_kendall_tau.pdf` "
            "(operators separate on Kendall $\\tau$)."
        )
    else:
        paper_bits.append(
            "Do **not** include a Kendall figure: operators do not separate enough "
            "on this grid."
        )
    if plot_jaccard:
        paper_bits.append(
            "Optional paper figure: `figures/overconfidence_jaccard.pdf` "
            "(Jaccard is not redundant with $\\tau$)."
        )
    else:
        paper_bits.append(
            "Do **not** include a Jaccard figure: it is redundant with Kendall "
            "$\\tau$, or operators do not separate."
        )
    paper_bits.append(
        "Do **not** plot $\\lambda=0.50$ or $\\lambda=0.90$ by default; they are "
        "robustness checks in `results/aggregated.csv`."
    )

    def series_block(metric: str, title: str) -> str:
        lines = [f"### {title}", "", _metric_grid(primary_df, metric), ""]
        for operator in OPERATORS:
            lines.append(f"- {_describe_series(primary_df, operator, metric)}")
        return "\n".join(lines)

    robustness_v = "\n".join(
        f"- {_robustness_line(agg_df, lam, 'policy_violation_rate')}"
        for lam in OVERCONFIDENCE_LAMBDAS
    )
    robustness_r = "\n".join(
        f"- {_robustness_line(agg_df, lam, 'predictive_utility')}"
        for lam in OVERCONFIDENCE_LAMBDAS
    )
    robustness_tau = "\n".join(
        f"- {_robustness_line(agg_df, lam, 'kendall_tau')}"
        for lam in OVERCONFIDENCE_LAMBDAS
    )

    onset_txt = []
    for op in OPERATORS:
        onset_txt.append(f"- {_describe_series(primary_df, op, 'policy_violation_rate')}")

    robustness_notes = []
    for lam in OVERCONFIDENCE_LAMBDAS:
        row0 = _lookup_lam(agg_df, "linear", lam, 0.00)
        row_end = _lookup_lam(agg_df, "linear", lam, dmax)
        sub = agg_df[
            (agg_df["operator"] == "linear") & np.isclose(agg_df["lambda"].to_numpy(), lam)
        ].sort_values("delta")
        peak = sub.loc[sub["policy_violation_rate_mean"].idxmax()]
        v0 = float(row0["policy_violation_rate_mean"])
        v1 = float(row_end["policy_violation_rate_mean"])
        vp = float(peak["policy_violation_rate_mean"])
        dp = float(peak["delta"])
        if v0 <= 1e-12 and v1 <= 1e-12 and vp <= 1e-12:
            robustness_notes.append(
                f"- $\\lambda={lam:.2f}$: linear mean $V$ remains $0$ over the whole "
                r"$\delta$ grid."
            )
        elif dp + 1e-12 < dmax and vp > max(v0, v1) + 1e-12:
            robustness_notes.append(
                f"- $\\lambda={lam:.2f}$: linear mean $V$ is non-monotonic, with an "
                f"interior maximum {_ci(peak, 'policy_violation_rate')} at "
                f"$\\delta={dp:.2f}$ (ends at {_ci(row_end, 'policy_violation_rate')})."
            )
        elif v1 > v0 + 1e-12:
            robustness_notes.append(
                f"- $\\lambda={lam:.2f}$: linear mean $V$ increases from "
                f"{_ci(row0, 'policy_violation_rate')} to "
                f"{_ci(row_end, 'policy_violation_rate')}."
            )
        else:
            robustness_notes.append(
                f"- $\\lambda={lam:.2f}$: linear mean $V$ goes from "
                f"{_ci(row0, 'policy_violation_rate')} to "
                f"{_ci(row_end, 'policy_violation_rate')}."
            )

    text = f"""# Narrative summary — Experiment 6.2 Effects of predictive overconfidence

This file is a **text-only briefing** for humans and LLM agents working on
`first_round_v2/manuscriptR1V2.tex` Section 6.2. Do not invent numerical claims.
Do not describe any operator as globally superior. Do not assume the reader
can see the figures.

- Code: `simulations/02_effects_of_predictive_overconfidence/run.py`
- Config: `simulations/common/config.py`
- Seeds: {MC_SEED_BASE} through {MC_SEED_BASE + N_MONTE_CARLO - 1}

## Scientific message

Section 6.2 shows the **population-level consequences** of a systematic
upward bias in predictive scores on contextually weak alternatives. The
question is how increasing $\\delta$ changes Top-$K$ composition under each
aggregation semantics, not which operator is better.

The mechanism is $R_i'=\\mathrm{{clip}}(R_i+\\delta,0,1)$ applied only when
$Q_i\\le q_{{\\mathrm{{weak}}}}={settings.q_weak_threshold:.2f}$. Predictive
utility uses the **original** $R$ of the selected set, so an increase in
$\\bar{{R}}$ is not automatic from inflating $R'$.

## Design (this run)

| Parameter | Value |
|---|---|
| $N$ | {N_CASES} |
| `veto_fraction` | {settings.veto_fraction:.2f} |
| Veto cases | {settings.n_veto} |
| Standard cases | {settings.n_std} |
| $K$ | {TOP_K} |
| $q_{{\\mathrm{{weak}}}}$ | {settings.q_weak_threshold:.2f} |
| $\\delta$ | {DELTA_VALUES} |
| $\\lambda$ stored | {OVERCONFIDENCE_LAMBDAS} |
| $\\lambda$ primary | {OVERCONFIDENCE_PRIMARY_LAMBDA:.2f} |
| Monte Carlo | {N_MONTE_CARLO} |

## Headline results ($\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$)

At $\\delta=0.00$ (no overconfidence):
- {OPERATOR_SHORT['linear']}: $V={_ci(_lookup(primary_df, 'linear', 0.0), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'linear', 0.0), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'linear', 0.0), 'kendall_tau')}$.
- {OPERATOR_SHORT['geometric']}: $V={_ci(_lookup(primary_df, 'geometric', 0.0), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'geometric', 0.0), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'geometric', 0.0), 'kendall_tau')}$.
- {OPERATOR_SHORT['min']}: $V={_ci(_lookup(primary_df, 'min', 0.0), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'min', 0.0), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'min', 0.0), 'kendall_tau')}$.

At $\\delta={dmax:.2f}$ (largest overconfidence on this grid):
- {OPERATOR_SHORT['linear']}: $V={_ci(_lookup(primary_df, 'linear', dmax), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'linear', dmax), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'linear', dmax), 'kendall_tau')}$.
- {OPERATOR_SHORT['geometric']}: $V={_ci(_lookup(primary_df, 'geometric', dmax), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'geometric', dmax), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'geometric', dmax), 'kendall_tau')}$.
- {OPERATOR_SHORT['min']}: $V={_ci(_lookup(primary_df, 'min', dmax), 'policy_violation_rate')}$; $\\bar{{R}}={_ci(_lookup(primary_df, 'min', dmax), 'predictive_utility')}$; $\\tau={_ci(_lookup(primary_df, 'min', dmax), 'kendall_tau')}$.

### Notable transitions in $V$

{chr(10).join(onset_txt)}

## Figure descriptions

### Primary two-panel: `figures/overconfidence_v_utility.pdf`

Shared x-axis: overconfidence level $\\delta\\in[{DELTA_VALUES[0]:.2f},{dmax:.2f}]$
with ticks at {', '.join(f'{x:.2f}' for x in DELTA_VALUES)}.
Three series in both panels, drawn Linear → Geometric → Min so overlaps remain
visible: {OPERATOR_SHORT['linear']} blue downward triangles (largest markers),
{OPERATOR_SHORT['geometric']} orange squares (medium), {OPERATOR_SHORT['min']}
green circles (smallest). White marker edges. Shaded $95\\%$ percentile bands.
Legends in the upper-left corner of each square axes box. Panel letters (a)
and (b) above the boxes.

**(a) Policy violation rate $V$.** Vertical axis from just below 0 to 1
(ticks $0.0,0.2,\\ldots,1.0$) so series at $V=0$ sit slightly above the bottom
spine. {_describe_series(primary_df, 'linear', 'policy_violation_rate')}
{_describe_series(primary_df, 'geometric', 'policy_violation_rate')}
{_describe_series(primary_df, 'min', 'policy_violation_rate')}
If two series lie on the bottom axis they overlap: the green circle sits on
the orange square, which sits on the blue triangle.

**(b) Predictive utility $\\bar{{R}}$.** Vertical axis is the mean original
$R$ of the Top-$K$ set (not $R'$). {_describe_series(primary_df, 'linear', 'predictive_utility')}
{_describe_series(primary_df, 'geometric', 'predictive_utility')}
{_describe_series(primary_df, 'min', 'predictive_utility')}

{'' if not plot_tau else f'''### Kendall $\\tau$: `figures/overconfidence_kendall_tau.pdf`

Single square panel, same x-axis, markers, colours, and upper-left legend as
panel (a). Y-axis is Kendall $\\tau$ between $P_{{\\mathrm{{ref}}}}$ ($\\delta=0$)
and $P(\\delta)$. At $\\delta=0$ every operator is $\\tau=1$ by construction.
{_describe_series(primary_df, 'linear', 'kendall_tau')}
{_describe_series(primary_df, 'geometric', 'kendall_tau')}
{_describe_series(primary_df, 'min', 'kendall_tau')}
'''}
{'' if not plot_jaccard else f'''### Jaccard Top-$K$: `figures/overconfidence_jaccard.pdf`

Same layout as the Kendall figure. {_describe_series(primary_df, 'linear', 'jaccard_top_k')}
{_describe_series(primary_df, 'geometric', 'jaccard_top_k')}
{_describe_series(primary_df, 'min', 'jaccard_top_k')}
'''}

## Table walkthrough

`tables/table_predictive_overconfidence.tex` groups rows by $\\delta$
({', '.join(f'{x:.2f}' for x in DELTA_TABLE_VALUES)}); $\\delta$ is printed
once per operator triplet. Columns: $V$ mean/std/CI and $\\bar{{R}}$ mean/CI
at $\\lambda={OVERCONFIDENCE_PRIMARY_LAMBDA:.2f}$.

{series_block('policy_violation_rate', 'Full $V$ grid (primary $\\lambda$)')}

{series_block('predictive_utility', r'Full $\bar{R}$ grid (primary $\lambda$)')}

{series_block('kendall_tau', r'Full Kendall $\tau$ grid (primary $\lambda$)')}

{series_block('jaccard_top_k', 'Full Jaccard Top-$K$ grid (primary $\\lambda$)')}

## Robustness: other $\\lambda$ ($V$ from $\\delta=0$ to $\\delta={dmax:.2f}$)

{robustness_v}

Predictive utility:

{robustness_r}

Kendall $\\tau$:

{robustness_tau}

Linear $V$ shape by $\\lambda$ (observed, not extrapolated):

{chr(10).join(robustness_notes)}

## Paper recommendations

{chr(10).join(paper_bits)}

Table for the paper: `tables/table_predictive_overconfidence.tex`. Include the
Kendall/Jaccard table only if the Kendall figure is used.

## Interpretation limits

All statements above are Monte Carlo summaries for this synthetic design
($N={N_CASES}$, $K={TOP_K}$, veto fraction {settings.veto_fraction:.2f},
$q_{{\\mathrm{{weak}}}}={settings.q_weak_threshold:.2f}$, adversarial veto
$R\\sim\\mathrm{{Beta}}{VETO_R_BETA}$). They do not establish that any
operator is globally preferable, and they do not transfer automatically to
other populations or other $q_{{\\mathrm{{weak}}}}$ values.
"""
    (exp_dir / "results_narrative.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute Monte Carlo trials even if a matching cache exists.",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=Q_WEAK_THRESHOLD,
        help=(
            "Apply overconfidence when Q_i is at most this value "
            f"(default {Q_WEAK_THRESHOLD:.2f})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _settings_from_args(args)
    paths = ensure_dirs(EXP_DIR)
    raw = run_monte_carlo(settings, results_dir=paths["results"])
    agg = summarize_trials(raw, ["lambda", "delta", "operator"], METRICS)
    agg = _order_operators(agg, extra_cols=("lambda", "delta"))
    agg_path = paths["results"] / "aggregated.csv"
    agg.to_csv(agg_path, index=False)
    primary = _primary(agg)
    primary_path = paths["results"] / "aggregated_primary.csv"
    primary.to_csv(primary_path, index=False)
    print(f"[agg] wrote {agg_path} ({len(agg)} rows)")
    print(f"[agg] wrote {primary_path} ({len(primary)} rows)")

    figures, plot_tau, plot_jaccard = write_figures(primary, paths["figures"])
    for path in figures:
        print(f"[fig] {path}")
    for path in write_tables(primary, paths["tables"], settings, plot_tau=plot_tau):
        print(f"[tex] {path}")
    write_captions(EXP_DIR, settings, plot_tau=plot_tau, plot_jaccard=plot_jaccard)
    write_readme(
        primary, EXP_DIR, settings, plot_tau=plot_tau, plot_jaccard=plot_jaccard
    )
    write_narrative(
        agg,
        primary,
        EXP_DIR,
        settings,
        plot_tau=plot_tau,
        plot_jaccard=plot_jaccard,
    )
    print("[doc] captions.md, README.md, results_narrative.md")
    print("done.")


if __name__ == "__main__":
    main()
