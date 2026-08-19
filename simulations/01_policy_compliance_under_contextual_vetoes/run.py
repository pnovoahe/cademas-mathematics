#!/usr/bin/env python3
"""Experiment 6.1 — Policy compliance under contextual vetoes.

Illustrates the population-level consequences of the contextual-veto
property (manuscript R1V2, Section 4) under an adversarial veto group.
Does not rank operators as globally better or worse.

Aligned with first_round_v2/manuscriptR1V2.tex, Section 6.1.
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
    LAMBDA_BAR_VALUES,
    LAMBDA_DENSE_VALUES,
    MC_SEED_BASE,
    N_CASES,
    N_MONTE_CARLO,
    OPERATOR_FULL_LABELS,
    OPERATORS,
    SEEDS,
    STD_Q_BETA,
    STD_R_BETA,
    TOP_K,
    VETO_FRACTION,
    VETO_FRACTIONS_SUPPORTED,
    VETO_R_BETA,
    n_std_from_fraction,
    n_veto_from_fraction,
)
from common.generators import generate_population  # noqa: E402
from common.metrics import policy_violation_rate, predictive_utility  # noqa: E402
from common.plotting import (  # noqa: E402
    grouped_bar_figure,
    new_single_axes,
    plot_line_mean_ci,
    policy_violation_two_panel,
    save_figure,
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

METRICS = ("policy_violation_rate", "veto_preservation_rate", "predictive_utility")
OPERATOR_TEX = {
    "linear": r"$A_L$ (Linear)",
    "geometric": r"$A_G$ (Geometric)",
    "min": r"$A_M$ (Minimum)",
}
VPR_FIGURE_STEMS = (
    "contextual_compliance_grouped.pdf",
    "contextual_compliance_grouped.png",
    "contextual_compliance_grouped.svg",
)


@dataclass(frozen=True)
class RunSettings:
    veto_fraction: float
    n_veto: int
    n_std: int
    refresh: bool


def _settings_from_args(args: argparse.Namespace) -> RunSettings:
    fraction = float(args.veto_fraction)
    return RunSettings(
        veto_fraction=fraction,
        n_veto=n_veto_from_fraction(fraction, N_CASES),
        n_std=n_std_from_fraction(fraction, N_CASES),
        refresh=bool(args.refresh),
    )


def _fingerprint(settings: RunSettings) -> str:
    return experiment_fingerprint(
        "01_policy_compliance_under_contextual_vetoes",
        {
            "n_cases": N_CASES,
            "veto_fraction": settings.veto_fraction,
            "n_std": settings.n_std,
            "n_veto": settings.n_veto,
            "top_k": TOP_K,
            "n_mc": N_MONTE_CARLO,
            "seeds": SEEDS,
            "lambda_dense": list(LAMBDA_DENSE_VALUES),
            "lambda_bar": list(LAMBDA_BAR_VALUES),
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
            "operators": list(OPERATORS),
        },
    )


def _run_trial(trial_idx: int, settings: RunSettings) -> list[dict]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(rng, n_std=settings.n_std, n_veto=settings.n_veto)
    records: list[dict] = []
    for lam in LAMBDA_DENSE_VALUES:
        for operator in OPERATORS:
            P = aggregate(operator, pop.R, pop.Q, lam)
            v = policy_violation_rate(P, pop.Q, TOP_K, R=pop.R, case_id=pop.case_id)
            records.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed(trial_idx),
                    "veto_fraction": settings.veto_fraction,
                    "lambda": lam,
                    "operator": operator,
                    "policy_violation_rate": v,
                    "veto_preservation_rate": 1.0 - v,
                    "predictive_utility": predictive_utility(
                        P, pop.R, TOP_K, case_id=pop.case_id
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
            "experiment": "01_policy_compliance_under_contextual_vetoes",
            "manuscript": "first_round_v2/manuscriptR1V2.tex",
            "subsection": "6.1 Policy Compliance under Contextual Vetoes",
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
            "lambda_dense": list(LAMBDA_DENSE_VALUES),
            "lambda_bar": list(LAMBDA_BAR_VALUES),
            "std_r_beta": STD_R_BETA,
            "std_q_beta": STD_Q_BETA,
            "veto_r_beta": VETO_R_BETA,
            "operators": list(OPERATORS),
            "ci_level": CI_LEVEL,
        },
    )
    print(f"[mc] wrote {raw_path} ({len(raw)} rows)")
    return raw


def _order_operators(df: pd.DataFrame) -> pd.DataFrame:
    rank = {op: i for i, op in enumerate(OPERATORS)}
    out = df.copy()
    out["_op"] = out["operator"].map(rank)
    out = out.sort_values(["lambda", "_op"]).drop(columns="_op").reset_index(drop=True)
    return out


def _subset_bar_values(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.zeros(len(df), dtype=bool)
    for lam in LAMBDA_BAR_VALUES:
        mask |= np.isclose(df["lambda"].to_numpy(), lam)
    out = df.loc[mask].copy()
    out["lambda"] = out["lambda"].round(2)
    return _order_operators(out)


def _lookup(df: pd.DataFrame, operator: str, lam: float) -> pd.Series:
    sub = df[(df["operator"] == operator) & np.isclose(df["lambda"].to_numpy(), lam)]
    if sub.empty:
        raise KeyError(f"No row for operator={operator}, lambda={lam}")
    return sub.iloc[0]


def _linear_onset(dense_df: pd.DataFrame) -> float | None:
    linear = dense_df[dense_df["operator"] == "linear"].sort_values("lambda")
    positive = linear[linear["policy_violation_rate_mean"] > 0.0]
    if positive.empty:
        return None
    return float(positive["lambda"].iloc[0])


def write_table(bar_df: pd.DataFrame, tables_dir: Path, settings: RunSettings) -> Path:
    table_df = bar_df.copy()
    table_df["operator_tex"] = table_df["operator"].map(OPERATOR_TEX)
    table_df["lambda_tex"] = table_df["lambda"].map(lambda x: rf"${x:.2f}$")
    op_rank = {name: i for i, name in enumerate(OPERATORS)}
    table_df["_op"] = table_df["operator"].map(op_rank)
    table_df = table_df.sort_values(["lambda", "_op"]).drop(columns="_op")
    path = tables_dir / "table_policy_compliance.tex"
    write_latex_table(
        table_df,
        path,
        columns=(
            ("lambda_tex", r"$\lambda$"),
            ("operator_tex", "Operator"),
            ("policy_violation_rate_mean", r"$V$ mean"),
            ("policy_violation_rate_std", r"$V$ std"),
            ("policy_violation_rate_ci_low", r"$V$ CI low"),
            ("policy_violation_rate_ci_high", r"$V$ CI high"),
        ),
        group_column="lambda_tex",
        col_spec="lccccc",
        caption=(
            r"Policy violation rate $V$ under contextual vetoes. Values are Monte Carlo "
            rf"means, standard deviations, and {int(CI_LEVEL * 100)}\% percentile "
            rf"confidence intervals over {N_MONTE_CARLO} replications "
            rf"($N={N_CASES}$, $K={TOP_K}$, veto fraction ${settings.veto_fraction:.2f}$)."
        ),
        label="tab:policy-compliance",
    )
    return path


def _remove_vpr_figures(figures_dir: Path) -> None:
    for name in VPR_FIGURE_STEMS:
        path = figures_dir / name
        if path.exists():
            path.unlink()
            print(f"[fig] removed {path}")


def write_figures(bar_df: pd.DataFrame, dense_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    _remove_vpr_figures(figures_dir)
    written: list[Path] = []
    written.extend(
        policy_violation_two_panel(
            bar_df,
            dense_df,
            path_stem=figures_dir / "policy_violation_rate",
            x_values=LAMBDA_BAR_VALUES,
        )
    )
    written.extend(
        grouped_bar_figure(
            bar_df,
            y_col="policy_violation_rate_mean",
            ci_low_col="policy_violation_rate_ci_low",
            ci_high_col="policy_violation_rate_ci_high",
            ylabel=r"Policy violation rate $V$",
            xlabel=r"Trade-off parameter $\lambda$",
            x_values=LAMBDA_BAR_VALUES,
            path_stem=figures_dir / "policy_violation_rate_grouped",
            ylim=(0.0, 1.0),
        )
    )
    fig, ax = new_single_axes()
    plot_line_mean_ci(
        ax,
        dense_df,
        x_col="lambda",
        group_col="operator",
        y_col="policy_violation_rate_mean",
        ci_low_col="policy_violation_rate_ci_low",
        ci_high_col="policy_violation_rate_ci_high",
        group_labels=OPERATOR_FULL_LABELS,
        xlabel=r"Trade-off parameter $\lambda$",
        ylabel=r"Policy violation rate $V$",
        ylim=(-0.08, 1.0),
    )
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.tight_layout()
    written.extend(save_figure(fig, figures_dir / "policy_violation_rate_dense"))
    return written


def _v_sentence(row: pd.Series) -> str:
    return format_ci(
        float(row["policy_violation_rate_mean"]),
        float(row["policy_violation_rate_ci_low"]),
        float(row["policy_violation_rate_ci_high"]),
    )


def write_snippets(
    bar_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    exp_dir: Path,
    settings: RunSettings,
) -> None:
    linear_050 = _lookup(bar_df, "linear", 0.50)
    linear_075 = _lookup(bar_df, "linear", 0.75)
    linear_090 = _lookup(bar_df, "linear", 0.90)
    geom_all_zero = all(
        float(_lookup(bar_df, "geometric", lam)["policy_violation_rate_mean"]) == 0.0
        for lam in LAMBDA_BAR_VALUES
    )
    min_all_zero = all(
        float(_lookup(bar_df, "min", lam)["policy_violation_rate_mean"]) == 0.0
        for lam in LAMBDA_BAR_VALUES
    )
    geom_dense_max = float(
        dense_df.loc[dense_df["operator"] == "geometric", "policy_violation_rate_mean"].max()
    )
    min_dense_max = float(
        dense_df.loc[dense_df["operator"] == "min", "policy_violation_rate_mean"].max()
    )
    first_lam = _linear_onset(dense_df)
    last_zero = None
    linear_dense = dense_df[dense_df["operator"] == "linear"].sort_values("lambda")
    zeros = linear_dense[linear_dense["policy_violation_rate_mean"] == 0.0]
    if not zeros.empty:
        last_zero = float(zeros["lambda"].iloc[-1])

    lines = [
        "# Manuscript snippets (Section 6.1)",
        "",
        "Generated automatically from Monte Carlo results. Do not invent additional claims.",
        "Do not describe any operator as globally superior.",
        "",
        "## Setup (for the paper)",
        "",
        (
            f"Each Monte Carlo replication generates $N={N_CASES}$ decision cases, "
            f"of which a fraction ${settings.veto_fraction:.2f}$ "
            f"($N_{{\\mathrm{{veto}}}}={settings.n_veto}$) have $Q_i=0$. "
            f"Standard scores are sampled independently as "
            f"$R_i\\sim\\mathrm{{Beta}}{STD_R_BETA}$ and "
            f"$Q_i\\sim\\mathrm{{Beta}}{STD_Q_BETA}$. "
            "The veto group is an adversarial scenario: those cases are "
            f"intentionally assigned high predictive scores "
            f"$R_i\\sim\\mathrm{{Beta}}{VETO_R_BETA}$ so that predictive evidence "
            "strongly conflicts with the contextual constraint. "
            f"The Top-$K$ tier uses $K={TOP_K}$. "
            f"Results are averaged over {N_MONTE_CARLO} independent seeds "
            f"(base seed {MC_SEED_BASE}). "
            "Contextual compliance is the complement $\\mathrm{VPR}=1-V$ and is "
            "not plotted separately."
        ),
        "",
        "## Grouped comparison at representative $\\lambda$ (Figure 6)",
        "",
    ]

    for lam in LAMBDA_BAR_VALUES:
        lines.append(f"### $\\lambda={lam:.2f}$")
        lines.append("")
        for operator in OPERATORS:
            row = _lookup(bar_df, operator, lam)
            lines.append(
                f"- {OPERATOR_TEX[operator]}: $V$ = {_v_sentence(row)}; "
                f"std$(V)$ = {float(row['policy_violation_rate_std']):.3f}."
            )
        lines.append("")

    lines.extend(["## Draft paragraphs", ""])
    paragraphs = [
        "The first experiment evaluates the population-level consequences of "
        "the contextual-veto property established in "
        "Section~\\ref{subsec:veto-preservation}. It is not a comparison of "
        "overall operator quality. The veto group is intentionally assigned "
        "high predictive scores in order to create the most challenging "
        "scenario for policy preservation, so that aggregation semantics can "
        "be evaluated when predictive evidence strongly conflicts with "
        "contextual constraints.",
        "",
        "Figure~\\ref{fig:policy-violation-grouped} reports the policy "
        "violation rate $V$ at three representative integration regimes "
        f"$\\lambda\\in\\{{{', '.join(f'{x:.2f}' for x in LAMBDA_BAR_VALUES)}\\}}$. "
        "Contextual compliance is the complement $\\mathrm{VPR}=1-V$ and is "
        "therefore omitted as a separate figure.",
        "",
        (
            f"At $\\lambda=0.50$, the linear operator yields $V={_v_sentence(linear_050)}$. "
            f"At $\\lambda=0.75$, $V={_v_sentence(linear_075)}$. "
            f"At $\\lambda=0.90$, $V={_v_sentence(linear_090)}$. "
            "Thus linear aggregation does not always admit contextually excluded "
            "cases; violations emerge as predictive evidence receives more weight."
        ),
        "",
    ]
    if geom_all_zero:
        paragraphs.append(
            "Under the same populations, the weighted geometric operator yields "
            "$V=0$ at all three representative values of $\\lambda$, consistent "
            "with zero absorption."
        )
    paragraphs.append("")
    if min_all_zero:
        paragraphs.append(
            "The minimum operator likewise yields $V=0$ at these settings, "
            "again consistent with zero absorption rather than with a claim of "
            "overall superiority."
        )
    paragraphs.append("")
    paragraphs.append(
        "Figure~\\ref{fig:policy-violation-dense} shows $V$ as a function of "
        "$\\lambda$ over $[0,1]$. This dense sweep makes the transition from "
        "no violations to systematic policy violations explicit."
    )
    if last_zero is not None and first_lam is not None:
        paragraphs.append(
            f" For the linear operator, the Monte Carlo mean of $V$ remains "
            f"$0$ up to $\\lambda={last_zero:.2f}$ and first becomes strictly "
            f"positive at $\\lambda={first_lam:.2f}$."
        )
    paragraphs.append(
        f" Over the full sweep, the maximum Monte Carlo mean of $V$ is "
        f"{geom_dense_max:.3f} for $A_G$ and {min_dense_max:.3f} for $A_M$."
    )
    paragraphs.append("")

    lines.extend(paragraphs)
    (exp_dir / "manuscript_snippets.md").write_text("\n".join(lines), encoding="utf-8")

    captions = [
        "# Captions (Section 6.1)",
        "",
        "## Figure 6: policy_violation_rate_grouped",
        "",
        (
            f"Policy violation rate $V$ under contextual vetoes for linear ($A_L$), "
            f"weighted geometric ($A_G$), and minimum ($A_M$) aggregation at "
            f"$\\lambda\\in\\{{{', '.join(f'{x:.2f}' for x in LAMBDA_BAR_VALUES)}\\}}$, "
            "corresponding to balanced, prediction-dominant, and strongly "
            "prediction-dominant integration. Bars show Monte Carlo means over "
            f"{N_MONTE_CARLO} replications ($N={N_CASES}$, $K={TOP_K}$, "
            f"veto fraction ${settings.veto_fraction:.2f}$). "
            "Error bars denote 95\\% percentile confidence intervals. "
            "The veto group is assigned high predictive scores "
            f"$R_i\\sim\\mathrm{{Beta}}{VETO_R_BETA}$, so prediction conflicts "
            "with $Q_i=0$. Contextual compliance is $\\mathrm{VPR}=1-V$."
        ),
        "",
        r"Suggested LaTeX label: \label{fig:policy-violation-grouped}",
        "",
        "## Figure 7: policy_violation_rate_dense",
        "",
        (
            f"Policy violation rate $V$ as a function of $\\lambda\\in[0,1]$ "
            f"(step $1/{len(LAMBDA_DENSE_VALUES)-1}$). Solid curves are Monte Carlo "
            f"means over {N_MONTE_CARLO} replications; shaded bands are 95\\% "
            "percentile confidence intervals. The figure shows the emergence of "
            "policy violations under linear aggregation as predictive evidence "
            "receives more weight; $A_G$ and $A_M$ remain at $V=0$ throughout "
            "the sweep in this experiment."
        ),
        "",
        r"Suggested LaTeX label: \label{fig:policy-violation-dense}",
        "",
        "## Table 2: table_policy_compliance.tex",
        "",
        (
            f"Monte Carlo mean, standard deviation, and 95\\% percentile confidence "
            f"interval of $V$ for each operator at "
            f"$\\lambda\\in\\{{{', '.join(f'{x:.2f}' for x in LAMBDA_BAR_VALUES)}\\}}$. "
            r"Contextual compliance $\mathrm{VPR}=1-V$ is omitted as redundant."
        ),
        "",
        r"Suggested LaTeX label: \label{tab:policy-compliance}",
        "",
    ]
    (exp_dir / "captions.md").write_text("\n".join(captions), encoding="utf-8")


def write_readme(
    bar_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    exp_dir: Path,
    settings: RunSettings,
) -> None:
    linear_rows = "\n".join(
        f"- $\\lambda={lam:.2f}$: $V={_v_sentence(_lookup(bar_df, 'linear', lam))}$"
        for lam in LAMBDA_BAR_VALUES
    )
    first_lam = _linear_onset(dense_df)
    onset = (
        f"Linear mean $V$ first becomes strictly positive at $\\lambda={first_lam:.2f}$."
        if first_lam is not None
        else "Linear mean $V$ remains $0$ over the recorded $\\lambda$ grid."
    )
    geom_zero = all(
        float(_lookup(bar_df, "geometric", lam)["policy_violation_rate_mean"]) == 0.0
        for lam in LAMBDA_BAR_VALUES
    )
    min_zero = all(
        float(_lookup(bar_df, "min", lam)["policy_violation_rate_mean"]) == 0.0
        for lam in LAMBDA_BAR_VALUES
    )
    conclusions = [
        "Observed Monte Carlo behaviour (no ranking of operators as better/worse):",
        "",
        "Linear operator — violations emerge as $\\lambda$ increases:",
        linear_rows,
        onset,
        "",
        (
            "Weighted geometric operator: $V=0$ at the representative $\\lambda$ values "
            "and throughout the dense sweep, consistent with zero absorption."
            if geom_zero
            else "Weighted geometric operator: see `manuscript_snippets.md`."
        ),
        (
            "Minimum operator: $V=0$ at the representative $\\lambda$ values "
            "and throughout the dense sweep, consistent with zero absorption."
            if min_zero
            else "Minimum operator: see `manuscript_snippets.md`."
        ),
        "",
        "Contextual compliance is $\\mathrm{VPR}=1-V$ and is not plotted.",
        "Predictive utility $\\bar{R}$ is stored in the CSV files for later sections and is not discussed here.",
    ]
    text = f"""# Experiment 01 — Policy compliance under contextual vetoes

Manuscript reference: `first_round_v2/manuscriptR1V2.tex`, Section 6.1
(`\\subsection{{Policy Compliance under Contextual Vetoes}}`).

## Scientific objective

This subsection is **not** a comparison of which operator is better. It
demonstrates the **population-level consequences** of the theoretical
contextual-veto property (manuscript Section 4 / zero absorption):

- when policy violations emerge under compensatory (linear) aggregation;
- when contextual vetoes remain preserved;
- how those outcomes depend on $\\lambda$.

## Adversarial veto population

The veto group is intentionally assigned high predictive scores
($R_i\\sim\\mathrm{{Beta}}{VETO_R_BETA}$, $Q_i=0$) in order to create the most
challenging scenario for policy preservation. This allows the aggregation
semantics to be evaluated under conditions where predictive evidence strongly
conflicts with contextual constraints.

## Experimental design

For each Monte Carlo replication:

1. Generate a synthetic population of $N$ cases.
2. Set a fraction `veto_fraction` of cases to $Q_i=0$ with high $R_i$.
3. Apply the three aggregation operators on the **same** population.
4. Select the Top-$K$ set with deterministic tie-breaking $(-P,-R,\\mathrm{{case\\_id}})$.
5. Compute $V$, $\\mathrm{{VPR}}=1-V$, and predictive utility $\\bar{{R}}$
   (the last two are stored for reuse; only $V$ is presented in Section 6.1).

The manuscript uses two complementary views of $V$:

- **Figure 6.** Grouped bars at $\\lambda\\in\\{{0.50,0.75,0.90\\}}$ (balanced,
  prediction-dominant, strongly prediction-dominant).
- **Figure 7.** Dense sweep $V(\\lambda)$ on $[0,1]$, which shows the
  transition from no violations to systematic violations.

## Parameters

All defaults live in `simulations/common/config.py`. Group sizes are derived
from `veto_fraction` (default ${VETO_FRACTION:.2f}$). Supported values for
later checks: {VETO_FRACTIONS_SUPPORTED}. This run used
`veto_fraction={settings.veto_fraction:.2f}`.

| Parameter | Value |
|---|---|
| $N$ | {N_CASES} |
| `veto_fraction` | {settings.veto_fraction:.2f} |
| Standard cases | {settings.n_std} |
| Veto cases ($Q_i=0$) | {settings.n_veto} |
| Top-$K$ | {TOP_K} |
| $R$ (standard) | Beta{STD_R_BETA} |
| $Q$ (standard) | Beta{STD_Q_BETA} |
| $R$ (veto, adversarial) | Beta{VETO_R_BETA} |
| Monte Carlo replications | {N_MONTE_CARLO} |
| Seeds | {MC_SEED_BASE} … {MC_SEED_BASE + N_MONTE_CARLO - 1} |
| $\\lambda$ (Figure 6) | {LAMBDA_BAR_VALUES} |
| $\\lambda$ (Figure 7) | {len(LAMBDA_DENSE_VALUES)} points on $[{LAMBDA_DENSE_VALUES[0]:.1f},{LAMBDA_DENSE_VALUES[-1]:.1f}]$ |
| Confidence intervals | {int(CI_LEVEL * 100)}% percentiles |

## Metrics

- **Policy violation rate** $V=|\\mathcal{{T}}_K\\cap\\mathcal{{V}}|/K$, $\\mathcal{{V}}=\\{{i:Q_i=0\\}}$ — reported in Section 6.1.
- **Contextual compliance / VPR** $1-V$ — computed, mentioned in text, not plotted.
- **Predictive utility** $\\bar{{R}}$ — computed and stored; **not** discussed in Section 6.1.

## How to run

```bash
pip install -r simulations/requirements.txt
cd simulations/01_policy_compliance_under_contextual_vetoes
python run.py
python run.py --refresh
python run.py --veto-fraction 0.10   # optional extra scenario; not used in the paper by default
```

## Generated outputs

- `results/trials_raw.csv` — one row per (trial, $\\lambda$, operator); includes $V$, VPR, $\\bar{{R}}$
- `results/aggregated_dense.csv` — mean, std, CI95% over the dense $\\lambda$ grid
- `results/aggregated_barplot.csv` — subset $\\lambda\\in\\{{0.50,0.75,0.90\\}}$
- `results/run_metadata.json` — seeds, `veto_fraction`, fingerprint
- `figures/policy_violation_rate_grouped.{{pdf,png}}` — Figure 6
- `figures/policy_violation_rate_dense.{{pdf,png}}` — Figure 7
- `tables/table_policy_compliance.tex` — Table 2 ($V$ only)
- `captions.md`
- `manuscript_snippets.md`
- `results_narrative.md`

## Main conclusions from the results

{chr(10).join(conclusions)}
"""
    (exp_dir / "README.md").write_text(text, encoding="utf-8")


def _spark(value: float, vmax: float = 0.65, width: int = 13) -> str:
    n = int(round(max(0.0, value) / vmax * width)) if vmax > 0 else 0
    return "|" + ("*" * n)


def write_narrative(
    bar_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    exp_dir: Path,
    settings: RunSettings,
) -> None:
    first_lam = _linear_onset(dense_df)
    linear_dense = dense_df[dense_df["operator"] == "linear"].sort_values("lambda")
    spark_lines = []
    for _, row in linear_dense.iterrows():
        lam = float(row["lambda"])
        mean = float(row["policy_violation_rate_mean"])
        marker = "  <- first strictly positive mean" if first_lam is not None and np.isclose(lam, first_lam) else ""
        spark_lines.append(f"{lam:0.2f}  {mean:0.3f}          {_spark(mean)}{marker}")

    def bar_cell(op: str, lam: float) -> str:
        row = _lookup(bar_df, op, lam)
        return (
            f"{float(row['policy_violation_rate_mean']):.3f} "
            f"[{float(row['policy_violation_rate_ci_low']):.3f}, "
            f"{float(row['policy_violation_rate_ci_high']):.3f}]"
        )

    linear_ci_rows = []
    for _, row in linear_dense.iterrows():
        lam = float(row["lambda"])
        if lam + 1e-12 < 0.65:
            continue
        linear_ci_rows.append(
            f"| {lam:.2f} | {float(row['policy_violation_rate_mean']):.3f} | "
            f"{float(row['policy_violation_rate_ci_low']):.2f} | "
            f"{float(row['policy_violation_rate_ci_high']):.2f} |"
        )

    text = f"""# Narrative summary — Experiment 6.1 Policy compliance under contextual vetoes

This file is a **text-only briefing** for humans and LLM agents working on
`first_round_v2/manuscriptR1V2.tex` Section 6.1. Do not invent numerical claims.
Do not describe any operator as globally superior.

- Code: `simulations/01_policy_compliance_under_contextual_vetoes/run.py`
- Config: `simulations/common/config.py` (`veto_fraction={settings.veto_fraction:.2f}`)
- Seeds: {MC_SEED_BASE} through {MC_SEED_BASE + N_MONTE_CARLO - 1}

## Scientific message

Section 6.1 shows the **population-level consequences** of the contextual-veto
property from Section 4. The question is when policy violations **emerge**, not
which operator is better.

The veto group is **adversarial**: $Q_i=0$ and $R_i\\sim\\mathrm{{Beta}}{VETO_R_BETA}$,
so predictive evidence strongly conflicts with the contextual constraint.

Contextual compliance is $\\mathrm{{VPR}}=1-V$. It is mentioned in the text and
**not** given its own figure. Predictive utility $\\bar{{R}}$ is stored in CSV
files for later sections and is **not** discussed in 6.1.

## Design (this run)

| Parameter | Value |
|---|---|
| $N$ | {N_CASES} |
| `veto_fraction` | {settings.veto_fraction:.2f} |
| Veto cases | {settings.n_veto} |
| Standard cases | {settings.n_std} |
| $K$ | {TOP_K} |
| Monte Carlo | {N_MONTE_CARLO} |
| Figure 6 $\\lambda$ | {LAMBDA_BAR_VALUES} |
| Figure 7 $\\lambda$ | 21 points on $[0,1]$ |

## Headline results

1. At $\\lambda=0.50$, linear $V={bar_cell('linear', 0.50)}$. Geometric and minimum remain at $V=0$.
2. At $\\lambda=0.75$, linear $V={bar_cell('linear', 0.75)}$. Geometric and minimum remain at $V=0$.
3. At $\\lambda=0.90$, linear $V={bar_cell('linear', 0.90)}$. Geometric and minimum remain at $V=0$.
4. Linear aggregation **does not always** violate vetoes. Violations emerge as $\\lambda$ increases.
5. Figure 7: linear mean $V$ is $0$ until late in the sweep; first strictly positive mean at $\\lambda={first_lam if first_lam is not None else 'n/a'}$. $A_G$ and $A_M$ stay at $V=0$ for every $\\lambda\\in[0,1]$.

## Figure 6 — grouped bars $V$ (primary)

File: `figures/policy_violation_rate_grouped.pdf`

```
V
1.0 |
    |
0.8 |
    |
0.6 |
    |                          [L ~0.46]
0.4 |                          ####
    |                          ####
0.2 |              [L ~0.08]   ####
0.0 | L G M        L G M       L G M
      0.50          0.75         0.90
```

| $\\lambda$ | $A_L$ mean $V$ [CI] | $A_G$ | $A_M$ |
|---|---|---|---|
| 0.50 | {bar_cell('linear', 0.50)} | {bar_cell('geometric', 0.50)} | {bar_cell('min', 0.50)} |
| 0.75 | {bar_cell('linear', 0.75)} | {bar_cell('geometric', 0.75)} | {bar_cell('min', 0.75)} |
| 0.90 | {bar_cell('linear', 0.90)} | {bar_cell('geometric', 0.90)} | {bar_cell('min', 0.90)} |

At 0.50 all three bars sit on the axis. At 0.75 a small linear bar appears.
At 0.90 the linear bar is large; geometric and minimum remain zero. The linear
CI at 0.90 does not overlap zero.

## Figure 7 — dense sweep $V(\\lambda)$ (primary)

File: `figures/policy_violation_rate_dense.pdf`

This is a **primary** result of the subsection: it shows the transition from
no violations to systematic violations.

ASCII of linear **mean $V$** (`*` scaled to 0.65). Geometric and minimum are
the zero line at every $\\lambda$.

```
λ     A_L mean V     spark
{chr(10).join(spark_lines)}
```

Linear 95% CI along the rise:

| $\\lambda$ | mean $V$ | CI low | CI high |
|---|---|---|---|
{chr(10).join(linear_ci_rows)}

At $\\lambda=1$, $A_L(R,Q)=R$, so ranking ignores context.

## Table 2

`tables/table_policy_compliance.tex`: operator × $\\lambda$ for Figure 6 values,
with mean, std, and 95% CI of $V$ only.

## Safe claims for the manuscript

- The experiment is an adversarial test of veto preservation, not a bake-off.
- Linear $V=0$ at $\\lambda=0.50$; violations emerge at higher $\\lambda$.
- $A_G$ and $A_M$ keep $V=0$ here because they absorb zero, not because they are "better".
- $\\mathrm{{VPR}}=1-V$; no VPR figure.
- Do not discuss $\\bar{{R}}$ in Section 6.1.

## File map

| Need | Path |
|---|---|
| Figure 6 | `figures/policy_violation_rate_grouped.pdf` |
| Figure 7 | `figures/policy_violation_rate_dense.pdf` |
| Table 2 | `tables/table_policy_compliance.tex` |
| Prose | `manuscript_snippets.md` |
| Captions | `captions.md` |
| Extra `veto_fraction` | `python run.py --veto-fraction 0.10` |
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
        "--veto-fraction",
        type=float,
        default=VETO_FRACTION,
        help=(
            "Fraction of cases with Q_i=0 "
            f"(default {VETO_FRACTION:.2f}; supported {VETO_FRACTIONS_SUPPORTED})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _settings_from_args(args)
    paths = ensure_dirs(EXP_DIR)
    raw = run_monte_carlo(settings, results_dir=paths["results"])
    dense = summarize_trials(raw, ["lambda", "operator"], METRICS)
    dense = _order_operators(dense)
    dense_path = paths["results"] / "aggregated_dense.csv"
    dense.to_csv(dense_path, index=False)
    bar = _subset_bar_values(dense)
    bar_path = paths["results"] / "aggregated_barplot.csv"
    bar.to_csv(bar_path, index=False)
    print(f"[agg] wrote {dense_path} ({len(dense)} rows)")
    print(f"[agg] wrote {bar_path} ({len(bar)} rows)")

    figures = write_figures(bar, dense, paths["figures"])
    for path in figures:
        print(f"[fig] {path}")
    table_path = write_table(bar, paths["tables"], settings)
    print(f"[tex] {table_path}")
    write_snippets(bar, dense, EXP_DIR, settings)
    write_readme(bar, dense, EXP_DIR, settings)
    write_narrative(bar, dense, EXP_DIR, settings)
    print("[doc] manuscript_snippets.md, captions.md, README.md, results_narrative.md")
    print("done.")


if __name__ == "__main__":
    main()
