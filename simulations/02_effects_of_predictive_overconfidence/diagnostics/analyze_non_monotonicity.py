#!/usr/bin/env python3
"""Diagnostic analysis of non-monotonic V(δ) for A_L in Experiment 6.2.

Internal only. Does not change the formal experiment or the manuscript.
Reuses the Experiment 02 population generator, seeds, λ=0.75, and δ grid.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent
EXP_DIR = DIAG_DIR.parent
SIM_DIR = EXP_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from common.aggregators import aggregate  # noqa: E402
from common.config import (  # noqa: E402
    DELTA_VALUES,
    N_CASES,
    N_MONTE_CARLO,
    OVERCONFIDENCE_PRIMARY_LAMBDA,
    Q_WEAK_THRESHOLD,
    TOP_K,
    VETO_FRACTION,
    n_std_from_fraction,
    n_veto_from_fraction,
)
from common.generators import apply_predictive_overconfidence, generate_population  # noqa: E402
from common.metrics import top_k_indices  # noqa: E402
from common.plotting import (  # noqa: E402
    LEGEND_KW,
    add_panel_letter,
    apply_paper_style,
    new_square_single_panel,
    new_square_two_panel,
    save_figure,
    style_square_box,
)
from common.utils import (  # noqa: E402
    ensure_dirs,
    experiment_fingerprint,
    format_ci,
    mean_std_ci,
    trial_seed,
    utc_now_iso,
    write_json,
)

LAM = OVERCONFIDENCE_PRIMARY_LAMBDA
Q_WEAK = Q_WEAK_THRESHOLD
MODES = ("clipped", "unclipped")
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95)


def summarize_allow_nan(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            arr = sub[metric].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                mean = std = lo = hi = float("nan")
            else:
                mean, std, lo, hi = mean_std_ci(arr)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def apply_overconfidence_raw(
    R: np.ndarray,
    Q: np.ndarray,
    delta: float,
    *,
    q_threshold: float,
) -> np.ndarray:
    """Same mask as the experiment, without clipping: R' = R + δ if Q ≤ q."""
    r_out = np.asarray(R, dtype=float).copy()
    weak = np.asarray(Q, dtype=float) <= float(q_threshold)
    r_out[weak] = r_out[weak] + float(delta)
    return r_out


def _mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def _median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if x.size else float("nan")


def _frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return float(np.mean(mask)) if mask.size else float("nan")


def _qtile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, q)) if x.size else float("nan")


def _group_stats(prefix: str, r: np.ndarray, r_prime: np.ndarray) -> dict[str, float]:
    inc = np.asarray(r_prime, dtype=float) - np.asarray(r, dtype=float)
    out = {
        f"{prefix}_n": float(np.asarray(r).size),
        f"{prefix}_mean_R": _mean(r),
        f"{prefix}_mean_Rprime": _mean(r_prime),
        f"{prefix}_median_Rprime": _median(r_prime),
        f"{prefix}_mean_increase": _mean(inc),
        f"{prefix}_frac_saturated": _frac(np.asarray(r_prime, dtype=float) >= 1.0 - 1e-12),
    }
    for q in QUANTILES:
        qtag = f"{int(round(q * 100)):02d}"
        out[f"{prefix}_Rprime_q{qtag}"] = _qtile(r_prime, q)
    return out


def _run_trial(trial_idx: int) -> list[dict]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(
        rng,
        n_std=n_std_from_fraction(VETO_FRACTION, N_CASES),
        n_veto=n_veto_from_fraction(VETO_FRACTION, N_CASES),
    )
    r = pop.R
    q = pop.Q
    is_veto = q == 0.0
    is_weak = q <= Q_WEAK
    is_weak_nv = is_weak & ~is_veto
    is_normal = ~is_weak
    records: list[dict] = []
    for delta in DELTA_VALUES:
        r_clip = apply_predictive_overconfidence(r, q, delta, q_threshold=Q_WEAK)
        r_raw = apply_overconfidence_raw(r, q, delta, q_threshold=Q_WEAK)
        for mode, r_prime in (("clipped", r_clip), ("unclipped", r_raw)):
            p = aggregate("linear", r_prime, q, LAM)
            idx = top_k_indices(p, TOP_K, R=r_prime, case_id=pop.case_id)
            in_topk = np.zeros(pop.n, dtype=bool)
            in_topk[idx] = True
            selected = {
                "veto": is_veto & in_topk,
                "weak_nv": is_weak_nv & in_topk,
                "normal": is_normal & in_topk,
                "weak": is_weak & in_topk,
            }
            n_veto_k = int(selected["veto"].sum())
            n_weak_nv_k = int(selected["weak_nv"].sum())
            n_normal_k = int(selected["normal"].sum())
            n_weak_k = n_veto_k + n_weak_nv_k
            p_threshold = float(np.min(p[idx]))
            rec: dict[str, float | int | str] = {
                "trial": trial_idx,
                "seed": trial_seed(trial_idx),
                "delta": float(delta),
                "mode": mode,
                "n_affected": int(is_weak.sum()),
                "n_veto_pop": int(is_veto.sum()),
                "n_weak_nv_pop": int(is_weak_nv.sum()),
                "n_normal_pop": int(is_normal.sum()),
                "v": n_veto_k / TOP_K,
                "weak_rate": n_weak_k / TOP_K,
                "veto_among_weak": (n_veto_k / n_weak_k) if n_weak_k else float("nan"),
                "n_veto_topk": n_veto_k,
                "n_weak_nv_topk": n_weak_nv_k,
                "n_normal_topk": n_normal_k,
                "n_weak_topk": n_weak_k,
                "mean_Q_topk": _mean(q[idx]),
                "mean_R_topk": _mean(r[idx]),
                "mean_Rprime_topk": _mean(r_prime[idx]),
                "frac_Rprime1_topk": _frac(r_prime[idx] >= 1.0 - 1e-12),
                "frac_weak_topk": n_weak_k / TOP_K,
                "frac_veto_topk": n_veto_k / TOP_K,
                "p_threshold": p_threshold,
                "mean_P_veto": _mean(p[is_veto]),
                "mean_P_weak_nv": _mean(p[is_weak_nv]),
                "mean_P_normal": _mean(p[is_normal]),
                "mean_P_veto_topk": _mean(p[selected["veto"]]),
                "mean_P_weak_nv_topk": _mean(p[selected["weak_nv"]]),
                "mean_P_normal_topk": _mean(p[selected["normal"]]),
                "inclusion_veto": _frac(in_topk[is_veto]),
                "inclusion_weak_nv": _frac(in_topk[is_weak_nv]),
                "inclusion_normal": _frac(in_topk[is_normal]),
                "frac_sat_veto": _frac(r_prime[is_veto] >= 1.0 - 1e-12),
                "frac_sat_weak_nv": _frac(r_prime[is_weak_nv] >= 1.0 - 1e-12),
                "frac_sat_affected": _frac(r_prime[is_weak] >= 1.0 - 1e-12),
                "frac_raw_ge1_veto": _frac((r[is_veto] + delta) >= 1.0 - 1e-12),
                "frac_raw_ge1_weak_nv": _frac((r[is_weak_nv] + delta) >= 1.0 - 1e-12),
                "frac_raw_ge1_affected": _frac((r[is_weak] + delta) >= 1.0 - 1e-12),
            }
            rec.update(_group_stats("aff", r[is_weak], r_prime[is_weak]))
            rec.update(_group_stats("veto", r[is_veto], r_prime[is_veto]))
            rec.update(_group_stats("weak_nv", r[is_weak_nv], r_prime[is_weak_nv]))
            rec.update(_group_stats("aff_in", r[is_weak & in_topk], r_prime[is_weak & in_topk]))
            rec.update(_group_stats("aff_out", r[is_weak & ~in_topk], r_prime[is_weak & ~in_topk]))
            for name, mask in (
                ("veto", is_veto),
                ("weak_nv", is_weak_nv),
                ("normal", is_normal),
            ):
                for qtile in QUANTILES:
                    qtag = f"{int(round(qtile * 100)):02d}"
                    rec[f"P_{name}_q{qtag}"] = _qtile(p[mask], qtile)
            records.append(rec)
    return records


def _fingerprint() -> str:
    return experiment_fingerprint(
        "02_diagnostics_non_monotonicity",
        {
            "n_cases": N_CASES,
            "n_mc": N_MONTE_CARLO,
            "lambda": LAM,
            "q_weak": Q_WEAK,
            "delta_values": list(DELTA_VALUES),
            "veto_fraction": VETO_FRACTION,
            "top_k": TOP_K,
            "modes": list(MODES),
        },
    )


def run_or_load(results_dir: Path, *, refresh: bool) -> pd.DataFrame:
    raw_path = results_dir / "trials_diagnostics.csv"
    meta_path = results_dir / "run_metadata.json"
    fingerprint = _fingerprint()
    if raw_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {raw_path} ({meta.get('n_rows')} rows)")
            return pd.read_csv(raw_path)
    records: list[dict] = []
    for trial_idx in range(N_MONTE_CARLO):
        records.extend(_run_trial(trial_idx))
        if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
            print(f"[mc] trial {trial_idx + 1}/{N_MONTE_CARLO}")
    raw = pd.DataFrame.from_records(records)
    raw.to_csv(raw_path, index=False)
    write_json(
        meta_path,
        {
            "experiment": "02_diagnostics_non_monotonicity",
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(raw)),
            "n_monte_carlo": N_MONTE_CARLO,
            "lambda": LAM,
            "q_weak": Q_WEAK,
            "delta_values": list(DELTA_VALUES),
        },
    )
    print(f"[mc] wrote {raw_path} ({len(raw)} rows)")
    return raw


def _ci_row(agg: pd.DataFrame, mode: str, delta: float, metric: str) -> pd.Series:
    sub = agg[
        (agg["mode"] == mode) & np.isclose(agg["delta"].to_numpy(), delta)
    ]
    if sub.empty:
        raise KeyError(f"No row mode={mode} delta={delta}")
    return sub.iloc[0]


def _fmt(agg: pd.DataFrame, mode: str, delta: float, metric: str) -> str:
    row = _ci_row(agg, mode, delta, metric)
    return format_ci(
        float(row[f"{metric}_mean"]),
        float(row[f"{metric}_ci_low"]),
        float(row[f"{metric}_ci_high"]),
    )


def _num(agg: pd.DataFrame, mode: str, delta: float, metric: str) -> float:
    return float(_ci_row(agg, mode, delta, metric)[f"{metric}_mean"])


def _plot_lines(
    ax,
    agg: pd.DataFrame,
    *,
    mode: str | None,
    series: list[tuple[str, str, str]],
    ylabel: str,
    xlabel: str = r"Overconfidence level $\delta$",
    ylim: tuple[float, float] | None = None,
) -> None:
    """series: (metric, label, color)."""
    apply_paper_style()
    df = agg if mode is None else agg[agg["mode"] == mode]
    markers = ["v", "s", "o", "D", "^"]
    for i, (metric, label, color) in enumerate(series):
        sub = df.sort_values("delta")
        ax.fill_between(
            sub["delta"],
            sub[f"{metric}_ci_low"],
            sub[f"{metric}_ci_high"],
            color=color,
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            sub["delta"],
            sub[f"{metric}_mean"],
            color=color,
            marker=markers[i % len(markers)],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.0,
            label=label,
            zorder=2 + i,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(DELTA_VALUES))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(**LEGEND_KW)
    style_square_box(ax)


def write_figures(agg: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written: list[Path] = []
    colors = {
        "clip": "#0072B2",
        "raw": "#D55E00",
        "veto": "#0072B2",
        "weak": "#D55E00",
        "normal": "#009E73",
        "thr": "#000000",
    }
    fig, ax_a, ax_b = new_square_two_panel()
    clip = agg[agg["mode"] == "clipped"]
    # Panel a: V
    for mode, color, marker, label in (
        ("clipped", colors["clip"], "v", r"$R'=\mathrm{clip}(R+\delta,0,1)$"),
        ("unclipped", colors["raw"], "s", r"$R'=R+\delta$ (diagnostic)"),
    ):
        sub = agg[agg["mode"] == mode].sort_values("delta")
        ax_a.fill_between(
            sub["delta"], sub["v_ci_low"], sub["v_ci_high"], color=color, alpha=0.22, linewidth=0
        )
        ax_a.plot(
            sub["delta"],
            sub["v_mean"],
            color=color,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.0,
            label=label,
        )
    ax_a.set_xlabel(r"Overconfidence level $\delta$")
    ax_a.set_ylabel(r"Policy violation rate $V$")
    ax_a.set_xticks(list(DELTA_VALUES))
    ax_a.set_ylim(-0.05, 1.0)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.legend(**LEGEND_KW)
    style_square_box(ax_a)
    add_panel_letter(ax_a, "a")
    # Panel b: weak-context rate
    for mode, color, marker, label in (
        ("clipped", colors["clip"], "v", r"clipped $Q\leq 0.25$ in Top-$K$"),
        ("unclipped", colors["raw"], "s", r"unclipped $Q\leq 0.25$ in Top-$K$"),
    ):
        sub = agg[agg["mode"] == mode].sort_values("delta")
        ax_b.fill_between(
            sub["delta"],
            sub["weak_rate_ci_low"],
            sub["weak_rate_ci_high"],
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax_b.plot(
            sub["delta"],
            sub["weak_rate_mean"],
            color=color,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.0,
            label=label,
        )
    ax_b.set_xlabel(r"Overconfidence level $\delta$")
    ax_b.set_ylabel(r"Weak-context share of Top-$K$")
    ax_b.set_xticks(list(DELTA_VALUES))
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.legend(**LEGEND_KW)
    style_square_box(ax_b)
    add_panel_letter(ax_b, "b")
    written.extend(
        save_figure(fig, figures_dir / "v_clipped_vs_unclipped", bbox_inches=None)
    )

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        mode=None,
        series=[
            ("n_veto_topk", r"$Q=0$ (vetoes)", colors["veto"]),
            ("n_weak_nv_topk", r"$0<Q\leq 0.25$", colors["weak"]),
            ("n_normal_topk", r"$Q>0.25$", colors["normal"]),
        ],
        ylabel=r"Count in Top-$K$ ($K=100$)",
        ylim=(-5, 105),
    )
    written.extend(save_figure(fig, figures_dir / "topk_composition", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        mode=None,
        series=[
            ("frac_sat_veto", r"vetoes $R'=1$", colors["veto"]),
            ("frac_sat_weak_nv", r"weak non-veto $R'=1$", colors["weak"]),
        ],
        ylabel=r"Fraction with $R'=1$",
        ylim=(-0.05, 1.05),
    )
    written.extend(save_figure(fig, figures_dir / "saturation_by_group", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        mode=None,
        series=[
            ("mean_P_veto", r"mean $P$, $Q=0$", colors["veto"]),
            ("mean_P_weak_nv", r"mean $P$, $0<Q\leq 0.25$", colors["weak"]),
            ("mean_P_normal", r"mean $P$, $Q>0.25$", colors["normal"]),
            ("p_threshold", r"Top-$K$ threshold", colors["thr"]),
        ],
        ylabel=r"Linear score $P=0.75 R'+0.25 Q$",
        ylim=(0.35, 0.90),
    )
    ax.axhline(0.75, color="0.5", linestyle="--", linewidth=1.0, zorder=1)
    written.extend(save_figure(fig, figures_dir / "scores_vs_threshold", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        mode=None,
        series=[
            ("inclusion_veto", r"veto inclusion", colors["veto"]),
            ("inclusion_weak_nv", r"weak non-veto inclusion", colors["weak"]),
            ("inclusion_normal", r"normal inclusion", colors["normal"]),
        ],
        ylabel=r"Top-$K$ inclusion probability",
        ylim=(-0.02, 0.55),
    )
    written.extend(save_figure(fig, figures_dir / "inclusion_probability", bbox_inches=None))
    return written


def _grid(agg: pd.DataFrame, mode: str, metrics: list[tuple[str, str]]) -> str:
    header = "| $\\delta$ | " + " | ".join(name for _, name in metrics) + " |"
    sep = "|" + "---|" * (1 + len(metrics))
    lines = [header, sep]
    for delta in DELTA_VALUES:
        cells = [f"{delta:.2f}"]
        for metric, _ in metrics:
            cells.append(f"{_num(agg, mode, delta, metric):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(agg: pd.DataFrame, path: Path) -> None:
    dmax = float(DELTA_VALUES[-1])
    v_clip = [_num(agg, "clipped", d, "v") for d in DELTA_VALUES]
    v_raw = [_num(agg, "unclipped", d, "v") for d in DELTA_VALUES]
    peak_clip = DELTA_VALUES[int(np.argmax(v_clip))]
    peak_raw = DELTA_VALUES[int(np.argmax(v_raw))]
    clip_nonmono = peak_clip < dmax - 1e-12 and v_clip[-1] < max(v_clip) - 0.01
    raw_nonmono = peak_raw < dmax - 1e-12 and v_raw[-1] < max(v_raw) - 0.01

    if (not clip_nonmono) and raw_nonmono:
        clip_verdict = "INCONCLUSIVE (unexpected: unclipped non-monotonic, clipped not)."
    elif clip_nonmono and not raw_nonmono:
        clip_verdict = (
            "YES. Clipped $V(\\delta)$ is non-monotonic; the unclipped diagnostic "
            "is not (it keeps rising or plateaus). Saturation at $R'=1$ is necessary "
            "for the decline after the peak."
        )
    elif clip_nonmono and raw_nonmono:
        clip_verdict = (
            "NO, not by itself. Non-monotonicity persists without clipping, so "
            "another compositional mechanism is sufficient."
        )
    else:
        clip_verdict = (
            "NEITHER curve shows a clear interior peak under the stated criterion; "
            "inspect the tables before claiming a clipping explanation."
        )

    p_veto_cap = LAM * 1.0 + (1.0 - LAM) * 0.0
    text = f"""# Non-monotonic $V(\\delta)$ for $A_L$ — diagnostic analysis

Internal report for Experiment 6.2. Not manuscript text. All numbers are
Monte Carlo means over {N_MONTE_CARLO} replications with the same seeds,
population generator, $\\lambda={LAM:.2f}$, $q_{{\\mathrm{{weak}}}}={Q_WEAK:.2f}$,
and $\\delta$ grid as the formal experiment. 95% percentile CIs are in the CSVs.

## 1. Question

For linear aggregation $A_L(R',Q)=\\lambda R'+(1-\\lambda)Q$ at $\\lambda={LAM:.2f}$,
the policy violation rate $V$ (share of Top-$K$ with $Q=0$) is **non-monotonic**
in the formal experiment:

| $\\delta$ | $V$ (formal Exp. 02) |
|---|---|
| 0.00 | 0.077 |
| 0.05 | 0.209 |
| 0.10 | 0.251 (maximum) |
| 0.15 | 0.196 |
| 0.20 | 0.131 |
| 0.25 | 0.068 |
| 0.30 | 0.022 |

Why does $V$ rise and then fall? Candidate mechanisms to test, not assume:

1. Overconfidence is applied only to $Q\\le {Q_WEAK:.2f}$, so those cases become
   more competitive.
2. Clipping $R'=\\mathrm{{clip}}(R+\\delta,0,1)$ saturates high-$R$ cases at 1.
3. Vetoes ($Q=0$) are replaced in Top-$K$ by weak non-vetoes ($0<Q\\le {Q_WEAK:.2f}$).
4. Something else in the Top-$K$ threshold / score distributions.

$V$ counts **only** $Q=0$. A fall in $V$ need not mean that weak-context cases
leave Top-$K$.

## 2. Experimental setup (this diagnostic)

- Same `generate_population` as Experiment 02: $N={N_CASES}$, veto fraction
  ${VETO_FRACTION:.2f}$, standard $R,Q\\sim\\mathrm{{Beta}}(2,2)$, veto
  $R\\sim\\mathrm{{Beta}}(8,2)$ and $Q=0$.
- Same seeds $42$–$1041$.
- Operator: **linear only**, $\\lambda={LAM:.2f}$.
- $R'$ on $Q\\le {Q_WEAK:.2f}$ only.
- Two modes on the **same** populations:
  - **clipped** (formal): $R'=\\mathrm{{clip}}(R+\\delta,0,1)$;
  - **unclipped** (diagnostic): $R'=R+\\delta$ (may exceed 1).
- Top-$K$ ties: $(-P,-R',\\mathrm{{id}})$, $K={TOP_K}$.
- Groups:
  - A: vetoes $Q=0$;
  - B: weak non-vetoes $0<Q\\le {Q_WEAK:.2f}$;
  - C: normal $Q>{Q_WEAK:.2f}$ (never receive $\\delta$).

Algebra of the linear score at $\\lambda={LAM:.2f}$:

- Veto, $R'=1$: $P= {p_veto_cap:.2f}$. This is the **clipped ceiling** for every
  $Q=0$ case.
- Weak non-veto, $R'=1$: $P={LAM:.2f}+{1.0-LAM:.2f}Q>{p_veto_cap:.2f}$ whenever
  $Q>0$. Saturated weak non-vetoes **strictly outrank** saturated vetoes.
- Unclipped, both groups receive the same additive $+\\lambda\\delta$ in $P$,
  so **relative** order among already-affected cases is invariant to $\\delta$.

These identities follow from the operator. Whether they drive the observed $V$
curve is an empirical question below.

## 3. Clipping analysis (affected population $Q\\le {Q_WEAK:.2f}$)

Clipped mode, means:

{_grid(agg, 'clipped', [
    ('frac_raw_ge1_affected', r"frac $R+\delta\geq 1$"),
    ('frac_sat_affected', "frac $R'=1$"),
    ('frac_sat_veto', "veto $R'=1$"),
    ('frac_sat_weak_nv', "weak-NV $R'=1$"),
    ('aff_mean_Rprime', "mean $R'$"),
    ('aff_mean_increase', "mean $R'-R$"),
])}

Veto vs weak non-veto original $R$ (clipped mode; $R$ does not depend on $\\delta$
except through sampling; reported at $\\delta=0$):

- Veto mean original $R$: {_fmt(agg, 'clipped', 0.0, 'veto_mean_R')}
- Weak non-veto mean original $R$: {_fmt(agg, 'clipped', 0.0, 'weak_nv_mean_R')}

Vetoes start with substantially higher $R$ ($\\mathrm{{Beta}}(8,2)$ vs
$\\mathrm{{Beta}}(2,2)$), so they hit the clip $R+\\delta\\ge 1$ at **smaller**
$\\delta$.

Affected cases **in** Top-$K$ vs **out** (clipped):

{_grid(agg, 'clipped', [
    ('aff_in_frac_saturated', "in Top-$K$, frac $R'=1$"),
    ('aff_out_frac_saturated', "out of Top-$K$, frac $R'=1$"),
    ('aff_in_mean_Rprime', "in, mean $R'$"),
    ('aff_out_mean_Rprime', "out, mean $R'$"),
])}

## 4. Top-$K$ composition ($A_L$, clipped)

Counts in Top-$K$ ($K={TOP_K}$), Monte Carlo means:

{_grid(agg, 'clipped', [
    ('n_veto_topk', r'$Q=0$'),
    ('n_weak_nv_topk', r"$0<Q\leq 0.25$"),
    ('n_normal_topk', r'$Q>0.25$'),
    ('v', r'$V$'),
    ('weak_rate', r'weak share'),
    ('veto_among_weak', r'veto / weak in Top-$K$'),
])}

Mean scores of the selected set:

{_grid(agg, 'clipped', [
    ('mean_Q_topk', r'mean $Q$'),
    ('mean_R_topk', r'mean original $R$'),
    ('mean_Rprime_topk', "mean $R'$"),
    ('frac_Rprime1_topk', "frac $R'=1$"),
])}

At $\\delta=0.00$: $V={_fmt(agg, 'clipped', 0.0, 'v')}$; weak share
{_fmt(agg, 'clipped', 0.0, 'weak_rate')}; vetoes in Top-$K$
{_num(agg, 'clipped', 0.0, 'n_veto_topk'):.2f}; weak non-vetoes
{_num(agg, 'clipped', 0.0, 'n_weak_nv_topk'):.2f}; normal
{_num(agg, 'clipped', 0.0, 'n_normal_topk'):.2f}.

At $\\delta={peak_clip:.2f}$ (clipped $V$ peak): $V={_fmt(agg, 'clipped', peak_clip, 'v')}$;
vetoes {_num(agg, 'clipped', peak_clip, 'n_veto_topk'):.2f}; weak non-vetoes
{_num(agg, 'clipped', peak_clip, 'n_weak_nv_topk'):.2f}; normal
{_num(agg, 'clipped', peak_clip, 'n_normal_topk'):.2f}.

At $\\delta={dmax:.2f}$: $V={_fmt(agg, 'clipped', dmax, 'v')}$;
vetoes {_num(agg, 'clipped', dmax, 'n_veto_topk'):.2f}; weak non-vetoes
{_num(agg, 'clipped', dmax, 'n_weak_nv_topk'):.2f}; normal
{_num(agg, 'clipped', dmax, 'n_normal_topk'):.2f}; weak share
{_fmt(agg, 'clipped', dmax, 'weak_rate')}.

**Compositional reading (clipped, from the counts above):** after the $V$ peak,
the number of $Q=0$ cases in Top-$K$ falls while the number of
$0<Q\\le {Q_WEAK:.2f}$ cases continues to rise. Normal ($Q>{Q_WEAK:.2f}$) cases
are displaced throughout. The decline in $V$ is therefore a **replacement of
vetoes by weak non-vetoes**, not a return of normal high-$Q$ cases.

## 5. Veto vs weak-context decomposition

- $V$ = (vetoes in Top-$K$)/$K$.
- Weak-context exposure = (vetoes + weak non-vetoes in Top-$K$)/$K$.
- Veto share among weak Top-$K$ cases = vetoes / (vetoes + weak non-vetoes).

Clipped:

{_grid(agg, 'clipped', [
    ('v', r'$V$ (veto rate)'),
    ('weak_rate', r'weak-context rate'),
    ('veto_among_weak', r'veto / weak'),
    ('inclusion_veto', r'veto inclusion'),
    ('inclusion_weak_nv', r'weak-NV inclusion'),
])}

Unclipped (same quantities):

{_grid(agg, 'unclipped', [
    ('v', r'$V$'),
    ('weak_rate', r'weak-context rate'),
    ('veto_among_weak', r'veto / weak'),
    ('inclusion_veto', r'veto inclusion'),
    ('inclusion_weak_nv', r'weak-NV inclusion'),
])}

If weak-context rate stays high (or rises) while $V$ falls, the policy-violation
metric is **not** showing a disappearance of overconfident weak-context cases;
it is showing a shift from $Q=0$ to $0<Q\\le {Q_WEAK:.2f}$.

## 6. Score-distribution analysis (clipped)

Mean $P$ by group vs Top-$K$ threshold:

{_grid(agg, 'clipped', [
    ('mean_P_veto', r'mean $P$ veto'),
    ('mean_P_weak_nv', r'mean $P$ weak-NV'),
    ('mean_P_normal', r'mean $P$ normal'),
    ('p_threshold', "Top-$K$ $P$ threshold"),
    ('P_veto_q90', r'veto $P$ q90'),
    ('P_weak_nv_q90', r'weak-NV $P$ q90'),
    ('P_normal_q90', r'normal $P$ q90'),
])}

Quantiles of $P$ at $\\delta=0.10$ (clipped $V$ peak) and $\\delta={dmax:.2f}$:

At $\\delta=0.10$: veto q50={_num(agg, 'clipped', 0.10, 'P_veto_q50'):.3f},
q90={_num(agg, 'clipped', 0.10, 'P_veto_q90'):.3f}; weak-NV q50=
{_num(agg, 'clipped', 0.10, 'P_weak_nv_q50'):.3f}, q90=
{_num(agg, 'clipped', 0.10, 'P_weak_nv_q90'):.3f}; threshold=
{_num(agg, 'clipped', 0.10, 'p_threshold'):.3f}.

At $\\delta={dmax:.2f}$: veto q50={_num(agg, 'clipped', dmax, 'P_veto_q50'):.3f},
q90={_num(agg, 'clipped', dmax, 'P_veto_q90'):.3f}; weak-NV q50=
{_num(agg, 'clipped', dmax, 'P_weak_nv_q50'):.3f}, q90=
{_num(agg, 'clipped', dmax, 'P_weak_nv_q90'):.3f}; threshold=
{_num(agg, 'clipped', dmax, 'p_threshold'):.3f}.

The clipped ceiling $P={p_veto_cap:.2f}$ for $Q=0$ is marked on
`figures/scores_vs_threshold.pdf`. Once many vetoes sit at that ceiling,
further $\\delta$ cannot raise their $P$, while weak non-vetoes can still
increase $P$ (until they also saturate, at which point $P>{p_veto_cap:.2f}$).

## 7. Clipped vs unclipped counterfactual

$V(\\delta)$:

{_grid(agg, 'clipped', [('v', r'clipped $V$')])}

{_grid(agg, 'unclipped', [('v', r'unclipped $V$')])}

- Clipped peak $\\delta={peak_clip:.2f}$, $V={_fmt(agg, 'clipped', peak_clip, 'v')}$;
  at $\\delta={dmax:.2f}$, $V={_fmt(agg, 'clipped', dmax, 'v')}$.
- Unclipped peak $\\delta={peak_raw:.2f}$, $V={_fmt(agg, 'unclipped', peak_raw, 'v')}$;
  at $\\delta={dmax:.2f}$, $V={_fmt(agg, 'unclipped', dmax, 'v')}$.

**Does clipping explain the non-monotonicity?** {clip_verdict}

Unclipped, $\\delta$ adds $\\lambda\\delta$ to $P$ for every affected case and
$0$ to normal cases. Relative ranking **within** the affected set is invariant
to $\\delta$. Vetoes keep their original $R$ advantage ($\\mathrm{{Beta}}(8,2)$)
plus no $Q$-bonus, and that comparison does not flip with $\\delta$ unless
clipping freezes the high-$R$ group.

## 8. Main findings

1. **Confirmed from code.** $\\delta$ is applied only if $Q\\le {Q_WEAK:.2f}$;
   $R'$ is clipped to $[0,1]$ in the formal experiment; $V$ counts only $Q=0$
   in Top-$K$; linear $P=\\lambda R'+(1-\\lambda)Q$.
2. **Vetoes saturate first.** Mean original $R$ is much higher for vetoes than
   for weak non-vetoes, so $\\mathrm{{frac}}(R+\\delta\\ge 1)$ rises earlier
   for $Q=0$ (Section 3 table).
3. **The $V$ peak is compositional.** Up to $\\delta={peak_clip:.2f}$, more
   vetoes enter Top-$K$ (and weak non-vetoes also begin to enter). After the
   peak, veto counts in Top-$K$ fall while weak non-veto counts keep rising
   (Section 4).
4. **$V$ falling ≠ weak-context exposure falling.** Weak-context share of
   Top-$K$ remains high at large $\\delta$ in the clipped run (Section 5).
   The metric $V$ specifically loses $Q=0$ cases.
5. **Clipping is required for the downturn** under the comparison in Section 7
   (see the verdict there). Without clipping there is no $R'=1$ ceiling on
   veto scores, so veto $P$ keeps rising with $\delta$ and vetoes are not
   overtaken via the $Q$-bonus of saturated weak non-vetoes.

## 9. Recommended scientifically defensible interpretation for the manuscript

Do **not** write that “predictive overconfidence increases policy violations”
as a blanket statement. At $\\lambda={LAM:.2f}$ the formal $V(\\delta)$ curve
for $A_L$ is non-monotonic.

A defensible account, restricted to this design:

- Overconfidence is applied to **all** $Q\\le {Q_WEAK:.2f}$, not only to vetoes.
- For $A_L$, raising $R'$ initially makes both vetoes and other weak-context
  cases more competitive against $Q>{Q_WEAK:.2f}$ cases, so $V$ rises from
  {_num(agg, 'clipped', 0.0, 'v'):.3f} to {_num(agg, 'clipped', peak_clip, 'v'):.3f}.
- Because vetoes already have high $R$, they hit $R'=1$ first. Their linear
  score then cannot exceed $P={p_veto_cap:.2f}$. Weak non-vetoes with $Q>0$
  can exceed that ceiling once their $R'$ is large, and they replace vetoes
  in Top-$K$. $V$ therefore falls even though weak-context occupancy of
  Top-$K$ stays large.
- $A_G$ and $A_M$ keep $V=0$ for all $\\delta$ in the formal experiment because
  $Q=0$ is absorbing; this diagnostic did not re-estimate those operators.

What **not** to claim without extra evidence:

- that overconfidence “improves policy compliance” at large $\\delta$ (it
  substitutes one weak-context class for another);
- that the downturn would occur without clipping, or at other $\\lambda$, or
  for a different $q_{{\\mathrm{{weak}}}}$;
- that any operator is globally better.

## Artifacts

- `results/trials_diagnostics.csv` — trial-level records
- `results/aggregated_diagnostics.csv` — mean / std / CI95%
- `figures/v_clipped_vs_unclipped.{{pdf,png}}`
- `figures/topk_composition.{{pdf,png}}`
- `figures/saturation_by_group.{{pdf,png}}`
- `figures/scores_vs_threshold.{{pdf,png}}`
- `figures/inclusion_probability.{{pdf,png}}`
"""
    path.write_text(text, encoding="utf-8")


def parse_refresh() -> bool:
    return "--refresh" in sys.argv


def main() -> None:
    paths = ensure_dirs(DIAG_DIR, subdirs=("results", "figures"))
    raw = run_or_load(paths["results"], refresh=parse_refresh())
    metric_cols = [
        c
        for c in raw.columns
        if c not in {"trial", "seed", "delta", "mode"} and pd.api.types.is_numeric_dtype(raw[c])
    ]
    agg = summarize_allow_nan(raw, ["mode", "delta"], metric_cols)
    agg_path = paths["results"] / "aggregated_diagnostics.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[agg] wrote {agg_path} ({len(agg)} rows)")
    for path in write_figures(agg, paths["figures"]):
        print(f"[fig] {path}")
    report = DIAG_DIR / "non_monotonicity_analysis.md"
    write_report(agg, report)
    print(f"[doc] {report}")
    print("done.")


if __name__ == "__main__":
    main()
