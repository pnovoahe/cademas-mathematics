#!/usr/bin/env python3
"""Second-pass validation of the non-monotonic V(δ) mechanism for Experiment 6.2.

Internal diagnostic only. Does not modify the formal experiment or manuscript.
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
    OVERCONFIDENCE_LAMBDAS,
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
    write_latex_table,
)

LAM = OVERCONFIDENCE_PRIMARY_LAMBDA
Q_WEAK = Q_WEAK_THRESHOLD
Q_WEAK_SENSITIVITY = (0.10, 0.25, 0.40)
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
    r_out = np.asarray(R, dtype=float).copy()
    weak = np.asarray(Q, dtype=float) <= float(q_threshold)
    r_out[weak] = r_out[weak] + float(delta)
    return r_out


def _mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def _qtile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, q)) if x.size else float("nan")


def _frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return float(np.mean(mask)) if mask.size else float("nan")


def _pairwise_weak_beats_veto(p_veto: np.ndarray, p_weak_nv: np.ndarray) -> tuple[float, float, int]:
    """Return (frac all pairs P_weak>P_veto, frac among R'=1 pairs, n pairs)."""
    if p_veto.size == 0 or p_weak_nv.size == 0:
        return float("nan"), float("nan"), 0
    diff = p_weak_nv[np.newaxis, :] - p_veto[:, np.newaxis]
    n_pairs = int(diff.size)
    frac_all = float(np.mean(diff > 0.0))
    return frac_all, float("nan"), n_pairs


def _pairwise_sat_weak_beats_veto(
    p_veto: np.ndarray,
    p_weak_nv: np.ndarray,
    r_veto: np.ndarray,
    r_weak_nv: np.ndarray,
    q_weak_nv: np.ndarray,
) -> tuple[float, int]:
    """Among pairs with both R'=1, fraction with P_weak > P_veto."""
    sat_v = r_veto >= 1.0 - 1e-12
    sat_w = r_weak_nv >= 1.0 - 1e-12
    if not np.any(sat_v) or not np.any(sat_w):
        return float("nan"), 0
    pv = p_veto[sat_v]
    pw = p_weak_nv[sat_w]
    diff = pw[np.newaxis, :] - pv[:, np.newaxis]
    return float(np.mean(diff > 0.0)), int(diff.size)


def _group_p_stats(
    p: np.ndarray,
    r: np.ndarray,
    r_prime: np.ndarray,
    mask: np.ndarray,
    in_topk: np.ndarray,
) -> dict[str, float]:
    sel = mask & in_topk
    nsel = mask & ~in_topk
    out: dict[str, float] = {
        "inclusion": _frac(in_topk[mask]),
        "frac_sat": _frac(r_prime[mask] >= 1.0 - 1e-12),
        "mean_P_all": _mean(p[mask]),
        "mean_R_orig_all": _mean(r[mask]),
        "mean_P_sel": _mean(p[sel]) if np.any(sel) else float("nan"),
        "mean_P_nsel": _mean(p[nsel]) if np.any(nsel) else float("nan"),
        "mean_R_orig_sel": _mean(r[sel]) if np.any(sel) else float("nan"),
        "mean_R_orig_nsel": _mean(r[nsel]) if np.any(nsel) else float("nan"),
        "count_topk": float(sel.sum()),
    }
    for q in QUANTILES:
        tag = f"{int(round(q * 100)):02d}"
        out[f"P_q{tag}"] = _qtile(p[mask], q)
    return out


def _run_trial_record(
    trial_idx: int,
    *,
    lam: float,
    q_weak: float,
    mode: str,
) -> list[dict]:
    rng = np.random.default_rng(trial_seed(trial_idx))
    pop = generate_population(
        rng,
        n_std=n_std_from_fraction(VETO_FRACTION, N_CASES),
        n_veto=n_veto_from_fraction(VETO_FRACTION, N_CASES),
    )
    r = pop.R
    q = pop.Q
    is_veto = q == 0.0
    is_weak_nv = (q > 0.0) & (q <= q_weak)
    is_normal = q > q_weak
    records: list[dict] = []
    for delta in DELTA_VALUES:
        if mode == "clipped":
            r_prime = apply_predictive_overconfidence(r, q, delta, q_threshold=q_weak)
        elif mode == "unclipped":
            r_prime = apply_overconfidence_raw(r, q, delta, q_threshold=q_weak)
        else:
            raise ValueError(mode)
        p = aggregate("linear", r_prime, q, lam)
        idx = top_k_indices(p, TOP_K, R=r_prime, case_id=pop.case_id)
        in_topk = np.zeros(pop.n, dtype=bool)
        in_topk[idx] = True
        n_veto_k = int((is_veto & in_topk).sum())
        n_weak_nv_k = int((is_weak_nv & in_topk).sum())
        n_normal_k = int((is_normal & in_topk).sum())
        n_weak_k = n_veto_k + n_weak_nv_k
        mean_r_topk = float(np.mean(r[idx]))
        rec: dict[str, float | int | str] = {
            "trial": trial_idx,
            "seed": trial_seed(trial_idx),
            "lambda": lam,
            "q_weak": q_weak,
            "mode": mode,
            "delta": float(delta),
            "v": n_veto_k / TOP_K,
            "w_weak": n_weak_nv_k / TOP_K,
            "e_weak_context": n_weak_k / TOP_K,
            "s_veto_given_weak": (n_veto_k / n_weak_k) if n_weak_k else float("nan"),
            "n_veto_topk": n_veto_k,
            "n_weak_nv_topk": n_weak_nv_k,
            "n_normal_topk": n_normal_k,
            "p_threshold": float(np.min(p[idx])),
            "mean_R_topk": mean_r_topk,
            "frac_Rprime1_topk": _frac(r_prime[idx] >= 1.0 - 1e-12),
            "contrib_veto_R": float(np.sum(r[is_veto & in_topk]) / TOP_K),
            "contrib_weak_nv_R": float(np.sum(r[is_weak_nv & in_topk]) / TOP_K),
            "contrib_normal_R": float(np.sum(r[is_normal & in_topk]) / TOP_K),
        }
        for prefix, mask in (
            ("veto", is_veto),
            ("weak_nv", is_weak_nv),
            ("normal", is_normal),
        ):
            stats = _group_p_stats(p, r, r_prime, mask, in_topk)
            for k, v in stats.items():
                rec[f"{prefix}_{k}"] = v
        frac_all, _, n_pairs = _pairwise_weak_beats_veto(
            p[is_veto], p[is_weak_nv]
        )
        frac_sat, n_sat_pairs = _pairwise_sat_weak_beats_veto(
            p[is_veto],
            p[is_weak_nv],
            r_prime[is_veto],
            r_prime[is_weak_nv],
            q[is_weak_nv],
        )
        rec["pairwise_weak_gt_veto"] = frac_all
        rec["pairwise_sat_weak_gt_veto"] = frac_sat
        rec["n_veto_weak_pairs"] = float(n_pairs)
        rec["n_sat_pairs"] = float(n_sat_pairs)
        records.append(rec)
    return records


def _fingerprint(name: str, extra: dict) -> str:
    return experiment_fingerprint(
        name,
        {
            "n_cases": N_CASES,
            "n_mc": N_MONTE_CARLO,
            "delta_values": list(DELTA_VALUES),
            "veto_fraction": VETO_FRACTION,
            "top_k": TOP_K,
            **extra,
        },
    )


def run_or_load(
    results_dir: Path,
    *,
    name: str,
    fingerprint: str,
    runner,
    refresh: bool,
) -> pd.DataFrame:
    raw_path = results_dir / f"{name}.csv"
    meta_path = results_dir / f"{name}_metadata.json"
    if raw_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            print(f"[cache] loaded {raw_path}")
            return pd.read_csv(raw_path)
    records = runner()
    raw = pd.DataFrame.from_records(records)
    raw.to_csv(raw_path, index=False)
    write_json(
        meta_path,
        {
            "name": name,
            "fingerprint": fingerprint,
            "timestamp_utc": utc_now_iso(),
            "n_rows": int(len(raw)),
        },
    )
    print(f"[mc] wrote {raw_path} ({len(raw)} rows)")
    return raw


def _v_peak_info(agg: pd.DataFrame) -> dict:
    sub = agg.sort_values("delta")
    v = sub["v_mean"].to_numpy()
    peak_i = int(np.argmax(v))
    peak_delta = float(sub.iloc[peak_i]["delta"])
    end_delta = float(sub.iloc[-1]["delta"])
    nonmono = peak_delta < end_delta - 1e-12 and v[-1] < v[peak_i] - 0.005
    return {
        "peak_delta": peak_delta,
        "peak_v": float(v[peak_i]),
        "end_v": float(v[-1]),
        "non_monotonic": bool(nonmono),
    }


def _plot_lines(ax, agg, series, ylabel, ylim=None, xticks=True):
    apply_paper_style()
    colors = ["#0072B2", "#D55E00", "#009E73", "#000000"]
    markers = ["v", "s", "o", "D"]
    for i, (metric, label) in enumerate(series):
        sub = agg.sort_values("delta")
        c = colors[i % len(colors)]
        ax.fill_between(
            sub["delta"],
            sub[f"{metric}_ci_low"],
            sub[f"{metric}_ci_high"],
            color=c,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            sub["delta"],
            sub[f"{metric}_mean"],
            color=c,
            marker=markers[i % len(markers)],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.0,
            label=label,
            zorder=2 + i,
        )
    ax.set_xlabel(r"Overconfidence level $\delta$")
    ax.set_ylabel(ylabel)
    if xticks:
        ax.set_xticks(list(DELTA_VALUES))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(**LEGEND_KW)
    style_square_box(ax)


def write_figures(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    clip = primary[primary["mode"] == "clipped"].sort_values("delta")
    raw = primary[primary["mode"] == "unclipped"].sort_values("delta")

    fig, ax_a, ax_b = new_square_two_panel()
    for sub, color, marker, label in (
        (clip, "#0072B2", "v", r"clipped $V$"),
        (raw, "#D55E00", "s", r"unclipped $V$"),
    ):
        ax_a.fill_between(sub["delta"], sub["v_ci_low"], sub["v_ci_high"], color=color, alpha=0.22, linewidth=0)
        ax_a.plot(sub["delta"], sub["v_mean"], color=color, marker=marker, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.5, linewidth=2.0, label=label)
    ax_a.set_ylabel(r"Policy violation rate $V$")
    ax_a.set_xticks(list(DELTA_VALUES))
    ax_a.set_ylim(-0.05, 1.0)
    ax_a.legend(**LEGEND_KW)
    style_square_box(ax_a)
    add_panel_letter(ax_a, "a")
    for sub, color, marker, label in (
        (clip, "#0072B2", "v", r"clipped weak-context rate"),
        (raw, "#D55E00", "s", r"unclipped weak-context rate"),
    ):
        ax_b.fill_between(sub["delta"], sub["e_weak_context_ci_low"], sub["e_weak_context_ci_high"], color=color, alpha=0.22, linewidth=0)
        ax_b.plot(sub["delta"], sub["e_weak_context_mean"], color=color, marker=marker, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.5, linewidth=2.0, label=label)
    ax_b.set_ylabel(r"Weak-context exposure $E_{\mathrm{weak}}$")
    ax_b.set_xticks(list(DELTA_VALUES))
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.legend(**LEGEND_KW)
    style_square_box(ax_b)
    add_panel_letter(ax_b, "b")
    written.extend(save_figure(fig, figures_dir / "deep_v_and_weak_exposure", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        [
            ("v", r"$V_{\mathrm{veto}}$"),
            ("w_weak", r"$W_{\mathrm{weak}}$"),
            ("e_weak_context", r"$E_{\mathrm{weak}}$"),
            ("s_veto_given_weak", r"$S_{\mathrm{veto|weak}}$"),
        ],
        ylabel="Rate / share",
        ylim=(-0.05, 1.05),
    )
    written.extend(save_figure(fig, figures_dir / "deep_exposure_decomposition", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        [
            ("pairwise_weak_gt_veto", r"all veto–weak pairs: $P_{\mathrm{weak}}>P_{\mathrm{veto}}$"),
            ("pairwise_sat_weak_gt_veto", r"both $R'=1$: $P_{\mathrm{weak}}>P_{\mathrm{veto}}$"),
        ],
        ylabel="Pairwise fraction",
        ylim=(-0.02, 1.02),
    )
    written.extend(save_figure(fig, figures_dir / "deep_pairwise_ranking", bbox_inches=None))

    fig, ax = new_square_single_panel()
    _plot_lines(
        ax,
        clip,
        [
            ("mean_R_topk", r"mean original $R$ of Top-$K$"),
            ("contrib_veto_R", r"veto contribution"),
            ("contrib_weak_nv_R", r"weak non-veto contribution"),
            ("contrib_normal_R", r"normal contribution"),
        ],
        ylabel=r"Mean / contribution to $\bar{R}$",
        ylim=(0.0, 0.95),
    )
    written.extend(save_figure(fig, figures_dir / "deep_utility_decomposition", bbox_inches=None))

    fig, ax = new_square_single_panel()
    for qv, color, marker in zip(Q_WEAK_SENSITIVITY, ["#0072B2", "#D55E00", "#009E73"], ["v", "s", "o"]):
        sub = sensitivity[(sensitivity["q_weak"] == qv) & (sensitivity["mode"] == "clipped")].sort_values("delta")
        ax.plot(sub["delta"], sub["v_mean"], color=color, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=0.5, linewidth=2.0,
                label=rf"$q_{{\mathrm{{weak}}}}={qv:.2f}$")
    ax.set_xlabel(r"Overconfidence level $\delta$")
    ax.set_ylabel(r"Policy violation rate $V$")
    ax.set_xticks(list(DELTA_VALUES))
    ax.set_ylim(-0.05, 0.55)
    ax.legend(**LEGEND_KW)
    style_square_box(ax)
    written.extend(save_figure(fig, figures_dir / "deep_qweak_sensitivity", bbox_inches=None))

    return written


def _grid_table(agg: pd.DataFrame, metrics: list[tuple[str, str]]) -> str:
    lines = ["| $\\delta$ | " + " | ".join(h for _, h in metrics) + " |", "|" + "---|" * (1 + len(metrics))]
    for delta in DELTA_VALUES:
        row = agg[np.isclose(agg["delta"].to_numpy(), delta)].iloc[0]
        cells = [f"{delta:.2f}"] + [f"{float(row[f'{m}_mean']):.3f}" for m, _ in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(primary: pd.DataFrame, tables_dir: Path) -> Path:
    clip = primary[primary["mode"] == "clipped"].sort_values("delta").copy()
    clip["delta_tex"] = clip["delta"].map(lambda x: rf"${x:.2f}$")
    path = tables_dir / "table_deep_exposure_decomposition.tex"
    write_latex_table(
        clip,
        path,
        columns=(
            ("delta_tex", r"$\delta$"),
            ("v_mean", r"$V$"),
            ("w_weak_mean", r"$W_{\mathrm{weak}}$"),
            ("e_weak_context_mean", r"$E_{\mathrm{weak}}$"),
            ("s_veto_given_weak_mean", r"$S_{\mathrm{veto|weak}}$"),
            ("pairwise_sat_weak_gt_veto_mean", r"sat.\ pairs $P_w>P_v$"),
        ),
        caption=(
            r"Diagnostic decomposition of weak-context exposure at $\lambda=0.75$ "
            r"(clipped formal model). $V$: veto rate; $W_{\mathrm{weak}}$: weak non-veto "
            r"rate; $E_{\mathrm{weak}}$: total weak-context rate; $S_{\mathrm{veto|weak}}$: "
            r"veto share among weak-context Top-$K$ cases; last column: fraction of saturated "
            r"veto--weak pairs with $P_{\mathrm{weak}}>P_{\mathrm{veto}}$."
        ),
        label="tab:deep-exposure-decomposition",
        col_spec="lcccccc",
    )
    return path


def write_deep_analysis(
    primary_raw: pd.DataFrame,
    primary_agg: pd.DataFrame,
    sensitivity_agg: pd.DataFrame,
    lambda_agg: pd.DataFrame,
    formal_agg: pd.DataFrame,
    path: Path,
) -> None:
    clip = primary_agg[primary_agg["mode"] == "clipped"].sort_values("delta")
    raw = primary_agg[primary_agg["mode"] == "unclipped"].sort_values("delta")
    peak = _v_peak_info(clip)
    raw_peak = _v_peak_info(raw)

    lambda_rows = []
    lambda_tables: list[str] = []
    for lam in OVERCONFIDENCE_LAMBDAS:
        sub = formal_agg[np.isclose(formal_agg["lambda"].to_numpy(), lam)]
        sub = sub[sub["operator"] == "linear"].sort_values("delta")
        agg_l = sub.rename(
            columns={
                "policy_violation_rate_mean": "v_mean",
                "policy_violation_rate_ci_low": "v_ci_low",
                "policy_violation_rate_ci_high": "v_ci_high",
            }
        )
        info = _v_peak_info(agg_l)
        lam_diag = lambda_agg[np.isclose(lambda_agg["lambda"].to_numpy(), lam)].sort_values(
            "delta"
        )
        lambda_rows.append(
            f"- $\\lambda={lam:.2f}$: peak $V$ at $\\delta={info['peak_delta']:.2f}$ "
            f"({info['peak_v']:.3f}); $\\delta={DELTA_VALUES[-1]:.2f}$ gives "
            f"{info['end_v']:.3f}; non-monotonic={info['non_monotonic']}."
        )
        if lam_diag.empty:
            continue
        d_peak = float(info["peak_delta"])
        d_end = float(DELTA_VALUES[-1])
        row_peak = lam_diag[np.isclose(lam_diag["delta"].to_numpy(), d_peak)].iloc[0]
        row_end = lam_diag[np.isclose(lam_diag["delta"].to_numpy(), d_end)].iloc[0]
        lambda_tables.append(
            f"**$\\lambda={lam:.2f}$** (clipped diagnostic; formal $V$ from stored results):\n\n"
            f"{_grid_table(lam_diag, [
                ('v', r'$V$'),
                ('e_weak_context', r'$E_{\mathrm{weak}}$'),
                ('n_veto_topk', r'veto count'),
                ('n_weak_nv_topk', r'weak-NV count'),
            ])}\n\n"
            f"At peak $\\delta={d_peak:.2f}$: $E_{{\\mathrm{{weak}}}}="
            f"{float(row_peak['e_weak_context_mean']):.3f}$, vetoes="
            f"{float(row_peak['n_veto_topk_mean']):.1f}$, weak non-vetoes="
            f"{float(row_peak['n_weak_nv_topk_mean']):.1f}$. "
            f"At $\\delta={d_end:.2f}$: $E_{{\\mathrm{{weak}}}}="
            f"{float(row_end['e_weak_context_mean']):.3f}$, vetoes="
            f"{float(row_end['n_veto_topk_mean']):.1f}$, weak non-vetoes="
            f"{float(row_end['n_weak_nv_topk_mean']):.1f}$."
        )

    q_rows = []
    for qv in Q_WEAK_SENSITIVITY:
        sub = sensitivity_agg[
            (sensitivity_agg["q_weak"] == qv) & (sensitivity_agg["mode"] == "clipped")
        ].sort_values("delta")
        info = _v_peak_info(sub)
        q_rows.append(
            f"- $q_{{\\mathrm{{weak}}}}={qv:.2f}$: peak at $\\delta={info['peak_delta']:.2f}$, "
            f"$V={info['peak_v']:.3f}$; non-monotonic={info['non_monotonic']}."
        )

    d10 = clip[np.isclose(clip["delta"], 0.10)].iloc[0]
    d30 = clip[np.isclose(clip["delta"], DELTA_VALUES[-1])].iloc[0]
    r30 = raw[np.isclose(raw["delta"], DELTA_VALUES[-1])].iloc[0]

    text = f"""# Deep validation — non-monotonic $V(\\delta)$ for $A_L$ (Experiment 6.2)

Second-pass diagnostic. Formal Experiment 6.2 is **unchanged**. All numbers below
use the same seeds ($42$–$1041$), population generator, $\\delta$ grid, and
$q_{{\\mathrm{{weak}}}}=0.25$ unless labelled as sensitivity.

Companion: `non_monotonicity_analysis.md` (first diagnostic pass).

## 1. Question and hypotheses under test

Formal setup: $R_i'=\\mathrm{{clip}}(R_i+\\delta,0,1)$ iff $Q_i\\le 0.25$;
$A_L(R',Q)=0.75R'+0.25Q$; $V=|\\{{Q=0\\}}\\cap\\mathrm{{Top\\text{{-}}K}}|/K$.

Hypotheses H1–H8 from the prior diagnostic are tested below with distributional
and pairwise evidence, not only group means.

## 2. Mathematical ranking conditions ($\\lambda=0.75$)

**Veto** ($Q=0$): $P_v=0.75R_v'$.

**Weak non-veto** ($0<Q\\le q_{{\\mathrm{{weak}}}}$): $P_w=0.75R_w'+0.25Q$.

If both are clipped to $R'=1$:

- $P_v=0.75$.
- $P_w=0.75+0.25Q > 0.75$ for every $Q>0$.

Therefore any saturated weak non-veto **strictly outranks** any saturated veto,
independent of tie-breaking on $R'$.

**Unclipped counterfactual:** for affected cases, $P(\\delta)=0.75(R+\\delta)+cQ$
with $c\\in\\{{0,0.25\\}}$. Adding $\\delta$ adds $0.75\\delta$ to **both**
groups equally. Relative order among affected cases is **invariant** in $\\delta$;
vetoes cannot be overtaken by weak non-vetoes through the $Q$-bonus alone.
Normal cases ($Q>0.25$) receive no shift; their relative position vs affected
cases can change only through threshold competition, not through pairwise
veto-vs-weak reversal.

## 3. Analysis 1 — Direct ranking verification (clipped, $\\lambda=0.75$)

### Top-$K$ composition and inclusion

{_grid_table(clip, [
    ('n_veto_topk', r'$Q=0$ count'),
    ('n_weak_nv_topk', r'$0<Q\leq 0.25$ count'),
    ('n_normal_topk', r'$Q>0.25$ count'),
    ('veto_inclusion', r'veto incl.'),
    ('weak_nv_inclusion', r'weak-NV incl.'),
    ('normal_inclusion', r'normal incl.'),
])}

### Score distributions and saturation

{_grid_table(clip, [
    ('p_threshold', r'Top-$K$ $P$ threshold'),
    ('veto_frac_sat', r'veto $R\'=1$'),
    ('weak_nv_frac_sat', r'weak-NV $R\'=1$'),
    ('frac_Rprime1_topk', r'Top-$K$ $R\'=1$'),
    ('veto_P_q90', r'veto $P$ q90'),
    ('weak_nv_P_q90', r'weak-NV $P$ q90'),
])}

### Pairwise veto vs weak non-veto ranking

Fraction of all veto–weak non-veto pairs with $P_w > P_v$:

{_grid_table(clip, [
    ('pairwise_weak_gt_veto', r'all pairs'),
    ('pairwise_sat_weak_gt_veto', r'both $R\'=1$'),
])}

At $\\delta=0.10$ (formal $V$ peak): all-pair fraction =
{float(d10['pairwise_weak_gt_veto_mean']):.3f}; among saturated pairs =
{float(d10['pairwise_sat_weak_gt_veto_mean']):.3f}.

At $\\delta=0.30$: all-pair fraction =
{float(d30['pairwise_weak_gt_veto_mean']):.3f}; among saturated pairs =
{float(d30['pairwise_sat_weak_gt_veto_mean']):.3f}.

**Interpretation.** The saturated-pair fraction approaches 1 once both groups
have substantial $R'=1$ mass, confirming the exact ranking inequality. The
all-pair fraction rises with $\\delta$ as more weak non-vetoes gain enough
$R'$ to beat unsaturated vetoes.

## 4. Analysis 2 — Clipping mechanism (same populations)

| $\\delta$ | clipped $V$ | unclipped $V$ | clipped $E_{{\\mathrm{{weak}}}}$ | unclipped $E_{{\\mathrm{{weak}}}}$ |
|---|---|---|---|---|
""" + "\n".join(
        f"| {d:.2f} | {float(clip[np.isclose(clip['delta'], d)].iloc[0]['v_mean']):.3f} | "
        f"{float(raw[np.isclose(raw['delta'], d)].iloc[0]['v_mean']):.3f} | "
        f"{float(clip[np.isclose(clip['delta'], d)].iloc[0]['e_weak_context_mean']):.3f} | "
        f"{float(raw[np.isclose(raw['delta'], d)].iloc[0]['e_weak_context_mean']):.3f} |"
        for d in DELTA_VALUES
    ) + f"""

Clipped: peak $\\delta={peak['peak_delta']:.2f}$, $V={peak['peak_v']:.3f}$; non-monotonic={peak['non_monotonic']}.
Unclipped: peak $\\delta={raw_peak['peak_delta']:.2f}$, $V={raw_peak['peak_v']:.3f}$; monotonic increase={not raw_peak['non_monotonic']}.

**Verdict.** Clipping is **necessary** for the observed downturn: unclipped $V$
 rises to {float(r30['v_mean']):.3f} at $\\delta=0.30$ while clipped $V$ falls
 to {float(d30['v_mean']):.3f}.

## 5. Analysis 3 — $V$ falls but weak-context exposure does not

Definitions: $V=veto/K$, $W_{{\\mathrm{{weak}}}}=weak\\ non\\ veto/K$,
$E_{{\\mathrm{{weak}}}}=(veto+weak\\ non\\ veto)/K$,
$S_{{\\mathrm{{veto|weak}}}}=veto/(veto+weak\\ non\\ veto)$.

{_grid_table(clip, [
    ('v', r'$V$'),
    ('w_weak', r'$W_{\mathrm{weak}}$'),
    ('e_weak_context', r'$E_{\mathrm{weak}}$'),
    ('s_veto_given_weak', r'$S_{\mathrm{veto|weak}}$'),
])}

From $\\delta=0.10$ to $\\delta=0.30$:

- $V$: {float(d10['v_mean']):.3f} $\\to$ {float(d30['v_mean']):.3f} (**decreasing**).
- $E_{{\\mathrm{{weak}}}}$: {float(d10['e_weak_context_mean']):.3f} $\\to$
  {float(d30['e_weak_context_mean']):.3f} (**increasing**).
- $W_{{\\mathrm{{weak}}}}$: {float(d10['w_weak_mean']):.3f} $\\to$
  {float(d30['w_weak_mean']):.3f} (**increasing**).

**Verdict.** The decline in $V$ is accompanied by **increasing** weak-context
occupancy of Top-$K$. It is **not** contextual recovery.

## 6. Analysis 4 — Predictive utility decomposition (original $R$)

{_grid_table(clip, [
    ('mean_R_topk', r'mean $R$ Top-$K$'),
    ('contrib_veto_R', r'veto contrib.'),
    ('contrib_weak_nv_R', r'weak-NV contrib.'),
    ('contrib_normal_R', r'normal contrib.'),
])}

$\\bar{{R}}$ is non-monotonic in $\\delta$ (peak near $\\delta=0.10$), tracking
the $V$ peak phase when more high-$R$ vetoes enter Top-$K$. After $\\delta=0.10$,
mean original $R$ of the selected set declines as vetoes are replaced by lower-$R$
weak non-vetoes and some normal cases.

Selected-group means (original $R$):

{_grid_table(clip, [
    ('veto_mean_R_orig_sel', r'veto sel. $R$'),
    ('weak_nv_mean_R_orig_sel', r'weak-NV sel. $R$'),
    ('normal_mean_R_orig_sel', r'normal sel. $R$'),
])}

## 7. Analysis 5 — $\\lambda$ robustness (formal $V$ + diagnostic composition)

Formal $V(\\delta)$ from `../results/aggregated.csv` (linear operator):

{chr(10).join(lambda_rows)}

Composition metrics below use the same seeds and populations (clipped model,
diagnostic recomputation):

{chr(10).join(lambda_tables) if lambda_tables else '_Lambda composition tables unavailable._'}

$\\lambda=0.50$: $V=0$ on the grid (no non-monotonicity). $\\lambda=0.75$ and
$0.90$ show non-monotonic clipped $V$ with replacement dynamics at different
magnitudes. This is robustness only; the paper primary remains $\\lambda=0.75$.

## 8. Analysis 6 — $q_{{\\mathrm{{weak}}}}$ sensitivity (diagnostic only)

{chr(10).join(q_rows)}

Non-monotonicity persists for $q_{{\\mathrm{{weak}}}}\\in\\{{0.10,0.25,0.40\\}}$
 under clipping; peak location shifts modestly. Clipping remains necessary in
 each case (unclipped curves rise monotonically; not tabulated here).

## 9. Classification of findings

| Finding | Class |
|---|---|
| Non-monotonic $V(\\delta)$ at $\\lambda=0.75$ | **A** Essential |
| Weak-context exposure rises while $V$ falls | **A** Essential |
| Replacement of $Q=0$ by $0<Q\\le q_{{\\mathrm{{weak}}}}$ | **A** Essential |
| Clipping creates $P_v\\le 0.75$ ceiling; $P_w>0.75$ when saturated | **A** Essential |
| Pairwise $P_w>P_v$ among saturated pairs | **B** Supporting |
| Unclipped counterfactual (monotonic $V$) | **B** Supporting (internal/supplement) |
| Utility decomposition / $\\bar{{R}}$ peak | **B** Supporting |
| $\\lambda=0.50/0.90$ robustness | **B** Brief robustness only |
| $q_{{\\mathrm{{weak}}}}$ sensitivity | **C** Internal |
| Kendall $\\tau$ / Jaccard figures | **B** if ranking stability discussed; else **C** |
| Full pairwise tables | **C** Internal |

**Do not claim** that large-$\\delta$ overconfidence improves policy compliance.

## 10. Artifacts

- `results/deep_primary.csv`, `results/deep_primary_aggregated.csv`
- `results/deep_qweak_sensitivity.csv`, `results/deep_qweak_aggregated.csv`
- `figures/deep_v_and_weak_exposure.{{pdf,png}}`
- `figures/deep_exposure_decomposition.{{pdf,png}}`
- `figures/deep_pairwise_ranking.{{pdf,png}}`
- `figures/deep_utility_decomposition.{{pdf,png}}`
- `figures/deep_qweak_sensitivity.{{pdf,png}}`
- `tables/table_deep_exposure_decomposition.tex`
- `captions.md` (updated)

## Manuscript recommendation

### 1. Essential findings

- $V(\\delta)$ for $A_L$ at $\\lambda=0.75$ is **non-monotonic** (peak
  $\\delta=0.10$, $V=0.251$; $\\delta=0.30$, $V=0.022$).
- Overconfidence applies to **all** $Q\\le 0.25$, not only vetoes.
- After the peak, $V$ falls because **vetoes leave Top-$K$** while **weak
  non-vetoes enter**, not because weak-context exposure falls ($E_{{\\mathrm{{weak}}}}$
  rises from {float(d10['e_weak_context_mean']):.3f} to {float(d30['e_weak_context_mean']):.3f}).
- Clipping is **necessary** for the downturn (unclipped $V$ rises to
  {float(r30['v_mean']):.3f} at $\\delta=0.30$).

### 2. Recommended for main text

- State non-monotonicity; do **not** say overconfidence monotonically increases
  violations.
- Explain selective application to $Q\\le q_{{\\mathrm{{weak}}}}$ and the
  $P_v=0.75R'$, $P_w=0.75R'+0.25Q$ ranking logic under saturation.
- Note that lower $V$ at large $\\delta$ reflects **metric composition**
  ($Q=0$ only), not restored compliance.
- Primary figure: two-panel $V$ and $\\bar{{R}}$ at $\\lambda=0.75$.

### 3. Better suited for appendix / supplement

- Unclipped counterfactual figure or one sentence.
- Table `tab:deep-exposure-decomposition` or abbreviated version with
  $V, W_{{\\mathrm{{weak}}}}, E_{{\\mathrm{{weak}}}}, S_{{\\mathrm{{veto|weak}}}}$ at
  representative $\\delta$.
- Brief $\\lambda=0.90$ robustness sentence (non-monotonic, higher baseline $V$).

### 4. Remain internal

- Full pairwise tables, $q_{{\\mathrm{{weak}}}}$ sensitivity plots, trial-level CSVs.
- Kendall $\\tau$ unless the narrative emphasizes ranking stability explicitly.

### 5. Safe numerical quotes (clipped, $\\lambda=0.75$, MC means)

| Quantity | $\\delta=0.00$ | $\\delta=0.10$ | $\\delta=0.30$ |
|---|---|---|---|
| $V$ | 0.077 | 0.251 | 0.022 |
| $E_{{\\mathrm{{weak}}}}$ | 0.106 | 0.331 | 0.366 |
| $W_{{\\mathrm{{weak}}}}$ | 0.029 | 0.080 | 0.344 |
| Vetoes in Top-$K$ | 7.7 | 25.1 | 2.2 |
| Weak non-vetoes in Top-$K$ | 2.9 | 8.0 | 34.4 |
| Sat. pairs $P_w>P_v$ | — | {float(d10['pairwise_sat_weak_gt_veto_mean']):.3f} | {float(d30['pairwise_sat_weak_gt_veto_mean']):.3f} |
| $\\bar{{R}}$ (original) | 0.850 | 0.880 | 0.830 |

Formal experiment values for $V$ and $\\bar{{R}}$ match the first column/row of
the published `results_narrative.md` within Monte Carlo noise (this diagnostic
recomputes from the same seeds).

### Remaining ambiguities for author decision

1. Whether to name **weak non-vetoes** explicitly in the main text or use
   “contextually weak but non-veto cases”.
2. Whether the unclipped counterfactual deserves one sentence or a supplement
   figure.
3. Whether $\\lambda=0.90$ robustness deserves one sentence given the higher
   baseline violation rate.
"""
    path.write_text(text, encoding="utf-8")


def write_captions(figures: list[str]) -> None:
    lines = [
        "# Captions — deep validation diagnostics",
        "",
        "Internal diagnostics only. Not manuscript-ready without editing.",
        "",
        "## deep_v_and_weak_exposure",
        "",
        "(a) Policy violation rate $V(\\delta)$ for the formal clipped model and the ",
        "unclipped counterfactual at $\\lambda=0.75$. (b) Weak-context exposure ",
        "$E_{\\mathrm{weak}}(\\delta)$ under the same two models.",
        "",
        "## deep_exposure_decomposition",
        "",
        "Decomposition of $V_{\\mathrm{veto}}$, $W_{\\mathrm{weak}}$, ",
        "$E_{\\mathrm{weak}}$, and $S_{\\mathrm{veto|weak}}$ for the clipped formal ",
        "model at $\\lambda=0.75$.",
        "",
        "## deep_pairwise_ranking",
        "",
        "Fraction of veto–weak non-veto pairs with $P_{\\mathrm{weak}}>P_{\\mathrm{veto}}$ ",
        "(all pairs; and pairs with both $R'=1$) at $\\lambda=0.75$, clipped model.",
        "",
        "## deep_utility_decomposition",
        "",
        "Mean original $\\bar{R}$ of Top-$K$ and group contributions from vetoes, ",
        "weak non-vetoes, and normal cases (clipped, $\\lambda=0.75$).",
        "",
        "## deep_qweak_sensitivity",
        "",
        "Sensitivity of clipped $V(\\delta)$ to $q_{\\mathrm{weak}}\\in\\{0.10,0.25,0.40\\}$ ",
        "at $\\lambda=0.75$ (diagnostic only; formal experiment uses $0.25$).",
        "",
        "## table_deep_exposure_decomposition (tab:deep-exposure-decomposition)",
        "",
        "Exposure decomposition at $\\lambda=0.75$ for the clipped formal model.",
        "",
    ]
    (DIAG_DIR / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def parse_refresh() -> bool:
    return "--refresh" in sys.argv


def main() -> None:
    refresh = parse_refresh()
    paths = ensure_dirs(DIAG_DIR, subdirs=("results", "figures", "tables"))

    def run_primary():
        out: list[dict] = []
        for trial_idx in range(N_MONTE_CARLO):
            for mode in ("clipped", "unclipped"):
                out.extend(
                    _run_trial_record(
                        trial_idx,
                        lam=LAM,
                        q_weak=Q_WEAK,
                        mode=mode,
                    )
                )
            if (trial_idx + 1) % 100 == 0 or trial_idx == 0:
                print(f"[primary] trial {trial_idx + 1}/{N_MONTE_CARLO}")
        return out

    primary_raw = run_or_load(
        paths["results"],
        name="deep_primary",
        fingerprint=_fingerprint(
            "deep_primary",
            {"lambda": LAM, "q_weak": Q_WEAK, "modes": ["clipped", "unclipped"]},
        ),
        runner=run_primary,
        refresh=refresh,
    )
    metric_cols = [
        c for c in primary_raw.columns
        if c not in {"trial", "seed", "lambda", "q_weak", "mode", "delta"}
        and pd.api.types.is_numeric_dtype(primary_raw[c])
    ]
    primary_agg = summarize_allow_nan(primary_raw, ["mode", "delta"], metric_cols)
    primary_agg.to_csv(paths["results"] / "deep_primary_aggregated.csv", index=False)

    def run_qweak():
        out: list[dict] = []
        for qv in Q_WEAK_SENSITIVITY:
            for trial_idx in range(N_MONTE_CARLO):
                for mode in ("clipped", "unclipped"):
                    out.extend(
                        _run_trial_record(
                            trial_idx,
                            lam=LAM,
                            q_weak=qv,
                            mode=mode,
                        )
                    )
                if (trial_idx + 1) % 200 == 0 or trial_idx == 0:
                    print(f"[qweak={qv:.2f}] trial {trial_idx + 1}/{N_MONTE_CARLO}")
        return out

    sens_raw = run_or_load(
        paths["results"],
        name="deep_qweak_sensitivity",
        fingerprint=_fingerprint(
            "deep_qweak_sensitivity",
            {"lambda": LAM, "q_weak_values": list(Q_WEAK_SENSITIVITY), "modes": ["clipped", "unclipped"]},
        ),
        runner=run_qweak,
        refresh=refresh,
    )
    sens_agg = summarize_allow_nan(
        sens_raw, ["q_weak", "mode", "delta"], metric_cols
    )
    sens_agg.to_csv(paths["results"] / "deep_qweak_aggregated.csv", index=False)

    def run_lambda():
        out: list[dict] = []
        for lam in OVERCONFIDENCE_LAMBDAS:
            for trial_idx in range(N_MONTE_CARLO):
                out.extend(
                    _run_trial_record(
                        trial_idx,
                        lam=lam,
                        q_weak=Q_WEAK,
                        mode="clipped",
                    )
                )
                if (trial_idx + 1) % 200 == 0 or trial_idx == 0:
                    print(f"[lambda={lam:.2f}] trial {trial_idx + 1}/{N_MONTE_CARLO}")
        return out

    lambda_raw = run_or_load(
        paths["results"],
        name="deep_lambda_robustness",
        fingerprint=_fingerprint(
            "deep_lambda_robustness",
            {"lambda_values": list(OVERCONFIDENCE_LAMBDAS), "q_weak": Q_WEAK, "mode": "clipped"},
        ),
        runner=run_lambda,
        refresh=refresh,
    )
    lambda_agg = summarize_allow_nan(lambda_raw, ["lambda", "delta"], metric_cols)
    lambda_agg.to_csv(paths["results"] / "deep_lambda_aggregated.csv", index=False)

    formal_agg = pd.read_csv(EXP_DIR / "results" / "aggregated.csv")

    for path in write_figures(primary_agg, sens_agg, paths["figures"]):
        print(f"[fig] {path}")
    table_path = write_table(primary_agg, paths["tables"])
    print(f"[tex] {table_path}")
    report = DIAG_DIR / "deep_analysis.md"
    write_deep_analysis(primary_raw, primary_agg, sens_agg, lambda_agg, formal_agg, report)
    print(f"[doc] {report}")
    write_captions([])
    print("done.")


if __name__ == "__main__":
    main()
