"""Shared utilities: Monte Carlo summaries, fingerprints, I/O helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from common.config import CI_PERCENTILES, SEEDS


def trial_seed(trial_idx: int) -> int:
    """Return the registered seed for Monte Carlo trial ``trial_idx``."""
    return int(SEEDS[trial_idx])


def ensure_dirs(exp_dir: Path, subdirs: Sequence[str] = ("results", "figures", "tables")) -> dict[str, Path]:
    """Create standard experiment subdirectories and return their paths."""
    exp_dir = Path(exp_dir)
    paths = {"root": exp_dir}
    for name in subdirs:
        path = exp_dir / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def mean_std_ci(values: Iterable[float]) -> tuple[float, float, float, float]:
    """Return mean, sample std, and 95% percentile CI."""
    arr = np.asarray(list(values), dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    lo = float(np.percentile(arr, CI_PERCENTILES[0]))
    hi = float(np.percentile(arr, CI_PERCENTILES[1]))
    return mean, std, lo, hi


def summarize_trials(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Aggregate trial-level metrics into mean, std, and 95% CI columns."""
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_cols, keys))
        for metric in metrics:
            mean, std, lo, hi = mean_std_ci(sub[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_fingerprint(name: str, params: dict[str, Any]) -> str:
    """Stable hash of experiment identity and configuration."""
    payload = {"name": name, **params}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_for_metadata(path: Path, *, base: Path) -> str:
    """Return a portable relative POSIX path for JSON metadata."""
    resolved = Path(path).resolve()
    root = Path(base).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    """Plain-text interval used in manuscript snippets."""
    return f"{mean:.{digits}f} (95% CI [{lo:.{digits}f}, {hi:.{digits}f}])"


def format_ci_tex(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    return rf"{mean:.{digits}f} (95\% CI $[{lo:.{digits}f},{hi:.{digits}f}]$)"


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[tuple[str, str]],
    caption: str,
    label: str,
    group_column: str | None = None,
    col_spec: str | None = None,
) -> None:
    """Write a booktabs tabular from selected (source_column, header) pairs.

    If ``group_column`` is set, that factor is printed once per consecutive
    block (via ``\\multirow``), with ``\\midrule`` separators between groups.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(columns)
    if col_spec is None:
        col_spec = "l" * n_cols

    records: list[list[str]] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for src, _ in columns:
            value = row[src]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.3f}")
            else:
                cells.append(str(value))
        records.append(cells)

    group_spans: list[tuple[int, int]] = []
    if group_column is not None:
        group_idx = next(i for i, (src, _) in enumerate(columns) if src == group_column)
        i = 0
        while i < len(records):
            j = i + 1
            while j < len(records) and records[j][group_idx] == records[i][group_idx]:
                j += 1
            n = j - i
            value = records[i][group_idx]
            records[i][group_idx] = rf"\multirow{{{n}}}{{*}}{{{value}}}"
            for k in range(i + 1, j):
                records[k][group_idx] = ""
            group_spans.append((i, j))
            i = j

    lines = [
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.2}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
    ]
    if group_spans:
        for g, (start, end) in enumerate(group_spans):
            if g > 0:
                lines.append(r"\midrule")
            for cells in records[start:end]:
                lines.append(" & ".join(cells) + r" \\")
    else:
        for cells in records:
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
