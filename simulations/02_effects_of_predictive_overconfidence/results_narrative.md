# Narrative summary — Experiment 6.2 Effects of predictive overconfidence

This file is a **text-only briefing** for humans and LLM agents working on
`first_round_v2/manuscriptR1V2.tex` Section 6.2. Do not invent numerical claims.
Do not describe any operator as globally superior. Do not assume the reader
can see the figures.

- Code: `simulations/02_effects_of_predictive_overconfidence/run.py`
- Config: `simulations/common/config.py`
- Seeds: 42 through 1041

## Scientific message

Section 6.2 shows the **population-level consequences** of a systematic
upward bias in predictive scores on contextually weak alternatives. The
question is how increasing $\delta$ changes Top-$K$ composition under each
aggregation semantics, not which operator is better.

The mechanism is $R_i'=\mathrm{clip}(R_i+\delta,0,1)$ applied only when
$Q_i\le q_{\mathrm{weak}}=0.25$. Predictive
utility uses the **original** $R$ of the selected set, so an increase in
$\bar{R}$ is not automatic from inflating $R'$.

## Design (this run)

| Parameter | Value |
|---|---|
| $N$ | 1000 |
| `veto_fraction` | 0.20 |
| Veto cases | 200 |
| Standard cases | 800 |
| $K$ | 100 |
| $q_{\mathrm{weak}}$ | 0.25 |
| $\delta$ | (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3) |
| $\lambda$ stored | (0.5, 0.75, 0.9) |
| $\lambda$ primary | 0.75 |
| Monte Carlo | 1000 |

## Headline results ($\lambda=0.75$)

At $\delta=0.00$ (no overconfidence):
- $A_L$: $V=0.077 (95% CI [0.020, 0.150])$; $\bar{R}=0.850 (95% CI [0.835, 0.866])$; $\tau=1.000 (95% CI [1.000, 1.000])$.
- $A_G$: $V=0.000 (95% CI [0.000, 0.000])$; $\bar{R}=0.818 (95% CI [0.798, 0.837])$; $\tau=1.000 (95% CI [1.000, 1.000])$.
- $A_M$: $V=0.000 (95% CI [0.000, 0.000])$; $\bar{R}=0.744 (95% CI [0.721, 0.766])$; $\tau=1.000 (95% CI [1.000, 1.000])$.

At $\delta=0.30$ (largest overconfidence on this grid):
- $A_L$: $V=0.022 (95% CI [0.000, 0.130])$; $\bar{R}=0.830 (95% CI [0.812, 0.847])$; $\tau=0.736 (95% CI [0.717, 0.757])$.
- $A_G$: $V=0.000 (95% CI [0.000, 0.000])$; $\bar{R}=0.819 (95% CI [0.802, 0.837])$; $\tau=0.914 (95% CI [0.900, 0.928])$.
- $A_M$: $V=0.000 (95% CI [0.000, 0.000])$; $\bar{R}=0.744 (95% CI [0.721, 0.766])$; $\tau=0.997 (95% CI [0.995, 0.999])$.

### Notable transitions in $V$

- $A_L$: 0.077 (95% CI [0.020, 0.150]) at $\delta=0.00$; 0.022 (95% CI [0.000, 0.130]) at $\delta=0.30$; interior maximum 0.251 (95% CI [0.080, 0.390]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
- $A_G$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.
- $A_M$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

## Figure descriptions

### Primary two-panel: `figures/overconfidence_v_utility.pdf`

Shared x-axis: overconfidence level $\delta\in[0.00,0.30]$
with ticks at 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30.
Three series in both panels, drawn Linear → Geometric → Min so overlaps remain
visible: $A_L$ blue downward triangles (largest markers),
$A_G$ orange squares (medium), $A_M$
green circles (smallest). White marker edges. Shaded $95\%$ percentile bands.
Legends in the upper-left corner of each square axes box. Panel letters (a)
and (b) above the boxes.

**(a) Policy violation rate $V$.** Vertical axis from just below 0 to 1
(ticks $0.0,0.2,\ldots,1.0$) so series at $V=0$ sit slightly above the bottom
spine. $A_L$: 0.077 (95% CI [0.020, 0.150]) at $\delta=0.00$; 0.022 (95% CI [0.000, 0.130]) at $\delta=0.30$; interior maximum 0.251 (95% CI [0.080, 0.390]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
$A_G$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.
$A_M$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.
If two series lie on the bottom axis they overlap: the green circle sits on
the orange square, which sits on the blue triangle.

**(b) Predictive utility $\bar{R}$.** Vertical axis is the mean original
$R$ of the Top-$K$ set (not $R'$). $A_L$: 0.850 (95% CI [0.835, 0.866]) at $\delta=0.00$; 0.830 (95% CI [0.812, 0.847]) at $\delta=0.30$; interior maximum 0.880 (95% CI [0.862, 0.896]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
$A_G$: 0.818 (95% CI [0.798, 0.837]) at $\delta=0.00$; 0.819 (95% CI [0.802, 0.837]) at $\delta=0.30$.
$A_M$: 0.744 (95% CI [0.721, 0.766]) at $\delta=0.00$; 0.744 (95% CI [0.721, 0.766]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

### Kendall $\tau$: `figures/overconfidence_kendall_tau.pdf`

Single square panel, same x-axis, markers, colours, and upper-left legend as
panel (a). Y-axis is Kendall $\tau$ between $P_{\mathrm{ref}}$ ($\delta=0$)
and $P(\delta)$. At $\delta=0$ every operator is $\tau=1$ by construction.
$A_L$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.736 (95% CI [0.717, 0.757]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
$A_G$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.914 (95% CI [0.900, 0.928]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
$A_M$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.997 (95% CI [0.995, 0.999]) at $\delta=0.30$; mean decreases over the $\delta$ grid.



## Table walkthrough

`tables/table_predictive_overconfidence.tex` groups rows by $\delta$
(0.00, 0.10, 0.20, 0.30); $\delta$ is printed
once per operator triplet. Columns: $V$ mean/std/CI and $\bar{R}$ mean/CI
at $\lambda=0.75$.

### Full $V$ grid (primary $\lambda$)

| $\delta$ | $A_L$ | $A_G$ | $A_M$ |
|---|---:|---:|---:|
| 0.00 | 0.077 (95% CI [0.020, 0.150]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.05 | 0.209 (95% CI [0.120, 0.300]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.10 | 0.251 (95% CI [0.080, 0.390]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.15 | 0.196 (95% CI [0.030, 0.350]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.20 | 0.131 (95% CI [0.000, 0.280]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.25 | 0.068 (95% CI [0.000, 0.220]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |
| 0.30 | 0.022 (95% CI [0.000, 0.130]) | 0.000 (95% CI [0.000, 0.000]) | 0.000 (95% CI [0.000, 0.000]) |

- $A_L$: 0.077 (95% CI [0.020, 0.150]) at $\delta=0.00$; 0.022 (95% CI [0.000, 0.130]) at $\delta=0.30$; interior maximum 0.251 (95% CI [0.080, 0.390]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
- $A_G$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.
- $A_M$: 0.000 (95% CI [0.000, 0.000]) at $\delta=0.00$; 0.000 (95% CI [0.000, 0.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

### Full $\bar{R}$ grid (primary $\lambda$)

| $\delta$ | $A_L$ | $A_G$ | $A_M$ |
|---|---:|---:|---:|
| 0.00 | 0.850 (95% CI [0.835, 0.866]) | 0.818 (95% CI [0.798, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.05 | 0.875 (95% CI [0.862, 0.889]) | 0.818 (95% CI [0.799, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.10 | 0.880 (95% CI [0.862, 0.896]) | 0.818 (95% CI [0.800, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.15 | 0.868 (95% CI [0.853, 0.881]) | 0.819 (95% CI [0.801, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.20 | 0.855 (95% CI [0.842, 0.868]) | 0.819 (95% CI [0.802, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.25 | 0.843 (95% CI [0.828, 0.858]) | 0.819 (95% CI [0.802, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |
| 0.30 | 0.830 (95% CI [0.812, 0.847]) | 0.819 (95% CI [0.802, 0.837]) | 0.744 (95% CI [0.721, 0.766]) |

- $A_L$: 0.850 (95% CI [0.835, 0.866]) at $\delta=0.00$; 0.830 (95% CI [0.812, 0.847]) at $\delta=0.30$; interior maximum 0.880 (95% CI [0.862, 0.896]) at $\delta=0.10$; first mean increase after $\delta=0$ at $\delta=0.05$.
- $A_G$: 0.818 (95% CI [0.798, 0.837]) at $\delta=0.00$; 0.819 (95% CI [0.802, 0.837]) at $\delta=0.30$.
- $A_M$: 0.744 (95% CI [0.721, 0.766]) at $\delta=0.00$; 0.744 (95% CI [0.721, 0.766]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

### Full Kendall $\tau$ grid (primary $\lambda$)

| $\delta$ | $A_L$ | $A_G$ | $A_M$ |
|---|---:|---:|---:|
| 0.00 | 1.000 (95% CI [1.000, 1.000]) | 1.000 (95% CI [1.000, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.05 | 0.947 (95% CI [0.944, 0.950]) | 0.986 (95% CI [0.983, 0.988]) | 0.998 (95% CI [0.997, 0.999]) |
| 0.10 | 0.897 (95% CI [0.891, 0.904]) | 0.971 (95% CI [0.966, 0.975]) | 0.998 (95% CI [0.996, 0.999]) |
| 0.15 | 0.851 (95% CI [0.842, 0.861]) | 0.956 (95% CI [0.949, 0.963]) | 0.997 (95% CI [0.995, 0.999]) |
| 0.20 | 0.809 (95% CI [0.796, 0.823]) | 0.941 (95% CI [0.932, 0.951]) | 0.997 (95% CI [0.995, 0.999]) |
| 0.25 | 0.771 (95% CI [0.754, 0.788]) | 0.927 (95% CI [0.916, 0.939]) | 0.997 (95% CI [0.995, 0.999]) |
| 0.30 | 0.736 (95% CI [0.717, 0.757]) | 0.914 (95% CI [0.900, 0.928]) | 0.997 (95% CI [0.995, 0.999]) |

- $A_L$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.736 (95% CI [0.717, 0.757]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_G$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.914 (95% CI [0.900, 0.928]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_M$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.997 (95% CI [0.995, 0.999]) at $\delta=0.30$; mean decreases over the $\delta$ grid.

### Full Jaccard Top-$K$ grid (primary $\lambda$)

| $\delta$ | $A_L$ | $A_G$ | $A_M$ |
|---|---:|---:|---:|
| 0.00 | 1.000 (95% CI [1.000, 1.000]) | 1.000 (95% CI [1.000, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.05 | 0.740 (95% CI [0.653, 0.818]) | 0.997 (95% CI [0.980, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.10 | 0.596 (95% CI [0.493, 0.739]) | 0.993 (95% CI [0.961, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.15 | 0.565 (95% CI [0.439, 0.724]) | 0.986 (95% CI [0.942, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.20 | 0.549 (95% CI [0.418, 0.695]) | 0.978 (95% CI [0.923, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.25 | 0.532 (95% CI [0.408, 0.653]) | 0.969 (95% CI [0.887, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |
| 0.30 | 0.499 (95% CI [0.399, 0.600]) | 0.962 (95% CI [0.869, 1.000]) | 1.000 (95% CI [1.000, 1.000]) |

- $A_L$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.499 (95% CI [0.399, 0.600]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_G$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 0.962 (95% CI [0.869, 1.000]) at $\delta=0.30$; mean decreases over the $\delta$ grid.
- $A_M$: 1.000 (95% CI [1.000, 1.000]) at $\delta=0.00$; 1.000 (95% CI [1.000, 1.000]) at $\delta=0.30$; no change relative to $\delta=0$ on this grid.

## Robustness: other $\lambda$ ($V$ from $\delta=0$ to $\delta=0.30$)

- $\lambda=0.50$: $A_L$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]); $A_G$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]); $A_M$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]).
- $\lambda=0.75$: $A_L$ 0.077 (95% CI [0.020, 0.150]) → 0.022 (95% CI [0.000, 0.130]); $A_G$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]); $A_M$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]).
- $\lambda=0.90$: $A_L$ 0.462 (95% CI [0.370, 0.550]) → 0.619 (95% CI [0.510, 0.730]); $A_G$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]); $A_M$ 0.000 (95% CI [0.000, 0.000]) → 0.000 (95% CI [0.000, 0.000]).

Predictive utility:

- $\lambda=0.50$: $A_L$ 0.758 (95% CI [0.734, 0.785]) → 0.758 (95% CI [0.734, 0.785]); $A_G$ 0.758 (95% CI [0.734, 0.781]) → 0.758 (95% CI [0.734, 0.781]); $A_M$ 0.744 (95% CI [0.721, 0.766]) → 0.744 (95% CI [0.721, 0.766]).
- $\lambda=0.75$: $A_L$ 0.850 (95% CI [0.835, 0.866]) → 0.830 (95% CI [0.812, 0.847]); $A_G$ 0.818 (95% CI [0.798, 0.837]) → 0.819 (95% CI [0.802, 0.837]); $A_M$ 0.744 (95% CI [0.721, 0.766]) → 0.744 (95% CI [0.721, 0.766]).
- $\lambda=0.90$: $A_L$ 0.913 (95% CI [0.902, 0.923]) → 0.840 (95% CI [0.825, 0.857]); $A_G$ 0.847 (95% CI [0.830, 0.863]) → 0.829 (95% CI [0.810, 0.849]); $A_M$ 0.744 (95% CI [0.721, 0.766]) → 0.744 (95% CI [0.721, 0.766]).

Kendall $\tau$:

- $\lambda=0.50$: $A_L$ 1.000 (95% CI [1.000, 1.000]) → 0.788 (95% CI [0.767, 0.808]); $A_G$ 1.000 (95% CI [1.000, 1.000]) → 0.965 (95% CI [0.958, 0.972]); $A_M$ 1.000 (95% CI [1.000, 1.000]) → 0.997 (95% CI [0.995, 0.999]).
- $\lambda=0.75$: $A_L$ 1.000 (95% CI [1.000, 1.000]) → 0.736 (95% CI [0.717, 0.757]); $A_G$ 1.000 (95% CI [1.000, 1.000]) → 0.914 (95% CI [0.900, 0.928]); $A_M$ 1.000 (95% CI [1.000, 1.000]) → 0.997 (95% CI [0.995, 0.999]).
- $\lambda=0.90$: $A_L$ 1.000 (95% CI [1.000, 1.000]) → 0.771 (95% CI [0.752, 0.791]); $A_G$ 1.000 (95% CI [1.000, 1.000]) → 0.891 (95% CI [0.874, 0.907]); $A_M$ 1.000 (95% CI [1.000, 1.000]) → 0.997 (95% CI [0.995, 0.999]).

Linear $V$ shape by $\lambda$ (observed, not extrapolated):

- $\lambda=0.50$: linear mean $V$ remains $0$ over the whole $\delta$ grid.
- $\lambda=0.75$: linear mean $V$ is non-monotonic, with an interior maximum 0.251 (95% CI [0.080, 0.390]) at $\delta=0.10$ (ends at 0.022 (95% CI [0.000, 0.130])).
- $\lambda=0.90$: linear mean $V$ is non-monotonic, with an interior maximum 0.809 (95% CI [0.730, 0.880]) at $\delta=0.15$ (ends at 0.619 (95% CI [0.510, 0.730])).

## Paper recommendations

Primary paper figure: `figures/overconfidence_v_utility.pdf` (two square
panels: (a) $V$ vs $\delta$, (b) $\bar{R}$ vs $\delta$) at
$\lambda=0.75$.
Optional paper figure: `figures/overconfidence_kendall_tau.pdf` (operators separate on Kendall $\tau$).
Do **not** include a Jaccard figure: it is redundant with Kendall $\tau$, or operators do not separate.
Do **not** plot $\lambda=0.50$ or $\lambda=0.90$ by default; they are robustness checks in `results/aggregated.csv`.

Table for the paper: `tables/table_predictive_overconfidence.tex`. Include the
Kendall/Jaccard table only if the Kendall figure is used.

## Interpretation limits

All statements above are Monte Carlo summaries for this synthetic design
($N=1000$, $K=100$, veto fraction 0.20,
$q_{\mathrm{weak}}=0.25$, adversarial veto
$R\sim\mathrm{Beta}(8.0, 2.0)$). They do not establish that any
operator is globally preferable, and they do not transfer automatically to
other populations or other $q_{\mathrm{weak}}$ values.
