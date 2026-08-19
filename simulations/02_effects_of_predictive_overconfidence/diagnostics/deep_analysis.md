# Deep validation — non-monotonic $V(\delta)$ for $A_L$ (Experiment 6.2)

Second-pass diagnostic. Formal Experiment 6.2 is **unchanged**. All numbers below
use the same seeds ($42$–$1041$), population generator, $\delta$ grid, and
$q_{\mathrm{weak}}=0.25$ unless labelled as sensitivity.

Companion: `non_monotonicity_analysis.md` (first diagnostic pass).

## 1. Question and hypotheses under test

Formal setup: $R_i'=\mathrm{clip}(R_i+\delta,0,1)$ iff $Q_i\le 0.25$;
$A_L(R',Q)=0.75R'+0.25Q$; $V=|\{Q=0\}\cap\mathrm{Top\text{-}K}|/K$.

Hypotheses H1–H8 from the prior diagnostic are tested below with distributional
and pairwise evidence, not only group means.

## 2. Mathematical ranking conditions ($\lambda=0.75$)

**Veto** ($Q=0$): $P_v=0.75R_v'$.

**Weak non-veto** ($0<Q\le q_{\mathrm{weak}}$): $P_w=0.75R_w'+0.25Q$.

If both are clipped to $R'=1$:

- $P_v=0.75$.
- $P_w=0.75+0.25Q > 0.75$ for every $Q>0$.

Therefore any saturated weak non-veto **strictly outranks** any saturated veto,
independent of tie-breaking on $R'$.

**Unclipped counterfactual:** for affected cases, $P(\delta)=0.75(R+\delta)+cQ$
with $c\in\{0,0.25\}$. Adding $\delta$ adds $0.75\delta$ to **both**
groups equally. Relative order among affected cases is **invariant** in $\delta$;
vetoes cannot be overtaken by weak non-vetoes through the $Q$-bonus alone.
Normal cases ($Q>0.25$) receive no shift; their relative position vs affected
cases can change only through threshold competition, not through pairwise
veto-vs-weak reversal.

## 3. Analysis 1 — Direct ranking verification (clipped, $\lambda=0.75$)

### Top-$K$ composition and inclusion

| $\delta$ | $Q=0$ count | $0<Q\leq 0.25$ count | $Q>0.25$ count | veto incl. | weak-NV incl. | normal incl. |
|---|---|---|---|---|---|---|
| 0.00 | 7.748 | 2.851 | 89.401 | 0.039 | 0.023 | 0.132 |
| 0.05 | 20.864 | 4.724 | 74.412 | 0.104 | 0.038 | 0.110 |
| 0.10 | 25.055 | 8.002 | 66.943 | 0.125 | 0.064 | 0.099 |
| 0.15 | 19.569 | 13.518 | 66.913 | 0.098 | 0.108 | 0.099 |
| 0.20 | 13.133 | 20.153 | 66.714 | 0.066 | 0.161 | 0.099 |
| 0.25 | 6.827 | 27.257 | 65.916 | 0.034 | 0.218 | 0.098 |
| 0.30 | 2.185 | 34.427 | 63.388 | 0.011 | 0.276 | 0.094 |

### Score distributions and saturation

| $\delta$ | Top-$K$ $P$ threshold | veto $R\'=1$ | weak-NV $R\'=1$ | Top-$K$ $R\'=1$ | veto $P$ q90 | weak-NV $P$ q90 |
|---|---|---|---|---|---|---|
| 0.00 | 0.724 | 0.000 | 0.000 | 0.000 | 0.704 | 0.640 |
| 0.05 | 0.741 | 0.072 | 0.007 | 0.151 | 0.741 | 0.678 |
| 0.10 | 0.750 | 0.227 | 0.027 | 0.285 | 0.750 | 0.715 |
| 0.15 | 0.750 | 0.402 | 0.060 | 0.271 | 0.750 | 0.751 |
| 0.20 | 0.750 | 0.566 | 0.104 | 0.261 | 0.750 | 0.775 |
| 0.25 | 0.751 | 0.701 | 0.156 | 0.263 | 0.750 | 0.788 |
| 0.30 | 0.755 | 0.805 | 0.215 | 0.287 | 0.750 | 0.795 |

### Pairwise veto vs weak non-veto ranking

Fraction of all veto–weak non-veto pairs with $P_w > P_v$:

| $\delta$ | all pairs | both $R\'=1$ |
|---|---|---|
| 0.00 | 0.179 | nan |
| 0.05 | 0.180 | 1.000 |
| 0.10 | 0.185 | 1.000 |
| 0.15 | 0.199 | 1.000 |
| 0.20 | 0.225 | 1.000 |
| 0.25 | 0.263 | 1.000 |
| 0.30 | 0.313 | 1.000 |

At $\delta=0.10$ (formal $V$ peak): all-pair fraction =
0.185; among saturated pairs =
1.000.

At $\delta=0.30$: all-pair fraction =
0.313; among saturated pairs =
1.000.

**Interpretation.** The saturated-pair fraction approaches 1 once both groups
have substantial $R'=1$ mass, confirming the exact ranking inequality. The
all-pair fraction rises with $\delta$ as more weak non-vetoes gain enough
$R'$ to beat unsaturated vetoes.

## 4. Analysis 2 — Clipping mechanism (same populations)

| $\delta$ | clipped $V$ | unclipped $V$ | clipped $E_{\mathrm{weak}}$ | unclipped $E_{\mathrm{weak}}$ |
|---|---|---|---|---|
| 0.00 | 0.077 | 0.077 | 0.106 | 0.106 |
| 0.05 | 0.209 | 0.210 | 0.256 | 0.257 |
| 0.10 | 0.251 | 0.351 | 0.331 | 0.417 |
| 0.15 | 0.196 | 0.488 | 0.331 | 0.573 |
| 0.20 | 0.131 | 0.611 | 0.333 | 0.714 |
| 0.25 | 0.068 | 0.712 | 0.341 | 0.830 |
| 0.30 | 0.022 | 0.784 | 0.366 | 0.916 |

Clipped: peak $\delta=0.10$, $V=0.251$; non-monotonic=True.
Unclipped: peak $\delta=0.30$, $V=0.784$; monotonic increase=True.

**Verdict.** Clipping is **necessary** for the observed downturn: unclipped $V$
 rises to 0.784 at $\delta=0.30$ while clipped $V$ falls
 to 0.022.

## 5. Analysis 3 — $V$ falls but weak-context exposure does not

Definitions: $V=veto/K$, $W_{\mathrm{weak}}=weak\ non\ veto/K$,
$E_{\mathrm{weak}}=(veto+weak\ non\ veto)/K$,
$S_{\mathrm{veto|weak}}=veto/(veto+weak\ non\ veto)$.

| $\delta$ | $V$ | $W_{\mathrm{weak}}$ | $E_{\mathrm{weak}}$ | $S_{\mathrm{veto|weak}}$ |
|---|---|---|---|---|
| 0.00 | 0.077 | 0.029 | 0.106 | 0.715 |
| 0.05 | 0.209 | 0.047 | 0.256 | 0.814 |
| 0.10 | 0.251 | 0.080 | 0.331 | 0.743 |
| 0.15 | 0.196 | 0.135 | 0.331 | 0.569 |
| 0.20 | 0.131 | 0.202 | 0.333 | 0.370 |
| 0.25 | 0.068 | 0.273 | 0.341 | 0.183 |
| 0.30 | 0.022 | 0.344 | 0.366 | 0.055 |

From $\delta=0.10$ to $\delta=0.30$:

- $V$: 0.251 $\to$ 0.022 (**decreasing**).
- $E_{\mathrm{weak}}$: 0.331 $\to$
  0.366 (**increasing**).
- $W_{\mathrm{weak}}$: 0.080 $\to$
  0.344 (**increasing**).

**Verdict.** The decline in $V$ is accompanied by **increasing** weak-context
occupancy of Top-$K$. It is **not** contextual recovery.

## 6. Analysis 4 — Predictive utility decomposition (original $R$)

| $\delta$ | mean $R$ Top-$K$ | veto contrib. | weak-NV contrib. | normal contrib. |
|---|---|---|---|---|
| 0.00 | 0.850 | 0.076 | 0.027 | 0.748 |
| 0.05 | 0.875 | 0.200 | 0.043 | 0.632 |
| 0.10 | 0.880 | 0.235 | 0.072 | 0.573 |
| 0.15 | 0.868 | 0.178 | 0.117 | 0.573 |
| 0.20 | 0.855 | 0.116 | 0.168 | 0.571 |
| 0.25 | 0.843 | 0.059 | 0.219 | 0.565 |
| 0.30 | 0.830 | 0.018 | 0.267 | 0.544 |

$\bar{R}$ is non-monotonic in $\delta$ (peak near $\delta=0.10$), tracking
the $V$ peak phase when more high-$R$ vetoes enter Top-$K$. After $\delta=0.10$,
mean original $R$ of the selected set declines as vetoes are replaced by lower-$R$
weak non-vetoes and some normal cases.

Selected-group means (original $R$):

| $\delta$ | veto sel. $R$ | weak-NV sel. $R$ | normal sel. $R$ |
|---|---|---|---|
| 0.00 | 0.977 | 0.935 | 0.837 |
| 0.05 | 0.960 | 0.918 | 0.849 |
| 0.10 | 0.938 | 0.895 | 0.856 |
| 0.15 | 0.910 | 0.864 | 0.856 |
| 0.20 | 0.886 | 0.833 | 0.856 |
| 0.25 | 0.865 | 0.803 | 0.857 |
| 0.30 | 0.848 | 0.775 | 0.859 |

## 7. Analysis 5 — $\lambda$ robustness (formal $V$ + diagnostic composition)

Formal $V(\delta)$ from `../results/aggregated.csv` (linear operator):

- $\lambda=0.50$: peak $V$ at $\delta=0.00$ (0.000); $\delta=0.30$ gives 0.000; non-monotonic=False.
- $\lambda=0.75$: peak $V$ at $\delta=0.10$ (0.251); $\delta=0.30$ gives 0.022; non-monotonic=True.
- $\lambda=0.90$: peak $V$ at $\delta=0.15$ (0.809); $\delta=0.30$ gives 0.619; non-monotonic=True.

Composition metrics below use the same seeds and populations (clipped model,
diagnostic recomputation):

**$\lambda=0.50$** (clipped diagnostic; formal $V$ from stored results):

| $\delta$ | $V$ | $E_{\mathrm{weak}}$ | veto count | weak-NV count |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.15 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.20 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.25 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.30 | 0.000 | 0.000 | 0.000 | 0.000 |

At peak $\delta=0.00$: $E_{\mathrm{weak}}=0.000$, vetoes=0.0$, weak non-vetoes=0.0$. At $\delta=0.30$: $E_{\mathrm{weak}}=0.000$, vetoes=0.0$, weak non-vetoes=0.0$.
**$\lambda=0.75$** (clipped diagnostic; formal $V$ from stored results):

| $\delta$ | $V$ | $E_{\mathrm{weak}}$ | veto count | weak-NV count |
|---|---|---|---|---|
| 0.00 | 0.077 | 0.106 | 7.748 | 2.851 |
| 0.05 | 0.209 | 0.256 | 20.864 | 4.724 |
| 0.10 | 0.251 | 0.331 | 25.055 | 8.002 |
| 0.15 | 0.196 | 0.331 | 19.569 | 13.518 |
| 0.20 | 0.131 | 0.333 | 13.133 | 20.153 |
| 0.25 | 0.068 | 0.341 | 6.827 | 27.257 |
| 0.30 | 0.022 | 0.366 | 2.185 | 34.427 |

At peak $\delta=0.10$: $E_{\mathrm{weak}}=0.331$, vetoes=25.1$, weak non-vetoes=8.0$. At $\delta=0.30$: $E_{\mathrm{weak}}=0.366$, vetoes=2.2$, weak non-vetoes=34.4$.
**$\lambda=0.90$** (clipped diagnostic; formal $V$ from stored results):

| $\delta$ | $V$ | $E_{\mathrm{weak}}$ | veto count | weak-NV count |
|---|---|---|---|---|
| 0.00 | 0.462 | 0.510 | 46.188 | 4.841 |
| 0.05 | 0.599 | 0.664 | 59.907 | 6.541 |
| 0.10 | 0.720 | 0.801 | 71.950 | 8.129 |
| 0.15 | 0.809 | 0.906 | 80.865 | 9.752 |
| 0.20 | 0.765 | 0.916 | 76.458 | 15.175 |
| 0.25 | 0.695 | 0.916 | 69.487 | 22.146 |
| 0.30 | 0.619 | 0.916 | 61.881 | 29.752 |

At peak $\delta=0.15$: $E_{\mathrm{weak}}=0.906$, vetoes=80.9$, weak non-vetoes=9.8$. At $\delta=0.30$: $E_{\mathrm{weak}}=0.916$, vetoes=61.9$, weak non-vetoes=29.8$.

$\lambda=0.50$: $V=0$ on the grid (no non-monotonicity). $\lambda=0.75$ and
$0.90$ show non-monotonic clipped $V$ with replacement dynamics at different
magnitudes. This is robustness only; the paper primary remains $\lambda=0.75$.

## 8. Analysis 6 — $q_{\mathrm{weak}}$ sensitivity (diagnostic only)

- $q_{\mathrm{weak}}=0.10$: peak at $\delta=0.10$, $V=0.310$; non-monotonic=True.
- $q_{\mathrm{weak}}=0.25$: peak at $\delta=0.10$, $V=0.251$; non-monotonic=True.
- $q_{\mathrm{weak}}=0.40$: peak at $\delta=0.05$, $V=0.180$; non-monotonic=True.

Non-monotonicity persists for $q_{\mathrm{weak}}\in\{0.10,0.25,0.40\}$
 under clipping; peak location shifts modestly. Clipping remains necessary in
 each case (unclipped curves rise monotonically; not tabulated here).

## 9. Classification of findings

| Finding | Class |
|---|---|
| Non-monotonic $V(\delta)$ at $\lambda=0.75$ | **A** Essential |
| Weak-context exposure rises while $V$ falls | **A** Essential |
| Replacement of $Q=0$ by $0<Q\le q_{\mathrm{weak}}$ | **A** Essential |
| Clipping creates $P_v\le 0.75$ ceiling; $P_w>0.75$ when saturated | **A** Essential |
| Pairwise $P_w>P_v$ among saturated pairs | **B** Supporting |
| Unclipped counterfactual (monotonic $V$) | **B** Supporting (internal/supplement) |
| Utility decomposition / $\bar{R}$ peak | **B** Supporting |
| $\lambda=0.50/0.90$ robustness | **B** Brief robustness only |
| $q_{\mathrm{weak}}$ sensitivity | **C** Internal |
| Kendall $\tau$ / Jaccard figures | **B** if ranking stability discussed; else **C** |
| Full pairwise tables | **C** Internal |

**Do not claim** that large-$\delta$ overconfidence improves policy compliance.

## 10. Artifacts

- `results/deep_primary.csv`, `results/deep_primary_aggregated.csv`
- `results/deep_qweak_sensitivity.csv`, `results/deep_qweak_aggregated.csv`
- `figures/deep_v_and_weak_exposure.{pdf,png}`
- `figures/deep_exposure_decomposition.{pdf,png}`
- `figures/deep_pairwise_ranking.{pdf,png}`
- `figures/deep_utility_decomposition.{pdf,png}`
- `figures/deep_qweak_sensitivity.{pdf,png}`
- `tables/table_deep_exposure_decomposition.tex`
- `captions.md` (updated)

## Manuscript recommendation

### 1. Essential findings

- $V(\delta)$ for $A_L$ at $\lambda=0.75$ is **non-monotonic** (peak
  $\delta=0.10$, $V=0.251$; $\delta=0.30$, $V=0.022$).
- Overconfidence applies to **all** $Q\le 0.25$, not only vetoes.
- After the peak, $V$ falls because **vetoes leave Top-$K$** while **weak
  non-vetoes enter**, not because weak-context exposure falls ($E_{\mathrm{weak}}$
  rises from 0.331 to 0.366).
- Clipping is **necessary** for the downturn (unclipped $V$ rises to
  0.784 at $\delta=0.30$).

### 2. Recommended for main text

- State non-monotonicity; do **not** say overconfidence monotonically increases
  violations.
- Explain selective application to $Q\le q_{\mathrm{weak}}$ and the
  $P_v=0.75R'$, $P_w=0.75R'+0.25Q$ ranking logic under saturation.
- Note that lower $V$ at large $\delta$ reflects **metric composition**
  ($Q=0$ only), not restored compliance.
- Primary figure: two-panel $V$ and $\bar{R}$ at $\lambda=0.75$.

### 3. Better suited for appendix / supplement

- Unclipped counterfactual figure or one sentence.
- Table `tab:deep-exposure-decomposition` or abbreviated version with
  $V, W_{\mathrm{weak}}, E_{\mathrm{weak}}, S_{\mathrm{veto|weak}}$ at
  representative $\delta$.
- Brief $\lambda=0.90$ robustness sentence (non-monotonic, higher baseline $V$).

### 4. Remain internal

- Full pairwise tables, $q_{\mathrm{weak}}$ sensitivity plots, trial-level CSVs.
- Kendall $\tau$ unless the narrative emphasizes ranking stability explicitly.

### 5. Safe numerical quotes (clipped, $\lambda=0.75$, MC means)

| Quantity | $\delta=0.00$ | $\delta=0.10$ | $\delta=0.30$ |
|---|---|---|---|
| $V$ | 0.077 | 0.251 | 0.022 |
| $E_{\mathrm{weak}}$ | 0.106 | 0.331 | 0.366 |
| $W_{\mathrm{weak}}$ | 0.029 | 0.080 | 0.344 |
| Vetoes in Top-$K$ | 7.7 | 25.1 | 2.2 |
| Weak non-vetoes in Top-$K$ | 2.9 | 8.0 | 34.4 |
| Sat. pairs $P_w>P_v$ | — | 1.000 | 1.000 |
| $\bar{R}$ (original) | 0.850 | 0.880 | 0.830 |

Formal experiment values for $V$ and $\bar{R}$ match the first column/row of
the published `results_narrative.md` within Monte Carlo noise (this diagnostic
recomputes from the same seeds).

### Remaining ambiguities for author decision

1. Whether to name **weak non-vetoes** explicitly in the main text or use
   “contextually weak but non-veto cases”.
2. Whether the unclipped counterfactual deserves one sentence or a supplement
   figure.
3. Whether $\lambda=0.90$ robustness deserves one sentence given the higher
   baseline violation rate.
