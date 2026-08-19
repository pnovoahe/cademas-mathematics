# Non-monotonic $V(\delta)$ for $A_L$ — diagnostic analysis

Internal report for Experiment 6.2. Not manuscript text. All numbers are
Monte Carlo means over 1000 replications with the same seeds,
population generator, $\lambda=0.75$, $q_{\mathrm{weak}}=0.25$,
and $\delta$ grid as the formal experiment. 95% percentile CIs are in the CSVs.

## 1. Question

For linear aggregation $A_L(R',Q)=\lambda R'+(1-\lambda)Q$ at $\lambda=0.75$,
the policy violation rate $V$ (share of Top-$K$ with $Q=0$) is **non-monotonic**
in the formal experiment:

| $\delta$ | $V$ (formal Exp. 02) |
|---|---|
| 0.00 | 0.077 |
| 0.05 | 0.209 |
| 0.10 | 0.251 (maximum) |
| 0.15 | 0.196 |
| 0.20 | 0.131 |
| 0.25 | 0.068 |
| 0.30 | 0.022 |

Why does $V$ rise and then fall? Candidate mechanisms to test, not assume:

1. Overconfidence is applied only to $Q\le 0.25$, so those cases become
   more competitive.
2. Clipping $R'=\mathrm{clip}(R+\delta,0,1)$ saturates high-$R$ cases at 1.
3. Vetoes ($Q=0$) are replaced in Top-$K$ by weak non-vetoes ($0<Q\le 0.25$).
4. Something else in the Top-$K$ threshold / score distributions.

$V$ counts **only** $Q=0$. A fall in $V$ need not mean that weak-context cases
leave Top-$K$.

## 2. Experimental setup (this diagnostic)

- Same `generate_population` as Experiment 02: $N=1000$, veto fraction
  $0.20$, standard $R,Q\sim\mathrm{Beta}(2,2)$, veto
  $R\sim\mathrm{Beta}(8,2)$ and $Q=0$.
- Same seeds $42$–$1041$.
- Operator: **linear only**, $\lambda=0.75$.
- $R'$ on $Q\le 0.25$ only.
- Two modes on the **same** populations:
  - **clipped** (formal): $R'=\mathrm{clip}(R+\delta,0,1)$;
  - **unclipped** (diagnostic): $R'=R+\delta$ (may exceed 1).
- Top-$K$ ties: $(-P,-R',\mathrm{id})$, $K=100$.
- Groups:
  - A: vetoes $Q=0$;
  - B: weak non-vetoes $0<Q\le 0.25$;
  - C: normal $Q>0.25$ (never receive $\delta$).

Algebra of the linear score at $\lambda=0.75$:

- Veto, $R'=1$: $P= 0.75$. This is the **clipped ceiling** for every
  $Q=0$ case.
- Weak non-veto, $R'=1$: $P=0.75+0.25Q>0.75$ whenever
  $Q>0$. Saturated weak non-vetoes **strictly outrank** saturated vetoes.
- Unclipped, both groups receive the same additive $+\lambda\delta$ in $P$,
  so **relative** order among already-affected cases is invariant to $\delta$.

These identities follow from the operator. Whether they drive the observed $V$
curve is an empirical question below.

## 3. Clipping analysis (affected population $Q\le 0.25$)

Clipped mode, means:

| $\delta$ | frac $R+\delta\geq 1$ | frac $R'=1$ | veto $R'=1$ | weak-NV $R'=1$ | mean $R'$ | mean $R'-R$ |
|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.685 | 0.000 |
| 0.05 | 0.047 | 0.047 | 0.072 | 0.007 | 0.734 | 0.049 |
| 0.10 | 0.150 | 0.150 | 0.227 | 0.027 | 0.779 | 0.094 |
| 0.15 | 0.271 | 0.271 | 0.402 | 0.060 | 0.819 | 0.134 |
| 0.20 | 0.389 | 0.389 | 0.566 | 0.104 | 0.852 | 0.167 |
| 0.25 | 0.492 | 0.492 | 0.701 | 0.156 | 0.880 | 0.195 |
| 0.30 | 0.578 | 0.578 | 0.805 | 0.215 | 0.903 | 0.218 |

Veto vs weak non-veto original $R$ (clipped mode; $R$ does not depend on $\delta$
except through sampling; reported at $\delta=0$):

- Veto mean original $R$: 0.800 (95% CI [0.782, 0.817])
- Weak non-veto mean original $R$: 0.500 (95% CI [0.461, 0.540])

Vetoes start with substantially higher $R$ ($\mathrm{Beta}(8,2)$ vs
$\mathrm{Beta}(2,2)$), so they hit the clip $R+\delta\ge 1$ at **smaller**
$\delta$.

Affected cases **in** Top-$K$ vs **out** (clipped):

| $\delta$ | in Top-$K$, frac $R'=1$ | out of Top-$K$, frac $R'=1$ | in, mean $R'$ | out, mean $R'$ |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.965 | 0.675 |
| 0.05 | 0.597 | 0.001 | 0.992 | 0.712 |
| 0.10 | 0.853 | 0.069 | 0.995 | 0.755 |
| 0.15 | 0.810 | 0.208 | 0.994 | 0.799 |
| 0.20 | 0.776 | 0.343 | 0.993 | 0.836 |
| 0.25 | 0.766 | 0.459 | 0.993 | 0.867 |
| 0.30 | 0.782 | 0.552 | 0.994 | 0.892 |

## 4. Top-$K$ composition ($A_L$, clipped)

Counts in Top-$K$ ($K=100$), Monte Carlo means:

| $\delta$ | $Q=0$ | $0<Q\leq 0.25$ | $Q>0.25$ | $V$ | weak share | veto / weak in Top-$K$ |
|---|---|---|---|---|---|---|
| 0.00 | 7.748 | 2.851 | 89.401 | 0.077 | 0.106 | 0.715 |
| 0.05 | 20.864 | 4.724 | 74.412 | 0.209 | 0.256 | 0.814 |
| 0.10 | 25.055 | 8.002 | 66.943 | 0.251 | 0.331 | 0.743 |
| 0.15 | 19.569 | 13.518 | 66.913 | 0.196 | 0.331 | 0.569 |
| 0.20 | 13.133 | 20.153 | 66.714 | 0.131 | 0.333 | 0.370 |
| 0.25 | 6.827 | 27.257 | 65.916 | 0.068 | 0.341 | 0.183 |
| 0.30 | 2.185 | 34.427 | 63.388 | 0.022 | 0.366 | 0.055 |

Mean scores of the selected set:

| $\delta$ | mean $Q$ | mean original $R$ | mean $R'$ | frac $R'=1$ |
|---|---|---|---|---|
| 0.00 | 0.587 | 0.850 | 0.850 | 0.000 |
| 0.05 | 0.500 | 0.875 | 0.886 | 0.151 |
| 0.10 | 0.460 | 0.880 | 0.902 | 0.285 |
| 0.15 | 0.469 | 0.868 | 0.902 | 0.271 |
| 0.20 | 0.479 | 0.855 | 0.902 | 0.261 |
| 0.25 | 0.485 | 0.843 | 0.903 | 0.263 |
| 0.30 | 0.482 | 0.830 | 0.908 | 0.287 |

At $\delta=0.00$: $V=0.077 (95% CI [0.020, 0.150])$; weak share
0.106 (95% CI [0.040, 0.190]); vetoes in Top-$K$
7.75; weak non-vetoes
2.85; normal
89.40.

At $\delta=0.10$ (clipped $V$ peak): $V=0.251 (95% CI [0.080, 0.390])$;
vetoes 25.05; weak non-vetoes
8.00; normal
66.94.

At $\delta=0.30$: $V=0.022 (95% CI [0.000, 0.130])$;
vetoes 2.19; weak non-vetoes
34.43; normal
63.39; weak share
0.366 (95% CI [0.270, 0.470]).

**Compositional reading (clipped, from the counts above):** after the $V$ peak,
the number of $Q=0$ cases in Top-$K$ falls while the number of
$0<Q\le 0.25$ cases continues to rise. Normal ($Q>0.25$) cases
are displaced throughout. The decline in $V$ is therefore a **replacement of
vetoes by weak non-vetoes**, not a return of normal high-$Q$ cases.

## 5. Veto vs weak-context decomposition

- $V$ = (vetoes in Top-$K$)/$K$.
- Weak-context exposure = (vetoes + weak non-vetoes in Top-$K$)/$K$.
- Veto share among weak Top-$K$ cases = vetoes / (vetoes + weak non-vetoes).

Clipped:

| $\delta$ | $V$ (veto rate) | weak-context rate | veto / weak | veto inclusion | weak-NV inclusion |
|---|---|---|---|---|---|
| 0.00 | 0.077 | 0.106 | 0.715 | 0.039 | 0.023 |
| 0.05 | 0.209 | 0.256 | 0.814 | 0.104 | 0.038 |
| 0.10 | 0.251 | 0.331 | 0.743 | 0.125 | 0.064 |
| 0.15 | 0.196 | 0.331 | 0.569 | 0.098 | 0.108 |
| 0.20 | 0.131 | 0.333 | 0.370 | 0.066 | 0.161 |
| 0.25 | 0.068 | 0.341 | 0.183 | 0.034 | 0.218 |
| 0.30 | 0.022 | 0.366 | 0.055 | 0.011 | 0.276 |

Unclipped (same quantities):

| $\delta$ | $V$ | weak-context rate | veto / weak | veto inclusion | weak-NV inclusion |
|---|---|---|---|---|---|
| 0.00 | 0.077 | 0.106 | 0.715 | 0.039 | 0.023 |
| 0.05 | 0.210 | 0.257 | 0.815 | 0.105 | 0.038 |
| 0.10 | 0.351 | 0.417 | 0.842 | 0.176 | 0.053 |
| 0.15 | 0.488 | 0.573 | 0.851 | 0.244 | 0.068 |
| 0.20 | 0.611 | 0.714 | 0.855 | 0.305 | 0.083 |
| 0.25 | 0.712 | 0.830 | 0.857 | 0.356 | 0.095 |
| 0.30 | 0.784 | 0.916 | 0.856 | 0.392 | 0.106 |

If weak-context rate stays high (or rises) while $V$ falls, the policy-violation
metric is **not** showing a disappearance of overconfident weak-context cases;
it is showing a shift from $Q=0$ to $0<Q\le 0.25$.

## 6. Score-distribution analysis (clipped)

Mean $P$ by group vs Top-$K$ threshold:

| $\delta$ | mean $P$ veto | mean $P$ weak-NV | mean $P$ normal | Top-$K$ $P$ threshold | veto $P$ q90 | weak-NV $P$ q90 | normal $P$ q90 |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.600 | 0.415 | 0.516 | 0.724 | 0.704 | 0.640 | 0.749 |
| 0.05 | 0.637 | 0.453 | 0.516 | 0.741 | 0.741 | 0.678 | 0.749 |
| 0.10 | 0.669 | 0.490 | 0.516 | 0.750 | 0.750 | 0.715 | 0.749 |
| 0.15 | 0.695 | 0.525 | 0.516 | 0.750 | 0.750 | 0.751 | 0.749 |
| 0.20 | 0.714 | 0.560 | 0.516 | 0.750 | 0.750 | 0.775 | 0.749 |
| 0.25 | 0.728 | 0.593 | 0.516 | 0.751 | 0.750 | 0.788 | 0.749 |
| 0.30 | 0.737 | 0.623 | 0.516 | 0.755 | 0.750 | 0.795 | 0.749 |

Quantiles of $P$ at $\delta=0.10$ (clipped $V$ peak) and $\delta=0.30$:

At $\delta=0.10$: veto q50=0.691,
q90=0.750; weak-NV q50=
0.490, q90=
0.715; threshold=
0.750.

At $\delta=0.30$: veto q50=0.750,
q90=0.750; weak-NV q50=
0.640, q90=
0.795; threshold=
0.755.

The clipped ceiling $P=0.75$ for $Q=0$ is marked on
`figures/scores_vs_threshold.pdf`. Once many vetoes sit at that ceiling,
further $\delta$ cannot raise their $P$, while weak non-vetoes can still
increase $P$ (until they also saturate, at which point $P>0.75$).

## 7. Clipped vs unclipped counterfactual

$V(\delta)$:

| $\delta$ | clipped $V$ |
|---|---|
| 0.00 | 0.077 |
| 0.05 | 0.209 |
| 0.10 | 0.251 |
| 0.15 | 0.196 |
| 0.20 | 0.131 |
| 0.25 | 0.068 |
| 0.30 | 0.022 |

| $\delta$ | unclipped $V$ |
|---|---|
| 0.00 | 0.077 |
| 0.05 | 0.210 |
| 0.10 | 0.351 |
| 0.15 | 0.488 |
| 0.20 | 0.611 |
| 0.25 | 0.712 |
| 0.30 | 0.784 |

- Clipped peak $\delta=0.10$, $V=0.251 (95% CI [0.080, 0.390])$;
  at $\delta=0.30$, $V=0.022 (95% CI [0.000, 0.130])$.
- Unclipped peak $\delta=0.30$, $V=0.784 (95% CI [0.710, 0.860])$;
  at $\delta=0.30$, $V=0.784 (95% CI [0.710, 0.860])$.

**Does clipping explain the non-monotonicity?** YES. Clipped $V(\delta)$ is non-monotonic; the unclipped diagnostic is not (it keeps rising or plateaus). Saturation at $R'=1$ is necessary for the decline after the peak.

Unclipped, $\delta$ adds $\lambda\delta$ to $P$ for every affected case and
$0$ to normal cases. Relative ranking **within** the affected set is invariant
to $\delta$. Vetoes keep their original $R$ advantage ($\mathrm{Beta}(8,2)$)
plus no $Q$-bonus, and that comparison does not flip with $\delta$ unless
clipping freezes the high-$R$ group.

## 8. Main findings

1. **Confirmed from code.** $\delta$ is applied only if $Q\le 0.25$;
   $R'$ is clipped to $[0,1]$ in the formal experiment; $V$ counts only $Q=0$
   in Top-$K$; linear $P=\lambda R'+(1-\lambda)Q$.
2. **Vetoes saturate first.** Mean original $R$ is much higher for vetoes than
   for weak non-vetoes, so $\mathrm{frac}(R+\delta\ge 1)$ rises earlier
   for $Q=0$ (Section 3 table).
3. **The $V$ peak is compositional.** Up to $\delta=0.10$, more
   vetoes enter Top-$K$ (and weak non-vetoes also begin to enter). After the
   peak, veto counts in Top-$K$ fall while weak non-veto counts keep rising
   (Section 4).
4. **$V$ falling ≠ weak-context exposure falling.** Weak-context share of
   Top-$K$ remains high at large $\delta$ in the clipped run (Section 5).
   The metric $V$ specifically loses $Q=0$ cases.
5. **Clipping is required for the downturn** under the comparison in Section 7
   (see the verdict there). Without clipping there is no $R'=1$ ceiling on
   veto scores, so veto $P$ keeps rising with $\delta$ and vetoes are not
   overtaken via the $Q$-bonus of saturated weak non-vetoes.

## 9. Recommended scientifically defensible interpretation for the manuscript

Do **not** write that “predictive overconfidence increases policy violations”
as a blanket statement. At $\lambda=0.75$ the formal $V(\delta)$ curve
for $A_L$ is non-monotonic.

A defensible account, restricted to this design:

- Overconfidence is applied to **all** $Q\le 0.25$, not only to vetoes.
- For $A_L$, raising $R'$ initially makes both vetoes and other weak-context
  cases more competitive against $Q>0.25$ cases, so $V$ rises from
  0.077 to 0.251.
- Because vetoes already have high $R$, they hit $R'=1$ first. Their linear
  score then cannot exceed $P=0.75$. Weak non-vetoes with $Q>0$
  can exceed that ceiling once their $R'$ is large, and they replace vetoes
  in Top-$K$. $V$ therefore falls even though weak-context occupancy of
  Top-$K$ stays large.
- $A_G$ and $A_M$ keep $V=0$ for all $\delta$ in the formal experiment because
  $Q=0$ is absorbing; this diagnostic did not re-estimate those operators.

What **not** to claim without extra evidence:

- that overconfidence “improves policy compliance” at large $\delta$ (it
  substitutes one weak-context class for another);
- that the downturn would occur without clipping, or at other $\lambda$, or
  for a different $q_{\mathrm{weak}}$;
- that any operator is globally better.

## Artifacts

- `results/trials_diagnostics.csv` — trial-level records
- `results/aggregated_diagnostics.csv` — mean / std / CI95%
- `figures/v_clipped_vs_unclipped.{pdf,png}`
- `figures/topk_composition.{pdf,png}`
- `figures/saturation_by_group.{pdf,png}`
- `figures/scores_vs_threshold.{pdf,png}`
- `figures/inclusion_probability.{pdf,png}`
