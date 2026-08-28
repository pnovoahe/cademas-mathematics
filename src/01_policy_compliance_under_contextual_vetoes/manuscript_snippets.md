# Manuscript snippets (Section 6.1)

Generated automatically from Monte Carlo results. Do not invent additional claims.
Do not describe any operator as globally superior.

## Setup (for the paper)

Each Monte Carlo replication generates $N=1000$ decision cases, of which a fraction $0.20$ ($N_{\mathrm{veto}}=200$) have $Q_i=0$. Standard scores are sampled independently as $R_i\sim\mathrm{Beta}(2.0, 2.0)$ and $Q_i\sim\mathrm{Beta}(2.0, 2.0)$. The veto group is an adversarial scenario: those cases are intentionally assigned high predictive scores $R_i\sim\mathrm{Beta}(8.0, 2.0)$ so that predictive evidence strongly conflicts with the contextual constraint. The Top-$K$ tier uses $K=100$. Results are averaged over 1000 independent seeds (base seed 42). Contextual compliance is the complement $\mathrm{VPR}=1-V$ and is not plotted separately.

## Grouped comparison at representative $\lambda$ (Figure 6)

### $\lambda=0.50$

- $A_L$ (Linear): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.
- $A_G$ (Geometric): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.
- $A_M$ (Minimum): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.

### $\lambda=0.75$

- $A_L$ (Linear): $V$ = 0.077 (95% CI [0.020, 0.150]); std$(V)$ = 0.036.
- $A_G$ (Geometric): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.
- $A_M$ (Minimum): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.

### $\lambda=0.90$

- $A_L$ (Linear): $V$ = 0.462 (95% CI [0.370, 0.550]); std$(V)$ = 0.047.
- $A_G$ (Geometric): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.
- $A_M$ (Minimum): $V$ = 0.000 (95% CI [0.000, 0.000]); std$(V)$ = 0.000.

## Draft paragraphs

The first experiment evaluates the population-level consequences of the contextual-veto property established in Section~\ref{subsec:veto-preservation}. It is not a comparison of overall operator quality. The veto group is intentionally assigned high predictive scores in order to create the most challenging scenario for policy preservation, so that aggregation semantics can be evaluated when predictive evidence strongly conflicts with contextual constraints.

Figure~\ref{fig:policy-violation} summarizes the experiment in three panels. Panel~(a) shows the adversarial $(R,Q)$ population; panel~(b) the resulting $P_i$ distributions by operator at $\lambda\in\{0.50, 0.75, 0.90\}$; panel~(c) the Monte Carlo mean $V(\lambda)$ sweep.

At $\lambda=0.50$, the linear operator yields $V=0.000 (95% CI [0.000, 0.000])$. At $\lambda=0.75$, $V=0.077 (95% CI [0.020, 0.150])$. At $\lambda=0.90$, $V=0.462 (95% CI [0.370, 0.550])$. Thus linear aggregation does not always admit contextually excluded cases; violations emerge as predictive evidence receives more weight.

Under the same populations, the weighted geometric operator yields $V=0$ at all three representative values of $\lambda$, consistent with zero absorption.

The minimum operator likewise yields $V=0$ at these settings, again consistent with zero absorption rather than with a claim of overall superiority.

Panel~(c) of Figure~\ref{fig:policy-violation} shows $V$ as a function of $\lambda$ over $[0,1]$. This dense sweep makes the transition from no violations to systematic policy violations explicit.
 For the linear operator, the Monte Carlo mean of $V$ remains $0$ up to $\lambda=0.65$ and first becomes strictly positive at $\lambda=0.70$.
 Over the full sweep, the maximum Monte Carlo mean of $V$ is 0.000 for $A_G$ and 0.000 for $A_M$.
