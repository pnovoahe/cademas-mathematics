# Narrative summary — Experiment 6.1 Policy compliance under contextual vetoes

This file is a **text-only briefing** for humans and LLM agents working on
`first_round_v2/manuscriptR1V2.tex` Section 6.1. Do not invent numerical claims.
Do not describe any operator as globally superior.

- Code: `simulations/01_policy_compliance_under_contextual_vetoes/run.py`
- Config: `simulations/common/config.py` (`veto_fraction=0.20`)
- Seeds: 42 through 1041

## Scientific message

Section 6.1 shows the **population-level consequences** of the contextual-veto
property from Section 4. The question is when policy violations **emerge**, not
which operator is better.

The veto group is **adversarial**: $Q_i=0$ and $R_i\sim\mathrm{Beta}(8.0, 2.0)$,
so predictive evidence strongly conflicts with the contextual constraint.

Contextual compliance is $\mathrm{VPR}=1-V$. It is mentioned in the text and
**not** given its own figure. Predictive utility $\bar{R}$ is stored in CSV
files for later sections and is **not** discussed in 6.1.

## Design (this run)

| Parameter | Value |
|---|---|
| $N$ | 1000 |
| `veto_fraction` | 0.20 |
| Veto cases | 200 |
| Standard cases | 800 |
| $K$ | 100 |
| Monte Carlo | 1000 |
| Figure 6 $\lambda$ | (0.5, 0.75, 0.9) |
| Figure 7 $\lambda$ | 21 points on $[0,1]$ |

## Headline results

1. At $\lambda=0.50$, linear $V=0.000 [0.000, 0.000]$. Geometric and minimum remain at $V=0$.
2. At $\lambda=0.75$, linear $V=0.077 [0.020, 0.150]$. Geometric and minimum remain at $V=0$.
3. At $\lambda=0.90$, linear $V=0.462 [0.370, 0.550]$. Geometric and minimum remain at $V=0$.
4. Linear aggregation **does not always** violate vetoes. Violations emerge as $\lambda$ increases.
5. Figure 7: linear mean $V$ is $0$ until late in the sweep; first strictly positive mean at $\lambda=0.7000000000000001$. $A_G$ and $A_M$ stay at $V=0$ for every $\lambda\in[0,1]$.

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

| $\lambda$ | $A_L$ mean $V$ [CI] | $A_G$ | $A_M$ |
|---|---|---|---|
| 0.50 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 0.75 | 0.077 [0.020, 0.150] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 0.90 | 0.462 [0.370, 0.550] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

At 0.50 all three bars sit on the axis. At 0.75 a small linear bar appears.
At 0.90 the linear bar is large; geometric and minimum remain zero. The linear
CI at 0.90 does not overlap zero.

## Figure 7 — dense sweep $V(\lambda)$ (primary)

File: `figures/policy_violation_rate_dense.pdf`

This is a **primary** result of the subsection: it shows the transition from
no violations to systematic violations.

ASCII of linear **mean $V$** (`*` scaled to 0.65). Geometric and minimum are
the zero line at every $\lambda$.

```
λ     A_L mean V     spark
0.00  0.000          |
0.05  0.000          |
0.10  0.000          |
0.15  0.000          |
0.20  0.000          |
0.25  0.000          |
0.30  0.000          |
0.35  0.000          |
0.40  0.000          |
0.45  0.000          |
0.50  0.000          |
0.55  0.000          |
0.60  0.000          |
0.65  0.000          |
0.70  0.001          |  <- first strictly positive mean
0.75  0.077          |**
0.80  0.214          |****
0.85  0.345          |*******
0.90  0.462          |*********
0.95  0.561          |***********
1.00  0.644          |*************
```

Linear 95% CI along the rise:

| $\lambda$ | mean $V$ | CI low | CI high |
|---|---|---|---|
| 0.65 | 0.000 | 0.00 | 0.00 |
| 0.70 | 0.001 | 0.00 | 0.02 |
| 0.75 | 0.077 | 0.02 | 0.15 |
| 0.80 | 0.214 | 0.13 | 0.30 |
| 0.85 | 0.345 | 0.25 | 0.43 |
| 0.90 | 0.462 | 0.37 | 0.55 |
| 0.95 | 0.561 | 0.47 | 0.65 |
| 1.00 | 0.644 | 0.55 | 0.73 |

At $\lambda=1$, $A_L(R,Q)=R$, so ranking ignores context.

## Table 2

`tables/table_policy_compliance.tex`: operator × $\lambda$ for Figure 6 values,
with mean, std, and 95% CI of $V$ only.

## Safe claims for the manuscript

- The experiment is an adversarial test of veto preservation, not a bake-off.
- Linear $V=0$ at $\lambda=0.50$; violations emerge at higher $\lambda$.
- $A_G$ and $A_M$ keep $V=0$ here because they absorb zero, not because they are "better".
- $\mathrm{VPR}=1-V$; no VPR figure.
- Do not discuss $\bar{R}$ in Section 6.1.

## File map

| Need | Path |
|---|---|
| Figure 6 | `figures/policy_violation_rate_grouped.pdf` |
| Figure 7 | `figures/policy_violation_rate_dense.pdf` |
| Table 2 | `tables/table_policy_compliance.tex` |
| Prose | `manuscript_snippets.md` |
| Captions | `captions.md` |
| Extra `veto_fraction` | `python run.py --veto-fraction 0.10` |
