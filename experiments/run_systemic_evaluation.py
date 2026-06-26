#!/usr/bin/env python3
"""
Systemic CADEMAS evaluation: three publication figures.

Outputs (paper/figures/):
  - opportunity_cost.pdf
  - algorithmic_firewall.pdf
  - noise_propagation.pdf
"""

from systemic_config import FIREWALL_LAMBDA, MC_SEED_BASE, N_MC_TRIALS, NOISE_LAMBDA
from systemic_experiments import run_all_systemic_experiments_mc
from systemic_plotting import (
    plot_algorithmic_firewall,
    plot_noise_propagation,
    plot_opportunity_cost,
)


def main() -> None:
    print("Systemic CADEMAS Evaluation (Monte Carlo)")
    print(
        f"Trials: {N_MC_TRIALS} | Seed base: {MC_SEED_BASE} | "
        f"Firewall lambda: {FIREWALL_LAMBDA} | Noise lambda: {NOISE_LAMBDA}"
    )
    print("-" * 50)

    opp, firewall, noise = run_all_systemic_experiments_mc()

    p1 = plot_opportunity_cost(opp)
    print(f"[Exp 1] Opportunity cost  -> {p1}")

    p2 = plot_algorithmic_firewall(firewall, lam=FIREWALL_LAMBDA)
    print(f"[Exp 2] Algorithmic firewall -> {p2}")

    p3 = plot_noise_propagation(noise)
    print(f"[Exp 3] Noise propagation -> {p3}")

    print("\nSummary (mean over trials, max V_K):")
    for op in ("linear", "geometric", "min"):
        sub = firewall[firewall["operator"] == op]
        mx = sub["v_k_mean"].max()
        print(f"  Firewall {op:10s}: max mean V_K = {mx:.3f}")

    print("\nSummary (mean tau at sigma=0.5):")
    for op in ("linear", "min"):
        row = noise[(noise["operator"] == op) & (noise["sigma"] == 0.5)]
        if len(row):
            print(f"  Noise {op:10s}: tau = {row['tau_mean'].values[0]:.3f}")

    lin = opp[opp["operator"] == "linear"].sort_values("lambda")
    compliant = lin[lin["v_k_ci_high"] == 0]
    if len(compliant):
        print(f"\nRQ1 linear max lambda (95% CI upper bound = 0): {compliant['lambda'].max():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
