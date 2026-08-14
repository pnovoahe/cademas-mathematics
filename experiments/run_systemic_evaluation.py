#!/usr/bin/env python3
"""
Systemic CADEMAS evaluation: publication figures.

Outputs (first_round/figures/):
  - opportunity_cost.pdf
  - predictive_overconfidence.pdf
  - noise_propagation.pdf
  - intermediate_context.pdf

Aggregated Monte Carlo summaries are cached in experiments/.cache/mc/.
Recompute with --refresh (or CADEMAS_MC_REFRESH=1).
"""

import argparse

from systemic_config import (
    MC_SEED_BASE,
    N_MC_TRIALS,
    NOISE_LAMBDAS,
    PREDICTIVE_OVERCONFIDENCE_LAMBDA,
)
from systemic_experiments import run_all_systemic_experiments_mc
from systemic_plotting import (
    plot_intermediate_context,
    plot_noise_propagation,
    plot_opportunity_cost,
    plot_predictive_overconfidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Systemic CADEMAS evaluation figures")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute Monte Carlo trials and overwrite the result cache",
    )
    args = parser.parse_args()

    print("Systemic CADEMAS Evaluation (Monte Carlo)")
    print(
        f"Trials: {N_MC_TRIALS} | Seed base: {MC_SEED_BASE} | "
        f"Predictive overconfidence lambda: {PREDICTIVE_OVERCONFIDENCE_LAMBDA} | "
        f"Rank stability lambdas: {NOISE_LAMBDAS} | "
        f"refresh={args.refresh}"
    )
    print("-" * 50)

    opp, overconfidence, noise, intermediate = run_all_systemic_experiments_mc(
        refresh=args.refresh
    )

    p1 = plot_opportunity_cost(opp)
    print(f"[Exp 1] Opportunity cost  -> {p1}")

    p2 = plot_predictive_overconfidence(overconfidence, lam=PREDICTIVE_OVERCONFIDENCE_LAMBDA)
    print(f"[Exp 2] Predictive overconfidence -> {p2}")

    p3 = plot_noise_propagation(noise)
    print(f"[Exp 3] Noise propagation -> {p3}")

    p4 = plot_intermediate_context(intermediate)
    print(f"[Exp 4] Intermediate context -> {p4}")

    print("\nSummary (mean over trials, max V_K):")
    for op in ("linear", "geometric", "min"):
        sub = overconfidence[overconfidence["operator"] == op]
        mx = sub["v_k_mean"].max()
        print(f"  Overconfidence {op:10s}: max mean V_K = {mx:.3f}")

    print("\nSummary (rank stability at sigma=0.5, averaged over population/noise):")
    for lam in NOISE_LAMBDAS:
        sub = noise[(noise["lambda"] == lam) & (noise["sigma"] >= 0.499)]
        for op in ("linear", "min"):
            row = sub[sub["operator"] == op]
            if len(row):
                tau = row["tau_mean"].mean()
                jac = row["jaccard_mean"].mean()
                print(f"  lambda={lam:.2f} {op:6s}: tau={tau:.3f}, jaccard={jac:.3f}")

    lin = opp[opp["operator"] == "linear"].sort_values("lambda")
    compliant = lin[lin["v_k_ci_high"] == 0]
    if len(compliant):
        print(f"\nRQ1 linear max lambda (95% CI upper bound = 0): {compliant['lambda'].max():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
