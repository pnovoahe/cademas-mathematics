#!/usr/bin/env python3
"""
CADEMAS Hybrid Decision-Making Simulation Pipeline.

Generates synthetic data, runs aggregation experiments (RQ1-RQ3),
and saves publication-quality figures to paper/figures/.
"""

from experiments import (
    print_proposition_2_confirmation,
    run_all_experiments,
)
from plotting import plot_ml_overconfidence, plot_pareto_frontier


def main() -> None:
    print("CADEMAS Simulation Pipeline")
    print("-" * 40)

    df, pareto, overconfidence, homogeneity = run_all_experiments()

    print(f"Dataset generated: n={len(df)} alternatives")
    print(f"  std={sum(df['group'] == 'std')}, "
          f"veto={sum(df['group'] == 'veto')}, "
          f"safe={sum(df['group'] == 'safe')}")

    pareto_path = plot_pareto_frontier(pareto)
    print(f"\n[RQ1] Pareto frontier saved to: {pareto_path}")

    overconf_path = plot_ml_overconfidence(overconfidence)
    print(f"[RQ2] ML overconfidence plot saved to: {overconf_path}")

    print_proposition_2_confirmation(homogeneity)

    print("Homogeneity table (all Q strata):")
    pivot = homogeneity.pivot(index="q_level", columns="operator", values="tau")
    pivot = pivot[["linear", "min", "max", "geometric"]]
    print(pivot.round(2).to_string())
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
