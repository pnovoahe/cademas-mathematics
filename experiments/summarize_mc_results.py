#!/usr/bin/env python3
"""Print key manuscript statistics from Monte Carlo experiments."""

import argparse

from systemic_experiments import run_all_systemic_experiments_mc


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print manuscript Monte Carlo statistics")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute Monte Carlo trials and overwrite the result cache",
    )
    args = parser.parse_args()
    opp, overconf, noise, inter = run_all_systemic_experiments_mc(refresh=args.refresh)

    print("=== Policy compliance (linear) ===")
    lin = opp[opp["operator"] == "linear"].sort_values("lambda")
    compliant = lin[lin["v_k_ci_high"] == 0]
    if len(compliant):
        print(f"Max lambda with CI_hi(V_K)=0: {compliant['lambda'].max():.2f}")
    row05 = lin.iloc[0]
    row95 = lin[lin["lambda"] >= 0.949].iloc[0]
    lo_lam = lin[lin["lambda"] <= 0.05 + 1e-9].iloc[0]
    hi_compliant = compliant.iloc[-1] if len(compliant) else None
    if hi_compliant is not None:
        print(
            f"At max compliant lambda={hi_compliant['lambda']:.2f}: "
            f"R_K={_fmt(hi_compliant['efficiency_mean'])} "
            f"[{_fmt(hi_compliant['efficiency_ci_low'])},{_fmt(hi_compliant['efficiency_ci_high'])}]"
        )
    print(
        f"At lambda=0.95: V_K={_fmt(row95['v_k_mean'])} "
        f"[{_fmt(row95['v_k_ci_low'])},{_fmt(row95['v_k_ci_high'])}], "
        f"R_K={_fmt(row95['efficiency_mean'])}"
    )

    print("\n=== Predictive overconfidence (linear, mu=1.0) ===")
    lin_oc = overconf[overconf["operator"] == "linear"]
    row1 = lin_oc[lin_oc["mu_shift"] >= 0.999].iloc[0]
    print(
        f"V_K={_fmt(row1['v_k_mean'])} "
        f"[{_fmt(row1['v_k_ci_low'])},{_fmt(row1['v_k_ci_high'])}]"
    )
    # threshold where CI lower bound > 0
    rising = lin_oc[lin_oc["v_k_ci_low"] > 0]
    if len(rising):
        print(f"First mu with CI_lo>0: {rising['mu_shift'].min():.2f}")

    print("\n=== Rank stability (sigma=0.5, avg over population/noise) ===")
    sub = noise[noise["sigma"] >= 0.499]
    for lam in (0.25, 0.50, 0.75):
        block = sub[sub["lambda"] == lam]
        print(f"lambda={lam:.2f}")
        for op in ("linear", "min"):
            rows = block[block["operator"] == op]
            tau_m = rows["tau_mean"].mean()
            tau_lo = rows["tau_ci_low"].mean()
            tau_hi = rows["tau_ci_high"].mean()
            jac_m = rows["jaccard_mean"].mean()
            jac_lo = rows["jaccard_ci_low"].mean()
            jac_hi = rows["jaccard_ci_high"].mean()
            print(
                f"  {op}: tau={_fmt(tau_m)} [{_fmt(tau_lo)},{_fmt(tau_hi)}], "
                f"jaccard={_fmt(jac_m)} [{_fmt(jac_lo)},{_fmt(jac_hi)}]"
            )

    print("\n=== Intermediate context ===")
    for op in ("linear", "geometric"):
        sub = inter[inter["operator"] == op].sort_values("lambda")
        # first lambda where mean > 0.01
        onset = sub[sub["acceptance_mean"] > 0.01]
        row90 = sub[sub["lambda"] >= 0.899].iloc[0]
        print(f"{op}: onset lambda>{onset.iloc[0]['lambda']:.2f}" if len(onset) else f"{op}: no onset")
        print(
            f"  lambda=0.9: acc={_fmt(row90['acceptance_mean'])} "
            f"[{_fmt(row90['acceptance_ci_low'])},{_fmt(row90['acceptance_ci_high'])}]"
        )
        mid = sub[(sub["lambda"] >= 0.59) & (sub["lambda"] <= 0.61)].iloc[0]
        print(
            f"  lambda=0.6: acc={_fmt(mid['acceptance_mean'])} "
            f"[{_fmt(mid['acceptance_ci_low'])},{_fmt(mid['acceptance_ci_high'])}]"
        )


if __name__ == "__main__":
    main()
