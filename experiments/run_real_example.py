"""Run the real attrition example and generate the manuscript figure."""

from real_example import describe_cohort, print_summary, summarize_example
from real_example_plotting import plot_real_example, results_table_latex


def main() -> None:
    stats = describe_cohort()
    print("Cohort composition:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print()

    df = summarize_example()
    plane_path, bump_path = plot_real_example(df)
    print_summary(df)
    print(f"\nFigures saved to:\n  {plane_path}\n  {bump_path}")
    print("\n--- LaTeX table rows (Top-5 by P_linear at lambda=0.5) ---")
    print(results_table_latex(df))


if __name__ == "__main__":
    main()
