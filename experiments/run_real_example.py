"""Run the real attrition example and generate the manuscript figure."""

from real_example import print_summary, summarize_example
from real_example_plotting import plot_real_example, results_table_latex


def main() -> None:
    df = summarize_example()
    out = plot_real_example(df)
    print_summary(df)
    print(f"\nFigure saved to: {out}")
    print("\n--- LaTeX table rows (sorted by P_linear) ---")
    print(results_table_latex(df))


if __name__ == "__main__":
    main()
