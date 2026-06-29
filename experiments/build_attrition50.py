"""Build the curated 50-case attrition subsample from the 100-case cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from operators import aggregate

DATA100 = Path(__file__).resolve().parent / "data" / "attrition100"
DATA50 = Path(__file__).resolve().parent / "data" / "attrition50"

KEEP = {
    "Laura Baker",
    "Andrew Wood",
    "Alice Brown",
    "Andrew Ross",
    "Caroline Bennett",
    "Jessica Wilson",
    "Eleanor Foster",
    "Madison Foster",
    "Lucy Scott",
    "Camila Phillips",
    "Julia Smith",
}
EXCLUDE = {
    "Amelia Davis",
    "Dominic Collins",
    "Leah Cooper",
    "Richard Johnson",
    "Jason Phillips",
    "Evelyn Taylor",
    "Oliver Walker",
    "Noah Murphy",
    "Christian Bennett",
}


def select_cohort_ids(final: pd.DataFrame) -> set[str]:
    """Return 50 case IDs preserving narrative contrasts at Top-K = 5."""
    R, Q = final["R"].to_numpy(), final["Q"].to_numpy()
    final = final.copy()
    final["rank100_l05"] = pd.Series(aggregate("linear", R, Q, 0.5)).rank(
        ascending=False, method="min"
    ).astype(int)
    final["P90"] = aggregate("linear", R, Q, 0.9)

    selected = set(KEEP)
    selected |= set(
        final[(final["rank100_l05"] <= 35) & (final["Q"] > 0) & ~final["case_id"].isin(EXCLUDE)][
            "case_id"
        ]
    )
    selected |= set(
        final[(final["Q"] == 0) & (final["R"] < 0.88)].sample(6, random_state=2)["case_id"]
    )
    selected -= EXCLUDE

    while len(selected) > 50:
        droppable = final[final["case_id"].isin(selected) & ~final["case_id"].isin(KEEP)].sort_values(
            "P90", ascending=False
        )
        selected.remove(droppable.iloc[0]["case_id"])

    while len(selected) < 50:
        sub = final[final["case_id"].isin(selected)]
        yes = (sub["attrition"].str.lower() == "yes").sum()
        pool = final[~final["case_id"].isin(selected) & ~final["case_id"].isin(EXCLUDE)]
        side = pool[pool["attrition"].str.lower() == ("yes" if yes < 25 else "no")]
        if side.empty:
            side = pool
        selected.add(side.nsmallest(1, "rank100_l05").iloc[0]["case_id"])

    return selected


def build_attrition50() -> None:
    final_path = DATA100 / "cases_attrition_100_final.csv"
    raw_path = DATA100 / "cases_attrition_100.csv"
    final = pd.read_csv(final_path, sep=";")
    final = final.rename(
        columns={"Case_ID": "case_id", "Ri_Global_Risk": "R", "Ci_Context_Score": "Q"}
    )
    raw = pd.read_csv(raw_path, sep=";")

    ids = select_cohort_ids(final)
    DATA50.mkdir(parents=True, exist_ok=True)
    raw[raw["Case_ID"].isin(ids)].sort_values("Case_ID").to_csv(
        DATA50 / "cases_attrition_50.csv", sep=";", index=False
    )
    pd.read_csv(final_path, sep=";")[pd.read_csv(final_path, sep=";")["Case_ID"].isin(ids)].sort_values(
        "Case_ID"
    ).to_csv(DATA50 / "cases_attrition_50_final.csv", sep=";", index=False)


if __name__ == "__main__":
    build_attrition50()
    print(f"Wrote 50-case cohort to {DATA50}")
