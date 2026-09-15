"""Filter data_news/ground_truth_combined.csv into tune / holdout subsets.

Reuses the split already committed for the Lexicon_based branch (mentor
feedback point A: tune/hyperparameter-search on one part, report the final
number on a part that was never looked at while iterating) - so every method
(Lexicon, Traditional ML, later PhoBERT) is tuned and reported on the exact
same rows. Does NOT create a new split.

    python News/Build_sentiment_label/Traditional_ML/improve/tune_holdout.py

Writes only into improve/tune_holdout/ (two CSVs, same schema as
ground_truth_combined.csv, ready to feed straight into
run_improve.py --ground-truth-csv / experiment_lexicon_features/run_experiment.py).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "tune_holdout"

GROUND_TRUTH_COMBINED_PATH = PROJECT_ROOT / "data_news" / "ground_truth_combined.csv"
SPLIT_PATH = (
    PROJECT_ROOT
    / "News"
    / "Build_sentiment_label"
    / "Lexicon_based"
    / "data"
    / "ground_truth_tune_holdout_split.csv"
)

TUNE_CSV_PATH = OUT_DIR / "ground_truth_tune.csv"
HOLDOUT_CSV_PATH = OUT_DIR / "ground_truth_holdout.csv"


def load_split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GROUND_TRUTH_COMBINED_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_COMBINED_PATH}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Tune/holdout split not found: {SPLIT_PATH}")

    combined = pd.read_csv(GROUND_TRUTH_COMBINED_PATH, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")[["source_row_id", "split"]]

    if "source_row_id" not in combined.columns:
        raise ValueError(f"{GROUND_TRUTH_COMBINED_PATH} is missing source_row_id")

    merged = combined.merge(split, on="source_row_id", how="left", validate="one_to_one")

    unmatched = merged["split"].isna()
    if unmatched.any():
        bad_ids = merged.loc[unmatched, "source_row_id"].tolist()
        raise ValueError(
            f"{int(unmatched.sum())} row(s) in {GROUND_TRUTH_COMBINED_PATH} have no "
            f"entry in {SPLIT_PATH} (source_row_id not found): {bad_ids[:10]}"
        )
    unexpected_values = set(merged["split"].unique()) - {"tune", "holdout"}
    if unexpected_values:
        raise ValueError(f"Unexpected split values: {sorted(unexpected_values)}")

    tune = (
        merged.loc[merged["split"].eq("tune")]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )
    holdout = (
        merged.loc[merged["split"].eq("holdout")]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    # tune and holdout must partition the combined set exactly - no row lost,
    # none duplicated, none shared between the two.
    overlap = set(tune["source_row_id"]) & set(holdout["source_row_id"])
    if overlap:
        raise AssertionError(f"tune/holdout overlap on source_row_id: {sorted(overlap)[:5]}")
    if len(tune) + len(holdout) != len(combined):
        raise AssertionError(
            f"tune ({len(tune)}) + holdout ({len(holdout)}) != combined ({len(combined)})"
        )

    return tune, holdout


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tune, holdout = load_split_frames()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tune.to_csv(TUNE_CSV_PATH, index=False, encoding="utf-8-sig")
    holdout.to_csv(HOLDOUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Ground truth combined:", GROUND_TRUTH_COMBINED_PATH, f"({len(tune) + len(holdout)} rows)")
    print("Split source         :", SPLIT_PATH)
    print()
    print(f"tune    -> {TUNE_CSV_PATH}  ({len(tune)} rows)")
    print(tune["sentiment"].value_counts().to_string())
    print()
    print(f"holdout -> {HOLDOUT_CSV_PATH}  ({len(holdout)} rows)")
    print(holdout["sentiment"].value_counts().to_string())
    print()
    print("Verified: tune and holdout partition the combined set exactly "
          "(no overlap, no missing row).")
    print()
    print("Use ONLY tune for iterating (feature ideas, hyperparameter search).")
    print("Touch holdout exactly once, at the very end, for the number you report.")


if __name__ == "__main__":
    main()
