"""Tune(730) -> Holdout(314), ONCE - the production Random Forest (+lexicon,
isotonic) number needed to compare this branch against Lexicon_based and
Transfer_Learning on the same protocol: train on Tune, predict once on
Holdout (see ``../../Transfer_Learning/transfer_learning_report.qmd``'s E2/E3
"Holdout (314 dong, kiem tra mot lan)" and
``../../Lexicon_based/LEXICON_SUMMARY.qmd``'s "Chi tap Holdout").

Every number that already existed in this branch is either 10 x 5-fold CV on the
full 1044 rows (RESULTS_SUMMARY.txt) or a repeated-CV diagnostic on Tune or
on Holdout treated as its own little dataset
(``experiment_only_TF_IDF/RESULTS.txt``, ``holdout_check_1044/RESULTS.txt``)
- none of those is "fit once on Tune, predict once on Holdout", so this
number did not exist anywhere in this branch until now.

WARNING - holdout re-use: these same 314 rows were already looked at once in
``ML_SUMMARY.qmd`` section 6.2, to decide whether random_forest should use
the lexicon feature block at all. This run answers a DIFFERENT question (how
does the whole Traditional_ML branch compare to the other two branches?), so
a first look for THIS question is still within the "look once per
independent question" discipline established across this branch - but the
resulting holdout number is not from a fully untouched split anymore.
``compare_branches_holdout.py`` states this explicitly rather than presenting
it as a clean confirmation.

    python News/Build_sentiment_label/Traditional_ML/experiment_Lexicon_features/holdout_once.py

Writes holdout_once/random_forest_holdout_predictions.csv - same columns as
Transfer_Learning/improve/data/finetune_holdout_predictions.csv - so
compare_branches_holdout.py can join the two on source_row_id.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
    load_frame_from_csv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    VALID_LABELS,
    build_prediction_output,
    compute_metrics,
    confusion_matrix,
    encode_labels,
    run_train_test_split,
)
from News.Build_sentiment_label.Traditional_ML.Common.model import random_forest
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    build_lexicon_feature_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TUNE_HOLDOUT_DIR = SCRIPT_DIR.parent / "Common" / "tune_holdout"
TUNE_CSV_PATH = TUNE_HOLDOUT_DIR / "ground_truth_tune.csv"
HOLDOUT_CSV_PATH = TUNE_HOLDOUT_DIR / "ground_truth_holdout.csv"

OUT_DIR = SCRIPT_DIR / "holdout_once"
PREDICTIONS_PATH = OUT_DIR / "random_forest_holdout_predictions.csv"
METRICS_PATH = OUT_DIR / "random_forest_holdout_metrics.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tune_df = load_frame_from_csv(TUNE_CSV_PATH)
    holdout_df = load_frame_from_csv(HOLDOUT_CSV_PATH)
    print(f"Tune: {len(tune_df)} rows | Holdout: {len(holdout_df)} rows (used ONCE below)")

    tune_counts = build_document_term_counts(tune_df)
    holdout_counts = build_document_term_counts(holdout_df)
    y_tune = encode_labels(tune_df["ground_truth_label"])
    y_holdout = encode_labels(holdout_df["ground_truth_label"])

    lexicon_tune = build_lexicon_feature_matrix(tune_df["Tokenize_content"].tolist())
    lexicon_holdout = build_lexicon_feature_matrix(holdout_df["Tokenize_content"].tolist())

    probabilities, predictions = run_train_test_split(
        random_forest.build_estimator,
        tune_counts,
        y_tune,
        holdout_counts,
        extra_features_train=lexicon_tune,
        extra_features_test=lexicon_holdout,
    )

    metrics_df = compute_metrics(y_holdout, predictions)
    prediction_df = build_prediction_output(holdout_df, probabilities, predictions)
    confusion_df = pd.DataFrame(
        confusion_matrix(y_holdout, predictions),
        index=[f"true_{lbl}" for lbl in VALID_LABELS],
        columns=[f"pred_{lbl}" for lbl in VALID_LABELS],
    ).reset_index(names="true_label")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    print(
        f"\nRandom Forest (+lexicon, isotonic) - Tune({len(tune_df)}) -> "
        f"Holdout({len(holdout_df)}), ONCE"
    )
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print("\nOutput predictions:", PREDICTIONS_PATH)
    print("Output metrics:", METRICS_PATH)
    print(
        f"\nNOTE: these {len(holdout_df)} holdout rows were already used once (ML_SUMMARY.qmd "
        "section 6.2, deciding whether random_forest should use the lexicon "
        "feature block). This run answers a different question (3-branch "
        "comparison) but is not a fully untouched holdout any more - see "
        "compare_branches_holdout.py's RESULTS.txt for the full disclosure."
    )


if __name__ == "__main__":
    main()
