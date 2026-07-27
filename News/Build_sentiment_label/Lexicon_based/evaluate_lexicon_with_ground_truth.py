from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
LEXICON_DATA_DIR = SCRIPT_DIR / "data"

GROUND_TRUTH_PATH = PROJECT_ROOT / "data_News" / "ground_truth_label.csv"
LEXICON_OUTPUT_PATH = LEXICON_DATA_DIR / "equity_news_content_sentiment_ratios.parquet"

OUTPUT_DETAIL_PATH = LEXICON_DATA_DIR / "lexicon_ground_truth_evaluation_detail.csv"
OUTPUT_ERRORS_PATH = LEXICON_DATA_DIR / "lexicon_ground_truth_evaluation_errors.csv"
OUTPUT_METRICS_PATH = LEXICON_DATA_DIR / "lexicon_ground_truth_metrics.json"
OUTPUT_CONFUSION_MATRIX_PATH = (
    LEXICON_DATA_DIR / "lexicon_ground_truth_confusion_matrix.csv"
)

GT_LABEL_COLUMN = "sentiment"
LEXICON_LABEL_COLUMN = "sentiment_label"
VALID_LABELS = ["positive", "negative", "neutral"]

LABEL_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "tích cực": "positive",
    "tich cuc": "positive",
    "negative": "negative",
    "neg": "negative",
    "tiêu cực": "negative",
    "tieu cuc": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "trung lập": "neutral",
    "trung lap": "neutral",
}


def normalize_label(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    label = str(value).strip().casefold()
    return LABEL_ALIASES.get(label, label)


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {"source_row_id", GT_LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Ground truth file is missing columns: {sorted(missing_columns)}"
        )

    out = df.copy()
    out["source_row_id"] = pd.to_numeric(out["source_row_id"], errors="coerce")
    out = out.loc[out["source_row_id"].notna()].copy()
    out["source_row_id"] = out["source_row_id"].astype("int64")
    out["ground_truth_label"] = out[GT_LABEL_COLUMN].apply(normalize_label)
    return out


def load_lexicon_output(path: Path = LEXICON_OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index(names="source_row_id")
    required_columns = {"source_row_id", LEXICON_LABEL_COLUMN, "sentiment_score"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Lexicon output file is missing columns: {sorted(missing_columns)}"
        )

    out = df.copy()
    out["source_row_id"] = pd.to_numeric(out["source_row_id"], errors="coerce")
    out = out.loc[out["source_row_id"].notna()].copy()
    out["source_row_id"] = out["source_row_id"].astype("int64")
    out["lexicon_label"] = out[LEXICON_LABEL_COLUMN].apply(normalize_label)
    return out


def build_evaluation_dataframe(
    ground_truth_df: pd.DataFrame,
    lexicon_df: pd.DataFrame,
) -> pd.DataFrame:
    keep_columns = [
        "source_row_id",
        "sentiment_score",
        "lexicon_label",
        "pos_count",
        "neg_count",
        "neutral_count",
        "polarity_count",
        "pos_ratio",
        "neg_ratio",
        "neutral_ratio",
        "sentiment_coverage_ratio",
    ]
    keep_columns = [column for column in keep_columns if column in lexicon_df.columns]

    merged_df = ground_truth_df.merge(
        lexicon_df[keep_columns],
        on="source_row_id",
        how="left",
    )

    eval_df = merged_df.loc[
        merged_df["ground_truth_label"].isin(VALID_LABELS)
        & merged_df["lexicon_label"].isin(VALID_LABELS)
    ].copy()
    eval_df["is_correct"] = eval_df["ground_truth_label"].eq(
        eval_df["lexicon_label"]
    )
    return eval_df


def build_metrics(eval_df: pd.DataFrame) -> dict[str, object]:
    y_true = eval_df["ground_truth_label"]
    y_pred = eval_df["lexicon_label"]

    report = classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {
        "rows_evaluated": int(len(eval_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
    }


def save_outputs(eval_df: pd.DataFrame, metrics: dict[str, object]) -> None:
    y_true = eval_df["ground_truth_label"]
    y_pred = eval_df["lexicon_label"]

    confusion = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    )

    OUTPUT_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_DETAIL_PATH, index=False, encoding="utf-8-sig")
    eval_df.loc[~eval_df["is_correct"]].to_csv(
        OUTPUT_ERRORS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, encoding="utf-8-sig")
    OUTPUT_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Rows evaluated:", metrics["rows_evaluated"])
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro F1:", round(metrics["macro_f1"], 4))
    print("Weighted F1:", round(metrics["weighted_f1"], 4))
    print("\nConfusion matrix:")
    print(confusion_df.to_string())
    print("\nSaved detail:", OUTPUT_DETAIL_PATH)
    print("Saved errors:", OUTPUT_ERRORS_PATH)
    print("Saved metrics:", OUTPUT_METRICS_PATH)
    print("Saved confusion matrix:", OUTPUT_CONFUSION_MATRIX_PATH)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ground_truth_df = load_ground_truth()
    lexicon_df = load_lexicon_output()
    eval_df = build_evaluation_dataframe(ground_truth_df, lexicon_df)

    if eval_df.empty:
        raise ValueError("No valid rows to evaluate after merging ground truth and lexicon output.")

    metrics = build_metrics(eval_df)
    save_outputs(eval_df, metrics)


if __name__ == "__main__":
    main()
