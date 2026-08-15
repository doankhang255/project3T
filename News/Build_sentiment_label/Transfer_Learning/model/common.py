from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_LEARNING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRANSFER_LEARNING_DIR.parents[2]
DATA_DIR = TRANSFER_LEARNING_DIR / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.tokenize_underthesea import (  # noqa: E402
    tokenize_vietnamese_text,
)


GROUND_TRUTH_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"

BASE_MODEL_PATH = (
    TRANSFER_LEARNING_DIR
    / "Model_Output"
    / "hub"
    / "models--wonrax--phobert-base-vietnamese-sentiment"
    / "snapshots"
    / "9076a5896971b5d551588fe8a51c722c89731d36"
)

RAW_TEXT_COLUMN = "content"
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"

# Same ordering as Traditional_ML/model/common.py so label ids line up
# across methods when results are compared side by side later.
VALID_LABELS = ["negative", "neutral", "positive"]
LABEL_ALIASES = {
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}

RANDOM_SEED = 42
MAX_LENGTH = 256
VAL_SIZE = 0.2


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return LABEL_ALIASES.get(str(value).strip().casefold())


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> pd.DataFrame:
    """Load the 152-row manually labeled ground truth and word-segment its
    raw article text so it matches the format PhoBERT's BPE tokenizer was
    pretrained on (underscore-joined compound words), e.g. "cong_ty" not
    "cong ty". Uses the same ground_truth_labeled.csv as Traditional_ML, so
    results are comparable across methods.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {RAW_TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Ground truth file is missing columns: {sorted(missing_columns)}")

    out = df.copy()
    out["ground_truth_label"] = out[LABEL_COLUMN].apply(normalize_label)
    out = out.loc[out["ground_truth_label"].isin(VALID_LABELS)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid labels found in ground truth file.")

    out[TEXT_COLUMN] = out[RAW_TEXT_COLUMN].apply(
        lambda text: " ".join(tokenize_vietnamese_text(text))
    )
    out = out.loc[out[TEXT_COLUMN].str.len().gt(0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No non-empty article text left after tokenization.")

    return out


def encode_labels(labels: pd.Series) -> list[int]:
    label_to_id = {label: index for index, label in enumerate(VALID_LABELS)}
    return labels.map(label_to_id).tolist()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def compute_metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Same precision/recall/f1/accuracy layout as
    Traditional_ML/model/common.py's compute_metrics, kept as a separate
    copy here (rather than imported) so Transfer_Learning stays a
    self-contained method branch, consistent with Lexicon_based and
    Traditional_ML each owning their own evaluation code.
    """
    rows = []
    f1_values = []
    for label_id, label in enumerate(VALID_LABELS):
        true_positive = int(((y_true == label_id) & (y_pred == label_id)).sum())
        false_positive = int(((y_true != label_id) & (y_pred == label_id)).sum())
        false_negative = int(((y_true == label_id) & (y_pred != label_id)).sum())
        support = int((y_true == label_id).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_values.append(f1_score)
        rows.append(
            {
                "metric_scope": label,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": support,
            }
        )

    rows.append(
        {
            "metric_scope": "overall",
            "precision": np.nan,
            "recall": np.nan,
            "f1": float(np.mean(f1_values)),
            "support": int(len(y_true)),
            "accuracy": float((y_true == y_pred).mean()),
        }
    )
    return pd.DataFrame(rows)
