from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_LEARNING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRANSFER_LEARNING_DIR.parents[2]
DATA_DIR = TRANSFER_LEARNING_DIR / "data"


GROUND_TRUTH_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"

# The ground truth is joined to this VNCoreNLP word-segmentation of the corpus
# by ``source_row_id`` instead of being re-tokenized here. VNCoreNLP is the
# segmentation PhoBERT was pretrained on, the one E1 (pretrain/) adapts on, and
# the one Lexicon_based / Traditional_ML compare against - so all branches now
# feed PhoBERT identically segmented text.
VNCORENLP_TOKENIZED_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
SOURCE_ROW_ID_COLUMN = "source_row_id"
TOKENIZED_COLUMN = "Tokenize_content"

# E1 output: vinai/phobert-base-v2 after domain-adaptive MLM pretraining on the
# 126k unlabeled equity-news corpus (val perplexity 8.25 -> 3.82). E2 (frozen
# feature probe) and E3 (fine-tune) both start from this instead of the raw base
# or the wonrax e-commerce-review sentiment checkpoint.
BASE_MODEL_PATH = TRANSFER_LEARNING_DIR / "Model_Output" / "phobert_domain_adapted"

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
    """Load the manually labeled ground truth and attach the VNCoreNLP word
    segmentation of each article (underscore-joined compounds, e.g. "cong_ty"
    not "cong ty" - the format PhoBERT's BPE expects).

    Instead of re-tokenizing ``content`` here, the rows are looked up in
    ``equity_news_tokenized_vncorenlp.parquet`` by ``source_row_id`` (the same
    positional join ``Traditional_ML/prepare_ground_truth.py`` uses). Keeps the
    segmentation identical to E1 pretraining and to the other method branches.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    if not VNCORENLP_TOKENIZED_PATH.exists():
        raise FileNotFoundError(
            f"VNCoreNLP tokenized corpus not found: {VNCORENLP_TOKENIZED_PATH}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {SOURCE_ROW_ID_COLUMN, LABEL_COLUMN, "title"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Ground truth file is missing columns: {sorted(missing_columns)}")
    if df[SOURCE_ROW_ID_COLUMN].isna().any():
        raise ValueError("Ground truth file has null source_row_id values.")

    out = df.copy()
    out[SOURCE_ROW_ID_COLUMN] = out[SOURCE_ROW_ID_COLUMN].astype(int)
    out["ground_truth_label"] = out[LABEL_COLUMN].apply(normalize_label)
    out = out.loc[out["ground_truth_label"].isin(VALID_LABELS)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid labels found in ground truth file.")

    corpus = pd.read_parquet(
        VNCORENLP_TOKENIZED_PATH, columns=["title", TOKENIZED_COLUMN]
    )
    row_ids = out[SOURCE_ROW_ID_COLUMN].to_numpy()
    if ((row_ids < 0) | (row_ids >= len(corpus))).any():
        raise ValueError(
            f"source_row_id values fall outside the VNCoreNLP corpus (0..{len(corpus) - 1})."
        )
    corpus_rows = corpus.iloc[row_ids].reset_index(drop=True)

    # The join is positional; make sure we pulled the article the annotator saw.
    title_mismatch = (
        out["title"].astype(str).to_numpy() != corpus_rows["title"].astype(str).to_numpy()
    )
    if title_mismatch.any():
        raise ValueError(
            "source_row_id no longer lines up with the VNCoreNLP corpus for "
            f"{int(title_mismatch.sum())} row(s)."
        )

    out[TEXT_COLUMN] = [
        " ".join(str(token) for token in tokens if str(token).strip())
        for tokens in corpus_rows[TOKENIZED_COLUMN]
    ]
    out = out.loc[out[TEXT_COLUMN].str.len().gt(0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No non-empty article text left after joining VNCoreNLP tokens.")

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
