"""Attach VNCoreNLP tokens to the manually labeled ground-truth rows.

The rest of the sentiment project (Lexicon_based, Build_sentiment_index) works
off ``data_news/data_tokenized/equity_news_tokenized_vncorenlp.parquet`` — a
word-segmentation of ``equity_news_clean_content.parquet`` produced by
VNCoreNLP. The Traditional_ML branch used to re-tokenize the ground truth with
``underthesea`` instead, so its features were built on a different segmentation
than every method it is meant to be compared against.

``ground_truth_labeled.csv`` carries a ``source_row_id`` column that is the
positional row index into that VNCoreNLP parquet (verified: ``title`` and
``publication_date`` match on every row). So instead of tokenizing again we
just look the rows up and copy the token columns across. No ``underthesea`` /
Java dependency is needed here anymore.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = SCRIPT_DIR / "data"

GROUND_TRUTH_CSV_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"
VNCORENLP_TOKENIZED_PATH = (
    PROJECT_ROOT
    / "data_news"
    / "data_tokenized"
    / "equity_news_tokenized_vncorenlp.parquet"
)

OUTPUT_PARQUET_PATH = DATA_DIR / "ground_truth_labeled_tokenized.parquet"
OUTPUT_CSV_PATH = DATA_DIR / "ground_truth_labeled_tokenized.csv"

SOURCE_ROW_ID_COLUMN = "source_row_id"
TITLE_COLUMN = "title"
PUBLICATION_DATE_COLUMN = "publication_date"
TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOTAL_TOKENIZER_COLUMN = "total_tokenizer"

CARRIED_CORPUS_COLUMNS = [
    TITLE_COLUMN,
    PUBLICATION_DATE_COLUMN,
    TOKENIZED_COLUMN,
    TOKENIZED_SENTENCES_COLUMN,
    TOTAL_TOKENIZER_COLUMN,
]

OUTPUT_COLUMNS = [
    "id",
    SOURCE_ROW_ID_COLUMN,
    "sentiment",
    "title",
    "content",
    TOKENIZED_COLUMN,
    TOKENIZED_SENTENCES_COLUMN,
    TOTAL_TOKENIZER_COLUMN,
]


def load_ground_truth(path: Path = GROUND_TRUTH_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {SOURCE_ROW_ID_COLUMN, "sentiment", "content", "title"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Ground truth file is missing columns: {sorted(missing_columns)}"
        )
    if df[SOURCE_ROW_ID_COLUMN].isna().any():
        raise ValueError("Ground truth file has null source_row_id values.")

    df = df.copy()
    df[SOURCE_ROW_ID_COLUMN] = df[SOURCE_ROW_ID_COLUMN].astype(int)
    return df


def lookup_vncorenlp_rows(ground_truth: pd.DataFrame) -> pd.DataFrame:
    if not VNCORENLP_TOKENIZED_PATH.exists():
        raise FileNotFoundError(
            f"VNCoreNLP tokenized corpus not found: {VNCORENLP_TOKENIZED_PATH}"
        )

    corpus = pd.read_parquet(
        VNCORENLP_TOKENIZED_PATH,
        columns=CARRIED_CORPUS_COLUMNS,
    )

    source_row_ids = ground_truth[SOURCE_ROW_ID_COLUMN].to_numpy()
    out_of_range = source_row_ids[
        (source_row_ids < 0) | (source_row_ids >= len(corpus))
    ]
    if out_of_range.size:
        raise ValueError(
            "source_row_id values fall outside the VNCoreNLP corpus "
            f"(0..{len(corpus) - 1}): {sorted(set(out_of_range.tolist()))}"
        )

    return corpus.iloc[source_row_ids].reset_index(drop=True)


def assert_alignment(ground_truth: pd.DataFrame, corpus_rows: pd.DataFrame) -> None:
    """The join is positional, so lock the implied contract down: the row we
    pulled from the corpus must be the same article the annotator labeled.
    """
    gt_title = ground_truth[TITLE_COLUMN].astype(str).to_numpy()
    corpus_title = corpus_rows[TITLE_COLUMN].astype(str).to_numpy()
    title_mismatch = gt_title != corpus_title

    gt_date = pd.to_datetime(ground_truth[PUBLICATION_DATE_COLUMN]).to_numpy()
    corpus_date = pd.to_datetime(corpus_rows[PUBLICATION_DATE_COLUMN]).to_numpy()
    date_mismatch = gt_date != corpus_date

    mismatch = title_mismatch | date_mismatch
    if mismatch.any():
        bad = ground_truth.loc[mismatch, [SOURCE_ROW_ID_COLUMN, TITLE_COLUMN]]
        raise ValueError(
            "source_row_id no longer lines up with the VNCoreNLP corpus for "
            f"{int(mismatch.sum())} row(s). Offending source_row_id / title:\n"
            f"{bad.to_string(index=False)}"
        )


def build_output(ground_truth: pd.DataFrame, corpus_rows: pd.DataFrame) -> pd.DataFrame:
    out = ground_truth.drop(columns=["title"], errors="ignore").copy()
    out["title"] = corpus_rows[TITLE_COLUMN].to_numpy()
    out[TOKENIZED_COLUMN] = list(corpus_rows[TOKENIZED_COLUMN])
    out[TOKENIZED_SENTENCES_COLUMN] = list(corpus_rows[TOKENIZED_SENTENCES_COLUMN])
    out[TOTAL_TOKENIZER_COLUMN] = corpus_rows[TOTAL_TOKENIZER_COLUMN].to_numpy()

    empty_tokens = out[TOKENIZED_COLUMN].map(len).eq(0)
    if empty_tokens.any():
        raise ValueError(
            f"{int(empty_tokens.sum())} ground-truth row(s) have no VNCoreNLP "
            "tokens; cannot build features for them."
        )

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ground_truth = load_ground_truth(GROUND_TRUTH_CSV_PATH)
    corpus_rows = lookup_vncorenlp_rows(ground_truth)
    assert_alignment(ground_truth, corpus_rows)
    output_df = build_output(ground_truth, corpus_rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Ground truth csv :", GROUND_TRUTH_CSV_PATH)
    print("VNCoreNLP corpus :", VNCORENLP_TOKENIZED_PATH)
    print("Output parquet   :", OUTPUT_PARQUET_PATH)
    print("Output csv       :", OUTPUT_CSV_PATH)
    print("Rows             :", len(output_df))
    print("Total tokens     :", int(output_df[TOTAL_TOKENIZER_COLUMN].sum()))
    print("Label counts:")
    print(output_df["sentiment"].value_counts().to_string())
    print("\nHead:")
    preview_columns = ["id", SOURCE_ROW_ID_COLUMN, "sentiment", "title", TOTAL_TOKENIZER_COLUMN]
    print(output_df[preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
