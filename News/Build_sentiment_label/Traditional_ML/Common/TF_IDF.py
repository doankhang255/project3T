"""TF-IDF feature builder for the Traditional_ML sentiment models.

Two entry points:

* ``fit_tfidf_vocabulary`` / ``transform_tfidf`` — the fit/transform pair the
  cross-validation loop in ``model/common.py`` calls **per fold**, so the
  vocabulary, document frequencies and IDF weights are learned from the
  training rows only and never see the held-out fold.

* ``main`` — fits the vocabulary on the **whole** ground-truth set and
  writes ``tfidf_vocabulary.csv``, a descriptive artifact (run_pipeline.py
  reports its size / n-gram mix in RESULTS_SUMMARY.txt); it is **not** the
  input the models evaluate on.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import (
    NGRAM_SEPARATOR,
    build_document_terms as build_common_document_terms,
)
from News.Build_sentiment_label.Common.ngram_filter import (
    choose_ngram_terms,
    scaled_min_df_by_ngram,
)
from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

INPUT_PARQUET_PATH = DATA_DIR / "ground_truth_labeled_tokenized.parquet"

OUTPUT_VOCAB_PATH = DATA_DIR / "tfidf_vocabulary.csv"

TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"

NGRAM_RANGE = (1, 3)
REMOVE_STOPWORDS = True

ML_MIN_DF_RATIO_BY_NGRAM = {1: 0.02, 2: 0.013, 3: 0.013}
ML_MIN_DF_FLOOR = 2
ML_MAX_DF_RATIO = 0.85


def build_document_terms(row: pd.Series) -> list[str]:
    min_ngram, max_ngram = NGRAM_RANGE

    if TOKENIZED_SENTENCES_COLUMN in row.index:
        terms = build_common_document_terms(
            row[TOKENIZED_SENTENCES_COLUMN],
            min_n=min_ngram,
            max_n=max_ngram,
        )
        if terms:
            return terms

    return build_common_document_terms(
        row[TOKENIZED_COLUMN],
        min_n=min_ngram,
        max_n=max_ngram,
    )


def load_tokenized_ground_truth() -> pd.DataFrame:
    if not INPUT_PARQUET_PATH.exists():
        raise FileNotFoundError(
            "Tokenized ground truth file not found. Run "
            "News/Build_sentiment_label/Traditional_ML/prepare_ground_truth.py first."
        )

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    required_columns = {TOKENIZED_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input file {INPUT_PARQUET_PATH} is missing columns: {sorted(missing_columns)}"
        )

    out = df.copy()
    out["_document_terms"] = out.apply(build_document_terms, axis=1)
    out = out.loc[out["_document_terms"].map(len).gt(0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid tokenized rows found.")

    print("Input:", INPUT_PARQUET_PATH)
    return out


def build_document_term_counts(df: pd.DataFrame) -> list[Counter[str]]:
    return [Counter(terms) for terms in df["_document_terms"]]


def build_ngram_terms_dataframe(
    term_counts_by_document: list[Counter[str]],
) -> pd.DataFrame:
    total_tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    for term_counts in term_counts_by_document:
        total_tf_counter.update(term_counts)
        df_counter.update(term_counts.keys())

    terms = list(df_counter.keys())
    return pd.DataFrame(
        {
            "term": terms,
            "ngram_n": [term.count(NGRAM_SEPARATOR) + 1 for term in terms],
            "tf": [total_tf_counter[term] for term in terms],
            "df": [df_counter[term] for term in terms],
        }
    )


def fit_tfidf_vocabulary(
    term_counts_by_document: list[Counter[str]],
    total_documents: int,
    stopwords: set[str] | None = None,
) -> pd.DataFrame:
    ngram_terms_df = build_ngram_terms_dataframe(term_counts_by_document)
    min_df_by_ngram = scaled_min_df_by_ngram(
        total_documents=total_documents,
        min_df_ratio_by_ngram=ML_MIN_DF_RATIO_BY_NGRAM,
        floor=ML_MIN_DF_FLOOR,
    )
    candidate_terms_df = choose_ngram_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        max_df_ratio=ML_MAX_DF_RATIO,
        remove_stopwords=REMOVE_STOPWORDS,
        stopwords=stopwords,
    )
    if candidate_terms_df.empty:
        raise ValueError("No terms left after vocabulary filtering.")

    vocabulary_df = candidate_terms_df.rename(columns={"tf": "total_tf"})[
        ["term", "ngram_n", "total_tf", "df", "df_ratio"]
    ].copy()
    vocabulary_df["idf"] = np.log(total_documents / vocabulary_df["df"])
    vocabulary_df = vocabulary_df.sort_values(
        ["ngram_n", "df", "total_tf", "term"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    vocabulary_df.insert(0, "term_id", np.arange(len(vocabulary_df), dtype=int))
    return vocabulary_df


def _document_tfidf_weights(
    term_counts: Counter[str],
    term_to_id: dict[str, int],
    term_to_idf: dict[str, float],
) -> tuple[dict[int, float], int, float]:
    filtered_counts = {
        term: count for term, count in term_counts.items() if term in term_to_id
    }
    document_term_count = sum(filtered_counts.values())
    document_length_norm = (
        1.0 + np.log(document_term_count) if document_term_count >= 1 else 1.0
    )

    weights: dict[int, float] = {}
    for term, term_frequency in filtered_counts.items():
        term_id = int(term_to_id[term])
        tf_log = 1.0 + np.log(term_frequency)
        idf = float(term_to_idf[term])
        weights[term_id] = float((tf_log / document_length_norm) * idf)
    return weights, document_term_count, document_length_norm


def transform_tfidf(
    term_counts_by_document: list[Counter[str]],
    vocabulary_df: pd.DataFrame,
    with_tf_rows: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    term_to_id = dict(zip(vocabulary_df["term"], vocabulary_df["term_id"], strict=False))
    term_to_idf = dict(zip(vocabulary_df["term"], vocabulary_df["idf"], strict=False))

    n_documents = len(term_counts_by_document)
    n_terms = len(vocabulary_df)
    dense = np.zeros((n_documents, n_terms), dtype=np.float32)
    tf_rows: list[dict[str, object]] = []

    id_to_term = dict(
        zip(vocabulary_df["term_id"], vocabulary_df["term"], strict=False)
    )
    for document_id, term_counts in enumerate(term_counts_by_document):
        weights, document_term_count, document_length_norm = _document_tfidf_weights(
            term_counts, term_to_id, term_to_idf
        )
        for term_id, weight in weights.items():
            dense[document_id, term_id] = weight

        if with_tf_rows:
            for term_id in sorted(weights):
                term = id_to_term[term_id]
                term_frequency = int(term_counts[term])
                tf_rows.append(
                    {
                        "document_id": document_id,
                        "term_id": term_id,
                        "term": term,
                        "tf": term_frequency,
                        "document_term_count": int(document_term_count),
                        "tf_log": float(1.0 + np.log(term_frequency)),
                        "document_length_norm": float(document_length_norm),
                        "idf": float(term_to_idf[term]),
                        "tfidf_weight": float(dense[document_id, term_id]),
                    }
                )

    tf_rows_df = pd.DataFrame(tf_rows) if with_tf_rows else None
    return dense, tf_rows_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_tokenized_ground_truth()
    term_counts_by_document = build_document_term_counts(df)
    stopwords = load_stopwords(DEFAULT_STOPWORDS_PATH) if REMOVE_STOPWORDS else set()
    vocabulary_df = fit_tfidf_vocabulary(
        term_counts_by_document,
        total_documents=len(df),
        stopwords=stopwords,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vocabulary_df.to_csv(OUTPUT_VOCAB_PATH, index=False, encoding="utf-8-sig")

    min_df_by_ngram = scaled_min_df_by_ngram(
        total_documents=len(df),
        min_df_ratio_by_ngram=ML_MIN_DF_RATIO_BY_NGRAM,
        floor=ML_MIN_DF_FLOOR,
    )

    print("NOTE: this vocabulary describes a whole-corpus fit; the models re-fit")
    print("      TF-IDF inside each CV fold (see model/common.py).")
    print("TF-IDF formula:")
    print("w_i,j = ((1 + log(tf_i,j)) / (1 + log(a_j))) * log(N / df_i)")
    print("N-gram range:", NGRAM_RANGE)
    print("Min df by n-gram (scaled from ratio):", min_df_by_ngram)
    print("Max df_ratio:", ML_MAX_DF_RATIO)
    print("Remove stopwords:", REMOVE_STOPWORDS)
    print("Stopwords path:", DEFAULT_STOPWORDS_PATH)
    print("Documents:", len(df))
    print("Terms:", len(vocabulary_df))
    print("Output vocabulary:", OUTPUT_VOCAB_PATH)
    print("\nTop terms by df:")
    print(
        vocabulary_df.sort_values(["df", "total_tf"], ascending=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
