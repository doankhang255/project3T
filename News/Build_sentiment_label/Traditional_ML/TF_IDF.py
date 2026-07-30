from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
DATA_DIR = SCRIPT_DIR / "data"

INPUT_PARQUET_PATH = DATA_DIR / "ground_truth_labeled_tokenized.parquet"

OUTPUT_CSR_NPZ_PATH = DATA_DIR / "tfidf_matrix_csr.npz"
OUTPUT_DOCUMENT_INDEX_PATH = DATA_DIR / "tfidf_document_index.csv"
OUTPUT_VOCAB_PATH = DATA_DIR / "tfidf_vocabulary.csv"
OUTPUT_TERM_STATS_PATH = DATA_DIR / "tfidf_term_statistics.csv"
OUTPUT_DOCUMENT_TERM_TF_PATH = DATA_DIR / "tfidf_document_term_tf.csv"

ID_COLUMN = "id"
TITLE_COLUMN = "title"
LABEL_COLUMN = "sentiment"
TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"

NGRAM_RANGE = (1, 3)
REMOVE_STOPWORDS = True

# Ratios instead of fixed counts: the ground-truth set keeps growing as more
# rows get labeled in Label Studio, so a hardcoded min_df tuned for today's
# row count would silently stop filtering anything once the corpus outgrows
# it (this is exactly how the project ended up with hundreds of thousands of
# near-useless candidate terms before). MIN_DF_FLOOR still guarantees a term
# must appear in at least 2 documents no matter how small the corpus is.
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
    if INPUT_PARQUET_PATH.exists():
        df = pd.read_parquet(INPUT_PARQUET_PATH)
        input_path = INPUT_PARQUET_PATH
    else:
        raise FileNotFoundError(
            "Tokenized ground truth file not found. Run "
            "News/Build_sentiment_label/Traditional_ML/tokenize_underthesea.py first."
        )

    required_columns = {TOKENIZED_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input file {input_path} is missing columns: {sorted(missing_columns)}"
        )

    out = df.copy()
    out["_document_terms"] = out.apply(build_document_terms, axis=1)
    out = out.loc[out["_document_terms"].map(len).gt(0)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid tokenized rows found.")

    print("Input:", input_path)
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


def build_vocabulary(
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


def build_tfidf_csr_arrays(
    term_counts_by_document: list[Counter[str]],
    vocabulary_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], pd.DataFrame]:
    term_to_id = dict(zip(vocabulary_df["term"], vocabulary_df["term_id"], strict=False))
    term_to_idf = dict(zip(vocabulary_df["term"], vocabulary_df["idf"], strict=False))

    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    tf_rows = []

    for document_id, term_counts in enumerate(term_counts_by_document):
        filtered_counts = {
            term: count
            for term, count in term_counts.items()
            if term in term_to_id
        }
        document_term_count = sum(filtered_counts.values())
        document_length_norm = (
            1.0 + np.log(document_term_count)
            if document_term_count >= 1
            else 1.0
        )

        for term, term_frequency in sorted(
            filtered_counts.items(),
            key=lambda item: term_to_id[item[0]],
        ):
            term_id = int(term_to_id[term])
            tf_log = 1.0 + np.log(term_frequency)
            idf = float(term_to_idf[term])
            weight = float((tf_log / document_length_norm) * idf)

            indices.append(term_id)
            data.append(weight)
            tf_rows.append(
                {
                    "document_id": document_id,
                    "term_id": term_id,
                    "term": term,
                    "tf": int(term_frequency),
                    "document_term_count": int(document_term_count),
                    "tf_log": float(tf_log),
                    "document_length_norm": float(document_length_norm),
                    "idf": idf,
                    "tfidf_weight": weight,
                }
            )

        indptr.append(len(data))

    shape = (len(term_counts_by_document), len(vocabulary_df))
    return (
        np.asarray(data, dtype=np.float64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(indptr, dtype=np.int64),
        shape,
        pd.DataFrame(tf_rows),
    )


def build_document_index(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in [ID_COLUMN, TITLE_COLUMN, LABEL_COLUMN]
        if column in df.columns
    ]
    document_index = df[columns].copy()
    document_index.insert(0, "document_id", np.arange(len(df), dtype=int))
    document_index["document_ngram_count"] = df["_document_terms"].map(len)
    return document_index


def save_csr_npz(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    path: Path = OUTPUT_CSR_NPZ_PATH,
) -> None:
    np.savez_compressed(
        path,
        data=data,
        indices=indices,
        indptr=indptr,
        shape=np.asarray(shape, dtype=np.int64),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_tokenized_ground_truth()
    term_counts_by_document = build_document_term_counts(df)
    stopwords = load_stopwords(DEFAULT_STOPWORDS_PATH) if REMOVE_STOPWORDS else set()
    vocabulary_df = build_vocabulary(
        term_counts_by_document,
        total_documents=len(df),
        stopwords=stopwords,
    )
    data, indices, indptr, shape, document_term_tf_df = build_tfidf_csr_arrays(
        term_counts_by_document,
        vocabulary_df,
    )
    document_index_df = build_document_index(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_csr_npz(data, indices, indptr, shape)
    document_index_df.to_csv(
        OUTPUT_DOCUMENT_INDEX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    vocabulary_df.to_csv(OUTPUT_VOCAB_PATH, index=False, encoding="utf-8-sig")
    vocabulary_df.to_csv(OUTPUT_TERM_STATS_PATH, index=False, encoding="utf-8-sig")
    document_term_tf_df.to_csv(
        OUTPUT_DOCUMENT_TERM_TF_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    min_df_by_ngram = scaled_min_df_by_ngram(
        total_documents=len(df),
        min_df_ratio_by_ngram=ML_MIN_DF_RATIO_BY_NGRAM,
        floor=ML_MIN_DF_FLOOR,
    )

    print("TF-IDF formula:")
    print("w_i,j = ((1 + log(tf_i,j)) / (1 + log(a_j))) * log(N / df_i)")
    print("N-gram range:", NGRAM_RANGE)
    print("Min df by n-gram (scaled from ratio):", min_df_by_ngram)
    print("Max df_ratio:", ML_MAX_DF_RATIO)
    print("Remove stopwords:", REMOVE_STOPWORDS)
    print("Stopwords path:", DEFAULT_STOPWORDS_PATH)
    print("Documents:", shape[0])
    print("Terms:", shape[1])
    print("Non-zero matrix values:", len(data))
    print("Output CSR npz:", OUTPUT_CSR_NPZ_PATH)
    print("Output document index:", OUTPUT_DOCUMENT_INDEX_PATH)
    print("Output vocabulary:", OUTPUT_VOCAB_PATH)
    print("Output term statistics:", OUTPUT_TERM_STATS_PATH)
    print("Output document-term tf:", OUTPUT_DOCUMENT_TERM_TF_PATH)
    print("\nTop terms by df:")
    print(
        vocabulary_df.sort_values(["df", "total_tf"], ascending=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
