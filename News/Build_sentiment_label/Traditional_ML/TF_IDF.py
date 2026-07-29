from __future__ import annotations

from ast import literal_eval
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
MIN_DF = 1
MAX_DF_RATIO = 1.0


def parse_serialized_list(value: object) -> object:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return literal_eval(value)
        except (SyntaxError, ValueError):
            return value.split()
    return []


def normalize_sentence_tokens(raw_value: object) -> list[list[str]]:
    parsed_value = parse_serialized_list(raw_value)
    if len(parsed_value) == 0:
        return []

    if all(isinstance(item, str) for item in parsed_value):
        return [[str(item) for item in parsed_value if str(item).strip()]]

    sentences: list[list[str]] = []
    for raw_sentence in parsed_value:
        if not isinstance(raw_sentence, (list, tuple, np.ndarray)):
            continue
        sentence_tokens = [
            str(token).strip()
            for token in raw_sentence
            if str(token).strip()
        ]
        if sentence_tokens:
            sentences.append(sentence_tokens)
    return sentences


def normalize_flat_tokens(raw_value: object) -> list[str]:
    parsed_value = parse_serialized_list(raw_value)
    if not isinstance(parsed_value, (list, tuple, np.ndarray)):
        return []
    return [str(token).strip() for token in parsed_value if str(token).strip()]


def build_ngrams_from_tokens(tokens: list[str], ngram_n: int) -> list[str]:
    if ngram_n <= 0 or len(tokens) < ngram_n:
        return []
    return [
        " ".join(tokens[index : index + ngram_n])
        for index in range(0, len(tokens) - ngram_n + 1)
    ]


def build_document_terms(row: pd.Series) -> list[str]:
    if TOKENIZED_SENTENCES_COLUMN in row.index:
        sentences = normalize_sentence_tokens(row[TOKENIZED_SENTENCES_COLUMN])
    else:
        sentences = []

    if not sentences:
        flat_tokens = normalize_flat_tokens(row[TOKENIZED_COLUMN])
        sentences = [flat_tokens] if flat_tokens else []

    terms: list[str] = []
    min_ngram, max_ngram = NGRAM_RANGE
    for sentence_tokens in sentences:
        for ngram_n in range(min_ngram, max_ngram + 1):
            terms.extend(build_ngrams_from_tokens(sentence_tokens, ngram_n))
    return terms


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


def build_vocabulary(term_counts_by_document: list[Counter[str]]) -> pd.DataFrame:
    total_documents = len(term_counts_by_document)
    total_tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()

    for term_counts in term_counts_by_document:
        total_tf_counter.update(term_counts)
        df_counter.update(term_counts.keys())

    rows = []
    for term, document_frequency in df_counter.items():
        df_ratio = document_frequency / total_documents
        if document_frequency < MIN_DF or df_ratio > MAX_DF_RATIO:
            continue

        rows.append(
            {
                "term": term,
                "ngram_n": term.count(" ") + 1,
                "total_tf": int(total_tf_counter[term]),
                "df": int(document_frequency),
                "df_ratio": df_ratio,
                "idf": float(np.log(total_documents / document_frequency)),
            }
        )

    vocabulary_df = pd.DataFrame(rows)
    if vocabulary_df.empty:
        raise ValueError("No terms left after vocabulary filtering.")

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
    vocabulary_df = build_vocabulary(term_counts_by_document)
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

    print("TF-IDF formula:")
    print("w_i,j = ((1 + log(tf_i,j)) / (1 + log(a_j))) * log(N / df_i)")
    print("N-gram range:", NGRAM_RANGE)
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
