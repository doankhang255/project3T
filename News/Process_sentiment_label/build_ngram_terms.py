from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast
import math
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "equity_news_tokenized_vncorenlp.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ngram_terms.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "ngram_terms.csv"
STOPWORDS_PATH = Path(__file__).resolve().parent / "vietnamese-stopwords-dash.txt"

TOKENIZED_COLUMN = "Tokenize_content_sentences"
FALLBACK_TOKENIZED_COLUMN = "Tokenize_content"
NGRAM_SEPARATOR = " "
MAX_DF_RATIO = 0.50


def normalize_token_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        tokens = raw
    elif isinstance(raw, tuple):
        tokens = list(raw)
    elif raw is None:
        return []
    elif isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return []

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text.split()

        if isinstance(parsed, list):
            tokens = parsed
        elif isinstance(parsed, tuple):
            tokens = list(parsed)
        else:
            tokens = str(parsed).split()
    else:
        try:
            if pd.isna(raw):
                return []
        except (TypeError, ValueError):
            pass

        try:
            tokens = list(raw)
        except TypeError:
            return []

    return [str(token).strip() for token in tokens if str(token).strip()]


def normalize_sentence_token_lists(raw: object) -> list[list[str]]:
    if raw is None:
        return []

    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return []

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [normalize_token_list(text)]

        raw = parsed

    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass

    try:
        values = list(raw)
    except TypeError:
        return []

    if not values:
        return []

    first_value = next((value for value in values if value is not None), None)
    if first_value is None:
        return []

    if isinstance(first_value, str):
        return [normalize_token_list(values)]

    try:
        list(first_value)
    except TypeError:
        return [normalize_token_list(values)]

    return [
        tokens
        for tokens in (normalize_token_list(sentence_tokens) for sentence_tokens in values)
        if tokens
    ]


def load_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    with open(path, encoding="utf-8") as file:
        return {line.strip().casefold() for line in file if line.strip()}


def has_stopword_boundary(term: str, stopwords: set[str]) -> bool:
    if not stopwords:
        return False

    tokens = term.split(NGRAM_SEPARATOR)
    if not tokens:
        return False

    return tokens[0].casefold() in stopwords or tokens[-1].casefold() in stopwords


def build_ngrams(
    tokens: list[str],
    min_n: int = 1,
    max_n: int = 3,
    separator: str = NGRAM_SEPARATOR,
) -> list[str]:
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")

    ngrams: list[str] = []
    token_count = len(tokens)
    for n in range(min_n, max_n + 1):
        if token_count < n:
            continue

        for start_index in range(token_count - n + 1):
            ngrams.append(separator.join(tokens[start_index : start_index + n]))

    return ngrams


def build_ngram_tf_df_dataframe(
    tokenized_documents: Iterable[object],
    total_documents: int,
    min_n: int = 1,
    max_n: int = 3,
    max_df_ratio: float = MAX_DF_RATIO,
    stopwords: set[str] | None = None,
    return_filter_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    stopwords = stopwords or set()

    for raw_document_tokens in tokenized_documents:
        document_ngrams: list[str] = []
        for tokens in normalize_sentence_token_lists(raw_document_tokens):
            ngrams = build_ngrams(tokens, min_n=min_n, max_n=max_n)
            if not ngrams:
                continue

            document_ngrams.extend(ngrams)

        if not document_ngrams:
            continue

        tf_counter.update(document_ngrams)
        df_counter.update(set(document_ngrams))

    filter_summary = {
        "unique_ngrams_before_filter": len(df_counter),
        "removed_by_stopword_boundary_terms": 0,
        "removed_by_max_df_ratio_terms": 0,
        "kept_ngram_terms": 0,
    }

    records = []
    for term, df in df_counter.items():
        tf = tf_counter[term]
        if has_stopword_boundary(term, stopwords):
            filter_summary["removed_by_stopword_boundary_terms"] += 1
            continue

        df_ratio = df / total_documents
        if df_ratio > max_df_ratio:
            filter_summary["removed_by_max_df_ratio_terms"] += 1
            continue

        filter_summary["kept_ngram_terms"] += 1
        records.append(
            {
                "term": term,
                "ngram_n": term.count(NGRAM_SEPARATOR) + 1,
                "tf": tf,
                "df": df,
                "df_ratio": df_ratio,
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        empty_df = pd.DataFrame(
            columns=[
                "term",
                "ngram_n",
                "tf",
                "df",
                "df_ratio",
                "avg_tf_per_doc",
                "candidate_score",
            ]
        )
        if return_filter_summary:
            return empty_df, filter_summary

        return empty_df

    out["avg_tf_per_doc"] = out["tf"] / out["df"]
    out["candidate_score"] = out["tf"] * out["df"].apply(
        lambda df: math.log((total_documents + 1) / (df + 1))
    )
    out = out.sort_values(
        by=["candidate_score", "tf", "df", "ngram_n", "term"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    out = out.reset_index(drop=True)
    if return_filter_summary:
        return out, filter_summary

    return out


def build_ngram_terms_with_filter_summary(
    path: Path = INPUT_PATH,
    tokenized_column: str = TOKENIZED_COLUMN,
    min_n: int = 1,
    max_n: int = 3,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = STOPWORDS_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        df = pd.read_parquet(path, columns=[tokenized_column])
    except Exception:
        if tokenized_column != TOKENIZED_COLUMN:
            raise

        tokenized_column = FALLBACK_TOKENIZED_COLUMN
        df = pd.read_parquet(path, columns=[tokenized_column])

    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()
    return build_ngram_tf_df_dataframe(
        tokenized_documents=df[tokenized_column],
        total_documents=len(df),
        min_n=min_n,
        max_n=max_n,
        max_df_ratio=max_df_ratio,
        stopwords=stopwords,
        return_filter_summary=True,
    )


def build_ngram_terms(
    path: Path = INPUT_PATH,
    tokenized_column: str = TOKENIZED_COLUMN,
    min_n: int = 1,
    max_n: int = 3,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = STOPWORDS_PATH,
) -> pd.DataFrame:
    ngram_terms_df, _ = build_ngram_terms_with_filter_summary(
        path=path,
        tokenized_column=tokenized_column,
        min_n=min_n,
        max_n=max_n,
        max_df_ratio=max_df_ratio,
        remove_stopwords=remove_stopwords,
        stopwords_path=stopwords_path,
    )
    return ngram_terms_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ngram_terms_df, filter_summary = build_ngram_terms_with_filter_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ngram_terms_df.to_parquet(OUTPUT_PATH, index=False)
    ngram_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input path:", INPUT_PATH)
    print("Stopwords path:", STOPWORDS_PATH)
    print("Maximum df_ratio:", MAX_DF_RATIO)
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Unique n-grams before filter:", filter_summary["unique_ngrams_before_filter"])
    print(
        "N-grams removed by stopword boundary:",
        filter_summary["removed_by_stopword_boundary_terms"],
    )
    print(
        "N-grams removed by df_ratio > max_df_ratio:",
        filter_summary["removed_by_max_df_ratio_terms"],
    )
    print("N-gram terms:", len(ngram_terms_df))
    print(ngram_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
