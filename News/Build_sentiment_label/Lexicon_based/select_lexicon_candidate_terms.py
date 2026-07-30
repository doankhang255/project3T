from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import (
    DEFAULT_TOKENIZED_NEWS_PATH,
    NGRAM_SEPARATOR,
    build_ngram_terms_with_summary,
)
from News.Build_sentiment_label.Common.ngram_filter import (
    MAX_DF_RATIO,
)
from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    has_stopword_boundary,
    load_stopwords,
)
from News.Build_sentiment_label.Common.tf_df_utils import (
    add_candidate_statistics,
    infer_total_documents,
    split_by_mask,
    validate_ngram_terms,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LEXICON_DATA_DIR = SCRIPT_DIR / "data"
RESOURCES_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Resources"

INPUT_PATH = DEFAULT_TOKENIZED_NEWS_PATH
OUTPUT_NGRAM_TERMS_PATH = LEXICON_DATA_DIR / "ngram_terms.parquet"
OUTPUT_NGRAM_TERMS_CSV_PATH = LEXICON_DATA_DIR / "ngram_terms.csv"
OUTPUT_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.parquet"
OUTPUT_CSV_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.csv"
STOPWORDS_PATH = DEFAULT_STOPWORDS_PATH
SENTIMENT_WORD_PATH = RESOURCES_DIR / "sentiment_word.txt"
LEXICON_MIN_N = 2
LEXICON_MAX_N = 3
LEXICON_MIN_DF_BY_NGRAM = {2: 200, 3: 50}
KEEP_LOW_DF_SENTIMENT_TERMS = False


def load_sentiment_tokens(path: Path = SENTIMENT_WORD_PATH) -> set[str]:
    text = path.read_text(encoding="utf-8")
    tokens: set[str] = set()
    for raw_item in text.replace("\n", ",").split(","):
        item = raw_item.strip().casefold()
        if not item:
            continue

        tokens.add(item)
        tokens.add("_".join(item.split()))

    return tokens


def find_sentiment_tokens(term: str, sentiment_tokens: set[str]) -> set[str]:
    term_text = str(term).casefold()
    return {token for token in sentiment_tokens if token in term_text}


def build_sentiment_priority_mask(
    term_stats: pd.DataFrame,
    sentiment_tokens: set[str] | None = None,
    preferred_ngram_n: int = 2,
    fallback_ngram_n: int = 1,
) -> pd.Series:
    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    matched_tokens = term_stats["term"].apply(
        lambda term: find_sentiment_tokens(term, sentiment_tokens)
    )
    preferred_tokens: set[str] = set()
    for tokens in matched_tokens.loc[term_stats["ngram_n"].eq(preferred_ngram_n)]:
        preferred_tokens.update(tokens)

    def is_priority_match(index: int) -> bool:
        tokens = matched_tokens.loc[index]
        if not tokens:
            return False

        ngram_n = term_stats.at[index, "ngram_n"]
        if ngram_n == preferred_ngram_n:
            return True
        if ngram_n == fallback_ngram_n:
            return bool(tokens.difference(preferred_tokens))

        return True

    return pd.Series(
        (is_priority_match(index) for index in term_stats.index),
        index=term_stats.index,
    )


def build_min_df_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    sentiment_tokens: set[str] | None = None,
) -> pd.Series:
    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    if not KEEP_LOW_DF_SENTIMENT_TERMS:
        return below_min_df_mask

    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return below_min_df_mask

    sentiment_token_mask = build_sentiment_priority_mask(
        term_stats,
        sentiment_tokens=sentiment_tokens,
    )
    return below_min_df_mask & ~sentiment_token_mask


def build_sentiment_min_df_keep_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    sentiment_tokens: set[str] | None = None,
) -> pd.Series:
    if not KEEP_LOW_DF_SENTIMENT_TERMS:
        return pd.Series(False, index=term_stats.index)

    sentiment_tokens = sentiment_tokens or set()
    if not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    sentiment_token_mask = build_sentiment_priority_mask(
        term_stats,
        sentiment_tokens=sentiment_tokens,
    )
    return below_min_df_mask & sentiment_token_mask


def choose_lexicon_candidate_terms(
    ngram_terms_df: pd.DataFrame,
    total_documents: int | None = None,
    min_df_by_ngram: dict[int, int] | None = None,
    sentiment_tokens: set[str] | None = None,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords: set[str] | None = None,
    return_groups: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    validate_ngram_terms(ngram_terms_df)
    total_documents = (
        infer_total_documents(ngram_terms_df)
        if total_documents is None
        else total_documents
    )
    term_stats = add_candidate_statistics(
        term_df=ngram_terms_df,
        total_documents=total_documents,
    )
    term_stats["rejection_reason"] = None

    if remove_stopwords:
        stopwords = stopwords or set()
        stopword_boundary_mask = term_stats["term"].apply(
            lambda term: has_stopword_boundary(
                term,
                stopwords=stopwords,
                separator=NGRAM_SEPARATOR,
            )
        )
    else:
        stopword_boundary_mask = pd.Series(False, index=term_stats.index)

    non_stopword_df, stopword_boundary_df = split_by_mask(
        term_stats,
        stopword_boundary_mask,
    )
    stopword_boundary_df["rejection_reason"] = "stopword_boundary"

    min_df_by_ngram = min_df_by_ngram or LEXICON_MIN_DF_BY_NGRAM
    below_min_df_mask = build_min_df_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
    )
    sentiment_min_df_keep_mask = build_sentiment_min_df_keep_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
    )
    min_df_pass_df, below_min_df_df = split_by_mask(
        non_stopword_df,
        below_min_df_mask,
    )
    below_min_df_df["rejection_reason"] = "below_min_df_by_ngram"
    sentiment_min_df_keep_df = non_stopword_df.loc[
        sentiment_min_df_keep_mask
    ].copy().reset_index(drop=True)

    above_max_df_ratio_mask = min_df_pass_df["df_ratio"].gt(max_df_ratio)
    candidate_terms_df, above_max_df_ratio_df = split_by_mask(
        min_df_pass_df,
        above_max_df_ratio_mask,
    )
    above_max_df_ratio_df["rejection_reason"] = "above_max_df_ratio"

    candidate_terms_df = candidate_terms_df.drop(columns=["rejection_reason"])
    candidate_terms_df = candidate_terms_df.sort_values(
        by=["candidate_score", "tf", "df", "ngram_n", "term"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    if not return_groups:
        return candidate_terms_df

    return {
        "all_ngram_terms_df": term_stats.reset_index(drop=True),
        "stopword_boundary_df": stopword_boundary_df,
        "sentiment_min_df_keep_df": sentiment_min_df_keep_df,
        "below_min_df_df": below_min_df_df,
        "above_max_df_ratio_df": above_max_df_ratio_df,
        "candidate_terms_df": candidate_terms_df,
    }


def build_lexicon_candidate_terms(
    path: Path = INPUT_PATH,
    min_df_by_ngram: dict[int, int] | None = None,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = STOPWORDS_PATH,
    sentiment_word_path: Path = SENTIMENT_WORD_PATH,
    total_documents: int | None = None,
) -> dict[str, pd.DataFrame]:
    ngram_terms_df, _ = build_ngram_terms_with_summary(
        path=path,
        min_n=LEXICON_MIN_N,
        max_n=LEXICON_MAX_N,
    )
    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()
    sentiment_tokens = load_sentiment_tokens(sentiment_word_path)
    return choose_lexicon_candidate_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
        max_df_ratio=max_df_ratio,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        return_groups=True,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_lexicon_candidate_terms()
    candidate_terms_df = result["candidate_terms_df"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result["all_ngram_terms_df"].drop(columns=["rejection_reason"]).to_parquet(
        OUTPUT_NGRAM_TERMS_PATH,
        index=False,
    )
    result["all_ngram_terms_df"].drop(columns=["rejection_reason"]).to_csv(
        OUTPUT_NGRAM_TERMS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    candidate_terms_df.to_parquet(OUTPUT_PATH, index=False)
    candidate_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input n-gram terms:", INPUT_PATH)
    print("Stopwords path:", STOPWORDS_PATH)
    print("Sentiment word path:", SENTIMENT_WORD_PATH)
    print("N-gram range:", f"{LEXICON_MIN_N} to {LEXICON_MAX_N}")
    print("Minimum df by n-gram:", LEXICON_MIN_DF_BY_NGRAM)
    print("Keep low-df sentiment terms:", KEEP_LOW_DF_SENTIMENT_TERMS)
    print("Maximum df_ratio:", MAX_DF_RATIO)
    print("Output n-gram terms parquet:", OUTPUT_NGRAM_TERMS_PATH)
    print("Output n-gram terms csv:", OUTPUT_NGRAM_TERMS_CSV_PATH)
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("N-grams removed by stopword boundary:", len(result["stopword_boundary_df"]))
    print(
        "N-grams kept below min_df because they contain sentiment tokens:",
        len(result["sentiment_min_df_keep_df"]),
    )
    print("N-grams removed by min_df by n-gram:", len(result["below_min_df_df"]))
    print("N-grams removed by df_ratio > max_df_ratio:", len(result["above_max_df_ratio_df"]))
    print("Candidate n-gram terms:", len(candidate_terms_df))
    print(candidate_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
