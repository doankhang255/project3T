from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import NGRAM_SEPARATOR
from News.Build_sentiment_label.Common.stopword_utils import has_stopword_boundary
from News.Build_sentiment_label.Common.tf_df_utils import (
    add_candidate_statistics,
    infer_total_documents,
    split_by_mask,
    validate_ngram_terms,
)


def scaled_min_df_by_ngram(
    total_documents: int,
    min_df_ratio_by_ngram: dict[int, float],
    floor: int = 2,
) -> dict[int, int]:
    return {
        ngram_n: max(floor, math.ceil(total_documents * ratio))
        for ngram_n, ratio in min_df_ratio_by_ngram.items()
    }


def load_sentiment_tokens(path: Path) -> set[str]:
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
    keep_low_df_sentiment_terms: bool = False,
) -> pd.Series:
    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    if not keep_low_df_sentiment_terms or not sentiment_tokens:
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
    keep_low_df_sentiment_terms: bool = False,
) -> pd.Series:
    if not keep_low_df_sentiment_terms or not sentiment_tokens:
        return pd.Series(False, index=term_stats.index)

    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    below_min_df_mask = term_stats["df"].lt(min_df_threshold)
    sentiment_token_mask = build_sentiment_priority_mask(
        term_stats,
        sentiment_tokens=sentiment_tokens,
    )
    return below_min_df_mask & sentiment_token_mask


def choose_ngram_terms(
    ngram_terms_df: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
    max_df_ratio: float,
    total_documents: int | None = None,
    remove_stopwords: bool = True,
    stopwords: set[str] | None = None,
    sentiment_tokens: set[str] | None = None,
    keep_low_df_sentiment_terms: bool = False,
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

    below_min_df_mask = build_min_df_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
        keep_low_df_sentiment_terms=keep_low_df_sentiment_terms,
    )
    sentiment_min_df_keep_mask = build_sentiment_min_df_keep_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
        sentiment_tokens=sentiment_tokens,
        keep_low_df_sentiment_terms=keep_low_df_sentiment_terms,
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
