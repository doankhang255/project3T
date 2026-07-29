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


INPUT_PATH = DEFAULT_TOKENIZED_NEWS_PATH
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "filtered_ngram_terms.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_News" / "filtered_ngram_terms.csv"

MAX_DF_RATIO = 0.22
MIN_DF_BY_NGRAM = {1: 5000, 2: 200}


def build_min_df_mask(
    term_stats: pd.DataFrame,
    min_df_by_ngram: dict[int, int],
) -> pd.Series:
    min_df_threshold = term_stats["ngram_n"].map(min_df_by_ngram).fillna(0)
    return term_stats["df"].lt(min_df_threshold)


def choose_ngram_terms(
    ngram_terms_df: pd.DataFrame,
    total_documents: int | None = None,
    min_df_by_ngram: dict[int, int] | None = None,
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

    min_df_by_ngram = min_df_by_ngram or MIN_DF_BY_NGRAM
    below_min_df_mask = build_min_df_mask(
        non_stopword_df,
        min_df_by_ngram=min_df_by_ngram,
    )
    min_df_pass_df, below_min_df_df = split_by_mask(
        non_stopword_df,
        below_min_df_mask,
    )
    below_min_df_df["rejection_reason"] = "below_min_df_by_ngram"

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
        "below_min_df_df": below_min_df_df,
        "above_max_df_ratio_df": above_max_df_ratio_df,
        "candidate_terms_df": candidate_terms_df,
    }


def build_filtered_ngram_terms(
    path: Path = INPUT_PATH,
    min_df_by_ngram: dict[int, int] | None = None,
    max_df_ratio: float = MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = DEFAULT_STOPWORDS_PATH,
    total_documents: int | None = None,
) -> dict[str, pd.DataFrame]:
    ngram_terms_df, _ = build_ngram_terms_with_summary(path=path)
    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()
    return choose_ngram_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        max_df_ratio=max_df_ratio,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        return_groups=True,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_filtered_ngram_terms()
    candidate_terms_df = result["candidate_terms_df"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_terms_df.to_parquet(OUTPUT_PATH, index=False)
    candidate_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input n-gram terms:", INPUT_PATH)
    print("Stopwords path:", DEFAULT_STOPWORDS_PATH)
    print("Minimum df by n-gram:", MIN_DF_BY_NGRAM)
    print("Maximum df_ratio:", MAX_DF_RATIO)
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("N-grams removed by stopword boundary:", len(result["stopword_boundary_df"]))
    print("N-grams removed by min_df by n-gram:", len(result["below_min_df_df"]))
    print("N-grams removed by df_ratio > max_df_ratio:", len(result["above_max_df_ratio_df"]))
    print("Filtered n-gram terms:", len(candidate_terms_df))
    print(candidate_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
