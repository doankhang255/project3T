from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import (
    DEFAULT_TOKENIZED_NEWS_PATH,
    build_ngram_terms_with_summary,
)
from News.Build_sentiment_label.Common.ngram_filter import (
    choose_ngram_terms,
    load_sentiment_tokens,
    scaled_min_df_by_ngram,
)
from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
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
POSITIVE_SEED_PATH = RESOURCES_DIR / "positive_word.txt"
NEGATIVE_SEED_PATH = RESOURCES_DIR / "negative_word.txt"
SENTIMENT_WORD_PATHS = [POSITIVE_SEED_PATH, NEGATIVE_SEED_PATH]
LEXICON_MIN_N = 2
LEXICON_MAX_N = 3
LEXICON_MIN_DF_RATIO_BY_NGRAM = {2: 0.0016, 3: 0.0004}
LEXICON_MIN_DF_FLOOR = 20
LEXICON_MAX_DF_RATIO = 0.22
KEEP_LOW_DF_SENTIMENT_TERMS = True


def build_lexicon_candidate_terms(
    path: Path = INPUT_PATH,
    min_df_ratio_by_ngram: dict[int, float] | None = None,
    min_df_floor: int = LEXICON_MIN_DF_FLOOR,
    max_df_ratio: float = LEXICON_MAX_DF_RATIO,
    remove_stopwords: bool = True,
    stopwords_path: Path = STOPWORDS_PATH,
    sentiment_word_path: Path | list[Path] = SENTIMENT_WORD_PATHS,
    total_documents: int | None = None,
) -> dict[str, pd.DataFrame]:
    ngram_terms_df, ngram_summary = build_ngram_terms_with_summary(
        path=path,
        min_n=LEXICON_MIN_N,
        max_n=LEXICON_MAX_N,
    )
    total_documents = total_documents or ngram_summary["total_documents"]
    min_df_by_ngram = scaled_min_df_by_ngram(
        total_documents=total_documents,
        min_df_ratio_by_ngram=min_df_ratio_by_ngram or LEXICON_MIN_DF_RATIO_BY_NGRAM,
        floor=min_df_floor,
    )
    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()
    sentiment_tokens = load_sentiment_tokens(sentiment_word_path)
    result = choose_ngram_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        max_df_ratio=max_df_ratio,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        sentiment_tokens=sentiment_tokens,
        keep_low_df_sentiment_terms=KEEP_LOW_DF_SENTIMENT_TERMS,
        return_groups=True,
    )
    result["total_documents"] = total_documents
    result["min_df_by_ngram"] = min_df_by_ngram
    return result


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
    print("Sentiment word paths:", SENTIMENT_WORD_PATHS)
    print("N-gram range:", f"{LEXICON_MIN_N} to {LEXICON_MAX_N}")
    print("Total documents:", result["total_documents"])
    print("Minimum df ratio by n-gram:", LEXICON_MIN_DF_RATIO_BY_NGRAM)
    print("Minimum df by n-gram (scaled):", result["min_df_by_ngram"])
    print("Keep low-df sentiment terms:", KEEP_LOW_DF_SENTIMENT_TERMS)
    print("Maximum df_ratio:", LEXICON_MAX_DF_RATIO)
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
