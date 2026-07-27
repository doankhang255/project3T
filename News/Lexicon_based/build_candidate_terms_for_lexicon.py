from __future__ import annotations

from pathlib import Path
import sys

from News.Common.candidate_term_ngram import (
    MAX_DF_RATIO,
    MIN_DF_BY_NGRAM,
    SENTIMENT_WORD_PATH,
    STOPWORDS_PATH,
    build_candidate_ngram_terms,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "ngram_terms.parquet"
OUTPUT_PATH = SCRIPT_DIR / "data" / "candidate_ngram_terms.parquet"
OUTPUT_CSV_PATH = SCRIPT_DIR / "data" / "candidate_ngram_terms.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_candidate_ngram_terms(path=INPUT_PATH)
    candidate_terms_df = result["candidate_terms_df"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_terms_df.to_parquet(OUTPUT_PATH, index=False)
    candidate_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input n-gram terms:", INPUT_PATH)
    print("Stopwords path:", STOPWORDS_PATH)
    print("Sentiment word path:", SENTIMENT_WORD_PATH)
    print("Minimum df by n-gram:", MIN_DF_BY_NGRAM)
    print("Maximum df_ratio:", MAX_DF_RATIO)
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print(
        "N-grams removed by stopword boundary:",
        len(result["stopword_boundary_df"]),
    )
    print(
        "N-grams kept below min_df because they contain sentiment tokens:",
        len(result["sentiment_min_df_keep_df"]),
    )
    print(
        "Sentiment unigram terms skipped because n-gram 2 matched first:",
        len(result["sentiment_shadowed_ngram1_df"]),
    )
    print(
        "N-grams removed by min_df by n-gram:",
        len(result["below_min_df_df"]),
    )
    print(
        "N-grams removed by df_ratio > max_df_ratio:",
        len(result["above_max_df_ratio_df"]),
    )
    print("Candidate n-gram terms:", len(candidate_terms_df))
    print(candidate_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
