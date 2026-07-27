from __future__ import annotations

from pathlib import Path
import sys

from News.Common.build_ngram_terms import (
    INPUT_PATH,
    build_ngram_terms_with_summary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "data" / "ngram_terms.parquet"
OUTPUT_CSV_PATH = SCRIPT_DIR / "data" / "ngram_terms.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ngram_terms_df, summary = build_ngram_terms_with_summary(path=INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ngram_terms_df.to_parquet(OUTPUT_PATH, index=False)
    ngram_terms_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input path:", INPUT_PATH)
    print("Tokenized column:", summary["tokenized_column"])
    print("N-gram range:", f"{summary['min_n']} to {summary['max_n']}")
    print("Output parquet:", OUTPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Total documents:", summary["total_documents"])
    print("Documents with n-grams:", summary["documents_with_ngrams"])
    print("Total n-gram occurrences:", summary["total_ngram_occurrences"])
    print("Unique n-grams:", summary["unique_ngrams"])
    print("N-gram terms:", len(ngram_terms_df))
    print(ngram_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
