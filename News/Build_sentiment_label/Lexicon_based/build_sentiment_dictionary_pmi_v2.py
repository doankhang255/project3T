from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Lexicon_based.build_sentiment_dictionary_pmi import (
    CANDIDATE_TERMS_PATH,
    INPUT_TOKENIZED_PATH,
    LEXICON_DATA_DIR,
    NEGATIVE_SEED_PATH,
    POSITIVE_SEED_PATH,
    build_pmi_dictionary,
)


# v2: loai candidate co df < MIN_CANDIDATE_DF (mac dinh 2, tuc bo het term chi
# xuat hien dung 1 bai) truoc khi tinh PMI va truoc khi chuan hoa z-score, de
# so sanh xem phan bo so_score_z co bot nhieu hon so voi ban goc (v1, khong
# loc df) hay khong. Cac buoc con lai giong het v1 - chi khac dau vao candidate.
MIN_CANDIDATE_DF = 2

OUTPUT_DICTIONARY_PARQUET_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi_v2.parquet"
OUTPUT_DICTIONARY_CSV_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi_v2.csv"
OUTPUT_SEED_RESOLUTION_CSV_PATH = LEXICON_DATA_DIR / "pmi_seed_resolution_v2.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary_df, seed_resolution_df = build_pmi_dictionary(
        tokenized_path=INPUT_TOKENIZED_PATH,
        candidate_terms_path=CANDIDATE_TERMS_PATH,
        positive_seed_path=POSITIVE_SEED_PATH,
        negative_seed_path=NEGATIVE_SEED_PATH,
        min_candidate_df=MIN_CANDIDATE_DF,
    )

    LEXICON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_parquet(OUTPUT_DICTIONARY_PARQUET_PATH, index=False)
    dictionary_df.to_csv(OUTPUT_DICTIONARY_CSV_PATH, index=False, encoding="utf-8-sig")
    seed_resolution_df.to_csv(OUTPUT_SEED_RESOLUTION_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Minimum candidate df (v2):", MIN_CANDIDATE_DF)
    print("Output dictionary parquet:", OUTPUT_DICTIONARY_PARQUET_PATH)
    print("Output dictionary csv:", OUTPUT_DICTIONARY_CSV_PATH)
    print("Output seed resolution csv:", OUTPUT_SEED_RESOLUTION_CSV_PATH)
    print("Candidate terms scored:", len(dictionary_df))
    print("Label counts:")
    print(dictionary_df["sentiment_label"].value_counts(dropna=False).to_string())
    print("Confidence counts:")
    print(dictionary_df["pmi_confidence"].value_counts(dropna=False).to_string())
    print("\nso_score_z describe:")
    print(dictionary_df["so_score_z"].describe().to_string())


if __name__ == "__main__":
    main()
