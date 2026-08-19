from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Lexicon_based.build_sentiment_dictionary_pmi import (
    LEXICON_DATA_DIR,
    PMI_VARIANT_CDS,
    build_pmi_dictionary,
)


# Bien the Context Distribution Smoothing (Levy, Goldberg & Dagan 2015, kieu
# word2vec): lam phang do lech tan suat GIUA CAC SEED trong cung 1 nhom cuc
# bang luy thua beta (< 1) truoc khi dung lam mau so trong PMI - seed hiem
# trong nhom duoc nang ty trong len, seed pho bien trong nhom bi ha xuong
# (xem giai thich chi tiet trong compute_cds_smoothed_seed_df cua
# build_sentiment_dictionary_pmi.py).
CDS_BETA = 0.7

# Context window cap cau: moi cau la 1 don vi dong-xuat-hien, thay vi ca bai -
# doi hoi corpus co cau that (VNCoreNLP). Khop voi DEFAULT_TOKENIZED_NEWS_PATH
# hien tai trong matrix_csr_utils.py (da duoc doi sang file VNCoreNLP) - ghi
# ro lai o day de khong phai doi sang file khac de biet dang dung corpus nao.
TOKENIZED_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)

# select_lexicon_candidate_terms.py da tu dong dung DEFAULT_TOKENIZED_NEWS_PATH
# (VNCoreNLP) sau khi ban cap nhat matrix_csr_utils.py, nen chi can chay lai
# nguyen file do - khong can doi ten output, van la candidate_ngram_terms.parquet
# nhung noi dung se la candidate tu VNCoreNLP thay vi underthesea.
CANDIDATE_TERMS_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.parquet"

# Dung final_seed/ (ban da gop Master Dictionary + seed thu cong, da loai
# seed df=0 - xem Seed_set_Prepare/lexicon_md_pipeline.ipynb Buoc 4) thay vi
# manual_seed/ tho, de co seed set day du va da kiem chung hon.
FINAL_SEED_DIR = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "final_seed"
)
POSITIVE_SEED_PATH = FINAL_SEED_DIR / "positive_word.txt"
NEGATIVE_SEED_PATH = FINAL_SEED_DIR / "negative_word.txt"

# Tran df_ratio (cap cau): candidate/seed xuat hien trong hon 10% so cau bi
# loai khoi buoc tinh PMI - qua pho bien de mang gia tri phan biet sentiment.
MAX_DF_RATIO = 0.1

OUTPUT_DICTIONARY_PARQUET_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi_cds.parquet"
OUTPUT_DICTIONARY_CSV_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi_cds.csv"
OUTPUT_SEED_RESOLUTION_CSV_PATH = LEXICON_DATA_DIR / "pmi_seed_resolution_cds.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary_df, seed_resolution_df = build_pmi_dictionary(
        tokenized_path=TOKENIZED_PATH,
        candidate_terms_path=CANDIDATE_TERMS_PATH,
        positive_seed_path=POSITIVE_SEED_PATH,
        negative_seed_path=NEGATIVE_SEED_PATH,
        context_window="sentence",
        pmi_variant=PMI_VARIANT_CDS,
        cds_beta=CDS_BETA,
        max_df_ratio=MAX_DF_RATIO,
    )

    LEXICON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_parquet(OUTPUT_DICTIONARY_PARQUET_PATH, index=False)
    dictionary_df.to_csv(OUTPUT_DICTIONARY_CSV_PATH, index=False, encoding="utf-8-sig")
    seed_resolution_df.to_csv(OUTPUT_SEED_RESOLUTION_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Variant: Context distribution smoothing (beta =", CDS_BETA, ")")
    print("Context window: sentence")
    print("Max df_ratio:", MAX_DF_RATIO)
    print("Input tokenized corpus:", TOKENIZED_PATH)
    print("Candidate terms:", CANDIDATE_TERMS_PATH)
    print("Positive seed file:", POSITIVE_SEED_PATH)
    print("Negative seed file:", NEGATIVE_SEED_PATH)
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
    print("\nSeed resolution:")
    print(seed_resolution_df.to_string(index=False))


if __name__ == "__main__":
    main()
