from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Lexicon_based.build_sentiment_dictionary_pmi import (
    CATEGORY_NAMES,
    FINAL_SEED_DIR,
    LEXICON_DATA_DIR,
    PMI_VARIANT_ADD_ALPHA,
    aggregate_category_labels,
    compute_multi_category_pmi,
)


# Add-alpha (Laplace-style) smoothing: cong pseudo-count alpha vao ca 3 so dem
# (co_df, df_candidate, df_seed) truoc khi tinh ty le trong PMI.
SMOOTHING_ALPHA = 1.0

# Chay tren toan bo 7 danh muc doc lap (khong ghep cap doi lap), moi danh muc
# dung dung seed cua chinh no trong final_seed/.
CATEGORIES = CATEGORY_NAMES

# Corpus cap cau (VNCoreNLP) - khop DEFAULT_TOKENIZED_NEWS_PATH hien tai.
TOKENIZED_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
CANDIDATE_TERMS_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.parquet"
MAX_DF_RATIO = 0.1

# Da kiem chung thuc te: voi add_alpha, so seed match luon gan nhu toi da
# (~90%+ seed cua danh muc) cho MOI candidate, vi add_alpha khong bao gio ra
# NaN - nen min_seed_matches KHONG anh huong ket qua (da thu 1/5/15 ra y het
# nhau). Chi giu 1 gia tri hop ly lam nguong toi thieu, khong can thu nghiem
# them o tham so nay nua.
MIN_SEED_MATCHES = 1

# Nguon nhieu thuc su: candidate hiem O CAP CAU (candidate_unit_df) bi PMI
# add_alpha thoi phong diem DEU tren ca 7 danh muc (da kiem chung: nhom bi gan
# ca 7 co co df trung binh 21.3, sat nguong san 16 cua buoc chon candidate;
# nhom khong co nao ca co df trung binh 175.5). Thu 3 nguong loc candidate qua
# hiem TRUOC KHI tinh z-score: None (khong loc - de doi chieu voi ket qua sai
# truoc do), 50, 100.
MIN_CANDIDATE_UNIT_DF_OPTIONS = [None, 50, 100]

OUTPUT_DIR = LEXICON_DATA_DIR
RESULT_PARQUET_TEMPLATE = "candidate_multicategory_addalpha_mindf{n}.parquet"
RESULT_CSV_TEMPLATE = "candidate_multicategory_addalpha_mindf{n}.csv"
SUMMARY_CSV_PATH = LEXICON_DATA_DIR / "multicategory_addalpha_mindf_summary.csv"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Variant: Add-alpha smoothing (alpha =", SMOOTHING_ALPHA, ")")
    print("Context window: sentence")
    print("Max df_ratio (candidate/seed qua pho bien):", MAX_DF_RATIO)
    print("min_seed_matches (co dinh, da xac nhan khong anh huong add_alpha):", MIN_SEED_MATCHES)
    print("Danh muc:", CATEGORIES)
    print("Input tokenized corpus:", TOKENIZED_PATH)
    print("Candidate terms:", CANDIDATE_TERMS_PATH)
    print("Seed dir:", FINAL_SEED_DIR)
    print()

    # Buoc nang: build ma tran + tinh PMI cho tung cap (candidate, seed) CHI 1
    # LAN cho ca 7 danh muc, KHONG loc theo candidate_unit_df o day. Cac
    # nguong min_candidate_unit_df ben duoi chi loc + tinh lai z-score (nhe),
    # tai su dung dung 1 lan tinh PMI nay.
    result = compute_multi_category_pmi(
        tokenized_path=TOKENIZED_PATH,
        candidate_terms_path=CANDIDATE_TERMS_PATH,
        seed_dir=FINAL_SEED_DIR,
        categories=CATEGORIES,
        context_window="sentence",
        pmi_variant=PMI_VARIANT_ADD_ALPHA,
        smoothing_alpha=SMOOTHING_ALPHA,
        max_df_ratio=MAX_DF_RATIO,
    )

    LEXICON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for min_unit_df in MIN_CANDIDATE_UNIT_DF_OPTIONS:
        labeled_df = aggregate_category_labels(
            result,
            min_seed_matches=MIN_SEED_MATCHES,
            min_candidate_unit_df=min_unit_df,
        )

        label_tag = min_unit_df if min_unit_df is not None else "none"
        parquet_path = OUTPUT_DIR / RESULT_PARQUET_TEMPLATE.format(n=label_tag)
        csv_path = OUTPUT_DIR / RESULT_CSV_TEMPLATE.format(n=label_tag)
        labeled_df.to_parquet(parquet_path, index=False)
        labeled_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        flag_cols = [f"{c}_flag" for c in CATEGORIES]
        n_flags = labeled_df[flag_cols].sum(axis=1)
        all_seven = int((n_flags == len(CATEGORIES)).sum())

        print(f"=== min_candidate_unit_df = {min_unit_df} ===")
        print("Da luu:", parquet_path)
        print(f"  Tong candidate con lai: {len(labeled_df)}")
        print(f"  Candidate bi gan CA {len(CATEGORIES)} co cung luc: {all_seven}")
        row = {
            "min_candidate_unit_df": label_tag,
            "total_candidates": len(labeled_df),
            "flagged_all_categories": all_seven,
        }
        for category in CATEGORIES:
            flagged = int(labeled_df[f"{category}_flag"].sum())
            print(f"  {category:13s}: {flagged:6d} candidate duoc gan co")
            row[f"{category}_flagged"] = flagged
        summary_rows.append(row)
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
    print("Bang tong hop so sanh 3 nguong min_candidate_unit_df:")
    print(summary_df.to_string(index=False))
    print("\nDa luu bang tong hop:", SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()
