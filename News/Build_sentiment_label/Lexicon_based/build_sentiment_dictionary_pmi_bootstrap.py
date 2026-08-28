from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# build_sentiment_dictionary_pmi.py chi con cung cap PMI add-alpha (bien the
# duy nhat con dung) - file nay chi import cac ham/hang so DUNG CHUNG.
from News.Build_sentiment_label.Lexicon_based.build_sentiment_dictionary_pmi import (
    CATEGORY_NAMES,
    FINAL_SEED_DIR,
    LEXICON_DATA_DIR,
    compute_multi_category_pmi,
    load_seed_words,
)


BOOTSTRAP_DATA_DIR = LEXICON_DATA_DIR / "bootstrap"
PROVENANCE_PATH = BOOTSTRAP_DATA_DIR / "seed_provenance.csv"

# Moi vong bootstrap co 1 thu muc seed rieng (seed_round1, seed_round2, ...)
# nam canh final_seed trong Seed_set_Prepare - final_seed luon giu nguyen la
# seed GOC, khong bao gio bi ghi de boi ket qua bootstrap.
SEED_SET_PREPARE_DIR = FINAL_SEED_DIR.parent


def resolve_seed_round_dir(round_number: int) -> Path:
    """Thu muc seed LA KET QUA cua vong `round_number` (seed goc + moi term
    duyet tinh den het vong do)."""
    return SEED_SET_PREPARE_DIR / f"seed_round{round_number}"


def resolve_seed_source_dir(round_number: int) -> Path:
    """Thu muc seed dung LAM DAU VAO khi tinh PMI hoac ingest cho vong
    `round_number`: la ket qua cua vong ngay truoc do (seed_round{N-1}), hoac
    final_seed goc neu day la vong dau tien (round_number <= 1)."""
    if round_number <= 1:
        return FINAL_SEED_DIR
    return resolve_seed_round_dir(round_number - 1)

# Top 3% candidate (theo diem centered) moi category se duoc dua vao file
# review moi vong.
TOP_PERCENTILE_DEFAULT = 0.03

# Cac gia tri duoc coi la "duyet" trong cot approve cua file review (khong
# phan biet hoa/thuong, tu dong strip khoang trang).
APPROVE_TRUE_VALUES = {"1", "x", "yes", "y", "true", "approve", "approved", "ok", "co", "duyet"}


def wrap_terms_preserving_boundaries(terms: list[str], width: int = 79) -> list[str]:
    """Xep cac term (VD "quyet_dinh dinh_chi", nhieu tu cach nhau boi dau
    cach that) thanh cac dong ~width ky tu, kieu nhu textwrap, nhung KHONG
    BAO GIO xuong dong o giua 1 term - chi xuong dong giua 2 term voi nhau.

    Ly do khong dung textwrap.wrap truc tiep tren chuoi da noi dau phay: no
    chi biet ngat o khoang trang, khong biet ranh gioi that su la dau phay,
    nen co the cat doi mot term nhieu tu ngay giua 2 tu cua no. Khi doc lai
    bang load_seed_words (tach theo ca dau phay LAN xuong dong), nua dau va
    nua sau cua term bi cat se tro thanh 2 muc rieng biet - vua mat term
    goc vua sinh ra muc trung lap voi term khac co san.
    """
    lines: list[str] = []
    current = ""
    for term in terms:
        piece = f"{term}, "
        if current and len(current) + len(piece) > width:
            lines.append(current.rstrip())
            current = piece
        else:
            current += piece
    if current:
        lines.append(current.rstrip())
    return lines


def aggregate_category_labels_percentile(
    result: dict,
    categories: list[str] = CATEGORY_NAMES,
    top_percentile: float = TOP_PERCENTILE_DEFAULT,
    min_candidate_unit_st: int | None = None,
) -> pd.DataFrame:
    """Xep hang candidate theo PHAN TRAM (percentile) diem centered, lay dung
    top_percentile% candidate moi category bat ke hinh dang phan phoi diem.

    Dung centering 2 CHIEU (row + column, kieu two-way ANOVA):
    - Tru row_mean (trung binh 7 category CUNG 1 candidate): khu bias
      "candidate hiem bi doi diem deu tren ca 7 category" (xem giai thich
      chi tiet trong build_sentiment_dictionary_pmi.py). Buoc nay BAT BUOC
      phai giu - da kiem chung thuc te: bo buoc nay thi co toi 105 candidate
      bi flag=True o CA 7 category cung luc (dung raw score de xep percentile
      truc tiep), vi row_mean khac nhau theo TUNG candidate nen no thuc su
      doi thu hang ben trong 1 category.
    - Tru them col_mean (trung binh 1 category qua TOAN BO candidate, cong
      lai grand_mean): khu bias "category nay co PMI trung binh cao/thap hon
      hang loat category khac mot cach he thong" (VD weak_modal luon +1.15,
      strong_modal luon -1.33 so voi trung binh 7 category, du category
      khong lien quan gi den candidate cu the). Buoc nay KHONG doi candidate
      nao duoc chon top X% (da kiem chung: 100% trung khop danh sach so voi
      khi chi dung 1 chieu), vi day la 1 hang so cong them GIONG NHAU cho
      moi candidate trong cung 1 category - khong doi thu hang trong noi bo
      category. Chi giup gia tri score_centered so sanh duoc GIUA cac
      category voi nhau (vd danh gia candidate nao "litigious" hon la
      "constraining" mot cach cong bang), thay vi chi so sanh duoc trong
      noi bo 1 category nhu truoc.

    `min_candidate_unit_st`: loc candidate qua hiem O CAP CAU truoc khi xep
    hang (van nen dung, vi ly do khac voi 2 bias tren - candidate hiem cho
    diem tung category kem on dinh hon, xem hoi thoai truoc).
    """
    candidate_terms_df = result["candidate_terms_df"]

    if min_candidate_unit_st is not None:
        keep_mask = (candidate_terms_df["candidate_unit_st"] >= min_candidate_unit_st).to_numpy()
    else:
        keep_mask = np.ones(len(candidate_terms_df), dtype=bool)

    out = candidate_terms_df.loc[keep_mask].reset_index(drop=True)

    # Tinh raw_score cho ca 7 category truoc (can du de tinh row_mean).
    raw_scores: dict[str, np.ndarray] = {}
    for category in categories:
        pmi_by_seed_full, _ = result["category_pmi_by_seed"][category]
        pmi_by_seed = pmi_by_seed_full[keep_mask]
        if pmi_by_seed.shape[1] == 0:
            raw_scores[category] = np.full(len(out), np.nan)
            continue
        with np.errstate(invalid="ignore"):
            raw_scores[category] = np.nanmean(pmi_by_seed, axis=1)

    score_matrix = np.column_stack([raw_scores[c] for c in categories])
    with np.errstate(invalid="ignore"):
        row_mean = np.nanmean(score_matrix, axis=1)
        col_mean = np.nanmean(score_matrix, axis=0)
        grand_mean = np.nanmean(score_matrix)

    cutoff_fraction = 1.0 - top_percentile
    for col_idx, category in enumerate(categories):
        centered = raw_scores[category] - row_mean - col_mean[col_idx] + grand_mean
        out[f"{category}_score_centered"] = centered

        valid = centered[~np.isnan(centered)]
        if len(valid) == 0:
            out[f"{category}_flag"] = False
            out[f"{category}_percentile_cutoff"] = np.nan
            continue

        cutoff_value = np.quantile(valid, cutoff_fraction)
        out[f"{category}_percentile_cutoff"] = cutoff_value
        out[f"{category}_flag"] = np.where(np.isnan(centered), False, centered >= cutoff_value)

    return out


def export_review_batch(
    labeled_df: pd.DataFrame,
    round_number: int,
    categories: list[str] = CATEGORY_NAMES,
    output_dir: Path = BOOTSTRAP_DATA_DIR,
) -> Path:
    """Xuat toan bo candidate duoc chon (flag=True) o MOI category ra 1 file
    CSV de review thu cong. Moi dong la 1 cap (term, category) - 1 term co
    the xuat hien nhieu dong neu duoc chon o nhieu category cung luc (hop le,
    xem hoi thoai truoc ve multi-label).

    Cot "approve" de trong san - nguoi review dien 1 trong cac gia tri
    APPROVE_TRUE_VALUES (VD "x") de duyet, de trong = tu choi/bo qua.
    Cot "notes" de trong, danh cho ghi chu tuy y khi review.
    """
    rows = []
    for category in categories:
        flagged = labeled_df.loc[labeled_df[f"{category}_flag"]]
        for _, row in flagged.iterrows():
            rows.append(
                {
                    "round": round_number,
                    "category": category,
                    "term": row["term"],
                    "ngram_n": row["ngram_n"],
                    "candidate_unit_st": row["candidate_unit_st"],
                    "score_centered": row[f"{category}_score_centered"],
                    "approve": "",
                    "notes": "",
                }
            )

    review_df = pd.DataFrame(rows).sort_values(
        ["category", "score_centered"], ascending=[True, False]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"review_round{round_number}.csv"
    review_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Da xuat {len(review_df)} dong ({len(categories)} danh muc) de review:")
    print("  ->", output_path)
    counts = review_df["category"].value_counts()
    for category in categories:
        print(f"  {category:13s}: {int(counts.get(category, 0))} candidate can review")

    return output_path


def ingest_review_batch(
    review_csv_path: Path,
    round_number: int,
    source_seed_dir: Path | None = None,
    output_seed_dir: Path | None = None,
    provenance_path: Path = PROVENANCE_PATH,
) -> pd.DataFrame:
    """Doc lai file review (da duoc dien cot 'approve'), gop voi seed dau vao
    (source_seed_dir - mac dinh la ket qua vong truoc, hoac final_seed neu la
    vong 1) thanh 1 bo seed MOI, ghi ra output_seed_dir (mac dinh la thu muc
    rieng seed_round{round_number} - KHONG BAO GIO ghi de len final_seed hay
    len source_seed_dir). Ghi ca 7 danh muc du category do co term moi hay
    khong, de moi thu muc seed_round{N} luon la 1 ban chup day du, doc lap,
    khong phu thuoc thu muc vong truoc con ton tai hay khong. Ghi log vao
    provenance de sau nay biet term nao them o vong nao, luc nao.
    """
    if source_seed_dir is None:
        source_seed_dir = resolve_seed_source_dir(round_number)
    if output_seed_dir is None:
        output_seed_dir = resolve_seed_round_dir(round_number)
    output_seed_dir.mkdir(parents=True, exist_ok=True)

    review_df = pd.read_csv(review_csv_path, encoding="utf-8-sig", dtype=str)
    review_df["approve_norm"] = review_df["approve"].fillna("").str.strip().str.lower()
    approved_df = review_df.loc[review_df["approve_norm"].isin(APPROVE_TRUE_VALUES)].copy()
    # Phong ve: 1 term co the vo tinh lap dong trong file review (VD candidate
    # goc bi trung trong candidate_ngram_terms.parquet) - chi giu 1 ban ghi
    # moi cap (category, term) de tranh them trung vao seed / log trung.
    approved_df = approved_df.drop_duplicates(subset=["category", "term"], keep="first")
    approved_by_category = {
        category: group for category, group in approved_df.groupby("category")
    }

    print(f"Seed dau vao (vong truoc): {source_seed_dir}")
    print(f"Seed dau ra (vong {round_number}): {output_seed_dir}")

    provenance_rows = []
    for category in CATEGORY_NAMES:
        source_path = source_seed_dir / f"{category}_word.txt"
        existing_terms = load_seed_words(source_path)
        existing_set = set(existing_terms)

        group = approved_by_category.get(category)
        new_terms = (
            [t for t in group["term"].tolist() if t not in existing_set]
            if group is not None
            else []
        )

        combined_terms = existing_terms + new_terms
        output_path = output_seed_dir / f"{category}_word.txt"
        wrapped_lines = wrap_terms_preserving_boundaries(combined_terms, width=79)
        output_path.write_text("\n".join(wrapped_lines) + "\n", encoding="utf-8")

        if not new_terms:
            print(f"[{category}] khong co term moi, sao chep nguyen seed cu ({len(existing_terms)} term).")
            continue

        print(f"[{category}] them {len(new_terms)} term moi vao seed:")
        print("  ", sorted(new_terms))

        for term in new_terms:
            row = group.loc[group["term"] == term].iloc[0]
            provenance_rows.append(
                {
                    "term": term,
                    "category": category,
                    "round": round_number,
                    "date_added": date.today().isoformat(),
                    "score_centered_at_approval": row["score_centered"],
                }
            )

    provenance_df = pd.DataFrame(provenance_rows)
    if not provenance_df.empty:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        if provenance_path.exists():
            old_provenance_df = pd.read_csv(provenance_path, encoding="utf-8-sig")
            provenance_df = pd.concat([old_provenance_df, provenance_df], ignore_index=True)
        provenance_df.to_csv(provenance_path, index=False, encoding="utf-8-sig")
        print("\nDa cap nhat provenance log:", provenance_path)
        print("Tong so term da them qua tat ca cac vong:", len(provenance_df))

    print(f"\nSeed vong {round_number} da luu day du tai: {output_seed_dir}")
    return provenance_df


def run_export_round(
    round_number: int,
    top_percentile: float = TOP_PERCENTILE_DEFAULT,
    min_candidate_unit_st: int | None = 50,
    tokenized_path: Path | None = None,
    candidate_terms_path: Path | None = None,
    seed_dir: Path | None = None,
    max_df_ratio: float = 0.1,
    smoothing_alpha: float = 1.0,
    export_categories: list[str] | None = None,
) -> Path:
    """Chay 1 vong bootstrap - phan 'xuat file review'. Dung seed la KET QUA
    cua vong truoc (seed_round{round_number-1}, hoac final_seed goc neu day
    la vong 1 - xem resolve_seed_source_dir) de tinh PMI, roi xuat
    top_percentile% candidate moi category ra file review_round{N}.csv.

    `export_categories`: neu truyen vao (VD ["negative", "positive", ...]),
    CHI xuat candidate cua nhung category nay ra file review - dung khi 1 vai
    category da bao hoa (VD strong_modal/weak_modal ra rat it hoac 0 candidate
    o vong truoc) va muon bo qua khong xuat lai o vong nay. Luu y: PMI va
    2-chieu centering VAN duoc tinh tren DU CA 7 category nhu binh thuong
    (khong doi) - chi buoc xuat file la bi loc bot, de khong pha vo co che
    row-centering/col-centering da kiem chung (row_mean/col_mean can tinh
    tren toan bo 7 category moi dung). Mac dinh (None) = xuat het 7 category.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if seed_dir is None:
        seed_dir = resolve_seed_source_dir(round_number)

    tokenized_path = tokenized_path or (
        PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
    )
    candidate_terms_path = candidate_terms_path or (LEXICON_DATA_DIR / "candidate_ngram_terms.parquet")

    print(f"=== Vong bootstrap {round_number}: tinh PMI voi seed hien tai ===")
    print("Seed dir:", seed_dir)
    result = compute_multi_category_pmi(
        tokenized_path=tokenized_path,
        candidate_terms_path=candidate_terms_path,
        seed_dir=seed_dir,
        categories=CATEGORY_NAMES,
        context_window="sentence",
        smoothing_alpha=smoothing_alpha,
        max_df_ratio=max_df_ratio,
    )

    labeled_df = aggregate_category_labels_percentile(
        result,
        top_percentile=top_percentile,
        min_candidate_unit_st=min_candidate_unit_st,
    )

    return export_review_batch(
        labeled_df,
        round_number=round_number,
        categories=export_categories or CATEGORY_NAMES,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 3:
        print("Cach dung:")
        print("  python build_sentiment_dictionary_pmi_bootstrap.py export <round_number> [--skip cat1,cat2]")
        print("  python build_sentiment_dictionary_pmi_bootstrap.py ingest <round_number>")
        print("  (--skip: bo qua khong xuat lai candidate cua nhung category da bao hoa,")
        print("   VD --skip strong_modal,weak_modal)")
        sys.exit(1)

    mode = sys.argv[1]
    round_arg = int(sys.argv[2])

    if mode == "export":
        skip_categories: list[str] = []
        if "--skip" in sys.argv:
            skip_idx = sys.argv.index("--skip")
            skip_categories = [c.strip() for c in sys.argv[skip_idx + 1].split(",") if c.strip()]
            unknown = set(skip_categories) - set(CATEGORY_NAMES)
            if unknown:
                print(f"Category khong hop le trong --skip: {sorted(unknown)}")
                print(f"Cac category hop le: {CATEGORY_NAMES}")
                sys.exit(1)
        export_categories = [c for c in CATEGORY_NAMES if c not in skip_categories] or None
        run_export_round(round_number=round_arg, export_categories=export_categories)
    elif mode == "ingest":
        review_path = BOOTSTRAP_DATA_DIR / f"review_round{round_arg}.csv"
        ingest_review_batch(review_path, round_number=round_arg)
    else:
        print(f"Mode khong hop le: {mode!r} (chi nhan 'export' hoac 'ingest')")
        sys.exit(1)
