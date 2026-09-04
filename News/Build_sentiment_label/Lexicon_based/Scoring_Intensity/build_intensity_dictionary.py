"""CÁCH 2 (khác với Lexicon_based/Scoring/ - dùng PMI đo liên kết với corpus):
gán TRỌNG SỐ NỘI TẠI cho từng từ trong dictionary, kiểu VADER (từ điển có sẵn
điểm cường độ cho từng từ, KHÔNG phụ thuộc corpus) thay vì score_centered
(PMI, đo mức độ đồng-xuất-hiện với seed TRONG corpus).

Vì tiếng Việt chưa có sẵn 1 bộ VADER/AFINN cho tài chính, trọng số được gán
bằng 1 QUY TẮC TƯỜNG MINH (rule-based), có thể kiểm tra/điều chỉnh lại, gồm
2 lớp:

Lớp 1 - "độ trung tâm/mức độ tin cậy" (base_weight, theo nguồn gốc từ):
- final_seed (người curate tay ngay từ đầu, chưa qua PMI): base = 4.0
- round1 (vòng bootstrap sớm nhất, PMI liên kết mạnh nhất với seed gốc): 3.5
- round2: base = 3.0
- round3: base = 2.5
- round4 (vòng muộn nhất, PMI yếu nhất, biên rìa nhất): base = 2.0

Lớp 2 - "độ cực đoan ngữ nghĩa" (marker_adjustment, CỘNG THÊM vào base):
- Chứa 1 từ mang nghĩa TUYỆT ĐỐI/CỰC ĐOAN (VD "tuyệt_đối", "cực_kỳ",
  "chắc_chắn", "nghiêm_cấm", "thảm_họa", "phá_sản"...): +1.0
- Chứa 1 từ mang nghĩa NHẸ/DÈ DẶT (VD "hơi", "khá", "tương_đối", "có_thể"):
  -1.0
- Không khớp marker nào: giữ nguyên base_weight

Kết quả cuối = base_weight + marker_adjustment, CHẶN (clip) trong [1.0, 5.0].

VD đúng ý tưởng ban đầu: "tăng_trưởng" (final_seed, không marker) = 4.0,
cao hơn hẳn 1 từ tìm được ở round4 không có marker gì = 2.0.

GIỚI HẠN: đây là quy tắc TỰ THIẾT KẾ, KHÔNG phải điểm do con người/LLM chấm
riêng từng từ (không khả thi thủ công cho ~1700 từ trong 1 lượt) - cần
review/spot-check trước khi tin tưởng hoàn toàn, và các con số +1.0/-1.0
cũng như base_weight theo round là CHỌN BAN ĐẦU, CHƯA hiệu chỉnh thực
nghiệm (giống tình trạng NEGATION_WINDOW trước khi được compare_*.py kiểm
chứng) - nên chạy classify_and_evaluate_intensity.py để so sánh với PMI
trước khi quyết định dùng chính thức.

Output: data/intensity_dictionary.csv với các cột:
    category, term, ngram_n, intensity_weight, source, matched_markers
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SEED_SET_PREPARE_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare"
LATEST_SEED_ROUND_DIR = SEED_SET_PREPARE_DIR / "seed_round4"
BOOTSTRAP_DATA_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based" / "data" / "bootstrap"
PROVENANCE_PATH = BOOTSTRAP_DATA_DIR / "seed_provenance.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "intensity_dictionary.csv"

CATEGORY_NAMES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]

BASE_WEIGHT_BY_SOURCE = {
    "final_seed": 5.0,
    "round1": 4.0,
    "round2": 3.0,
    "round3": 2.0,
    "round4": 1.0,
}

# Da do 125 to hop qua tune_intensity_coefficients.py tren 152 bai ground
# truth: marker_delta=0.0 cho accuracy trung binh CAO NHAT (0.630 vs ~0.617
# cho moi gia tri >0) - bat ngo, nhung du lieu ro rang. Giu nguyen co che +
# 2 danh sach marker de sau nay co them ground truth thi do lai duoc, nhung
# hien tai KHONG ap dung dieu chinh (delta=0).
MARKER_DELTA = 0.0

EXTREME_MARKERS = {
    "tuyệt_đối", "hoàn_toàn", "tối_đa", "tối_thiểu", "tối_ưu", "chắc_chắn",
    "nghiêm_cấm", "nghiêm_ngặt", "nghiêm_trọng", "không_bao_giờ", "luôn_luôn",
    "luôn_đúng", "duy_nhất", "số_một", "cực_kỳ", "vô_cùng", "hết_sức",
    "thảm_họa", "khủng_hoảng", "phá_sản", "sụp_đổ", "vỡ_nợ", "bắt_buộc",
    "cấm", "tuyệt_vời", "hoàn_hảo", "kỷ_lục", "bùng_nổ", "khẳng_định",
    "dứt_khoát", "kiên_quyết", "quyết_liệt", "vượt_trội", "đột_phá",
    "cao_nhất", "thấp_nhất", "nhất_định", "hiển_nhiên", "đương_nhiên",
    "không_thể_tranh_cãi", "vô_song", "không_nghi_ngờ", "mạnh_mẽ_nhất",
    "tuyệt_nhiên_không", "tuyệt_đối_không", "hoàn_toàn_không",
}

MILD_MARKERS = {
    "hơi", "khá", "phần_nào", "tương_đối", "ít", "sơ_bộ", "tạm", "gần_như",
    "có_thể", "có_lẽ", "dường_như", "đôi_khi", "thỉnh_thoảng", "chưa_chắc",
    "không_hẳn", "hiếm_khi", "khả_dĩ", "biết_đâu", "chưa_hẳn", "có_phần",
    "nhẹ", "có_vẻ_như", "hình_như",
}


def load_seed_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    items = [item.strip() for item in re.split(r"[,\n]", text)]
    return [item for item in items if item]


def load_source_lookup() -> dict[tuple[str, str], str]:
    """Trả về {(category, term): source} - term không có trong provenance
    thì mặc định source='final_seed' (giống logic build_weighted_dictionary.py
    ở Scoring/)."""
    provenance_df = pd.read_csv(PROVENANCE_PATH, encoding="utf-8-sig", dtype=str)
    provenance_df["round"] = provenance_df["round"].astype(int)
    provenance_df = provenance_df.sort_values("round")
    lookup: dict[tuple[str, str], str] = {}
    for row in provenance_df.itertuples(index=False):
        key = (row.category, row.term)
        if key not in lookup:
            lookup[key] = f"round{row.round}"
    return lookup


def compute_intensity_weight(term: str) -> tuple[float, list[str]]:
    """Trả về (marker_adjustment, matched_markers) - quét từng sub-token
    (tách theo dấu cách) của term, tìm marker khớp. adjustment = 0 khi
    MARKER_DELTA = 0.0 (xem giải thích ở khai báo hằng số)."""
    subtokens = term.split(" ")
    matched: list[str] = []
    adjustment = 0.0
    for tok in subtokens:
        if tok in EXTREME_MARKERS:
            matched.append(f"+{tok}")
            adjustment += MARKER_DELTA
        elif tok in MILD_MARKERS:
            matched.append(f"-{tok}")
            adjustment -= MARKER_DELTA
    return adjustment, matched


def build_intensity_dictionary() -> pd.DataFrame:
    source_lookup = load_source_lookup()

    rows: list[dict] = []
    for category in CATEGORY_NAMES:
        path = LATEST_SEED_ROUND_DIR / f"{category}_word.txt"
        terms = load_seed_words(path)
        for term in terms:
            source = source_lookup.get((category, term), "final_seed")
            base_weight = BASE_WEIGHT_BY_SOURCE[source]
            adjustment, matched = compute_intensity_weight(term)
            final_weight = max(1.0, min(5.0, base_weight + adjustment))
            rows.append(
                {
                    "category": category,
                    "term": term,
                    "ngram_n": term.count(" ") + 1,
                    "intensity_weight": final_weight,
                    "source": source,
                    "matched_markers": ";".join(matched),
                }
            )

    dictionary_df = pd.DataFrame(rows)
    dictionary_df = dictionary_df.sort_values(
        ["category", "intensity_weight"], ascending=[True, False]
    ).reset_index(drop=True)
    return dictionary_df


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary_df = build_intensity_dictionary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Đã xuất {len(dictionary_df)} term vào: {OUTPUT_PATH}")
    print()
    print("Phân phối intensity_weight / category:")
    print(dictionary_df.groupby("category")["intensity_weight"].describe().to_string())
    print()
    print("Số term có marker điều chỉnh (khác 0):", (dictionary_df["matched_markers"] != "").sum())
    print()
    print("10 term intensity_weight cao nhất mỗi category (mẫu để spot-check):")
    for category in CATEGORY_NAMES:
        sub = dictionary_df.loc[dictionary_df["category"] == category].head(10)
        print(f"\n--- {category} ---")
        print(sub[["term", "intensity_weight", "source", "matched_markers"]].to_string(index=False))
