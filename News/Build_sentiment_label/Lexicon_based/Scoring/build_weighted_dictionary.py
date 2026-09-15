""" Quy ước trọng số:
- Term có trong seed_round4 nhưng KHÔNG có trong seed_provenance.csv => là
  term gốc từ final_seed (người curate tay, chưa bao giờ là candidate PMI
  nên không có score_centered) => weight = 1.0, coi là đơn vị tham chiếu
  "mạnh nhất", giống cách đếm nhị phân của LIWC/LM gốc.
- Term có trong seed_provenance.csv (được duyệt qua bootstrap, có sẵn
  score_centered từ PMI) => weight = score_centered_at_approval của nó. Các
  term này thường < 1.0 (hiếm khi vượt 1.5), nên đóng góp ít hơn 1 unit so
  với từ seed gốc - hợp lý vì đây là từ suy luận/mở rộng qua PMI, độ tin cậy
  thấp hơn từ đã được con người xác nhận từ đầu.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SEED_SET_PREPARE_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare"
LATEST_SEED_ROUND_DIR = SEED_SET_PREPARE_DIR / "seed_round4"
BOOTSTRAP_DATA_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based" / "data" / "bootstrap"
PROVENANCE_PATH = BOOTSTRAP_DATA_DIR / "seed_provenance.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "weighted_dictionary.csv"
TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"

CATEGORY_NAMES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]

FINAL_SEED_WEIGHT = 1.0


def load_seed_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    items = [item.strip() for item in re.split(r"[,\n]", text)]
    return [item for item in items if item]


def build_single_token_vocab(tokenized_path: Path) -> set[str]:
    """Tập hợp mọi token literal (đã tokenize bởi VNCoreNLP) từng xuất hiện
    trong corpus - dùng để phân biệt 1 chuỗi nối gạch dưới trong seed là
    compound THẬT (VNCoreNLP tự gộp thành 1 token, VD "tăng_trưởng") hay chỉ
    là người viết seed nối tay nhiều từ riêng biệt (VD "tăng_mạnh" - thực ra
    là 2 token "tăng" + "mạnh", không bao giờ xuất hiện gộp)."""
    vocab: set[str] = set()
    df = pd.read_parquet(tokenized_path, columns=["Tokenize_content_sentences"])
    for sentences in df["Tokenize_content_sentences"]:
        for sent in sentences:
            vocab.update(sent)
    return vocab


def resolve_term_and_ngram(term: str, single_token_vocab: set[str]) -> tuple[str, int]:
    """QUAN TRỌNG: score_sentence() (Scoring/score_articles.py) tra dictionary
    bằng candidate = " ".join(tokens[pos:pos+n]) - tức multi-token phải nối
    bằng DẤU CÁCH, không phải gạch dưới. Nếu term giữ nguyên gạch dưới nhưng
    không tồn tại như 1 token literal nào trong corpus (không phải compound
    thật), nó sẽ có ngram_n=1 (đếm nhầm bằng term.count(" ")+1 = 1 vì không
    có dấu cách) và KHÔNG BAO GIỜ khớp được với 2 token rời "tăng"+"mạnh"
    trong corpus - đây là bug đã phát hiện (209/1729 term "chết" hoàn toàn,
    gồm nhiều từ tần suất cao như tăng_mạnh/giảm_mạnh/có_lãi/nợ_xấu). Sửa:
    nếu chuỗi gốc là 1 token thật trong corpus -> giữ nguyên (ngram_n=1); nếu
    không -> tách theo "_" thành các token thật, nối lại bằng dấu cách,
    ngram_n = số token.

    CHỈ áp dụng suy luận này cho term CHƯA có dấu cách (tức term thô từ
    final_seed, nơi gạch dưới là quy ước MƠ HỒ - có thể là compound thật
    hoặc chỉ là nối tay). Term đã có dấu cách (từ round1-4, do
    select_lexicon_candidate_terms.py sinh ra bằng cách nối các TOKEN THẬT
    bằng dấu cách - xem CANDIDATE_NGRAM_RANGE) giữ NGUYÊN, không tách thêm
    theo "_" - nếu không sẽ phá vỡ compound thật nằm bên trong cụm nhiều từ,
    VD "tăng_trưởng doanh_thu" (đúng là 2 token) bị tách nhầm thành 4."""
    if " " in term:
        return term, term.count(" ") + 1
    if term in single_token_vocab:
        return term, 1
    parts = [p for p in term.split("_") if p]
    return " ".join(parts), len(parts)


def load_bootstrap_weight_lookup() -> dict[tuple[str, str], tuple[float, str]]:
    provenance_df = pd.read_csv(PROVENANCE_PATH, encoding="utf-8-sig", dtype=str)
    provenance_df["round"] = provenance_df["round"].astype(int)
    provenance_df["score_centered_at_approval"] = provenance_df["score_centered_at_approval"].astype(float)
    provenance_df = provenance_df.sort_values("round")

    lookup: dict[tuple[str, str], tuple[float, str]] = {}
    for row in provenance_df.itertuples(index=False):
        key = (row.category, row.term)
        if key not in lookup:  # vòng sớm nhất thắng (đã sort theo round)
            lookup[key] = (row.score_centered_at_approval, f"round{row.round}")
    return lookup


def build_weighted_dictionary() -> pd.DataFrame:
    weight_lookup = load_bootstrap_weight_lookup()
    single_token_vocab = build_single_token_vocab(TOKENIZED_CORPUS_PATH)

    rows: list[dict] = []
    for category in CATEGORY_NAMES:
        path = LATEST_SEED_ROUND_DIR / f"{category}_word.txt"
        terms = load_seed_words(path)
        for term in terms:
            key = (category, term)
            if key in weight_lookup:
                weight, source = weight_lookup[key]
            else:
                weight, source = FINAL_SEED_WEIGHT, "final_seed"
            resolved_term, ngram_n = resolve_term_and_ngram(term, single_token_vocab)
            rows.append(
                {
                    "category": category,
                    "term": resolved_term,
                    "ngram_n": ngram_n,
                    "weight": weight,
                    "source": source,
                }
            )

    dictionary_df = pd.DataFrame(rows)

    # Phòng vệ trùng lặp trong chính seed_round4/*.txt (đã xác nhận sạch ở
    # lần kiểm tra trước, nhưng giữ lại để bảo vệ nếu file bị sửa tay sau này).
    before = len(dictionary_df)
    dictionary_df = dictionary_df.drop_duplicates(subset=["category", "term"], keep="first")
    after = len(dictionary_df)
    if before != after:
        print(f"CẢNH BÁO: đã loại {before - after} dòng trùng (category, term).")

    dictionary_df = dictionary_df.sort_values(["category", "weight"], ascending=[True, False]).reset_index(
        drop=True
    )
    return dictionary_df


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary_df = build_weighted_dictionary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Đã xuất {len(dictionary_df)} term vào: {OUTPUT_PATH}")
    print()
    summary = dictionary_df.groupby(["category", "source"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print()
    print("Tổng số term / category:")
    print(dictionary_df.groupby("category").size().to_string())
