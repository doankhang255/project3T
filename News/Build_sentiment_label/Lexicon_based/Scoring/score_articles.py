"""Áp dụng weighted_dictionary.csv (+ negation_cue_words.txt) lên corpus đã
tokenize, tính điểm 7 category cho từng bài báo.

Công thức (cho từng category, từng bài báo):

    S_cat = ( sum_i  weight(w_i) * sign(w_i) )  /  N_total

- w_i: mỗi lần match 1 term trong dictionary, thuộc category đó.
- weight(w_i): tra từ weighted_dictionary.csv (final_seed = 1.0, từ bootstrap
  = score_centered của nó).
- sign(w_i) = -1 nếu quét NGƯỢC tối đa NEGATION_WINDOW token trước điểm bắt
  đầu của w_i (cùng 1 câu) gặp 1 cue-word phủ định (negation_cue_words.txt)
  TRƯỚC KHI gặp 1 từ nối tương phản (clause_boundary_words.txt, VD "nhưng",
  "tuy_nhiên", "mặc_dù"); ngược lại = +1. Từ nối tương phản là ĐIỂM DỪNG
  CỨNG - nếu gặp nó trước khi gặp từ phủ định thì dừng quét, không cho phủ
  định ở mệnh đề trước lan sang mệnh đề đang xét (VD "mặc_dù lợi_nhuận giảm
  nhưng doanh_thu tăng mạnh" - phủ định nếu có ở mệnh đề 1 không được ảnh
  hưởng "tăng" ở mệnh đề 2). Corpus đã tokenize không giữ dấu câu, nên chỉ
  dừng được ở TỪ nối, không dừng được ở dấu phẩy.
- N_total: tổng số token của bài báo (cột total_tokenizer có sẵn trong
  corpus) - chuẩn hóa để bài dài/ngắn không bị lệch điểm chỉ vì độ dài khác
  nhau.

Giới hạn còn lại:
- Chưa xử lý "từ nhấn mạnh" (degree modifier, VD "rất", "cực_kỳ") - đang là
  sub-project riêng, chưa tích hợp vào đây.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORING_DIR = Path(__file__).resolve().parent
DICTIONARY_PATH = SCORING_DIR / "data" / "weighted_dictionary.csv"
NEGATION_PATH = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "negation_cue_words.txt"
)
CLAUSE_BOUNDARY_PATH = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "clause_boundary_words.txt"
)
TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
OUTPUT_PATH = SCORING_DIR / "data" / "article_scores.parquet"
OUTPUT_CSV_SAMPLE_PATH = SCORING_DIR / "data" / "article_scores_sample.csv"

CATEGORY_NAMES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]

NEGATION_WINDOW_DEFAULT = 4  # số token quét ngược trước mỗi match để tìm cue-word


def load_word_set_from_file(path: Path) -> set[str]:
    """Đọc 1 file danh sách từ dạng comma/newline-separated, bỏ dòng comment
    (bắt đầu bằng '#') - dùng chung cho negation_cue_words.txt và
    clause_boundary_words.txt."""
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    cleaned = "\n".join(lines)
    items = [item.strip() for item in re.split(r"[,\n]", cleaned)]
    return {item for item in items if item}


def is_negated(
    tokens: list[str],
    pos: int,
    negation_words: set[str],
    clause_boundary_words: set[str],
    negation_window: int,
) -> bool:
    """Quét ngược từ token ngay trước vị trí `pos` (match trong dictionary
    bắt đầu tại `pos`), tối đa `negation_window` token. Gặp cue-word phủ
    định trước -> True. Gặp từ nối tương phản (nhưng, tuy_nhiên...) trước khi
    gặp phủ định -> dừng ngay, trả về False (không cho phủ định ở mệnh đề
    trước lan sang mệnh đề đang xét)."""
    window_start = max(0, pos - negation_window)
    for i in range(pos - 1, window_start - 1, -1):
        tok = tokens[i]
        if tok in negation_words:
            return True
        if tok in clause_boundary_words:
            return False
    return False


def load_dictionary_by_ngram(dictionary_df: pd.DataFrame) -> dict[int, dict[str, list[tuple[str, float]]]]:
    """Trả về {ngram_n: {term_string: [(category, weight), ...]}} để tra cứu
    O(1) khi quét qua từng n-gram trong câu.
    """
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for row in dictionary_df.itertuples(index=False):
        by_ngram.setdefault(row.ngram_n, {}).setdefault(row.term, []).append((row.category, row.weight))
    return by_ngram


def score_sentence(
    tokens: list[str],
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]],
    negation_words: set[str],
    clause_boundary_words: set[str],
    max_ngram: int,
    negation_window: int,
    category_sums: dict[str, float],
    category_counts: dict[str, int],
) -> None:
    n_tokens = len(tokens)
    pos = 0
    while pos < n_tokens:
        matched_len = 0
        for n in range(min(max_ngram, n_tokens - pos), 0, -1):
            term_map = by_ngram.get(n)
            if not term_map:
                continue
            candidate = " ".join(tokens[pos : pos + n])
            hits = term_map.get(candidate)
            if not hits:
                continue

            negated = is_negated(tokens, pos, negation_words, clause_boundary_words, negation_window)
            sign = -1.0 if negated else 1.0
            for category, weight in hits:
                category_sums[category] += weight * sign
                category_counts[category] += 1

            matched_len = n
            break 
        pos += matched_len if matched_len else 1


def score_article(
    sentences: np.ndarray,
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]],
    negation_words: set[str],
    clause_boundary_words: set[str],
    max_ngram: int,
    negation_window: int,
) -> tuple[dict[str, float], dict[str, int]]:
    category_sums = {c: 0.0 for c in CATEGORY_NAMES}
    category_counts = {c: 0 for c in CATEGORY_NAMES}
    for sent in sentences:
        tokens = list(sent)
        score_sentence(
            tokens,
            by_ngram,
            negation_words,
            clause_boundary_words,
            max_ngram,
            negation_window,
            category_sums,
            category_counts,
        )
    return category_sums, category_counts


def score_corpus(
    corpus_df: pd.DataFrame,
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]],
    negation_words: set[str],
    clause_boundary_words: set[str],
    max_ngram: int,
    negation_window: int = NEGATION_WINDOW_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Chấm điểm toàn bộ corpus_df (đã load sẵn) với 1 giá trị negation_window
    cụ thể. Tách riêng khỏi việc đọc file để dùng lại corpus_df/dictionary đã
    load, không phải đọc lại mỗi lần đổi negation_window (xem
    compare_negation_windows.py)."""
    raw_sum_records: list[dict] = []
    match_count_records: list[dict] = []
    for idx, row in enumerate(corpus_df.itertuples(index=False)):
        sentences = getattr(row, "Tokenize_content_sentences")
        sums, counts = score_article(
            sentences, by_ngram, negation_words, clause_boundary_words, max_ngram, negation_window
        )
        raw_sum_records.append(sums)
        match_count_records.append(counts)
        if verbose and (idx + 1) % 20000 == 0:
            print(f"    ... đã xử lý {idx + 1}/{len(corpus_df)} bài (negation_window={negation_window})")

    sums_df = pd.DataFrame(raw_sum_records)
    counts_df = pd.DataFrame(match_count_records).add_suffix("_match_count")

    result_df = corpus_df[["link", "publication_date", "domain_norm", "title", "total_tokenizer"]].copy()
    result_df = result_df.reset_index(drop=True)
    for category in CATEGORY_NAMES:
        result_df[f"{category}_score"] = sums_df[category] / result_df["total_tokenizer"].replace(0, np.nan)
    result_df = pd.concat([result_df, counts_df], axis=1)
    result_df["net_sentiment_score"] = result_df["positive_score"] - result_df["negative_score"]
    return result_df


def load_dictionary_and_negation() -> tuple[
    pd.DataFrame, dict[int, dict[str, list[tuple[str, float]]]], set[str], set[str], int
]:
    dictionary_df = pd.read_csv(DICTIONARY_PATH, encoding="utf-8-sig")
    max_ngram = int(dictionary_df["ngram_n"].max())
    by_ngram = load_dictionary_by_ngram(dictionary_df)
    negation_words = load_word_set_from_file(NEGATION_PATH)
    clause_boundary_words = load_word_set_from_file(CLAUSE_BOUNDARY_PATH)
    return dictionary_df, by_ngram, negation_words, clause_boundary_words, max_ngram


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc weighted_dictionary.csv ...")
    dictionary_df, by_ngram, negation_words, clause_boundary_words, max_ngram = load_dictionary_and_negation()
    print(f"  {len(dictionary_df)} term, ngram_n tối đa = {max_ngram}")
    print(f"  {len(negation_words)} cue-word phủ định, {len(clause_boundary_words)} từ nối tương phản")

    print("Đọc corpus đã tokenize (có thể mất vài phút) ...")
    corpus_df = pd.read_parquet(TOKENIZED_CORPUS_PATH)
    print(f"  {len(corpus_df)} bài báo")

    result_df = score_corpus(
        corpus_df, by_ngram, negation_words, clause_boundary_words, max_ngram, NEGATION_WINDOW_DEFAULT
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Đã lưu kết quả đầy đủ vào: {OUTPUT_PATH}")

    sample_pos = result_df.nlargest(15, "net_sentiment_score").assign(sample_group="top_positive")
    sample_neg = result_df.nsmallest(15, "net_sentiment_score").assign(sample_group="top_negative")
    sample_random = result_df.sample(n=min(15, len(result_df)), random_state=42).assign(sample_group="random")
    sample_df = pd.concat([sample_pos, sample_neg, sample_random], ignore_index=True)
    sample_df.to_csv(OUTPUT_CSV_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã lưu mẫu review nhanh (45 bài) vào: {OUTPUT_CSV_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
