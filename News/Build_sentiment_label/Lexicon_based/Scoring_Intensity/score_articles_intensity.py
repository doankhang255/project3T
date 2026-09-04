"""CÁCH 2: giống hệt cơ chế match n-gram + phủ định + ranh giới mệnh đề của
Lexicon_based/Scoring/score_articles.py, nhưng:

1. weight(w_i) = intensity_weight (nội tại, không phụ thuộc corpus - xem
   build_intensity_dictionary.py) THAY VÌ score_centered (PMI).
2. THÊM hệ số nhân từ intensifier_words.txt: nếu token NGAY TRƯỚC vị trí
   match là 1 từ nhấn mạnh (VD "rất", "cực_kỳ"), nhân weight với hệ số của
   nhóm từ đó TRƯỚC KHI cộng vào tổng.

Công thức:
    S_cat = ( sum_i  intensity_weight(w_i) * intensifier_multiplier(w_i) * sign(w_i) )  /  N_total

GIỚI HẠN: các hệ số nhân trong INTENSIFIER_MULTIPLIERS (0.7/1.3/1.4/1.6) là
lựa chọn BAN ĐẦU dựa theo 4 nhóm cường độ đã phân loại trong
Intensifier_words/intensifier_words.txt, CHƯA hiệu chỉnh thực nghiệm (khác
VADER có hệ số -0.74/1.295 đo trên người đánh giá thật) - cần so sánh với
Cách 1 (PMI) qua classify_and_evaluate_intensity.py trước khi tin dùng.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORING_DIR = Path(__file__).resolve().parent
DICTIONARY_PATH = SCORING_DIR / "data" / "intensity_dictionary.csv"
NEGATION_PATH = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "negation_cue_words.txt"
)
CLAUSE_BOUNDARY_PATH = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "clause_boundary_words.txt"
)
TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
OUTPUT_PATH = SCORING_DIR / "data" / "article_scores_intensity.parquet"
OUTPUT_CSV_SAMPLE_PATH = SCORING_DIR / "data" / "article_scores_intensity_sample.csv"

CATEGORY_NAMES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]

NEGATION_WINDOW_DEFAULT = 4

# Nhóm 1 (giảm nhẹ), 2 (cao), 3 (rất cao/tuyệt đối), 4 (thông dụng tài chính)
# - xem Intensifier_words/intensifier_words.txt để biết nguồn/lý do phân nhóm.
INTENSIFIER_MULTIPLIERS: dict[str, float] = {
    "hơi": 0.7, "khá": 0.7, "phần_nào": 0.7, "tương_đối": 0.7, "ít": 0.7,
    "rất": 1.3, "quá": 1.3, "lắm": 1.3, "thật": 1.3, "thật_sự": 1.3,
    "cực_kỳ": 1.6, "vô_cùng": 1.6, "hết_sức": 1.6, "tuyệt_đối": 1.6,
    "hoàn_toàn": 1.6, "đặc_biệt": 1.6,
    "đáng_kể": 1.4, "nghiêm_trọng": 1.4, "mạnh_mẽ": 1.4, "sâu_sắc": 1.4,
    "rõ_rệt": 1.4,
}


def load_word_set_from_file(path: Path) -> set[str]:
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
    window_start = max(0, pos - negation_window)
    for i in range(pos - 1, window_start - 1, -1):
        tok = tokens[i]
        if tok in negation_words:
            return True
        if tok in clause_boundary_words:
            return False
    return False


def intensifier_multiplier(tokens: list[str], pos: int) -> float:
    """Token NGAY TRƯỚC vị trí match (pos-1) - nếu là từ nhấn mạnh thì trả
    về hệ số nhân của nhóm đó, ngược lại 1.0 (không đổi)."""
    if pos == 0:
        return 1.0
    return INTENSIFIER_MULTIPLIERS.get(tokens[pos - 1], 1.0)


def load_dictionary_by_ngram(dictionary_df: pd.DataFrame) -> dict[int, dict[str, list[tuple[str, float]]]]:
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for row in dictionary_df.itertuples(index=False):
        by_ngram.setdefault(row.ngram_n, {}).setdefault(row.term, []).append(
            (row.category, row.intensity_weight)
        )
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
            multiplier = intensifier_multiplier(tokens, pos)
            for category, weight in hits:
                category_sums[category] += weight * multiplier * sign
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
            print(f"    ... đã xử lý {idx + 1}/{len(corpus_df)} bài")

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

    print("Đọc intensity_dictionary.csv ...")
    dictionary_df, by_ngram, negation_words, clause_boundary_words, max_ngram = load_dictionary_and_negation()
    print(f"  {len(dictionary_df)} term, ngram_n tối đa = {max_ngram}")
    print(f"  {len(negation_words)} cue-word phủ định, {len(clause_boundary_words)} từ nối tương phản")
    print(f"  {len(INTENSIFIER_MULTIPLIERS)} từ nhấn mạnh có hệ số nhân")

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
