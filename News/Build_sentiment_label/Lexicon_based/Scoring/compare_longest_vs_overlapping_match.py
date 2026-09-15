"""So sánh lại "greedy longest-match" (đang dùng, KHÔNG chồng lấn) với
"khớp chồng lấn" (cách cũ trước khi đổi, xem LEXICON_SUMMARY.qmd mục 3) -
từng thấy 63,8% (chồng lấn) -> 61,2% (longest-match) trên 152 bài, nhưng
"chưa rõ đây là nhiễu do mẫu nhỏ hay longest-match" (MENTOR_FEEDBACK_PLAN.md
mục A). Giờ test lại với dictionary đã sửa bug ngram_n + ground truth 599
bài (thay vì 152), dùng đúng quy trình Tune/Holdout.

2 cách khớp:
    - Longest-match (đang dùng, score_articles.py::score_sentence): tại mỗi
      vị trí, chỉ lấy n-gram DÀI NHẤT khớp được, rồi nhảy qua hết độ dài đó
      (không chồng lấn). VD "tạm_ngừng hoạt_động" (nếu có trong dict) sẽ che
      mất "hoạt_động" đứng một mình ở cùng vị trí.
    - Overlapping-match (cách cũ): tại MỌI vị trí, đếm TẤT CẢ n-gram khớp
      được (1..max_ngram), không bỏ qua n-gram ngắn hơn dù bị 1 n-gram dài
      hơn "chứa" nó. VD vừa đếm "hoạt_động" (n=1) vừa đếm "tạm_ngừng
      hoạt_động" (n=2) trong cùng câu.

Quy trình: chấm cả Tune (419 bài) VÀ Holdout (180 bài) bằng CẢ 2 cách khớp,
báo cáo cả 2 tập (chỉ 1 phép so sánh nhị phân, không phải grid-search nhiều
lựa chọn, nên rủi ro overfit thấp hơn - nhưng vẫn theo đúng tinh thần không
chỉ tin số trên Tune).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
LEXICON_DIR = SCORING_DIR.parent
PROJECT_ROOT = SCORING_DIR.parents[3]

sys.path.insert(0, str(SCORING_DIR))
from score_articles import (  # noqa: E402
    CATEGORY_NAMES,
    is_negated,
    load_dictionary_and_negation,
    score_corpus,
)

TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
SPLIT_PATH = LEXICON_DIR / "data" / "ground_truth_tune_holdout_split.csv"
OUTPUT_PATH = SCORING_DIR / "data" / "longest_vs_overlapping_match.csv"

NEGATION_WINDOW = 4  # đã xác nhận qua tune/holdout - xem tune_negation_window.py
LABELS = ["Positive", "Neutral", "Negative"]


def score_sentence_overlapping(
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
    for pos in range(n_tokens):
        for n in range(1, min(max_ngram, n_tokens - pos) + 1):
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


def score_corpus_overlapping(
    corpus_df: pd.DataFrame,
    by_ngram,
    negation_words,
    clause_boundary_words,
    max_ngram,
    negation_window: int,
) -> pd.DataFrame:
    raw_sum_records: list[dict] = []
    for row in corpus_df.itertuples(index=False):
        sentences = getattr(row, "Tokenize_content_sentences")
        category_sums = {c: 0.0 for c in CATEGORY_NAMES}
        category_counts = {c: 0 for c in CATEGORY_NAMES}
        for sent in sentences:
            score_sentence_overlapping(
                list(sent), by_ngram, negation_words, clause_boundary_words, max_ngram, negation_window,
                category_sums, category_counts,
            )
        raw_sum_records.append(category_sums)

    sums_df = pd.DataFrame(raw_sum_records)
    result_df = corpus_df[["total_tokenizer"]].copy().reset_index(drop=True)
    for category in CATEGORY_NAMES:
        result_df[f"{category}_score"] = sums_df[category] / result_df["total_tokenizer"].replace(0, np.nan)
    result_df["net_sentiment_score"] = result_df["positive_score"] - result_df["negative_score"]
    return result_df


def assign_three_class_label(row: pd.Series) -> str:
    if row["positive_score"] > row["negative_score"]:
        return "Positive"
    if row["positive_score"] < row["negative_score"]:
        return "Negative"
    return "Neutral"


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    accuracy = float((y_true.values == y_pred.values).mean())
    f1_scores = []
    for label in LABELS:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return accuracy, sum(f1_scores) / len(f1_scores)


def score_and_evaluate(strategy: str, subset_corpus_df, gt_df, by_ngram, negation_words, clause_boundary_words, max_ngram) -> tuple[float, float]:
    if strategy == "longest":
        scored_df = score_corpus(
            subset_corpus_df, by_ngram, negation_words, clause_boundary_words, max_ngram, NEGATION_WINDOW, verbose=False
        )
    else:
        scored_df = score_corpus_overlapping(
            subset_corpus_df, by_ngram, negation_words, clause_boundary_words, max_ngram, NEGATION_WINDOW
        )
    scored_df["source_row_id"] = subset_corpus_df.index
    scored_df["predicted_label"] = scored_df.apply(assign_three_class_label, axis=1)
    merged = gt_df.merge(scored_df[["source_row_id", "predicted_label"]], on="source_row_id", how="left")
    return evaluate(merged["sentiment"], merged["predicted_label"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc dictionary + negation words ...")
    _, by_ngram, negation_words, clause_boundary_words, max_ngram = load_dictionary_and_negation()

    print("Đọc split Tune/Holdout ...")
    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    tune_gt = split_df.loc[split_df["split"] == "tune", ["source_row_id", "sentiment"]].reset_index(drop=True)
    holdout_gt = split_df.loc[split_df["split"] == "holdout", ["source_row_id", "sentiment"]].reset_index(drop=True)
    print(f"  Tune: {len(tune_gt)} bài | Holdout: {len(holdout_gt)} bài")

    print("Đọc corpus, trích riêng các bài trong Tune + Holdout ...")
    all_row_ids = pd.concat([tune_gt["source_row_id"], holdout_gt["source_row_id"]]).astype(int).tolist()
    full_corpus_df = pd.read_parquet(
        TOKENIZED_CORPUS_PATH, columns=["Tokenize_content_sentences", "total_tokenizer", "link", "publication_date", "domain_norm", "title"]
    )
    subset_df = full_corpus_df.loc[all_row_ids]
    del full_corpus_df
    tune_corpus_df = subset_df.loc[tune_gt["source_row_id"].astype(int).tolist()]
    holdout_corpus_df = subset_df.loc[holdout_gt["source_row_id"].astype(int).tolist()]

    results = []
    for strategy, label in [("longest", "Longest-match (đang dùng)"), ("overlapping", "Overlapping-match (cách cũ)")]:
        acc_tune, f1_tune = score_and_evaluate(strategy, tune_corpus_df, tune_gt, by_ngram, negation_words, clause_boundary_words, max_ngram)
        acc_holdout, f1_holdout = score_and_evaluate(strategy, holdout_corpus_df, holdout_gt, by_ngram, negation_words, clause_boundary_words, max_ngram)
        print(f"\n=== {label} ===")
        print(f"  Tune    (419 bài): accuracy={acc_tune:.4f}  macro_f1={f1_tune:.4f}")
        print(f"  Holdout (180 bài): accuracy={acc_holdout:.4f}  macro_f1={f1_holdout:.4f}")
        results.append(
            {
                "strategy": strategy,
                "accuracy_tune": acc_tune,
                "macro_f1_tune": f1_tune,
                "accuracy_holdout": acc_holdout,
                "macro_f1_holdout": f1_holdout,
            }
        )

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\nĐã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
