"""Dò lại `negation_window` cho Cách 2 (Intensity) bằng quy trình TUNE/HOLDOUT
- Cách 2 có constant NEGATION_WINDOW_DEFAULT RIÊNG (score_articles_intensity.py),
  tách biệt với Cách 1, nhưng trước giờ chưa từng được tune/kiểm chứng riêng -
  chỉ copy giá trị =4 từ Cách 1 sang mà chưa test. Xem
  Scoring/tune_negation_window.py (bản Cách 1) để đối chiếu quy trình - 2
  script độc lập, không dùng chung state, chỉ giống cấu trúc.

Quy trình: xem docstring Scoring/tune_negation_window.py (giống hệt, đổi
sang dictionary + score_corpus() của Cách 2).

Output: data/negation_window_tune_holdout.csv
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
from score_articles_intensity import (  # noqa: E402
    load_dictionary_and_negation,
    score_corpus,
)

TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
SPLIT_PATH = LEXICON_DIR / "data" / "ground_truth_tune_holdout_split.csv"
OUTPUT_PATH = SCORING_DIR / "data" / "negation_window_tune_holdout.csv"

CANDIDATE_WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]
LABELS = ["Positive", "Neutral", "Negative"]


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


def score_and_evaluate(
    subset_corpus_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    by_ngram,
    negation_words,
    clause_boundary_words,
    max_ngram,
    negation_window: int,
) -> tuple[float, float]:
    scored_df = score_corpus(
        subset_corpus_df, by_ngram, negation_words, clause_boundary_words, max_ngram, negation_window, verbose=False
    )
    scored_df["source_row_id"] = subset_corpus_df.index
    scored_df["predicted_label"] = scored_df.apply(assign_three_class_label, axis=1)
    merged = gt_df.merge(scored_df[["source_row_id", "predicted_label"]], on="source_row_id", how="left")
    return evaluate(merged["sentiment"], merged["predicted_label"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc intensity_dictionary.csv + negation words ...")
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

    print(f"\n=== BƯỚC 1: grid search {len(CANDIDATE_WINDOWS)} giá trị window TRÊN TUNE ({len(tune_gt)} bài) ===")
    tune_results = []
    for window in CANDIDATE_WINDOWS:
        accuracy, macro_f1 = score_and_evaluate(
            tune_corpus_df, tune_gt, by_ngram, negation_words, clause_boundary_words, max_ngram, window
        )
        tune_results.append({"negation_window": window, "accuracy_tune": accuracy, "macro_f1_tune": macro_f1})
        print(f"  window={window:>2}: accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}")

    tune_results_df = pd.DataFrame(tune_results).sort_values(["accuracy_tune", "macro_f1_tune"], ascending=False)
    best_window = int(tune_results_df.iloc[0]["negation_window"])
    print(f"\nTốt nhất trên Tune: window={best_window} (accuracy_tune={tune_results_df.iloc[0]['accuracy_tune']:.4f})")

    print(f"\n=== BƯỚC 2: chấm Holdout ({len(holdout_gt)} bài) ĐÚNG 1 LẦN, so sánh window={best_window} vs mặc định=4 ===")
    acc_best_holdout, f1_best_holdout = score_and_evaluate(
        holdout_corpus_df, holdout_gt, by_ngram, negation_words, clause_boundary_words, max_ngram, best_window
    )
    acc_default_holdout, f1_default_holdout = score_and_evaluate(
        holdout_corpus_df, holdout_gt, by_ngram, negation_words, clause_boundary_words, max_ngram, 4
    )
    print(f"  window={best_window} (chọn từ Tune) trên Holdout: accuracy={acc_best_holdout:.4f}  macro_f1={f1_best_holdout:.4f}")
    print(f"  window=4    (mặc định hiện tại) trên Holdout: accuracy={acc_default_holdout:.4f}  macro_f1={f1_default_holdout:.4f}")

    tune_results_df["accuracy_holdout_if_chosen"] = np.nan
    tune_results_df.loc[tune_results_df["negation_window"] == best_window, "accuracy_holdout_if_chosen"] = acc_best_holdout
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tune_results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\nĐã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
