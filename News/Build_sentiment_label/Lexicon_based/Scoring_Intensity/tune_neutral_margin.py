"""Bản Cách 2 (Intensity) của Scoring/tune_neutral_margin.py - CHƯA từng thử
margin riêng cho Cách 2 trước đây (luôn dùng chung luật "Cách 3" không
margin với Cách 1). Cùng công thức margin + cùng quy trình TUNE/HOLDOUT, chỉ
đổi nguồn điểm số sang article_scores_intensity.parquet.

Quy tắc:
    diff = positive_score - negative_score
    |diff| <= margin   -> Neutral
    diff > margin      -> Positive
    diff < -margin     -> Negative

    1. Đọc split cố định từ Lexicon_based/data/ground_truth_tune_holdout_split.csv
       (419 bài Tune / 180 bài Holdout).
    2. Quét 1 dải margin CHỈ trên Tune, chọn margin cho accuracy cao nhất.
    3. Chấm lại Holdout ĐÚNG 1 LẦN với margin đã chọn, so với margin=0 (Cách
       3 gốc, không đệm) cũng đo trên Holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
LEXICON_DIR = SCORING_DIR.parent
ARTICLE_SCORES_PATH = SCORING_DIR / "data" / "article_scores_intensity.parquet"
SPLIT_PATH = LEXICON_DIR / "data" / "ground_truth_tune_holdout_split.csv"
OUTPUT_PATH = SCORING_DIR / "data" / "neutral_margin_tuning.csv"

LABELS = ["Positive", "Neutral", "Negative"]


def label_with_margin(diff: float, margin: float) -> str:
    if diff > margin:
        return "Positive"
    if diff < -margin:
        return "Negative"
    return "Neutral"


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    f1_scores = []
    for label in LABELS:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def load_diff_for(row_ids: list[int], scores_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    subset_gt = gt_df.loc[gt_df["source_row_id"].isin(row_ids)]
    merged = subset_gt.merge(
        scores_df[["source_row_id", "positive_score", "negative_score"]], on="source_row_id", how="left"
    )
    merged = merged.dropna(subset=["positive_score", "negative_score"])
    merged["diff"] = merged["positive_score"] - merged["negative_score"]
    return merged


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc article_scores_intensity.parquet (Cách 2, đầy đủ) + split Tune/Holdout ...")
    scores_df = pd.read_parquet(ARTICLE_SCORES_PATH, columns=["positive_score", "negative_score"]).reset_index(drop=True)
    scores_df["source_row_id"] = scores_df.index

    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    tune_ids = split_df.loc[split_df["split"] == "tune", "source_row_id"].astype(int).tolist()
    holdout_ids = split_df.loc[split_df["split"] == "holdout", "source_row_id"].astype(int).tolist()

    tune_merged = load_diff_for(tune_ids, scores_df, split_df)
    holdout_merged = load_diff_for(holdout_ids, scores_df, split_df)
    print(f"  Tune: {len(tune_merged)} bài | Holdout: {len(holdout_merged)} bài")
    print(f"  Phân phối |diff| trên Tune: min={tune_merged['diff'].abs().min():.5f}, "
          f"median={tune_merged['diff'].abs().median():.5f}, max={tune_merged['diff'].abs().max():.5f}")
    print()

    print(f"=== BƯỚC 1: quét margin TRÊN TUNE ({len(tune_merged)} bài) ===")
    max_margin = tune_merged["diff"].abs().max()
    margins = np.linspace(0, max_margin, 60)

    results = []
    for margin in margins:
        y_pred = tune_merged["diff"].apply(lambda d: label_with_margin(d, margin))
        accuracy = (tune_merged["sentiment"] == y_pred).mean()
        f1 = macro_f1(tune_merged["sentiment"], y_pred)
        results.append({"margin": margin, "accuracy_tune": accuracy, "macro_f1_tune": f1})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    best_acc_row = results_df.loc[results_df["accuracy_tune"].idxmax()]
    print(f"  Margin tối ưu trên Tune (theo accuracy): margin={best_acc_row['margin']:.5f} "
          f"-> accuracy_tune={best_acc_row['accuracy_tune']:.4f}, macro_f1_tune={best_acc_row['macro_f1_tune']:.4f}")

    print("\n  Độ ổn định quanh điểm tối ưu - 5 margin lân cận mỗi phía:")
    best_idx = results_df["accuracy_tune"].idxmax()
    lo = max(0, best_idx - 5)
    hi = min(len(results_df), best_idx + 6)
    print(results_df.iloc[lo:hi].to_string(index=False))

    best_margin = float(best_acc_row["margin"])
    print(f"\n=== BƯỚC 2: chấm Holdout ({len(holdout_merged)} bài) ĐÚNG 1 LẦN, so sánh margin={best_margin:.5f} vs margin=0 (Cách 3 gốc) ===")
    y_pred_best_holdout = holdout_merged["diff"].apply(lambda d: label_with_margin(d, best_margin))
    acc_best_holdout = (holdout_merged["sentiment"] == y_pred_best_holdout).mean()
    f1_best_holdout = macro_f1(holdout_merged["sentiment"], y_pred_best_holdout)

    y_pred_zero_holdout = holdout_merged["diff"].apply(lambda d: label_with_margin(d, 0.0))
    acc_zero_holdout = (holdout_merged["sentiment"] == y_pred_zero_holdout).mean()
    f1_zero_holdout = macro_f1(holdout_merged["sentiment"], y_pred_zero_holdout)

    print(f"  margin={best_margin:.5f} (chọn từ Tune) trên Holdout: accuracy={acc_best_holdout:.4f}  macro_f1={f1_best_holdout:.4f}")
    print(f"  margin=0 (Cách 3 gốc, hiện đang dùng) trên Holdout:   accuracy={acc_zero_holdout:.4f}  macro_f1={f1_zero_holdout:.4f}")

    print(f"\nConfusion matrix trên Holdout tại margin={best_margin:.5f}:")
    confusion = pd.crosstab(
        holdout_merged["sentiment"], y_pred_best_holdout, rownames=["ground_truth"], colnames=["predicted"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    print(confusion.to_string())

    print(f"\nĐã lưu dải margin đã quét (trên Tune) vào: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
