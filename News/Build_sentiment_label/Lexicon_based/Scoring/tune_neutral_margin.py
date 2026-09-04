"""Thêm "vùng đệm" (margin) quanh 0 cho lớp Neutral, thay vì so sánh trực
tiếp positive_score == negative_score (Cách 3 gốc, không margin).

Quy tắc mới:
    diff = positive_score - negative_score
    |diff| <= margin   -> Neutral
    diff > margin      -> Positive
    diff < -margin     -> Negative

Thay vì BỊA 1 con số margin tùy tiện, script này QUÉT 1 dải giá trị margin
và chọn giá trị cho accuracy/macro-F1 CAO NHẤT khi đối chiếu với 152 bài
ground_truth_labeled.csv đã gán tay - tức margin được HIỆU CHỈNH THỰC
NGHIỆM trên dữ liệu thật, không phải số đoán mò.

CẢNH BÁO trung thực: tập ground truth chỉ có 152 bài - rất nhỏ, nên margin
"tối ưu" tìm được có rủi ro overfit vào đúng 152 bài này, không chắc tổng
quát hoá tốt cho toàn bộ 126,576 bài. Vì vậy script in ra CẢ DẢI margin xung
quanh điểm tối ưu để người đọc tự đánh giá độ ổn định (nếu accuracy dao động
mạnh giữa các margin gần nhau => dấu hiệu overfit, không nên tin tưởng con số
tối ưu tuyệt đối).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORING_DIR = Path(__file__).resolve().parent
ARTICLE_SCORES_LABELED_PATH = SCORING_DIR / "data" / "article_scores_labeled.parquet"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    labeled_df = pd.read_parquet(ARTICLE_SCORES_LABELED_PATH)
    gt_df = pd.read_csv(GROUND_TRUTH_PATH, encoding="utf-8-sig")
    merged = gt_df.merge(
        labeled_df[["source_row_id", "positive_score", "negative_score"]], on="source_row_id", how="left"
    )
    merged = merged.dropna(subset=["positive_score", "negative_score"])
    merged["diff"] = merged["positive_score"] - merged["negative_score"]

    print(f"Số bài đối chiếu: {len(merged)}")
    print(f"Phân phối |diff| trong tập ground truth: min={merged['diff'].abs().min():.5f}, "
          f"median={merged['diff'].abs().median():.5f}, max={merged['diff'].abs().max():.5f}")
    print()

    # Quét margin từ 0 đến giá trị đủ lớn để mọi bài đều thành Neutral.
    max_margin = merged["diff"].abs().max()
    margins = np.linspace(0, max_margin, 60)

    results = []
    for margin in margins:
        y_pred = merged["diff"].apply(lambda d: label_with_margin(d, margin))
        accuracy = (merged["sentiment"] == y_pred).mean()
        f1 = macro_f1(merged["sentiment"], y_pred)
        results.append({"margin": margin, "accuracy": accuracy, "macro_f1": f1})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    best_acc_row = results_df.loc[results_df["accuracy"].idxmax()]
    best_f1_row = results_df.loc[results_df["macro_f1"].idxmax()]

    print("=== Margin tối ưu theo ACCURACY ===")
    print(f"  margin = {best_acc_row['margin']:.5f} -> accuracy = {best_acc_row['accuracy']:.4f}, "
          f"macro_f1 = {best_acc_row['macro_f1']:.4f}")
    print()
    print("=== Margin tối ưu theo MACRO-F1 (cân bằng hơn giữa 3 lớp) ===")
    print(f"  margin = {best_f1_row['margin']:.5f} -> accuracy = {best_f1_row['accuracy']:.4f}, "
          f"macro_f1 = {best_f1_row['macro_f1']:.4f}")

    print()
    print("=== Độ ổn định quanh điểm tối ưu (accuracy) - 5 margin lân cận mỗi phía ===")
    best_idx = results_df["accuracy"].idxmax()
    lo = max(0, best_idx - 5)
    hi = min(len(results_df), best_idx + 6)
    print(results_df.iloc[lo:hi].to_string(index=False))

    print()
    print("=== So sánh với margin=0 (Cách 3 gốc, không đệm) ===")
    base_row = results_df.iloc[0]
    print(f"  margin=0     -> accuracy = {base_row['accuracy']:.4f}, macro_f1 = {base_row['macro_f1']:.4f}")
    print(f"  margin={best_acc_row['margin']:.5f} -> accuracy = {best_acc_row['accuracy']:.4f} "
          f"(cải thiện {best_acc_row['accuracy'] - base_row['accuracy']:+.4f})")

    # Confusion matrix tại margin tối ưu theo accuracy.
    best_margin = best_acc_row["margin"]
    y_pred_best = merged["diff"].apply(lambda d: label_with_margin(d, best_margin))
    confusion = pd.crosstab(
        merged["sentiment"], y_pred_best, rownames=["ground_truth"], colnames=["predicted"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    print()
    print(f"Confusion matrix tại margin tối ưu ({best_margin:.5f}):")
    print(confusion.to_string())

    print(f"\nĐã lưu toàn bộ dải margin đã quét vào: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
