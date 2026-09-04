"""Gán nhãn 3 lớp (Positive / Negative / Neutral) từ positive_score và
negative_score trong article_scores.parquet, theo "Cách 3" đã thống nhất:

    - positive_score == negative_score (kể cả cùng bằng 0, tức KHÔNG có từ
      sentiment nào match)              -> Neutral
    - positive_score > negative_score   -> Positive
    - positive_score < negative_score   -> Negative

Không dùng ngưỡng số tùy ý (epsilon) - so sánh trực tiếp 2 điểm. Giới hạn đã
biết: cách này không bắt được ca "vừa phải cả 2 chiều nhưng lệch rất nhẹ"
(VD positive=0.01 vs negative=0.0099 vẫn bị gán Positive dứt khoát, không có
vùng đệm "gần bằng nhau"). Xem đánh giá bên dưới để biết cách này có đủ tốt
không trước khi quyết định có cần thêm vùng đệm hay không.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORING_DIR = Path(__file__).resolve().parent
ARTICLE_SCORES_PATH = SCORING_DIR / "data" / "article_scores.parquet"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"
OUTPUT_LABELED_PATH = SCORING_DIR / "data" / "article_scores_labeled.parquet"
OUTPUT_EVAL_PATH = SCORING_DIR / "data" / "evaluation_vs_ground_truth.csv"


def assign_three_class_label(row: pd.Series) -> str:
    if row["positive_score"] > row["negative_score"]:
        return "Positive"
    if row["positive_score"] < row["negative_score"]:
        return "Negative"
    return "Neutral"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc article_scores.parquet ...")
    scores_df = pd.read_parquet(ARTICLE_SCORES_PATH)
    scores_df = scores_df.reset_index(drop=True)
    scores_df["source_row_id"] = scores_df.index

    print("Gán nhãn 3 lớp theo Cách 3 ...")
    scores_df["predicted_label"] = scores_df.apply(assign_three_class_label, axis=1)
    print(scores_df["predicted_label"].value_counts().to_string())

    scores_df.to_parquet(OUTPUT_LABELED_PATH, index=False)
    print(f"Đã lưu điểm + nhãn dự đoán vào: {OUTPUT_LABELED_PATH}")

    print()
    print("Đọc ground_truth_labeled.csv (152 bài đã gán tay) ...")
    gt_df = pd.read_csv(GROUND_TRUTH_PATH, encoding="utf-8-sig")

    merged_df = gt_df.merge(
        scores_df[["source_row_id", "positive_score", "negative_score", "net_sentiment_score", "predicted_label"]],
        on="source_row_id",
        how="left",
    )
    missing = merged_df["predicted_label"].isna().sum()
    if missing:
        print(f"CẢNH BÁO: {missing} bài trong ground truth không khớp được source_row_id.")
    merged_df = merged_df.dropna(subset=["predicted_label"])

    merged_df.to_csv(SCORING_DIR / "data" / "ground_truth_merged.csv", index=False, encoding="utf-8-sig")

    # --- Đánh giá: predicted_label (lexicon Cách 3) so với sentiment (người gán tay) ---
    y_true = merged_df["sentiment"]
    y_pred = merged_df["predicted_label"]

    accuracy = (y_true == y_pred).mean()
    print()
    print(f"Số bài đối chiếu được: {len(merged_df)} / {len(gt_df)}")
    print(f"Accuracy (lexicon Cách 3 vs người gán tay): {accuracy:.4f}")

    labels = ["Positive", "Neutral", "Negative"]
    confusion = pd.crosstab(y_true, y_pred, rownames=["ground_truth"], colnames=["predicted"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    print()
    print("Confusion matrix (hàng = ground truth, cột = dự đoán):")
    print(confusion.to_string())

    print()
    print("Precision / Recall / F1 theo từng lớp:")
    rows = []
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = (y_true == label).sum()
        rows.append(
            {"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(OUTPUT_EVAL_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu bảng đánh giá vào: {OUTPUT_EVAL_PATH}")


if __name__ == "__main__":
    main()
