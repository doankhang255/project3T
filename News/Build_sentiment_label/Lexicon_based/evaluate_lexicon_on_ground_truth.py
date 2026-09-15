"""Đánh giá pipeline Lexicon-Based (Cách 1 PMI = Scoring/, Cách 2 intensity =
Scoring_Intensity/) trên ground truth GỘP (ground_truth_combined.csv =
ground_truth_labeled.csv + ground_truth_news.csv, xác nhận 0 trùng
source_row_id). Tên file KHÔNG chứa số bài (khác bản cũ
ground_truth_combined_402.csv đã xóa) vì số bài sẽ tăng dần khi có thêm nhãn
mới - LƯU Ý: nếu 2 file nguồn được cập nhật thêm bài, cần chạy lại bước gộp
thủ công (xem lịch sử chat) trước khi chạy script này, file
ground_truth_combined.csv KHÔNG tự động đồng bộ.

QUYẾT ĐỊNH (theo yêu cầu người dùng 2026-09-12): xử lý trên đúng 1 tập gộp
này, KHÔNG tách riêng in-sample (152 cũ) / out-of-sample (bài mới) nữa. LƯU Ý
cho lần tune tham số sau: vì không còn tách in-sample/OOS, MỌI accuracy đo
trên tập gộp này từ nay là IN-SAMPLE cho bất kỳ thay đổi nào được quyết định
dựa trên chính con số đó (đúng vấn đề mentor nêu ở điểm 2) - muốn có ước
lượng khách quan cần tune/holdout split riêng, không dùng lại toàn bộ tập gộp
để vừa tune vừa báo cáo.

Nhãn dự đoán = "Cách 3": so trực tiếp positive_score vs negative_score trong
article_scores*.parquet (không margin). source_row_id = chỉ số dòng 0-based
trong article_scores*.parquet (đã kiểm chứng trước đó).

Output: data/evaluation_on_ground_truth_sets.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCORING_DIR.parents[2]
DATA_NEWS_DIR = PROJECT_ROOT / "data_news"

METHOD_SCORE_PATHS = {
    "Cach1_PMI": SCORING_DIR / "Scoring" / "data" / "article_scores.parquet",
    "Cach2_Intensity": SCORING_DIR / "Scoring_Intensity" / "data" / "article_scores_intensity.parquet",
}
GROUND_TRUTH_SETS = {
    "all_combined": DATA_NEWS_DIR / "ground_truth_combined.csv",
}
LABELS = ["Positive", "Neutral", "Negative"]
OUTPUT_PATH = SCORING_DIR / "data" / "evaluation_on_ground_truth_sets.csv"


def assign_three_class_label(row: pd.Series) -> str:
    if row["positive_score"] > row["negative_score"]:
        return "Positive"
    if row["positive_score"] < row["negative_score"]:
        return "Negative"
    return "Neutral"


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    accuracy = float((y_true.values == y_pred.values).mean())
    per_class = {}
    macro_f1_parts = []
    for label in LABELS:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int((y_true == label).sum())}
        macro_f1_parts.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(macro_f1_parts) / len(macro_f1_parts), "per_class": per_class}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ground_truth_frames = {}
    for name, path in GROUND_TRUTH_SETS.items():
        df = pd.read_csv(path, encoding="utf-8-sig")[["source_row_id", "sentiment"]].copy()
        df["source_row_id"] = df["source_row_id"].astype(int)
        ground_truth_frames[name] = df

    rows = []
    for method_name, score_path in METHOD_SCORE_PATHS.items():
        scores_df = pd.read_parquet(score_path, columns=["positive_score", "negative_score"]).reset_index(drop=True)
        scores_df["source_row_id"] = scores_df.index
        scores_df["predicted_label"] = scores_df.apply(assign_three_class_label, axis=1)

        for set_name, gt_df in ground_truth_frames.items():
            merged = gt_df.merge(scores_df[["source_row_id", "predicted_label"]], on="source_row_id", how="left")
            missing = int(merged["predicted_label"].isna().sum())
            merged = merged.dropna(subset=["predicted_label"])
            metrics = evaluate(merged["sentiment"], merged["predicted_label"])

            print(f"\n=== {method_name}  |  {set_name}  (n={len(merged)}, thiếu {missing}) ===")
            print(f"Accuracy = {metrics['accuracy']:.4f}   Macro-F1 = {metrics['macro_f1']:.4f}")
            confusion = pd.crosstab(
                merged["sentiment"], merged["predicted_label"], rownames=["thực"], colnames=["dự đoán"]
            ).reindex(index=LABELS, columns=LABELS, fill_value=0)
            print(confusion.to_string())
            for label in LABELS:
                pc = metrics["per_class"][label]
                print(f"  {label:>8}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (support {pc['support']})")

            rows.append(
                {
                    "method": method_name,
                    "ground_truth_set": set_name,
                    "n": len(merged),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    summary_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n\n=== TỔNG HỢP ===")
    print(summary_df.to_string(index=False))
    print("\nĐã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
