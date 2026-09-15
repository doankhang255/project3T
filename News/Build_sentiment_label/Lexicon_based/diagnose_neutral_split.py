"""Điểm 7 feedback mentor: tách 2 loại "Neutral" trong luật gán nhãn 3 lớp
để xem tỷ lệ mỗi loại (chưa đổi cách gán nhãn - chỉ chẩn đoán):

    - Neutral_balanced : positive_score == negative_score VÀ > 0
      -> bài CÓ từ sentiment, cân bằng -> 1 dự đoán thật.
    - Neutral_no_match : positive_score == negative_score == 0
      -> bài KHÔNG match từ positive/negative nào -> model không có thông
      tin, chỉ đoán mặc định Neutral. (Bài vẫn có thể match category khác
      như uncertainty/litigious nhưng không ảnh hưởng luật pos-vs-neg.)

In ra:
    1. Toàn corpus 126k: % mỗi loại Neutral / tổng.
    2. Trên ground truth gộp (ground_truth_combined.csv = ground_truth_labeled.csv
       + ground_truth_news.csv): trong số bài ĐƯỢC GÁN Neutral, bao nhiêu
       thuộc mỗi loại, và accuracy riêng từng loại (Neutral_no_match có thực
       sự là Neutral không, hay chỉ ăn may).

Output: data/neutral_split_diagnosis.csv
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
OUTPUT_PATH = SCORING_DIR / "data" / "neutral_split_diagnosis.csv"


def label_with_neutral_split(row: pd.Series) -> str:
    if row["positive_score"] > row["negative_score"]:
        return "Positive"
    if row["positive_score"] < row["negative_score"]:
        return "Negative"
    if row["positive_score"] == 0 and row["negative_score"] == 0:
        return "Neutral_no_match"
    return "Neutral_balanced"


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
        scores_df["label_split"] = scores_df.apply(label_with_neutral_split, axis=1)

        print(f"\n{'='*70}\n{method_name}\n{'='*70}")

        # 1. Toàn corpus
        corpus_dist = scores_df["label_split"].value_counts()
        total = len(scores_df)
        print(f"\n[1] Phân phối nhãn toàn corpus ({total:,} bài):")
        for label, count in corpus_dist.items():
            print(f"  {label:>18}: {count:>7,}  ({count/total*100:5.1f}%)")
        n_neutral_total = corpus_dist.get("Neutral_balanced", 0) + corpus_dist.get("Neutral_no_match", 0)
        if n_neutral_total:
            share_no_match = corpus_dist.get("Neutral_no_match", 0) / n_neutral_total * 100
            print(f"  -> Trong tổng số bài Neutral, {share_no_match:.1f}% là NO-MATCH (đoán mặc định, không có tín hiệu)")

        # 2. Trên từng tập ground truth
        for set_name, gt_df in ground_truth_frames.items():
            merged = gt_df.merge(scores_df[["source_row_id", "label_split"]], on="source_row_id", how="left").dropna(
                subset=["label_split"]
            )
            neutral_mask = merged["label_split"].isin(["Neutral_balanced", "Neutral_no_match"])
            n_pred_neutral = int(neutral_mask.sum())
            print(f"\n[2] {set_name} (n={len(merged)}): số bài ĐƯỢC GÁN Neutral = {n_pred_neutral}")
            for neutral_type in ["Neutral_balanced", "Neutral_no_match"]:
                sub = merged.loc[merged["label_split"] == neutral_type]
                if len(sub) == 0:
                    print(f"    {neutral_type:>18}: 0 bài")
                    continue
                correct = int((sub["sentiment"] == "Neutral").sum())
                true_dist = sub["sentiment"].value_counts().to_dict()
                print(
                    f"    {neutral_type:>18}: {len(sub):>3} bài | đúng Neutral {correct}/{len(sub)} = {correct/len(sub)*100:.1f}% | nhãn thực: {true_dist}"
                )
                rows.append(
                    {
                        "method": method_name,
                        "ground_truth_set": set_name,
                        "neutral_type": neutral_type,
                        "n": len(sub),
                        "n_correct_neutral": correct,
                        "acc_within_type": correct / len(sub),
                    }
                )

    summary_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n\nĐã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
