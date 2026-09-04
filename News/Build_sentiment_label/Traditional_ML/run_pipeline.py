"""Flow 1 - run the whole Traditional_ML pipeline in order.

    python News/Build_sentiment_label/Traditional_ML/run_pipeline.py

Steps, stopping on the first failure:

1. prepare_ground_truth.py  - join VNCoreNLP tokens onto the labeled rows
2. TF_IDF.py                - whole-corpus TF-IDF artifact (descriptive only)
3. model/*.py               - leak-free 5-fold CV for each classifier
4. compare_models.py        - merge per-model metrics + rank by macro F1
5. RESULTS_SUMMARY.txt      - regenerated from the fresh CSV outputs
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_SUMMARY_PATH = SCRIPT_DIR / "RESULTS_SUMMARY.txt"

PIPELINE_STEPS = [
    SCRIPT_DIR / "prepare_ground_truth.py",
    SCRIPT_DIR / "TF_IDF.py",
    SCRIPT_DIR / "model" / "logistic_regression.py",
    SCRIPT_DIR / "model" / "naive_bayes.py",
    SCRIPT_DIR / "model" / "random_forest.py",
    SCRIPT_DIR / "model" / "svm.py",
    SCRIPT_DIR / "compare_models.py",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression (class_weight=balanced)",
    "naive_bayes": "Naive Bayes (MultinomialNB)",
    "random_forest": "Random Forest (300 trees, class_weight=balanced)",
    "svm": "SVM (LinearSVC + Platt calibration)",
}


def run_step(script_path: Path) -> None:
    rel = script_path.relative_to(SCRIPT_DIR)
    print(f"\n{'=' * 70}\n>>> {rel}\n{'=' * 70}", flush=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR.parents[3]),
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(f"Step failed ({result.returncode}): {rel}")


def _fmt(value: float, width: int = 10) -> str:
    return f"{value:<{width}.3f}"


def _confusion_block(model_name: str) -> list[str]:
    cm = pd.read_csv(DATA_DIR / f"{model_name}_confusion_matrix.csv")
    lines = [
        "  Confusion matrix (hàng = nhãn thật, cột = nhãn dự đoán):",
        "                 pred_neg  pred_neu  pred_pos",
    ]
    for _, row in cm.iterrows():
        lines.append(
            f"  {str(row['true_label']):<14} {int(row['pred_negative']):>5}    "
            f"{int(row['pred_neutral']):>6}   {int(row['pred_positive']):>6}"
        )
    return lines


def write_results_summary() -> None:
    comparison = pd.read_csv(DATA_DIR / "traditional_ml_model_comparison.csv")
    tokenized = pd.read_parquet(DATA_DIR / "ground_truth_labeled_tokenized.parquet")
    vocabulary = pd.read_csv(DATA_DIR / "tfidf_vocabulary.csv")

    label_counts = tokenized["sentiment"].str.strip().str.lower().value_counts().to_dict()
    n_rows = len(tokenized)
    ngram_counts = vocabulary["ngram_n"].value_counts().to_dict()

    overall = (
        comparison.loc[comparison["metric_scope"].eq("overall")]
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )

    lines: list[str] = []
    add = lines.append

    add("TÓM TẮT KẾT QUẢ - TRADITIONAL ML (Sentiment Classification)")
    add("=" * 64)
    add("")
    add("File này được sinh tự động bởi run_pipeline.py - không sửa tay.")
    add("")
    add("1. DỮ LIỆU ĐẦU VÀO")
    add("-" * 64)
    add(f"Tổng số dòng: {n_rows} bài báo đã được gán nhãn tay (ground truth)")
    add("")
    add("Phân bố nhãn:")
    for label in ["negative", "neutral", "positive"]:
        count = int(label_counts.get(label, 0))
        pct = 100.0 * count / n_rows if n_rows else 0.0
        add(f"  {label:<9}: {count} bài  ({pct:.1f}%)")
    add("")
    add(
        "Tokenizer: VNCoreNLP - token lấy từ equity_news_tokenized_vncorenlp.parquet"
    )
    add(
        "theo source_row_id (chung với nhánh Lexicon_based và Build_sentiment_index)."
    )
    add("")
    add(
        "Feature: TF-IDF trên n-gram tiếng Việt (1-3 gram), lọc stopword + min_df"
    )
    add("theo tỷ lệ + max_df_ratio = 0.85.")
    add(
        f"Vocab (fit trên toàn bộ GT, dùng để mô tả): {len(vocabulary)} term "
        f"({int(ngram_counts.get(1, 0))} unigram, {int(ngram_counts.get(2, 0))} bigram, "
        f"{int(ngram_counts.get(3, 0))} trigram)."
    )
    add("")
    add(
        "Đánh giá bằng 5-fold stratified cross-validation. TF-IDF + bước chọn"
    )
    add(
        "top-feature được fit RIÊNG trong từng train-fold (không rò rỉ fold validation)."
    )
    add("")
    add("")
    add("2. KẾT QUẢ TỪNG MODEL")
    add("-" * 64)

    for model_name, model_label in MODEL_LABELS.items():
        block = comparison.loc[comparison["model"].eq(model_name)]
        per_class = block.loc[
            block["metric_scope"].isin(["negative", "neutral", "positive"])
        ]
        overall_row = block.loc[block["metric_scope"].eq("overall")].iloc[0]

        add("")
        add(f"--- {model_label} ---")
        add("  Class      Precision  Recall   F1")
        for _, row in per_class.iterrows():
            add(
                f"  {row['metric_scope']:<9}  {_fmt(row['precision'])}"
                f" {_fmt(row['recall'])} {_fmt(row['f1'])}"
            )
        add("  " + "-" * 40)
        add(f"  Accuracy      : {overall_row['accuracy']:.3f}")
        add(f"  Macro F1      : {overall_row['f1']:.3f}")
        add("")
        lines.extend(_confusion_block(model_name))

    add("")
    add("")
    add("3. BẢNG XẾP HẠNG TỔNG HỢP (macro F1)")
    add("-" * 64)
    for rank, row in enumerate(overall.itertuples(index=False), start=1):
        add(
            f"  {rank}. {row.model:<22} F1 = {row.f1:.3f}   Accuracy = {row.accuracy:.3f}"
        )

    add("")
    add("")
    add("4. NHẬN XÉT")
    add("-" * 64)
    for model_name in MODEL_LABELS:
        block = comparison.loc[
            comparison["model"].eq(model_name)
            & comparison["metric_scope"].isin(["negative", "neutral", "positive"])
        ]
        weakest = block.loc[block["f1"].idxmin()]
        add(
            f"- {model_name}: class yếu nhất = {weakest['metric_scope']} "
            f"(F1 = {weakest['f1']:.3f}, recall = {weakest['recall']:.3f})."
        )
    add(
        "- Class 'positive' ít dữ liệu nhất (35/152) nên thường là class yếu nhất."
    )

    add("")
    add("")
    add("5. HẠN CHẾ QUAN TRỌNG NHẤT: GROUND TRUTH QUÁ ÍT")
    add("-" * 64)
    add(
        f"{n_rows} dòng vẫn là bộ dữ liệu RẤT NHỎ cho bài toán phân loại 3 lớp. Mỗi"
    )
    add(
        "fold validation chỉ ~30 dòng, class positive chỉ ~7 dòng/fold, nên các con"
    )
    add("số accuracy/F1 ở trên chưa thật sự ổn định.")
    add("")
    add("Đã xử lý trong đợt refactor này:")
    add(
        "  (2) hết rò rỉ - vocab/IDF/chọn feature fit trong từng train-fold."
    )
    add("  (3) giữ MAX_FEATURES nhưng gọi bên trong fold (train-only).")
    add(
        "  (4) tokenizer = VNCoreNLP, đồng bộ với phần còn lại của project."
    )
    add(
        "  (7) một đường CV chung, n_splits = min(5, số dòng của lớp nhỏ nhất)."
    )
    add("")
    add(
        "Còn lại: cần gán nhãn thêm ground truth (vài trăm - vài nghìn dòng, cân"
    )
    add(
        "bằng hơn giữa 3 class) trước khi kết luận model nào tốt hơn và so sánh"
    )
    add("công bằng với Lexicon-based / Transformer (PhoBERT).")
    add("")

    text = "\n".join(line.rstrip() for line in lines)
    RESULTS_SUMMARY_PATH.write_text(text, encoding="utf-8")
    print("\nRegenerated:", RESULTS_SUMMARY_PATH)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for script_path in PIPELINE_STEPS:
        run_step(script_path)

    write_results_summary()

    print(f"\n{'=' * 70}")
    print("Flow 1 complete. Next: verify_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
