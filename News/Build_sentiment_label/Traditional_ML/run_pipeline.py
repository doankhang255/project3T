"""Flow 1 - run the whole Traditional_ML pipeline in order.

    python News/Build_sentiment_label/Traditional_ML/run_pipeline.py

Steps, stopping on the first failure:

1. Common/prepare_ground_truth.py            - join VNCoreNLP tokens onto the labeled rows
2. Common/TF_IDF.py                          - whole-corpus TF-IDF artifact (descriptive only)
3. experiment_only_TF_IDF/run_model.py       - LR/NB/ComplementNB/SVM: leak-free 10 x 5-fold CV, TF-IDF only
4. experiment_Lexicon_features/run_model.py  - RF (+lexicon), then the ensemble (averages saved LR/NB/RF OOF)
5. compare_models.py                         - merge per-model metrics + rank by macro F1
6. RESULTS_SUMMARY.txt                       - regenerated from the fresh CSV outputs

Model definitions (build_estimator) live in Common/model/*.py, shared by
every runnable step above - only the "run CV + write CSV" step is split
across the two experiment folders, by whether that model uses the lexicon
feature block (see ML_SUMMARY.qmd section 6).
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
    SCRIPT_DIR / "Common" / "prepare_ground_truth.py",
    SCRIPT_DIR / "Common" / "TF_IDF.py",
    SCRIPT_DIR / "experiment_only_TF_IDF" / "run_model.py",
    SCRIPT_DIR / "experiment_Lexicon_features" / "run_model.py",
    SCRIPT_DIR / "compare_models.py",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression (class_weight=balanced, C=0.1 tuned)",
    "naive_bayes": "Naive Bayes (MultinomialNB)",
    "complement_nb": "Complement Naive Bayes (Rennie et al. 2003)",
    "random_forest": "Random Forest (300 trees, isotonic-calibrated, +lexicon features)",
    "svm": "SVM (LinearSVC, C=0.1 tuned, margin-softmax - not a calibrated probability)",
    "ensemble": "Ensemble (average probability of LR + NB + RF)",
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
        "  Confusion matrix, trung bình qua các lần lặp (hàng = nhãn thật, cột = dự đoán):",
        "                 pred_neg  pred_neu  pred_pos",
    ]
    for _, row in cm.iterrows():
        lines.append(
            f"  {str(row['true_label']):<14} {row['pred_negative']:>7.1f}  "
            f"{row['pred_neutral']:>8.1f}  {row['pred_positive']:>8.1f}"
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
    n_repeats = int(overall["n_repeats"].iloc[0])
    add(
        f"Đánh giá bằng {n_repeats} lần lặp x 5-fold stratified cross-validation"
    )
    add(
        "(mỗi lần lặp chia fold khác nhau; số dưới đây = trung bình +/- độ lệch"
    )
    add(
        "chuẩn qua các lần lặp). TF-IDF + bước chọn top-feature được fit RIÊNG"
    )
    add("trong từng train-fold (không rò rỉ fold validation).")
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
        add(
            f"  Accuracy      : {overall_row['accuracy']:.3f} +/- {overall_row['accuracy_std']:.3f}"
        )
        add(f"  Macro F1      : {overall_row['f1']:.3f} +/- {overall_row['f1_std']:.3f}")
        add("")
        lines.extend(_confusion_block(model_name))

    add("")
    add("")
    add("3. BẢNG XẾP HẠNG TỔNG HỢP (macro F1)")
    add("-" * 64)
    for rank, row in enumerate(overall.itertuples(index=False), start=1):
        add(
            f"  {rank}. {row.model:<22} F1 = {row.f1:.3f} +/- {row.f1_std:.3f}   "
            f"Accuracy = {row.accuracy:.3f}"
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
    smallest_label = min(label_counts, key=label_counts.get)
    smallest_count = int(label_counts[smallest_label])
    add(
        f"- Class '{smallest_label}' ít dữ liệu nhất ({smallest_count}/{n_rows}) "
        "nên thường là class yếu nhất."
    )

    add("")
    add("")
    add("5. HẠN CHẾ QUAN TRỌNG NHẤT: GROUND TRUTH VẪN CÒN NHỎ")
    add("-" * 64)
    fold_rows = n_rows // 5
    fold_smallest = smallest_count // 5
    add(
        f"{n_rows} dòng (tăng từ 152 dòng ban đầu) vẫn chưa lớn cho bài toán phân"
    )
    add(
        f"loại 3 lớp. Mỗi fold validation chỉ ~{fold_rows} dòng, class "
        f"'{smallest_label}' chỉ ~{fold_smallest} dòng/fold, nên các con số"
    )
    add("accuracy/F1 ở trên vẫn còn dao động theo cách chia fold.")
    add("")
    add("Đã xử lý qua các đợt refactor:")
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
    add(
        "  Ground truth: 152 -> 1044 dòng (data_news/ground_truth_combined.csv)."
    )
    add(
        "  Random Forest: thêm isotonic calibration (CalibratedClassifierCV) -"
    )
    add(
        "  predict_proba dạng vote-fraction trước đây bị lệch calibration"
    )
    add(
        "  (đo bằng reliability diagram, xem Common/calibration_check.py)."
    )
    add(
        "  SVM: bỏ Platt scaling (CalibratedClassifierCV) - dữ liệu calibrate"
    )
    add(
        "  trong từng inner-fold vẫn không đủ ổn định; predict_proba giờ là"
    )
    add(
        "  softmax của decision_function, CHỈ để xếp hạng, không phải xác suất"
    )
    add("  đã hiệu chỉnh - không dùng trong ensemble hay so sánh CI.")
    add(
        "  Thêm ensemble: trung bình xác suất LR + NB + RF (loại SVM vì lý do"
    )
    add("  trên).")
    add(
        "  Random Forest: thêm feature 7 nhóm lexicon tài chính + negation"
    )
    add(
        "  (model/lexicon_features.py) - đã test tune+holdout 3 lần (2 lần"
    )
    add(
        "  đầu không lặp lại được, lần 3 với GT lớn hơn + holdout mới thì"
    )
    add(
        "  replicate: holdout Delta=+0.049 CI[+0.017,+0.082] p=0.003). Chỉ RF"
    )
    add(
        "  dùng feature này (LR/SVM không có bằng chứng); xem ML_SUMMARY.qmd"
    )
    add("  mục 6.2.")
    add(
        "  Nested-CV tuning (Common/tune_hyperparameters.py): LogReg/SVM"
    )
    add(
        "  C=1.0 -> tuned: Delta=-0.008 CI[-0.019,+0.002] (LogReg), -0.003"
    )
    add(
        "  CI[-0.014,+0.008] (SVM) - chạm 0; NB/RF cũng không có cải thiện"
    )
    add(
        "  đáng tin. Code vẫn giữ C=0.1, xem ML_SUMMARY.qmd mục 5.3."
    )
    add("")
    add(
        "Còn lại: cần gán nhãn thêm ground truth (hàng nghìn dòng, cân bằng hơn"
    )
    add(
        "giữa 3 class) trước khi kết luận model nào tốt hơn và so sánh công bằng"
    )
    add("với Lexicon-based / Transformer (PhoBERT).")
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
    print("Flow 1 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
