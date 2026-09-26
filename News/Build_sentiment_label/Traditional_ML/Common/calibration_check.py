"""Before/after calibration check for RF and SVM (ML_SUMMARY.qmd section 7.2).

    python News/Build_sentiment_label/Traditional_ML/Common/calibration_check.py

Compares, on the SAME leak-free 5-fold split - repeat 0 (fold seed 0) of the
production 10 x 5-fold CV (``production_cv.py``):

  RF   raw       - RandomForestClassifier, no calibration (old production default)
  RF   isotonic  - CalibratedClassifierCV(rf, method="isotonic") (new production default)
  SVM  platt     - CalibratedClassifierCV(LinearSVC, method="sigmoid") (old production
                   default - reconstructed here for comparison only, no longer in
                   model/svm.py)
  SVM  softmax   - MarginSoftmaxSVC: softmax(decision_function), no fitted
                   calibration (new production default)

The four variants above all run on plain TF-IDF features. But the isotonic
vs. raw decision for Random Forest is also the actual production default in
``model/random_forest.py::build_estimator``, and production Random Forest
runs with the +lexicon feature block hstacked on (see
``experiment_Lexicon_features/run_model.py``) - random_forest being the only
model among the 3 lexicon-eligible ones (logistic_regression, random_forest,
svm) that showed a reliable, replicated improvement from adding lexicon
features (see ML_SUMMARY.qmd section 6.2). The 4 variants above never
exercise that feature space, so the calibration decision was previously
validated only on TF-IDF-only features. Two more variants close that gap
(same fix already applied to the leak check in ``sanity_checks.py``, mirrored
here for calibration quality):

  RF   raw_lexicon      - build_rf_raw on TF-IDF + lexicon feature block
  RF   isotonic_lexicon - build_rf_isotonic on TF-IDF + lexicon feature block

(SVM has no lexicon variant here: SVM+lexicon was never promoted to
production, so its calibration quality on that unused path isn't a
production concern.)

No duplicate CV: ``rf_isotonic`` is read from the saved run of
``experiment_only_TF_IDF/compare_models_tfidf_only.py`` (repeat 0, run that
first). ``rf_isotonic_lexicon`` and ``svm_softmax`` ARE the
production random_forest and svm, so their probabilities are read from the
saved production run (repeat 0) instead of being re-trained here - run
``../run_pipeline.py`` first. The other 3 variants are not production
models and are run here, on the same fold seed 0 split.

Two measures per variant, both on the out-of-fold probabilities (never the
resubstitution predictions):

- Brier score (multiclass): mean over rows of sum_c (p_c - 1{y=c})^2. Lower is
  better; a model that always outputs the empirical class frequency already
  gets a nontrivial baseline score, so compare across variants of the SAME
  model, not across models.
- Reliability table: bin rows by the predicted (argmax) class's own
  probability, compare the bin's mean confidence to its empirical accuracy.
  A well-calibrated model has confidence ~= accuracy in every bin.
  Summarized as ECE (expected calibration error - the accuracy-weighted mean
  |confidence - accuracy| over bins).

Also reports macro-F1 per variant, since isotonic calibration (unlike a
monotonic per-class rescaling) can change which class wins the argmax.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.Common.production_cv import load_production_oof
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    load_stopword_set,
    run_single_cv_proba,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    compute_metrics,
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    build_lexicon_feature_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.random_forest import (
    build_base_random_forest,
    build_estimator as build_rf_isotonic,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.svm import (
    build_base_svm,
    build_estimator as build_svm_softmax,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OLD_CALIBRATION_CV = 3
N_BINS = 10


def build_rf_raw(random_state: int):
    return build_base_random_forest(random_state)


def build_svm_platt(random_state: int) -> CalibratedClassifierCV:
    """The pre-fix production SVM (Platt/sigmoid calibration) - reconstructed
    here only for this comparison; model/svm.py no longer builds this."""
    return CalibratedClassifierCV(build_base_svm(random_state), cv=OLD_CALIBRATION_CV)


VARIANTS = {
    "rf_raw": build_rf_raw,
    "rf_isotonic": build_rf_isotonic,
    "rf_raw_lexicon": build_rf_raw,
    "rf_isotonic_lexicon": build_rf_isotonic,
    "svm_platt": build_svm_platt,
    "svm_softmax": build_svm_softmax,
}

# Variant names that run on TF-IDF + lexicon feature block instead of plain
# TF-IDF - same factories as their non-"_lexicon" counterparts, only the
# extra_features passed to run_single_cv_proba differs (see main()).
USES_LEXICON = {"rf_raw_lexicon", "rf_isotonic_lexicon"}

# Variants that are exactly a production model -> read that model's saved
# production OOF probabilities (repeat 0) instead of re-training it.
PRODUCTION_VARIANTS = {
    "rf_isotonic_lexicon": "random_forest",
    "svm_softmax": "svm",
    # not production, but saved by experiment_only_TF_IDF/compare_models_tfidf_only.py
    "rf_isotonic": "random_forest_tfidf_only",
}
PRODUCTION_REPEAT = 0


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_labels: int = 3) -> float:
    onehot = np.eye(n_labels)[y_true]
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def reliability_table(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = N_BINS
) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        n = int(mask.sum())
        if n == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        empirical_accuracy = float(correct[mask].mean())
        rows.append(
            {
                "bin": f"[{bin_edges[bin_id]:.1f}, {bin_edges[bin_id + 1]:.1f}]",
                "n": n,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "gap": mean_confidence - empirical_accuracy,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(reliability_df: pd.DataFrame, n_total: int) -> float:
    if reliability_df.empty:
        return float("nan")
    weights = reliability_df["n"] / n_total
    return float(np.sum(weights * reliability_df["gap"].abs()))


def _fmt_reliability(reliability_df: pd.DataFrame) -> list[str]:
    lines = [f"  {'bin':<14}{'n':>6}{'mean_conf':>12}{'emp_acc':>10}{'gap':>9}"]
    for row in reliability_df.itertuples(index=False):
        lines.append(
            f"  {row.bin:<14}{row.n:>6}{row.mean_confidence:>12.3f}"
            f"{row.empirical_accuracy:>10.3f}{row.gap:>+9.3f}"
        )
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "calibration_check_out")
    args = parser.parse_args()

    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    stopwords = load_stopword_set()
    lexicon_matrix = build_lexicon_feature_matrix(df["Tokenize_content"].tolist())
    label_counts = df["ground_truth_label"].value_counts().to_dict()
    print(f"Ground truth: {len(df)} rows {label_counts}")

    lines: list[str] = []
    add = lines.append
    add(
        "CALIBRATION CHECK - RF (raw vs isotonic, TF-IDF and +lexicon) / "
        "SVM (Platt vs softmax)"
    )
    add("=" * 68)
    add("")
    add(f"Ground truth: {len(df)} rows {label_counts}")
    add(
        f"One 5-fold split (fold seed {PRODUCTION_REPEAT} = production repeat "
        f"{PRODUCTION_REPEAT}); {', '.join(PRODUCTION_VARIANTS)} read from the "
        "saved production run, not re-trained."
    )
    add(
        "Brier score: mean_i sum_c (p_ic - 1{y_i=c})^2 - lower is better, "
        "compare within a model (raw vs isotonic / platt vs softmax), not across."
    )
    add(
        "ECE: accuracy-weighted mean |confidence - empirical accuracy| over "
        f"{N_BINS} confidence bins of the predicted (argmax) class."
    )
    add("")

    summary_rows = []
    for name, factory in VARIANTS.items():
        if name in PRODUCTION_VARIANTS:
            production_name = PRODUCTION_VARIANTS[name]
            print(f"\n[{name}] reusing saved OOF ({production_name}, repeat {PRODUCTION_REPEAT})")
            probabilities = load_production_oof(
                production_name, df["source_row_id"].to_numpy(), y
            )[PRODUCTION_REPEAT]
        else:
            print(f"\n[{name}] 5-fold CV (fold seed {PRODUCTION_REPEAT}) ...", flush=True)
            extra_features = lexicon_matrix if name in USES_LEXICON else None
            probabilities = run_single_cv_proba(
                factory, term_counts, y, stopwords, PRODUCTION_REPEAT, extra_features=extra_features
            )
        predictions = probabilities.argmax(axis=1)
        metrics = compute_metrics(y, predictions)
        macro_f1 = float(metrics.loc[metrics["metric_scope"].eq("overall"), "f1"].iloc[0])
        brier = brier_score(y, probabilities)
        reliability_df = reliability_table(y, probabilities)
        ece = expected_calibration_error(reliability_df, len(y))
        summary_rows.append(
            {"variant": name, "macro_f1": macro_f1, "brier": brier, "ece": ece}
        )
        print(f"    macro_f1={macro_f1:.3f}  brier={brier:.4f}  ece={ece:.4f}")

        add(f"--- {name} ---")
        add(f"  macro_f1 = {macro_f1:.3f}   brier = {brier:.4f}   ece = {ece:.4f}")
        lines.extend(_fmt_reliability(reliability_df))
        add("")

    summary_df = pd.DataFrame(summary_rows)
    add("SUMMARY")
    add("-" * 68)
    add(f"  {'variant':<14}{'macro_f1':>10}{'brier':>10}{'ece':>10}")
    for row in summary_df.itertuples(index=False):
        add(f"  {row.variant:<14}{row.macro_f1:>10.3f}{row.brier:>10.4f}{row.ece:>10.4f}")
    add("")
    for model_name, raw_name, cal_name in (
        ("random_forest", "rf_raw", "rf_isotonic"),
        ("random_forest_lexicon", "rf_raw_lexicon", "rf_isotonic_lexicon"),
        ("svm", "svm_platt", "svm_softmax"),
    ):
        raw_row = summary_df.set_index("variant").loc[raw_name]
        cal_row = summary_df.set_index("variant").loc[cal_name]
        add(
            f"  {model_name}: ece {raw_row['ece']:.4f} -> {cal_row['ece']:.4f}  "
            f"({'improved' if cal_row['ece'] < raw_row['ece'] else 'WORSE'}), "
            f"brier {raw_row['brier']:.4f} -> {cal_row['brier']:.4f} "
            f"({'improved' if cal_row['brier'] < raw_row['brier'] else 'WORSE'}), "
            f"macro_f1 {raw_row['macro_f1']:.3f} -> {cal_row['macro_f1']:.3f}"
        )
    add("")

    report = "\n".join(lines)
    print("\n" + report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "RESULTS.txt"
    results_path.write_text(report + "\n", encoding="utf-8")
    summary_df.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten: {results_path}")


if __name__ == "__main__":
    main()
