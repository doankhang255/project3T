"""External sanity checks - independent of the leak-free CV harness itself.

M2.1-M2.4 (repeated CV, McNemar, bootstrap) all assume the harness is
correctly leak-free; they cannot catch a bug in the harness's own logic. The
two checks here are deliberately independent of it:

1. permutation_test - Ojala & Garriga (2010), "Permutation Tests for Studying
   Classifier Performance": shuffle the labels so there is provably zero
   relationship between text and label, then run the exact same leak-free CV.
   If the harness is honestly leak-free, macro-F1 on shuffled labels must
   collapse to chance level. If it stays close to the real score, something
   leaks (vocab fit on validation rows, a fold boundary bug, ...) that the
   "looks correct on paper" checks would not catch.

2. sklearn_reference_check - a second, completely independent implementation
   (sklearn's own TfidfVectorizer + LogisticRegression + StratifiedKFold,
   not one line of this project's TF_IDF.py / repeated_cv.py) on the same
   rows. If the hand-rolled pipeline's score is wildly outside what this
   independent implementation gets, that is a discrepancy to explain before
   trusting the hand-rolled numbers. It is wrapped in a sklearn Pipeline (not
   a pre-fit matrix) specifically so TfidfVectorizer is refit per training
   fold inside cross_val_predict - a pre-fit matrix would leak exactly the
   way the original (pre-fix) TF_IDF.py used to.

    python News/Build_sentiment_label/Traditional_ML/improve/sanity_checks.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.improve.repeated_cv import (
    load_stopword_set,
    run_single_cv,
)
from News.Build_sentiment_label.Traditional_ML.improve.run_improve import load_frame_from_csv
from News.Build_sentiment_label.Traditional_ML.model.common import (
    compute_metrics,
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.model.naive_bayes import (
    build_estimator as build_naive_bayes,
)

DEFAULT_GROUND_TRUTH_CSV = (
    Path(__file__).resolve().parent / "tune_holdout" / "ground_truth_tune.csv"
)
N_SHUFFLES = 30


def _macro_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    metrics = compute_metrics(y_true, predictions)
    return float(metrics.loc[metrics["metric_scope"].eq("overall"), "f1"].iloc[0])


def permutation_test(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    n_shuffles: int = N_SHUFFLES,
    seed: int = 20260913,
) -> dict:
    y = np.asarray(y, dtype=int)
    real_predictions = run_single_cv(estimator_factory, term_counts, y, stopwords, fold_seed=0)
    real_score = _macro_f1(y, real_predictions)

    rng = np.random.default_rng(seed)
    shuffled_scores = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        predictions = run_single_cv(estimator_factory, term_counts, y_shuffled, stopwords, fold_seed=i)
        shuffled_scores[i] = _macro_f1(y_shuffled, predictions)

    # standard permutation-test p-value: how often does a shuffled run match
    # or beat the real score, +1/+1 correction so p is never exactly 0.
    p_value = float((np.sum(shuffled_scores >= real_score) + 1) / (n_shuffles + 1))
    return {
        "real_score": real_score,
        "shuffled_mean": float(shuffled_scores.mean()),
        "shuffled_std": float(shuffled_scores.std(ddof=1)),
        "shuffled_max": float(shuffled_scores.max()),
        "n_shuffles": n_shuffles,
        "p_value": p_value,
        "shuffled_scores": shuffled_scores,
    }


def sklearn_reference_check(frame, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    texts = [" ".join(str(t) for t in tokens) for tokens in frame["Tokenize_content"]]
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.85, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = cross_val_predict(pipeline, texts, y, cv=skf)
    return {
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "accuracy": float((predictions == y).mean()),
        "vocab_size": len(pipeline.fit(texts, y).named_steps["tfidf"].vocabulary_),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV)
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    args = parser.parse_args()

    if args.ground_truth_csv is not None and args.ground_truth_csv.exists():
        frame = load_frame_from_csv(args.ground_truth_csv)
        source = args.ground_truth_csv
    else:
        frame = load_ground_truth_frame()
        source = "model/common.load_ground_truth_frame() (152-row committed set)"

    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    label_counts = frame["ground_truth_label"].value_counts().to_dict()

    print(f"Ground truth: {source} ({len(frame)} rows) {label_counts}")

    print(f"\n=== 1. PERMUTATION TEST (naive_bayes, {args.n_shuffles} shuffles) ===")
    perm = permutation_test(build_naive_bayes, term_counts, y, stopwords, n_shuffles=args.n_shuffles)
    print(f"  real macro-F1            : {perm['real_score']:.4f}")
    print(
        f"  shuffled-label macro-F1  : {perm['shuffled_mean']:.4f} +/- "
        f"{perm['shuffled_std']:.4f}  (max over {perm['n_shuffles']} shuffles: "
        f"{perm['shuffled_max']:.4f})"
    )
    gap_in_std = (
        (perm["real_score"] - perm["shuffled_mean"]) / perm["shuffled_std"]
        if perm["shuffled_std"] > 0
        else float("inf")
    )
    print(f"  gap                      : {gap_in_std:.1f} std above the shuffled-label mean")
    print(f"  permutation p-value      : {perm['p_value']:.4f}")
    if perm["real_score"] > perm["shuffled_max"]:
        print("  -> real score exceeds EVERY shuffled run: no sign of leakage.")
    else:
        print("  -> WARNING: a shuffled (label-free) run matched or beat the real score.")

    print("\n=== 2. INDEPENDENT sklearn REFERENCE (Pipeline: TfidfVectorizer + LogisticRegression) ===")
    ref = sklearn_reference_check(frame, y)
    print(f"  sklearn pipeline macro-F1: {ref['macro_f1']:.4f}  accuracy: {ref['accuracy']:.4f}")
    print(f"  sklearn vocab size (fit on all rows, for reference only): {ref['vocab_size']}")
    print(
        "  Compare to this project's logistic_regression result on the same rows "
        "(see improve/RESULTS.txt or improve/gt599_tune/RESULTS.txt) - should land in "
        "the same ballpark (+/- ~0.05), not wildly higher or lower."
    )


if __name__ == "__main__":
    main()
