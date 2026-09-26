"""The one production CV run per model, persisted so nothing re-runs it.

Every production model (``experiment_only_TF_IDF/run_model.py``,
``experiment_Lexicon_features/run_model.py``) is scored ONCE with
``repeated_cv.run_repeated_cv_proba`` (10 repeats x 5-fold, fold seed =
repeat index) on the full ground truth. The out-of-fold probabilities of
every repeat are saved to ``data/{name}_oof_probabilities.npz`` and reused by:

  - the reported production numbers (``{name}_metrics.csv`` /
    ``{name}_confusion_matrix.csv``: mean over repeats, plus std) - so each
    model has exactly one CV number on the full ground truth;
  - ``ensemble`` (averages its members' saved probabilities per repeat
    instead of re-running each member's CV);
  - ``experiment_only_TF_IDF/compare_models_tfidf_only.py`` (bootstrap /
    McNemar / Nadeau-Bengio on the same predictions);
  - ``Common/calibration_check.py`` (repeat 0 of the production RF +lexicon
    and SVM, instead of re-training those two variants).

``load_production_oof`` refuses a file whose rows / labels no longer match
the current ground truth, so a stale file can never be silently reused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    DATA_DIR,
    VALID_LABELS,
    compute_metrics,
    confusion_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    N_REPEATS,
    load_stopword_set,
    run_repeated_cv_proba,
)


def oof_path(name: str) -> Path:
    return DATA_DIR / f"{name}_oof_probabilities.npz"


def save_production_oof(
    name: str, probabilities: np.ndarray, source_row_id: np.ndarray, y: np.ndarray
) -> Path:
    path = oof_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=probabilities,
        source_row_id=np.asarray(source_row_id, dtype=np.int64),
        y=np.asarray(y, dtype=int),
    )
    return path


def load_production_oof(name: str, source_row_id: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Saved ``(n_repeats, n_rows, n_labels)`` probabilities for ``name``,
    after checking they were produced on exactly these rows and labels."""
    path = oof_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run News/Build_sentiment_label/Traditional_ML/run_pipeline.py first."
        )
    saved = np.load(path)
    if not (
        np.array_equal(saved["source_row_id"], np.asarray(source_row_id, dtype=np.int64))
        and np.array_equal(saved["y"], np.asarray(y, dtype=int))
    ):
        raise ValueError(
            f"{path} was produced on different ground-truth rows/labels - "
            "re-run News/Build_sentiment_label/Traditional_ML/run_pipeline.py."
        )
    return saved["probabilities"]


def run_production_cv(
    name: str,
    estimator_factory,
    df: pd.DataFrame,
    term_counts,
    y: np.ndarray,
    extra_features: np.ndarray | None = None,
    n_repeats: int = N_REPEATS,
) -> np.ndarray:
    probabilities = run_repeated_cv_proba(
        estimator_factory,
        term_counts,
        y,
        load_stopword_set(),
        n_repeats=n_repeats,
        extra_features=extra_features,
    )
    save_production_oof(name, probabilities, df["source_row_id"].to_numpy(), y)
    return probabilities


def summarize_repeats(
    y: np.ndarray, probabilities: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (metrics_df, confusion_df).

    metrics_df   : per-class + overall metrics averaged over repeats, with
                   ``f1_std`` / ``accuracy_std`` across repeats.
    confusion_df : confusion matrix averaged over repeats (row counts / repeat).
    """
    y = np.asarray(y, dtype=int)
    predictions_by_repeat = probabilities.argmax(axis=2)
    n_repeats = len(predictions_by_repeat)

    per_repeat = pd.concat(
        [compute_metrics(y, predictions) for predictions in predictions_by_repeat],
        ignore_index=True,
    )
    scope_order = [*VALID_LABELS, "overall"]
    grouped = per_repeat.groupby("metric_scope", sort=False)
    metrics_df = grouped.mean(numeric_only=True).reindex(scope_order)
    metrics_df["f1_std"] = grouped["f1"].std(ddof=1).reindex(scope_order)
    metrics_df["accuracy_std"] = grouped["accuracy"].std(ddof=1).reindex(scope_order)
    metrics_df["n_repeats"] = n_repeats
    metrics_df = metrics_df.reset_index()[
        [
            "metric_scope",
            "precision",
            "recall",
            "f1",
            "f1_std",
            "support",
            "accuracy",
            "accuracy_std",
            "n_repeats",
        ]
    ]

    mean_confusion = np.mean(
        [confusion_matrix(y, predictions) for predictions in predictions_by_repeat], axis=0
    )
    confusion_df = pd.DataFrame(
        mean_confusion,
        index=[f"true_{lbl}" for lbl in VALID_LABELS],
        columns=[f"pred_{lbl}" for lbl in VALID_LABELS],
    ).reset_index(names="true_label")
    return metrics_df, confusion_df


def write_production_outputs(name: str, y: np.ndarray, probabilities: np.ndarray) -> None:
    metrics_df, confusion_df = summarize_repeats(y, probabilities)

    paths = {
        "metrics": DATA_DIR / f"{name}_metrics.csv",
        "confusion matrix": DATA_DIR / f"{name}_confusion_matrix.csv",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    confusion_df.to_csv(paths["confusion matrix"], index=False, encoding="utf-8-sig")

    print(f"Metrics (mean over {len(probabilities)} repeats x 5-fold CV):")
    print(metrics_df.to_string(index=False))
    print("Confusion matrix (mean over repeats):")
    print(confusion_df.round(1).to_string(index=False))
    if oof_path(name).exists():  # ensemble has no OOF file of its own
        print("Output OOF probabilities:", oof_path(name))
    for label, path in paths.items():
        print(f"Output {label}:", path)
