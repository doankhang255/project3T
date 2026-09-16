"""Does the Nadeau-Bengio (2003) correction change any conclusion here?

    python News/Build_sentiment_label/Traditional_ML/Common/run_nadeau_bengio.py

Reuses ``run_repeated_cv`` (same OOF arrays as ``run_improve.py``) and applies
``nadeau_bengio.py`` on top - no retraining beyond the repeated CV itself.

The naive per-repeat std (what ``Common/README.md`` / ``run_improve.py``
already report) treats the ``n_repeats`` repeat-level scores as independent.
They are not independent at the *fold* level - every repeat's training folds
overlap. Nadeau & Bengio's correction re-derives the variance from the
``n_repeats * n_splits`` individual fold scores and inflates it by
``n_test/n_train`` to account for that overlap. It cannot move the point
estimate (the mean) and cannot narrow a CI - only widen it or leave it
unchanged, per ``nadeau_bengio.py``'s module docstring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.Common.nadeau_bengio import (
    nadeau_bengio_ci,
    nadeau_bengio_paired_test,
)
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    N_REPEATS,
    load_stopword_set,
    run_repeated_cv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model_factories import (
    MODEL_FACTORIES,
    summarize,
)
from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
    load_frame_from_csv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import encode_labels

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_CSV = SCRIPT_DIR / "tune_holdout" / "ground_truth_tune.csv"
DEFAULT_MODELS = ["logistic_regression", "multinomial_nb", "random_forest", "svm"]


def _fmt_ci(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def render_report(
    frame_len: int,
    label_counts: dict[str, int],
    n_repeats: int,
    per_repeat_by_model: dict[str, pd.DataFrame],
    ci_by_model: dict[str, dict],
    pair_results: list[tuple[str, str, dict]],
) -> str:
    lines: list[str] = []
    add = lines.append

    total = sum(label_counts.values())
    add("NADEAU-BENGIO CORRECTED VARIANCE - does it change any conclusion?")
    add("=" * 72)
    add("")
    add(
        f"Ground truth: {total} rows "
        f"(neg {label_counts.get('negative', 0)} / "
        f"neu {label_counts.get('neutral', 0)} / "
        f"pos {label_counts.get('positive', 0)}) | {n_repeats} repeats x 5-fold CV"
    )
    add(
        "Naive SE below = std/sqrt(n) over the n_repeats*5 individual FOLD "
        "scores (not the repeat-level mean+/-std elsewhere in Common/ - that"
    )
    add(
        "one already averages 5 folds per repeat first). Corrected SE inflates "
        "it by (1/n + n_test/n_train) per Nadeau & Bengio (2003)."
    )
    add("")

    add("PER-MODEL CI ON MEAN MACRO-F1")
    add("-" * 72)
    add(f"  {'model':<20}{'mean':>8}{'naive 95% CI':>20}{'corrected 95% CI':>24}{'widen x':>10}")
    for name in DEFAULT_MODELS:
        if name not in ci_by_model:
            continue
        result = ci_by_model[name]
        add(
            f"  {name:<20}{result['mean']:>8.3f}"
            f"{_fmt_ci(*result['naive_ci']):>20}"
            f"{_fmt_ci(*result['corrected_ci']):>24}"
            f"{result['widen_factor']:>10.2f}"
        )
    add("")

    add("PAIRWISE  (delta = model_a - model_b, corrected resampled paired t-test)")
    add("-" * 72)
    add(f"  {'pair':<38}{'delta':>9}{'naive p<.05?':>14}{'corrected p':>13}{'corrected CI excl. 0?':>24}")
    for model_a, model_b, result in pair_results:
        naive_significant = not (
            result["naive_ci"][0] <= 0.0 <= result["naive_ci"][1]
        )
        pair = f"{model_a} vs {model_b}"
        add(
            f"  {pair:<38}{result['mean_delta']:>+9.3f}"
            f"{str(naive_significant):>14}{result['corrected_p']:>13.3f}"
            f"{str(not result['corrected_crosses_zero']):>24}"
        )
    add("")

    add("READ")
    add("-" * 72)
    flips = [
        (a, b)
        for a, b, result in pair_results
        if not (result["naive_ci"][0] <= 0.0 <= result["naive_ci"][1])
        and result["corrected_crosses_zero"]
    ]
    if flips:
        pairs = ", ".join(f"{a} vs {b}" for a, b in flips)
        add(f"  Correction FLIPS the conclusion (naive: real gap -> corrected: noise) for: {pairs}.")
    else:
        add("  No pair flips from 'real gap' to 'noise' under the correction.")
    add(
        "  Point estimates (mean macro-F1) are identical to what run_improve.py "
        "already reports - this only widens uncertainty, never the score itself."
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_FACTORIES), default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "nadeau_bengio_out")
    args = parser.parse_args()

    frame = load_frame_from_csv(args.ground_truth_csv)
    print(f"Ground truth source: {args.ground_truth_csv} ({len(frame)} rows)")
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    label_counts = frame["ground_truth_label"].value_counts().to_dict()

    per_repeat_by_model: dict[str, pd.DataFrame] = {}
    oof_by_model: dict[str, np.ndarray] = {}
    for name in args.models:
        print(f"\n[{name}] {args.repeats} repeats x 5-fold CV ...", flush=True)
        per_repeat, oof = run_repeated_cv(
            MODEL_FACTORIES[name], term_counts, y, stopwords, n_repeats=args.repeats
        )
        per_repeat_by_model[name] = per_repeat
        oof_by_model[name] = oof
        summary = summarize(per_repeat)["macro_f1"]
        print(f"    macro_f1 {summary[0]:.3f} +/- {summary[1]:.3f} (repeat-level, naive)")

    ci_by_model = {
        name: nadeau_bengio_ci(np.asarray(y), oof_by_model[name]) for name in args.models
    }

    pair_results: list[tuple[str, str, dict]] = []
    for i in range(len(args.models)):
        for j in range(i + 1, len(args.models)):
            model_a, model_b = args.models[i], args.models[j]
            result = nadeau_bengio_paired_test(
                np.asarray(y), oof_by_model[model_a], oof_by_model[model_b]
            )
            pair_results.append((model_a, model_b, result))

    report = render_report(
        len(frame), label_counts, args.repeats, per_repeat_by_model, ci_by_model, pair_results
    )
    print("\n" + report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "RESULTS.txt"
    results_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten: {results_path}")


if __name__ == "__main__":
    main()
