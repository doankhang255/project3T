"""Experiment - do financial-lexicon category features + negation handling
beat the TF-IDF-only baseline? (IMPROVEMENTS.md priority 1; Loughran &
McDonald 2011, Tetlock 2007.)

Self-contained: imports pure helpers from ``../Common`` (TF-IDF fit/transform,
the leak-free repeated-CV harness, the bootstrap CI) and adds only the
lexicon-feature block. Does not edit ``Common/model/*.py`` or
``Common/model/common.py``. The one change made to shared code
(``Common/repeated_cv.py``) is an optional ``extra_features`` parameter,
default ``None`` - every existing caller is unaffected (verified: reproduces
the pre-change out-of-fold predictions exactly).

Runs on the TUNE split by default (see ``../Common/tune_holdout.py``) - this
is exploratory feature engineering, not the final number for the holdout.

    python News/Build_sentiment_label/Traditional_ML/experiment_Lexicon_features/compare_baseline_vs_lexicon.py
    python .../compare_baseline_vs_lexicon.py --repeats 3 --n-boot 400   # quick smoke run

MultinomialNB / ComplementNB are excluded from the "+lexicon" arm: both
require non-negative input, and ``net_polarity`` can be negative.
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
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    FEATURE_NAMES,
    build_lexicon_feature_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.bootstrap import (
    N_BOOT,
    bootstrap_samples,
    ci,
    mean_macro_f1,
    two_sided_p,
)
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    N_REPEATS,
    load_stopword_set,
    run_repeated_cv,
)
from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
    load_frame_from_csv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.random_forest import (
    build_estimator as build_random_forest,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.svm import (
    build_estimator as build_svm,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_CSV = (
    SCRIPT_DIR.parent / "Common" / "tune_holdout" / "ground_truth_tune.csv"
)

MODEL_FACTORIES = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "svm": build_svm,
}
METRIC_COLUMNS = ["macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"]


def load_frame(ground_truth_csv: Path | None) -> pd.DataFrame:
    if ground_truth_csv is not None:
        return load_frame_from_csv(ground_truth_csv)
    return load_ground_truth_frame()


def summarize(per_repeat: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        column: (float(per_repeat[column].mean()), float(per_repeat[column].std(ddof=1)))
        for column in METRIC_COLUMNS
    }


def _fmt(mean_std: tuple[float, float]) -> str:
    return f"{mean_std[0]:.3f} +/- {mean_std[1]:.3f}"


def render_report(
    ground_truth_csv: Path,
    n_rows: int,
    label_counts: dict[str, int],
    n_repeats: int,
    n_boot: int,
    summary: dict[str, dict[str, dict[str, tuple[float, float]]]],
    delta_rows: list[dict],
) -> str:
    lines: list[str] = []
    add = lines.append

    add("EXPERIMENT - lexicon-category features + negation vs TF-IDF-only baseline")
    add("=" * 72)
    add("")
    add(f"Ground truth: {ground_truth_csv}")
    add(
        f"  {n_rows} rows (neg {label_counts.get('negative', 0)} / "
        f"neu {label_counts.get('neutral', 0)} / pos {label_counts.get('positive', 0)})"
    )
    add(
        f"Repeated stratified 5-fold CV, {n_repeats} repeats, {n_boot} bootstrap "
        "resamples. TF-IDF fit per fold (leak-free); lexicon features are fixed "
        "word lists (Seed_set_Prepare + Loughran & McDonald 2011 negation rule), "
        "not fit from data, so adding them leaks nothing."
    )
    add(f"Feature columns added: {', '.join(FEATURE_NAMES)}")
    add("NB family (Multinomial/Complement) skipped: net_polarity can be negative.")
    add("")

    add("MACRO-F1: baseline (TF-IDF only) vs +lexicon")
    add("-" * 72)
    add(f"  {'model':<22}{'baseline':>18}{'+lexicon':>18}{'delta':>10}")
    for name in MODEL_FACTORIES:
        base = summary[name]["baseline"]["macro_f1"]
        lex = summary[name]["lexicon"]["macro_f1"]
        add(
            f"  {name:<22}{_fmt(base):>18}{_fmt(lex):>18}"
            f"{lex[0] - base[0]:>+10.3f}"
        )
    add("")

    add("PER-CLASS F1 (+lexicon arm)")
    add("-" * 72)
    add(f"  {'model':<22}{'f1_negative':>18}{'f1_neutral':>18}{'f1_positive':>18}")
    for name in MODEL_FACTORIES:
        lex = summary[name]["lexicon"]
        add(
            f"  {name:<22}{_fmt(lex['f1_negative']):>18}"
            f"{_fmt(lex['f1_neutral']):>18}{_fmt(lex['f1_positive']):>18}"
        )
    add("")

    add("BOOTSTRAP  -  95% CI on delta(macro-F1) = lexicon - baseline, paired, per model")
    add("-" * 72)
    add(f"  {'model':<22}{'delta':>9}  {'95% CI':<20}{'p':>8}  verdict")
    for row in delta_rows:
        ci_text = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        verdict = "EXCLUDES 0" if not row["crosses_zero"] else "overlaps 0"
        add(
            f"  {row['model']:<22}{row['delta']:>+9.3f}  {ci_text:<20}"
            f"{row['p_value']:>8.3f}  {verdict}"
        )
    add("")
    add("  delta > 0 and CI excludes 0 -> lexicon features reliably help this model.")
    add("  CI straddling 0 -> not distinguishable from split/resample noise.")
    add("")

    add("READ")
    add("-" * 72)
    helped = [row["model"] for row in delta_rows if not row["crosses_zero"] and row["delta"] > 0]
    hurt = [row["model"] for row in delta_rows if not row["crosses_zero"] and row["delta"] < 0]
    if helped:
        add(f"  Reliably helped: {', '.join(helped)}.")
    if hurt:
        add(f"  Reliably hurt: {', '.join(hurt)}.")
    if not helped and not hurt:
        add("  No model shows a reliable change - lexicon features are not distinguishable")
        add("  from noise here. Do not promote into Common/model/*.py yet.")
    add(
        "  This ran on the TUNE split only. If promoting, re-check once on the "
        "HOLDOUT split (../Common/tune_holdout.py) - exactly once, not iteratively."
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help="defaults to the tune split (../Common/tune_holdout.py output)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="where to write RESULTS.txt + data/ (default: this folder) - "
        "override when running against the holdout split so it does not "
        "overwrite the tune-split RESULTS.txt",
    )
    args = parser.parse_args()
    data_dir = args.out_dir / "data"
    results_path = args.out_dir / "RESULTS.txt"

    frame = load_frame(args.ground_truth_csv)
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    lexicon_matrix = build_lexicon_feature_matrix(frame["Tokenize_content"].tolist())
    label_counts = frame["ground_truth_label"].value_counts().to_dict()

    print(f"Ground truth: {args.ground_truth_csv} ({len(frame)} rows)")
    print("Label counts:", label_counts)
    print(f"Lexicon feature matrix: {lexicon_matrix.shape}")

    summary: dict[str, dict[str, dict]] = {}
    oof: dict[str, dict[str, np.ndarray]] = {}
    per_repeat_frames: list[pd.DataFrame] = []

    for name, factory in MODEL_FACTORIES.items():
        print(f"\n[{name}] baseline ({args.repeats} repeats x 5-fold CV) ...", flush=True)
        base_repeat, base_oof = run_repeated_cv(factory, term_counts, y, stopwords, n_repeats=args.repeats)
        print(f"[{name}] +lexicon ({args.repeats} repeats x 5-fold CV) ...", flush=True)
        lex_repeat, lex_oof = run_repeated_cv(
            factory, term_counts, y, stopwords, n_repeats=args.repeats, extra_features=lexicon_matrix
        )
        summary[name] = {"baseline": summarize(base_repeat), "lexicon": summarize(lex_repeat)}
        oof[name] = {"baseline": base_oof, "lexicon": lex_oof}
        for arm_name, repeat_df in (("baseline", base_repeat), ("lexicon", lex_repeat)):
            tagged = repeat_df.copy()
            tagged.insert(0, "arm", arm_name)
            tagged.insert(0, "model", name)
            per_repeat_frames.append(tagged)
        print(
            f"    baseline macro_f1 {summary[name]['baseline']['macro_f1'][0]:.3f} +/- "
            f"{summary[name]['baseline']['macro_f1'][1]:.3f}   "
            f"+lexicon macro_f1 {summary[name]['lexicon']['macro_f1'][0]:.3f} +/- "
            f"{summary[name]['lexicon']['macro_f1'][1]:.3f}"
        )

    delta_rows: list[dict] = []
    for name in MODEL_FACTORIES:
        arm_oof = {"baseline": oof[name]["baseline"], "lexicon": oof[name]["lexicon"]}
        samples = bootstrap_samples(np.asarray(y), arm_oof, n_boot=args.n_boot)
        delta_sample = samples["lexicon"] - samples["baseline"]
        point_base = mean_macro_f1(np.asarray(y), oof[name]["baseline"])
        point_lex = mean_macro_f1(np.asarray(y), oof[name]["lexicon"])
        ci_low, ci_high = ci(delta_sample)
        delta_rows.append(
            {
                "model": name,
                "delta": point_lex - point_base,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": two_sided_p(delta_sample),
                "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
            }
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_repeat_frames, ignore_index=True).to_csv(
        data_dir / "per_repeat.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        data_dir / "bootstrap_delta.csv", index=False, encoding="utf-8-sig"
    )

    report = render_report(
        args.ground_truth_csv,
        len(frame),
        label_counts,
        args.repeats,
        args.n_boot,
        summary,
        delta_rows,
    )
    results_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nWritten: {results_path}")
    print(f"         {data_dir / 'per_repeat.csv'}")
    print(f"         {data_dir / 'bootstrap_delta.csv'}")


if __name__ == "__main__":
    main()
