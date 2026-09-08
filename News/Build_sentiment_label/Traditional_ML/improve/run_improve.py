"""M2.1 + M2.2 + M2.3 for the Traditional_ML branch, run together.

    python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py
    python .../improve/run_improve.py --repeats 3        # quick smoke run

Writes only into ``improve/`` (``RESULTS.txt`` + ``data/*.csv``). Does not
touch the main pipeline. Once a change here is confirmed useful, fold it into
``model/*.py`` and regenerate ``RESULTS_SUMMARY.txt``.

M2.1  Multinomial vs Complement Naive Bayes. Rennie et al. (2003): Complement
      NB is built for class-imbalanced text; here ``positive`` is 23% of rows.
M2.2  Repeated stratified 5-fold CV -> mean +/- std over ``--repeats`` runs,
      so a ~0.02 macro-F1 gap is visibly inside the noise band.
M2.3  McNemar's test between every model pair on the out-of-fold predictions
      -> is model A really better than model B, or is it the split?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.improve.mcnemar import mcnemar_test
from News.Build_sentiment_label.Traditional_ML.improve.repeated_cv import (
    N_REPEATS,
    load_stopword_set,
    run_repeated_cv,
)
from News.Build_sentiment_label.Traditional_ML.model.common import (
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.model.naive_bayes import (
    build_estimator as build_multinomial_nb,
)
from News.Build_sentiment_label.Traditional_ML.model.random_forest import (
    build_estimator as build_random_forest,
)
from News.Build_sentiment_label.Traditional_ML.model.svm import (
    build_estimator as build_svm,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_PATH = SCRIPT_DIR / "RESULTS.txt"

METRIC_COLUMNS = ["macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"]

# Single-split numbers from the committed RESULTS_SUMMARY.txt, shown for
# reference only (one fold seed, MultinomialNB).
SINGLE_RUN_REFERENCE = {
    "random_forest": (0.634, 0.658),
    "naive_bayes": (0.605, 0.632),
    "logistic_regression": (0.593, 0.632),
    "svm": (0.562, 0.638),
}


def build_complement_nb(random_state: int) -> ComplementNB:
    del random_state  # ComplementNB has no randomness; uniform factory signature
    return ComplementNB()


MODEL_FACTORIES = {
    "logistic_regression": build_logistic_regression,
    "multinomial_nb": build_multinomial_nb,
    "complement_nb": build_complement_nb,
    "random_forest": build_random_forest,
    "svm": build_svm,
}


def summarize(per_repeat: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        column: (
            float(per_repeat[column].mean()),
            float(per_repeat[column].std(ddof=1)),
        )
        for column in METRIC_COLUMNS
    }


def mcnemar_pairwise(
    model_names: list[str],
    y: np.ndarray,
    oof_by_model: dict[str, np.ndarray],
    n_repeats: int,
) -> pd.DataFrame:
    rows = []
    for first_index in range(len(model_names)):
        for second_index in range(first_index + 1, len(model_names)):
            model_a = model_names[first_index]
            model_b = model_names[second_index]
            per_repeat = [
                mcnemar_test(
                    y, oof_by_model[model_a][repeat], oof_by_model[model_b][repeat]
                )
                for repeat in range(n_repeats)
            ]
            b_counts = np.array([result["b"] for result in per_repeat])
            c_counts = np.array([result["c"] for result in per_repeat])
            p_values = np.array([result["p_value"] for result in per_repeat])
            pooled = mcnemar_test(
                np.tile(y, n_repeats),
                oof_by_model[model_a].ravel(),
                oof_by_model[model_b].ravel(),
            )
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_b": float(b_counts.mean()),
                    "mean_c": float(c_counts.mean()),
                    "median_p": float(np.median(p_values)),
                    "sig_repeats": int(np.sum(p_values < 0.05)),
                    "n_repeats": n_repeats,
                    "pooled_b": int(pooled["b"]),
                    "pooled_c": int(pooled["c"]),
                    "pooled_p": float(pooled["p_value"]),
                    "pooled_method": pooled["method"],
                }
            )
    return pd.DataFrame(rows)


def _fmt_mean_std(mean_std: tuple[float, float]) -> str:
    mean, std = mean_std
    return f"{mean:.3f} +/- {std:.3f}"


def render_report(
    summary_by_model: dict[str, dict[str, tuple[float, float]]],
    per_repeat_by_model: dict[str, pd.DataFrame],
    mcnemar_df: pd.DataFrame,
    n_repeats: int,
    label_counts: dict[str, int],
) -> str:
    lines: list[str] = []
    add = lines.append

    total = sum(label_counts.values())
    add("IMPROVE - M2.1 / M2.2 / M2.3  (Traditional_ML methodology fixes)")
    add("=" * 68)
    add("")
    add(
        f"Ground truth: {total} rows "
        f"(neg {label_counts.get('negative', 0)} / "
        f"neu {label_counts.get('neutral', 0)} / "
        f"pos {label_counts.get('positive', 0)}) | tokenizer VNCoreNLP"
    )
    add(
        f"Repeated stratified 5-fold CV, {n_repeats} repeats, TF-IDF fit per "
        "fold (leak-free)."
    )
    add(
        "Repeat r uses fold seed r; every model sees the SAME split in repeat r "
        "(paired)."
    )
    add("")

    ranked = sorted(
        summary_by_model,
        key=lambda name: summary_by_model[name]["macro_f1"][0],
        reverse=True,
    )

    add(f"M2.2  REPEATED CV  -  mean +/- std over {n_repeats} repeats")
    add("-" * 68)
    add(f"  {'model':<20}{'macro_f1':>16}{'accuracy':>16}")
    for name in ranked:
        stats = summary_by_model[name]
        add(
            f"  {name:<20}{_fmt_mean_std(stats['macro_f1']):>16}"
            f"{_fmt_mean_std(stats['accuracy']):>16}"
        )
    add("")
    add(f"  {'model':<20}{'f1_negative':>16}{'f1_neutral':>16}{'f1_positive':>16}")
    for name in ranked:
        stats = summary_by_model[name]
        add(
            f"  {name:<20}{_fmt_mean_std(stats['f1_negative']):>16}"
            f"{_fmt_mean_std(stats['f1_neutral']):>16}"
            f"{_fmt_mean_std(stats['f1_positive']):>16}"
        )
    add("")
    add("  Single-split reference (RESULTS_SUMMARY.txt, one fold seed):")
    for name, (macro_f1, accuracy) in SINGLE_RUN_REFERENCE.items():
        add(f"    {name:<20} macro_f1 {macro_f1:.3f}   accuracy {accuracy:.3f}")
    add("")

    add("M2.1  MULTINOMIAL vs COMPLEMENT NAIVE BAYES")
    add("-" * 68)
    multinomial = summary_by_model["multinomial_nb"]
    complement = summary_by_model["complement_nb"]
    for name, stats in (("multinomial_nb", multinomial), ("complement_nb", complement)):
        add(
            f"  {name:<16} macro_f1 {_fmt_mean_std(stats['macro_f1'])}   "
            f"accuracy {_fmt_mean_std(stats['accuracy'])}"
        )
    delta_macro = complement["macro_f1"][0] - multinomial["macro_f1"][0]
    pooled_std = max(multinomial["macro_f1"][1], complement["macro_f1"][1])
    inside = abs(delta_macro) <= pooled_std
    add(
        f"  delta (complement - multinomial): {delta_macro:+.3f} macro_f1  "
        f"({'inside' if inside else 'outside'} one std -> "
        f"{'noise' if inside else 'possibly real'})"
    )
    nb_pair = mcnemar_df[
        (mcnemar_df["model_a"] == "multinomial_nb")
        & (mcnemar_df["model_b"] == "complement_nb")
    ]
    if not nb_pair.empty:
        row = nb_pair.iloc[0]
        add(
            f"  McNemar: mean_b(mnb right, cnb wrong)={row['mean_b']:.1f}  "
            f"mean_c={row['mean_c']:.1f}  median_p={row['median_p']:.3f}  "
            f"sig_repeats={row['sig_repeats']}/{row['n_repeats']}  "
            f"-> {'reliably different' if row['sig_repeats'] > row['n_repeats'] / 2 else 'not reliably different'}"
        )
    add("")

    add("M2.3  McNEMAR PAIRWISE  (are the model gaps real, or the split?)")
    add("-" * 68)
    add(
        f"  {'pair':<40}{'mean_b':>8}{'mean_c':>8}{'med_p':>8}"
        f"{'sig/N':>8}{'pool_p':>9}"
    )
    for row in mcnemar_df.itertuples(index=False):
        pair = f"{row.model_a} vs {row.model_b}"
        add(
            f"  {pair:<40}{row.mean_b:>8.1f}{row.mean_c:>8.1f}"
            f"{row.median_p:>8.3f}{f'{row.sig_repeats}/{row.n_repeats}':>8}"
            f"{row.pooled_p:>9.3f}"
        )
    add("")
    add("  mean_b = rows first model right & second wrong (averaged over repeats)")
    add("  sig/N  = repeats with p < 0.05 out of N; pool_p pools all repeats")
    add("           (optimistic - repeats are not independent)")
    add("")

    add("READ")
    add("-" * 68)
    best = ranked[0]
    best_macro = summary_by_model[best]["macro_f1"]
    add(
        f"  Best mean macro-F1: {best} {best_macro[0]:.3f} +/- {best_macro[1]:.3f}."
    )
    any_reliable = (mcnemar_df["sig_repeats"] > mcnemar_df["n_repeats"] / 2).any()
    if any_reliable:
        reliable = mcnemar_df.loc[
            mcnemar_df["sig_repeats"] > mcnemar_df["n_repeats"] / 2,
            ["model_a", "model_b"],
        ]
        pairs = ", ".join(f"{a} vs {b}" for a, b in reliable.itertuples(index=False))
        add(f"  Reliably different (majority of repeats significant): {pairs}.")
    else:
        add(
            "  No model pair is reliably different (no pair significant in a "
            "majority of repeats)."
        )
    add(
        f"  delta(complement - multinomial) = {delta_macro:+.3f} macro-F1; "
        f"{'within' if inside else 'beyond'} one std."
    )
    add(
        "  With 152 rows most gaps sit inside the +/- std band. Do not lock in "
        "a model choice; grow ground truth first (see ../IMPROVEMENTS.md)."
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help=f"number of repeated CV passes (default {N_REPEATS})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FACTORIES),
        default=list(MODEL_FACTORIES),
        help="subset of models to run (default: all)",
    )
    args = parser.parse_args()

    frame = load_ground_truth_frame()
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    label_counts = frame["ground_truth_label"].value_counts().to_dict()

    print(f"Ground truth rows: {len(frame)}  |  repeats: {args.repeats}")
    print("Label counts:", label_counts)

    per_repeat_by_model: dict[str, pd.DataFrame] = {}
    summary_by_model: dict[str, dict[str, tuple[float, float]]] = {}
    oof_by_model: dict[str, np.ndarray] = {}

    for name in args.models:
        print(f"\n[{name}] {args.repeats} repeats x 5-fold CV ...", flush=True)
        per_repeat, oof = run_repeated_cv(
            MODEL_FACTORIES[name], term_counts, y, stopwords, n_repeats=args.repeats
        )
        per_repeat.insert(0, "model", name)
        per_repeat_by_model[name] = per_repeat
        summary_by_model[name] = summarize(per_repeat)
        oof_by_model[name] = oof
        macro = summary_by_model[name]["macro_f1"]
        print(f"    macro_f1 {macro[0]:.3f} +/- {macro[1]:.3f}")

    mcnemar_df = mcnemar_pairwise(list(args.models), np.asarray(y), oof_by_model, args.repeats)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    per_repeat_all = pd.concat(per_repeat_by_model.values(), ignore_index=True)
    per_repeat_all.to_csv(
        DATA_DIR / "repeated_cv_per_repeat.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for name, stats in summary_by_model.items():
        row: dict[str, object] = {"model": name}
        for metric, (mean, std) in stats.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        DATA_DIR / "repeated_cv_summary.csv", index=False, encoding="utf-8-sig"
    )
    mcnemar_df.to_csv(
        DATA_DIR / "mcnemar_pairwise.csv", index=False, encoding="utf-8-sig"
    )

    report = render_report(
        summary_by_model, per_repeat_by_model, mcnemar_df, args.repeats, label_counts
    )
    RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nWritten: {RESULTS_PATH}")
    print(f"         {DATA_DIR / 'repeated_cv_per_repeat.csv'}")
    print(f"         {DATA_DIR / 'repeated_cv_summary.csv'}")
    print(f"         {DATA_DIR / 'mcnemar_pairwise.csv'}")


if __name__ == "__main__":
    main()
