"""M2.1 + M2.2 + M2.3 + M2.4 - compare all 5 models on TF-IDF features only
(no lexicon feature block - see ../experiment_Lexicon_features/ for that).

    python News/Build_sentiment_label/Traditional_ML/experiment_only_TF_IDF/compare_models_tfidf_only.py
    python .../compare_models_tfidf_only.py --repeats 3        # quick smoke run

    # retest on a different ground-truth CSV without touching this folder's own report:
    python .../compare_models_tfidf_only.py \\
        --ground-truth-csv data_news/ground_truth_combined.csv \\
        --out-dir News/Build_sentiment_label/Traditional_ML/experiment_only_TF_IDF/gt_custom

By default writes only into this folder (``RESULTS.txt`` + ``data/*.csv``).
Does not touch the main pipeline or model/*.py.

M2.1  Multinomial vs Complement Naive Bayes. Rennie et al. (2003): Complement
      NB is built for class-imbalanced text; here ``positive`` is ~25% of rows.
M2.2  Repeated stratified 5-fold CV -> mean +/- std over ``--repeats`` runs,
      so a ~0.02 macro-F1 gap is visibly inside the noise band.
M2.3  McNemar's test between every model pair on the out-of-fold predictions
      -> is model A really better than model B, or is it the split?
M2.4  Bootstrap 95% CI on macro-F1 and on every pairwise delta(macro-F1).
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

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import (
    build_document_term_counts,
)
from News.Build_sentiment_label.Traditional_ML.Common.bootstrap import (
    N_BOOT,
    bootstrap_samples,
    ci,
    mean_macro_f1,
    two_sided_p,
)
from News.Build_sentiment_label.Traditional_ML.Common.mcnemar import mcnemar_test
from News.Build_sentiment_label.Traditional_ML.Common.model_factories import (
    MODEL_FACTORIES,
    SINGLE_RUN_REFERENCE,
    summarize,
)
from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
    load_frame_from_csv,
)
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    N_REPEATS,
    load_stopword_set,
    run_repeated_cv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    encode_labels,
    load_ground_truth_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
# default output location; --out-dir overrides both (see main())


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


def bootstrap_delta_pairwise(
    model_names: list[str],
    boot_samples: dict[str, np.ndarray],
    point_by_model: dict[str, float],
) -> pd.DataFrame:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) for every model pair.

    ``boot_samples[name]`` are the resampled macro-F1 values from a shared set
    of row resamples, so ``boot_samples[a] - boot_samples[b]`` is the gap under
    the same resample. A CI that straddles 0 means the ranking is not
    distinguishable from split noise.
    """
    rows = []
    for first_index in range(len(model_names)):
        for second_index in range(first_index + 1, len(model_names)):
            model_a = model_names[first_index]
            model_b = model_names[second_index]
            delta_sample = boot_samples[model_a] - boot_samples[model_b]
            ci_low, ci_high = ci(delta_sample)
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_macro_f1": point_by_model[model_a] - point_by_model[model_b],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": two_sided_p(delta_sample),
                    "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
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
    boot_point_by_model: dict[str, float],
    boot_ci_by_model: dict[str, tuple[float, float]],
    boot_delta_df: pd.DataFrame,
    n_repeats: int,
    n_boot: int,
    label_counts: dict[str, int],
) -> str:
    lines: list[str] = []
    add = lines.append

    total = sum(label_counts.values())
    add("COMPARE MODELS - TF-IDF ONLY  (M2.1 / M2.2 / M2.3 / M2.4)")
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
        "fold (leak-free). No lexicon feature block (see "
        "../experiment_Lexicon_features/ for that)."
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
    add(
        f"  95% bootstrap CI on mean macro-F1 (B={n_boot}, resample the {total} rows):"
    )
    for name in ranked:
        low, high = boot_ci_by_model[name]
        add(f"    {name:<20} {boot_point_by_model[name]:.3f}  [{low:.3f}, {high:.3f}]")
    add("")
    add("  RESULTS_SUMMARY.txt reports the production pipeline (random_forest")
    add("  there uses the lexicon feature block; the random_forest here does not -")
    add("  the two are not directly comparable).")
    reference_to_repeat = {"naive_bayes": "multinomial_nb"}
    for name, (macro_f1, _accuracy) in SINGLE_RUN_REFERENCE.items():
        repeat_name = reference_to_repeat.get(name, name)
        repeat_mean = summary_by_model.get(repeat_name, {}).get(
            "macro_f1", (float("nan"), 0.0)
        )[0]
        add(
            f"    {name:<20} single {macro_f1:.3f}   repeated mean {repeat_mean:.3f}"
        )
    add("")

    have_nb_pair = "multinomial_nb" in summary_by_model and "complement_nb" in summary_by_model
    delta_macro = None
    inside = None
    if have_nb_pair:
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

    add(
        f"M2.4  BOOTSTRAP  -  95% CI on delta(macro-F1) between models "
        f"(paired, B={n_boot})"
    )
    add("-" * 68)
    add(f"  {'pair':<40}{'d_macroF1':>11}{'  95% CI':<20}{'p':>7}")
    for row in boot_delta_df.itertuples(index=False):
        pair = f"{row.model_a} vs {row.model_b}"
        ci_text = f"  [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
        add(
            f"  {pair:<40}{row.delta_macro_f1:>+11.3f}{ci_text:<20}{row.p_value:>7.3f}"
            f"{'   overlaps 0' if row.crosses_zero else '   EXCLUDES 0'}"
        )
    add("")
    add("  delta = (row1 mean macro-F1) - (row2). McNemar (M2.3) tests the")
    add("  accuracy gap; this tests the macro-F1 gap, which is the ranking")
    add("  metric. CI straddling 0 -> gap not distinguishable from split noise.")
    add("")

    add("READ")
    add("-" * 68)
    best = ranked[0]
    best_macro = summary_by_model[best]["macro_f1"]
    best_low, best_high = boot_ci_by_model[best]
    add(
        f"  Best mean macro-F1: {best} {best_macro[0]:.3f} +/- {best_macro[1]:.3f} "
        f"(95% CI [{best_low:.3f}, {best_high:.3f}])."
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
    if delta_macro is not None:
        add(
            f"  delta(complement - multinomial) = {delta_macro:+.3f} macro-F1; "
            f"{'within' if inside else 'beyond'} one std."
        )
    excludes_zero = boot_delta_df.loc[~boot_delta_df["crosses_zero"], ["model_a", "model_b"]]
    if excludes_zero.empty:
        add(
            "  Bootstrap: every model-pair macro-F1 gap CI straddles 0 - no "
            "ranking survives at this sample size."
        )
    else:
        pairs = ", ".join(
            f"{a} vs {b}" for a, b in excludes_zero.itertuples(index=False)
        )
        add(f"  Bootstrap: macro-F1 gap CI excludes 0 only for: {pairs}.")
    add(
        f"  With {total} rows, check whether gaps still sit inside the +/- std "
        "band before locking in a model choice (see ../IMPROVEMENTS.md)."
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
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help=f"bootstrap resamples for the macro-F1 CI (default {N_BOOT})",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help=(
            "retest against a different ground-truth CSV (same schema as "
            "data_news/ground_truth_labeled.csv, joined to the VNCoreNLP "
            "corpus by source_row_id) instead of the pipeline's committed "
            "tokenized parquet"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write RESULTS.txt + data/ (default: this folder)",
    )
    args = parser.parse_args()

    if args.ground_truth_csv is not None:
        frame = load_frame_from_csv(args.ground_truth_csv)
        print(f"Ground truth source: {args.ground_truth_csv}")
    else:
        frame = load_ground_truth_frame()
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    label_counts = frame["ground_truth_label"].value_counts().to_dict()

    out_dir = args.out_dir if args.out_dir is not None else SCRIPT_DIR
    data_dir = out_dir / "data"
    results_path = out_dir / "RESULTS.txt"

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

    print(f"\n[bootstrap] {args.n_boot} paired resamples of {len(frame)} rows ...", flush=True)
    boot_sample_by_model = bootstrap_samples(
        np.asarray(y), oof_by_model, n_boot=args.n_boot
    )
    boot_point_by_model = {
        name: mean_macro_f1(np.asarray(y), oof_by_model[name]) for name in args.models
    }
    boot_ci_by_model = {
        name: ci(boot_sample_by_model[name]) for name in args.models
    }
    boot_delta_df = bootstrap_delta_pairwise(
        list(args.models), boot_sample_by_model, boot_point_by_model
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    per_repeat_all = pd.concat(per_repeat_by_model.values(), ignore_index=True)
    per_repeat_all.to_csv(
        data_dir / "repeated_cv_per_repeat.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for name, stats in summary_by_model.items():
        row: dict[str, object] = {"model": name}
        for metric, (mean, std) in stats.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        data_dir / "repeated_cv_summary.csv", index=False, encoding="utf-8-sig"
    )
    mcnemar_df.to_csv(
        data_dir / "mcnemar_pairwise.csv", index=False, encoding="utf-8-sig"
    )

    boot_ci_rows = [
        {
            "model": name,
            "macro_f1_point": boot_point_by_model[name],
            "ci_low": boot_ci_by_model[name][0],
            "ci_high": boot_ci_by_model[name][1],
        }
        for name in args.models
    ]
    pd.DataFrame(boot_ci_rows).to_csv(
        data_dir / "bootstrap_macro_f1_ci.csv", index=False, encoding="utf-8-sig"
    )
    boot_delta_df.to_csv(
        data_dir / "bootstrap_delta.csv", index=False, encoding="utf-8-sig"
    )

    report = render_report(
        summary_by_model,
        per_repeat_by_model,
        mcnemar_df,
        boot_point_by_model,
        boot_ci_by_model,
        boot_delta_df,
        args.repeats,
        args.n_boot,
        label_counts,
    )
    results_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nWritten: {results_path}")
    for csv_name in (
        "repeated_cv_per_repeat.csv",
        "repeated_cv_summary.csv",
        "mcnemar_pairwise.csv",
        "bootstrap_macro_f1_ci.csv",
        "bootstrap_delta.csv",
    ):
        print(f"         {data_dir / csv_name}")


if __name__ == "__main__":
    main()
