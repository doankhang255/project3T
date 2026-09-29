"""3-branch comparison on the SHARED 314-row Holdout, using the PhoBERT
DISTILLATION model (N=20.000 unlabeled, pilot: Tune-trained, Holdout-checked
once) instead of the plain E3 fine-tune that compare_branches_holdout.py
uses.

This is a DIFFERENT question from compare_branches_holdout.py's ("which of
the 3 branches wins" was already answered there for E3 pre-distillation) -
the question here is specifically "does replacing E3 with the distillation
model change the 3-branch ranking / does distillation's improvement hold up
under a paired, honest comparison instead of just a bare macro-F1 number".
Looking once at this Holdout for THIS question is still within the "one
look per independent question" discipline both other branches document.

Inputs (all read-only):
  Traditional_ML     - experiment_Lexicon_features/holdout_once/
                       random_forest_holdout_predictions.csv (same as
                       compare_branches_holdout.py)
  Transfer_Learning  - improve/data/distill_holdout_predictions.csv
                       (distillation, N=20000 unlabeled, pilot run:
                       Tune-trained, Holdout-checked once - produced by
                       distill_phobert.py --n-unlabeled 20000 --skip-cv)
  Lexicon_based      - same reproduction as compare_branches_holdout.py
                       (Scoring/data/article_scores.parquet, Cach 1 / PMI)

IMPORTANT - same Holdout re-use caveat as compare_branches_holdout.py: these
314 rows have already been looked at multiple times for other questions
(lexicon-feature promotion, each branch's own Holdout number, the E3 3-way
comparison, and choosing N=20000 over N=5000 for the distillation pilot
itself - see PROJECT_SUMMARY.qmd's note that N=20000 was picked AFTER
seeing N=5000's Holdout score). This script's own comparison is therefore
NOT a pristine first look at distillation's true generalization - it is an
honest paired comparison of numbers that were already going to be reported
anyway, just now with a CI and a paired test attached instead of a bare
point estimate. Do not re-run this with a different N or a different
distillation config.

    python News/Build_sentiment_label/Traditional_ML/compare_branches_holdout_distillation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.bootstrap import (
    bootstrap_samples,
    ci,
    two_sided_p,
)
from News.Build_sentiment_label.Traditional_ML.Common.mcnemar import mcnemar_test
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    compute_metrics,
    encode_labels,
    normalize_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
LEXICON_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based"
TRANSFER_LEARNING_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Transfer_Learning"

RF_PREDICTIONS_PATH = (
    SCRIPT_DIR / "experiment_Lexicon_features" / "holdout_once" / "random_forest_holdout_predictions.csv"
)
DISTILL_PREDICTIONS_PATH = TRANSFER_LEARNING_DIR / "improve" / "data" / "distill_holdout_predictions.csv"
LEXICON_SCORES_PATH = LEXICON_DIR / "Scoring" / "data" / "article_scores.parquet"
SPLIT_PATH = LEXICON_DIR / "data" / "ground_truth_tune_holdout_split.csv"
GROUND_TRUTH_COMBINED_PATH = PROJECT_ROOT / "data_news" / "ground_truth_combined.csv"

RESULTS_PATH = SCRIPT_DIR / "compare_branches_holdout_distillation_RESULTS.txt"

BRANCH_ORDER = ["lexicon_based", "traditional_ml", "transfer_learning"]
BRANCH_LABELS = {
    "lexicon_based": "Lexicon-Based (Cach 1 PMI, longest-match)",
    "traditional_ml": "Traditional ML (Random Forest +lexicon, isotonic)",
    "transfer_learning": "Transfer Learning (PhoBERT distillation, N=20000, pilot)",
}
PAIRS = [
    ("transfer_learning", "traditional_ml"),
    ("transfer_learning", "lexicon_based"),
    ("traditional_ml", "lexicon_based"),
]


def load_random_forest_holdout() -> pd.DataFrame:
    if not RF_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"{RF_PREDICTIONS_PATH} not found.")
    df = pd.read_csv(RF_PREDICTIONS_PATH, encoding="utf-8-sig")
    df["ground_truth_label"] = df["ground_truth_label"].apply(normalize_label)
    df["predicted_label"] = df["predicted_label"].apply(normalize_label)
    return df[["source_row_id", "ground_truth_label", "predicted_label"]]


def load_distillation_holdout() -> pd.DataFrame:
    if not DISTILL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{DISTILL_PREDICTIONS_PATH} not found - run distill_phobert.py "
            "--n-unlabeled 20000 --skip-cv first (GPU venv)."
        )
    df = pd.read_csv(DISTILL_PREDICTIONS_PATH, encoding="utf-8-sig")
    df["ground_truth_label"] = df["ground_truth_label"].apply(normalize_label)
    df["predicted_label"] = df["predicted_label"].apply(normalize_label)
    return df[["source_row_id", "ground_truth_label", "predicted_label"]]


def load_lexicon_holdout() -> pd.DataFrame:
    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    holdout_ids = split_df.loc[split_df["split"] == "holdout", "source_row_id"].astype(int)

    ground_truth = pd.read_csv(GROUND_TRUTH_COMBINED_PATH, encoding="utf-8-sig")[
        ["source_row_id", "sentiment"]
    ].copy()
    ground_truth["source_row_id"] = ground_truth["source_row_id"].astype(int)

    scores = pd.read_parquet(
        LEXICON_SCORES_PATH, columns=["positive_score", "negative_score"]
    ).reset_index(drop=True)
    scores["source_row_id"] = scores.index
    scores["predicted_label"] = np.select(
        [
            scores["positive_score"] > scores["negative_score"],
            scores["positive_score"] < scores["negative_score"],
        ],
        ["positive", "negative"],
        default="neutral",
    )

    merged = ground_truth.merge(
        scores[["source_row_id", "predicted_label"]], on="source_row_id", how="inner"
    )
    merged = merged[merged["source_row_id"].isin(holdout_ids)].reset_index(drop=True)
    merged["ground_truth_label"] = merged["sentiment"].apply(normalize_label)
    return merged[["source_row_id", "ground_truth_label", "predicted_label"]]


def build_joined_frame() -> pd.DataFrame:
    loaders = {
        "traditional_ml": load_random_forest_holdout,
        "transfer_learning": load_distillation_holdout,
        "lexicon_based": load_lexicon_holdout,
    }

    merged: pd.DataFrame | None = None
    for name in BRANCH_ORDER:
        frame = loaders[name]().rename(
            columns={
                "ground_truth_label": f"ground_truth_label__{name}",
                "predicted_label": f"predicted_label__{name}",
            }
        )
        merged = frame if merged is None else merged.merge(
            frame, on="source_row_id", how="inner", validate="one_to_one"
        )
    assert merged is not None

    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    expected_rows = int((split_df["split"] == "holdout").sum())
    if len(merged) != expected_rows:
        raise AssertionError(
            f"Expected the shared {expected_rows}-row Holdout to join across all 3 "
            f"branches with no drops; got {len(merged)} rows instead."
        )

    gt_columns = [f"ground_truth_label__{name}" for name in BRANCH_ORDER]
    disagreement = merged[gt_columns].nunique(axis=1) != 1
    if disagreement.any():
        raise AssertionError(
            "Ground-truth label disagrees across branches for some rows:\n"
            f"{merged.loc[disagreement, ['source_row_id', *gt_columns]]}"
        )

    return merged


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    merged = build_joined_frame()
    n_rows = len(merged)
    y_true = encode_labels(merged[f"ground_truth_label__{BRANCH_ORDER[0]}"])

    oof_by_model = {
        name: encode_labels(merged[f"predicted_label__{name}"])[None, :]
        for name in BRANCH_ORDER
    }

    point = {}
    for name in BRANCH_ORDER:
        overall = compute_metrics(y_true, oof_by_model[name][0])
        overall = overall.loc[overall["metric_scope"].eq("overall")].iloc[0]
        point[name] = {"macro_f1": float(overall["f1"]), "accuracy": float(overall["accuracy"])}

    samples = bootstrap_samples(y_true, oof_by_model)
    branch_ci = {name: ci(samples[name]) for name in BRANCH_ORDER}

    pair_results = []
    for a, b in PAIRS:
        delta_sample = samples[a] - samples[b]
        delta_ci = ci(delta_sample)
        p_value = two_sided_p(delta_sample)
        mcnemar = mcnemar_test(y_true, oof_by_model[a][0], oof_by_model[b][0])
        pair_results.append(
            {
                "a": a,
                "b": b,
                "delta_macro_f1": point[a]["macro_f1"] - point[b]["macro_f1"],
                "ci_low": delta_ci[0],
                "ci_high": delta_ci[1],
                "p_bootstrap": p_value,
                "mcnemar_b": mcnemar["b"],
                "mcnemar_c": mcnemar["c"],
                "mcnemar_p": mcnemar["p_value"],
                "mcnemar_method": mcnemar["method"],
            }
        )

    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(f"SO SANH 3 NHANH TREN HOLDOUT DUNG CHUNG - BAN DISTILLATION ({n_rows} bai, kiem tra 1 lan)")
    add("=" * 72)
    add()
    add(
        f"Ca 3 nhanh dung CHUNG dung {n_rows} dong holdout. Transfer_Learning o day "
        "la ban DISTILLATION (N=20000 unlabeled, pilot: train tren Tune 730 dong + "
        "unlabeled, check Holdout dung 1 lan) - KHAC voi compare_branches_holdout.py "
        "(dung ban E3 thuan-supervised)."
    )
    add()
    label_width = max(len(label) for label in BRANCH_LABELS.values()) + 2

    add(f"DIEM SO TREN HOLDOUT (n={n_rows})")
    add("-" * 72)
    add(f"{'Nhanh':<{label_width}}{'Macro F1':>12}{'Accuracy':>12}")
    for name in BRANCH_ORDER:
        add(
            f"{BRANCH_LABELS[name]:<{label_width}}{point[name]['macro_f1']:>12.4f}"
            f"{point[name]['accuracy']:>12.4f}"
        )
    add()
    add(f"BOOTSTRAP 95% CI TREN MACRO-F1 (B=2000, resample {n_rows} dong)")
    add("-" * 72)
    for name in BRANCH_ORDER:
        low, high = branch_ci[name]
        add(f"{BRANCH_LABELS[name]:<{label_width}}{point[name]['macro_f1']:.4f}  [{low:.4f}, {high:.4f}]")
    add()
    add("SO SANH TUNG CAP - BOOTSTRAP (paired, tren delta macro-F1) + McNEMAR (tren accuracy)")
    add("-" * 72)
    for row in pair_results:
        a_label, b_label = BRANCH_LABELS[row["a"]], BRANCH_LABELS[row["b"]]
        verdict = "loai 0" if row["ci_low"] > 0 or row["ci_high"] < 0 else "cham 0"
        add(f"{a_label}  vs  {b_label}")
        add(
            f"  Bootstrap: delta={row['delta_macro_f1']:+.4f}  "
            f"CI=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  "
            f"p={row['p_bootstrap']:.4f}  ({verdict})"
        )
        add(
            f"  McNemar ({row['mcnemar_method']}): b={row['mcnemar_b']} "
            f"c={row['mcnemar_c']}  p={row['mcnemar_p']:.4f}"
        )
        add()

    add("DOC KET QUA")
    add("-" * 72)
    ranked = sorted(BRANCH_ORDER, key=lambda name: point[name]["macro_f1"], reverse=True)
    add(
        "Xep hang single-point macro-F1: "
        + " > ".join(f"{BRANCH_LABELS[name]} ({point[name]['macro_f1']:.4f})" for name in ranked)
    )
    reliable_pairs = [
        f"{BRANCH_LABELS[row['a']]} vs {BRANCH_LABELS[row['b']]}"
        for row in pair_results
        if row["ci_low"] > 0 or row["ci_high"] < 0
    ]
    if reliable_pairs:
        add("Cap co CI loai 0 (khac biet dang tin, khong phai nhieu chia mau): " + "; ".join(reliable_pairs))
    else:
        add(
            f"KHONG cap nao co CI loai 0 o n={n_rows} - voi quy mo mau nay, ca 3 "
            "nhanh khong tach duoc mot cach dang tin khoi nhieu resample."
        )
    add()

    add("CANH BAO QUAN TRONG - HOLDOUT DA BI DUNG LAI NHIEU LAN, VA N=20000 DA CHON SAU KHI XEM HOLDOUT")
    add("-" * 72)
    add(
        f"{n_rows} dong holdout nay da duoc dung nhieu lan truoc do (lexicon-feature "
        "promotion trong Traditional_ML, so Holdout rieng cua tung nhanh, phep so "
        "sanh E3-vs-RF-vs-Lexicon trong compare_branches_holdout.py). Quan trong hon: "
        "N=20000 (thay vi N=5000 da thu truoc) duoc chon SAU KHI da xem ket qua "
        "Holdout cua N=5000 - nen ban than con so macro-F1 cua distillation o day "
        "co the lac quan hon kha nang tong quat hoa thuc te, khong chi la van de "
        "'holdout da dung lai' nhu 2 nhanh kia."
    )
    add(
        "Vi vay CI/p-value o day nen doc la 'so sanh trung thuc giua cac con so DA "
        "DINH CHAY nay voi nhau', khong phai la mot phep kiem dinh doc lap hoan toan "
        "cho cau hoi 'distillation N=20000 co thuc su tot hon RF/Lexicon khong'."
    )

    report = "\n".join(lines)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print("\nOutput report:", RESULTS_PATH)


if __name__ == "__main__":
    main()
