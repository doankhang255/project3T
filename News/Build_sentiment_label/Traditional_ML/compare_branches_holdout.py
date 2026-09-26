"""3-branch comparison on the SHARED, once-only 314-row Holdout: Lexicon_based
vs Traditional_ML (this branch, Random Forest +lexicon) vs Transfer_Learning
(PhoBERT E3 fine-tune). Motivated by a review finding: RESULTS_SUMMARY.txt's
random_forest number is 10 x 5-fold CV on the full 1044 rows, not a
Holdout number, so comparing it directly against Lexicon_based's and
Transfer_Learning's genuine train(Tune)->predict(Holdout)-once numbers was
comparing apples to oranges. This script fixes that by using
``experiment_Lexicon_features/holdout_once.py``'s genuine Holdout-once Random
Forest predictions instead (run that script first if its output is missing).

All three branches share the exact same 314 Holdout rows (verified: identical
source_row_id sets), sourced from
``Lexicon_based/data/ground_truth_tune_holdout_split.csv`` - see
``Common/tune_holdout.py`` and
``Transfer_Learning/improve/repeated_cv_tune_holdout.py``'s docstrings, both
of which build off that same split file by design (mentor feedback: tune on
one part, report once on a part nobody iterated against).

Inputs (all read-only - nothing in Lexicon_based or Transfer_Learning is
modified by this script):

  Traditional_ML     - experiment_Lexicon_features/holdout_once/
                       random_forest_holdout_predictions.csv (this branch's
                       own output)
  Transfer_Learning  - improve/data/finetune_holdout_predictions.csv (E3
                       fine-tuned PhoBERT; already a genuine Holdout-once file)
  Lexicon_based      - Scoring/data/article_scores.parquet (Cach 1 / PMI,
                       longest-match - the config LEXICON_SUMMARY.qmd section
                       3 reports as production, and which that section shows
                       gives essentially the same Holdout number as Cach 2 /
                       intensity) + data_news/ground_truth_combined.csv for
                       the ground-truth label. Re-applies the same
                       positive_score-vs-negative_score rule
                       ``evaluate_lexicon_on_ground_truth.py`` already uses -
                       reproduced here (not imported) because that branch
                       exposes it as a script, not an importable function.
                       Verified below to reproduce the exact published
                       number (accuracy 0.6624, macro-F1 0.6500 -
                       LEXICON_SUMMARY.qmd section 3) before trusting it.

IMPORTANT - holdout re-use, read before citing these numbers anywhere: these
are the SAME 314 rows already spent once in this branch (ML_SUMMARY.qmd
section 6.2, deciding whether random_forest should use the lexicon feature
block) and already used for the Lexicon and PhoBERT Holdout numbers each
branch reports on its own. This script's 3-way ranking is a NEW question
(which branch does best?), so a first look at it here is still within the
"look once per independent question" discipline - but none of the three
numbers below should be read as an untouched, "fresh eyes" holdout check
anymore. Do not re-run this after changing any one branch's model and expect
the result to still mean the same thing.

    python News/Build_sentiment_label/Traditional_ML/compare_branches_holdout.py
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
PHOBERT_PREDICTIONS_PATH = TRANSFER_LEARNING_DIR / "improve" / "data" / "finetune_holdout_predictions.csv"
LEXICON_SCORES_PATH = LEXICON_DIR / "Scoring" / "data" / "article_scores.parquet"
SPLIT_PATH = LEXICON_DIR / "data" / "ground_truth_tune_holdout_split.csv"
GROUND_TRUTH_COMBINED_PATH = PROJECT_ROOT / "data_news" / "ground_truth_combined.csv"

RESULTS_PATH = SCRIPT_DIR / "compare_branches_holdout_RESULTS.txt"

BRANCH_ORDER = ["lexicon_based", "traditional_ml", "transfer_learning"]
BRANCH_LABELS = {
    "lexicon_based": "Lexicon-Based (Cach 1 PMI, longest-match)",
    "traditional_ml": "Traditional ML (Random Forest +lexicon, isotonic)",
    "transfer_learning": "Transfer Learning (PhoBERT E3, fine-tune toan phan)",
}
PAIRS = [
    ("transfer_learning", "traditional_ml"),
    ("transfer_learning", "lexicon_based"),
    ("traditional_ml", "lexicon_based"),
]


def load_random_forest_holdout() -> pd.DataFrame:
    if not RF_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{RF_PREDICTIONS_PATH} not found - run experiment_Lexicon_features/"
            "holdout_once.py first."
        )
    df = pd.read_csv(RF_PREDICTIONS_PATH, encoding="utf-8-sig")
    df["ground_truth_label"] = df["ground_truth_label"].apply(normalize_label)
    df["predicted_label"] = df["predicted_label"].apply(normalize_label)
    return df[["source_row_id", "ground_truth_label", "predicted_label"]]


def load_phobert_holdout() -> pd.DataFrame:
    df = pd.read_csv(PHOBERT_PREDICTIONS_PATH, encoding="utf-8-sig")
    df["ground_truth_label"] = df["ground_truth_label"].apply(normalize_label)
    df["predicted_label"] = df["predicted_label"].apply(normalize_label)
    return df[["source_row_id", "ground_truth_label", "predicted_label"]]


def load_lexicon_holdout() -> pd.DataFrame:
    """Re-derives Lexicon_based's Holdout predictions from its own committed,
    already-final score file (Cach 1 / PMI, longest-match) - same
    positive_score-vs-negative_score rule as
    ``Lexicon_based/evaluate_lexicon_on_ground_truth.py::assign_three_class_label``.
    Read-only: nothing in Lexicon_based is written or modified here.
    """
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
        "transfer_learning": load_phobert_holdout,
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
            "Ground-truth label disagrees across branches for some rows "
            f"(should be the same 3-class label everywhere):\n"
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

    add(f"SO SANH 3 NHANH TREN HOLDOUT DUNG CHUNG ({n_rows} bai, kiem tra 1 lan)")
    add("=" * 72)
    add()
    add(
        f"Ca 3 nhanh dung CHUNG dung {n_rows} dong holdout (source_row_id khop tuyet "
        "doi), nguon tu Lexicon_based/data/ground_truth_tune_holdout_split.csv."
    )
    add(
        "Moi nhanh: fit/tune tren phan Tune, predict DUNG 1 LAN tren Holdout - "
        "cung 1 giao thuc cho ca 3, khac voi RESULTS_SUMMARY.txt (RF o do la "
        "10 x 5-fold CV tren toan bo ground truth, KHONG phai so holdout)."
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
            "nhanh khong tach duoc mot cach dang tin khoi nhieu resample, du "
            "thu hang single-point co the khac nhau ro."
        )
    add()

    add("CANH BAO QUAN TRONG - HOLDOUT DA BI DUNG LAI (khong phai lan nhin dau tien)")
    add("-" * 72)
    add(
        f"{n_rows} dong holdout nay DA duoc dung 1 lan truoc do trong chinh nhanh "
        "Traditional_ML (ML_SUMMARY.qmd muc 6.2) de quyet dinh random_forest "
        "co nen dung lexicon feature hay khong (ket qua: co, Delta=+0.049 "
        "CI[+0.017,+0.082] p=0.003). Lexicon_based va Transfer_Learning cung "
        f"da tung dung dung {n_rows} dong nay de bao cao so 'Holdout' rieng cua ho."
    )
    add(
        "So sanh 3 nhanh o day la MOT CAU HOI KHAC (nhanh nao tot hon, khong "
        "phai lexicon feature co giup RF khong), nen lan nhin dau tien CHO "
        "CAU HOI NAY van con hop le theo nguyen tac 'nhin 1 lan cho 1 cau "
        "hoi doc lap'. Nhung day KHONG con la mot holdout 'trinh nguyen' "
        "theo nghia tuyet doi - cau hinh Random Forest duoc chon (co dung "
        "lexicon feature) it nhieu da chiu anh huong tu lan nhin truoc do o "
        f"chinh {n_rows} dong nay. Khi trich dan ket qua nay o bao cao, PHAI ghi ro "
        "diem nay, khong duoc trinh bay nhu mot kiem dinh doc lap hoan toan."
    )
    add(
        "Ky luat tiep theo: KHONG chay lai script nay voi mot cau hinh Random "
        "Forest / Lexicon / PhoBERT khac de xem thu hang co doi khong - do se "
        "la lan nhin thu 2 cho DUNG cau hoi nay, vi pham chinh nguyen tac vua "
        "neu."
    )

    report = "\n".join(lines)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print("\nOutput report:", RESULTS_PATH)


if __name__ == "__main__":
    main()
