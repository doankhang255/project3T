"""Flow 2 - independent correctness checks for the Traditional_ML pipeline.

    python News/Build_sentiment_label/Traditional_ML/verify_pipeline.py

Run this after run_pipeline.py (Flow 1). Every check prints PASS / FAIL with a
short reason; any FAIL makes the script exit non-zero. No pytest dependency -
plain asserts on the committed CSV/parquet outputs plus a few in-process
recomputations.

Covers the four issues the refactor targeted:
  #2 no data leakage        -> CHECK 2
  #3 MAX_FEATURES kept,
     applied inside the fold -> CHECK 4
  #4 tokenizer == VNCoreNLP  -> CHECK 1
  #7 one shared CV path      -> CHECK 3
plus reproducibility (CHECK 5) and output schema (CHECK 6).
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = SCRIPT_DIR / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)
from News.Build_sentiment_label.Traditional_ML.TF_IDF import (
    build_document_term_counts,
    fit_tfidf_vocabulary,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.model.common import (
    MAX_FEATURES,
    N_SPLITS,
    RANDOM_SEED,
    VALID_LABELS,
    build_stratified_folds,
    compute_metrics,
    encode_labels,
    load_ground_truth_frame,
    resolve_n_splits,
    run_cross_validation,
    select_top_features,
)

MODEL_NAMES = ["logistic_regression", "naive_bayes", "random_forest", "svm"]
N_LABELS = len(VALID_LABELS)

VNCORENLP_TOKENIZED_PATH = (
    PROJECT_ROOT
    / "data_news"
    / "data_tokenized"
    / "equity_news_tokenized_vncorenlp.parquet"
)
GROUND_TRUTH_CSV_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"
TOKENIZED_PARQUET_PATH = DATA_DIR / "ground_truth_labeled_tokenized.parquet"


class CheckError(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _as_token_list(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(token) for token in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(token) for token in value]
    return []


# --------------------------------------------------------------------------- #
# shared fixtures (loaded once)
# --------------------------------------------------------------------------- #
def _load_fixtures() -> dict[str, object]:
    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    stopwords = load_stopwords(DEFAULT_STOPWORDS_PATH)
    return {
        "df": df,
        "term_counts": term_counts,
        "y": y,
        "stopwords": stopwords,
    }


# --------------------------------------------------------------------------- #
# CHECK 1 - #4 tokenizer alignment
# --------------------------------------------------------------------------- #
def check_1_tokenizer_alignment(fixtures: dict[str, object]) -> str:
    _require(TOKENIZED_PARQUET_PATH.exists(), f"missing {TOKENIZED_PARQUET_PATH}")
    tok = pd.read_parquet(TOKENIZED_PARQUET_PATH)
    gt = pd.read_csv(GROUND_TRUTH_CSV_PATH, encoding="utf-8-sig")
    corpus = pd.read_parquet(
        VNCORENLP_TOKENIZED_PATH,
        columns=["title", "publication_date", "Tokenize_content"],
    )

    _require(len(tok) == len(gt), f"row count {len(tok)} != ground truth {len(gt)}")
    source_row_ids = tok["source_row_id"].to_numpy()
    corpus_rows = corpus.iloc[source_row_ids].reset_index(drop=True)

    title_mismatch = (
        tok["title"].astype(str).to_numpy() != corpus_rows["title"].astype(str).to_numpy()
    )
    _require(not title_mismatch.any(), f"{int(title_mismatch.sum())} title mismatch(es)")

    date_mismatch = (
        pd.to_datetime(gt["publication_date"]).to_numpy()
        != pd.to_datetime(corpus_rows["publication_date"]).to_numpy()
    )
    _require(not date_mismatch.any(), f"{int(date_mismatch.sum())} publication_date mismatch(es)")

    token_mismatch = 0
    for got, expected in zip(
        tok["Tokenize_content"], corpus_rows["Tokenize_content"], strict=True
    ):
        if _as_token_list(got) != _as_token_list(expected):
            token_mismatch += 1
    _require(token_mismatch == 0, f"{token_mismatch} row(s) have non-VNCoreNLP tokens")

    scanned = [
        SCRIPT_DIR / "prepare_ground_truth.py",
        SCRIPT_DIR / "TF_IDF.py",
        SCRIPT_DIR / "run_pipeline.py",
        SCRIPT_DIR / "model" / "common.py",
        *[SCRIPT_DIR / "model" / f"{name}.py" for name in MODEL_NAMES],
    ]
    # only real usage counts (import / attribute access), not prose in a docstring
    usage_pattern = re.compile(
        r"^\s*(?:import|from)\s+[\w.]*underthesea|underthesea\s*\.", re.MULTILINE
    )
    with_underthesea = [
        str(path.relative_to(SCRIPT_DIR))
        for path in scanned
        if usage_pattern.search(path.read_text(encoding="utf-8"))
    ]
    _require(not with_underthesea, f"underthesea still imported/used in {with_underthesea}")

    return (
        f"152 rows: title + publication_date + Tokenize_content all match "
        f"equity_news_tokenized_vncorenlp.parquet via source_row_id; "
        f"no underthesea import in the branch"
    )


# --------------------------------------------------------------------------- #
# CHECK 2 - #2 no data leakage in the per-fold TF-IDF
# --------------------------------------------------------------------------- #
def check_2_no_leakage(fixtures: dict[str, object]) -> str:
    term_counts = fixtures["term_counts"]
    y = fixtures["y"]
    stopwords = fixtures["stopwords"]

    n_splits = resolve_n_splits(y, N_SPLITS)
    folds = build_stratified_folds(y, n_splits=n_splits)
    all_indices = np.arange(len(y))

    vocab_full = fit_tfidf_vocabulary(term_counts, total_documents=len(term_counts), stopwords=stopwords)
    full_terms = set(vocab_full["term"])

    validation_indices = folds[0]
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[validation_indices] = False
    train_indices = all_indices[train_mask]
    train_counts = [term_counts[i] for i in train_indices]
    val_counts = [term_counts[i] for i in validation_indices]

    vocab_train = fit_tfidf_vocabulary(
        train_counts, total_documents=len(train_indices), stopwords=stopwords
    )
    train_terms = set(vocab_train["term"])

    # (a) whole-corpus fit and train-only fit genuinely differ -> the old code
    #     (fit once on all 152) really was leaking into every fold.
    only_in_full = full_terms - train_terms
    _require(
        len(only_in_full) > 0,
        "train-only vocab == whole-corpus vocab; fold fit is not actually train-only",
    )

    # (b) df counts in the fitted vocab come from the training rows only.
    train_df = Counter()
    for counts in train_counts:
        train_df.update(counts.keys())
    df_mismatch = [
        row["term"]
        for _, row in vocab_train.iterrows()
        if int(row["df"]) != int(train_df[row["term"]])
    ]
    _require(not df_mismatch, f"df not train-only for: {df_mismatch[:5]}")

    # (c) idf == log(n_train / df_train), not log(152 / df).
    n_train = len(train_indices)
    idf_expected = np.log(n_train / vocab_train["df"].to_numpy())
    _require(
        np.allclose(vocab_train["idf"].to_numpy(), idf_expected, atol=1e-9),
        "idf is not log(n_train / df_train)",
    )
    idf_if_leaked = np.log(len(y) / vocab_train["df"].to_numpy())
    _require(
        not np.allclose(vocab_train["idf"].to_numpy(), idf_if_leaked, atol=1e-9),
        "idf matches log(152 / df) - denominator is the full corpus, not the fold",
    )

    # (d) terms that occur only in held-out docs never enter the fold vocab.
    val_only_terms = set()
    for counts in val_counts:
        val_only_terms.update(counts.keys())
    val_only_terms -= set(train_df.keys())
    leaked = val_only_terms & train_terms
    _require(not leaked, f"held-out-only terms present in fold vocab: {list(leaked)[:5]}")

    return (
        f"fold 1: train vocab {len(train_terms)} vs full vocab {len(full_terms)} "
        f"({len(only_in_full)} full-only terms); df + idf are train-only; "
        f"{len(val_only_terms)} held-out-only terms all excluded"
    )


# --------------------------------------------------------------------------- #
# CHECK 3 - #7 one shared CV path / consistent folds
# --------------------------------------------------------------------------- #
def check_3_fold_consistency(fixtures: dict[str, object]) -> str:
    y = fixtures["y"]
    expected_splits = resolve_n_splits(y, N_SPLITS)

    predictions = {
        name: pd.read_csv(DATA_DIR / f"{name}_predictions.csv") for name in MODEL_NAMES
    }
    reference = predictions[MODEL_NAMES[0]]
    _require("validation_fold" in reference.columns, "predictions missing validation_fold column")

    ref_ids = reference["id"].tolist()
    ref_folds = reference["validation_fold"].to_numpy()
    for name, frame in predictions.items():
        _require(frame["id"].tolist() == ref_ids, f"{name}: id order differs")
        _require(
            np.array_equal(frame["validation_fold"].to_numpy(), ref_folds),
            f"{name}: validation_fold differs from {MODEL_NAMES[0]}",
        )

    fold_ids = sorted(set(ref_folds.tolist()))
    _require(
        fold_ids == list(range(1, expected_splits + 1)),
        f"fold ids {fold_ids} != 1..{expected_splits}",
    )
    counts = np.bincount(ref_folds, minlength=expected_splits + 1)[1:]
    _require(int(counts.sum()) == len(y), "folds do not partition all rows")

    # stratified: each (fold, class) count is floor or ceil of class_size / n_splits
    labels_by_row = y
    fold_by_row = ref_folds
    strat_errors = []
    for label_id in range(N_LABELS):
        class_size = int((labels_by_row == label_id).sum())
        low, high = class_size // expected_splits, math.ceil(class_size / expected_splits)
        for fold_id in fold_ids:
            in_cell = int(((fold_by_row == fold_id) & (labels_by_row == label_id)).sum())
            if not low <= in_cell <= high:
                strat_errors.append(
                    f"class {VALID_LABELS[label_id]} fold {fold_id}: {in_cell} not in [{low},{high}]"
                )
    _require(not strat_errors, "; ".join(strat_errors))

    return (
        f"4 models share one fold partition; n_splits = {expected_splits} = "
        f"min({N_SPLITS}, smallest class); per-fold sizes {counts.tolist()}, stratified"
    )


# --------------------------------------------------------------------------- #
# CHECK 4 - #3 MAX_FEATURES kept and applied inside the fold
# --------------------------------------------------------------------------- #
def check_4_feature_cap(fixtures: dict[str, object]) -> str:
    _require(isinstance(MAX_FEATURES, int) and MAX_FEATURES > 0, "MAX_FEATURES not a positive int")

    common_src = (SCRIPT_DIR / "model" / "common.py").read_text(encoding="utf-8")
    match = re.search(r"def run_cross_validation\(.*?\n(?=\ndef |\Z)", common_src, re.S)
    _require(match is not None, "run_cross_validation not found in common.py")
    _require(
        "select_top_features(" in match.group(0),
        "select_top_features is not called inside run_cross_validation",
    )
    for name in MODEL_NAMES:
        src = (SCRIPT_DIR / "model" / f"{name}.py").read_text(encoding="utf-8")
        _require(
            "select_top_features(" not in src,
            f"{name}.py calls select_top_features directly (should go through the CV driver)",
        )

    # behaviour: below the cap every feature is kept; above the cap it truncates.
    term_counts = fixtures["term_counts"]
    y = fixtures["y"]
    stopwords = fixtures["stopwords"]
    folds = build_stratified_folds(y, n_splits=resolve_n_splits(y, N_SPLITS))
    val_idx = folds[0]
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    train_counts = [term_counts[i] for i in train_idx]
    val_counts = [term_counts[i] for i in val_idx]
    vocab = fit_tfidf_vocabulary(train_counts, total_documents=len(train_idx), stopwords=stopwords)
    x_train, _ = transform_tfidf(train_counts, vocab)
    x_val, _ = transform_tfidf(val_counts, vocab)

    _require(len(vocab) <= MAX_FEATURES, "unexpected: current vocab already exceeds MAX_FEATURES")
    x_sel, _, sel_idx = select_top_features(x_train, vocab, max_features=MAX_FEATURES)
    _require(
        x_sel.shape[1] == len(vocab) and np.array_equal(sel_idx, np.arange(len(vocab))),
        "below the cap select_top_features dropped features",
    )

    capped_x, _, capped_idx = select_top_features(x_train, vocab, max_features=100)
    _require(capped_x.shape[1] == 100, f"cap=100 gave {capped_x.shape[1]} features")
    _require(x_val[:, capped_idx].shape[1] == 100, "held-out matrix not projected onto the cap")

    return (
        f"MAX_FEATURES={MAX_FEATURES}, called only inside run_cross_validation; "
        f"vocab {len(vocab)} <= cap -> all kept; forced cap=100 truncates train+val"
    )


# --------------------------------------------------------------------------- #
# CHECK 5 - reproducibility + RESULTS_SUMMARY in sync
# --------------------------------------------------------------------------- #
def check_5_reproducible(fixtures: dict[str, object]) -> str:
    from News.Build_sentiment_label.Traditional_ML.model.naive_bayes import build_estimator

    term_counts = fixtures["term_counts"]
    y = fixtures["y"]
    probabilities, predictions, _ = run_cross_validation(build_estimator, term_counts, y)
    fresh = compute_metrics(y, predictions).reset_index(drop=True)
    saved = pd.read_csv(DATA_DIR / "naive_bayes_metrics.csv").reset_index(drop=True)

    for column in ["precision", "recall", "f1", "accuracy"]:
        a = fresh[column].to_numpy(dtype=float)
        b = saved[column].to_numpy(dtype=float)
        _require(
            np.allclose(a, b, atol=1e-9, equal_nan=True),
            f"naive_bayes {column} not reproducible: {a} vs {b}",
        )

    comparison = pd.read_csv(DATA_DIR / "traditional_ml_model_comparison.csv")
    overall = comparison.loc[comparison["metric_scope"].eq("overall")].set_index("model")
    summary_text = (SCRIPT_DIR / "RESULTS_SUMMARY.txt").read_text(encoding="utf-8")

    rank_rows = re.findall(
        r"\d+\.\s+(\w+)\s+F1 = ([0-9.]+)\s+Accuracy = ([0-9.]+)", summary_text
    )
    _require(len(rank_rows) == len(MODEL_NAMES), f"ranking table has {len(rank_rows)} rows")
    for model_name, f1_text, acc_text in rank_rows:
        _require(model_name in overall.index, f"unknown model in summary: {model_name}")
        _require(
            abs(float(f1_text) - float(overall.loc[model_name, "f1"])) < 5e-4,
            f"{model_name}: summary F1 {f1_text} != csv {overall.loc[model_name, 'f1']:.3f}",
        )
        _require(
            abs(float(acc_text) - float(overall.loc[model_name, "accuracy"])) < 5e-4,
            f"{model_name}: summary accuracy {acc_text} != csv",
        )

    return (
        "naive_bayes CV metrics reproduce the saved CSV (atol 1e-9); "
        "RESULTS_SUMMARY ranking matches traditional_ml_model_comparison.csv"
    )


# --------------------------------------------------------------------------- #
# CHECK 6 - output schema / internal consistency
# --------------------------------------------------------------------------- #
def check_6_output_schema(fixtures: dict[str, object]) -> str:
    n_rows = len(fixtures["y"])
    for name in MODEL_NAMES:
        metrics = pd.read_csv(DATA_DIR / f"{name}_metrics.csv")
        scopes = metrics["metric_scope"].tolist()
        _require(
            scopes == ["negative", "neutral", "positive", "overall"],
            f"{name}_metrics.csv scopes = {scopes}",
        )
        per_class = metrics.loc[metrics["metric_scope"] != "overall"]
        _require(
            int(per_class["support"].sum()) == n_rows,
            f"{name}: per-class support sums to {int(per_class['support'].sum())} != {n_rows}",
        )
        overall = metrics.loc[metrics["metric_scope"].eq("overall")].iloc[0]
        _require(int(overall["support"]) == n_rows, f"{name}: overall support != {n_rows}")
        _require(0.0 <= float(overall["accuracy"]) <= 1.0, f"{name}: accuracy out of range")

        cm = pd.read_csv(DATA_DIR / f"{name}_confusion_matrix.csv")
        pred_cols = ["pred_negative", "pred_neutral", "pred_positive"]
        _require(int(cm[pred_cols].to_numpy().sum()) == n_rows, f"{name}: confusion total != {n_rows}")
        for label_id, label in enumerate(VALID_LABELS):
            row_total = int(cm.loc[cm["true_label"].eq(f"true_{label}"), pred_cols].to_numpy().sum())
            support = int(per_class.loc[per_class["metric_scope"].eq(label), "support"].iloc[0])
            _require(
                row_total == support,
                f"{name}: confusion row true_{label} sums to {row_total} != support {support}",
            )

        preds = pd.read_csv(DATA_DIR / f"{name}_predictions.csv")
        _require(len(preds) == n_rows, f"{name}: predictions row count {len(preds)} != {n_rows}")
        prob_cols = ["prob_negative", "prob_neutral", "prob_positive"]
        prob_sum = preds[prob_cols].to_numpy().sum(axis=1)
        _require(np.allclose(prob_sum, 1.0, atol=1e-6), f"{name}: prob_* rows do not sum to 1")
        argmax_label = np.array(VALID_LABELS)[preds[prob_cols].to_numpy().argmax(axis=1)]
        _require(
            np.array_equal(argmax_label, preds["predicted_label"].to_numpy()),
            f"{name}: predicted_label != argmax(prob_*)",
        )
        score = preds["prob_positive"].to_numpy() - preds["prob_negative"].to_numpy()
        _require(
            np.allclose(score, preds["sentiment_score_ml"].to_numpy(), atol=1e-9),
            f"{name}: sentiment_score_ml != prob_positive - prob_negative",
        )

    return "metrics / confusion / predictions CSVs are internally consistent for all 4 models"


CHECKS = [
    ("CHECK 1  (#4 tokenizer = VNCoreNLP)", check_1_tokenizer_alignment),
    ("CHECK 2  (#2 no data leakage)", check_2_no_leakage),
    ("CHECK 3  (#7 one shared CV path)", check_3_fold_consistency),
    ("CHECK 4  (#3 MAX_FEATURES kept, in-fold)", check_4_feature_cap),
    ("CHECK 5  (reproducible + summary in sync)", check_5_reproducible),
    ("CHECK 6  (output schema)", check_6_output_schema),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fixtures = _load_fixtures()
    failures = 0
    print("=" * 72)
    for title, check_fn in CHECKS:
        try:
            detail = check_fn(fixtures)
        except CheckError as exc:
            failures += 1
            print(f"FAIL  {title}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error as a failure
            failures += 1
            print(f"FAIL  {title}\n      unexpected error: {exc!r}")
        else:
            print(f"PASS  {title}\n      {detail}")
    print("=" * 72)

    if failures:
        print(f"{failures} check(s) FAILED")
        raise SystemExit(1)
    print("all checks PASSED")


if __name__ == "__main__":
    main()
