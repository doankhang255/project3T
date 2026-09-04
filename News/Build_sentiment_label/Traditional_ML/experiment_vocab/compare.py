"""Experiment: does a leaner vocabulary help the Traditional_ML models?

Compares two feature configs, same 4 models, same leak-free 5-fold CV:

  baseline : n-gram (1,3), min_df floor = 2   (= the main pipeline today)
  leaner   : n-gram (1,2), min_df floor = 3   (drop trigrams + the df=2 tail)

Self-contained: it only *imports* pure helpers from the existing code
(n-gram builder, the df filter, the per-model estimator factories, the
metric functions) and does its own CV wiring so the one differing knob
(min_df floor) is an explicit parameter. It does NOT edit or monkeypatch
any existing module, does NOT re-run prepare_ground_truth.py, and writes
NO data files - only RESULTS.txt + stdout.

    python News/Build_sentiment_label/Traditional_ML/experiment_vocab/compare.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import (
    build_document_terms as build_ngram_terms,
)
from News.Build_sentiment_label.Common.ngram_filter import (
    choose_ngram_terms,
    scaled_min_df_by_ngram,
)
from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)
from News.Build_sentiment_label.Traditional_ML.TF_IDF import (
    ML_MAX_DF_RATIO,
    ML_MIN_DF_RATIO_BY_NGRAM,
    build_ngram_terms_dataframe,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.model.common import (
    RANDOM_SEED,
    VALID_LABELS,
    build_stratified_folds,
    compute_metrics,
    encode_labels,
    resolve_n_splits,
)
from News.Build_sentiment_label.Traditional_ML.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.model.naive_bayes import (
    build_estimator as build_naive_bayes,
)
from News.Build_sentiment_label.Traditional_ML.model.random_forest import (
    build_estimator as build_random_forest,
)
from News.Build_sentiment_label.Traditional_ML.model.svm import (
    build_estimator as build_svm,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PARQUET = SCRIPT_DIR.parent / "data" / "ground_truth_labeled_tokenized.parquet"
RESULTS_PATH = SCRIPT_DIR / "RESULTS.txt"

TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOKENIZED_COLUMN = "Tokenize_content"

MODELS = {
    "logistic_regression": build_logistic_regression,
    "naive_bayes": build_naive_bayes,
    "random_forest": build_random_forest,
    "svm": build_svm,
}

CONFIGS = {
    "baseline": {"max_n": 3, "min_df_floor": 2},
    "leaner": {"max_n": 2, "min_df_floor": 3},
}


def load_frame() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found - run the main pipeline "
            "(prepare_ground_truth.py) first."
        )
    df = pd.read_parquet(INPUT_PARQUET)
    df["ground_truth_label"] = df["sentiment"].astype(str).str.strip().str.casefold()
    return df.loc[df["ground_truth_label"].isin(VALID_LABELS)].reset_index(drop=True)


def build_term_counts(df: pd.DataFrame, max_n: int) -> list[Counter[str]]:
    counts: list[Counter[str]] = []
    for _, row in df.iterrows():
        terms = build_ngram_terms(row[TOKENIZED_SENTENCES_COLUMN], min_n=1, max_n=max_n)
        if not terms:
            terms = build_ngram_terms(row[TOKENIZED_COLUMN], min_n=1, max_n=max_n)
        counts.append(Counter(terms))
    return counts


def fit_vocabulary(
    term_counts: list[Counter[str]],
    total_documents: int,
    min_df_floor: int,
    stopwords: set[str],
) -> pd.DataFrame:
    """Same recipe as TF_IDF.fit_tfidf_vocabulary, but min_df_floor is an
    explicit argument instead of the module constant.
    """
    ngram_terms_df = build_ngram_terms_dataframe(term_counts)
    min_df_by_ngram = scaled_min_df_by_ngram(
        total_documents=total_documents,
        min_df_ratio_by_ngram=ML_MIN_DF_RATIO_BY_NGRAM,
        floor=min_df_floor,
    )
    candidate = choose_ngram_terms(
        ngram_terms_df=ngram_terms_df,
        total_documents=total_documents,
        min_df_by_ngram=min_df_by_ngram,
        max_df_ratio=ML_MAX_DF_RATIO,
        remove_stopwords=True,
        stopwords=stopwords,
    )
    if candidate.empty:
        raise ValueError("No terms left after vocabulary filtering.")

    vocab = candidate.rename(columns={"tf": "total_tf"})[
        ["term", "ngram_n", "total_tf", "df", "df_ratio"]
    ].copy()
    vocab["idf"] = np.log(total_documents / vocab["df"])
    vocab = vocab.sort_values(
        ["ngram_n", "df", "total_tf", "term"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    vocab.insert(0, "term_id", np.arange(len(vocab), dtype=int))
    return vocab


def cross_validate(
    estimator_factory,
    term_counts: list[Counter[str]],
    y: np.ndarray,
    min_df_floor: int,
    stopwords: set[str],
) -> np.ndarray:
    """Mirror of model/common.run_cross_validation: TF-IDF vocab + idf fitted
    on the training rows of each fold only.
    """
    y = np.asarray(y, dtype=int)
    n_splits = resolve_n_splits(y)
    n_rows = len(y)
    probabilities = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    all_indices = np.arange(n_rows)

    for fold_id, validation_indices in enumerate(build_stratified_folds(y, n_splits), start=1):
        train_indices = all_indices[~np.isin(all_indices, validation_indices)]
        train_counts = [term_counts[i] for i in train_indices]
        val_counts = [term_counts[i] for i in validation_indices]

        vocab = fit_vocabulary(train_counts, len(train_indices), min_df_floor, stopwords)
        x_train, _ = transform_tfidf(train_counts, vocab)
        x_val, _ = transform_tfidf(val_counts, vocab)

        model = estimator_factory(RANDOM_SEED + fold_id)
        model.fit(x_train, y[train_indices])
        probabilities[validation_indices] = model.predict_proba(x_val)

    return probabilities.argmax(axis=1)


def run_config(
    df: pd.DataFrame, y: np.ndarray, max_n: int, min_df_floor: int, stopwords: set[str]
) -> tuple[dict[str, dict], int, dict[int, int]]:
    term_counts = build_term_counts(df, max_n=max_n)
    full_vocab = fit_vocabulary(term_counts, len(term_counts), min_df_floor, stopwords)
    by_ngram = {int(k): int(v) for k, v in full_vocab["ngram_n"].value_counts().items()}

    results: dict[str, dict] = {}
    for model_name, factory in MODELS.items():
        print(f"  [{model_name}] ...", flush=True)
        predictions = cross_validate(factory, term_counts, y, min_df_floor, stopwords)
        metrics = compute_metrics(y, predictions)
        overall = metrics.loc[metrics["metric_scope"].eq("overall")].iloc[0]
        per_class = metrics.set_index("metric_scope")["f1"]
        results[model_name] = {
            "macro_f1": float(overall["f1"]),
            "accuracy": float(overall["accuracy"]),
            "f1_negative": float(per_class["negative"]),
            "f1_neutral": float(per_class["neutral"]),
            "f1_positive": float(per_class["positive"]),
        }
    return results, len(full_vocab), by_ngram


def _delta(new: float, old: float) -> str:
    diff = new - old
    return f"{'+' if diff >= 0 else ''}{diff:.3f}"


def render_report(vocab_info: dict, all_results: dict, n_rows: int) -> str:
    lines: list[str] = []
    add = lines.append

    add("EXPERIMENT - LEANER VOCABULARY vs BASELINE")
    add("=" * 68)
    add("")
    add(f"Ground truth: {n_rows} bai | 5-fold stratified CV | TF-IDF fit per-fold")
    add("")
    add("baseline : n-gram (1,3), min_df floor = 2   (= main pipeline)")
    add("leaner   : n-gram (1,2), min_df floor = 3")
    add("")
    add("VOCAB (fit tren toan bo ground truth; moi fold model dung vocab train rieng)")
    add("-" * 68)
    for config_name, (size, by_ngram) in vocab_info.items():
        parts = ", ".join(f"{n}gram={by_ngram.get(n, 0)}" for n in sorted(by_ngram))
        add(f"  {config_name:<9}: {size:>5} term   ({parts})")
    base_size, lean_size = vocab_info["baseline"][0], vocab_info["leaner"][0]
    add(f"  -> leaner cat {base_size - lean_size} term "
        f"({100 * (base_size - lean_size) / base_size:.0f}%)")
    add("")

    ranked = sorted(
        MODELS, key=lambda m: all_results["baseline"][m]["macro_f1"], reverse=True
    )
    for metric_key, title in [("macro_f1", "MACRO F1"), ("accuracy", "ACCURACY")]:
        add(f"{title} (5-fold CV)")
        add("-" * 68)
        add(f"  {'model':<22}{'baseline':>10}{'leaner':>10}{'delta':>9}")
        for model_name in ranked:
            base = all_results["baseline"][model_name][metric_key]
            lean = all_results["leaner"][model_name][metric_key]
            add(f"  {model_name:<22}{base:>10.3f}{lean:>10.3f}{_delta(lean, base):>9}")
        add("")

    add("PER-CLASS F1")
    add("-" * 68)
    add(f"  {'model':<20}{'config':<10}{'neg':>9}{'neu':>9}{'pos':>9}")
    for model_name in ranked:
        for config_name in CONFIGS:
            r = all_results[config_name][model_name]
            add(f"  {model_name:<20}{config_name:<10}"
                f"{r['f1_negative']:>9.3f}{r['f1_neutral']:>9.3f}{r['f1_positive']:>9.3f}")
    add("")

    add("READ")
    add("-" * 68)
    for config_name in CONFIGS:
        best = max(MODELS, key=lambda m: all_results[config_name][m]["macro_f1"])
        add(f"  Best {config_name:<9}: {best} macro F1 = "
            f"{all_results[config_name][best]['macro_f1']:.3f}")
    mean_delta = float(np.mean([
        all_results["leaner"][m]["macro_f1"] - all_results["baseline"][m]["macro_f1"]
        for m in MODELS
    ]))
    add(f"  Mean macro-F1 delta (leaner - baseline), 4 models: {_delta(mean_delta, 0.0)}")
    add("")
    add("  Luu y: 152 dong -> +/- 0.02 macro-F1 co the chi la nhieu CV, chua ket luan duoc.")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_frame()
    y = encode_labels(df["ground_truth_label"])
    stopwords = load_stopwords(DEFAULT_STOPWORDS_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PARQUET.name}")

    vocab_info: dict = {}
    all_results: dict = {}
    for config_name, cfg in CONFIGS.items():
        print(f"\n=== {config_name}: max_n={cfg['max_n']}, min_df_floor={cfg['min_df_floor']} ===")
        results, size, by_ngram = run_config(
            df, y, cfg["max_n"], cfg["min_df_floor"], stopwords
        )
        all_results[config_name] = results
        vocab_info[config_name] = (size, by_ngram)

    report = render_report(vocab_info, all_results, n_rows=len(df))
    RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nWritten: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
