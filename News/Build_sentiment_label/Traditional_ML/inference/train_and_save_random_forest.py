"""Fit the production Random Forest (+lexicon features) ONCE on all 1044
ground-truth rows and persist it, so it can be applied to the 126k-article
corpus for the sentiment-index/VN-Index comparison
(Transfer_Learning/inference/score_corpus_phobert.py's counterpart).

Why this file exists: every existing Traditional_ML runner
(``experiment_Lexicon_features/run_model.py``) only ever fits inside a CV
loop and throws the fitted model away after scoring the held-out fold -
there was never a "fit once, save, reuse on new documents" step, because
until now nothing needed to score documents outside the ground truth.

Reuses the exact same feature recipe as the production run (do not
re-derive it here): ``Common/TF_IDF.py`` (fit_tfidf_vocabulary +
transform_tfidf), ``Common/model/common.py::select_top_features``, and
``Common/model/lexicon_features.py::build_lexicon_feature_matrix``, then
``Common/model/random_forest.py::build_estimator`` (isotonic-calibrated RF).
The only new code is the "fit on everything, persist the fitted vocabulary +
lexicon standardization stats + model together" wiring.

Output: inference/random_forest_production.joblib, containing:
    - model: the fitted CalibratedClassifierCV(RandomForestClassifier)
    - vocabulary_df: the FULL fitted TF-IDF vocabulary (needed by
      transform_tfidf on new documents)
    - selected_indices: which vocabulary columns survived select_top_features
    - lexicon_mean / lexicon_std: train-set standardization stats for the
      9 lexicon feature columns (same z-score discipline as every CV fold)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import (  # noqa: E402
    build_document_term_counts,
    fit_tfidf_vocabulary,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (  # noqa: E402
    MAX_FEATURES,
    RANDOM_SEED,
    encode_labels,
    load_ground_truth_frame,
    select_top_features,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (  # noqa: E402
    build_lexicon_feature_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.model import random_forest  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "random_forest_production.joblib"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_ground_truth_frame()
    print("Ground truth rows:", len(df))
    print(df["ground_truth_label"].value_counts().to_string())

    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    lexicon_matrix = build_lexicon_feature_matrix(df["Tokenize_content"].tolist())
    print("Lexicon feature matrix:", lexicon_matrix.shape)

    vocabulary_df = fit_tfidf_vocabulary(term_counts, total_documents=len(term_counts))
    x_full, _ = transform_tfidf(term_counts, vocabulary_df)
    x_selected, selected_vocabulary, selected_indices = select_top_features(
        x_full, vocabulary_df, max_features=MAX_FEATURES
    )
    print("Full vocabulary:", len(vocabulary_df), "| selected features:", x_selected.shape[1])

    lexicon_mean = lexicon_matrix.mean(axis=0)
    lexicon_std = lexicon_matrix.std(axis=0)
    lexicon_std = np.where(lexicon_std > 1e-8, lexicon_std, 1.0)
    lexicon_scaled = (lexicon_matrix - lexicon_mean) / lexicon_std

    x_train = np.hstack([x_selected, lexicon_scaled])
    print("Final feature matrix:", x_train.shape)

    model = random_forest.build_estimator(RANDOM_SEED)
    model.fit(x_train, y)
    if list(model.classes_) != list(range(3)):
        raise AssertionError(f"Unexpected class order: {list(model.classes_)}")
    print("Fitted. Train accuracy (in-sample, NOT a generalisation estimate):",
          float((model.predict(x_train) == y).mean()))

    joblib.dump(
        {
            "model": model,
            "vocabulary_df": vocabulary_df,
            "selected_indices": selected_indices,
            "lexicon_mean": lexicon_mean,
            "lexicon_std": lexicon_std,
            "n_train_rows": len(df),
        },
        OUTPUT_PATH,
    )
    print("\nSaved production model + feature-fit artifacts to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
