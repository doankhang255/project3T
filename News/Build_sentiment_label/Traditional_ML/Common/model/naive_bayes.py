"""Multinomial Naive Bayes - build_estimator() + build_top_features() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/naive_bayes.py`` for the
production run - this model never uses the lexicon feature block). No
``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.common import VALID_LABELS


def build_estimator(random_state: int) -> MultinomialNB:
    # MultinomialNB has no randomness; random_state is accepted for a uniform
    # estimator_factory signature across models.
    del random_state
    return MultinomialNB()


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    model = MultinomialNB()
    model.fit(x, y)

    rows = []
    for label_id, label in enumerate(VALID_LABELS):
        top_indices = np.argsort(model.feature_log_prob_[label_id])[-top_n:][::-1]
        for rank, feature_index in enumerate(top_indices, start=1):
            vocab_row = vocabulary.iloc[int(feature_index)]
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "selected_feature_id": int(feature_index),
                    "term_id": int(vocab_row["term_id"]),
                    "term": vocab_row["term"],
                    "ngram_n": int(vocab_row["ngram_n"]),
                    "feature_log_prob": float(
                        model.feature_log_prob_[label_id, feature_index]
                    ),
                }
            )
    return pd.DataFrame(rows)
