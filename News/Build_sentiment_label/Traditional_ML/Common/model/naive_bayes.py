"""Multinomial Naive Bayes - build_estimator() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/run_model.py`` for the
production run - this model never uses the lexicon feature block). No
``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB


def build_estimator(random_state: int) -> MultinomialNB:
    # MultinomialNB has no randomness; random_state is accepted for a uniform
    # estimator_factory signature across models.
    del random_state
    return MultinomialNB()
