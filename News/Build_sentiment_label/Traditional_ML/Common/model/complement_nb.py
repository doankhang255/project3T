"""Complement Naive Bayes - build_estimator() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/run_model.py`` for the
production run - this model never uses the lexicon feature block). No
``main()`` here on purpose: this module is a library, not a runnable step.

ComplementNB (Rennie et al. 2003, "Tackling the Poor Assumptions of Naive
Bayes Text Classifiers") was designed for class-imbalanced text
classification - it estimates each class's parameters using data from all
*other* classes, which tends to be more stable than MultinomialNB when
classes are skewed (see ML_SUMMARY.qmd section 5.2). It is reported here as
an additional model alongside ``naive_bayes.py`` (MultinomialNB), not as a
replacement for it - see ML_SUMMARY.qmd section 5.2 (M2.1) for the earlier
replace-question test and why MultinomialNB was kept as the sole
``naive_bayes`` model.
"""

from __future__ import annotations

from sklearn.naive_bayes import ComplementNB


def build_estimator(random_state: int) -> ComplementNB:
    # ComplementNB has no randomness; random_state is accepted for a uniform
    # estimator_factory signature across models.
    del random_state
    return ComplementNB()
