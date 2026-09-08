"""M2.3 - McNemar's test for comparing two classifiers on the same items
(Dietterich 1998).

``statsmodels`` is not a project dependency, so this is a small hand-rolled
implementation on ``scipy`` - consistent with the rest of the repo, where
Newey-West, the metric functions and the fold splitter are all hand-rolled.

Given the out-of-fold predictions of model A and model B on the *same* rows:

    b = # rows where A is right and B is wrong
    c = # rows where A is wrong and B is right

McNemar tests H0: b and c are draws from the same distribution, i.e. the two
models have the same error rate. Rows both models get right, or both get
wrong, carry no information and are ignored.

    b + c <  25  -> exact two-sided binomial test against p = 0.5
    b + c >= 25  -> chi-square with continuity correction, 1 degree of freedom
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EXACT_THRESHOLD = 25


def mcnemar_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> dict[str, object]:
    """Return ``{b, c, n_discordant, statistic, p_value, method}``.

    ``b`` counts rows where ``pred_a`` is correct and ``pred_b`` is not, so a
    small ``p_value`` with ``b > c`` means model A is reliably better.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "no discordant pairs",
        }

    if n_discordant < EXACT_THRESHOLD:
        p_value = float(
            stats.binomtest(
                min(b, c), n_discordant, 0.5, alternative="two-sided"
            ).pvalue
        )
        return {
            "b": b,
            "c": c,
            "n_discordant": n_discordant,
            "statistic": float(min(b, c)),
            "p_value": p_value,
            "method": "exact binomial",
        }

    statistic = (abs(b - c) - 1.0) ** 2 / n_discordant
    p_value = float(stats.chi2.sf(statistic, df=1))
    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "statistic": float(statistic),
        "p_value": p_value,
        "method": "chi-square (continuity corrected)",
    }
