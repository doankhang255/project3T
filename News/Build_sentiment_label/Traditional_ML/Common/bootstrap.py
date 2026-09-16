"""M2.4 - bootstrap confidence interval on macro-F1 and on the macro-F1 gap
between two models.

McNemar (``mcnemar.py``) tests the *accuracy* gap - per-row correct/wrong.
The Traditional_ML models are ranked by **macro-F1**, which on this set is
dominated by the 35-row ``positive`` class, so an accuracy tie does not imply
a macro-F1 tie. This resamples the 152 evaluation rows with replacement and
recomputes macro-F1, giving a CI on the quantity actually being ranked.

Design: one shared set of ``n_boot`` row-index draws is reused for every
model, so ``samples[a] - samples[b]`` is a *paired* bootstrap of the gap
(same articles in/out of each resample). Every repeat's out-of-fold
prediction is scored on the resample and averaged, so all repeats are used.
"""

from __future__ import annotations

import numpy as np

BOOT_SEED = 20260908
N_BOOT = 2000


def macro_f1_per_repeat(
    y_true: np.ndarray, oof: np.ndarray, n_labels: int = 3
) -> np.ndarray:
    """macro-F1 of each repeat's prediction vector.

    ``oof`` has shape ``(n_repeats, n_rows)`` (label ids); returns ``(n_repeats,)``.
    Matches ``model/common.compute_metrics``: per-class F1 = 2*tp / (2*tp+fp+fn),
    0 when that class is absent from both truth and prediction; macro-F1 = mean
    over the ``n_labels`` classes.
    """
    oof = np.atleast_2d(oof)
    truth = y_true[None, :]
    macro = np.zeros(oof.shape[0], dtype=float)
    for label_id in range(n_labels):
        true_pos = np.sum((truth == label_id) & (oof == label_id), axis=1)
        false_pos = np.sum((truth != label_id) & (oof == label_id), axis=1)
        false_neg = np.sum((truth == label_id) & (oof != label_id), axis=1)
        denom = 2 * true_pos + false_pos + false_neg
        f1_label = np.divide(
            2 * true_pos, denom, out=np.zeros(oof.shape[0]), where=denom > 0
        )
        macro += f1_label
    return macro / n_labels


def mean_macro_f1(y_true: np.ndarray, oof: np.ndarray, n_labels: int = 3) -> float:
    """Point estimate: macro-F1 averaged over the repeated-CV passes."""
    return float(macro_f1_per_repeat(y_true, oof, n_labels).mean())


def bootstrap_samples(
    y_true: np.ndarray,
    oof_by_model: dict[str, np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    n_labels: int = 3,
) -> dict[str, np.ndarray]:
    """One shared resample per iteration, scored for every model.

    Returns ``{model: array of shape (n_boot,)}`` - each entry is the
    resampled macro-F1 (averaged over repeats). Paired across models.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    names = list(oof_by_model)
    samples = {name: np.empty(n_boot, dtype=float) for name in names}
    for boot_index in range(n_boot):
        row_index = rng.integers(0, n_rows, n_rows)
        truth = y_true[row_index]
        for name in names:
            samples[name][boot_index] = macro_f1_per_repeat(
                truth, oof_by_model[name][:, row_index], n_labels
            ).mean()
    return samples


def ci(sample: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    low, high = np.quantile(sample, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def two_sided_p(delta_sample: np.ndarray) -> float:
    """Bootstrap two-sided p-value for H0: delta == 0 (fraction of resamples on
    the wrong side of 0, doubled)."""
    p = 2.0 * min(
        float(np.mean(delta_sample <= 0.0)), float(np.mean(delta_sample >= 0.0))
    )
    return min(p, 1.0)
