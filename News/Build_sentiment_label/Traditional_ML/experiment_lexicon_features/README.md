# experiment_lexicon_features/ — financial-lexicon category features (IMPROVEMENTS.md priority 1)

Standalone, same pattern as `../improve/` and `../experiment_vocab/`: this
folder imports pure helpers from the main pipeline and from `../improve`
(TF-IDF fit/transform, the leak-free repeated-CV harness, the bootstrap CI)
and adds only the lexicon-feature block. It does not edit `model/*.py`,
`TF_IDF.py` or `model/common.py`.

The one shared file it does change is `../improve/repeated_cv.py`: an
optional `extra_features` parameter on `run_single_cv` / `run_repeated_cv`,
default `None`. Every existing caller is unaffected — verified byte-identical
to the pre-change output (`0.632973, 0.580331, 0.617391` for
`naive_bayes` at n=152, 3 repeats, before and after).

## What it tests

Loughran & McDonald (2011) / Tetlock (2007): generic sentiment word lists
misclassify financial text; a finance-specific lexicon, split into
negative / positive / uncertainty / litigious / strong-modal / weak-modal /
constraining, carries real signal. `Seed_set_Prepare/seed_round4/` already has
a Vietnamese version of these 7 lists. This experiment asks: does appending
9 numeric columns derived from them (7 category proportions +
`net_polarity` + `coverage`, with LM's negation rule applied to the positive
list) improve macro-F1 on top of the existing TF-IDF matrix?

`lexicon_features.py` builds the 9-column block. Nothing in it is *fit* from
the ground truth — the word lists and the negation rule (cue word within 3
tokens before a positive hit) are fixed ahead of time — so computing it for a
held-out fold, or for the holdout split, leaks nothing.

## A real bug caught while building this (worth keeping in mind for any future `extra_features` use)

First smoke run: `logistic_regression` and `svm` came back **bit-identical**
to the baseline (not just "not significantly different" — literally the same
mean and std to 3 decimals). That is not a plausible result for continuous
features and was a sign to stop and check, not a preliminary result to trust.

Cause: the lexicon columns are proportions in roughly `[0, 0.06]`; the TF-IDF
weights they sit next to average `~0.5` (nonzero cells). A linear model with
fixed regularization effectively can't afford a large-enough coefficient on a
column at that scale, given `C=1`, so those 9 columns contributed almost
nothing to `predict_proba` (checked directly: max abs diff `0.0014` before
the fix) — never enough to flip an argmax over 3 classes across a 10-repeat
run. Random Forest didn't show this (tree splits don't care about scale),
which is what made the scale bug visible instead of silent.

Fix: `run_single_cv` now standardizes `extra_features` (z-score, mean/std
from the **train fold only**, same discipline as the TF-IDF idf) before
hstacking. Re-ran the same smoke test after the fix — all three models
produce distinct, non-degenerate deltas. This standardization step is now
part of `extra_features` generally, not specific to this experiment — any
future extra numeric block (e.g. the SO-PMI feature in `IMPROVEMENTS.md`
priority 3) gets it automatically.

## A second bug: pointed at the wrong seed folder

This experiment originally loaded word lists from `Seed_set_Prepare/final_seed/`.
Despite the name, `final_seed` is an **older, much smaller snapshot** —
verified every `final_seed` word is a subset of `seed_round4`'s (round4 is the
one actively being edited; e.g. `negative_word.txt` 99 vs 356 words,
`litigious_word.txt` 81 vs 587). `lexicon_features.py` now points at
`seed_round4/` instead.

This was not a cosmetic fix — with the fuller word lists, `random_forest`'s
`Δ(macro-F1)` CI now **excludes 0** (see current `RESULTS.txt`), where it
previously overlapped 0. Two things changed between the old and the corrected
`RESULTS.txt` (the seed folder here, and separately `model/random_forest.py` /
`model/svm.py` picked up isotonic calibration / margin-softmax in an unrelated
change — see `../IMPROVEMENTS.md` section D), so the size of the shift can't
be attributed to the seed fix alone from those two numbers - but the current
run is internally valid (baseline and +lexicon arm use the identical, current
`build_estimator` for each model), and it is the number that matters for the
promotion decision below.

## A third bug: single-token matching missed most multi-syllable entries

`lexicon_features.py` checked each lexicon word against single VNCoreNLP
tokens. But VNCoreNLP itself already merges 2-3 syllables into one token
(`co_quan`, `quan_ly`), so any 4+ syllable lexicon entry (`co_quan_quan_ly`,
904 of the 1731 words across all 7 categories have 3+ underscore segments) is
often split across 2+ *tokens* - a single-token check can never match it. Measured
directly on the tune set: only 5.6-37.5% of each category's words ever
occurred as one exact token; most of the rest genuinely appear in the text,
just spanning multiple tokens.

Fix: `_document_feature_vector` now builds every underscore-joined n-gram of
the document's tokens (length 1 up to the longest lexicon/cue entry's token
span, computed from the loaded lists via `_max_phrase_length` - not
hardcoded) and matches those, including for the negation-cue window. Verified
the fix actually catches more: nonzero-document counts per category rose
across the board (e.g. `strong_modal` 75→85 docs, `weak_modal` 68→80,
`constraining` 42→50 out of 419) — real recall gained on genuinely rare
categories, not just noise. Practical effect on the tune-set macro-F1 result
was modest (`random_forest` Δ stayed ≈+0.027) - the newly-caught phrases are
inherently rarer in this short-announcement-style news corpus than the
already-matching short words.

## Negation extended to `negative_prop` too

Loughran & McDonald (2011) only negate the **positive** list (they argue "not
terrible earnings" doesn't occur in financial text, so negating negative
words is unnecessary). By explicit request, `NEGATABLE_CATEGORIES = ("positive",
"negative")` now applies the same negation-window rule to both - combined
into the one feature block, not a separate arm; there is no
positive-only-negation variant left to compare against.

## Promotion attempt #2 - still doesn't clear holdout

This is a **second look at the same 180-row holdout**, for a different
feature design (negation added to `negative_prop`) than promotion attempt #1
above. Re-checking a holdout against a modified design after a first attempt
failed is exactly the "keep tweaking until holdout says yes" pattern the
tune/holdout split exists to prevent - done here because it was explicitly
requested, not as a template to repeat. Recorded honestly:

- Tune (419 rows): `random_forest` Δ=+0.029, 95% CI [+0.006, +0.053], p=0.011
  - excludes 0, again.
- Holdout (180 rows): `random_forest` Δ=+0.012, 95% CI **[-0.014, +0.040]**,
  p=0.381 - **does not replicate**, again.

Same failure pattern as promotion attempt #1 (tune says yes, holdout says
no), now observed for **two different feature designs** in a row. That
consistency is itself informative: it points at the tune split's own 419-row
sample being the source of the "random_forest improves" signal (i.e. some
quirk of that particular partition that isotonic-calibrated RF picks up on),
rather than anything about the lexicon feature or the negation rule
specifically. **Still do not promote.** The holdout has now been looked at
twice for this feature family - a third redesign-and-recheck cycle on this
same 180-row split would no longer be a meaningful check at all.

## Run

```bash
python News/Build_sentiment_label/Traditional_ML/experiment_lexicon_features/run_experiment.py
python .../run_experiment.py --repeats 3 --n-boot 400   # quick smoke run
python .../run_experiment.py --ground-truth-csv <path>  # override the ground truth (default: tune split)
```

Runs on `../improve/tune_holdout/ground_truth_tune.csv` by default (744 rows
as of the 1064-row ground truth - regenerate via `../improve/tune_holdout.py`
whenever the source grows) - this is exploratory feature engineering, so it
stays on the **tune** split. Do not run this against the holdout split while
iterating; only re-check there once, after a design is settled.

Outputs (all in this folder):

- `RESULTS.txt` — baseline vs +lexicon macro-F1 (mean ± std, per model),
  per-class F1, and the paired bootstrap `Δ(macro-F1)` with 95% CI + p-value.
- `data/per_repeat.csv` — every (model, arm, repeat) row.
- `data/bootstrap_delta.csv` — the pairwise delta table.

## Scope left out on purpose

- **MultinomialNB / ComplementNB** are excluded from the "+lexicon" arm:
  both require non-negative input, and `net_polarity` can be negative.
  Clipping or using only the non-negative columns for the NB family is a
  follow-up, not done here.
- Only the raw-proportion variant is tested (`IMPROVEMENTS.md` priority 1
  also mentions a tf-idf-weighted variant) — left for a follow-up if the
  raw-proportion version shows a reliable gain.

## Promotion path

If a model's `Δ(macro-F1)` CI reliably excludes 0 on the tune split, re-check
**once** on `../improve/tune_holdout/ground_truth_holdout.csv` before folding
the feature block into `model/common.py` (as an opt-in on
`run_cross_validation`, following the same `extra_features` parameter added
here) and regenerating `RESULTS_SUMMARY.txt`.

**Status: PROMOTED into `model/common.py` for `random_forest`, after 2
failures and a successful replication on a genuinely fresh holdout.**

```bash
python News/Build_sentiment_label/Traditional_ML/experiment_lexicon_features/run_experiment.py \
    --ground-truth-csv News/Build_sentiment_label/Traditional_ML/improve/tune_holdout/ground_truth_holdout.csv \
    --out-dir News/Build_sentiment_label/Traditional_ML/experiment_lexicon_features/holdout_check
```

| attempt | ground truth | feature design | tune Δ (random_forest) | holdout Δ (random_forest) | replicated? |
|---|---|---|---|---|---|
| #1 | 599 (419/180) | negation on `positive` only | +0.027, CI [+0.004,+0.051], p=0.021 | +0.003, CI [-0.023,+0.031], p=0.794 | No |
| #2 | 599 (419/180) | negation on `positive` AND `negative` | +0.029, CI [+0.006,+0.053], p=0.011 | +0.012, CI [-0.014,+0.040], p=0.381 | No |
| **#3** | **1064 (744/320, brand-new holdout)** | negation on both (same as #2) | +0.016, CI [+0.001,+0.033], p=0.039 | **+0.038, CI [+0.011,+0.066], p=0.004** | **Yes** |

Attempts #1/#2 both failed on the SAME 180-row holdout, pointing at that
particular split as the likely source of the earlier "random_forest
improves" tune signal rather than the feature itself. Attempt #3 used more
ground truth (1064 rows) and, critically, a **holdout that had never been
looked at before** (320 rows, freshly drawn when the split file was
regenerated for the larger ground truth) - and it replicated, with an even
larger effect on holdout than on tune. `logistic_regression` and `svm` never
showed a reliable effect across all 3 attempts; `MultinomialNB`/`ComplementNB`
remain excluded (`net_polarity` can be negative).

**Promoted:**
- `lexicon_features.py` moved to `model/lexicon_features.py` (canonical
  location - this folder's copy was deleted, not duplicated).
- `model/common.py::run_cross_validation` gained the `extra_features`
  parameter (same standardize-in-fold logic already validated in
  `improve/repeated_cv.py`).
- `model/random_forest.py` computes the lexicon matrix and passes it as
  `extra_features` - the only model with evidence for it.
- `model/ensemble.py` passes the same matrix to its `random_forest` member,
  so the ensemble does not silently score a different (TF-IDF-only) RF than
  what `random_forest_metrics.csv` reports.
- Production result on the full 1064-row set: `random_forest` macro-F1
  0.645 → **0.663** (+0.018, same direction as the validated effect).

This folder (`experiment_lexicon_features/`) is kept as the historical
record of the validation - `run_experiment.py` now imports
`build_lexicon_feature_matrix` from `model/lexicon_features.py` rather than
a local copy.
