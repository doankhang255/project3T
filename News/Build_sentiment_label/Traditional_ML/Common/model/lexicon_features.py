"""Financial-lexicon category features (ML_SUMMARY.qmd section 6, priority 1).

Promoted from ``experiment_Lexicon_features/`` after the feature block
cleared both a tune-split check AND an independent, previously-untouched
holdout check on the 1044-row ground truth (random_forest: tune
Delta=+0.024 CI[+0.007,+0.041] p=0.001, holdout Delta=+0.049
CI[+0.017,+0.082] p=0.003 - see ``ML_SUMMARY.qmd`` section 6.2
for the full validation history, including two earlier attempts that did
NOT replicate on holdout before this one did). Only ``model/random_forest.py``
passes this as ``extra_features`` to its production CV - logistic
regression and SVM never showed a reliable effect, and MultinomialNB /
ComplementNB can't take the negative ``net_polarity`` column.

Loads the 7 category word lists already built in
``Seed_set_Prepare/seed_round4`` (negative / positive / uncertainty /
litigious / strong_modal / weak_modal / constraining) plus
``negation_cue_words.txt``, and turns each article's token list into a small
numeric feature block:

    <category>_prop  = hits / total_tokenizer          (7 columns; "positive"
                        AND "negative" count only NON-negated hits - see
                        NEGATABLE_CATEGORIES below)
    net_polarity      = positive_prop - negative_prop   (1 column)
    coverage           = (positive_hits + negative_hits) / total_tokenizer
                          (1 column, using the same non-negated counts)

Negation rule: a lexicon hit is discarded if a negation cue occurs within
the ``NEGATION_WINDOW`` tokens immediately before it. Loughran & McDonald
(2011, section III) apply this only to the positive list - they argue "not
terrible earnings" does not occur in financial text, so negating the
negative list is unnecessary. This module deliberately deviates from that
and applies the same rule to BOTH ``positive`` and ``negative``
(``NEGATABLE_CATEGORIES``), as an empirical test of whether that assumption
holds for this Vietnamese equity-news corpus rather than taking it on
faith - both promotion-clearing checks used this symmetric version.

Lexicon words are NOT all single VNCoreNLP tokens - many entries are 3-6
syllables (e.g. ``co_quan_quan_ly``, ``boi_thuong_thiet_hai``), and VNCoreNLP
already merges 2-3 syllables into one token on its own (``co_quan``,
``quan_ly``), so a 4+ syllable entry usually spans 2+ *tokens* in
``Tokenize_content``. A plain single-token membership test misses these
silently: measured on the tune set, only 5.6-37.5% of each category's words
ever occurred as one exact token - most of the rest DO appear in the text,
just spanning multiple tokens. So matching builds every underscore-joined
n-gram of the document's token stream, up to the longest lexicon entry's
token-span (``_max_phrase_length``, computed from the loaded word lists, not
hardcoded), and tests those against the word sets - not just single tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare"
SEED_ROUND4_DIR = SEED_DIR / "seed_round4"
NEGATION_CUE_PATH = SEED_DIR / "negation_cue_words.txt"

CATEGORY_FILES = {
    "negative": SEED_ROUND4_DIR / "negative_word.txt",
    "positive": SEED_ROUND4_DIR / "positive_word.txt",
    "uncertainty": SEED_ROUND4_DIR / "uncertainty_word.txt",
    "litigious": SEED_ROUND4_DIR / "litigious_word.txt",
    "strong_modal": SEED_ROUND4_DIR / "strong_modal_word.txt",
    "weak_modal": SEED_ROUND4_DIR / "weak_modal_word.txt",
    "constraining": SEED_ROUND4_DIR / "constraining_word.txt",
}
CATEGORIES = list(CATEGORY_FILES)
NEGATION_WINDOW = 3  # tokens immediately preceding a hit
NEGATABLE_CATEGORIES = ("positive", "negative")

FEATURE_NAMES = [f"{name}_prop" for name in CATEGORIES] + ["net_polarity", "coverage"]
N_FEATURES = len(FEATURE_NAMES)


def _read_word_list(path: Path) -> set[str]:
    """Comma-separated, word-wrapped across lines (same format for every
    Seed_set_Prepare word list) - split each physical line on commas."""
    words: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            for token in line.strip().rstrip(",").split(","):
                token = token.strip().casefold()
                if token:
                    words.add(token)
    return words


def load_lexicon_categories() -> dict[str, set[str]]:
    return {name: _read_word_list(path) for name, path in CATEGORY_FILES.items()}


def load_negation_cues() -> set[str]:
    return _read_word_list(NEGATION_CUE_PATH)


def _max_phrase_length(categories: dict[str, set[str]], negation_cues: set[str]) -> int:
    """Longest lexicon/cue entry, in underscore-joined segments. Computed from
    the actual loaded lists (not hardcoded) so this stays correct if the
    seed lists grow - matching only up to a stale fixed length would silently
    reintroduce the same class of miss this function exists to fix."""
    all_words = set(negation_cues)
    for words in categories.values():
        all_words |= words
    if not all_words:
        return 1
    return max(word.count("_") + 1 for word in all_words)


def _phrases_in_span(folded_tokens: Sequence[str], max_n: int) -> set[str]:
    phrases: set[str] = set()
    length = len(folded_tokens)
    for start in range(length):
        for n in range(1, max_n + 1):
            end = start + n
            if end > length:
                break
            phrases.add("_".join(folded_tokens[start:end]))
    return phrases


def _document_feature_vector(
    tokens: Sequence[str],
    categories: dict[str, set[str]],
    negation_cues: set[str],
    max_phrase_n: int,
) -> np.ndarray:
    total = len(tokens)
    if total == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    folded = [str(token).casefold() for token in tokens]

    hits = dict.fromkeys(CATEGORIES, 0)
    kept = dict.fromkeys(NEGATABLE_CATEGORIES, 0)
    for start in range(total):
        for n in range(1, max_phrase_n + 1):
            end = start + n
            if end > total:
                break
            phrase = "_".join(folded[start:end])
            for name, words in categories.items():
                if name in NEGATABLE_CATEGORIES:
                    continue
                if phrase in words:
                    hits[name] += 1
            for name in NEGATABLE_CATEGORIES:
                if phrase not in categories[name]:
                    continue
                window = folded[max(0, start - NEGATION_WINDOW) : start]
                if not (_phrases_in_span(window, max_phrase_n) & negation_cues):
                    kept[name] += 1

    props = []
    for name in CATEGORIES:
        count = kept[name] if name in NEGATABLE_CATEGORIES else hits[name]
        props.append(count / total)

    positive_prop = props[CATEGORIES.index("positive")]
    negative_prop = props[CATEGORIES.index("negative")]
    net_polarity = positive_prop - negative_prop
    coverage = positive_prop + negative_prop
    return np.asarray(props + [net_polarity, coverage], dtype=np.float32)


def build_lexicon_feature_matrix(
    token_lists: Sequence[Sequence[str]],
    categories: dict[str, set[str]] | None = None,
    negation_cues: set[str] | None = None,
) -> np.ndarray:
    """``(n_docs, len(FEATURE_NAMES))`` array. Nothing here is *fit* from the
    ground-truth set - the word lists and the negation rule are both fixed
    ahead of time (from ``Seed_set_Prepare`` / Loughran & McDonald 2011), so
    computing this for a held-out fold (or the holdout split) leaks nothing.
    """
    categories = categories or load_lexicon_categories()
    negation_cues = negation_cues if negation_cues is not None else load_negation_cues()
    max_phrase_n = _max_phrase_length(categories, negation_cues)
    rows = [
        _document_feature_vector(tokens, categories, negation_cues, max_phrase_n)
        for tokens in token_lists
    ]
    return np.vstack(rows) if rows else np.zeros((0, N_FEATURES), dtype=np.float32)
