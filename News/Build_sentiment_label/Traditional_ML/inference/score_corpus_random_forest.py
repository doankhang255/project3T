"""Apply the persisted production Random Forest (+lexicon features,
``train_and_save_random_forest.py``) to the FULL 126.576-article corpus, so
it has a per-article sentiment score on equal footing with the Lexicon
(``Lexicon_based/Scoring*/data/article_scores*.parquet``) and PhoBERT
(``Transfer_Learning/inference/score_corpus_phobert.py``) arms - the 3-way
comparison this whole inference/ folder exists for (see the day-level
regression pipelines under ``News_Vnindex/*/daily/``).

Same feature recipe as training, replayed with the FITTED artifacts (no
re-fitting anything on the corpus - that would leak the corpus into the
"vocabulary"/"lexicon scaling" the model was trained with):
    1. n-gram document terms (Common/TF_IDF.py::build_document_terms) -> TF-IDF
       via the persisted vocabulary_df, sliced to the persisted
       selected_indices.
    2. Lexicon category features (Common/model/lexicon_features.py), scaled
       with the persisted train mean/std.
    3. hstack the two blocks, exactly as at train time.

``net_sentiment_score = prob_positive - prob_negative`` - same name/formula as
the Lexicon and PhoBERT arms, so
``News/Build_sentiment_index/build_sentiment_index_daily.py`` (the shared
daily aggregation for every method) just lists this file as one more input.

Processed in batches (default 10,000 rows) to bound peak memory - the TF-IDF
transform builds a dense (batch_rows x ~2,900 vocab) float32 matrix per batch.

Run::

    python News/Build_sentiment_label/Traditional_ML/inference/score_corpus_random_forest.py
    ... --max-docs 5000   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import (  # noqa: E402
    build_document_term_counts,
    build_document_terms,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (  # noqa: E402
    build_lexicon_feature_matrix,
)


TOKENIZED_CORPUS_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_production.joblib"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = OUTPUT_DIR / "article_scores_random_forest.parquet"
OUTPUT_CSV_SAMPLE_PATH = OUTPUT_DIR / "article_scores_random_forest_sample.csv"

VALID_LABELS = ["negative", "neutral", "positive"]
METADATA_COLUMNS = ["link", "publication_date", "domain_norm", "title", "total_tokenizer"]
REQUIRED_COLUMNS = METADATA_COLUMNS + ["Tokenize_content", "Tokenize_content_sentences"]

BATCH_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Cap số bài (smoke test). Mặc định: toàn bộ."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def score_batch(
    batch_df: pd.DataFrame,
    vocabulary_df: pd.DataFrame,
    selected_indices: np.ndarray,
    lexicon_mean: np.ndarray,
    lexicon_std: np.ndarray,
    model,
) -> np.ndarray:
    batch_df = batch_df.copy()
    batch_df["_document_terms"] = batch_df.apply(build_document_terms, axis=1)
    term_counts = build_document_term_counts(batch_df)

    x_full, _ = transform_tfidf(term_counts, vocabulary_df)
    x_selected = x_full[:, selected_indices]

    lexicon_matrix = build_lexicon_feature_matrix(batch_df["Tokenize_content"].tolist())
    lexicon_scaled = (lexicon_matrix - lexicon_mean) / lexicon_std

    x_batch = np.hstack([x_selected, lexicon_scaled])
    return model.predict_proba(x_batch)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print("Model:", MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    vocabulary_df = artifact["vocabulary_df"]
    selected_indices = artifact["selected_indices"]
    lexicon_mean = artifact["lexicon_mean"]
    lexicon_std = artifact["lexicon_std"]
    print(f"Trained on {artifact['n_train_rows']} rows | vocab {len(vocabulary_df)} | "
          f"selected features {len(selected_indices)}")

    print("Đọc corpus đã tokenize (có thể mất vài phút) ...")
    corpus_df = pd.read_parquet(TOKENIZED_CORPUS_PATH, columns=REQUIRED_COLUMNS)
    if args.max_docs is not None:
        corpus_df = corpus_df.iloc[: args.max_docs].reset_index(drop=True)
    n_docs = len(corpus_df)
    print(f"  {n_docs:,} bài báo")

    print(f"\nChấm điểm theo batch={args.batch_size} ...")
    started = time.perf_counter()
    all_probs: list[np.ndarray] = []
    for start in range(0, n_docs, args.batch_size):
        batch_df = corpus_df.iloc[start : start + args.batch_size].reset_index(drop=True)
        probs = score_batch(batch_df, vocabulary_df, selected_indices, lexicon_mean, lexicon_std, model)
        all_probs.append(probs)

        done = min(start + args.batch_size, n_docs)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (n_docs - done) / rate if rate > 0 else float("nan")
        print(f"  ... {done:,}/{n_docs:,} bài ({rate:.1f} bài/s, còn ~{remaining / 60:.1f} phút)")

    probs = np.vstack(all_probs)
    print(f"Xong trong {time.perf_counter() - started:.1f}s")

    result_df = corpus_df[METADATA_COLUMNS].copy().reset_index(drop=True)
    for label_id, label in enumerate(VALID_LABELS):
        result_df[f"prob_{label}"] = probs[:, label_id]
    predicted_ids = probs.argmax(axis=1)
    result_df["predicted_label"] = [VALID_LABELS[i] for i in predicted_ids]
    result_df["net_sentiment_score"] = result_df["prob_positive"] - result_df["prob_negative"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)
    print("\nĐã lưu kết quả đầy đủ vào:", OUTPUT_PATH)
    print("Phân bố nhãn dự đoán:")
    print(result_df["predicted_label"].value_counts().to_string())

    sample_pos = result_df.nlargest(15, "net_sentiment_score").assign(sample_group="top_positive")
    sample_neg = result_df.nsmallest(15, "net_sentiment_score").assign(sample_group="top_negative")
    sample_random = result_df.sample(n=min(15, len(result_df)), random_state=42).assign(
        sample_group="random"
    )
    sample_df = pd.concat([sample_pos, sample_neg, sample_random], ignore_index=True)
    sample_df.to_csv(OUTPUT_CSV_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print("Đã lưu mẫu review nhanh (45 bài) vào:", OUTPUT_CSV_SAMPLE_PATH)


if __name__ == "__main__":
    main()
