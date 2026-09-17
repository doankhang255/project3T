"""Áp dụng model E3 (fine-tune toàn phần, `Model_Output/phobert_finetuned_ground_truth`)
lên TOÀN BỘ corpus 126.576 bài báo (không chỉ 1.064 dòng ground truth), để có
1 điểm sentiment cho mỗi bài - phục vụ dựng chỉ số sentiment thị trường cấp
ngày và kiểm định tác động lên VN-Index, cùng lối đi mà Lexicon_based/Scoring
/score_articles.py đã làm cho phương pháp lexicon (`net_sentiment_score`).

Text đưa vào model giống hệt cách ``model/common.py::load_ground_truth`` build
cột ``text`` khi huấn luyện/đánh giá: nối các token của cột phẳng
``Tokenize_content`` bằng dấu cách (không dùng bản theo câu
``Tokenize_content_sentences``), rồi tokenize/truncate max_length=256 - đúng
những gì model đã thấy lúc train, không tự tạo thêm một cách xử lý văn bản
khác.

``net_sentiment_score = prob_positive - prob_negative`` - đặt cùng tên cột và
cùng công thức với Lexicon's ``score_articles.py`` để
``News/Build_sentiment_index/build_sentiment_index_daily.py`` và
``News_Vnindex/Common/merge_vnindex_daily_with_sentiment.py`` (đều đã viết
tổng quát theo (publication_date, net_sentiment_score)) dùng lại được nguyên
vẹn, không phải sửa.

Chạy (dùng venv gốc của repo, có torch/transformers)::

    venv/bin/python News/Build_sentiment_label/Transfer_Learning/inference/score_corpus_phobert.py

Smoke test nhanh (2000 bài đầu)::

    ... score_corpus_phobert.py --max-docs 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (  # noqa: E402
    MAX_LENGTH,
    VALID_LABELS,
)


TOKENIZED_CORPUS_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "News"
    / "Build_sentiment_label"
    / "Transfer_Learning"
    / "Model_Output"
    / "phobert_finetuned_ground_truth"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = OUTPUT_DIR / "article_scores_phobert.parquet"
OUTPUT_CSV_SAMPLE_PATH = OUTPUT_DIR / "article_scores_phobert_sample.csv"

BATCH_SIZE = 64
TOKENIZE_COLUMN = "Tokenize_content"
METADATA_COLUMNS = ["link", "publication_date", "domain_norm", "title", "total_tokenizer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Cap số bài (smoke test). Mặc định: toàn bộ."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def build_texts(tokenize_column: pd.Series) -> list[str]:
    return [
        " ".join(str(token) for token in tokens if str(token).strip())
        for tokens in tokenize_column
    ]


@torch.no_grad()
def score_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Trả về ma trận (n_docs, 3) xác suất [negative, neutral, positive] -
    đúng thứ tự VALID_LABELS."""
    all_probs: list[np.ndarray] = []
    n_docs = len(texts)
    started = time.perf_counter()
    for start in range(0, n_docs, batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        all_probs.append(probs)

        done = min(start + batch_size, n_docs)
        if done % (batch_size * 50) == 0 or done == n_docs:
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (n_docs - done) / rate if rate > 0 else float("nan")
            print(
                f"  ... {done}/{n_docs} bài ({rate:.1f} bài/s, "
                f"còn ~{remaining / 60:.1f} phút)"
            )
    return np.vstack(all_probs)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    on_cuda = torch.cuda.is_available()
    on_mps = (not on_cuda) and torch.backends.mps.is_available()
    device = torch.device("cuda" if on_cuda else "mps" if on_mps else "cpu")
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Model      :", MODEL_PATH)
    print("Device     :", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()
    print("Label map  :", model.config.id2label)
    assert [model.config.id2label[i] for i in range(len(VALID_LABELS))] == VALID_LABELS, (
        "Label order mismatch giữa model.config.id2label và VALID_LABELS."
    )

    print("Đọc corpus đã tokenize (có thể mất vài phút) ...")
    corpus_df = pd.read_parquet(TOKENIZED_CORPUS_PATH, columns=METADATA_COLUMNS + [TOKENIZE_COLUMN])
    if args.max_docs is not None:
        corpus_df = corpus_df.iloc[: args.max_docs].reset_index(drop=True)
    print(f"  {len(corpus_df):,} bài báo")

    texts = build_texts(corpus_df[TOKENIZE_COLUMN])

    print(f"\nChạy forward-pass (batch_size={args.batch_size}) ...")
    forward_started = time.perf_counter()
    probs = score_texts(texts, tokenizer, model, device, args.batch_size)
    print(f"Xong trong {time.perf_counter() - forward_started:.1f}s")

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
