"""E1 - Domain-adaptive pretraining (DAPT) for PhoBERT on the equity-news corpus.

What it does
------------
1. Read ``equity_news_tokenized_vncorenlp.parquet`` column
   ``Tokenize_content_sentences`` - the article already split into sentences,
   each a list of word-segmented tokens with ``_``-joined compounds (the
   VNCoreNLP/RDRSegmenter segmentation PhoBERT was pretrained on).
2. BPE-tokenize each sentence, then greedily pack *whole sentences* into blocks
   of up to ``MAX_SEQ_LENGTH`` tokens - never splitting a sentence, never
   crossing an article boundary. Every training sequence is therefore
   linguistically coherent (unlike a blind 256-token window that cuts
   mid-sentence and can merge two unrelated articles), while still staying long
   enough to give MLM real bidirectional context. Rare sentences longer than a
   block are hard-split as a fallback.
3. Hold out ``VAL_FRACTION`` of the blocks to track eval loss / perplexity - the
   signal that DAPT is actually adapting the model to the domain.
4. Train ``AutoModelForMaskedLM`` starting from the local ``vinai/phobert-base-v2``
   snapshot, with dynamic 15% masking (``DataCollatorForLanguageModeling``).
5. Save the adapted model + tokenizer to ``Model_Output/phobert_domain_adapted/``
   (this becomes the base checkpoint for E2 frozen-feature extraction and E3
   full fine-tuning) and append a row to ``data/phobert_dapt_eval.csv``.

The tokenized+blocked dataset is cached under ``Model_Output/phobert_dapt_cache/``
so re-runs (e.g. trying more epochs) skip the preprocessing.

Run (from the repo root, using the GPU venv)::

    News/Build_sentiment_label/Transfer_Learning/Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/pretrain/domain_adaptive_pretrain.py

Quick smoke test on 2000 docs / 1 epoch::

    ... domain_adaptive_pretrain.py --max-docs 2000 --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_LEARNING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_PARQUET = (
    PROJECT_ROOT
    / "data_news"
    / "data_tokenized"
    / "equity_news_tokenized_vncorenlp.parquet"
)
# List-of-lists: outer = sentences in the article, inner = tokens in the sentence.
SENTENCES_COLUMN = "Tokenize_content_sentences"

# vinai/phobert-base-v2 is the *base* masked-LM (not the wonrax sentiment head)
# - the right starting point for continued MLM pretraining. Already in the local
# HF hub cache, so no download / network needed.
BASE_MODEL_PATH = (
    TRANSFER_LEARNING_DIR
    / "Model_Output"
    / "hub"
    / "models--vinai--phobert-base-v2" three thousand two hundred forty nine dollars start FTMO Challenge a character. Ay shump sheet Phương Hàn gì anhcoi rồi chết phòng chờ chết bắt buộc share might. Another turned the same ten percent into fifty four euros same scale completely different account. Prove your edge in the FTMO challenge. Learn more with FTMO chuyen to anhering came trouble have you ever wondered what happens if you could generate a full thirty second cinematic story from just one single text prompts most i models tap out at five or ten seconds before the character start melting our physics fall apart but today everything changes because ingoing thought a train sheet stand out in the job market with the new Google AI professional certificate master effective prompting and responsible AI use partner with AI to transform your research, communications, and more start automating your workflows to save time and streamline your tasks. Go hands on with Google's most capable AI tools. Build smarter workflows, create AI powered solutions, and become AI fluent in just ten hours. Enroll today on Corsera Managed WordPress hosting from hostinger is the easiest way to launch, manage, and grow a WordPress website. Focus on the important bits, while powerful AI tools strip away the complexity and do the hard work for you. You're not the only one being backed up. Top tier security tools and automatic backups keeps your website fully secure. Join the three million people who trust us to help make their online dreams a reality. Head to hostinger.com to get started recommended by WordPress.org trans. Review let YouTube, then
    / "snapshots"
    / "e2375d266bdf39c6e8e9a87af16a5da3190b0cc8"
)

OUTPUT_DIR = TRANSFER_LEARNING_DIR / "Model_Output" / "phobert_domain_adapted"
CACHE_DIR = TRANSFER_LEARNING_DIR / "Model_Output" / "phobert_dapt_cache"
EVAL_CSV = TRANSFER_LEARNING_DIR / "data" / "phobert_dapt_eval.csv"

# PhoBERT's positional table is 258 wide -> 256 usable tokens after the two
# special tokens. Same length model_sentiment_v1/v2 and finetune_phobert.py use.
MAX_SEQ_LENGTH = 256
MLM_PROBABILITY = 0.15
VAL_FRACTION = 0.05
MIN_TOKENS_PER_DOC = 5
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap the number of articles (smoke tests). Default: use all.",
    )
    parser.add_argument(
        "--map-num-proc",
        type=int,
        default=1,
        help="Processes for datasets.map tokenization. >1 is faster but on "
        "Windows can be flaky; bump only if it works on your box.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision entirely (train in fp32).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 mixed precision instead of the bf16 default (bf16 is "
        "native on this RTX 5060 / Blackwell and needs no loss scaler).",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore any cached tokenized dataset and rebuild it.",
    )
    return parser.parse_args()


def load_corpus_sentences(max_docs: int | None) -> list[list[str]]:
    """Return one entry per article: a list of sentence strings (tokens joined
    by spaces). Articles with fewer than MIN_TOKENS_PER_DOC tokens total are
    dropped.
    """
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Tokenized corpus not found: {INPUT_PARQUET}")

    frame = pd.read_parquet(INPUT_PARQUET, columns=[SENTENCES_COLUMN])
    print(f"Loaded {len(frame):,} rows from {INPUT_PARQUET.name}")

    articles: list[list[str]] = []
    for sentences in frame[SENTENCES_COLUMN]:
        if sentences is None or len(sentences) == 0:
            continue
        cleaned = []
        total_tokens = 0
        for sentence in sentences:
            tokens = [str(tok) for tok in sentence if str(tok).strip()]
            if tokens:
                cleaned.append(" ".join(tokens))
                total_tokens += len(tokens)
        if cleaned and total_tokens >= MIN_TOKENS_PER_DOC:
            articles.append(cleaned)

    print(f"Kept {len(articles):,} articles with >= {MIN_TOKENS_PER_DOC} tokens")

    if max_docs is not None:
        articles = articles[:max_docs]
        print(f"--max-docs: truncated to {len(articles):,} articles")

    return articles


def build_block_dataset(
    articles: list[list[str]],
    tokenizer,
    map_num_proc: int,
    rebuild_cache: bool,
) -> Dataset:
    """BPE-tokenize every sentence, then greedily pack whole sentences into
    <= MAX_SEQ_LENGTH blocks without splitting a sentence or crossing an
    article boundary. Cached to disk keyed by corpus size + sequence length.
    """
    cache_path = CACHE_DIR / f"blocks_sent_{len(articles)}docs_{MAX_SEQ_LENGTH}tok"
    if cache_path.exists() and not rebuild_cache:
        print(f"Loading cached blocked dataset from {cache_path}")
        return Dataset.load_from_disk(str(cache_path))

    # Flatten sentences (keeping which article each belongs to) so the slow
    # PhoBERT BPE runs once over one batched dataset instead of per article.
    flat_sentences: list[str] = []
    article_of_sentence: list[int] = []
    for article_idx, sentences in enumerate(articles):
        for sentence in sentences:
            flat_sentences.append(sentence)
            article_of_sentence.append(article_idx)

    sentence_dataset = Dataset.from_dict({"text": flat_sentences})

    def tokenize_fn(batch: dict) -> dict:
        return {"ids": tokenizer(batch["text"], add_special_tokens=False)["input_ids"]}

    sentence_dataset = sentence_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=map_num_proc,
        remove_columns=["text"],
        desc="BPE tokenizing sentences",
    )
    sentence_ids = sentence_dataset["ids"]

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    content_len = MAX_SEQ_LENGTH - 2  # room for the leading bos + trailing eos

    input_ids_blocks: list[list[int]] = []
    attention_masks: list[list[int]] = []
    buffer: list[int] = []
    current_article = -1

    def flush() -> None:
        if buffer:
            sequence = [bos, *buffer, eos]
            input_ids_blocks.append(sequence)
            attention_masks.append([1] * len(sequence))
            buffer.clear()

    for article_idx, ids in zip(article_of_sentence, sentence_ids):
        if article_idx != current_article:
            flush()
            current_article = article_idx

        if len(ids) > content_len:
            # Rare (~1 in 80k): a single sentence longer than a block.
            flush()
            for start in range(0, len(ids), content_len):
                piece = ids[start : start + content_len]
                sequence = [bos, *piece, eos]
                input_ids_blocks.append(sequence)
                attention_masks.append([1] * len(sequence))
            continue

        if len(buffer) + len(ids) > content_len:
            flush()
        buffer.extend(ids)
    flush()

    blocks = Dataset.from_dict(
        {"input_ids": input_ids_blocks, "attention_mask": attention_masks}
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    blocks.save_to_disk(str(cache_path))
    print(f"Cached {len(blocks):,} blocks to {cache_path}")
    return blocks


def append_eval_row(row: dict) -> None:
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_CSV.exists()
    with EVAL_CSV.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        # Blackwell has fast TF32 matmul; free accuracy-neutral speedup.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_bf16 = on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print("Base model :", BASE_MODEL_PATH)
    print("Device     :", torch.cuda.get_device_name(0) if on_cuda else "cpu", "| precision:", precision)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

    articles = load_corpus_sentences(args.max_docs)
    blocks = build_block_dataset(articles, tokenizer, args.map_num_proc, args.rebuild_cache)

    split = blocks.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    train_dataset = split["train"].with_format("torch")
    eval_dataset = split["test"].with_format("torch")
    print(f"Blocks: {len(blocks):,} total | train {len(train_dataset):,} | val {len(eval_dataset):,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        # 0 on purpose: on Windows, DataLoader worker processes (spawn + pickle
        # the Arrow dataset) add far more overhead than they save here.
        dataloader_num_workers=0,
        dataloader_pin_memory=on_cuda,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    baseline = trainer.evaluate()
    baseline_ppl = math.exp(baseline["eval_loss"]) if baseline["eval_loss"] < 20 else float("inf")
    print(f"\nBaseline (phobert-base-v2, no adaptation): eval_loss={baseline['eval_loss']:.4f} "
          f"| perplexity={baseline_ppl:.2f}")

    trainer.train()

    final = trainer.evaluate()
    final_ppl = math.exp(final["eval_loss"]) if final["eval_loss"] < 20 else float("inf")
    print(f"\nAfter DAPT: eval_loss={final['eval_loss']:.4f} | perplexity={final_ppl:.2f}")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Saved domain-adapted model to:", OUTPUT_DIR)

    append_eval_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_docs": len(articles),
            "n_blocks": len(blocks),
            "packing": "sentence",
            "precision": precision,
            "max_seq_length": MAX_SEQ_LENGTH,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "baseline_eval_loss": round(baseline["eval_loss"], 5),
            "baseline_perplexity": round(baseline_ppl, 4),
            "final_eval_loss": round(final["eval_loss"], 5),
            "final_perplexity": round(final_ppl, 4),
        }
    )
    print("Appended eval summary to:", EVAL_CSV)


if __name__ == "__main__":
    main()
