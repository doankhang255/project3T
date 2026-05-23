from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "News"
    / "Process_sentiment_label"
    / "phobert__sentiment_model_v1"
)

INPUT_PATH = PROJECT_ROOT / "data" / "equity_news_content_sentiment_ratios.parquet"

TEXT_COLUMN = "Tokenize_content"
LABEL_COLUMN = "sentiment_label"

MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 42


label2id = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}

id2label = {
    0: "negative",
    1: "positive",
    2: "neutral",
}


def clean_text(text):
    if text is None:
        return ""

    if isinstance(text, (list, tuple, np.ndarray)):
        return " ".join(str(token).strip() for token in text if str(token).strip())

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(text).strip()
    text = " ".join(text.split())
    return text


def prepare_dataframe(path):
    df = pd.read_parquet(path)

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    df = df.loc[df[TEXT_COLUMN].ne("")].copy()
    df["label"] = df[LABEL_COLUMN].map(label2id)

    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), LABEL_COLUMN].unique()
        raise ValueError(f"Invalid labels: {bad}")

    return df[[TEXT_COLUMN, "label"]]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    df = prepare_dataframe(INPUT_PATH)

    _, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    valid_dataset = valid_dataset.map(tokenize_batch, batched=True)
    valid_dataset = valid_dataset.remove_columns([TEXT_COLUMN])
    valid_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "eval_tmp"),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    eval_result = trainer.evaluate()
    print("\nEvaluation result:")
    print(eval_result)

    predictions = trainer.predict(valid_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["negative", "positive", "neutral"],
        )
    )


if __name__ == "__main__":
    main()