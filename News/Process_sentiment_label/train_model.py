from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:
    raise ImportError(
        "Missing training dependencies. Install torch, transformers, and scikit-learn first."
    ) from exc


INPUT_PATH = Path(__file__).with_name("equity_news_tokenized.parquet")
OUTPUT_DIR = Path(__file__).with_name("phobert_sentiment_model")
TEXT_COLUMN = "Tokenize_des"
LABEL_COLUMN = "sentiment_label"
MODEL_NAME = "vinai/phobert-base"
DEFAULT_LABEL_ORDER = ["negative", "neutral", "positive"]


class NewsSentimentDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def read_input_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .parquet or .csv")


def normalize_text_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        return " ".join(str(token) for token in value.tolist())
    if isinstance(value, list):
        return " ".join(str(token) for token in value)
    if pd.isna(value):
        return ""

    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, list):
            return " ".join(str(token) for token in parsed)
    return text


def build_label_mapping(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    label_values = sorted(labels.astype("string").str.strip().str.lower().unique())
    known_labels = [label for label in DEFAULT_LABEL_ORDER if label in label_values]

    ordered_labels = known_labels if set(label_values).issubset(DEFAULT_LABEL_ORDER) else label_values
    label_to_id = {label: index for index, label in enumerate(ordered_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def prepare_training_dataframe(
    path: Path = INPUT_PATH,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    df = read_input_dataframe(path)
    missing_columns = [column for column in [text_column, label_column] if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required column(s): {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = df[[text_column, label_column]].copy()
    out["text"] = out[text_column].apply(normalize_text_value)
    out["label_text"] = out[label_column].astype("string").str.strip().str.lower()
    out = out.loc[out["text"].ne("") & out["label_text"].notna()].reset_index(drop=True)

    label_to_id, id_to_label = build_label_mapping(out["label_text"])
    out["label"] = out["label_text"].map(label_to_id)
    out = out.loc[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)

    if out.empty:
        raise ValueError("No training rows remain after cleaning text and labels.")
    return out, label_to_id, id_to_label


def compute_metrics(eval_pred: object) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
    }


def build_training_arguments(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    fp16: bool,
) -> TrainingArguments:
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "fp16": fp16,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**training_kwargs)


def train_phobert_sentiment_model(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    model_name: str = MODEL_NAME,
    validation_size: float = 0.20,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    seed: int = 42,
    fp16: bool = False,
) -> Trainer:
    set_seed(seed)
    df, label_to_id, id_to_label = prepare_training_dataframe(
        path=input_path,
        text_column=text_column,
        label_column=label_column,
    )

    stratify_labels = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, valid_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        max_length=max_length,
    )
    valid_encodings = tokenizer(
        valid_df["text"].tolist(),
        truncation=True,
        max_length=max_length,
    )

    train_dataset = NewsSentimentDataset(train_encodings, train_df["label"].tolist())
    valid_dataset = NewsSentimentDataset(valid_encodings, valid_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label={index: label for index, label in id_to_label.items()},
    )
    training_args = build_training_arguments(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_mapping_path = output_dir / "label_mapping.json"
    label_mapping_path.write_text(
        json.dumps(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for news sentiment.")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--text-column", default=TEXT_COLUMN)
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    trainer = train_phobert_sentiment_model(
        input_path=args.input_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        validation_size=args.validation_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        fp16=args.fp16,
    )
    print("Training finished.")
    print(trainer.state.log_history[-1] if trainer.state.log_history else {})


if __name__ == "__main__":
    main()
