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
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "News"
    / "Process_sentiment_label"
    / "hub"
    / "models--wonrax--phobert-base-vietnamese-sentiment"
    / "snapshots"
    / "9076a5896971b5d551588fe8a51c722c89731d36"
)

INPUT_PATH = PROJECT_ROOT / "data_News" / "equity_news_content_sentiment_ratios.parquet"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "News"
    / "Process_sentiment_label"
    / "phobert_financial_sentiment_model_v2_class_weight"
)

TEXT_COLUMN = "Tokenize_content"
LABEL_COLUMN = "sentiment_label"

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
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
    if Path(path).suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    label_aliases = {
        "pos": "positive",
        "posi": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
    }

    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(label_aliases)

    if df[LABEL_COLUMN].isna().any():
        bad_labels = df.loc[df[LABEL_COLUMN].isna(), LABEL_COLUMN].unique()
        raise ValueError(f"Invalid labels: {bad_labels}")

    df = df.loc[df[TEXT_COLUMN].ne("")].copy()
    df["label"] = df[LABEL_COLUMN].map(label2id)

    return df[[TEXT_COLUMN, "label"]]


def compute_balanced_class_weights(labels: pd.Series) -> torch.Tensor:
    label_counts = labels.value_counts().sort_index()
    missing_labels = set(id2label).difference(label_counts.index)
    if missing_labels:
        raise ValueError(f"Missing labels in training data: {sorted(missing_labels)}")

    total_samples = len(labels)
    num_classes = len(id2label)
    weights = [
        total_samples / (num_classes * label_counts[label_id])
        for label_id in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    torch.manual_seed(SEED)

    df = prepare_dataframe(INPUT_PATH)

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    class_weights = compute_balanced_class_weights(train_df["label"])

    print("\nTrain label counts:")
    print(train_df["label"].map(id2label).value_counts().to_string())
    print("\nClass weights used in CrossEntropyLoss:")
    for label_id, weight in enumerate(class_weights.tolist()):
        print(f"{id2label[label_id]}: {weight:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
        local_files_only=True,
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    valid_dataset = valid_dataset.map(tokenize_batch, batched=True)

    train_dataset = train_dataset.remove_columns([TEXT_COLUMN])
    valid_dataset = valid_dataset.remove_columns([TEXT_COLUMN])

    train_dataset.set_format("torch")
    valid_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,

        report_to="none",
        seed=SEED,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

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

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\nSaved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
