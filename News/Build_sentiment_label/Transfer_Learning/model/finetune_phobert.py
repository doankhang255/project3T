from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (
    BASE_MODEL_PATH,
    DATA_DIR,
    MAX_LENGTH,
    RANDOM_SEED,
    TEXT_COLUMN,
    VALID_LABELS,
    VAL_SIZE,
    compute_metrics_table,
    confusion_matrix,
    encode_labels,
    load_ground_truth,
)


# Single stratified train/val split rather than the 5-fold CV used by the
# Traditional_ML models: this machine has no GPU (CPU-only torch build), and
# fine-tuning a full PhoBERT model 5x over would take far longer than the
# sklearn models do. A held-out 20% split is the same evaluation protocol
# the earlier model_sentiment_v1.py/v2.py scripts already used.
OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "Model_Output" / "phobert_finetuned_ground_truth"
)
OUTPUT_METRICS_PATH = DATA_DIR / "phobert_finetuned_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "phobert_finetuned_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "phobert_finetuned_confusion_matrix.csv"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def compute_balanced_class_weights(labels: list[int]) -> torch.Tensor:
    label_counts = pd.Series(labels).value_counts().sort_index()
    missing_labels = set(range(len(VALID_LABELS))).difference(label_counts.index)
    if missing_labels:
        raise ValueError(f"Missing labels in training split: {sorted(missing_labels)}")

    total_samples = len(labels)
    num_classes = len(VALID_LABELS)
    weights = [
        total_samples / (num_classes * label_counts[label_id])
        for label_id in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_hf_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics_df = compute_metrics_table(np.asarray(labels), np.asarray(preds))
        overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
        return {
            "accuracy": float(overall["accuracy"]),
            "f1_macro": float(overall["f1"]),
        }

    return compute_metrics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(RANDOM_SEED)

    ground_truth_df = load_ground_truth()
    labels = encode_labels(ground_truth_df["ground_truth_label"])
    ground_truth_df = ground_truth_df.assign(label=labels)

    print("Base model:", BASE_MODEL_PATH)
    print("Documents:", len(ground_truth_df))
    print("Label counts:")
    print(ground_truth_df["ground_truth_label"].value_counts().to_string())

    train_df, val_df = train_test_split(
        ground_truth_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=ground_truth_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    class_weights = compute_balanced_class_weights(train_df["label"].tolist())
    print("\nTrain rows:", len(train_df), "| Val rows:", len(val_df))
    print("Class weights used in CrossEntropyLoss:")
    for label_id, weight in enumerate(class_weights.tolist()):
        print(f"  {VALID_LABELS[label_id]}: {weight:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=len(VALID_LABELS),
        label2id={label: index for index, label in enumerate(VALID_LABELS)},
        id2label=dict(enumerate(VALID_LABELS)),
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    train_dataset = Dataset.from_pandas(train_df[[TEXT_COLUMN, "label"]])
    val_dataset = Dataset.from_pandas(val_df[[TEXT_COLUMN, "label"]])
    train_dataset = train_dataset.map(tokenize_batch, batched=True).remove_columns([TEXT_COLUMN])
    val_dataset = val_dataset.map(tokenize_batch, batched=True).remove_columns([TEXT_COLUMN])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=build_hf_compute_metrics(),
        class_weights=class_weights,
    )

    trainer.train()

    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    y_true = np.asarray(predictions.label_ids)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = probabilities.argmax(axis=1)

    metrics_df = compute_metrics_table(y_true, y_pred)

    prediction_df = val_df[["id", "title", "ground_truth_label"]].copy()
    prediction_df["predicted_label"] = [VALID_LABELS[index] for index in y_pred]
    for label_id, label in enumerate(VALID_LABELS):
        prediction_df[f"prob_{label}"] = probabilities[:, label_id]
    prediction_df["sentiment_score_ml"] = (
        prediction_df["prob_positive"] - prediction_df["prob_negative"]
    )
    prediction_df["is_correct"] = prediction_df["ground_truth_label"].eq(
        prediction_df["predicted_label"]
    )

    confusion_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, index=False, encoding="utf-8-sig")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nMetrics (held-out validation split):")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print("\nSaved fine-tuned model to:", OUTPUT_DIR)
    print("Output metrics:", OUTPUT_METRICS_PATH)
    print("Output predictions:", OUTPUT_PREDICTIONS_PATH)
    print("Output confusion matrix:", OUTPUT_CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
