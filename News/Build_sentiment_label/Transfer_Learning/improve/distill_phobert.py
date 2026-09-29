"""E4 - Distillation (mentor plan muc 4): tan dung 126k bai chua nhan bang
cach train mot model "tro" tren CA nhan that (Tune, target = vector one-hot)
LAN xac suat mem cua model "thay" hien tai (E3, da cham san toan bo corpus o
inference/data/article_scores_phobert.parquet) tren mot mau bai CHUA NHAN -
thay vi chi hoc tu nhan da gan tay.

Loss = cross-entropy voi target MEM cho ca 2 loai vi du (target la one-hot
cho bai that, la xac suat cua thay cho bai chua nhan) - dung cong thuc, khong
can forward pass lai thay (da co san trong article_scores_phobert.parquet).

Pilot dau tien (Tune-train + 1 lan check Holdout, N=5000) cho macro F1 0.8105
so voi baseline thuan-supervised 0.7581 tren dung Holdout do - de chac chan
day khong phai may man theo 1 split, ban nay them 5-fold CV tren Tune (moi
fold: train = fold-train that (soft one-hot) + CUNG 1 mau unlabeled co dinh,
val = fold-val that, danh gia hard-label nhu thuong) truoc khi fit cuoi +
check Holdout 1 lan - dung cau truc voi improve/repeated_cv_tune_holdout.py.

Chay (dung venv GPU cua repo)::

    .../Model_Output/.venv-gpu/Scripts/python.exe \
        News/Build_sentiment_label/Transfer_Learning/improve/distill_phobert.py \
        --n-unlabeled 5000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Transfer_Learning.model.common import (  # noqa: E402
    BASE_MODEL_PATH,
    MAX_LENGTH,
    RANDOM_SEED,
    SENTENCES_COLUMN,
    TEXT_COLUMN,
    VALID_LABELS,
    build_sentence_strings,
    chunk_sentences_into_windows,
    compute_metrics_table,
    confusion_matrix,
    encode_labels,
    load_ground_truth,
    load_ground_truth_tune_holdout,
    pool_chunks_by_weight,
)
from News.Build_sentiment_label.Transfer_Learning.model.finetune_phobert import (  # noqa: E402
    OUTPUT_DIR as FINETUNE_OUTPUT_DIR,
    build_classifier,
    make_training_args,
)
from News.Build_sentiment_label.Transfer_Learning.pretrain.domain_adaptive_pretrain import (  # noqa: E402
    load_ground_truth_row_ids,
)

TOKENIZED_CORPUS_PATH = (
    PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
)
TEACHER_SCORES_PATH = (
    PROJECT_ROOT
    / "News"
    / "Build_sentiment_label"
    / "Transfer_Learning"
    / "inference"
    / "data"
    / "article_scores_phobert.parquet"
)
IMPROVE_DIR = Path(__file__).resolve().parent
DATA_DIR = IMPROVE_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-unlabeled", type=int, default=5000)
    parser.add_argument("--folds", type=int, default=5, help="5-fold CV tren Tune truoc khi fit cuoi + check Holdout.")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Bo qua vong K-fold CV (ton kem voi N lon), chi fit 1 lan tren toan bo Tune + unlabeled roi check Holdout.",
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--save-production",
        action="store_true",
        help="Sau khi check Holdout, fit them 1 lan tren TOAN BO 1044 dong (Tune+Holdout) "
        "+ unlabeled roi luu vao Model_Output/phobert_finetuned_ground_truth - de "
        "score_corpus_phobert.py dung ban distillation nay thay E3 thuan-supervised.",
    )
    return parser.parse_args()


def sample_unlabeled_documents(n_unlabeled: int, excluded_row_ids: set[int]) -> pd.DataFrame:
    """Lay ngau nhien n_unlabeled bai tu corpus 126k (loai het source_row_id
    da co trong ground truth), tra ve DataFrame [SENTENCES_COLUMN, soft_label]
    voi soft_label = [prob_negative, prob_neutral, prob_positive] cua thay E3
    (da cham san, khong forward pass lai)."""
    teacher_scores = pd.read_parquet(
        TEACHER_SCORES_PATH, columns=["prob_negative", "prob_neutral", "prob_positive"]
    )
    n_corpus = len(teacher_scores)
    all_row_ids = np.arange(n_corpus)
    candidate_row_ids = np.setdiff1d(all_row_ids, np.array(sorted(excluded_row_ids), dtype=np.int64))

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_row_ids = rng.choice(candidate_row_ids, size=min(n_unlabeled, len(candidate_row_ids)), replace=False)
    sampled_row_ids.sort()

    sentences_col = pd.read_parquet(TOKENIZED_CORPUS_PATH, columns=[SENTENCES_COLUMN])
    out = pd.DataFrame(
        {
            SENTENCES_COLUMN: sentences_col[SENTENCES_COLUMN].to_numpy()[sampled_row_ids],
        }
    )
    soft = teacher_scores.iloc[sampled_row_ids].reset_index(drop=True)
    out["soft_label"] = soft[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy().tolist()
    return out


def build_labeled_chunk_frame(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Nhu finetune_phobert.py::build_chunk_frame nhung target la vector
    one-hot (soft_label) thay vi nhan int, de dung chung 1 ham loss voi phan
    unlabeled (soft cross-entropy)."""
    n_classes = len(VALID_LABELS)
    rows = []
    for sentences, label in zip(frame[SENTENCES_COLUMN], frame["label"]):
        one_hot = [0.0] * n_classes
        one_hot[int(label)] = 1.0
        chunks = chunk_sentences_into_windows(build_sentence_strings(sentences), tokenizer, MAX_LENGTH)
        for chunk in chunks:
            rows.append({TEXT_COLUMN: chunk, "soft_label": one_hot})
    return pd.DataFrame(rows)


def build_unlabeled_chunk_frame(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for sentences, soft_label in zip(frame[SENTENCES_COLUMN], frame["soft_label"]):
        chunks = chunk_sentences_into_windows(build_sentence_strings(sentences), tokenizer, MAX_LENGTH)
        for chunk in chunks:
            rows.append({TEXT_COLUMN: chunk, "soft_label": soft_label})
    return pd.DataFrame(rows)


def build_eval_chunk_frame(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Cho Holdout: giu doc_position/chunk_weight de gop du doan lai theo
    bai (giong finetune_phobert.py::build_chunk_frame ban goc, dung nhan
    int, khong phai soft_label - vi day la de DANH GIA, khong phai train)."""
    rows = []
    for doc_position, (sentences, label) in enumerate(zip(frame[SENTENCES_COLUMN], frame["label"])):
        chunks = chunk_sentences_into_windows(build_sentence_strings(sentences), tokenizer, MAX_LENGTH)
        for chunk in chunks:
            rows.append(
                {
                    TEXT_COLUMN: chunk,
                    "label": label,
                    "doc_position": doc_position,
                    "chunk_weight": max(len(chunk.split(" ")), 1),
                }
            )
    return pd.DataFrame(rows)


def make_soft_label_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(frame[[TEXT_COLUMN, "soft_label"]], preserve_index=False)
    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=[TEXT_COLUMN])
    return dataset.with_format("torch")


def make_text_only_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(frame[[TEXT_COLUMN]], preserve_index=False)
    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=[TEXT_COLUMN])
    return dataset.with_format("torch")


class SoftLabelCollator:
    """Nhu DataCollatorWithPadding nhung tach rieng soft_label ra truoc khi
    goi tokenizer.pad (chi biet cac truong input_ids/attention_mask chuan),
    roi ghep lai thanh 1 tensor (batch, 3) sau."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        features = [dict(f) for f in features]
        soft_labels = torch.stack([torch.as_tensor(f.pop("soft_label"), dtype=torch.float32) for f in features])
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["soft_label"] = soft_labels
        return batch


class SoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        soft_labels = inputs.pop("soft_label")
        outputs = model(**inputs)
        logits = outputs["logits"]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(soft_labels.to(logits.dtype) * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def train_soft_label_model(
    train_chunk_frame: pd.DataFrame,
    tokenizer,
    args: argparse.Namespace,
    use_bf16: bool,
    use_fp16: bool,
    on_cuda: bool,
    scratch_prefix: str,
):
    """Train mot model tuoi tu checkpoint E1 tren train_chunk_frame (cot
    TEXT_COLUMN + soft_label), tra ve model da train - dung chung cho moi
    fold CV lan cho lan fit cuoi tren toan bo Tune."""
    train_dataset = make_soft_label_dataset(train_chunk_frame, tokenizer)
    data_collator = SoftLabelCollator(tokenizer)
    model = build_classifier()
    with tempfile.TemporaryDirectory(prefix=scratch_prefix) as scratch:
        training_args = make_training_args(Path(scratch), args, use_bf16, use_fp16, on_cuda)
        # "soft_label" isn't a column name Trainer recognizes (only the
        # singular "label"/"labels" survive its default column-pruning), so
        # without this it gets silently dropped before the collator ever
        # sees it.
        training_args.remove_unused_columns = False
        trainer = SoftLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        trained_model = trainer.model
    return trained_model


def predict_pooled_probabilities_plain(
    model, tokenizer, chunk_frame: pd.DataFrame, chunk_dataset: Dataset, n_docs: int, batch_size: int
) -> np.ndarray:
    """Du doan bang mot Trainer THUONG (khong phai SoftLabelTrainer) de
    tranh dung nham compute_loss ky vong soft_label trong luc predict tren
    du lieu nhan int cua Holdout."""
    with tempfile.TemporaryDirectory(prefix="phobert_distill_eval_") as scratch:
        eval_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=scratch,
                per_device_eval_batch_size=batch_size,
                report_to="none",
                dataloader_num_workers=0,
            ),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        chunk_logits = eval_trainer.predict(chunk_dataset).predictions
    chunk_probabilities = torch.softmax(torch.tensor(chunk_logits), dim=-1).numpy().astype(np.float64)
    doc_probabilities = pool_chunks_by_weight(
        chunk_probabilities,
        chunk_frame["doc_position"].to_numpy(),
        chunk_frame["chunk_weight"].to_numpy(dtype=np.float64),
        n_docs,
    )
    empty_mask = np.isnan(doc_probabilities).any(axis=1)
    if empty_mask.any():
        doc_probabilities[empty_mask] = 1.0 / len(VALID_LABELS)
    return doc_probabilities


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = on_cuda and not args.no_amp and not args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = on_cuda and not args.no_amp and not use_bf16
    print("Base model :", BASE_MODEL_PATH)
    print("Device     :", torch.cuda.get_device_name(0) if on_cuda else "cpu")

    tune_df, holdout_df = load_ground_truth_tune_holdout()
    tune_df = tune_df.assign(label=encode_labels(tune_df["ground_truth_label"])).reset_index(drop=True)
    holdout_df = holdout_df.assign(label=encode_labels(holdout_df["ground_truth_label"])).reset_index(drop=True)
    y_holdout = np.asarray(holdout_df["label"], dtype=int)
    print(f"Tune: {len(tune_df)} rows | Holdout: {len(holdout_df)} rows")

    ground_truth_row_ids = load_ground_truth_row_ids()
    print(f"Sampling {args.n_unlabeled} unlabeled documents (excluding {len(ground_truth_row_ids)} GT rows) ...")
    unlabeled_df = sample_unlabeled_documents(args.n_unlabeled, ground_truth_row_ids)
    print(f"  sampled {len(unlabeled_df)} unlabeled documents")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, local_files_only=True)

    print("Building chunk frame cho mau unlabeled (dung chung cho moi fold + fit cuoi) ...")
    unlabeled_chunk_frame = build_unlabeled_chunk_frame(unlabeled_df, tokenizer)
    print(f"  unlabeled chunks: {len(unlabeled_chunk_frame)}")

    # ------------------------------------------------------------------
    # K-fold CV tren Tune - moi fold: train = fold-train that (soft one-hot)
    # + CUNG 1 mau unlabeled co dinh; val = fold-val that, danh gia hard-label
    # nhu thuong (khong bao gio co du lieu unlabeled trong val).
    # --skip-cv: bo qua het khoi nay (ton kem tuyen tinh theo N unlabeled x
    # so fold) - dung khi chi muon 1 lan train-Tune + check Holdout.
    # ------------------------------------------------------------------
    if args.skip_cv:
        print("\n--skip-cv: bo qua K-fold CV tren Tune, chi fit 1 lan + check Holdout.")
    else:
        y_tune = np.asarray(tune_df["label"], dtype=int)
        n_tune = len(tune_df)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
        oof_proba = np.zeros((n_tune, len(VALID_LABELS)), dtype=np.float64)
        oof_pred = np.full(n_tune, -1, dtype=int)

        cv_start = time.perf_counter()
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_tune), y_tune), start=1):
            fold_start = time.perf_counter()
            train_frame = tune_df.iloc[train_idx].reset_index(drop=True)
            val_frame = tune_df.iloc[val_idx].reset_index(drop=True)

            labeled_chunk_frame = build_labeled_chunk_frame(train_frame, tokenizer)
            train_chunk_frame = pd.concat([labeled_chunk_frame, unlabeled_chunk_frame], ignore_index=True)
            train_chunk_frame = train_chunk_frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

            fold_model = train_soft_label_model(
                train_chunk_frame, tokenizer, args, use_bf16, use_fp16, on_cuda, f"phobert_distill_cv{fold_id}_"
            )

            val_chunk_frame = build_eval_chunk_frame(val_frame, tokenizer)
            val_dataset = make_text_only_dataset(val_chunk_frame, tokenizer)
            fold_proba = predict_pooled_probabilities_plain(
                fold_model, tokenizer, val_chunk_frame, val_dataset, len(val_frame), args.batch_size
            )
            oof_proba[val_idx] = fold_proba
            oof_pred[val_idx] = fold_proba.argmax(axis=1)

            fold_acc = float((oof_pred[val_idx] == y_tune[val_idx]).mean())
            print(
                f"Fold {fold_id}/{args.folds}: train={len(train_idx)} val={len(val_idx)} "
                f"acc={fold_acc:.4f} ({time.perf_counter() - fold_start:.1f}s)"
            )
            del fold_model
            if on_cuda:
                torch.cuda.empty_cache()

        cv_wall = time.perf_counter() - cv_start
        cv_metrics_df = compute_metrics_table(y_tune, oof_pred)
        cv_confusion_df = pd.DataFrame(
            confusion_matrix(y_tune, oof_pred),
            index=[f"true_{label}" for label in VALID_LABELS],
            columns=[f"pred_{label}" for label in VALID_LABELS],
        ).reset_index(names="true_label")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cv_metrics_df.to_csv(DATA_DIR / "distill_cv_metrics.csv", index=False, encoding="utf-8-sig")
        cv_confusion_df.to_csv(DATA_DIR / "distill_cv_confusion_matrix.csv", index=False, encoding="utf-8-sig")

        cv_overall = cv_metrics_df.loc[cv_metrics_df["metric_scope"].eq("overall")].iloc[0]
        print(f"\n=== {args.folds}-fold CV tren Tune ({n_tune} bai, OOF) - distillation N_unlabeled={args.n_unlabeled} ===")
        print(cv_metrics_df.to_string(index=False))
        print("\nConfusion matrix (OOF):")
        print(cv_confusion_df.to_string(index=False))
        print(f"\nTune OOF macro F1 : {cv_overall['f1']:.4f}")
        print(f"Tune OOF accuracy : {cv_overall['accuracy']:.4f}")
        print(f"CV wall time      : {cv_wall:.1f}s")
        print("(so sanh: baseline thuan-supervised, finetune_phobert.py tren toan bo 1044 dong - macro F1 0.7688)")

    # ------------------------------------------------------------------
    # Fit cuoi tren TOAN BO Tune + unlabeled, check Holdout 1 lan
    # ------------------------------------------------------------------
    print("\nFit cuoi tren toan bo Tune + unlabeled, check Holdout ...")
    labeled_chunk_frame = build_labeled_chunk_frame(tune_df, tokenizer)
    final_train_chunk_frame = pd.concat([labeled_chunk_frame, unlabeled_chunk_frame], ignore_index=True)
    final_train_chunk_frame = final_train_chunk_frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    print(
        f"  labeled chunks: {len(labeled_chunk_frame)} | unlabeled chunks: {len(unlabeled_chunk_frame)} "
        f"| total train chunks: {len(final_train_chunk_frame)}"
    )

    final_start = time.perf_counter()
    trained_model = train_soft_label_model(
        final_train_chunk_frame, tokenizer, args, use_bf16, use_fp16, on_cuda, "phobert_distill_final_"
    )

    holdout_chunk_frame = build_eval_chunk_frame(holdout_df, tokenizer)
    holdout_dataset = make_text_only_dataset(holdout_chunk_frame, tokenizer)

    holdout_proba = predict_pooled_probabilities_plain(
        trained_model, tokenizer, holdout_chunk_frame, holdout_dataset, len(holdout_df), args.batch_size
    )
    holdout_pred = holdout_proba.argmax(axis=1)
    print(f"Final fit + Holdout predict: {time.perf_counter() - final_start:.1f}s")

    metrics_df = compute_metrics_table(y_holdout, holdout_pred)
    confusion_df = pd.DataFrame(
        confusion_matrix(y_holdout, holdout_pred),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    metrics_df.to_csv(DATA_DIR / "distill_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    confusion_df.to_csv(DATA_DIR / "distill_holdout_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    # Per-article predictions (id/source_row_id/title/ground_truth_label/
    # predicted_label/prob_*/sentiment_score_ml/is_correct) - same schema as
    # improve/data/finetune_holdout_predictions.csv (E3) - so this file can
    # feed the same bootstrap/McNemar comparison utilities Traditional_ML's
    # compare_branches_holdout.py uses, instead of only reporting the
    # aggregate macro-F1 with no CI / no paired comparison against the other
    # branches.
    prediction_df = holdout_df[["id", "source_row_id", "title", "ground_truth_label"]].copy()
    prediction_df["predicted_label"] = [VALID_LABELS[i] for i in holdout_pred]
    for label_id, label in enumerate(VALID_LABELS):
        prediction_df[f"prob_{label}"] = holdout_proba[:, label_id]
    prediction_df["sentiment_score_ml"] = prediction_df["prob_positive"] - prediction_df["prob_negative"]
    prediction_df["is_correct"] = prediction_df["ground_truth_label"].eq(prediction_df["predicted_label"])
    prediction_df.to_csv(DATA_DIR / "distill_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Da luu du doan tung bai: {DATA_DIR / 'distill_holdout_predictions.csv'}")

    overall = metrics_df.loc[metrics_df["metric_scope"].eq("overall")].iloc[0]
    print(f"\n=== Holdout ({len(holdout_df)} rows) - distillation (N_unlabeled={args.n_unlabeled}) ===")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print(f"\nHoldout macro F1 : {overall['f1']:.4f}")
    print(f"Holdout accuracy : {overall['accuracy']:.4f}")
    print("\n(so sanh: baseline thuan-supervised tren dung Holdout nay - macro F1 0.7581, accuracy 0.7611)")

    if not args.save_production:
        return

    # ------------------------------------------------------------------
    # Model production: fit lai 1 LAN NUA tren TOAN BO 1044 dong (Tune +
    # Holdout - khong con ly do giu rieng Holdout mot khi da co con so danh
    # gia trung thuc o tren) + CUNG mau unlabeled, luu de score_corpus_phobert.py
    # dung thay E3 thuan-supervised - dung quy uoc finetune_phobert.py::main
    # (CV/Holdout de UOC LUONG hieu nang, fit cuoi tren het du lieu de DEPLOY).
    # ------------------------------------------------------------------
    print(f"\nFit production tren TOAN BO {len(tune_df) + len(holdout_df)} dong (Tune+Holdout) + unlabeled ...")
    full_ground_truth_df = load_ground_truth()
    full_ground_truth_df = full_ground_truth_df.assign(
        label=encode_labels(full_ground_truth_df["ground_truth_label"])
    ).reset_index(drop=True)
    print(f"  full ground truth: {len(full_ground_truth_df)} dong")

    full_labeled_chunk_frame = build_labeled_chunk_frame(full_ground_truth_df, tokenizer)
    production_train_chunk_frame = pd.concat(
        [full_labeled_chunk_frame, unlabeled_chunk_frame], ignore_index=True
    )
    production_train_chunk_frame = production_train_chunk_frame.sample(
        frac=1.0, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    print(
        f"  labeled chunks: {len(full_labeled_chunk_frame)} | unlabeled chunks: {len(unlabeled_chunk_frame)} "
        f"| total train chunks: {len(production_train_chunk_frame)}"
    )

    production_start = time.perf_counter()
    production_model = train_soft_label_model(
        production_train_chunk_frame, tokenizer, args, use_bf16, use_fp16, on_cuda, "phobert_distill_production_"
    )
    print(f"Production fit: {time.perf_counter() - production_start:.1f}s")

    FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FINETUNE_OUTPUT_DIR.glob("checkpoint-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    production_model.save_pretrained(str(FINETUNE_OUTPUT_DIR))
    tokenizer.save_pretrained(str(FINETUNE_OUTPUT_DIR))
    print(f"\nDa luu model production (distillation) vao: {FINETUNE_OUTPUT_DIR}")
    print(
        "Model nay thay the E3 thuan-supervised - score_corpus_phobert.py se tu dong "
        "dung ban nay trong lan chay tiep theo (cung duong dan, khong can sua gi)."
    )


if __name__ == "__main__":
    main()
