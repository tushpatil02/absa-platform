"""Transformer fine-tuning for both ABSA stages.

Written to run identically on a Colab T4 and on a CPU-only laptop -- device and
mixed precision are detected, never assumed. The CPU path is what makes this
testable locally; the GPU path is what makes DeBERTa-v3-large practical.

Stage A -- Aspect Category Detection
    Multi-label. 12 sigmoid outputs, ``BCEWithLogitsLoss`` with per-label
    ``pos_weight`` so the tail aspects (audio ~2% of reviews) are not ignored.

Stage B -- Aspect Sentiment Classification
    Sentence-pair, following Sun et al. (2019): the aspect is the second
    segment, so the model attends to "which aspect am I being asked about".
    Encoded as ``[CLS] review [SEP] <aspect description> [SEP]``. Class weights
    counter the 5.3% neutral share.

Deliberately no ``Trainer``: an explicit loop is ~80 lines, makes the class
weighting and threshold tuning visible rather than buried in kwargs, and avoids
a moving API surface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_MAX_LENGTH = 128  # covers 99.6% of reviews -- see docs/dataset.md


def set_seed(seed: int = 42) -> None:
    """Seed every RNG that affects training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(prefer: str | None = None) -> torch.device:
    """Pick a device. Never assumes a GPU is present."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Everything that changes between a laptop smoke-test and a real Colab run."""

    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = 42
    max_grad_norm: float = 1.0
    device: str | None = None
    # fp16 is a GPU-only win; on CPU it is slower. Auto-enabled when on CUDA.
    fp16: bool | None = None
    num_workers: int = 0  # 0 avoids Windows spawn overhead on small datasets.
    extra: dict = field(default_factory=dict)

    def resolved_fp16(self, device: torch.device) -> bool:
        if self.fp16 is not None:
            return self.fp16 and device.type == "cuda"
        return device.type == "cuda"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SingleTextDataset(Dataset):
    """Multi-label aspect detection: review -> 12 binary targets.

    Sequences are returned **unpadded**; :func:`collate` pads each batch to its
    own longest member. Padding everything to ``max_length`` wasted 68.6% of
    every batch here (median 32 tokens against a 128 limit).
    """

    def __init__(self, texts, labels, tokenizer, max_length: int, weights=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )
        item = {key: torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float)
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[index], dtype=torch.float)
        return item


class SentencePairDataset(Dataset):
    """Sentiment: ``[CLS] review [SEP] aspect description [SEP]`` -> 3 classes."""

    def __init__(self, texts, aspect_texts, labels, tokenizer, max_length: int, weights=None):
        self.texts = list(texts)
        self.aspect_texts = list(aspect_texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            self.aspect_texts[index],
            truncation="only_first",  # never truncate the aspect away
            max_length=self.max_length,
        )
        item = {key: torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[index], dtype=torch.float)
        return item


def make_collate(tokenizer):
    """Pad each batch to its own longest sequence rather than to max_length."""
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in batch)
        out: dict[str, torch.Tensor] = {}

        for key in batch[0]:
            if key in ("labels", "weight"):
                out[key] = torch.stack([item[key] for item in batch])
                continue
            fill = pad_id if key == "input_ids" else 0
            out[key] = torch.stack([
                torch.cat([
                    item[key],
                    torch.full((longest - len(item[key]),), fill, dtype=item[key].dtype),
                ])
                for item in batch
            ])
        return out

    return collate


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def multilabel_pos_weight(y: np.ndarray) -> torch.Tensor:
    """``pos_weight`` for BCEWithLogitsLoss: negatives / positives per label.

    Without this the model minimises loss on rare aspects by always predicting
    absent. Clipped at 20 so a very rare label cannot destabilise training.
    """
    y = np.asarray(y)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weight = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.tensor(np.clip(weight, 0.1, 20.0), dtype=torch.float)


def mixed_review_weights(frame, multiplier: float) -> np.ndarray:
    """Per-sample weights that upweight pairs from *mixed* reviews.

    A review is mixed when its gold polarities differ across aspects. Only 8.9%
    of training reviews are mixed (16.2% of pairs), so a model can score well by
    reading overall tone and ignoring the aspect entirely -- which is exactly
    what both models were measured doing.

    Computed from **training labels only**, so this uses no test information.
    """
    from ml.evaluation.mixed_reviews import find_mixed_reviews

    is_mixed = frame["review_id"].isin(find_mixed_reviews(frame)).to_numpy()
    return np.where(is_mixed, float(multiplier), 1.0).astype(np.float32)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is stable."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    weight = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weight / weight.mean(), dtype=torch.float)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _build_optimizer(model, config: TrainConfig):
    """AdamW with no weight decay on biases and LayerNorm -- the standard recipe."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if any(k in name for k in ("bias", "LayerNorm.weight")) else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def train_model(
    model,
    train_dataset: Dataset,
    config: TrainConfig,
    *,
    loss_fn: nn.Module,
    device: torch.device,
    collate_fn=None,
    log_every: int = 50,
) -> dict:
    """Fine-tune `model`. Returns a small history dict.

    The loss is computed here rather than by the model so the class weighting is
    explicit and auditable. ``loss_fn`` must use ``reduction="none"`` when the
    dataset supplies per-sample ``weight`` values; the reduction happens here so
    the weighting is visible.
    """
    from transformers import get_linear_schedule_with_warmup

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )
    optimizer = _build_optimizer(model, config)
    total_steps = len(loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    use_amp = config.resolved_fp16(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.to(device).train()
    history = {"epoch_loss": []}

    for epoch in range(config.epochs):
        running, seen = 0.0, 0
        for step, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            weights = batch.pop("weight", None)
            if weights is not None:
                weights = weights.to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
                if weights is not None:
                    # loss_fn ran with reduction="none": average per example
                    # first (multi-label returns one value per label), then take
                    # the weighted mean so the batch loss scale stays comparable.
                    per_example = loss.mean(dim=1) if loss.ndim > 1 else loss
                    loss = (per_example * weights).sum() / weights.sum().clamp(min=1e-8)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if log_every and step % log_every == 0:
                print(f"    epoch {epoch + 1}  step {step:>4}/{len(loader)}  loss {loss.item():.4f}")

        epoch_loss = running / max(seen, 1)
        history["epoch_loss"].append(epoch_loss)
        print(f"  epoch {epoch + 1}/{config.epochs}  mean loss {epoch_loss:.4f}")

    return history


@torch.no_grad()
def predict_logits(
    model, dataset: Dataset, config: TrainConfig, device: torch.device, collate_fn=None
) -> np.ndarray:
    """Run inference and return raw logits."""
    loader = DataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn
    )
    model.to(device).eval()
    outputs = []
    for batch in loader:
        batch.pop("labels", None)
        batch.pop("weight", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
