"""Tests for the training utilities.

Two things are worth pinning here because getting them wrong is silent:

* the **weighted loss reduction** -- an incorrect reduction trains a different
  objective than the one documented, and the only symptom is worse metrics an
  hour later;
* the **dynamic-padding collator** -- padding with the wrong token id, or
  forgetting to pad the attention mask, corrupts every batch while still
  producing tensors of the right shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn

from ml.training.transformer import (
    class_weights,
    make_collate,
    mixed_review_weights,
    multilabel_pos_weight,
)

# ---------------------------------------------------------------------------
# Weighted loss reduction
# ---------------------------------------------------------------------------


def weighted_reduce(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mirror of the reduction performed inside `train_model`."""
    per_example = loss.mean(dim=1) if loss.ndim > 1 else loss
    return (per_example * weights).sum() / weights.sum().clamp(min=1e-8)


def test_uniform_weights_match_plain_mean():
    """Weighting with all-ones must be identical to reduction='mean'."""
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))

    weighted = weighted_reduce(
        nn.CrossEntropyLoss(reduction="none")(logits, labels), torch.ones(8)
    )
    plain = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
    assert weighted.item() == pytest.approx(plain.item(), abs=1e-6)


def test_weighting_shifts_the_loss_toward_the_upweighted_examples():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    labels = torch.tensor([0, 0])  # first is right, second is badly wrong
    per_example = nn.CrossEntropyLoss(reduction="none")(logits, labels)

    equal = weighted_reduce(per_example, torch.tensor([1.0, 1.0]))
    favour_wrong = weighted_reduce(per_example, torch.tensor([1.0, 9.0]))
    favour_right = weighted_reduce(per_example, torch.tensor([9.0, 1.0]))

    assert favour_wrong > equal > favour_right


def test_weighted_loss_is_a_weighted_mean_not_a_weighted_sum():
    """The batch loss scale must not grow with the weights, or the effective
    learning rate changes whenever the mixed-review share of a batch does."""
    logits = torch.randn(16, 3)
    labels = torch.randint(0, 3, (16,))
    per_example = nn.CrossEntropyLoss(reduction="none")(logits, labels)

    small = weighted_reduce(per_example, torch.full((16,), 1.0))
    large = weighted_reduce(per_example, torch.full((16,), 100.0))
    assert small.item() == pytest.approx(large.item(), abs=1e-5)


def test_multilabel_loss_reduces_over_labels_before_weighting():
    """BCE returns one value per label; the per-example mean must come first."""
    logits = torch.randn(4, 12)
    labels = (torch.rand(4, 12) > 0.5).float()
    loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    assert loss.shape == (4, 12)

    reduced = weighted_reduce(loss, torch.ones(4))
    assert reduced.shape == ()
    assert reduced.item() == pytest.approx(loss.mean().item(), abs=1e-6)


# ---------------------------------------------------------------------------
# mixed_review_weights
# ---------------------------------------------------------------------------


def test_mixed_review_weights_only_upweights_mixed_reviews():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "review_id": ["a", "a", "b", "b"],
            "polarity": ["positive", "negative", "positive", "positive"],
        }
    )
    weights = mixed_review_weights(frame, 5.0)
    assert list(weights) == [5.0, 5.0, 1.0, 1.0]


def test_mixed_review_weights_of_one_is_a_no_op():
    import pandas as pd

    frame = pd.DataFrame(
        {"review_id": ["a", "a"], "polarity": ["positive", "negative"]}
    )
    assert list(mixed_review_weights(frame, 1.0)) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Dynamic padding collator
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    pad_token_id = 0


def test_collate_pads_to_the_longest_in_the_batch_not_to_max_length():
    collate = make_collate(_FakeTokenizer())
    batch = [
        {
            "input_ids": torch.tensor([5, 6, 7]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor(1),
        },
        {
            "input_ids": torch.tensor([8]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor(0),
        },
    ]
    out = collate(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [8, 0, 0]


def test_collate_pads_the_attention_mask_with_zeros():
    """Padding the mask with 1s would let the model attend to padding."""
    collate = make_collate(_FakeTokenizer())
    out = collate(
        [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]),
             "labels": torch.tensor(0)},
            {"input_ids": torch.tensor([4]), "attention_mask": torch.tensor([1]),
             "labels": torch.tensor(1)},
        ]
    )
    assert out["attention_mask"][1].tolist() == [1, 0, 0]


def test_collate_carries_sample_weights_through():
    collate = make_collate(_FakeTokenizer())
    out = collate(
        [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
             "labels": torch.tensor(0), "weight": torch.tensor(3.0)},
            {"input_ids": torch.tensor([3]), "attention_mask": torch.tensor([1]),
             "labels": torch.tensor(1), "weight": torch.tensor(1.0)},
        ]
    )
    assert out["weight"].tolist() == [3.0, 1.0]


def test_collate_handles_multilabel_targets():
    collate = make_collate(_FakeTokenizer())
    out = collate(
        [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]),
             "labels": torch.zeros(12)},
            {"input_ids": torch.tensor([3]), "attention_mask": torch.tensor([1]),
             "labels": torch.ones(12)},
        ]
    )
    assert out["labels"].shape == (2, 12)


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def test_class_weights_favour_the_rare_class():
    y = np.array([2] * 90 + [0] * 9 + [1] * 1)  # neutral is 1%
    weights = class_weights(y, 3)
    assert weights[1] > weights[0] > weights[2]


def test_class_weights_are_normalised_to_mean_one():
    """Keeps the loss scale comparable to the unweighted case."""
    y = np.array([2] * 90 + [0] * 9 + [1] * 1)
    assert class_weights(y, 3).mean().item() == pytest.approx(1.0, abs=1e-5)


def test_pos_weight_is_clipped():
    """An ultra-rare label must not produce an enormous, destabilising weight."""
    y = np.zeros((1000, 3))
    y[0, 0] = 1  # 1 positive in 1000
    y[:, 1] = 1  # always present
    weight = multilabel_pos_weight(y)
    assert weight[0].item() == pytest.approx(20.0)  # clipped from 999
    assert weight[1].item() == pytest.approx(0.1)   # clipped from 0


# ---------------------------------------------------------------------------
# Dynamic padding must not change the maths
# ---------------------------------------------------------------------------


def test_dynamic_padding_matches_fixed_padding_on_a_real_model():
    """Padding is an optimisation, not a modelling change.

    If the attention mask is padded correctly the model cannot see the padding,
    so batching to the longest member must give the same logits as padding every
    sequence to max_length. A wrong pad token or mask value would silently shift
    predictions while keeping every tensor shape valid.
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    model_dir = REPO_ROOT / "models" / "sentiment_classifier"
    processed = REPO_ROOT / "data" / "processed" / "asc_dev.csv"
    if not (model_dir / "model.safetensors").exists() or not processed.exists():
        pytest.skip("Needs a trained transformer and the processed dev split")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ml.preprocessing.transform import load_taxonomy
    from ml.training.transformer import SentencePairDataset

    taxonomy = load_taxonomy(REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml")
    frame = pd.read_csv(processed).head(48)
    aspects = [taxonomy.descriptions[a] for a in frame["aspect"]]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    with torch.no_grad():
        encoded = tokenizer(
            list(frame["text"]), aspects, truncation="only_first",
            max_length=128, padding="max_length", return_tensors="pt",
        )
        fixed = model(**encoded).logits.numpy()

    dataset = SentencePairDataset(
        frame["text"], aspects, frame["label"].to_numpy(), tokenizer, 128
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=make_collate(tokenizer))
    batches = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batches.append(model(**batch).logits.numpy())
    dynamic = np.concatenate(batches)

    assert np.abs(fixed - dynamic).max() < 1e-4
    assert (fixed.argmax(1) == dynamic.argmax(1)).all()
