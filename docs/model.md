# Models, evaluation and selection

Every number here is on the **held-out test split**, touched once. Protocol:
train on `train`, tune hyperparameters and the detection threshold on `dev`,
evaluate on `test` last. Splits are grouped by review and leakage-asserted
(see [dataset.md](dataset.md)).

---

## Why two stages

| | Two-stage classification | Generative triplet extraction |
|---|---|---|
| Per-aspect probability | Native (softmax over 3 classes) | Needs extra machinery |
| Fixed 12-aspect taxonomy | Guaranteed by the output head | Free-form; needs post-hoc mapping |
| CPU inference | Comfortable | Marginal |
| Failure modes | Legible per stage | Opaque |

The 1–10 slider is defined over a probability distribution
([scoring.md](scoring.md)), so a model that cannot produce one is disqualified
regardless of its F1. That settled the architecture before any training ran.

---

## Selection metrics

**Stage A (aspect detection)** — micro F1. It weights by support, which matches
the question "how many aspect mentions did we catch". Macro F1 is reported
beside it because the tail aspects are small.

**Stage B (sentiment)** — **macro F1**, never accuracy.

That is not a stylistic preference. `neutral` is 5.3% of the pairs, so a model
that never predicts it loses almost nothing in accuracy. The baselines
demonstrate this concretely:

| Model | Macro F1 | Accuracy | neg F1 | **neu F1** | pos F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | **0.6088** | 0.7869 | 0.720 | **0.244** | 0.863 |
| TF-IDF + LinearSVC | 0.5319 | **0.8169** | 0.712 | **0.000** | 0.884 |

The SVM is 3 points *better* on accuracy and predicts `neutral` **zero times**.
Selecting on accuracy would have shipped the model that silently dropped a third
of the label space. Both rows are kept in the comparison for exactly that reason.

---

## Stage B — Aspect Sentiment Classification

Sentence-pair input, following Sun et al. (2019):
`[CLS] review [SEP] aspect description [SEP]`. Class-weighted cross-entropy for
the neutral imbalance.

| Model | Macro F1 | Accuracy | neg F1 | neu F1 | pos F1 | Train time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 0.5319 | 0.8169 | 0.712 | 0.000 | 0.884 | 8 s |
| TF-IDF + LogReg | 0.6088 | 0.7869 | 0.720 | 0.244 | 0.863 | 7 s |
| **DistilBERT** | **0.6538** | **0.8148** | **0.761** | **0.316** | **0.885** | 39.5 min (CPU) |

DistilBERT wins on every column. The largest relative gain is on `neutral`
(0.244 → 0.316, +30%), which is the class the metric was chosen to protect.

Confusion matrix, DistilBERT (rows = true, cols = predicted):

```
              neg  neu  pos
negative      331   31   43
neutral        22   27   23
positive      112   41  839
```

Neutral remains the weak class: 27 of 72 correct. With 305 neutral pairs in the
whole dataset, that is the expected ceiling, not a fixable bug.

---

## The diagnostic that changes the conclusion

Headline metrics hide whether a model conditions on the aspect **at all**. Most
reviews are uniformly positive or negative, so a model that reads only overall
tone still scores well. The capability this product actually sells is the other
case.

Sliced to **mixed reviews** — the 94 test reviews (266 pairs) carrying different
polarities for different aspects:

| Model | Mixed acc | Uniform acc | Gap | **Collapsed** |
|---|---:|---:|---:|---:|
| TF-IDF + LogReg | 0.5451 | 0.8404 | +0.2953 | 0.7553 |
| DistilBERT | **0.5414** | **0.8753** | **+0.3340** | **0.8511** |

*Collapsed* = share of mixed reviews given one single polarity for every aspect.

**DistilBERT is better overall and no better — marginally worse — at the actual
task.** Its entire gain came from uniform reviews (0.8404 → 0.8753). On mixed
reviews it is flat, and it ignores the aspect *more* often than the baseline
(85.1% vs 75.5% collapsed).

Concretely:

```
"the product is great, but the customer support is horrible."
   overall            gold=positive   pred=negative   MISS
   customer_service   gold=negative   pred=negative   OK

"great price and great performance but terrible battery."
   overall            gold=positive   pred=positive   OK
   battery            gold=negative   pred=negative   OK
```

It *can* separate aspects; it usually does not, defaulting to the review's
dominant tone.

**Why.** Most likely dataset size. 3,464 training pairs, of which mixed reviews
are a small minority, is not enough signal for the model to learn that the
second segment should override the first's overall tone. Reading tone is the
easier hypothesis and fits nearly as well.

**What this means for the reported result.** DistilBERT is selected — it is
better on every metric that was defined in advance, and changing the criterion
after seeing results would be exactly the kind of post-hoc rationalisation this
project exists to avoid. But the headline macro F1 of 0.6538 should be read
alongside the fact that **neither model reliably does aspect-conditional
sentiment at this dataset size.**

---

## What would likely fix it

Ranked by expected value, not yet attempted:

1. **DeBERTa-v3-base on a Colab GPU.** Stronger encoder, better at the
   long-range attention this needs. Cheapest experiment (~12 min on a T4).
2. **Oversample or upweight mixed reviews** during training so the easy
   tone-reading hypothesis stops being sufficient.
3. **More data.** The single biggest constraint. M-ABSA's other five English
   domains would roughly triple the corpus at the cost of a taxonomy mapping.
4. **Aspect-aware augmentation** — split mixed reviews into single-aspect
   clauses as additional training pairs.

Track this with `python scripts/compare_models.py`; the diagnostic runs on every
sentiment model on disk.

---

## Reproducing

```bash
python scripts/train_baseline.py                                    # ~15 s, CPU
python scripts/train_transformer.py --stage asc --epochs 3          # 40 min CPU / ~4 min T4
python scripts/train_transformer.py --stage acd --epochs 3
python scripts/compare_models.py
```

Or [`notebooks/absa_training.ipynb`](../notebooks/absa_training.ipynb) in Colab,
which detects the GPU and upgrades the model accordingly.

Seeds are fixed (42) but CPU/GPU kernel differences mean runs are not
bit-identical across devices. With 5,763 pairs, treat differences under roughly
±0.02 macro F1 as noise — a proper claim needs seed-averaged runs, which is on
the roadmap and not yet done.
