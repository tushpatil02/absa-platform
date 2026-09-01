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

## Stage A — Aspect Category Detection

Multi-label, 12 sigmoid outputs. Threshold tuned on dev.

| Model | Micro F1 | Macro F1 | Subset acc | Micro P | Micro R | Train time |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + OvR LogReg** | **0.7755** | 0.7387 | 0.5523 | 0.783 | 0.768 | 6 s |
| DistilBERT | 0.6192 | 0.6114 | 0.3575 | **0.525** | 0.756 | 26 min (CPU) |

**The baseline wins by 15.6 micro-F1 points**, which was not the expected result.

The cause is visible in the precision column: DistilBERT's micro precision is
0.525 against the baseline's 0.783 at similar recall. It over-predicts —
`design` gets precision 0.218 at recall 0.829, meaning it flags that aspect
almost everywhere. With 2,298 training reviews, 12 labels and `pos_weight` up to
20 pushing hard toward positives on rare labels, the model learns to fire
liberally.

It is also the wrong tool for the job. Aspect detection is largely *lexical* —
the word "battery" all but determines the aspect — and word + character n-grams
model that directly. The character n-grams additionally absorb the misspellings
("batery", "camara") that pervade review text. There is little long-range
context for an encoder to exploit.

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

## Selection: different families per stage

The comparison selected **different model families for each stage**, so the
serving layer composes them rather than forcing one:

| Stage | Selected | Metric | Runner-up |
|---|---|---|---|
| A — aspect detection | **TF-IDF + OvR LogReg** | micro F1 0.7755 | DistilBERT 0.6192 |
| B — sentiment | **DistilBERT** | macro F1 0.6538 | TF-IDF 0.6088 |

`ml/inference/predictor.py` reads `models/metadata/comparison.json` — written
from held-out test metrics — and resolves each stage independently. With no
comparison file it defaults to the baseline for both, because defaulting to the
larger model would have shipped a detector 15.6 points worse.

Override per stage: `ABSA_ASPECT_MODEL` / `ABSA_SENTIMENT_MODEL`
(`auto` | `baseline` | `transformer`).

---

## Confidence is not trustworthy on mixed reviews

The UI shows a confidence figure, so a *confidently wrong* answer is worse than a
hedged one. Measured on test, at the 0.7 threshold:

| Model | Confidently wrong (all) | On mixed reviews | ECE (all) |
|---|---:|---:|---:|
| TF-IDF + LogReg | 5.2% | 17.7% | 0.0545 |
| DistilBERT | **11.4%** | **30.1%** | 0.0554 |

Aggregate ECE is effectively identical, so this is not a general calibration
gap — DistilBERT is simply far more assertive (mean confidence 0.857 vs 0.745),
which converts the same error rate into twice as many confident errors.

So the model that wins macro F1 is also confidently wrong roughly twice as often
on the cases that matter. That is a genuine tradeoff, documented rather than
hidden, and switchable with one environment variable.

### Temperature scaling was tried and rejected

The standard fix is temperature scaling (Guo et al., 2017): divide logits by a
scalar fitted on dev. It is monotonic, so it cannot change any argmax — model
selection is provably unaffected.

Fitted here: **T = 1.0226**, essentially a no-op, and test ECE got slightly
*worse* (0.0554 → 0.0604). Investigating why produced the more interesting
result — **the miscalibration is conditional, not global:**

| Slice | Mean confidence | Accuracy | Gap | ECE |
|---|---:|---:|---:|---:|
| uniform reviews | 0.8760 | 0.8753 | **+0.0007** | 0.0525 |
| mixed reviews | 0.7706 | 0.5414 | **+0.2292** | 0.2307 |

On uniform reviews the model is **near-perfectly calibrated**. On mixed reviews
it is wildly over-confident. Fitted separately the two slices want *opposite*
corrections — T = 2.097 and T = 0.885 — which cancel to T ≈ 1.

Applied per slice the effect would be dramatic: mixed-review confidently-wrong
falls **30.1% → 3.8%**. But "is this review mixed?" requires the gold labels.
Conditioning on **aspect count** was tested as an inference-time proxy and does
not work — the confidence gap is flat across counts (+0.03 to +0.05), including
1-aspect reviews that are 0% mixed.

The circularity is the point: the model could calibrate itself if it could tell
a mixed review from a uniform one, which is precisely the thing it is failing at.
`scripts/calibrate.py` reproduces the whole investigation.

---

## A units error worth recording

`docs/dataset.md` justified `max_length=128` with "covers 99.6% of reviews".
That figure was measured in **words**; the model truncates in **tokens**.

| Unit | Coverage at 128 |
|---|---:|
| words (whole reviews) | 99.6% |
| tokens (sentence-pair inputs) | **96.97%** |

So ~3% of training pairs are truncated, roughly 5× more than documented — median
32 tokens, p99 162, max 205. `truncation="only_first"` means the aspect segment
is always preserved and the review is what gets cut, which is the right side to
lose, but the claim was still wrong.

Raising the cap is now nearly free: with dynamic padding, `max_length` is a
truncation ceiling rather than a padding target, so only the ~3% of batches
containing a long pair pay for it. `max_length=192` would cover 99.71%. It is
deliberately **not** changed mid-experiment — the runs in progress use 128, and
changing it now would confound the mixed-weight comparison.

---

## Experiment: upweighting mixed reviews (result: no change)

Roadmap item 2 was "upweight mixed reviews so reading overall tone stops being a
sufficient hypothesis". It was implemented (`--mixed-weight`), run at w=3 and
w=8, and **rejected**. Recording it because the negative result is more useful
than the intervention would have been.

Selection ran on **dev**, with the objective fixed in advance
(`scripts/compare_mixed_weight.py`):

    score = mixed_accuracy - 1.0 * max(0, macro_f1(w=1) - macro_f1)

| dev | macro F1 | accuracy | mixed acc | uniform | collapsed | score |
|---|---:|---:|---:|---:|---:|---:|
| **w=1** | **0.7086** | 0.8446 | **0.5593** | 0.8919 | 0.9130 | **0.5593** |
| w=3 | 0.6691 | 0.8217 | 0.5508 | 0.8666 | 0.8043 | 0.5114 |
| w=8 | 0.6269 | 0.7747 | 0.5508 | 0.8118 | 0.7391 | 0.4692 |

**Selected: w=1.** No change to the shipped model.

### The part worth being careful about

On **test**, w=3 scored *better* on mixed reviews — 0.5789 against 0.5414, and a
paired bootstrap puts that at +0.038 with a 95% CI of [+0.008, +0.071], i.e.
nominally significant. On **dev** the same comparison is −0.009, CI
[−0.051, +0.025], not significant and pointing the other way.

Opposite signs on two splits, with the significant one only just clearing zero,
on a **single seed**. That is what a null effect looks like when a 3,464-pair
training set is evaluated on a 118-pair dev slice. Selecting on the test number
would have "discovered" a +3.8-point improvement that dev says does not exist --
which is exactly why the weight was chosen on dev.

### What *did* move, consistently

Collapse rate -- the share of mixed reviews given one polarity for every aspect --
falls monotonically with the weight, in the same direction on both splits:

| | w=1 | w=3 | w=8 |
|---|---:|---:|---:|
| dev collapsed | 0.9130 | 0.8043 | **0.7391** |
| test collapsed | 0.8511 | 0.7660 | — |

So upweighting does change behaviour: the model becomes **more willing to assign
different polarities within a review**. It just does not become more *accurate*
at it. It differentiates more, and is wrong about as often -- while macro F1 falls
monotonically (0.7086 → 0.6691 → 0.6269 on dev).

Differentiating without being right is not worth 4 points of macro F1, so the
unweighted model stays.

### What this says about the whole evaluation

Every mixed-review number in this document rests on 94 test reviews / 266 pairs
and a single seed. This experiment is the concrete demonstration that such
numbers can swing by ±4 points between splits. **Seed-averaged evaluation with
confidence intervals is no longer a nice-to-have** -- it is required before any
claim of a small improvement here, including the claims already made above.

---

## What would likely fix it

Ranked by expected value, not yet attempted:

1. **DeBERTa-v3-base on a Colab GPU.** Stronger encoder, better at the
   long-range attention this needs. Cheapest experiment (~12 min on a T4). Note
   it would only be adopted for Stage B — Stage A should stay TF-IDF unless a
   transformer actually beats 0.7755.
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
