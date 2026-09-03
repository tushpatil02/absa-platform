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

That is not a stylistic preference. `neutral` is 5.2% of the pairs, so a model
that never predicts it loses almost nothing in accuracy. The baselines
demonstrate this concretely (dev split, where selection happens):

| Model | Macro F1 | Accuracy | neg F1 | **neu F1** | pos F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | **0.5912** | 0.7842 | 0.700 | **0.207** | 0.866 |
| TF-IDF + LinearSVC | 0.5339 | **0.8219** | 0.713 | **0.000** | 0.888 |

The SVM is nearly 4 points *better* on accuracy and predicts `neutral` **zero
times**. Selecting on accuracy would have shipped the model that silently
dropped a third of the label space. Both rows are kept in the comparison for
exactly that reason.

---

## Stage A — Aspect Category Detection

Multi-label, 12 sigmoid outputs. Threshold tuned on dev.

| Model | Micro F1 | Macro F1 | Subset acc | Micro P | Micro R | Train time |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + OvR LogReg** | **0.7418** | 0.7391 | 0.4632 | 0.745 | 0.739 | 6 s |
| DistilBERT † | 0.6192 | 0.6114 | 0.3575 | **0.525** | 0.756 | 26 min (CPU) |

† Measured **before** the taxonomy fix; the detector has not been retrained
since, so this row is not strictly comparable to the one above it. It is kept
because the gap is far too large for a remapping to close, but it should be
read as indicative rather than current.

**The baseline wins by roughly 12 micro-F1 points**, which was not the expected
result.

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

Test split. The LinearSVC lost on dev and was never run on test — the test set
is touched once, by the winner.

| Model | Macro F1 | Accuracy | neg F1 | neu F1 | pos F1 | Train time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC *(dev)* | 0.5339 | 0.8219 | 0.713 | 0.000 | 0.888 | 8 s |
| TF-IDF + LogReg | 0.5993 | 0.7850 | 0.713 | 0.224 | 0.862 | 7 s |
| **DistilBERT** | **0.6637** | **0.8245** | **0.765** | **0.335** | **0.891** | 43 min (CPU) |

DistilBERT wins on every column. The largest relative gain is on `neutral`
(0.224 → 0.335, +50%), which is the class the metric was chosen to protect.

Confusion matrix, DistilBERT (rows = true, cols = predicted):

```
              neg  neu  pos
negative      329   27   49
neutral        20   27   25
positive      106   35  875
```

Neutral remains the weak class: 27 of 72 correct. With 305 neutral pairs in the
whole dataset, that is the expected ceiling, not a fixable bug.

---

## The diagnostic that changes the conclusion

Headline metrics hide whether a model conditions on the aspect **at all**. Most
reviews are uniformly positive or negative, so a model that reads only overall
tone still scores well. The capability this product actually sells is the other
case.

Sliced to **mixed reviews** — the 102 test reviews (285 pairs) carrying
different polarities for different aspects:

| Model | Mixed acc | Uniform acc | Gap | **Collapsed** |
|---|---:|---:|---:|---:|
| TF-IDF + LogReg | 0.5228 | 0.8469 | +0.3240 | 0.8137 |
| DistilBERT | **0.5474** | **0.8899** | **+0.3425** | **0.8824** |

*Collapsed* = share of mixed reviews given one single polarity for every aspect.

**DistilBERT is much better overall and barely better at the actual task.** It
gains 4.3 points on uniform reviews (0.8469 → 0.8899) and 2.5 on mixed, while
ignoring the aspect *more* often than the baseline (88.2% vs 81.4% collapsed).
The gap between the two slices is 34 points either way.

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

**Why.** Most likely dataset size. 3,518 training pairs, of which mixed reviews
are a small minority, is not enough signal for the model to learn that the
second segment should override the first's overall tone. Reading tone is the
easier hypothesis and fits nearly as well.

**What this means for the reported result.** DistilBERT is selected — it is
better on every metric that was defined in advance, and changing the criterion
after seeing results would be exactly the kind of post-hoc rationalisation this
project exists to avoid. But the headline macro F1 of 0.6637 should be read
alongside the fact that **neither model reliably does aspect-conditional
sentiment at this dataset size.**

---

## Selection: different families per stage

The comparison selected **different model families for each stage**, so the
serving layer composes them rather than forcing one:

| Stage | Selected | Metric | Runner-up |
|---|---|---|---|
| A — aspect detection | **TF-IDF + OvR LogReg** | micro F1 0.7418 | DistilBERT 0.6192 † |
| B — sentiment | **DistilBERT** | macro F1 0.6637 | TF-IDF 0.5993 |

`ml/inference/predictor.py` reads `models/metadata/comparison.json` — written
from held-out test metrics — and resolves each stage independently. With no
comparison file it defaults to the baseline for both, because defaulting to the
larger model would have shipped a detector roughly 12 points worse.

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

> **Historical.** This experiment was run on the dataset as it stood *before*
> the ENTITY#ATTRIBUTE taxonomy fix, which moved 402 pairs out of `overall` and
> changed every split. The numbers below are a faithful record of that run and
> have not been reproduced since; the conclusion — that upweighting did nothing
> on dev, and that the apparent test-set gain was noise — is what carried
> forward. The models are kept under `models/_experiments/`.

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

## The fix that worked: change the unit of inference

Every result above is measured **per review**. The mixed-review gap is not a
calibration problem and not, it turns out, mainly a training problem — it is a
problem with what is being handed to the model.

M-ABSA rows are single sentences, so a single sentence is the input distribution
the sentiment model actually learned. Real Amazon reviews average ~50 words and
routinely praise one aspect while criticising another, which puts essentially
the whole corpus in the 0.55 slice — **below the 0.674 majority-class
baseline**. Running the same model per sentence puts each input back in the 0.89
slice, without retraining anything.

### Measuring it needs a benchmark that does not exist

M-ABSA cannot measure this directly: its rows are already single sentences, so
splitting is a no-op and both modes score identically. So the benchmark is
*composed* from held-out test rows ([`ml/evaluation/multi_sentence.py`](../ml/evaluation/multi_sentence.py)):

* components are drawn from the **test split only** — no training text is touched;
* components must have **disjoint aspect sets**, so the union of their gold
  labels can never be self-contradictory;
* every label is a real human annotation. Only the concatenation is synthetic.

300 pseudo-reviews, 1,012 gold aspects, 71.7% of them mixed:

| | whole-review | **sentence** | Δ |
|---|---:|---:|---:|
| **mixed-review accuracy** | 0.5917 | **0.7976** | **+0.2059** |
| **collapsed rate** | 0.9442 | **0.2791** | −0.6651 |
| detection recall | 0.5563 | **0.7875** | +0.2312 |
| overall sentiment accuracy | 0.6927 | **0.8269** | +0.1342 |
| uniform-review accuracy | **0.9610** | 0.9091 | **−0.0519** |

The collapsed rate is the headline: whole-review inference gave **94% of mixed
reviews one polarity for every aspect**. Per sentence, that falls to 28%.

### The cost, stated plainly

**Uniform reviews get worse** — 0.9610 → 0.9091. Whole-review inference has more
context when the opinion is consistent, and a sentence in isolation can be
misread. The trade is worth taking because mixed reviews are 71.7% of this
benchmark and close to 100% of real Amazon prose, and overall accuracy still
rises 13.4 points. But it is a real regression, not a free win.

**Inference costs 1.5× more.** More units, more model calls.

**It depends on punctuation.** Stripping the sentence terminators — the
adversarial case, and 23.8% of the corpus has no terminators at all — cuts the
mixed-review gain from +20.6 to **+5.7**. Sentence splitting cannot help where
there are no sentence boundaries to find.

Reproduce with `python scripts/eval_sentence_level.py`.

### What this does not fix

The model is unchanged. It is still 0.335 F1 on `neutral`, still trained on
3,518 pairs, and still reads overall tone when a sentence carries two opinions
("great screen but awful battery" remains one unit — splitting on contrastive
conjunctions was considered and left as a measured change rather than an assumed
one). Sentence splitting improves the *inputs*, not the model.

---

## What would likely fix it

Ranked by expected value, not yet attempted:

1. **DeBERTa-v3-base on a Colab GPU.** Stronger encoder, better at the
   long-range attention this needs. Cheapest experiment (~12 min on a T4). Note
   it would only be adopted for Stage B — Stage A should stay TF-IDF unless a
   transformer actually beats 0.7418.
2. **Oversample or upweight mixed reviews** during training so the easy
   tone-reading hypothesis stops being sufficient.
3. **More data.** The single biggest constraint. M-ABSA's other five English
   domains would roughly triple the corpus at the cost of a taxonomy mapping.
4. **Aspect-aware augmentation** — split mixed reviews into single-aspect
   clauses as additional training *pairs*. (The inference-time half of this idea
   is now done and is the single largest improvement so far — see above. Doing
   it during training as well is still untried.)

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
bit-identical across devices. With 5,859 pairs, treat differences under roughly
±0.02 macro F1 as noise — a proper claim needs seed-averaged runs, which is on
the roadmap and not yet done.
