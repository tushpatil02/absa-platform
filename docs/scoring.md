# The 1–10 sentiment score

How the slider value is computed, and why it is computed that way.

The original college project had a 1–10 slider whose values were essentially
decorative. This document defines the score so that every value is derived from
the model's actual output distribution, with nothing invented.

---

## The mapping

The sentiment classifier emits a probability distribution over three ordered
classes. Treat polarity as an **ordinal axis** with fixed anchors:

| Class | Anchor |
|---|---|
| negative | 0.0 |
| neutral | 0.5 |
| positive | 1.0 |

Take the **expected value** under the predicted distribution, and rescale to 1–10:

```
positivity = 1.0·P(positive) + 0.5·P(neutral) + 0.0·P(negative)
score      = 1 + 9 · positivity
```

Implemented in [`ml/inference/scoring.py`](../ml/inference/scoring.py).

### Worked example

For *"The camera is excellent, but the battery drains very quickly"*, the model
returns for **camera**:

```
P(negative) = 0.04    P(neutral) = 0.11    P(positive) = 0.85
positivity  = 1.0(0.85) + 0.5(0.11) + 0.0(0.04) = 0.905
score       = 1 + 9(0.905) = 9.15
confidence  = max(P) = 0.85
```

---

## Why expected value rather than bucketing

The obvious alternative is to take the argmax class and assign it a band —
negative → 2, neutral → 5, positive → 9. That is worse in four specific ways.

**It is continuous.** A barely-positive prediction (0.45 / 0.10 / 0.45) scores
5.5, near the middle where it belongs. Bucketing would send it to the top of the
positive band, indistinguishable from an emphatic rave.

**It is monotonic.** The score rises with `P(positive)` and falls with
`P(negative)`, always. The slider can never move opposite to the model.

**The endpoints are exact.**

| Distribution | Score | Reading |
|---|---|---|
| P(negative) = 1 | **1.00** | Extremely negative |
| P(neutral) = 1 | **5.50** | Neutral — the midpoint of 1–10 |
| P(positive) = 1 | **10.00** | Extremely positive |
| uniform (⅓, ⅓, ⅓) | **5.50** | Maximally uncertain, also neutral in expectation |

**It has no free parameters.** There is nothing to tune to make the UI look
better, which is exactly the failure mode this project is avoiding.

> Note that 5.5 is the neutral point, not 5. The scale runs 1–10 and has no zero,
> so its midpoint is 5.5.

---

## Confidence is reported separately

`confidence = max(P)` is returned as its own field and is **never blended into
the score**. They answer different questions:

- **Score** — how positive is this?
- **Confidence** — how sure is the model?

Blending them makes both unreadable. Consider two predictions that both score
**5.50**:

| | P(neg) | P(neu) | P(pos) | Score | Confidence |
|---|---|---|---|---|---|
| Confident neutral | 0.00 | 1.00 | 0.00 | 5.50 | **1.00** |
| Three-way coin flip | 0.33 | 0.33 | 0.33 | 5.50 | **0.33** |

Identical scores, completely different epistemic states. The confidence field is
the only thing that distinguishes them, which is why the UI always shows both.

There is a test pinning this property
([`tests/test_scoring.py`](../tests/test_scoring.py)).

---

## Band labels

The score is continuous; these bands only supply the human-readable label shown
beside the number. They never alter the score.

| Score range | Label |
|---|---|
| 1.00 – 1.75 | Extremely Negative |
| 1.75 – 3.25 | Negative |
| 3.25 – 4.75 | Slightly Negative |
| **4.75 – 6.25** | **Neutral** (centred on 5.5) |
| 6.25 – 7.75 | Slightly Positive |
| 7.75 – 9.25 | Positive |
| 9.25 – 10.00 | Extremely Positive |

The bands are **symmetric about 5.5**: reflecting a score `s` to `11 − s` lands
in the mirrored band. This is not cosmetic. An earlier version of this table was
asymmetric and labelled the exact neutral midpoint "Slightly Positive" — a pure
neutral prediction was reported as leaning positive. A test now enforces the
symmetry.

---

## Product-level aggregation

For a product summary, per-aspect scores are averaged across reviews:

```
product_score(aspect) = mean(score of each review mentioning that aspect)
```

Because the score is linear in the probabilities, the mean of per-review scores
equals the score of the pooled distribution. Aggregation therefore stays
consistent with the per-review definition instead of introducing a second,
different rule. (Tested in `test_aggregate_scores_matches_pooled_expectation`.)

Polarity **shares** are computed over reviews that *mention* the aspect, not over
all reviews — "61% negative on battery" means 61% of the people who discussed
battery. Any other denominator would make the number meaningless.

---

## Limitations

Stated because they bound what the number means.

1. **Softmax outputs are over-confident.** Neural classifiers are known to be
   poorly calibrated: a reported 0.9 does not mean 90% of such predictions are
   correct. The *ordering* of scores is trustworthy; the absolute confidence is
   not a probability of correctness. The UI labels it "confidence", never
   "accuracy", for this reason.
2. **The anchors are a modelling choice.** Placing neutral at 0.5 assumes it sits
   exactly between the poles. That is the natural reading of an ordinal scale,
   but it is an assumption, not a measurement.
3. **Neutral is the weakest class.** It is 5.3% of the training data and has the
   lowest per-class F1. Scores near 5.5 are therefore the least reliable region
   of the scale.
4. **No human calibration study.** Nothing here has been checked against human
   ratings of the same reviews. A score of 8.2 is defined by the formula above,
   not validated as "what a person would call 8.2 out of 10".

A future improvement would be temperature scaling fitted on the dev split, which
would make the confidence figure meaningfully calibrated without changing the
score's definition.
