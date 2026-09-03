# The recommender

How five sliders turn into a ranked list of phones, and what the numbers mean.

---

## 1. Sliders are requirements, not weights

Each slider says **"I need at least this much"** on a 1–10 scale. A phone is
penalised for falling short and **never** for exceeding:

```
shortfall_i = max(0, requirement_i − profile_i)
```

Setting Camera to 9 and everything else to 1 means "the camera matters and
nothing else does", and the ranking follows — every phone trivially satisfies a
requirement of 1, so only the camera separates them.

### Two alternatives, and why they were rejected

**Importance weights** (a weighted mean of aspect scores) is what most
recommenders do, and it was the design one research pass recommended. It was
rejected because weights renormalise: moving Camera from 5 to 9 also changes how
much Battery counts, even though the shopper said nothing about batteries. A
requirement does not interact with the other requirements.

**Symmetric distance** — the literal reading of "closest to these desired
values" — punishes a phone for being *better* than asked. A shopper who sets
Battery to 6 does not want the 9-battery phone ranked below the 6-battery one.

The asymmetric form keeps the spirit of "closest to your preferences" while
matching what people actually mean when they move a slider up.

---

## 2. Match percentage

```
match = 100 × (1 − ‖shortfall‖ / ‖requirement − 1‖)
```

The denominator is the worst distance **this query** could produce: a
hypothetical phone scoring 1 on everything. So the number reads as *"how much of
what you asked for you actually get"*.

Two consequences worth stating plainly:

- It is **100% when every requirement is met**, however modest the request.
- It is **100% for every phone when all sliders sit at 1**, because nothing was
  asked for. The UI says so explicitly rather than presenting a meaningless
  ranking as a real one.

Normalising by a fixed constant instead would make an undemanding query report
suspiciously high matches for reasons the shopper cannot see.

**Ties** break on the mean profile score. Under a requirements model every phone
clearing every slider matches 100%, so without a tiebreak the order among them
would be arbitrary — and *"these all meet your needs, this one is best overall"*
is the sensible thing to say next.

---

## 3. Four axes from sentiment, one from price

| Axis | Source | Aspect id |
|---|---|---|
| Battery | Review sentiment | `battery` |
| Camera | Review sentiment | `camera` |
| Display | Review sentiment | `display` |
| Processor | Review sentiment | `performance` |
| **Price** | **Listed price** | — |

### Why Price is different

Price opinions in the training data are **85.6% positive** — 292 of 341 pairs,
with only 40 negative. (Before the [taxonomy fix](dataset.md) it was 90.8%.)
Shoppers record price complaints far less often than price praise, so a model
trained on this scores well by answering "positive", and every phone's price
sentiment lands in a narrow band near the top.

A Price slider driven by that would **look functional and do nothing**. Since
the corpus carries an actual listed price, the axis uses it. Price sentiment is
still computed and shown on the phone page — it is just not what the slider
ranks on, and the UI labels the axis accordingly.

**Direction.** Higher is cheaper: 10 is the least expensive phone in the
catalogue, 1 the most expensive. Every slider then points the same way, so
"Price 9" reads as "I want it cheap" exactly as "Camera 9" reads as "I want a
good camera".

**Scale.** Linear in *log* price. Price is perceived in ratios — the gap between
\$100 and \$200 feels like the gap between \$400 and \$800 — and a linear map
would squash three quarters of the catalogue (median \$200, max \$948) into the
top of the scale. Log spacing also keeps genuinely similar prices close, which a
percentile rank would not: percentiles push \$200 and \$210 apart whenever the
catalogue happens to be dense between them.

---

## 4. From reviews to a profile

```
per-sentence scores  →  per-review aspect scores  →  per-phone profile
   score_catalog.py         (already aggregated)        build_profiles.py
```

### Shrinkage

Phones carry wildly different amounts of evidence. A phone whose camera was
mentioned four times must not top the camera ranking on four enthusiastic
sentences. Each aspect mean is pulled toward the catalogue mean in proportion
to how little evidence supports it:

```
shrunk = (n · phone_mean + k · grand_mean) / (n + k)
```

`k` is estimated from the data (empirical Bayes) as within-phone variance over
between-phone variance — the point at which the two sources deserve equal
weight. A phone with n ≫ k keeps its own mean; a phone with n ≪ k is mostly told
what the average phone looks like, which is the honest answer when its own
evidence is thin.

If between-phone variance is zero — all observed spread explained by sampling
noise — `k` is infinite and every score collapses to the mean. That is a
**finding**, not a failure: it says the aspect does not discriminate.

### What is deliberately not done

The shrunk scores are **not** z-scored, percentile-ranked, or rescaled to fill
the 1–10 range.

An earlier design showed percentile ranks beside shrunk scores, in the name of
transparency. Simulated against a null where every phone is genuinely identical,
that presentation produced a confident **1.83-point spread and a full 0th-to-100th
percentile range out of pure noise**. Dividing shrunk scores by their own
compressed standard deviation *is* stretching them: the compression is the
uncertainty. `ml/evaluation/reliability.py::null_spread` reproduces it.

So a flat catalogue is displayed as a flat catalogue.

---

## 5. The gate: is any of this real?

Everything above assumes a phone's aspect score measures **the phone**. That is
testable, and until it is tested the rest is decoration.

```bash
python scripts/evaluate_recommender.py
```

**Split-half reliability.** Divide each phone's reviews at random, build the
profile twice, correlate the halves across phones. Agreement means the score
reflects the phone; disagreement means it reflects which reviews landed in which
half. Reported with the Spearman–Brown correction, since each half holds half
the data and the raw figure understates reliability at full length.

| Corrected ρ | Reading |
|---|---|
| ≥ 0.80 | strong |
| ≥ 0.60 | usable |
| ≥ 0.40 | weak |
| < 0.40 | **noise** |

**Beating the star rating.** Amazon already ships a per-product number. If the
aspect scores track it almost perfectly, the whole pipeline is an expensive
re-derivation of a column that was already in the CSV. A high correlation is not
a bug — good phones do get good stars *and* good battery reviews — but the
closer R² gets to 1, the less the aspect adds.

**Null baseline.** How much apparent spread identical phones produce for free.
Any observed spread below it is not evidence of anything.

The script prints **PASS**, **PARTIAL** or **FAIL** rather than a dashboard. A
FAIL means the sliders would be reordering noise, and the recommender should not
ship on those profiles.

---

## 6. Why phones are excluded

A phone is left out of the recommendations when:

| Reason | Where |
|---|---|
| Its listing title could not be resolved to a specific model | `normalise.py::is_usable` |
| Fewer than 20 reviews | `CatalogConfig.min_reviews` |
| No listed price (the Price axis needs one) | `CatalogConfig.require_price` |
| An aspect has fewer than 5 detected mentions | `profiles.py::MIN_MENTIONS` |

Missing axes are **never imputed**. Filling a gap with the catalogue mean would
let a phone be ranked on a measurement that was never taken; the phone page
shows a dash and says why.

---

## 7. Reproducing the pipeline

```bash
python scripts/download_phones.py      # CC0 corpus, no credentials
python scripts/build_catalog.py        # seconds
python scripts/score_catalog.py        # ~1 hour on CPU; resumable
python scripts/build_profiles.py       # seconds
python scripts/evaluate_recommender.py # the gate
```

Only `score_catalog.py` is expensive, and it writes per-review rows rather than
per-phone averages precisely so that aggregation can be re-tried without
re-running inference.
