"""Generate labelled phone reviews for phones the real corpus does not cover.

Why this exists
---------------
The Amazon corpus was scraped 2019-12-26. Every phone in it is at least six
years old. A search for a permissively-licensed 2025+ replacement found none:
the real corpora stop at 2023 (McAuley-Lab/Amazon-Reviews-2023, research-use,
not CC0), and the Kaggle sets advertising "2025" and "2026" are template
output. The "Global Mobile Reviews Dataset (2025 Edition)" ships 50,000 rows
containing **110 distinct review texts** -- one string repeats 724 times -- plus
`customer_name`, `age` and `exchange_rate_to_usd` columns.

So this module generates its own, and the bar it has to clear is exactly the
one those datasets fail: **distinct text**. :func:`uniqueness` reports it and
``scripts/generate_synthetic.py`` prints it, because a generator that emits 110
strings is worse than no generator at all -- it looks like data.

What the labels mean
--------------------
Labels here are **exact, not estimated**. The generator picks an aspect and a
polarity and then renders a sentence for it, so the annotation is the input to
generation rather than a judgement about the output. That is the one genuine
advantage synthetic data has, and it is why this is usable as *training*
augmentation.

What this data must never be used for
-------------------------------------
**Reporting accuracy.** Scoring a model on synthetic test data measures the
generator, not the model -- `docs/dataset.md` rejects a much larger corpus for
precisely this reason ("aspect labels would have to be manufactured, which is
exactly how you end up measuring your own label generator"). Every number in
`docs/model.md` is and stays on the real held-out M-ABSA split. Synthetic rows
are allowed into *training* only, and whether they help is then a real question
with a real answer, measured on real data.

Known limitations, stated up front
----------------------------------
* It is compositional. Diversity comes from recombining claim templates and
  slot fillers, so the *space* is large but the underlying syntax is narrower
  than human writing. A model can learn the scaffolding rather than the
  sentiment.
* It has no misspellings-by-accident, no digressions, no off-topic sentences,
  no sarcasm, and none of the formatting debris real reviews carry.
* Polarity is unambiguous by construction. Real reviews are frequently
  borderline, which is where the actual difficulty lives -- `neutral` is 5.2%
  of M-ABSA and its F1 is 0.335.

Together those mean synthetic data can plausibly teach *vocabulary* and
*aspect-to-sentiment association*, and is unlikely to teach the hard cases.
The experiment in ``scripts/compare_synthetic.py`` is what settles it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# The five slider axes. `performance` is shown as "Processor" in the UI.
ASPECTS: tuple[str, ...] = ("battery", "camera", "display", "performance", "price")
POLARITIES: tuple[str, ...] = ("negative", "neutral", "positive")

# ---------------------------------------------------------------------------
# Slot fillers. Kept generous on purpose: the whole failure mode being avoided
# is a small closed set of output strings.
# ---------------------------------------------------------------------------

SLOTS: dict[str, list[str]] = {
    "duration": [
        "a full day", "two days", "about 14 hours", "a day and a half",
        "roughly 9 hours", "an entire weekend", "most of a day", "around 11 hours",
        "a day and then some", "well over a day",
    ],
    "short_duration": [
        "half a day", "about four hours", "barely five hours", "a morning",
        "under six hours", "roughly three hours", "less than a working day",
    ],
    "usage": [
        "heavy use", "mixed use", "screen time", "gaming", "constant navigation",
        "streaming", "hotspot use", "video calls", "normal daily use",
    ],
    "great": [
        "excellent", "superb", "outstanding", "genuinely impressive", "fantastic",
        "brilliant", "very good", "seriously good", "hard to fault",
    ],
    "poor": [
        "terrible", "disappointing", "unacceptable", "poor", "frustrating",
        "well below par", "not good enough", "a real letdown",
    ],
    "ok": [
        "acceptable", "fine", "adequate", "about average", "nothing special",
        "roughly what I expected", "serviceable", "middle of the road",
    ],
    "intensifier": ["really ", "genuinely ", "absolutely ", "honestly ", "", "", ""],
    "hedge": ["I think ", "to be fair ", "in my experience ", "", "", ""],
    "pct": ["30", "40", "50", "60", "20", "45"],
    "charge_time": [
        "under an hour", "about 35 minutes", "roughly 50 minutes",
        "less than 40 minutes", "about an hour",
    ],
    "light": [
        "low light", "indoor lighting", "the evening", "dim rooms",
        "overcast weather", "night shots", "artificial light",
    ],
    "photo": ["photos", "pictures", "shots", "images", "stills"],
    "screen_use": [
        "in direct sunlight", "outdoors", "at full brightness", "on the bus",
        "in a bright room",
    ],
    "app": [
        "games", "the camera app", "heavy apps", "several apps at once",
        "video editing", "maps", "the browser with many tabs",
    ],
    "money": [
        "the money", "the price", "what it costs", "this price bracket",
        "the asking price",
    ],
    "opener": [
        "", "", "", "Overall, ", "So far, ", "After a month, ", "Two weeks in, ",
        "Coming from an older phone, ", "Honestly, ",
    ],
    "closer": [
        "", "", "", "", " Would recommend.", " No complaints there.",
        " Worth knowing before you buy.", " Just my two cents.",
        " Your mileage may vary.",
    ],
}

# ---------------------------------------------------------------------------
# Claim templates, keyed by (aspect, polarity). {slot} names index SLOTS.
# ---------------------------------------------------------------------------

CLAIMS: dict[tuple[str, str], list[str]] = {
    ("battery", "positive"): [
        "the battery {intensifier}lasts {duration} with {usage}",
        "I get {duration} on a single charge",
        "battery life is {intensifier}{great}",
        "it charges from {pct}% to full in {charge_time}",
        "still on {pct}% by the evening after {usage}",
        "the battery outlasts anything I have owned",
        "two days between charges is normal for me",
        "fast charging actually is fast, {charge_time} and done",
        "endurance is {great}, even with {usage}",
    ],
    ("battery", "negative"): [
        "the battery only lasts {short_duration}",
        "battery life is {intensifier}{poor}",
        "it drains overnight doing nothing",
        "I am hunting for a charger by mid afternoon with {usage}",
        "charging is slow, well over two hours from flat",
        "the battery degraded noticeably within months",
        "it dies at {short_duration} of {usage}",
        "battery drain during {usage} is {poor}",
    ],
    ("battery", "neutral"): [
        "battery life is {ok}",
        "the battery gets me through the day, nothing more",
        "charging speed is {ok} for the class",
        "battery is {ok}, neither a strength nor a problem",
        "I get about a day, which is what I expected",
    ],
    ("camera", "positive"): [
        "the camera is {intensifier}{great}",
        "{photo} come out sharp even in {light}",
        "photo quality is {great} for {money}",
        "the main sensor handles {light} well",
        "video stabilisation is {great}",
        "portrait shots look natural rather than over processed",
        "colours in the {photo} are accurate",
        "the ultrawide is {intensifier}{great}",
    ],
    ("camera", "negative"): [
        "the camera is {intensifier}{poor}",
        "{photo} are grainy in {light}",
        "photo quality is {poor} compared to the price",
        "the camera struggles badly in {light}",
        "colours are oversaturated and look unnatural",
        "video is soft and the stabilisation wobbles",
        "the zoom is {poor} past 2x",
        "{photo} come out blurry unless everything is perfectly still",
    ],
    ("camera", "neutral"): [
        "the camera is {ok}",
        "{photo} are {ok} in good light and weaker in {light}",
        "camera performance is {ok} for the class",
        "photos are {ok}, nothing I would show off",
    ],
    ("display", "positive"): [
        "the screen is {intensifier}{great}",
        "the display stays readable {screen_use}",
        "colours are vivid and the panel is {great}",
        "brightness is {great}, no trouble {screen_use}",
        "the refresh rate makes everything feel smooth",
        "text is razor sharp on this panel",
        "the display is easily the best part of the phone",
    ],
    ("display", "negative"): [
        "the screen is {intensifier}{poor}",
        "the display washes out {screen_use}",
        "brightness is {poor}, I cannot read it {screen_use}",
        "there is visible colour shift at an angle",
        "the panel has a green tint at low brightness",
        "the screen scratched within a fortnight",
        "touch response on the display is {poor}",
    ],
    ("display", "neutral"): [
        "the screen is {ok}",
        "the display is {ok}, bright enough indoors at least",
        "panel quality is {ok} for {money}",
        "the screen is {ok} but nothing remarkable",
    ],
    ("performance", "positive"): [
        "performance is {intensifier}{great}",
        "it handles {app} without breaking a sweat",
        "the processor is {great}, everything opens instantly",
        "no lag at all, even with {app}",
        "it stays cool and fast during {app}",
        "multitasking is {great} with this much memory",
        "the chip is {great} and it shows in {app}",
    ],
    ("performance", "negative"): [
        "performance is {intensifier}{poor}",
        "it stutters constantly with {app}",
        "the phone lags when I open {app}",
        "it overheats within minutes of {app}",
        "the processor is {poor} for {money}",
        "apps reload in the background because memory is {poor}",
        "it slowed down badly after a few months",
    ],
    ("performance", "neutral"): [
        "performance is {ok}",
        "it handles {app} {ok}, with the occasional stutter",
        "the processor is {ok} for everyday things",
        "speed is {ok}, not fast and not slow",
    ],
    ("price", "positive"): [
        "it is {intensifier}good value for {money}",
        "you cannot beat this for {money}",
        "excellent value, cheaper than the obvious alternatives",
        "for {money} this is hard to argue with",
        "it undercuts the competition and does not feel cheap",
    ],
    ("price", "negative"): [
        "it is overpriced for what you get",
        "far too expensive for {money}",
        "the price is {intensifier}{poor} given the specification",
        "you are paying a brand premium and little else",
        "I would not pay this again",
    ],
    ("price", "neutral"): [
        "the price is {ok} for the segment",
        "it costs about what you would expect",
        "value for {money} is {ok}",
    ],
}

# Joins between clauses of the same polarity, and of opposing polarity. The
# contrastive set matters most: mixed-polarity reviews are the case the model
# is measurably worst at, so a generator that only produces uniform reviews
# would augment the easy half of the problem.
SAME_POLARITY_JOINS = [". ", ". ", ", and ", ". Also, ", ". On top of that, "]
CONTRAST_JOINS = [
    ". That said, ", ", but ", ". However, ", ". The downside is that ",
    ", although ", ". On the other hand, ",
]


@dataclass(frozen=True)
class SyntheticReview:
    """One generated review and the labels it was generated from."""

    review_id: str
    text: str
    # aspect -> polarity, exact by construction.
    labels: dict[str, str]
    # aspect -> the clause generated for it, before assembly and surface noise.
    #
    # Kept because a keyword check cannot verify the labels: roughly 7% of
    # battery clauses contain no battery word at all ("it drains overnight
    # doing nothing"), which is realistic and is exactly the case a lexical
    # detector finds hard. Recording the clause makes the label auditable
    # without guessing at vocabulary.
    clauses: dict[str, str]
    phone: str
    n_sentences: int

    @property
    def is_mixed(self) -> bool:
        return len(set(self.labels.values())) > 1


def _fill(template: str, rng: random.Random) -> str:
    """Substitute {slot} placeholders with a random filler."""
    def pick(match: re.Match) -> str:
        return rng.choice(SLOTS[match.group(1)])

    return re.sub(r"\{(\w+)\}", pick, template)


def _clause(aspect: str, polarity: str, rng: random.Random) -> str:
    return _fill(rng.choice(CLAIMS[(aspect, polarity)]), rng)


def _surface_noise(text: str, rng: random.Random) -> str:
    """Roughen the output slightly.

    Real reviews are not uniformly well-formed. This does not pretend to model
    that -- it only prevents every generated sentence from sharing identical
    capitalisation and punctuation habits, which would be a trivially learnable
    shortcut.
    """
    roll = rng.random()
    if roll < 0.06:
        text = text.upper()
    elif roll < 0.14:
        text = text.lower()
    if rng.random() < 0.10:
        text = text.replace(".", "!", 1)
    if rng.random() < 0.06:
        text = text.rstrip(".") + "..."
    return text


def generate_review(
    phone: str,
    index: int,
    rng: random.Random,
    *,
    max_aspects: int = 3,
    mixed_probability: float = 0.55,
) -> SyntheticReview:
    """Generate one review with exact aspect/polarity labels.

    Args:
        phone: Product name, used only for the id and for occasional mention.
        index: Sequence number, for a stable id.
        rng: Seeded generator; the whole module is deterministic given a seed.
        max_aspects: Upper bound on aspects discussed in one review.
        mixed_probability: Share of multi-aspect reviews that carry opposing
            polarities. Set high on purpose -- 71.7% of the benchmark and
            effectively all of the real Amazon corpus is mixed, and that is the
            slice the model handles worst.
    """
    n = rng.randint(1, max_aspects)
    aspects = rng.sample(ASPECTS, n)

    # Real reviews skew positive: M-ABSA is 67.4% positive, 27.4% negative,
    # 5.2% neutral. Matching that keeps the augmented class balance from
    # drifting away from the distribution the model is evaluated on.
    weights = [0.27, 0.06, 0.67]
    labels: dict[str, str] = {}
    first = rng.choices(POLARITIES, weights=weights)[0]
    labels[aspects[0]] = first

    for aspect in aspects[1:]:
        if rng.random() < mixed_probability:
            # Deliberately oppose an earlier clause.
            opposite = "negative" if first == "positive" else "positive"
            labels[aspect] = opposite
        else:
            labels[aspect] = rng.choices(POLARITIES, weights=weights)[0]

    parts = [_clause(aspect, labels[aspect], rng) for aspect in aspects]
    clauses = dict(zip(aspects, parts, strict=True))

    text = rng.choice(SLOTS["opener"])
    text += parts[0]
    previous = labels[aspects[0]]
    for aspect, part in zip(aspects[1:], parts[1:], strict=True):
        contrast = labels[aspect] != previous
        joiner = rng.choice(CONTRAST_JOINS if contrast else SAME_POLARITY_JOINS)
        text += joiner + part
        previous = labels[aspect]

    text = text[0].upper() + text[1:] if text else text
    if not text.endswith((".", "!", "?")):
        text += "."
    text += rng.choice(SLOTS["closer"])

    return SyntheticReview(
        review_id=f"syn-{index}",
        text=_surface_noise(text.strip(), rng),
        labels=labels,
        clauses=clauses,
        phone=phone,
        n_sentences=text.count(".") + text.count("!") + text.count("?"),
    )


def generate(
    phones: list[str],
    n_reviews: int,
    *,
    seed: int = 0,
    max_aspects: int = 3,
    mixed_probability: float = 0.55,
) -> list[SyntheticReview]:
    """Generate ``n_reviews`` reviews spread evenly across ``phones``."""
    rng = random.Random(seed)
    return [
        generate_review(
            phones[index % len(phones)],
            index,
            rng,
            max_aspects=max_aspects,
            mixed_probability=mixed_probability,
        )
        for index in range(n_reviews)
    ]


def uniqueness(reviews: list[SyntheticReview]) -> float:
    """Share of generated texts that are distinct.

    The single most important quality check on this module. The Kaggle dataset
    that motivated writing it scores 110/50000 = 0.0022 here.
    """
    if not reviews:
        return 0.0
    return len({review.text for review in reviews}) / len(reviews)
