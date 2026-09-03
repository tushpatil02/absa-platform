"""Tests for the data pipeline.

The leakage tests matter most. Everything else can be wrong and merely produce a
worse model; leakage produces a *better-looking* model that is not real, which is
exactly the failure this rebuild exists to avoid.

Run with:  .venv/Scripts/python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.clean import clean_text, fix_encoding, is_usable
from ml.preprocessing.parse import ParseReport, RawReview, Triplet, parse_file, parse_line
from ml.preprocessing.split import (
    assert_no_leakage,
    deduplicate_across_splits,
    normalise_for_dedup,
)
from ml.preprocessing.transform import (
    TransformReport,
    load_taxonomy,
    resolve_polarity,
    transform,
)

TAXONOMY_PATH = REPO_ROOT / "ml" / "config" / "aspect_taxonomy.yaml"


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_line_reads_text_and_triplets():
    line = "The camera is great.####[['camera', 'Camera#General', 'positive']]"
    text, triplets = parse_line(line)
    assert text == "The camera is great."
    assert triplets == [Triplet("camera", "Camera#General", "positive")]


def test_parse_line_handles_multiple_triplets():
    line = (
        "Great camera, bad battery.####"
        "[['camera', 'Camera#General', 'positive'], "
        "['battery', 'Battery/Longevity#General', 'negative']]"
    )
    _, triplets = parse_line(line)
    assert len(triplets) == 2
    assert {t.polarity for t in triplets} == {"positive", "negative"}


@pytest.mark.parametrize(
    "line",
    [
        "no separator here",
        "####[['a', 'b', 'positive']]",          # empty text
        "text####not a literal",                  # unparseable payload
        "text####[['only', 'two']]",             # wrong triplet arity
        "text####'a string not a list'",
    ],
)
def test_parse_line_rejects_malformed(line):
    assert parse_line(line) is None


def test_parse_line_does_not_execute_code():
    """literal_eval must refuse a call expression rather than run it."""
    assert parse_line("text####__import__('os').system('echo pwned')") is None


def test_parse_file_records_invalid_polarity(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text(
        "Good phone.####[['phone', 'Overall#Overall', 'excellent']]\n"
        "Bad phone.####[['phone', 'Overall#Overall', 'negative']]\n",
        encoding="utf-8",
    )
    report = ParseReport()
    reviews = parse_file(path, "phone", "train", report)

    assert len(reviews) == 2
    assert reviews[0].triplets == ()          # 'excellent' dropped
    assert len(reviews[1].triplets) == 1
    assert len(report.invalid_polarity) == 1


# ---------------------------------------------------------------------------
# clean -- the point is what is PRESERVED
# ---------------------------------------------------------------------------


def test_clean_preserves_negation_and_case_and_punctuation():
    text = "The battery is NOT good! Why?"
    assert clean_text(text) == "The battery is NOT good! Why?"


def test_clean_preserves_sentiment_bearing_stopwords():
    for token in ("not", "no", "never", "but", "however"):
        assert token in clean_text(f"It is {token} working").lower()


def test_clean_strips_html_and_unescapes_entities():
    assert clean_text("<b>Great</b> value &amp; fast") == "Great value & fast"


def test_clean_removes_known_boilerplate():
    text = "Your browser does not support HTML5 video. The camera is good."
    assert "browser" not in clean_text(text)
    assert "camera is good" in clean_text(text)


def test_clean_masks_urls_and_emails():
    out = clean_text("See https://example.com or mail a@b.com")
    assert "[URL]" in out and "[EMAIL]" in out
    assert "example.com" not in out


def test_clean_collapses_long_punctuation_runs_but_keeps_emphasis():
    assert clean_text("Terrible!!!!!!") == "Terrible!!"


def test_fix_encoding_repairs_mojibake():
    """UTF-8 bytes mis-decoded as cp1252 must be recovered.

    The corrupted input is built programmatically rather than pasted as a
    literal, so this test cannot be silently broken by an editor re-encoding
    the file (which is exactly what happened while writing it).
    """
    original = "it" + chr(0x2019) + "s good"
    mojibake = original.encode("utf-8").decode("cp1252")
    assert mojibake != original                      # precondition: really corrupted
    assert fix_encoding(mojibake) == "it's good"     # repaired, then quote folded


def test_fix_encoding_folds_typographic_quotes():
    """NFKC does not fold U+2019 or en dashes, so the explicit table must."""
    assert fix_encoding("don" + chr(0x2019) + "t") == "don't"
    assert fix_encoding(chr(0x201C) + "great" + chr(0x201D)) == '"great"'
    assert fix_encoding("a " + chr(0x2013) + " b") == "a - b"


def test_fix_encoding_leaves_clean_ascii_untouched():
    assert fix_encoding("The battery is not good!") == "The battery is not good!"


def test_fix_encoding_removes_replacement_char():
    assert "�" not in fix_encoding("bad � byte")


@pytest.mark.parametrize("text,expected", [("Great phone!", True), ("", False), (".", False), ("12345", False)])
def test_is_usable(text, expected):
    assert is_usable(text) is expected


# ---------------------------------------------------------------------------
# taxonomy + transform
# ---------------------------------------------------------------------------


def test_taxonomy_loads_twelve_aspects():
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert len(taxonomy.aspect_ids) == 12
    assert len(set(taxonomy.aspect_ids)) == 12


def test_taxonomy_maps_both_disjoint_schemes_to_same_aspect():
    """phone and laptop use different label schemes; both must reach 'battery'."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("Battery/Longevity#Battery Life", "phone") == "battery"
    assert taxonomy.map_category("BATTERY#OPERATION_PERFORMANCE", "laptop") == "battery"


def test_taxonomy_returns_none_for_dropped_labels():
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("Product Accessories#Phone Cases", "phone") is None


def test_taxonomy_raises_on_unknown_label():
    """An unmapped label must fail the build, not vanish silently."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    with pytest.raises(KeyError, match="Unmapped entity"):
        taxonomy.map_category("Telepathy#Mind Reading", "phone")


def test_taxonomy_raises_on_unknown_product_attribute():
    """A new attribute on a whole-product entity must fail loudly too.

    Rule 2 has no entity fallback: silently routing an unrecognised
    ``LAPTOP#SOMETHING`` to `overall` is how LAPTOP#PRICE went unnoticed.
    """
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    with pytest.raises(KeyError, match="Unmapped attribute"):
        taxonomy.map_category("LAPTOP#TELEPATHY", "laptop")


def test_price_attribute_overrides_the_entity():
    """Rule 1: "<component> is good for the PRICE" is about price.

    The regression this pins: mapping on the entity alone sent all 116
    ``LAPTOP#PRICE`` rows to `overall`, and scattered the rest across the
    component aspects, leaving the price class trained on phone data only.
    """
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("LAPTOP#PRICE", "laptop") == "price"
    assert taxonomy.map_category("DISPLAY#PRICE", "laptop") == "price"
    assert taxonomy.map_category("HARD_DISC#PRICE", "laptop") == "price"


def test_entity_scoped_attributes_do_not_override():
    """Rule 1 is deliberately narrow.

    QUALITY and OPERATION_PERFORMANCE describe *the entity*, so they must keep
    resolving through the entity map. Promoting them would empty the component
    aspects into two catch-alls.
    """
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("DISPLAY#QUALITY", "laptop") == "display"
    assert taxonomy.map_category("CPU#OPERATION_PERFORMANCE", "laptop") == "performance"
    assert taxonomy.map_category("BATTERY#QUALITY", "laptop") == "battery"


def test_whole_product_entity_maps_on_its_attribute():
    """Rule 2: LAPTOP names no aspect, so the attribute has to carry it."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("LAPTOP#OPERATION_PERFORMANCE", "laptop") == "performance"
    assert taxonomy.map_category("LAPTOP#QUALITY", "laptop") == "build_quality"
    assert taxonomy.map_category("LAPTOP#PORTABILITY", "laptop") == "design"
    assert taxonomy.map_category("LAPTOP#GENERAL", "laptop") == "overall"


def test_phone_scheme_is_unaffected_by_the_new_rules():
    """Verified against all 4,810 phone triplets: `#Price` and `#Value for
    Money` occur under the `Price` entity and nowhere else, so rule 3 alone
    already handled phone. These pin that the new rules changed nothing there.
    """
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.map_category("Price#Price", "phone") == "price"
    assert taxonomy.map_category("Price#Value for Money", "phone") == "price"
    assert taxonomy.map_category("Screen#Clarity", "phone") == "display"
    assert taxonomy.map_category("Camera#Pixel", "phone") == "camera"


def test_split_category_handles_missing_and_extra_separators():
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.split_category("Overall") == ("Overall", "")
    assert taxonomy.split_category("BATTERY#QUALITY") == ("BATTERY", "QUALITY")
    # Everything after the first "#" is the attribute.
    assert taxonomy.split_category("A#B#C") == ("A", "B#C")
    assert taxonomy.split_category("  BATTERY # QUALITY ") == ("BATTERY", "QUALITY")


def test_polarity_ids_are_stable():
    """The model head, ONNX output and API all depend on this order."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    assert taxonomy.polarity_to_id == {"negative": 0, "neutral": 1, "positive": 2}


def test_resolve_polarity_majority_wins():
    counts = collections.Counter({"positive": 3, "negative": 1})
    assert resolve_polarity(counts) == "positive"


def test_resolve_polarity_tie_never_invents_a_label():
    """A positive/negative tie must resolve to one of those two, not 'neutral'.

    Manufacturing a label no annotator assigned would be exactly the kind of
    fabrication that makes a headline metric meaningless.
    """
    counts = collections.Counter({"positive": 1, "negative": 1})
    assert resolve_polarity(counts) == "negative"


def test_resolve_polarity_tie_prefers_annotated_neutral():
    """When neutral IS among the tied labels, it wins."""
    counts = collections.Counter({"positive": 1, "neutral": 1})
    assert resolve_polarity(counts) == "neutral"


def test_transform_deduplicates_repeated_triplets():
    """M-ABSA repeats identical triplets on ~3.5% of rows; they must not double-vote."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    review = RawReview(
        text="Memory is huge.",
        triplets=(
            Triplet("Memory", "Product Configuration#Operating Memory", "positive"),
            Triplet("Memory", "Product Configuration#Operating Memory", "positive"),
        ),
        source_domain="phone",
        source_split="train",
        line_number=1,
    )
    report = TransformReport()
    pairs = transform([review], taxonomy, report)

    assert report.duplicate_triplets_removed == 1
    assert len(pairs) == 1
    assert pairs[0].aspect == "performance"


def test_transform_merges_multiple_triplets_into_one_pair():
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    review = RawReview(
        text="Screen is bright but scratches.",
        triplets=(
            Triplet("Screen", "Screen#Clarity", "positive"),
            Triplet("screen", "Screen#General", "positive"),
            Triplet("scratches", "Screen#General", "negative"),
        ),
        source_domain="phone",
        source_split="train",
        line_number=1,
    )
    pairs = transform([review], taxonomy, TransformReport())
    assert len(pairs) == 1                    # one (review, display) row
    assert pairs[0].polarity == "positive"    # 2 positive vs 1 negative


# ---------------------------------------------------------------------------
# split -- leakage
# ---------------------------------------------------------------------------


def _pair(review_id, text, split, aspect="battery", polarity="positive"):
    from ml.preprocessing.transform import AspectPair

    return AspectPair(review_id, text, aspect, polarity, "phone", split, ())


def test_normalise_for_dedup_ignores_case_and_punctuation():
    assert normalise_for_dedup("Great phone!") == normalise_for_dedup("great phone")


def test_assert_no_leakage_passes_on_clean_split():
    pairs = [_pair("r1", "good phone", "train"), _pair("r2", "bad phone", "test")]
    assert_no_leakage(pairs)  # must not raise


def test_assert_no_leakage_catches_shared_review_id():
    pairs = [_pair("r1", "good phone", "train"), _pair("r1", "good phone", "test")]
    with pytest.raises(AssertionError, match="review_id"):
        assert_no_leakage(pairs)


def test_assert_no_leakage_catches_same_text_under_different_ids():
    """The subtle case: identical text, different ids, opposite splits."""
    pairs = [_pair("r1", "Great phone!", "train"), _pair("r2", "great phone", "test")]
    with pytest.raises(AssertionError, match="review text"):
        assert_no_leakage(pairs)


def test_deduplicate_moves_duplicates_to_test_and_output_is_leak_free():
    pairs = [_pair("r1", "Great phone!", "train"), _pair("r2", "great phone", "test")]
    kept, report = deduplicate_across_splits(pairs)

    assert report.cross_split_duplicates_resolved == 1
    assert {p.split for p in kept} == {"test"}   # test set wins
    assert_no_leakage(kept)                       # and the result is clean


def test_deduplicate_is_idempotent():
    pairs = [_pair("r1", "Great phone!", "train"), _pair("r2", "great phone", "test")]
    once, _ = deduplicate_across_splits(pairs)
    twice, report = deduplicate_across_splits(once)
    assert len(once) == len(twice)
    assert report.cross_split_duplicates_resolved == 0
