"""Tests for listing-title normalisation.

The failure modes are asymmetric, and the tests are weighted accordingly.
Under-merging splits one phone's reviews across several catalogue entries and
costs statistical power. **Over-merging pools reviews from different phones**
and attributes them to one, which is silently wrong. The over-merge tests below
are the ones that matter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.catalog.normalise import canonical_model, is_usable, model_key

# ---------------------------------------------------------------------------
# Merging variants of one phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Samsung Galaxy Note 5, Black 64GB (Verizon Wireless)",
        "Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon Wireless)",
        "Samsung Galaxy Note 5 SM-N920T 32GB Black Smartphone for T-Mobile",
        "Samsung Galaxy Note 5 SM-N920T 64gb - Unlocked Cellphone GSM - Platinum Gold",
    ],
)
def test_colour_storage_carrier_and_sku_all_collapse(title):
    """19 listings in the corpus are this one phone."""
    assert canonical_model(title, "Samsung") == "Samsung Galaxy Note 5"


def test_plus_and_the_word_plus_are_one_phone():
    assert canonical_model("Samsung Galaxy S8+ 64GB GSM Unlocked", "Samsung") == (
        canonical_model("Samsung Galaxy S8 Plus 64GB Unlocked", "Samsung")
    )


def test_case_differences_group_together():
    assert model_key(canonical_model("Samsung GALAXY S7 EDGE 32GB", "Samsung")) == model_key(
        canonical_model("Samsung Galaxy S7 Edge Smartphone", "Samsung")
    )


def test_moto_prefix_is_optional():
    """The corpus lists the same phone as "Motorola G6" and "Motorola Moto G6"."""
    assert canonical_model("Motorola Moto G6 32GB Unlocked", "Motorola") == canonical_model(
        "Motorola G6 32GB Unlocked", "Motorola"
    )


def test_spec_prose_is_dropped():
    # Compared through model_key: canonical_model preserves the title's own
    # casing, and grouping is case-insensitive by design.
    title = "Samsung GALAXY S6 G920 32GB Unlocked GSM 4G LTE Octa-Core Smartphone - Black Sapphire"
    assert model_key(canonical_model(title, "Samsung")) == "samsung galaxy s6"


# ---------------------------------------------------------------------------
# NOT merging different phones -- the expensive failure
# ---------------------------------------------------------------------------


def test_nokia_decimal_model_numbers_survive():
    """The regression this pins.

    Stripping decimals to kill "Windows 8.1" also deleted Nokia's actual model
    numbers, collapsing eight distinct handsets into a bare "Nokia" holding 692
    pooled reviews. Only a version attached to an OS name is noise.
    """
    for number in ("3.1", "6.1", "7.1", "2.2", "4.2", "7.2"):
        title = f"Nokia {number} - Android 9.0 Pie - 32 GB - Dual SIM Unlocked Smartphone"
        assert canonical_model(title, "Nokia") == f"Nokia {number}"


def test_distinct_nokia_models_stay_distinct():
    names = {
        canonical_model(f"Nokia {n} - Android 9.0 Pie - 32 GB - Dual SIM", "Nokia")
        for n in ("3.1", "6.1", "7.1", "2.2")
    }
    assert len(names) == 4


@pytest.mark.parametrize(
    "left,right",
    [
        ("Samsung Galaxy S7 32GB", "Samsung Galaxy S7 Edge 32GB"),
        ("Samsung Galaxy Note 8 64GB", "Samsung Galaxy Note 9 64GB"),
        ("Apple iPhone 7 32GB", "Apple iPhone 7 Plus 32GB"),
        ("Google Pixel 2 64GB", "Google Pixel 2 XL 64GB"),
        ("Samsung Galaxy S8 64GB", "Samsung Galaxy S8 Plus 64GB"),
    ],
)
def test_neighbouring_models_are_not_merged(left, right):
    """A base model and its variant are different phones with different scores."""
    assert canonical_model(left, None) != canonical_model(right, None)


def test_os_version_is_removed_but_not_model_numbers():
    title = "Nokia Lumia 635 8GB Unlocked GSM 4G LTE Windows 8.1 Quad-Core Phone - Black"
    assert canonical_model(title, "Nokia") == "Nokia Lumia 635"


# ---------------------------------------------------------------------------
# Normalisation failures must be excluded, not pooled
# ---------------------------------------------------------------------------


def test_a_bare_brand_is_not_usable():
    assert not is_usable("Nokia", "Nokia")
    assert not is_usable("Samsung", "Samsung")


def test_a_real_model_is_usable():
    assert is_usable("Samsung Galaxy S7", "Samsung")
    assert is_usable("Nokia 3.1", "Nokia")


def test_empty_input_is_not_usable():
    assert not is_usable(canonical_model("", None))
    assert not is_usable(canonical_model("Unlocked GSM Smartphone", None))


# ---------------------------------------------------------------------------
# Specific leaks that were caught by hand
# ---------------------------------------------------------------------------


def test_slash_suffixed_sku_is_fully_removed():
    """"SM-J500H/DS" once left a bare "SM" in the model name."""
    title = "Samsung Galaxy J5 SM-J500H/DS GSM Factory Unlocked Smartphone"
    assert canonical_model(title, "Samsung") == "Samsung Galaxy J5"


def test_w_slash_abbreviation_does_not_leak():
    """r"\\bw/\\b" never matches: "/" is a non-word char, so the trailing \\b
    requires a word character straight after the slash."""
    title = "Samsung Galaxy S5 G900V Verizon 4G LTE Smartphone w/ 16MP Camera - Black"
    assert canonical_model(title, "Samsung") == "Samsung Galaxy S5"


def test_mojibake_in_titles_is_repaired():
    title = "Samsung Galaxy Note 5, Black" + chr(0xFFFD) + " 64GB (Verizon Wireless)"
    assert canonical_model(title, "Samsung") == "Samsung Galaxy Note 5"


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_brand_is_prepended_when_missing_from_the_title():
    assert canonical_model("Galaxy S7 32GB Unlocked", "Samsung").startswith("Samsung")


def test_brand_casing_is_canonical():
    assert canonical_model("huawei Mate 10 Pro 64GB", "huawei").startswith("HUAWEI")


def test_normalisation_is_deterministic():
    title = "Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon Wireless)"
    assert canonical_model(title, "Samsung") == canonical_model(title, "Samsung")


@pytest.mark.parametrize("title", ["", None, "   ", "!!!"])
def test_degenerate_titles_do_not_crash(title):
    canonical_model(title, "Samsung")


def test_model_key_is_whitespace_insensitive():
    assert model_key("  Samsung   Galaxy  S7 ") == "samsung galaxy s7"
