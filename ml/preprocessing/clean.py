"""Conservative text cleaning for ABSA.

Cleaning sentiment data is mostly an exercise in restraint. Classic NLP
"cleaning" -- lowercasing, stripping punctuation, removing stopwords -- destroys
exactly the signal an ABSA model needs:

* ``not`` / ``never`` / ``no`` are stopwords in most lists, yet they invert
  polarity ("not bad" is positive).
* ``!`` and ``?`` and repeated punctuation carry intensity.
* Capitalisation carries emphasis ("BATTERY LIFE IS TERRIBLE").

So this module removes only things that are unambiguously *not* language:
scraper boilerplate, markup, encoding damage, and whitespace noise. Everything
that could plausibly carry sentiment is left exactly as written.

Every removal rule is listed in ``BOILERPLATE_PATTERNS`` with a comment saying
why, so the decisions are auditable.
"""

from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# Boilerplate observed in the M-ABSA source data. These are scraper artefacts
# from the review pages, not things a customer wrote.
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Video player fallback text scraped from review pages.
    re.compile(r"your browser does not support html5? video\.?", re.IGNORECASE),
    # Truncated-review "read more" affordances.
    re.compile(r"\bread more\b\.?", re.IGNORECASE),
    # "See more reviews like this" style trailers.
    re.compile(r"\bsee (?:all|more) reviews?\b\.?", re.IGNORECASE),
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]{1,80}>")

# U+FFFD is what a failed UTF-8 decode leaves behind; it is never meaningful.
REPLACEMENT_CHAR = "�"

# Typographic variants NFKC leaves alone. Folding them keeps "don't" and the
# U+2019 spelling as one token instead of two. Keep this file UTF-8: the keys
# below are literal characters (U+2018/2019/201A/201C/201D/201E/2013/2014/2026/00A0).
TYPOGRAPHIC_TRANSLATIONS = str.maketrans(
    {
        "‘": "'",    # left single quote
        "’": "'",    # right single quote / apostrophe
        "‚": "'",    # single low-9 quote
        "“": '"',    # left double quote
        "”": '"',    # right double quote
        "„": '"',    # double low-9 quote
        "–": "-",    # en dash
        "—": "-",    # em dash
        "…": "...",  # horizontal ellipsis
        " ": " ",    # non-breaking space
    }
)

# Collapse 3+ identical punctuation marks to 2. "!!!!!!" and "!!" mean the same
# thing to a tokenizer, but the long run wastes tokens and creates rare types.
# Two are kept so the emphasis itself survives.
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1{2,}")

WHITESPACE_PATTERN = re.compile(r"\s+")

# Placeholders keep the *fact* that a link/email was present without leaking a
# high-cardinality token. Removing them outright would silently change sentences
# like "see <url> for proof" into something ungrammatical.
URL_PLACEHOLDER = " [URL] "
EMAIL_PLACEHOLDER = " [EMAIL] "


def fix_encoding(text: str) -> str:
    """Repair mojibake and normalise Unicode.

    Three distinct problems are handled:

    1. **Mojibake** -- UTF-8 bytes previously decoded as cp1252, which turns
       an apostrophe into ``â€™``. Recovered by round-tripping through cp1252
       when that round-trip is lossless.
    2. **Compatibility forms** -- NFKC folds full-width Latin, ligatures and
       similar variants onto canonical forms.
    3. **Typographic punctuation** -- NFKC does *not* touch curly quotes or
       en/em dashes, so ``don't`` and ``don’t`` would otherwise be two distinct
       tokens for every contraction in the corpus. They are folded explicitly.
       This changes punctuation shape only, never wording or polarity.
    """
    if "â€" in text or "Ã" in text:
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass  # Not actually mojibake; leave the original alone.
        else:
            text = repaired

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TYPOGRAPHIC_TRANSLATIONS)
    return text.replace(REPLACEMENT_CHAR, "")


def clean_text(
    text: str,
    *,
    mask_urls: bool = True,
    mask_emails: bool = True,
) -> str:
    """Clean one review.

    Preserves casing, punctuation, negations and stopwords by design.

    Args:
        text: Raw review text.
        mask_urls: Replace URLs with ``[URL]`` rather than deleting them.
        mask_emails: Replace e-mail addresses with ``[EMAIL]``.

    Returns:
        The cleaned text, or ``""`` if nothing usable survived.
    """
    if not text:
        return ""

    text = html.unescape(text)  # &amp; -> &, &#39; -> '
    text = fix_encoding(text)
    text = HTML_TAG_PATTERN.sub(" ", text)

    if mask_urls:
        text = URL_PATTERN.sub(URL_PLACEHOLDER, text)
    if mask_emails:
        text = EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)

    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)

    text = REPEATED_PUNCT_PATTERN.sub(r"\1\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def is_usable(text: str, *, min_chars: int = 3, min_letters: int = 2) -> bool:
    """Whether a cleaned review carries enough signal to keep.

    Rejects the degenerate leftovers that cleaning can produce: empty strings,
    a lone punctuation mark, or a row of digits. Deliberately permissive --
    short reviews like "Great phone!" are perfectly valid training data.
    """
    if len(text) < min_chars:
        return False
    return sum(character.isalpha() for character in text) >= min_letters
