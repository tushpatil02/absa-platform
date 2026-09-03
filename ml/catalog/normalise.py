"""Turn Amazon listing titles into canonical phone models.

The corpus has 720 listings but far fewer phones. One model appears many times
over -- 19 listings for the Galaxy Note 5 alone -- differing only by colour,
storage, carrier and condition::

    Samsung Galaxy Note 5, Black 64GB (Verizon Wireless)
    Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon Wireless)
    Samsung Galaxy Note 5 SM-N920T 32GB Black Smartphone for T-Mobile

Left unmerged, the recommender shows the same phone nineteen times and splits
its reviews nineteen ways, so no listing accumulates enough evidence for a
stable sentiment profile. Merging is what makes per-phone scores possible.

Failure modes, and which one to prefer
--------------------------------------
Under-merging fragments a phone's reviews and costs statistical power.
**Over-merging is worse**: it pools reviews from genuinely different phones and
attributes them all to one, which is silently wrong rather than merely weak.
So where the two trade off, this module under-merges. In particular, a title
that reduces to nothing but the brand is treated as a *failure* and excluded --
see :func:`is_usable`. Pooling those would have produced one fake "Nokia"
product holding 692 reviews from eight different handsets.
"""

from __future__ import annotations

import re

from ml.preprocessing.clean import fix_encoding

# An OS and its version number, removed as a phrase.
#
# This must run before anything else and must NOT be generalised to "strip
# decimals". Nokia's model names *are* decimals -- Nokia 3.1, 6.1, 7.2 -- so a
# blanket decimal strip collapsed eight distinct Nokia phones into a bare
# "Nokia". Only the version attached to an OS name is noise.
OS_VERSION = re.compile(r"\b(?:windows|android|ios|android\s+one)\s*\d+(?:\.\d+)*\b", re.I)

# Everything after one of these is packaging, not identity.
#
# "w/" deliberately has no trailing word boundary: "/" is a non-word character,
# so r"\bw/\b" demands a word character immediately after the slash and never
# fires on "Smartphone w/ 16MP Camera", which left a stray "w" in the name.
CUT = re.compile(
    r"\s*(?:,|\(|\[|\||\bwith\b|\bw/| - |–|\bfor\b|\bincludes?\b|\bbundle\b)",
    re.I,
)

# Words that describe the offer or the spec sheet rather than the phone.
NOISE = re.compile(
    r"\b("
    r"unlocked|factory|international|version|global|model|gsm|cdma|lte|4g|5g|3g|volte|"
    r"at&t|att|verizon|sprint|t-?mobile|tmobile|metropcs|cricket|boost|xfinity|"
    r"tracfone|straight ?talk|net10|page ?plus|prepaid|no ?contract|carrier|"
    r"smartphone|cell ?phone|phone|mobile|handset|cellphone|"
    r"renewed|refurbished|certified|used|new|open ?box|warranty|excellent|"
    r"android|windows|ios|pie|oreo|nougat|marshmallow|"
    r"dual ?sim|single ?sim|sim ?free|microsd|"
    r"octa|quad|hexa|deca|core|ram|rom|display|camera|triple|rear|front|"
    r"fhd|hd|amoled|oled|lcd|inch|snapdragon|mediatek|exynos|helio|"
    r"black|white|blue|red|gold|silver|grey|gray|green|purple|pink|rose|onyx|"
    r"midnight|space|graphite|titanium|coral|lavender|platinum|mist|frost|sapphire|"
    r"\d+\s?gb|\d+\s?tb|\d+\s?mb|\d+\s?mp"
    r")\b",
    re.I,
)

# Manufacturer SKU codes: SM-N920V, G900V, XT1254, N950U, A505G, I917, G935FD.
# These name a carrier or region variant, not a different phone.
SKU = re.compile(
    r"(?:"
    r"SM-[A-Z0-9]+"                    # Samsung
    r"|SCH-[A-Z0-9]+"                  # Samsung CDMA
    r"|XT\d{3,}[A-Z]*"                 # Motorola
    r"|[A-Z]{1,2}\d{3,4}[A-Z]{0,3}"    # G900V, N950U, A505G, I917, G935FD
    r")",
    re.I,
)

# Short letter+digit combinations that are model names, not SKUs: S7, A10, G6,
# Note 9. Without this the SKU pattern would eat the model itself.
MODEL_TOKEN = re.compile(r"^(?:[A-Z]\d{1,2}|note\d?|mate\d+|nova\d+|redmi.*)$", re.I)

BRAND_CANONICAL = {
    "apple": "Apple",
    "asus": "ASUS",
    "blu": "BLU",
    "google": "Google",
    "htc": "HTC",
    "huawei": "HUAWEI",
    "lg": "LG",
    "motorola": "Motorola",
    "nokia": "Nokia",
    "oneplus": "OnePlus",
    "samsung": "Samsung",
    "sony": "Sony",
    "xiaomi": "Xiaomi",
    "zte": "ZTE",
}

# Brand plus up to three model tokens. Long enough for "Samsung Galaxy Note 5"
# and "Xiaomi Redmi Note 7"; short enough to drop trailing spec prose.
MAX_TOKENS = 4


def _strip_skus(text: str) -> str:
    """Drop SKU codes while keeping tokens that are themselves model names."""
    kept = []
    for token in text.split():
        # "SM-J500H/DS" is one SKU with a variant suffix. Without splitting on
        # "/" the full-match fails and the token survives as a bare "SM".
        bare = token.split("/")[0].strip("-+")
        if SKU.fullmatch(bare) and not MODEL_TOKEN.match(bare):
            continue
        kept.append(token)
    return " ".join(kept)


def canonical_model(title: str, brand: str | None = None) -> str:
    """Reduce a listing title to a canonical model name.

    >>> canonical_model("Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon)", "Samsung")
    'Samsung Galaxy Note 5'
    >>> canonical_model("Samsung Galaxy S8+ 64GB GSM Unlocked", "Samsung")
    'Samsung Galaxy S8 Plus'
    >>> canonical_model("Nokia 3.1 - Android 9.0 Pie - 16 GB - Dual SIM", "Nokia")
    'Nokia 3.1'
    """
    text = fix_encoding(str(title or ""))
    text = OS_VERSION.sub(" ", text)

    head = CUT.split(text)[0]
    head = NOISE.sub(" ", head)
    head = _strip_skus(head)

    # "S8+" and "S8 Plus" are the same phone.
    head = head.replace("+", " Plus ")
    # Keep "." so Nokia 3.1 survives; drop the rest of the punctuation.
    head = re.sub(r"[^\w\s.]", " ", head)
    head = re.sub(r"\.(?!\d)", " ", head)
    # Motorola is listed both as "Motorola G6" and "Motorola Moto G6".
    head = re.sub(r"\bMotorola\s+Moto\b", "Motorola", head, flags=re.I)
    head = re.sub(r"\s+", " ", head).strip()

    tokens = head.split()
    brand_name = BRAND_CANONICAL.get(str(brand or "").lower().strip(), str(brand or "").strip())
    if brand_name:
        lowered = [token.lower() for token in tokens]
        if brand_name.lower() in lowered:
            tokens = tokens[lowered.index(brand_name.lower()) :]
            tokens[0] = brand_name
        else:
            tokens = [brand_name, *tokens]

    return " ".join(tokens[:MAX_TOKENS])


def model_key(name: str) -> str:
    """Case-insensitive grouping key.

    "Samsung Galaxy S7 Edge" and "Samsung Galaxy S7 EDGE" are one phone.
    """
    return " ".join(str(name).lower().split())


def is_usable(name: str, brand: str | None = None) -> bool:
    """Whether a canonical name identifies a specific phone.

    A name that is only the brand means normalisation failed -- the model was
    stripped along with the noise. Those listings are excluded rather than
    grouped, because grouping them pools unrelated handsets under one product.

    >>> is_usable("Samsung Galaxy S7", "Samsung")
    True
    >>> is_usable("Nokia", "Nokia")
    False
    """
    tokens = str(name).split()
    if len(tokens) < 2:
        return False
    return not (brand and model_key(name) == model_key(brand))
