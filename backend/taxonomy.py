"""Taxonomy primitives for RetailGraph product normalization.

This module is intentionally deterministic and dependency-free. It defines the
controlled vocabulary that later pipeline stages can rely on when converting
messy retailer observations into canonical product representations.

The taxonomy is not a full normalization pipeline. It provides shared constants,
alias maps, category-specific attribute helpers, validation helpers, and simple
confidence scoring that `normalize.py` can import later.
"""

from __future__ import annotations

import re
import string
from typing import Any


# ---------------------------------------------------------------------------
# Category and product type vocabulary
# ---------------------------------------------------------------------------

CATEGORY_FAMILIES = [
    "beverages",
    "dairy_milk",
    "plant_based_milk",
    "yogurt",
    "eggs",
    "cheese",
    "butter",
    "produce",
    "meat_seafood",
    "frozen",
    "pantry",
    "snacks",
    "bakery",
    "household",
    "unknown",
]


PRODUCT_TYPES_BY_FAMILY = {
    "beverages": [
        "soda",
        "sparkling_water",
        "orange_juice",
        "coffee",
    ],
    "dairy_milk": ["milk"],
    "plant_based_milk": [
        "almond_milk",
        "oat_milk",
        "soy_milk",
    ],
    "yogurt": [
        "yogurt",
        "greek_yogurt",
        "skyr",
    ],
    "eggs": ["eggs"],
    "cheese": ["cheese"],
    "butter": ["butter"],
    "produce": [
        "apples",
        "bananas",
        "berries",
        "avocados",
        "lettuce",
        "spinach",
        "carrots",
    ],
    "meat_seafood": [
        "chicken_breast",
        "ground_beef",
        "salmon",
    ],
    "frozen": [
        "frozen_pizza",
        "ice_cream",
    ],
    "pantry": [
        "cereal",
        "pasta",
        "rice",
        "peanut_butter",
        "canned_beans",
        "soup",
        "condiments",
        "olive_oil",
        "honey",
    ],
    "snacks": [
        "chips",
        "cookies",
        "crackers",
        "granola_bars",
    ],
    "bakery": ["bread"],
    "household": ["paper_towels", "toilet_paper", "laundry_detergent"],
}


PRODUCT_TYPE_TO_FAMILY = {
    product_type: family
    for family, product_types in PRODUCT_TYPES_BY_FAMILY.items()
    for product_type in product_types
}


# Broad retailer categories are deliberately mapped to "unknown" when they are
# too ambiguous. Product name inference can resolve the family later.
RAW_CATEGORY_ALIASES = {
    "drinks": "beverages",
    "beverages": "beverages",
    "soft drinks": "beverages",
    "coffee tea": "beverages",
    "coffee and tea": "beverages",
    "coffee": "beverages",
    "dairy": "unknown",
    "dairy eggs": "unknown",
    "dairy and eggs": "unknown",
    "milk cream": "dairy_milk",
    "milk and cream": "dairy_milk",
    "milk": "dairy_milk",
    "yogurt": "yogurt",
    "cheese": "cheese",
    "butter": "butter",
    "eggs": "eggs",
    "plant based": "plant_based_milk",
    "plant based milk": "plant_based_milk",
    "fresh produce": "produce",
    "produce": "produce",
    "meat": "meat_seafood",
    "meat seafood": "meat_seafood",
    "meat and seafood": "meat_seafood",
    "seafood": "meat_seafood",
    "frozen": "frozen",
    "frozen foods": "frozen",
    "pantry": "pantry",
    "canned packaged": "pantry",
    "canned and packaged": "pantry",
    "snacks": "snacks",
    "bakery": "bakery",
    "breakfast": "pantry",
    "breakfast cereal": "pantry",
    "breakfast and cereal": "pantry",
    "household": "household",
}


# Keep this map at the vocabulary layer: semantic synonyms and a minimal set of
# legacy aliases that preserve current normalization behavior. OCR corruption
# should eventually move to a separate text-cleaning layer instead of growing
# here indefinitely.
PRODUCT_TYPE_ALIASES = {
    # TODO: Move OCR/noisy-text repairs such as "c0ca c0la" into a dedicated
    # text_cleaning.py layer once the pipeline has that boundary.
    "c0ca c0la": "soda",
    "coke zero": "soda",
    "coca cola": "soda",
    "coca-cola": "soda",
    "diet coke": "soda",
    "sprite": "soda",
    "pepsi": "soda",
    "sparkling water": "sparkling_water",
    "seltzer": "sparkling_water",
    "lac roix": "sparkling_water",
    "lacroix": "sparkling_water",
    "orange juice": "orange_juice",
    "oj": "orange_juice",
    "coffee": "coffee",
    "whole milk": "milk",
    "2% milk": "milk",
    "2 percent milk": "milk",
    "reduced fat milk": "milk",
    "skim milk": "milk",
    "lactose free milk": "milk",
    "milk": "milk",
    "almondmilk": "almond_milk",
    "almond milk": "almond_milk",
    "almond beverage": "almond_milk",
    "oatmilk": "oat_milk",
    "oat milk": "oat_milk",
    "soymilk": "soy_milk",
    "soy milk": "soy_milk",
    "greek yogurt": "greek_yogurt",
    "yogurt": "yogurt",
    "skyr": "skyr",
    "eggs": "eggs",
    "egg": "eggs",
    "fuji apple": "apples",
    "gala apple": "apples",
    "apple": "apples",
    "apples": "apples",
    "banana": "bananas",
    "bananas": "bananas",
    "strawberries": "berries",
    "blueberries": "berries",
    "raspberries": "berries",
    "blackberries": "berries",
    "avocado": "avocados",
    "avocados": "avocados",
    "lettuce": "lettuce",
    "romaine": "lettuce",
    "spinach": "spinach",
    "carrot": "carrots",
    "carrots": "carrots",
    "chicken breast": "chicken_breast",
    "ground beef": "ground_beef",
    "salmon": "salmon",
    "frozen pizza": "frozen_pizza",
    "pizza": "frozen_pizza",
    "ice cream": "ice_cream",
    "cheerios": "cereal",
    "cereal": "cereal",
    "oats": "cereal",
    "bread": "bread",
    "spaghetti": "pasta",
    "pasta": "pasta",
    "rice": "rice",
    "peanut butter": "peanut_butter",
    "black beans": "canned_beans",
    "beans": "canned_beans",
    "soup": "soup",
    "ketchup": "condiments",
    "mayonnaise": "condiments",
    "olive oil": "olive_oil",
    "honey": "honey",
    "chips": "chips",
    "cookies": "cookies",
    "crackers": "crackers",
    "granola bars": "granola_bars",
    "cheese": "cheese",
    "butter": "butter",
}


# ---------------------------------------------------------------------------
# Brand, retailer private-label, unit, and package vocabularies
# ---------------------------------------------------------------------------

BRAND_ALIASES = {
    "coke": "Coca-Cola",
    "coca cola": "Coca-Cola",
    "coca-cola": "Coca-Cola",
    "la croix": "LaCroix",
    "lac roix": "LaCroix",
    "lacroix": "LaCroix",
    "365 whole foods": "365",
    "365": "365",
    "simple truth": "Simple Truth",
    "kroger": "Kroger",
    "great value": "Great Value",
    "good gather": "Good & Gather",
    "good and gather": "Good & Gather",
    "good & gather": "Good & Gather",
    "market pantry": "Market Pantry",
    "chobani": "Chobani",
    "silk": "Silk",
    "oatly": "Oatly",
    "horizon organic": "Horizon Organic",
    "fairlife": "Fairlife",
    "cheerios": "Cheerios",
    "general mills": "General Mills",
    "fage": "FAGE",
    "oreo": "Oreo",
    "tostitos": "Tostitos",
    "doritos": "Doritos",
    "lays": "Lay's",
    "lay's": "Lay's",
    "lay s": "Lay's",
    "ritz": "Ritz",
    "cheez it": "Cheez-It",
    "cheez-it": "Cheez-It",
    "jif": "Jif",
    "skippy": "Skippy",
    "heinz": "Heinz",
    "hellmanns": "Hellmann's",
    "hellmann's": "Hellmann's",
    "hellmann s": "Hellmann's",
    "kraft": "Kraft",
    "barilla": "Barilla",
    "rao's": "Rao's",
    "raos": "Rao's",
    "rao s": "Rao's",
    "digiorno": "DiGiorno",
}


PRIVATE_LABEL_BRANDS_BY_RETAILER = {
    "whole_foods": {"365"},
    "kroger": {"Simple Truth", "Kroger"},
    "walmart": {"Great Value"},
    "target": {"Good & Gather", "Market Pantry"},
}


UNIT_ALIASES = {
    "fl oz": "fl_oz",
    "floz": "fl_oz",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "gal": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "liter": "liter",
    "litre": "liter",
    "l": "liter",
    "ct": "count",
    "count": "count",
    "each": "count",
    "dozen": "dozen",
}


UNIT_TO_QUANTITY_TYPE = {
    "fl_oz": "volume",
    "gal": "volume",
    "liter": "volume",
    "oz": "weight",
    "lb": "weight",
    "count": "count",
    "dozen": "count",
}


CANONICAL_UNIT_BY_QUANTITY_TYPE = {
    "volume": "fl_oz",
    "weight": "oz",
    "count": "count",
}


UNIT_TO_CANONICAL_MULTIPLIER = {
    "fl_oz": 1.0,
    "gal": 128.0,
    "liter": 33.814,
    "oz": 1.0,
    "lb": 16.0,
    "count": 1.0,
    "dozen": 12.0,
}


PACKAGE_TYPE_ALIASES = {
    "can": "can",
    "cans": "can",
    "bottle": "bottle",
    "bottles": "bottle",
    "carton": "carton_or_jug",
    "jug": "carton_or_jug",
    "bag": "bag",
    "bg": "bag",
    "pouch": "bag",
    "box": "box",
    "jar": "jar",
    "tub": "tub",
    "cup": "cup",
    "cups": "cup",
    "clamshell": "clamshell",
    "head": "head",
    "bunch": "bunch",
    "loaf": "loaf",
    "stick": "stick",
    "sticks": "stick",
    "each": "each",
    "bulk": "bulk",
}


# ---------------------------------------------------------------------------
# Category-specific controlled attributes and validation schema
# ---------------------------------------------------------------------------

ATTRIBUTE_VOCABULARIES = {
    "beverages": {
        "flavor": ["zero_sugar", "classic", "diet", "lime", "grapefruit", "lemon_lime", "orange"],
        "diet_type": ["regular", "diet", "zero_sugar", "unsweetened"],
        "carbonated": [True, False, "unknown"],
        "caffeine": ["caffeinated", "caffeine_free", "unknown"],
    },
    "dairy_milk": {
        "fat_content": ["whole", "two_percent", "one_percent", "skim", "unknown"],
        "lactose_free": [True, False, "unknown"],
        "organic": [True, False],
    },
    "plant_based_milk": {
        "base": ["almond", "oat", "soy", "coconut", "unknown"],
        "sweetness": ["unsweetened", "original", "vanilla", "sweetened", "unknown"],
        "barista_style": [True, False],
    },
    "yogurt": {
        "style": ["greek", "regular", "skyr", "unknown"],
        "flavor": ["blueberry", "strawberry", "plain", "vanilla", "unknown"],
        "fat_content": ["whole", "nonfat", "lowfat", "unknown"],
    },
    "produce": {
        "produce_form": ["loose", "bagged", "clamshell", "bunch", "head", "packaged", "unknown"],
        "variety": ["fuji", "gala", "hass", "romaine", "baby_spinach", "unknown"],
        "unit_basis": ["per_lb", "each", "package", "unknown"],
    },
    "meat_seafood": {
        "animal": ["chicken", "beef", "salmon", "unknown"],
        "cut": ["breast", "ground", "fillet", "thigh", "unknown"],
        "bone_status": ["boneless", "bone_in", "unknown"],
        "skin_status": ["skinless", "skin_on", "unknown"],
        "fresh_or_frozen": ["fresh", "frozen", "unknown"],
    },
    "eggs": {
        "egg_size": ["large", "extra_large", "unknown"],
        "color": ["brown", "white", "unknown"],
        "cage_claim": ["cage_free", "pasture_raised", "unknown"],
    },
    "pantry_snacks": {
        "flavor": ["original", "classic", "nacho_cheese", "honey_nut", "plain", "unknown"],
        "package_type": ["box", "bag", "jar", "can", "bottle", "unknown"],
    },
}


CATEGORY_SCHEMA = {
    "beverages": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["flavor", "diet_type", "carbonated", "caffeine", "package_type"],
    },
    "dairy_milk": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type", "fat_content"],
        "optional": ["organic", "lactose_free", "package_type"],
    },
    "plant_based_milk": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type", "base"],
        "optional": ["sweetness", "barista_style", "package_type"],
    },
    "yogurt": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type", "style"],
        "optional": ["flavor", "fat_content", "organic", "package_type", "multi_pack"],
    },
    "eggs": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["egg_size", "color", "cage_claim", "organic", "package_type"],
    },
    "cheese": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["package_type", "organic"],
    },
    "butter": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["package_type", "organic"],
    },
    "produce": {
        "required": ["product_type", "produce_form", "unit_basis"],
        "optional": ["variety", "organic", "total_quantity"],
    },
    "meat_seafood": {
        "required": ["product_type", "total_quantity", "quantity_type", "animal", "cut"],
        "optional": ["bone_status", "skin_status", "fresh_or_frozen", "organic", "package_type"],
    },
    "frozen": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["flavor", "package_type"],
    },
    "pantry": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["flavor", "package_type", "organic"],
    },
    "snacks": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["flavor", "package_type"],
    },
    "bakery": {
        "required": ["canonical_brand", "product_type"],
        "optional": ["total_quantity", "quantity_type", "package_type", "organic"],
    },
    "household": {
        "required": ["canonical_brand", "product_type", "total_quantity", "quantity_type"],
        "optional": ["package_type"],
    },
    "unknown": {
        "required": ["product_type"],
        "optional": ["canonical_brand", "total_quantity", "quantity_type", "package_type"],
    },
}

# CATEGORY_SCHEMA is for extraction and validation:
# "What fields should normalization try to extract for this family?"
#
# CATEGORY_SEMANTIC_SCHEMAS is for comparison and matching semantics:
# "Once fields are extracted, which fields matter for equivalence or
# substitution decisions?"
CATEGORY_SEMANTIC_SCHEMAS = {
    "beverages": {
        "identity_critical": ["product_type", "flavor", "diet_type", "quantity_type", "total_quantity", "package_type", "pack_count"],
        "substitute_critical": ["product_type", "flavor", "diet_type", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "flavor", "diet_type", "quantity_type", "package_type", "pack_count"],
        "substitute_hard_conflicts": ["product_type", "diet_type"],
        "substitute_soft_conflicts": ["flavor", "package_type", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "dairy_milk": {
        "identity_critical": ["product_type", "fat_content", "lactose_free", "organic", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "fat_content", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "fat_content", "lactose_free", "organic", "quantity_type"],
        "substitute_hard_conflicts": ["product_type", "quantity_type"],
        "substitute_soft_conflicts": ["fat_content", "organic", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "plant_based_milk": {
        "identity_critical": ["product_type", "base", "sweetness", "barista_style", "quantity_type", "total_quantity"],
        "substitute_critical": ["base", "sweetness", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "base", "sweetness", "barista_style", "quantity_type"],
        "substitute_hard_conflicts": ["base", "quantity_type"],
        "substitute_soft_conflicts": ["sweetness", "barista_style", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "yogurt": {
        "identity_critical": ["product_type", "style", "flavor", "fat_content", "quantity_type", "total_quantity", "multi_pack"],
        "substitute_critical": ["style", "flavor", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "style", "flavor", "quantity_type", "multi_pack"],
        "substitute_hard_conflicts": ["style", "quantity_type"],
        "substitute_soft_conflicts": ["flavor", "fat_content", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "produce": {
        "identity_critical": ["product_type", "variety", "produce_form", "unit_basis", "organic"],
        "substitute_critical": ["product_type", "produce_form", "unit_basis"],
        "equivalence_hard_conflicts": ["product_type", "variety", "produce_form", "unit_basis"],
        "substitute_hard_conflicts": ["product_type", "unit_basis"],
        "substitute_soft_conflicts": ["variety", "produce_form", "organic"],
        "allow_private_label_equivalence": False,
    },
    "meat_seafood": {
        "identity_critical": ["product_type", "animal", "cut", "bone_status", "skin_status", "fresh_or_frozen", "quantity_type"],
        "substitute_critical": ["animal", "cut", "fresh_or_frozen", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "animal", "cut", "quantity_type"],
        "substitute_hard_conflicts": ["animal", "quantity_type"],
        "substitute_soft_conflicts": ["cut", "bone_status", "skin_status", "fresh_or_frozen"],
        "allow_private_label_equivalence": False,
    },
    "eggs": {
        "identity_critical": ["product_type", "egg_size", "color", "cage_claim", "organic", "total_quantity"],
        "substitute_critical": ["product_type", "egg_size", "total_quantity"],
        "equivalence_hard_conflicts": ["product_type", "egg_size", "total_quantity"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["color", "cage_claim", "organic"],
        "allow_private_label_equivalence": True,
    },
    "snacks": {
        "identity_critical": ["product_type", "flavor", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "flavor", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "flavor", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["flavor", "package_type", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "pantry": {
        "identity_critical": ["product_type", "flavor", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["flavor", "package_type", "total_quantity", "organic"],
        "allow_private_label_equivalence": True,
    },
    "frozen": {
        "identity_critical": ["product_type", "flavor", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "flavor", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "flavor", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["flavor", "package_type", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "bakery": {
        "identity_critical": ["product_type", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type"],
        "equivalence_hard_conflicts": ["product_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["package_type", "total_quantity", "organic"],
        "allow_private_label_equivalence": True,
    },
    "cheese": {
        "identity_critical": ["product_type", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["package_type", "total_quantity", "organic"],
        "allow_private_label_equivalence": True,
    },
    "butter": {
        "identity_critical": ["product_type", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["package_type", "total_quantity", "organic"],
        "allow_private_label_equivalence": True,
    },
    "household": {
        "identity_critical": ["product_type", "package_type", "quantity_type", "total_quantity"],
        "substitute_critical": ["product_type", "quantity_type"],
        "equivalence_hard_conflicts": ["product_type", "quantity_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": ["package_type", "total_quantity"],
        "allow_private_label_equivalence": True,
    },
    "unknown": {
        "identity_critical": ["product_type"],
        "substitute_critical": ["product_type"],
        "equivalence_hard_conflicts": ["product_type"],
        "substitute_hard_conflicts": ["product_type"],
        "substitute_soft_conflicts": [],
        "allow_private_label_equivalence": False,
    },
}


PACKAGING_MATTERS_FAMILIES = {
    "beverages",
    "dairy_milk",
    "plant_based_milk",
    "yogurt",
    "eggs",
    "frozen",
    "pantry",
    "snacks",
    "household",
}


# ---------------------------------------------------------------------------
# Generic normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """Lowercase text, replace punctuation with spaces, and collapse whitespace.

    Decimal points between digits are preserved because sizes such as "1.5 L"
    should remain parseable by later normalization code.
    """
    if text is None:
        return ""

    lowered = str(text).lower()
    chars = []
    for index, char in enumerate(lowered):
        if char == ".":
            previous_is_digit = index > 0 and lowered[index - 1].isdigit()
            next_is_digit = index + 1 < len(lowered) and lowered[index + 1].isdigit()
            chars.append("." if previous_is_digit and next_is_digit else " ")
        elif char in string.punctuation:
            chars.append(" ")
        else:
            chars.append(char)

    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _contains_phrase(cleaned_text: str, phrase: str) -> bool:
    cleaned_phrase = normalize_text(phrase)
    if not cleaned_phrase:
        return False
    return re.search(rf"(^|\s){re.escape(cleaned_phrase)}($|\s)", cleaned_text) is not None


def canonicalize_brand(brand: str | None) -> str:
    """Return a canonical brand name, or a readable title-cased fallback."""
    cleaned = normalize_text(brand)
    if not cleaned:
        return "unknown"
    return BRAND_ALIASES.get(cleaned, cleaned.title())


def normalize_raw_category(raw_category: str | None) -> str:
    cleaned = normalize_text(raw_category)
    return RAW_CATEGORY_ALIASES.get(cleaned, "unknown")


def infer_product_type(text: str | None) -> str:
    """Infer product type from text using longest alias matches first."""
    cleaned = normalize_text(text)
    for alias, product_type in sorted(PRODUCT_TYPE_ALIASES.items(), key=lambda item: len(normalize_text(item[0])), reverse=True):
        if _contains_phrase(cleaned, alias):
            return product_type
    return "unknown"


def get_family_for_product_type(product_type: str | None) -> str:
    return PRODUCT_TYPE_TO_FAMILY.get(product_type or "", "unknown")


def is_private_label(retailer: str | None, brand: str | None) -> bool:
    retailer_key = normalize_text(retailer).replace(" ", "_")
    canonical_brand = canonicalize_brand(brand)
    return canonical_brand in PRIVATE_LABEL_BRANDS_BY_RETAILER.get(retailer_key, set())


def normalize_unit(unit: str | None) -> str:
    cleaned = normalize_text(unit)
    return UNIT_ALIASES.get(cleaned, "unknown")


def convert_to_canonical_quantity(amount: float | int | str | None, unit: str | None, pack_count: int | str | None = 1) -> dict[str, Any]:
    """Convert a size into the canonical unit for its quantity type.

    The returned quantity is per sellable item (`unit_quantity`) and multiplied
    by pack count (`total_quantity`). Unknown or invalid inputs are represented
    explicitly rather than raising, so later stages can score confidence.
    """
    normalized_unit = normalize_unit(unit)
    quantity_type = UNIT_TO_QUANTITY_TYPE.get(normalized_unit, "unknown")
    canonical_unit = CANONICAL_UNIT_BY_QUANTITY_TYPE.get(quantity_type)

    try:
        size_value = float(amount) if amount is not None and amount != "" else None
    except (TypeError, ValueError):
        size_value = None

    try:
        normalized_pack_count = int(pack_count) if pack_count not in (None, "") else 1
    except (TypeError, ValueError):
        normalized_pack_count = 1

    if normalized_pack_count < 1:
        normalized_pack_count = 1

    multiplier = UNIT_TO_CANONICAL_MULTIPLIER.get(normalized_unit)
    unit_quantity = size_value * multiplier if size_value is not None and multiplier is not None else None
    total_quantity = unit_quantity * normalized_pack_count if unit_quantity is not None else None

    return {
        "size_value": size_value,
        "size_unit": normalized_unit,
        "pack_count": normalized_pack_count,
        "unit_quantity": unit_quantity,
        "total_quantity": total_quantity,
        "quantity_type": quantity_type,
        "canonical_unit": canonical_unit,
    }


def infer_package_type(text: str | None) -> str:
    cleaned = normalize_text(text)
    for alias, package_type in sorted(PACKAGE_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if _contains_phrase(cleaned, alias):
            return package_type
    return "unknown"


def get_required_fields(category_family: str | None) -> list[str]:
    return list(CATEGORY_SCHEMA.get(category_family or "unknown", CATEGORY_SCHEMA["unknown"])["required"])


def get_optional_fields(category_family: str | None) -> list[str]:
    return list(CATEGORY_SCHEMA.get(category_family or "unknown", CATEGORY_SCHEMA["unknown"])["optional"])


def get_semantic_schema(category_family: str | None) -> dict[str, Any]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in schema.items()
    }


def get_identity_critical_fields(category_family: str | None) -> list[str]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return list(schema["identity_critical"])


def get_substitute_critical_fields(category_family: str | None) -> list[str]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return list(schema["substitute_critical"])


def get_equivalence_hard_conflict_fields(category_family: str | None) -> list[str]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return list(schema["equivalence_hard_conflicts"])


def get_substitute_hard_conflict_fields(category_family: str | None) -> list[str]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return list(schema["substitute_hard_conflicts"])


def get_substitute_soft_conflict_fields(category_family: str | None) -> list[str]:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return list(schema["substitute_soft_conflicts"])


def allows_private_label_equivalence(category_family: str | None) -> bool:
    schema = CATEGORY_SEMANTIC_SCHEMAS.get(category_family or "unknown", CATEGORY_SEMANTIC_SCHEMAS["unknown"])
    return bool(schema["allow_private_label_equivalence"])


def validate_required_fields(category_family: str | None, extracted: dict[str, Any]) -> list[str]:
    missing = []
    for field in get_required_fields(category_family):
        value = extracted.get(field)
        if value is None or value == "" or value == "unknown":
            missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Category-specific attribute extraction helpers
# ---------------------------------------------------------------------------

def extract_beverage_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    attrs: dict[str, Any] = {
        "flavor": "classic",
        "diet_type": "regular",
        "carbonated": "unknown",
        "caffeine": "unknown",
        "package_type": infer_package_type(cleaned),
    }

    if any(_contains_phrase(cleaned, phrase) for phrase in ["zero sugar", "zer0 sugar", "coke zero"]):
        attrs["flavor"] = "zero_sugar"
        attrs["diet_type"] = "zero_sugar"
    elif _contains_phrase(cleaned, "diet"):
        attrs["flavor"] = "diet"
        attrs["diet_type"] = "diet"
    elif _contains_phrase(cleaned, "unsweetened"):
        attrs["diet_type"] = "unsweetened"

    if _contains_phrase(cleaned, "lemon lime") or _contains_phrase(cleaned, "sprite"):
        attrs["flavor"] = "lemon_lime"
    elif _contains_phrase(cleaned, "grapefruit"):
        attrs["flavor"] = "grapefruit"
    elif _contains_phrase(cleaned, "lime"):
        attrs["flavor"] = "lime"
    elif _contains_phrase(cleaned, "orange"):
        attrs["flavor"] = "orange"

    if any(_contains_phrase(cleaned, phrase) for phrase in ["soda", "cola", "c0la", "sparkling water", "seltzer", "sprite", "pepsi"]):
        attrs["carbonated"] = True
    elif any(_contains_phrase(cleaned, phrase) for phrase in ["juice", "coffee"]):
        attrs["carbonated"] = False

    if any(_contains_phrase(cleaned, phrase) for phrase in ["caffeine free", "decaf"]):
        attrs["caffeine"] = "caffeine_free"
    elif any(_contains_phrase(cleaned, phrase) for phrase in ["cola", "c0la", "coffee", "coke", "pepsi"]):
        attrs["caffeine"] = "caffeinated"

    return attrs


def extract_dairy_milk_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    fat_content = "unknown"
    if _contains_phrase(cleaned, "whole milk") or _contains_phrase(cleaned, "whole"):
        fat_content = "whole"
    elif any(phrase in cleaned for phrase in ["2% milk", "2 percent milk", "reduced fat"]):
        fat_content = "two_percent"
    elif any(phrase in cleaned for phrase in ["1% milk", "1 percent milk"]):
        fat_content = "one_percent"
    elif _contains_phrase(cleaned, "skim"):
        fat_content = "skim"

    return {
        "fat_content": fat_content,
        "lactose_free": True if "lactose free" in cleaned else False,
        "organic": _contains_phrase(cleaned, "organic"),
        "package_type": infer_package_type(cleaned),
    }


def extract_plant_based_milk_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    base = "unknown"
    base_aliases = {
        "almond": "almond",
        "almondmilk": "almond",
        "oat": "oat",
        "oatmilk": "oat",
        "soy": "soy",
        "soymilk": "soy",
        "coconut": "coconut",
    }
    for alias, candidate in base_aliases.items():
        if _contains_phrase(cleaned, alias):
            base = candidate
            break

    sweetness = "unknown"
    if _contains_phrase(cleaned, "unsweetened"):
        sweetness = "unsweetened"
    elif _contains_phrase(cleaned, "vanilla"):
        sweetness = "vanilla"
    elif _contains_phrase(cleaned, "original"):
        sweetness = "original"
    elif _contains_phrase(cleaned, "sweetened"):
        sweetness = "sweetened"

    return {
        "base": base,
        "sweetness": sweetness,
        "barista_style": _contains_phrase(cleaned, "barista"),
        "package_type": infer_package_type(cleaned),
    }


def extract_yogurt_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    style = "regular"
    if _contains_phrase(cleaned, "greek"):
        style = "greek"
    elif _contains_phrase(cleaned, "skyr"):
        style = "skyr"

    flavor = "unknown"
    for candidate in ["blueberry", "strawberry", "plain", "vanilla"]:
        if _contains_phrase(cleaned, candidate):
            flavor = candidate
            break

    fat_content = "unknown"
    if _contains_phrase(cleaned, "whole"):
        fat_content = "whole"
    elif _contains_phrase(cleaned, "nonfat") or _contains_phrase(cleaned, "fat free"):
        fat_content = "nonfat"
    elif _contains_phrase(cleaned, "lowfat") or _contains_phrase(cleaned, "low fat"):
        fat_content = "lowfat"

    return {
        "style": style,
        "flavor": flavor,
        "fat_content": fat_content,
        "organic": _contains_phrase(cleaned, "organic"),
        "package_type": infer_package_type(cleaned),
        "multi_pack": bool(re.search(r"\b\d+\s*(pk|pack|ct|count)\b", cleaned)),
    }


def extract_produce_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    package_type = infer_package_type(cleaned)

    if package_type == "bag":
        produce_form = "bagged"
        unit_basis = "package"
    elif package_type == "clamshell":
        produce_form = "clamshell"
        unit_basis = "package"
    elif package_type == "bunch":
        produce_form = "bunch"
        unit_basis = "each"
    elif package_type == "head":
        produce_form = "head"
        unit_basis = "each"
    elif package_type == "bulk" or _contains_phrase(cleaned, "bulk"):
        produce_form = "loose"
        unit_basis = "per_lb"
    elif any(_contains_phrase(cleaned, phrase) for phrase in ["each", "ea"]):
        produce_form = "loose"
        unit_basis = "each"
    else:
        produce_form = "unknown"
        unit_basis = "unknown"

    variety = "unknown"
    variety_aliases = {
        "fuji": "fuji",
        "gala": "gala",
        "hass": "hass",
        "romaine": "romaine",
        "baby spinach": "baby_spinach",
    }
    for alias, value in variety_aliases.items():
        if _contains_phrase(cleaned, alias):
            variety = value
            break

    return {
        "produce_form": produce_form,
        "variety": variety,
        "unit_basis": unit_basis,
        "organic": _contains_phrase(cleaned, "organic"),
        "package_type": package_type,
    }


def extract_meat_seafood_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    animal = "unknown"
    for candidate in ["chicken", "beef", "salmon"]:
        if _contains_phrase(cleaned, candidate):
            animal = candidate
            break

    cut = "unknown"
    for candidate in ["breast", "ground", "fillet", "thigh"]:
        if _contains_phrase(cleaned, candidate):
            cut = candidate
            break

    return {
        "animal": animal,
        "cut": cut,
        "bone_status": "boneless" if _contains_phrase(cleaned, "boneless") else "bone_in" if _contains_phrase(cleaned, "bone in") else "unknown",
        "skin_status": "skinless" if _contains_phrase(cleaned, "skinless") else "skin_on" if _contains_phrase(cleaned, "skin on") else "unknown",
        "fresh_or_frozen": "frozen" if _contains_phrase(cleaned, "frozen") else "fresh" if _contains_phrase(cleaned, "fresh") else "unknown",
        "organic": _contains_phrase(cleaned, "organic"),
        "package_type": infer_package_type(cleaned),
    }


def extract_egg_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    egg_size = "extra_large" if _contains_phrase(cleaned, "extra large") else "large" if _contains_phrase(cleaned, "large") else "unknown"
    color = "brown" if _contains_phrase(cleaned, "brown") else "white" if _contains_phrase(cleaned, "white") else "unknown"

    cage_claim = "unknown"
    if _contains_phrase(cleaned, "cage free"):
        cage_claim = "cage_free"
    elif _contains_phrase(cleaned, "pasture raised"):
        cage_claim = "pasture_raised"

    return {
        "egg_size": egg_size,
        "color": color,
        "cage_claim": cage_claim,
        "organic": _contains_phrase(cleaned, "organic"),
        "package_type": infer_package_type(cleaned),
    }


def extract_pantry_snack_attributes(text: str | None) -> dict[str, Any]:
    cleaned = normalize_text(text)
    flavor = "unknown"
    flavor_aliases = {
        "nacho cheese": "nacho_cheese",
        "honey nut": "honey_nut",
        "original": "original",
        "classic": "classic",
        "plain": "plain",
    }
    for alias, value in flavor_aliases.items():
        if _contains_phrase(cleaned, alias):
            flavor = value
            break

    return {
        "flavor": flavor,
        "package_type": infer_package_type(cleaned),
        "organic": _contains_phrase(cleaned, "organic"),
    }


def extract_category_attributes(category_family: str | None, text: str | None) -> dict[str, Any]:
    family = category_family or "unknown"
    if family == "beverages":
        return extract_beverage_attributes(text)
    if family == "dairy_milk":
        return extract_dairy_milk_attributes(text)
    if family == "plant_based_milk":
        return extract_plant_based_milk_attributes(text)
    if family == "yogurt":
        return extract_yogurt_attributes(text)
    if family == "produce":
        return extract_produce_attributes(text)
    if family == "meat_seafood":
        return extract_meat_seafood_attributes(text)
    if family == "eggs":
        return extract_egg_attributes(text)
    if family in {"pantry", "snacks", "frozen", "bakery", "cheese", "butter", "household"}:
        return extract_pantry_snack_attributes(text)
    return {}


# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------

def estimate_taxonomy_confidence(extracted: dict[str, Any]) -> dict[str, Any]:
    """Estimate confidence for taxonomy-driven extraction.

    This is a transparent heuristic score, not a probabilistic model. It gives
    downstream code a consistent way to route low-quality rows to review.
    """
    category_family = extracted.get("category_family", "unknown")
    required_fields = get_required_fields(category_family)
    missing_required = validate_required_fields(category_family, extracted)
    warnings: list[str] = []

    brand = extracted.get("canonical_brand")
    product_type = extracted.get("product_type")
    quantity_type = extracted.get("quantity_type")
    package_type = extracted.get("package_type")

    brand_score = 1.0
    if not brand or brand == "unknown":
        brand_score = 0.4
        warnings.append("canonical_brand is unknown")

    product_type_score = 1.0
    if not product_type or product_type == "unknown":
        product_type_score = 0.3
        warnings.append("product_type is unknown")

    quantity_required = "total_quantity" in required_fields or "quantity_type" in required_fields
    quantity_score = 1.0
    if quantity_required:
        if extracted.get("total_quantity") in (None, "", "unknown"):
            quantity_score -= 0.35
            warnings.append("total_quantity is missing")
        if not quantity_type or quantity_type == "unknown":
            quantity_score -= 0.35
            warnings.append("quantity_type is unknown")
        quantity_score = max(0.2, quantity_score)

    attributes_score = 1.0
    attribute_missing = [
        field
        for field in missing_required
        if field not in {"canonical_brand", "product_type", "total_quantity", "quantity_type"}
    ]
    if attribute_missing:
        attributes_score = max(0.25, 1.0 - 0.25 * len(attribute_missing))
        warnings.append(f"missing required attributes: {', '.join(attribute_missing)}")

    raw_category_family = extracted.get("raw_category_family")
    inferred_family = get_family_for_product_type(product_type)
    family_conflict = (
        raw_category_family
        and raw_category_family != "unknown"
        and inferred_family != "unknown"
        and raw_category_family != inferred_family
    )
    if family_conflict:
        warnings.append(f"raw category family {raw_category_family} conflicts with inferred family {inferred_family}")

    package_penalty = 0.0
    if category_family in PACKAGING_MATTERS_FAMILIES and (not package_type or package_type == "unknown"):
        package_penalty = 0.08
        warnings.append("package_type is unknown")

    overall = (brand_score + product_type_score + quantity_score + attributes_score) / 4
    if family_conflict:
        overall -= 0.15
    overall -= package_penalty
    overall = round(max(0.0, min(1.0, overall)), 3)

    needs_review = (
        overall < 0.75
        or product_type in (None, "", "unknown")
        or bool(missing_required)
        or (quantity_required and quantity_type == "unknown")
        or (category_family == "produce" and extracted.get("produce_form") == "unknown")
    )

    return {
        "overall": overall,
        "brand": round(brand_score, 3),
        "product_type": round(product_type_score, 3),
        "quantity": round(quantity_score, 3),
        "attributes": round(attributes_score, 3),
        "warnings": warnings,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Smoke test for manual verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    semantic_schema_example = get_semantic_schema("plant_based_milk")
    print("semantic_schema_example:", semantic_schema_example)
    assert "fat_content" in get_identity_critical_fields("dairy_milk")
    assert allows_private_label_equivalence("produce") is False
    assert "base" in get_substitute_hard_conflict_fields("plant_based_milk")

    test_cases = [
        {
            "retailer": "walmart",
            "brand": "Coke",
            "raw_name": "C0ca C0la Zer0 Sugar 12pk 12 fl oz Cans",
            "category_raw": "Soft Drinks",
            "amount": 12,
            "unit": "fl oz",
            "pack_count": 12,
        },
        {
            "retailer": "whole_foods",
            "brand": "365",
            "raw_name": "365 Organic Whole Milk 128 fl oz",
            "category_raw": "Dairy & Eggs",
            "amount": 128,
            "unit": "fl oz",
            "pack_count": 1,
        },
        {
            "retailer": "kroger",
            "brand": "Simple Truth",
            "raw_name": "Simple Truth Organic Fuji Apples 3 lb Bag",
            "category_raw": "Fresh Produce",
            "amount": 3,
            "unit": "lb",
            "pack_count": 1,
        },
        {
            "retailer": "target",
            "brand": "Good & Gather",
            "raw_name": "Good & Gather Unsweetened Almond Milk 64 fl oz",
            "category_raw": "Plant-Based",
            "amount": 64,
            "unit": "fl oz",
            "pack_count": 1,
        },
    ]

    for index, case in enumerate(test_cases, start=1):
        canonical_brand = canonicalize_brand(case["brand"])
        product_type = infer_product_type(case["raw_name"])
        inferred_family = get_family_for_product_type(product_type)
        raw_category_family = normalize_raw_category(case["category_raw"])
        category_family = inferred_family if inferred_family != "unknown" else raw_category_family
        quantity = convert_to_canonical_quantity(case["amount"], case["unit"], case["pack_count"])
        package_type = infer_package_type(case["raw_name"])
        attributes = extract_category_attributes(category_family, case["raw_name"])

        extracted = {
            "canonical_brand": canonical_brand,
            "product_type": product_type,
            "category_family": category_family,
            "raw_category_family": raw_category_family,
            "package_type": package_type,
            **quantity,
            **attributes,
        }

        print(f"\nTest case {index}")
        print(f"canonical_brand: {canonical_brand}")
        print(f"product_type: {product_type}")
        print(f"category_family: {category_family}")
        print(f"quantity_conversion: {quantity}")
        print(f"package_type: {package_type}")
        print(f"category_attributes: {attributes}")
        print(f"is_private_label: {is_private_label(case['retailer'], case['brand'])}")
        print(f"confidence: {estimate_taxonomy_confidence(extracted)}")
