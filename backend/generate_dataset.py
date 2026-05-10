"""
backend/generate_dataset.py
===========================

Generates a synthetic but realistic grocery dataset for testing
entity resolution across four retailers: whole_foods, kroger, walmart, target.

Run:
    python backend/generate_dataset.py

Outputs:
    data/raw/whole_foods.csv
    data/raw/kroger.csv
    data/raw/walmart.csv
    data/raw/target.csv
    data/labels/ground_truth_matches.json
    data/labels/negative_pairs.json
    data/labels/substitute_pairs.json
    data/README_DATASET.md

Design idea (for the learner):
    We start from "canonical product concepts" (the real-world products
    that exist in the world, like "Coca-Cola Zero Sugar 12-pack of 12oz cans").
    Then for each retailer we generate a *distorted* row representing how that
    retailer would advertise the same product on its website. Because we know
    which rows came from the same concept, we can also write the ground-truth
    labels in the same pass. This is the trick that makes the dataset useful
    for testing entity resolution: the answer key is built alongside the data.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Determinism — seed everything so reruns produce identical files.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
README_PATH = PROJECT_ROOT / "data" / "README_DATASET.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETAILERS = ["whole_foods", "kroger", "walmart", "target"]
RETAILER_PREFIX = {
    "whole_foods": "wf",
    "kroger": "kr",
    "walmart": "wm",
    "target": "tg",
}

# Each retailer's private-label brand(s). We pick one per concept.
PRIVATE_LABELS = {
    "whole_foods": ["365"],
    "kroger": ["Simple Truth", "Kroger"],
    "walmart": ["Great Value"],
    "target": ["Good & Gather", "Market Pantry"],
}

# Each retailer uses its own category names for the same broad area.
# This mimics the real schema differences between retailer websites.
CATEGORY_NAMES = {
    "Dairy milk": {
        "whole_foods": "Dairy & Eggs", "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Milk & Cream",
    },
    "Plant-based milk": {
        "whole_foods": "Plant-Based",  "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Plant-Based",
    },
    "Yogurt": {
        "whole_foods": "Dairy & Eggs", "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Dairy & Eggs",
    },
    "Eggs": {
        "whole_foods": "Dairy & Eggs", "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Dairy & Eggs",
    },
    "Cheese": {
        "whole_foods": "Dairy & Eggs", "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Dairy & Eggs",
    },
    "Butter": {
        "whole_foods": "Dairy & Eggs", "kroger": "Dairy",
        "walmart": "Dairy & Eggs",     "target": "Dairy & Eggs",
    },
    "Apples": {
        "whole_foods": "Produce", "kroger": "Fresh Produce",
        "walmart": "Produce",     "target": "Fresh Produce",
    },
    "Bananas": {
        "whole_foods": "Produce", "kroger": "Fresh Produce",
        "walmart": "Produce",     "target": "Fresh Produce",
    },
    "Berries": {
        "whole_foods": "Produce", "kroger": "Fresh Produce",
        "walmart": "Produce",     "target": "Fresh Produce",
    },
    "Avocados": {
        "whole_foods": "Produce", "kroger": "Fresh Produce",
        "walmart": "Produce",     "target": "Fresh Produce",
    },
    "Lettuce": {
        "whole_foods": "Produce", "kroger": "Fresh Produce",
        "walmart": "Produce",     "target": "Fresh Produce",
    },
    "Chicken": {
        "whole_foods": "Meat & Seafood", "kroger": "Meat & Seafood",
        "walmart": "Meat",               "target": "Meat & Seafood",
    },
    "Beef": {
        "whole_foods": "Meat & Seafood", "kroger": "Meat & Seafood",
        "walmart": "Meat",               "target": "Meat & Seafood",
    },
    "Seafood": {
        "whole_foods": "Meat & Seafood", "kroger": "Meat & Seafood",
        "walmart": "Seafood",            "target": "Meat & Seafood",
    },
    "Frozen pizza": {
        "whole_foods": "Frozen Foods", "kroger": "Frozen",
        "walmart": "Frozen Foods",     "target": "Frozen",
    },
    "Ice cream": {
        "whole_foods": "Frozen Foods", "kroger": "Frozen",
        "walmart": "Frozen Foods",     "target": "Frozen",
    },
    "Soda": {
        "whole_foods": "Beverages", "kroger": "Soft Drinks",
        "walmart": "Beverages",     "target": "Beverages",
    },
    "Sparkling water": {
        "whole_foods": "Beverages", "kroger": "Soft Drinks",
        "walmart": "Beverages",     "target": "Beverages",
    },
    "Coffee": {
        "whole_foods": "Beverages", "kroger": "Coffee & Tea",
        "walmart": "Beverages",     "target": "Coffee",
    },
    "Cereal": {
        "whole_foods": "Breakfast", "kroger": "Breakfast",
        "walmart": "Breakfast & Cereal", "target": "Breakfast",
    },
    "Bread": {
        "whole_foods": "Bakery", "kroger": "Bakery",
        "walmart": "Bakery",     "target": "Bakery",
    },
    "Pasta": {
        "whole_foods": "Pantry", "kroger": "Pantry",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Rice": {
        "whole_foods": "Pantry", "kroger": "Pantry",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Chips": {
        "whole_foods": "Snacks", "kroger": "Snacks",
        "walmart": "Snacks",     "target": "Snacks",
    },
    "Granola bars": {
        "whole_foods": "Snacks", "kroger": "Snacks",
        "walmart": "Snacks",     "target": "Snacks",
    },
    "Orange juice": {
        "whole_foods": "Beverages", "kroger": "Beverages",
        "walmart": "Beverages",     "target": "Beverages",
    },
    "Peanut butter": {
        "whole_foods": "Pantry", "kroger": "Pantry",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Canned beans": {
        "whole_foods": "Pantry", "kroger": "Canned & Packaged",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Soup": {
        "whole_foods": "Pantry", "kroger": "Canned & Packaged",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Condiments": {
        "whole_foods": "Pantry", "kroger": "Pantry",
        "walmart": "Pantry",     "target": "Pantry",
    },
    "Snacks": {
        "whole_foods": "Snacks", "kroger": "Snacks",
        "walmart": "Snacks",     "target": "Snacks",
    },
}

# Retailer pricing tendency: per retailer, multiplier ranges applied to a
# "concept-level" base price. Whole Foods leans premium, Walmart leans cheap.
PRICE_MULTIPLIER = {
    "whole_foods": (1.08, 1.30),
    "kroger":      (0.85, 1.05),
    "walmart":     (0.78, 1.00),
    "target":      (0.95, 1.18),
}

# Promo strings used across retailers (some retailers love promos more).
PROMO_POOL_BY_RETAILER = {
    "whole_foods": ["", "", "", "Save $1", "Prime member deal"],
    "kroger":      ["", "", "Digital coupon", "Club price", "2 for $5",
                    "Save $1.50", "Save $1"],
    "walmart":     ["", "", "Rollback", "Save $1", "Save $2", "Buy 2 save $1"],
    "target":      ["", "", "Circle deal", "Save 20%", "BOGO 50% off",
                    "Save $1"],
}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def make_url(retailer: str, pid: str) -> str:
    """Fake but realistic URL string."""
    return f"https://example.com/{retailer}/products/{pid}"


def make_timestamp(idx: int) -> str:
    """ISO timestamp; spread across the last 30 days so they look real."""
    base = datetime(2026, 5, 8, 12, 0, 0)
    delta = timedelta(minutes=(idx * 17) % (30 * 24 * 60))
    return (base - delta).isoformat(timespec="seconds")


def random_price(base_low: float, base_high: float, retailer: str) -> float:
    """Pick a realistic price in retailer-specific range."""
    base = random.uniform(base_low, base_high)
    mlow, mhigh = PRICE_MULTIPLIER[retailer]
    price = base * random.uniform(mlow, mhigh)
    return round(price, 2)


def estimated_cost_for(price: float) -> float:
    """
    Wholesale/acquisition cost. Usually 55-78% of retail. About 5% of rows
    will be edge cases where margin is razor-thin (loss-leader-like).
    """
    if random.random() < 0.05:
        # low-margin edge case: cost very close to (or even slightly above) price
        cost_ratio = random.uniform(0.92, 1.02)
    else:
        cost_ratio = random.uniform(0.55, 0.78)
    return round(price * cost_ratio, 2)


# OCR/noise: cheap character-substitution table so we can produce names
# that look like they came from a flaky OCR pipeline.
OCR_SUBS = [
    ("o", "0"),
    ("O", "0"),
    ("l", "I"),
    ("i", "1"),
    ("s", "$"),
]


def apply_ocr_noise(text: str, intensity: float = 0.18) -> str:
    """Randomly corrupt a small fraction of characters."""
    chars = list(text)
    for i, ch in enumerate(chars):
        for src, dst in OCR_SUBS:
            if ch == src and random.random() < intensity:
                chars[i] = dst
                break
    # Also drop a vowel here and there (e.g., "Whle Milk").
    if random.random() < 0.5:
        # remove a single random lowercase vowel mid-word
        vowels = [i for i, c in enumerate(chars)
                  if c in "aeiou" and 1 < i < len(chars) - 1]
        if vowels:
            del chars[random.choice(vowels)]
    return "".join(chars)


def maybe_inject_promo_into_name(name: str, promo: str) -> str:
    """Sometimes retailers put the promo right into the listing name."""
    if not promo:
        return name
    if random.random() < 0.5:
        return f"{promo} {name}"
    return name


# ---------------------------------------------------------------------------
# Concept catalog
# ---------------------------------------------------------------------------
# Each concept describes ONE real-world product. For each retailer that
# carries it, we provide:
#   - the raw product name as it would appear on that retailer's site
#   - the brand string the retailer would publish
#   - amount/unit/pack as the retailer would express them (e.g., 1 gal vs 128 fl oz)
#
# match_type:
#   - "exact_equivalent"          : same national brand product across retailers
#   - "private_label_equivalent"  : same product but each retailer's own brand
#
# base_price is a concept-level reference; we multiply per retailer.
# ---------------------------------------------------------------------------

# Helper to keep concept dicts compact.
def C(concept_id, category, match_type, base_price, retailers):
    """Build a concept dict. `retailers` is a dict[retailer] -> dict of fields."""
    return {
        "concept_id": concept_id,
        "category": category,
        "match_type": match_type,
        "base_price": base_price,
        "retailers": retailers,
    }


# Note on naming style:
#   whole_foods: tends to spell things out, uses "fl oz" for liquids
#   kroger:      spells "Gallon", uses commas, sometimes "Pack of N"
#   walmart:     short forms ("gal", "pk"), all-caps occasionally
#   target:      Title Case, "12 Pack 12oz Cans"

CONCEPTS = []

# === Private-label dairy & milk ============================================
CONCEPTS.append(C(
    "organic_whole_milk_1gal_pl", "Dairy milk", "private_label_equivalent", 6.49,
    {
        "whole_foods": dict(name="365 Organic Whole Milk, 128 fl oz",
                            brand="365", amount=128, unit="fl oz", pack=1,
                            organic=True, pl=True),
        "kroger":      dict(name="Simple Truth Organic Whole Milk 1 Gallon",
                            brand="Simple Truth", amount=1, unit="gal", pack=1,
                            organic=True, pl=True),
        "walmart":     dict(name="Great Value Organic Whole Milk 1 gal",
                            brand="Great Value", amount=1, unit="gal", pack=1,
                            organic=True, pl=True),
        "target":      dict(name="Good & Gather Organic Whole Milk 1 Gal",
                            brand="Good & Gather", amount=1, unit="gal", pack=1,
                            organic=True, pl=True),
    },
))

CONCEPTS.append(C(
    "whole_milk_1gal_pl", "Dairy milk", "private_label_equivalent", 4.49,
    {
        "whole_foods": dict(name="365 Whole Milk 1 Gallon",
                            brand="365", amount=1, unit="gal", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Whole Milk 1 Gallon",
                            brand="Kroger", amount=1, unit="gal", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Whole Milk 1 gal",
                            brand="Great Value", amount=1, unit="gal", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Whole Milk, 1 Gallon",
                            brand="Good & Gather", amount=1, unit="gal", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "two_percent_milk_half_gal_pl", "Dairy milk", "private_label_equivalent", 2.99,
    {
        "whole_foods": dict(name="365 2% Reduced Fat Milk 64 fl oz",
                            brand="365", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger 2% Reduced Fat Milk 1/2 Gallon",
                            brand="Kroger", amount=0.5, unit="gal", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value 2% Reduced Fat Milk 0.5 gal",
                            brand="Great Value", amount=0.5, unit="gal", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "unsweet_almond_milk_64floz_pl", "Plant-based milk", "private_label_equivalent", 3.49,
    {
        "whole_foods": dict(name="365 Unsweetened Almond Milk 64 fl oz",
                            brand="365", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Simple Truth Unsweetened Almondmilk Half Gallon",
                            brand="Simple Truth", amount=0.5, unit="gal", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Unsweetened Almond Milk, 64 fl oz",
                            brand="Great Value", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Unsweetened Almond Milk 64 fl oz",
                            brand="Good & Gather", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "oat_milk_64floz_pl", "Plant-based milk", "private_label_equivalent", 3.99,
    {
        "whole_foods": dict(name="365 Original Oat Milk 64 fl oz",
                            brand="365", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Simple Truth Oatmilk Original 64 fl oz",
                            brand="Simple Truth", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Oat Milk 64 fl oz",
                            brand="Good & Gather", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "large_brown_eggs_dozen_pl", "Eggs", "private_label_equivalent", 4.49,
    {
        "whole_foods": dict(name="365 Large Brown Eggs, Dozen",
                            brand="365", amount=12, unit="ct", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Grade A Large Brown Eggs, 12 ct",
                            brand="Kroger", amount=12, unit="ct", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Cage Free Large Brown Eggs 12 Count",
                            brand="Great Value", amount=12, unit="ct", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Cage-Free Large Brown Grade A Eggs - 1 Dozen",
                            brand="Good & Gather", amount=1, unit="dozen", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "organic_large_brown_eggs_dozen_pl", "Eggs", "private_label_equivalent", 6.99,
    {
        "whole_foods": dict(name="365 Organic Large Brown Eggs, Dozen",
                            brand="365", amount=12, unit="ct", pack=1,
                            organic=True, pl=True),
        "kroger":      dict(name="Simple Truth Organic Cage Free Large Brown Eggs 12 ct",
                            brand="Simple Truth", amount=12, unit="ct", pack=1,
                            organic=True, pl=True),
        "walmart":     dict(name="Great Value Organic Large Brown Eggs, 12 Count",
                            brand="Great Value", amount=12, unit="ct", pack=1,
                            organic=True, pl=True),
        "target":      dict(name="Good & Gather Organic Cage-Free Grade A Large Brown Eggs - 1 Dozen",
                            brand="Good & Gather", amount=1, unit="dozen", pack=1,
                            organic=True, pl=True),
    },
))

CONCEPTS.append(C(
    "white_bread_loaf_pl", "Bread", "private_label_equivalent", 2.49,
    {
        "whole_foods": dict(name="365 Classic White Sandwich Bread 20 oz",
                            brand="365", amount=20, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger White Sandwich Bread 20 oz Loaf",
                            brand="Kroger", amount=20, unit="oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value White Sandwich Bread, 20 oz",
                            brand="Great Value", amount=20, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Market Pantry Classic White Bread - 20oz",
                            brand="Market Pantry", amount=20, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "whole_wheat_bread_loaf_pl", "Bread", "private_label_equivalent", 2.79,
    {
        "kroger":  dict(name="Kroger 100% Whole Wheat Bread 20 oz",
                        brand="Kroger", amount=20, unit="oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value 100% Whole Wheat Bread 20 oz",
                        brand="Great Value", amount=20, unit="oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Market Pantry 100% Whole Wheat Bread 20 oz",
                        brand="Market Pantry", amount=20, unit="oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "creamy_pb_16oz_pl", "Peanut butter", "private_label_equivalent", 2.99,
    {
        "whole_foods": dict(name="365 Creamy Peanut Butter 16 oz",
                            brand="365", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Creamy Peanut Butter 16 oz Jar",
                            brand="Kroger", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Creamy Peanut Butter, 16 oz",
                            brand="Great Value", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Creamy Peanut Butter 16 oz",
                            brand="Good & Gather", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "tortilla_chips_13oz_pl", "Chips", "private_label_equivalent", 2.99,
    {
        "whole_foods": dict(name="365 Restaurant Style Tortilla Chips 13 oz",
                            brand="365", amount=13, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Restaurant Style Tortilla Chips, 13 oz",
                            brand="Kroger", amount=13, unit="oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Restaurant Style White Corn Tortilla Chips 13 oz",
                            brand="Great Value", amount=13, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Restaurant Style Tortilla Chips 13oz",
                            brand="Good & Gather", amount=13, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "frozen_pepperoni_pizza_pl", "Frozen pizza", "private_label_equivalent", 3.99,
    {
        "kroger":  dict(name="Kroger Pepperoni Frozen Pizza 12 inch",
                        brand="Kroger", amount=12, unit="oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Pepperoni Frozen Pizza, 12 in",
                        brand="Great Value", amount=12, unit="oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Market Pantry Pepperoni Frozen Pizza 12in",
                        brand="Market Pantry", amount=12, unit="oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "organic_gala_apples_3lb_pl", "Apples", "private_label_equivalent", 5.99,
    {
        "whole_foods": dict(name="365 Organic Gala Apples 3 lb Bag",
                            brand="365", amount=3, unit="lb", pack=1,
                            organic=True, pl=True),
        "kroger":      dict(name="Simple Truth Organic Gala Apples 3 lb Bag",
                            brand="Simple Truth", amount=3, unit="lb", pack=1,
                            organic=True, pl=True),
        "walmart":     dict(name="Great Value Organic Gala Apples, 3 lb Bag",
                            brand="Great Value", amount=48, unit="oz", pack=1,
                            organic=True, pl=True),
        "target":      dict(name="Good & Gather Organic Gala Apples - 3lb Bag",
                            brand="Good & Gather", amount=3, unit="lb", pack=1,
                            organic=True, pl=True),
    },
))

CONCEPTS.append(C(
    "sparkling_water_lime_8pk_pl", "Sparkling water", "private_label_equivalent", 3.49,
    {
        "whole_foods": dict(name="365 Sparkling Water Lime 8 pack 12 fl oz cans",
                            brand="365", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Lime Sparkling Water 8 ct, 12 fl oz cans",
                            brand="Kroger", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Lime Sparkling Water 8pk 12oz",
                            brand="Great Value", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Lime Sparkling Water 8 Pack 12 fl oz Cans",
                            brand="Good & Gather", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "organic_baby_spinach_5oz_pl", "Lettuce", "private_label_equivalent", 3.99,
    {
        "whole_foods": dict(name="365 Organic Baby Spinach 5 oz Clamshell",
                            brand="365", amount=5, unit="oz", pack=1,
                            organic=True, pl=True),
        "kroger":      dict(name="Simple Truth Organic Baby Spinach 5 oz",
                            brand="Simple Truth", amount=5, unit="oz", pack=1,
                            organic=True, pl=True),
        "target":      dict(name="Good & Gather Organic Baby Spinach 5oz",
                            brand="Good & Gather", amount=5, unit="oz", pack=1,
                            organic=True, pl=True),
    },
))

CONCEPTS.append(C(
    "plain_greek_yogurt_32oz_pl", "Yogurt", "private_label_equivalent", 4.49,
    {
        "whole_foods": dict(name="365 Plain Whole Milk Greek Yogurt 32 oz Tub",
                            brand="365", amount=32, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Plain Greek Yogurt 32 oz",
                            brand="Kroger", amount=32, unit="oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Plain Nonfat Greek Yogurt 32 oz",
                            brand="Great Value", amount=32, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Plain Whole Milk Greek Yogurt - 32oz",
                            brand="Good & Gather", amount=32, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "unsalted_butter_1lb_pl", "Butter", "private_label_equivalent", 4.99,
    {
        "whole_foods": dict(name="365 Unsalted Butter 4 sticks 16 oz",
                            brand="365", amount=16, unit="oz", pack=4,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Unsalted Sweet Cream Butter 1 lb (4 Sticks)",
                            brand="Kroger", amount=1, unit="lb", pack=4,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Unsalted Sweet Cream Butter, 16 oz",
                            brand="Great Value", amount=16, unit="oz", pack=4,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Unsalted Sweet Cream Butter Sticks - 1 lb",
                            brand="Good & Gather", amount=1, unit="lb", pack=4,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "shredded_cheddar_8oz_pl", "Cheese", "private_label_equivalent", 3.49,
    {
        "kroger":  dict(name="Kroger Mild Shredded Cheddar Cheese 8 oz",
                        brand="Kroger", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Finely Shredded Mild Cheddar Cheese, 8 oz",
                        brand="Great Value", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Good & Gather Mild Cheddar Shredded Cheese - 8oz",
                        brand="Good & Gather", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "black_beans_can_15oz_pl", "Canned beans", "private_label_equivalent", 1.19,
    {
        "whole_foods": dict(name="365 Organic Black Beans 15 oz Can",
                            brand="365", amount=15, unit="oz", pack=1,
                            organic=True, pl=True),
        "kroger":      dict(name="Kroger Black Beans 15 oz Can",
                            brand="Kroger", amount=15, unit="oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Black Beans, 15 oz",
                            brand="Great Value", amount=15, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Black Beans 15oz",
                            brand="Good & Gather", amount=15, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "tomato_soup_can_pl", "Soup", "private_label_equivalent", 1.49,
    {
        "kroger":  dict(name="Kroger Condensed Tomato Soup 10.75 oz",
                        brand="Kroger", amount=10.75, unit="oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Condensed Tomato Soup, 10.75 oz",
                        brand="Great Value", amount=10.75, unit="oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Market Pantry Condensed Tomato Soup 10.75 oz",
                        brand="Market Pantry", amount=10.75, unit="oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "long_grain_white_rice_5lb_pl", "Rice", "private_label_equivalent", 5.99,
    {
        "kroger":  dict(name="Kroger Long Grain White Rice 5 lb Bag",
                        brand="Kroger", amount=5, unit="lb", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Long Grain Enriched White Rice, 5 lb",
                        brand="Great Value", amount=5, unit="lb", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Good & Gather Long Grain White Rice - 5lb",
                        brand="Good & Gather", amount=5, unit="lb", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "spaghetti_pasta_16oz_pl", "Pasta", "private_label_equivalent", 1.49,
    {
        "whole_foods": dict(name="365 Spaghetti Pasta 16 oz",
                            brand="365", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Spaghetti Pasta 1 lb Box",
                            brand="Kroger", amount=1, unit="lb", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Spaghetti Pasta, 16 oz",
                            brand="Great Value", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Spaghetti - 16oz",
                            brand="Good & Gather", amount=16, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "oj_no_pulp_52floz_pl", "Orange juice", "private_label_equivalent", 3.99,
    {
        "kroger":  dict(name="Kroger 100% Orange Juice No Pulp 52 fl oz",
                        brand="Kroger", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value 100% Orange Juice No Pulp, 52 fl oz",
                        brand="Great Value", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Good & Gather 100% Orange Juice No Pulp 52 fl oz",
                        brand="Good & Gather", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "baby_carrots_1lb_pl", "Apples", "private_label_equivalent", 1.99,
    # category bucket is loose — using "Apples" produce category mapping is wrong;
    # we'll override below by using a Lettuce-style mapping. To keep this simple,
    # we re-use the "Lettuce" mapping which routes to "Produce" for all retailers.
    {
        "kroger":  dict(name="Kroger Baby Cut Carrots 1 lb Bag",
                        brand="Kroger", amount=1, unit="lb", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Baby-Cut Carrots, 1 lb",
                        brand="Great Value", amount=1, unit="lb", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Good & Gather Cut & Peeled Baby Carrots - 1lb",
                        brand="Good & Gather", amount=1, unit="lb", pack=1,
                        organic=False, pl=True),
    },
))
# Fix the category mapping: baby carrots are produce, not apples.
CONCEPTS[-1]["category"] = "Lettuce"  # Lettuce maps to Produce for all retailers.

CONCEPTS.append(C(
    "cream_cheese_8oz_pl", "Cheese", "private_label_equivalent", 2.49,
    {
        "kroger":  dict(name="Kroger Original Cream Cheese 8 oz",
                        brand="Kroger", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
        "walmart": dict(name="Great Value Original Cream Cheese, 8 oz",
                        brand="Great Value", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
        "target":  dict(name="Market Pantry Original Cream Cheese - 8oz",
                        brand="Market Pantry", amount=8, unit="oz", pack=1,
                        organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "evoo_17floz_pl", "Condiments", "private_label_equivalent", 7.99,
    {
        "whole_foods": dict(name="365 Extra Virgin Olive Oil 17 fl oz",
                            brand="365", amount=17, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Extra Virgin Olive Oil 17 oz",
                            brand="Kroger", amount=17, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "walmart":     dict(name="Great Value Extra Virgin Olive Oil, 17 fl oz",
                            brand="Great Value", amount=17, unit="fl oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Extra Virgin Olive Oil - 17 fl oz",
                            brand="Good & Gather", amount=17, unit="fl oz", pack=1,
                            organic=False, pl=True),
    },
))

CONCEPTS.append(C(
    "honey_12oz_pl", "Condiments", "private_label_equivalent", 4.99,
    {
        "whole_foods": dict(name="365 Pure Honey 12 oz Squeeze Bottle",
                            brand="365", amount=12, unit="oz", pack=1,
                            organic=False, pl=True),
        "kroger":      dict(name="Kroger Pure Honey 12 oz",
                            brand="Kroger", amount=12, unit="oz", pack=1,
                            organic=False, pl=True),
        "target":      dict(name="Good & Gather Pure Honey 12oz Squeeze Bottle",
                            brand="Good & Gather", amount=12, unit="oz", pack=1,
                            organic=False, pl=True),
    },
))


# === National-brand exact equivalents =====================================
# Same product across retailers (e.g., Coke 12pk). match_type=exact_equivalent.

# --- Coca-Cola family ---
CONCEPTS.append(C(
    "coke_zero_12pk_12oz", "Soda", "exact_equivalent", 8.99,
    {
        "whole_foods": dict(name="Coca-Cola Zero Sugar, 12 x 12 fl oz cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "kroger":      dict(name="Coke Zero 12pk 12 oz cans",
                            brand="Coke", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "walmart":     dict(name="Coca-Cola Zero Sugar Soda 12-Pack 12 fl oz",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "target":      dict(name="Coca-Cola Zero Sugar 12 Pack 12oz Cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "coke_zero_6pk_12oz", "Soda", "exact_equivalent", 4.99,
    {
        "kroger":      dict(name="Coke Zero 6pk 12 oz cans",
                            brand="Coke", amount=12, unit="fl oz", pack=6,
                            organic=False, pl=False),
        "walmart":     dict(name="Coca-Cola Zero Sugar Soda 6-Pack 12 fl oz",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=6,
                            organic=False, pl=False),
        "target":      dict(name="Coca-Cola Zero Sugar 6 Pack 12oz Cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=6,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "coke_classic_12pk_12oz", "Soda", "exact_equivalent", 8.99,
    {
        "whole_foods": dict(name="Coca-Cola Classic, 12 x 12 fl oz cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "kroger":      dict(name="Coca-Cola Classic 12pk, 12 oz cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "walmart":     dict(name="Coca-Cola Classic 12-Pack 12 fl oz Cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "target":      dict(name="Coca-Cola 12 Pack 12oz Cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "coke_2L", "Soda", "exact_equivalent", 2.49,
    {
        "kroger":      dict(name="Coca-Cola Classic 2 L Bottle",
                            brand="Coca-Cola", amount=2, unit="liter", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Coca-Cola Classic, 2 L",
                            brand="Coca-Cola", amount=2, unit="liter", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Coca-Cola - 2 L Bottle",
                            brand="Coca-Cola", amount=2, unit="liter", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "diet_coke_12pk_12oz", "Soda", "exact_equivalent", 8.99,
    {
        "kroger":      dict(name="Diet Coke 12pk 12 oz cans",
                            brand="Coke", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "walmart":     dict(name="Diet Coke Soda 12-Pack 12 fl oz",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
        "target":      dict(name="Diet Coke 12 Pack 12oz Cans",
                            brand="Coca-Cola", amount=12, unit="fl oz", pack=12,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "sprite_12pk_12oz", "Soda", "exact_equivalent", 8.49,
    {
        "kroger":  dict(name="Sprite Lemon-Lime 12pk 12 oz cans",
                        brand="Sprite", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "walmart": dict(name="Sprite Lemon Lime Soda 12-Pack 12 fl oz",
                        brand="Sprite", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "target":  dict(name="Sprite 12 Pack 12oz Cans",
                        brand="Sprite", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "pepsi_12pk_12oz", "Soda", "exact_equivalent", 8.49,
    {
        "kroger":  dict(name="Pepsi Cola 12pk 12 oz cans",
                        brand="Pepsi", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "walmart": dict(name="Pepsi Soda 12-Pack 12 fl oz",
                        brand="Pepsi", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "target":  dict(name="Pepsi 12 Pack 12oz Cans",
                        brand="Pepsi", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
    },
))

# --- LaCroix family ---
CONCEPTS.append(C(
    "lacroix_lime_8pk", "Sparkling water", "exact_equivalent", 4.49,
    {
        "whole_foods": dict(name="LaCroix Sparkling Water, Lime, 8 x 12 fl oz cans",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
        "kroger":      dict(name="LaCroix Lime Sparkling Water 8pk 12 oz",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
        "walmart":     dict(name="LaCroix Sparkling Water Lime 8-Pack 12 fl oz Cans",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
        "target":      dict(name="LaCroix Lime Sparkling Water 8 Pack 12oz Cans",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "lacroix_lime_12pk", "Sparkling water", "exact_equivalent", 5.99,
    {
        "kroger":  dict(name="LaCroix Lime Sparkling Water 12pk 12 oz",
                        brand="LaCroix", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "walmart": dict(name="LaCroix Sparkling Water Lime 12-Pack 12 fl oz",
                        brand="LaCroix", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
        "target":  dict(name="LaCroix Lime Sparkling Water 12 Pack 12oz",
                        brand="LaCroix", amount=12, unit="fl oz", pack=12,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "lacroix_grapefruit_8pk", "Sparkling water", "exact_equivalent", 4.49,
    {
        "whole_foods": dict(name="LaCroix Sparkling Water, Pamplemousse (Grapefruit), 8 x 12 fl oz cans",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
        "kroger":      dict(name="LaCroix Pamplemousse Grapefruit Sparkling Water 8pk 12 oz",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
        "target":      dict(name="LaCroix Pamplemousse Sparkling Water 8 Pack 12oz",
                            brand="LaCroix", amount=12, unit="fl oz", pack=8,
                            organic=False, pl=False),
    },
))

# --- Dairy alternatives (national brands) ---
CONCEPTS.append(C(
    "silk_unsweet_almond_64floz", "Plant-based milk", "exact_equivalent", 4.49,
    {
        "whole_foods": dict(name="Silk Unsweetened Almondmilk 64 fl oz",
                            brand="Silk", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Silk Unsweet Almond Milk Half Gallon",
                            brand="Silk", amount=0.5, unit="gal", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Silk Unsweetened Almondmilk, 64 fl oz",
                            brand="Silk", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Silk Unsweetened Almondmilk - 64 fl oz",
                            brand="Silk", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "silk_orig_soy_64floz", "Plant-based milk", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Silk Original Soymilk Half Gallon",
                        brand="Silk", amount=0.5, unit="gal", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Silk Original Soymilk 64 fl oz",
                        brand="Silk", amount=64, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Silk Original Soymilk - 64 fl oz",
                        brand="Silk", amount=64, unit="fl oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "oatly_orig_64floz", "Plant-based milk", "exact_equivalent", 5.49,
    {
        "whole_foods": dict(name="Oatly Original Oatmilk 64 fl oz",
                            brand="Oatly", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Oatly Original Oat Milk 64 fl oz",
                            brand="Oatly", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Oatly Original Oat Milk - 64 fl oz",
                            brand="Oatly", amount=64, unit="fl oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "oatly_barista_32floz", "Plant-based milk", "exact_equivalent", 5.49,
    {
        "whole_foods": dict(name="Oatly Barista Edition Oatmilk 32 fl oz",
                            brand="Oatly", amount=32, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Oatly Barista Edition Oat Milk 32 fl oz",
                            brand="Oatly", amount=32, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Oatly Barista Edition Oatmilk - 32 fl oz",
                            brand="Oatly", amount=32, unit="fl oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "horizon_organic_whole_64floz", "Dairy milk", "exact_equivalent", 4.99,
    {
        "whole_foods": dict(name="Horizon Organic Whole Milk 64 fl oz",
                            brand="Horizon Organic", amount=64, unit="fl oz", pack=1,
                            organic=True, pl=False),
        "kroger":      dict(name="Horizon Organic Whole Milk Half Gallon",
                            brand="Horizon Organic", amount=0.5, unit="gal", pack=1,
                            organic=True, pl=False),
        "walmart":     dict(name="Horizon Organic Whole Milk, 64 fl oz",
                            brand="Horizon Organic", amount=64, unit="fl oz", pack=1,
                            organic=True, pl=False),
        "target":      dict(name="Horizon Organic Whole Milk - 0.5 Gal",
                            brand="Horizon Organic", amount=0.5, unit="gal", pack=1,
                            organic=True, pl=False),
    },
))

CONCEPTS.append(C(
    "horizon_organic_2pct_1gal", "Dairy milk", "exact_equivalent", 7.49,
    {
        "kroger":  dict(name="Horizon Organic 2% Reduced Fat Milk 1 Gallon",
                        brand="Horizon Organic", amount=1, unit="gal", pack=1,
                        organic=True, pl=False),
        "walmart": dict(name="Horizon Organic 2% Reduced Fat Milk 1 gal",
                        brand="Horizon Organic", amount=1, unit="gal", pack=1,
                        organic=True, pl=False),
        "target":  dict(name="Horizon Organic 2% Reduced Fat Milk - 1 Gal",
                        brand="Horizon Organic", amount=1, unit="gal", pack=1,
                        organic=True, pl=False),
    },
))

CONCEPTS.append(C(
    "fairlife_2pct_52floz", "Dairy milk", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Fairlife Lactose Free 2% Ultra-Filtered Milk 52 fl oz",
                        brand="Fairlife", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Fairlife Lactose Free 2% Ultra-Filtered Milk, 52 fl oz",
                        brand="Fairlife", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Fairlife Lactose Free 2% Reduced Fat Ultra Filtered Milk - 52 fl oz",
                        brand="Fairlife", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
    },
))

# --- Yogurt ---
CONCEPTS.append(C(
    "chobani_blueberry_5p3oz", "Yogurt", "exact_equivalent", 1.39,
    {
        "whole_foods": dict(name="Chobani Greek Yogurt, Blueberry on the Bottom, 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Chobani Blueberry Greek Yogurt 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Chobani Non-Fat Greek Yogurt, Blueberry, 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Chobani Blueberry on the Bottom Nonfat Greek Yogurt - 5.3oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "chobani_blueberry_4pk", "Yogurt", "exact_equivalent", 5.49,
    {
        "kroger":  dict(name="Chobani Blueberry Greek Yogurt, 4 ct, 5.3 oz Cups",
                        brand="Chobani", amount=5.3, unit="oz", pack=4,
                        organic=False, pl=False),
        "walmart": dict(name="Chobani Non-Fat Greek Yogurt Blueberry 4-Pack 5.3 oz Cups",
                        brand="Chobani", amount=5.3, unit="oz", pack=4,
                        organic=False, pl=False),
        "target":  dict(name="Chobani Blueberry Nonfat Greek Yogurt 4 Pack - 5.3oz Cups",
                        brand="Chobani", amount=5.3, unit="oz", pack=4,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "chobani_strawberry_5p3oz", "Yogurt", "exact_equivalent", 1.39,
    {
        "whole_foods": dict(name="Chobani Greek Yogurt, Strawberry on the Bottom, 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Chobani Strawberry Greek Yogurt 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Chobani Non-Fat Greek Yogurt, Strawberry, 5.3 oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Chobani Strawberry on the Bottom Nonfat Greek Yogurt - 5.3oz",
                            brand="Chobani", amount=5.3, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "fage_total_plain_35p3oz", "Yogurt", "exact_equivalent", 6.99,
    {
        "whole_foods": dict(name="FAGE Total Plain Whole Milk Greek Yogurt 35.3 oz",
                            brand="FAGE", amount=35.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="FAGE Total 5% Plain Greek Yogurt 35.3 oz",
                            brand="FAGE", amount=35.3, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="FAGE Total Plain Whole Milk Greek Yogurt - 35.3oz",
                            brand="FAGE", amount=35.3, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

# --- Cereal ---
CONCEPTS.append(C(
    "cheerios_orig_12oz", "Cereal", "exact_equivalent", 4.49,
    {
        "whole_foods": dict(name="General Mills Cheerios Original 12 oz",
                            brand="Cheerios", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Cheerios Original Cereal 12 oz",
                            brand="Cheerios", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Cheerios Original Cereal, 12 oz Box",
                            brand="Cheerios", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Cheerios Original Breakfast Cereal - 12oz",
                            brand="Cheerios", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "cheerios_honey_nut_10p8oz", "Cereal", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Honey Nut Cheerios Cereal 10.8 oz",
                        brand="Cheerios", amount=10.8, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Honey Nut Cheerios Cereal, 10.8 oz",
                        brand="Cheerios", amount=10.8, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Honey Nut Cheerios Breakfast Cereal - 10.8oz",
                        brand="Cheerios", amount=10.8, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "frosted_flakes_13p5oz", "Cereal", "exact_equivalent", 4.99,
    {
        "kroger":  dict(name="Kellogg's Frosted Flakes Cereal 13.5 oz",
                        brand="Kellogg's", amount=13.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Kellogg's Frosted Flakes Cereal, 13.5 oz",
                        brand="Kellogg's", amount=13.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Frosted Flakes Breakfast Cereal - 13.5oz - Kellogg's",
                        brand="Kellogg's", amount=13.5, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "quaker_oats_42oz", "Cereal", "exact_equivalent", 5.99,
    {
        "whole_foods": dict(name="Quaker Old Fashioned Oats 42 oz",
                            brand="Quaker", amount=42, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Quaker Old Fashioned Oats 42 oz Canister",
                            brand="Quaker", amount=42, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Quaker Oats Old Fashioned, 42 oz",
                            brand="Quaker", amount=42, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

# --- Chips & snacks ---
CONCEPTS.append(C(
    "tostitos_scoops_10oz", "Chips", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Tostitos Scoops Tortilla Chips 10 oz",
                        brand="Tostitos", amount=10, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Tostitos Scoops Tortilla Chips, 10 oz Bag",
                        brand="Tostitos", amount=10, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Tostitos Scoops! Tortilla Chips - 10oz",
                        brand="Tostitos", amount=10, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "doritos_nacho_9p25oz", "Chips", "exact_equivalent", 4.99,
    {
        "kroger":  dict(name="Doritos Nacho Cheese Tortilla Chips 9.25 oz",
                        brand="Doritos", amount=9.25, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Doritos Nacho Cheese Flavored Tortilla Chips, 9.25 oz",
                        brand="Doritos", amount=9.25, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Doritos Nacho Cheese Tortilla Chips - 9.25oz",
                        brand="Doritos", amount=9.25, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "lays_classic_8oz", "Chips", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Lay's Classic Potato Chips 8 oz",
                        brand="Lay's", amount=8, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Lay's Classic Potato Chips, 8 oz Bag",
                        brand="Lay's", amount=8, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Lay's Classic Potato Chips - 8oz",
                        brand="Lay's", amount=8, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "oreo_orig_14p3oz", "Snacks", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="OREO Original Cookies 14.3 oz Family Size",
                        brand="Oreo", amount=14.3, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="OREO Chocolate Sandwich Cookies, 14.3 oz",
                        brand="Oreo", amount=14.3, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Oreo Chocolate Sandwich Cookies - 14.3oz",
                        brand="Oreo", amount=14.3, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "ritz_orig_13p7oz", "Snacks", "exact_equivalent", 4.29,
    {
        "kroger":  dict(name="RITZ Original Crackers 13.7 oz",
                        brand="Ritz", amount=13.7, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="RITZ Original Crackers, 13.7 oz",
                        brand="Ritz", amount=13.7, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Ritz Original Crackers - 13.7oz",
                        brand="Ritz", amount=13.7, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "cheez_it_orig_12p4oz", "Snacks", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Cheez-It Original Baked Snack Crackers 12.4 oz",
                        brand="Cheez-It", amount=12.4, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Cheez-It Original Cheese Crackers, 12.4 oz",
                        brand="Cheez-It", amount=12.4, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Cheez-It Original Baked Snack Crackers - 12.4oz",
                        brand="Cheez-It", amount=12.4, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

# --- Pantry / sauces / condiments ---
CONCEPTS.append(C(
    "skippy_creamy_pb_16p3oz", "Peanut butter", "exact_equivalent", 3.49,
    {
        "kroger":  dict(name="Skippy Creamy Peanut Butter 16.3 oz",
                        brand="Skippy", amount=16.3, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Skippy Creamy Peanut Butter, 16.3 oz",
                        brand="Skippy", amount=16.3, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Skippy Creamy Peanut Butter - 16.3oz",
                        brand="Skippy", amount=16.3, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "jif_creamy_pb_16oz", "Peanut butter", "exact_equivalent", 3.49,
    {
        "whole_foods": dict(name="Jif Creamy Peanut Butter 16 oz",
                            brand="Jif", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Jif Creamy Peanut Butter 16 oz Jar",
                            brand="Jif", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Jif Creamy Peanut Butter, 16 oz",
                            brand="Jif", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Jif Creamy Peanut Butter - 16oz",
                            brand="Jif", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "heinz_ketchup_20oz", "Condiments", "exact_equivalent", 3.49,
    {
        "kroger":  dict(name="Heinz Tomato Ketchup 20 oz",
                        brand="Heinz", amount=20, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Heinz Tomato Ketchup, 20 oz",
                        brand="Heinz", amount=20, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Heinz Tomato Ketchup - 20oz",
                        brand="Heinz", amount=20, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "hellmanns_mayo_30oz", "Condiments", "exact_equivalent", 5.99,
    {
        "kroger":  dict(name="Hellmann's Real Mayonnaise 30 oz",
                        brand="Hellmann's", amount=30, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Hellmann's Real Mayonnaise, 30 oz",
                        brand="Hellmann's", amount=30, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Hellmann's Real Mayonnaise - 30oz",
                        brand="Hellmann's", amount=30, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "kraft_mac_cheese_7p25oz", "Pasta", "exact_equivalent", 1.49,
    {
        "kroger":  dict(name="Kraft Original Macaroni & Cheese Dinner 7.25 oz",
                        brand="Kraft", amount=7.25, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Kraft Original Mac & Cheese Dinner, 7.25 oz",
                        brand="Kraft", amount=7.25, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Kraft Original Macaroni & Cheese - 7.25oz",
                        brand="Kraft", amount=7.25, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "barilla_spaghetti_16oz", "Pasta", "exact_equivalent", 1.99,
    {
        "whole_foods": dict(name="Barilla Spaghetti Pasta 16 oz",
                            brand="Barilla", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Barilla Spaghetti 1 lb Box",
                            brand="Barilla", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Barilla Spaghetti, 16 oz",
                            brand="Barilla", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Barilla Spaghetti Pasta - 16oz",
                            brand="Barilla", amount=16, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "raos_marinara_24oz", "Pasta", "exact_equivalent", 8.99,
    {
        "whole_foods": dict(name="Rao's Homemade Marinara Sauce 24 oz",
                            brand="Rao's", amount=24, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Rao's Homemade Marinara Sauce 24 oz Jar",
                            brand="Rao's", amount=24, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Rao's Homemade Marinara Sauce - 24oz",
                            brand="Rao's", amount=24, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

# --- Frozen pizza & ice cream ---
CONCEPTS.append(C(
    "digiorno_pepperoni", "Frozen pizza", "exact_equivalent", 6.99,
    {
        "kroger":  dict(name="DiGiorno Original Rising Crust Pepperoni Pizza 27.5 oz",
                        brand="DiGiorno", amount=27.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="DiGiorno Pepperoni Frozen Pizza Original Rising Crust 27.5 oz",
                        brand="DiGiorno", amount=27.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="DiGiorno Original Rising Crust Pepperoni Frozen Pizza - 27.5oz",
                        brand="DiGiorno", amount=27.5, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "tombstone_pepperoni", "Frozen pizza", "exact_equivalent", 4.49,
    {
        "kroger":  dict(name="Tombstone Original Pepperoni Frozen Pizza 18.5 oz",
                        brand="Tombstone", amount=18.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Tombstone Pepperoni Original Frozen Pizza, 18.5 oz",
                        brand="Tombstone", amount=18.5, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "bj_cherry_garcia_pint", "Ice cream", "exact_equivalent", 5.99,
    {
        "whole_foods": dict(name="Ben & Jerry's Cherry Garcia Ice Cream 1 Pint",
                            brand="Ben & Jerry's", amount=16, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Ben & Jerry's Cherry Garcia Ice Cream 16 fl oz",
                            brand="Ben & Jerry's", amount=16, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Ben & Jerry's Cherry Garcia Ice Cream, 16 fl oz Pint",
                            brand="Ben & Jerry's", amount=16, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Ben & Jerry's Cherry Garcia Ice Cream - 1 pint",
                            brand="Ben & Jerry's", amount=16, unit="fl oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "haagen_vanilla_14floz", "Ice cream", "exact_equivalent", 5.49,
    {
        "kroger":  dict(name="Häagen-Dazs Vanilla Ice Cream 14 fl oz",
                        brand="Häagen-Dazs", amount=14, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Haagen-Dazs Vanilla Ice Cream, 14 fl oz",
                        brand="Häagen-Dazs", amount=14, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Häagen-Dazs Vanilla Ice Cream - 14 fl oz",
                        brand="Häagen-Dazs", amount=14, unit="fl oz", pack=1,
                        organic=False, pl=False),
    },
))

# --- OJ ---
CONCEPTS.append(C(
    "tropicana_pure_premium_52floz", "Orange juice", "exact_equivalent", 4.99,
    {
        "whole_foods": dict(name="Tropicana Pure Premium Orange Juice No Pulp 52 fl oz",
                            brand="Tropicana", amount=52, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Tropicana Pure Premium Orange Juice 52 fl oz",
                            brand="Tropicana", amount=52, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Tropicana Pure Premium 100% Orange Juice, 52 fl oz",
                            brand="Tropicana", amount=52, unit="fl oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Tropicana Pure Premium Orange Juice - 52 fl oz",
                            brand="Tropicana", amount=52, unit="fl oz", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "simply_oj_52floz", "Orange juice", "exact_equivalent", 4.99,
    {
        "kroger":  dict(name="Simply Orange Pulp Free Orange Juice 52 fl oz",
                        brand="Simply", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Simply Orange Pulp Free 100% Orange Juice, 52 fl oz",
                        brand="Simply", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Simply Orange Pulp Free Orange Juice - 52 fl oz",
                        brand="Simply", amount=52, unit="fl oz", pack=1,
                        organic=False, pl=False),
    },
))

# --- Coffee ---
CONCEPTS.append(C(
    "folgers_classic_30p5oz", "Coffee", "exact_equivalent", 8.99,
    {
        "kroger":  dict(name="Folgers Classic Roast Ground Coffee 30.5 oz",
                        brand="Folgers", amount=30.5, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Folgers Classic Roast Ground Coffee, 30.5 oz",
                        brand="Folgers", amount=30.5, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "starbucks_pikeplace_12oz", "Coffee", "exact_equivalent", 11.99,
    {
        "whole_foods": dict(name="Starbucks Pike Place Roast Whole Bean Coffee 12 oz",
                            brand="Starbucks", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Starbucks Pike Place Roast Whole Bean Coffee 12 oz",
                            brand="Starbucks", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Starbucks Pike Place Roast Whole Bean Coffee, 12 oz Bag",
                            brand="Starbucks", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Starbucks Pike Place Whole Bean Coffee - 12oz",
                            brand="Starbucks", amount=12, unit="oz", pack=1,
                            organic=False, pl=False),
    },
))

# --- Granola / bars ---
CONCEPTS.append(C(
    "kind_dark_choc_nut_6pk", "Granola bars", "exact_equivalent", 6.49,
    {
        "whole_foods": dict(name="KIND Dark Chocolate Nuts & Sea Salt Bars 6 ct",
                            brand="KIND", amount=1.4, unit="oz", pack=6,
                            organic=False, pl=False),
        "kroger":      dict(name="KIND Dark Chocolate Nuts & Sea Salt Bars 6 ct, 1.4 oz",
                            brand="KIND", amount=1.4, unit="oz", pack=6,
                            organic=False, pl=False),
        "target":      dict(name="KIND Dark Chocolate Nuts & Sea Salt Bars - 6ct",
                            brand="KIND", amount=1.4, unit="oz", pack=6,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "clif_choc_chip_6pk", "Granola bars", "exact_equivalent", 7.99,
    {
        "kroger":  dict(name="CLIF BAR Chocolate Chip Energy Bars 6 ct",
                        brand="Clif", amount=2.4, unit="oz", pack=6,
                        organic=False, pl=False),
        "walmart": dict(name="CLIF BAR Chocolate Chip Energy Bars, 6 Pack 2.4 oz",
                        brand="Clif", amount=2.4, unit="oz", pack=6,
                        organic=False, pl=False),
        "target":  dict(name="CLIF Bar Chocolate Chip Energy Bars - 6ct",
                        brand="Clif", amount=2.4, unit="oz", pack=6,
                        organic=False, pl=False),
    },
))

# --- Produce: branded organic (Driscoll's, etc.) and bulk ---
CONCEPTS.append(C(
    "driscolls_strawberries_1lb", "Berries", "exact_equivalent", 4.99,
    {
        "kroger":  dict(name="Driscoll's Strawberries 1 lb Clamshell",
                        brand="Driscoll's", amount=1, unit="lb", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Driscoll's Strawberries, 1 lb",
                        brand="Driscoll's", amount=16, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Driscoll's Strawberries - 1lb",
                        brand="Driscoll's", amount=1, unit="lb", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "driscolls_blueberries_6oz", "Berries", "exact_equivalent", 3.99,
    {
        "kroger":  dict(name="Driscoll's Blueberries 6 oz",
                        brand="Driscoll's", amount=6, unit="oz", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="Driscoll's Blueberries, 6 oz",
                        brand="Driscoll's", amount=6, unit="oz", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="Driscoll's Blueberries - 6oz",
                        brand="Driscoll's", amount=6, unit="oz", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "bulk_organic_fuji_apples", "Apples", "exact_equivalent", 2.49,
    {
        "whole_foods": dict(name="Organic Fuji Apples (each)",
                            brand="", amount=1, unit="each", pack=1,
                            organic=True, pl=False),
        "kroger":      dict(name="Bulk Organic Fuji Apples per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=True, pl=False),
        "walmart":     dict(name="Organic Fuji Apples Loose, per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=True, pl=False),
    },
))

CONCEPTS.append(C(
    "bulk_bananas", "Bananas", "exact_equivalent", 0.59,
    {
        "whole_foods": dict(name="Organic Bananas (each)",
                            brand="", amount=1, unit="each", pack=1,
                            organic=True, pl=False),
        "kroger":      dict(name="Bananas (per lb)",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Bananas, per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Bananas - each",
                            brand="", amount=1, unit="each", pack=1,
                            organic=False, pl=False),
    },
))

# --- Meat ---
CONCEPTS.append(C(
    "boneless_chicken_breast_perlb", "Chicken", "exact_equivalent", 5.99,
    {
        "whole_foods": dict(name="Boneless Skinless Chicken Breasts per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Boneless Skinless Chicken Breast (per lb)",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Boneless Skinless Chicken Breast, per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "target":      dict(name="Boneless Skinless Chicken Breast - 1 lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "ground_beef_80_20_perlb", "Beef", "exact_equivalent", 5.49,
    {
        "kroger":  dict(name="Ground Beef 80/20 1 lb",
                        brand="", amount=1, unit="lb", pack=1,
                        organic=False, pl=False),
        "walmart": dict(name="80% Lean / 20% Fat Ground Beef, 1 lb",
                        brand="", amount=1, unit="lb", pack=1,
                        organic=False, pl=False),
        "target":  dict(name="80/20 Ground Beef - 1lb",
                        brand="", amount=1, unit="lb", pack=1,
                        organic=False, pl=False),
    },
))

CONCEPTS.append(C(
    "atlantic_salmon_perlb", "Seafood", "exact_equivalent", 12.99,
    {
        "whole_foods": dict(name="Atlantic Salmon Fillet per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "kroger":      dict(name="Atlantic Salmon Fillet (per lb)",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
        "walmart":     dict(name="Atlantic Salmon Fillet, per lb",
                            brand="", amount=1, unit="lb", pack=1,
                            organic=False, pl=False),
    },
))


# ---------------------------------------------------------------------------
# Substitute pairs — these are concepts that map to each other "across brand
# space" but should NOT be considered exact matches. They map a private-label
# product to a national-brand alternative of the same item type and size.
# We declare these by concept_id of two existing concepts.
# ---------------------------------------------------------------------------
SUBSTITUTE_CONCEPT_PAIRS = [
    # (private label concept, national-brand concept, reason)
    ("unsweet_almond_milk_64floz_pl", "silk_unsweet_almond_64floz",
     "private-label unsweetened almond milk vs Silk; same form factor, different brand"),
    ("oat_milk_64floz_pl", "oatly_orig_64floz",
     "private-label oat milk vs Oatly; comparable substitute, not an exact equivalent"),
    ("tortilla_chips_13oz_pl", "tostitos_scoops_10oz",
     "private-label tortilla chips vs Tostitos; different brand and size, similar use"),
    ("sparkling_water_lime_8pk_pl", "lacroix_lime_8pk",
     "private-label lime sparkling water vs LaCroix; same pack/size, different brand"),
    ("creamy_pb_16oz_pl", "jif_creamy_pb_16oz",
     "private-label creamy peanut butter vs Jif; same size, different brand"),
    ("creamy_pb_16oz_pl", "skippy_creamy_pb_16p3oz",
     "private-label creamy peanut butter vs Skippy; near-equivalent size, different brand"),
    ("frozen_pepperoni_pizza_pl", "tombstone_pepperoni",
     "private-label pepperoni pizza vs Tombstone; comparable substitute"),
    ("oj_no_pulp_52floz_pl", "tropicana_pure_premium_52floz",
     "private-label OJ vs Tropicana Pure Premium; same size, different brand"),
    ("oj_no_pulp_52floz_pl", "simply_oj_52floz",
     "private-label OJ vs Simply Orange; same size, different brand"),
    ("plain_greek_yogurt_32oz_pl", "fage_total_plain_35p3oz",
     "private-label plain Greek yogurt vs FAGE Total; near-equivalent size"),
    ("spaghetti_pasta_16oz_pl", "barilla_spaghetti_16oz",
     "private-label spaghetti vs Barilla; same size, different brand"),
    ("organic_whole_milk_1gal_pl", "horizon_organic_2pct_1gal",
     "private-label organic whole milk vs Horizon Organic 2%; comparable but different fat content"),
]


# ---------------------------------------------------------------------------
# Negative-pair "near-miss" generators.
#
# For some concepts we ALSO produce a near-miss row in some retailer:
# same product but different size/pack/organic-status/sweetened-status.
# That near-miss row is NOT in the original match group — it goes into
# negative_pairs.json paired against a row from the original group.
# ---------------------------------------------------------------------------
# Each entry: (source_concept_id, retailer, mutation, reason_template)
# mutation is a callable: takes the source row dict and returns a mutated copy.

def _shrink_amount(row, factor=0.5):
    new = dict(row)
    new = new.copy()
    new["amount"] = round(new["amount"] * factor, 2) if isinstance(new["amount"], (int, float)) else new["amount"]
    new["name"] = new["name"] + f" (alt size {new['amount']} {new['unit']})"
    return new


def _flip_pack(row, new_pack):
    new = dict(row)
    new["pack"] = new_pack
    new["name"] = new["name"].replace(f"{row['pack']}pk", f"{new_pack}pk") \
                              .replace(f"{row['pack']} ct", f"{new_pack} ct") \
                              .replace(f"{row['pack']}-Pack", f"{new_pack}-Pack") \
                              .replace(f"{row['pack']} Pack", f"{new_pack} Pack")
    if str(new_pack) not in new["name"]:
        new["name"] = new["name"] + f" {new_pack}-Pack"
    return new


def _flip_organic(row):
    new = dict(row)
    new["organic"] = not new.get("organic", False)
    if new["organic"]:
        new["name"] = "Organic " + new["name"]
    else:
        new["name"] = new["name"].replace("Organic ", "").replace("365 Organic", "365")
    return new


def _flip_sweetened(row):
    new = dict(row)
    if "Unsweetened" in new["name"]:
        new["name"] = new["name"].replace("Unsweetened", "Original")
    else:
        new["name"] = new["name"].replace("Original", "Sweetened Vanilla")
    return new


# We'll programmatically generate near-miss rows below.
NEAR_MISS_RULES = [
    # Different pack count: shrink Coke 12pk → 8pk on a single retailer.
    ("coke_zero_12pk_12oz", "kroger",
     lambda r: _flip_pack(r, 8),
     "different pack count (8pk vs 12pk) of same Coke Zero"),
    ("lacroix_lime_12pk", "walmart",
     lambda r: _flip_pack(r, 8),
     "different pack count (8pk vs 12pk) of same LaCroix Lime"),
    # Different size: half gal vs full gal milk.
    ("organic_whole_milk_1gal_pl", "kroger",
     lambda r: _shrink_amount(r, 0.5),
     "half-gallon vs full-gallon size of same organic whole milk"),
    # Organic vs non-organic: flip organic flag for a Cheerios-style item.
    ("cheerios_orig_12oz", "target",
     _flip_organic,
     "organic vs non-organic same brand same size"),
    # Sweetened vs unsweetened.
    ("unsweet_almond_milk_64floz_pl", "walmart",
     _flip_sweetened,
     "sweetened vs unsweetened same brand same size"),
    # Bulk vs bagged apples.
    ("organic_gala_apples_3lb_pl", "whole_foods",
     lambda r: dict(r, name="Bulk Organic Gala Apples (each)",
                    amount=1, unit="each", pack=1),
     "bulk loose apples vs 3 lb bagged apples"),
    # Same brand different flavor (Chobani blueberry → strawberry).
    ("chobani_blueberry_5p3oz", "walmart",
     lambda r: dict(r, name=r["name"].replace("Blueberry", "Strawberry")
                                       .replace("blueberry", "strawberry")),
     "same brand same size but different flavor (blueberry vs strawberry)"),
    # Different berry size 6oz vs 18oz.
    ("driscolls_blueberries_6oz", "kroger",
     lambda r: dict(r, name="Driscoll's Blueberries 18 oz Family Pack",
                    amount=18),
     "6 oz clamshell vs 18 oz family pack same brand"),
]


# ---------------------------------------------------------------------------
# Unique unmatched product templates (per retailer).
# These products only exist at one retailer, so they should never appear in
# match groups. We just need them to push row count up and to test that the
# matcher doesn't over-match.
# ---------------------------------------------------------------------------
UNIQUE_PRODUCTS = {
    "whole_foods": [
        ("Allegro Coffee Organic French Roast 12 oz", "Allegro", "Coffee", 11.99, 12, "oz", 1, True, False),
        ("365 Organic Coconut Oil 14 oz", "365", "Condiments", 6.49, 14, "oz", 1, True, True),
        ("365 Organic Maple Syrup Grade A 12 fl oz", "365", "Condiments", 8.99, 12, "fl oz", 1, True, True),
        ("Vital Farms Pasture-Raised Eggs Dozen", "Vital Farms", "Eggs", 7.99, 12, "ct", 1, False, False),
        ("365 Organic Quinoa 16 oz", "365", "Pantry", 4.99, 16, "oz", 1, True, True),
        ("Whole Foods Bakery Sourdough Boule 24 oz", "Whole Foods Bakery", "Bread", 6.99, 24, "oz", 1, False, True),
        ("Siggi's Plain Skyr Yogurt 24 oz", "Siggi's", "Yogurt", 5.99, 24, "oz", 1, False, False),
        ("365 Organic Tomato Paste 6 oz Can", "365", "Pantry", 1.49, 6, "oz", 1, True, True),
        ("Mary's Free Range Whole Chicken per lb", "Mary's", "Chicken", 4.99, 1, "lb", 1, False, False),
        ("Manchego Cheese 1/4 lb wedge", "Whole Foods", "Cheese", 7.99, 0.25, "lb", 1, False, False),
        ("Burrata Cheese 8 oz tub", "Belgioioso", "Cheese", 6.99, 8, "oz", 1, False, False),
        ("Whole Foods Prepared Caesar Salad 8 oz", "Whole Foods", "Lettuce", 7.99, 8, "oz", 1, False, True),
        ("Bulk Pistachios per lb", "", "Snacks", 14.99, 1, "lb", 1, False, False),
        ("365 Organic Hummus Classic 10 oz", "365", "Snacks", 3.99, 10, "oz", 1, True, True),
        ("Honeycrisp Apples per lb", "", "Apples", 3.49, 1, "lb", 1, False, False),
        ("Organic Heirloom Tomatoes per lb", "", "Lettuce", 5.99, 1, "lb", 1, True, False),
        ("Whole Foods Sushi California Roll 8 pc", "Whole Foods", "Seafood", 9.99, 8, "ct", 1, False, True),
        ("Organic Avocados 4 ct bag", "", "Avocados", 6.99, 4, "ct", 1, True, False),
        ("365 Organic Black Pepper Ground 2.5 oz", "365", "Pantry", 4.49, 2.5, "oz", 1, True, True),
        ("365 Organic Vegetable Broth 32 fl oz", "365", "Soup", 3.49, 32, "fl oz", 1, True, True),
        ("Whole Foods Rotisserie Chicken each", "Whole Foods", "Chicken", 9.99, 1, "each", 1, False, True),
        ("Stonyfield Organic Whole Milk Yogurt 32 oz", "Stonyfield", "Yogurt", 5.99, 32, "oz", 1, True, False),
        ("365 Organic Frozen Wild Blueberries 10 oz", "365", "Frozen pizza", 4.99, 10, "oz", 1, True, True),
        ("Organic Kale Bunch each", "", "Lettuce", 2.99, 1, "each", 1, True, False),
        ("La Colombe Triple Draft Latte 9 fl oz can", "La Colombe", "Coffee", 3.99, 9, "fl oz", 1, False, False),
        ("365 Organic Almonds Raw 16 oz", "365", "Snacks", 9.99, 16, "oz", 1, True, True),
        ("Bulk Organic Quinoa per lb", "", "Pantry", 5.99, 1, "lb", 1, True, False),
        ("Hu Dark Chocolate Bar 2.1 oz", "Hu", "Snacks", 5.49, 2.1, "oz", 1, False, False),
        ("365 Organic Coconut Milk Can 13.5 fl oz", "365", "Pantry", 2.99, 13.5, "fl oz", 1, True, True),
        ("Whole Foods Garlic Naan 4 ct", "Whole Foods", "Bread", 4.99, 4, "ct", 1, False, True),
    ],
    "kroger": [
        ("Private Selection Sharp Cheddar Cheese 8 oz", "Private Selection", "Cheese", 4.49, 8, "oz", 1, False, True),
        ("Kroger Whole Wheat Pasta Penne 16 oz", "Kroger", "Pasta", 1.49, 16, "oz", 1, False, True),
        ("Simple Truth Organic Honey 12 oz", "Simple Truth", "Condiments", 5.49, 12, "oz", 1, True, True),
        ("Kroger Pancake Mix Original 32 oz", "Kroger", "Cereal", 2.49, 32, "oz", 1, False, True),
        ("Kroger Frozen Mixed Vegetables 12 oz", "Kroger", "Frozen pizza", 1.79, 12, "oz", 1, False, True),
        ("Kroger Vanilla Ice Cream 48 fl oz", "Kroger", "Ice cream", 3.99, 48, "fl oz", 1, False, True),
        ("Kroger Hot Dog Buns 8 ct", "Kroger", "Bread", 1.49, 8, "ct", 1, False, True),
        ("Kroger American Cheese Singles 16 ct", "Kroger", "Cheese", 3.49, 16, "ct", 1, False, True),
        ("Tillamook Cheddar Cheese Block 8 oz", "Tillamook", "Cheese", 4.99, 8, "oz", 1, False, False),
        ("Kroger Diced Tomatoes Can 14.5 oz", "Kroger", "Pantry", 0.99, 14.5, "oz", 1, False, True),
        ("Kroger Beef Hot Dogs 16 oz", "Kroger", "Beef", 4.99, 16, "oz", 1, False, True),
        ("Boneless Pork Chops per lb", "", "Beef", 4.49, 1, "lb", 1, False, False),
        ("Tilapia Fillet (per lb)", "", "Seafood", 6.99, 1, "lb", 1, False, False),
        ("Kroger Sliced Provolone Cheese 8 oz", "Kroger", "Cheese", 2.99, 8, "oz", 1, False, True),
        ("Bulk Russet Potatoes 5 lb Bag", "", "Apples", 4.99, 5, "lb", 1, False, False),
        ("Yellow Onions 3 lb Bag", "", "Lettuce", 3.49, 3, "lb", 1, False, False),
        ("Bell Peppers each", "", "Lettuce", 1.49, 1, "each", 1, False, False),
        ("Kroger Apple Sauce Cups 6 ct", "Kroger", "Snacks", 2.49, 4, "oz", 6, False, True),
        ("Kroger Granola Cereal Honey Almond 12 oz", "Kroger", "Cereal", 3.49, 12, "oz", 1, False, True),
        ("Annie's Organic Mac & Cheese Shells 6 oz", "Annie's", "Pasta", 2.49, 6, "oz", 1, True, False),
        ("Yoplait Original Strawberry Yogurt 6 oz", "Yoplait", "Yogurt", 0.99, 6, "oz", 1, False, False),
        ("Kroger Frozen Cheese Pizza 16 oz", "Kroger", "Frozen pizza", 2.99, 16, "oz", 1, False, True),
        ("Kroger Sliced Sandwich Pickles 16 fl oz", "Kroger", "Condiments", 2.49, 16, "fl oz", 1, False, True),
        ("Kroger Yellow Mustard 14 oz", "Kroger", "Condiments", 1.49, 14, "oz", 1, False, True),
        ("Kroger Frozen Hash Browns 30 oz", "Kroger", "Frozen pizza", 3.49, 30, "oz", 1, False, True),
        ("Kroger Sour Cream 16 oz", "Kroger", "Cheese", 2.49, 16, "oz", 1, False, True),
        ("Kroger Sliced Turkey Lunch Meat 9 oz", "Kroger", "Beef", 4.49, 9, "oz", 1, False, True),
        ("Kroger Tortillas Flour 10 ct", "Kroger", "Bread", 2.49, 10, "ct", 1, False, True),
        ("Kroger Salsa Medium 16 oz", "Kroger", "Condiments", 2.49, 16, "oz", 1, False, True),
        ("Kroger Garlic Bread Loaf 16 oz", "Kroger", "Bread", 3.99, 16, "oz", 1, False, True),
    ],
    "walmart": [
        ("Great Value Frozen Broccoli Florets 12 oz", "Great Value", "Frozen pizza", 1.49, 12, "oz", 1, False, True),
        ("Great Value 2% Reduced Fat Milk 1 gal", "Great Value", "Dairy milk", 3.99, 1, "gal", 1, False, True),
        ("Marketside Caesar Salad Kit 9.8 oz", "Marketside", "Lettuce", 3.99, 9.8, "oz", 1, False, True),
        ("Great Value Cinnamon Toast Crunch Style Cereal 12 oz", "Great Value", "Cereal", 2.49, 12, "oz", 1, False, True),
        ("Great Value Apple Juice 64 fl oz", "Great Value", "Orange juice", 2.49, 64, "fl oz", 1, False, True),
        ("Great Value Saltine Crackers 16 oz", "Great Value", "Snacks", 1.49, 16, "oz", 1, False, True),
        ("Great Value Whole Milk Mozzarella Shredded 16 oz", "Great Value", "Cheese", 4.49, 16, "oz", 1, False, True),
        ("Great Value Salted Butter 1 lb (4 Sticks)", "Great Value", "Butter", 4.49, 1, "lb", 4, False, True),
        ("Marketside Rotisserie Chicken each", "Marketside", "Chicken", 4.99, 1, "each", 1, False, True),
        ("Great Value Chicken Broth 32 fl oz", "Great Value", "Soup", 1.49, 32, "fl oz", 1, False, True),
        ("Great Value Frozen Chicken Nuggets 32 oz", "Great Value", "Chicken", 7.49, 32, "oz", 1, False, True),
        ("Great Value Hamburger Buns 8 ct", "Great Value", "Bread", 1.49, 8, "ct", 1, False, True),
        ("Marketside Italian Style Pizza 14 inch", "Marketside", "Frozen pizza", 6.49, 14, "oz", 1, False, True),
        ("Great Value Frozen French Fries 32 oz", "Great Value", "Frozen pizza", 2.99, 32, "oz", 1, False, True),
        ("Great Value Whole Almonds 16 oz", "Great Value", "Snacks", 6.99, 16, "oz", 1, False, True),
        ("Great Value Sliced White Bread 20 oz", "Great Value", "Bread", 1.29, 20, "oz", 1, False, True),
        ("Great Value Vanilla Ice Cream 48 fl oz", "Great Value", "Ice cream", 3.49, 48, "fl oz", 1, False, True),
        ("Great Value Granulated Sugar 4 lb", "Great Value", "Pantry", 3.49, 4, "lb", 1, False, True),
        ("Great Value Long Grain Brown Rice 5 lb", "Great Value", "Rice", 5.49, 5, "lb", 1, False, True),
        ("Great Value Pasta Sauce Marinara 24 oz", "Great Value", "Pasta", 1.99, 24, "oz", 1, False, True),
        ("Great Value Chunk Light Tuna in Water 5 oz", "Great Value", "Pantry", 0.89, 5, "oz", 1, False, True),
        ("Great Value Garlic Powder 3.4 oz", "Great Value", "Pantry", 1.99, 3.4, "oz", 1, False, True),
        ("Great Value Frozen Sweet Corn 12 oz", "Great Value", "Frozen pizza", 1.49, 12, "oz", 1, False, True),
        ("Great Value Chocolate Chip Cookies 14.4 oz", "Great Value", "Snacks", 2.49, 14.4, "oz", 1, False, True),
        ("Mountain Dew Soda 2 L", "Mountain Dew", "Soda", 2.49, 2, "liter", 1, False, False),
        ("Marketside Fresh Strawberries 1 lb", "Marketside", "Berries", 3.99, 1, "lb", 1, False, True),
        ("Great Value Vegetable Oil 48 fl oz", "Great Value", "Pantry", 4.49, 48, "fl oz", 1, False, True),
        ("Great Value Lemonade Frozen Concentrate 12 fl oz", "Great Value", "Orange juice", 1.29, 12, "fl oz", 1, False, True),
        ("Equate Vitamin D3 100 ct", "Equate", "Pantry", 4.99, 100, "ct", 1, False, True),
        ("Great Value Sliced American Cheese 24 ct", "Great Value", "Cheese", 4.49, 24, "ct", 1, False, True),
    ],
    "target": [
        ("Good & Gather Frozen Berry Blend 10 oz", "Good & Gather", "Frozen pizza", 4.49, 10, "oz", 1, False, True),
        ("Market Pantry Macaroni & Cheese Original 7.25 oz", "Market Pantry", "Pasta", 0.99, 7.25, "oz", 1, False, True),
        ("Good & Gather Roasted Almonds 16 oz", "Good & Gather", "Snacks", 7.99, 16, "oz", 1, False, True),
        ("Good & Gather Hummus Classic 10 oz", "Good & Gather", "Snacks", 3.49, 10, "oz", 1, False, True),
        ("Good & Gather Greek Yogurt Vanilla 32 oz", "Good & Gather", "Yogurt", 4.99, 32, "oz", 1, False, True),
        ("Good & Gather Sparkling Water Black Cherry 8pk", "Good & Gather", "Sparkling water", 3.49, 12, "fl oz", 8, False, True),
        ("Good & Gather Trail Mix 10 oz", "Good & Gather", "Snacks", 4.99, 10, "oz", 1, False, True),
        ("Good & Gather Frozen Pizza Margherita 12 inch", "Good & Gather", "Frozen pizza", 5.99, 12, "oz", 1, False, True),
        ("Good & Gather Tomato Basil Soup 16 fl oz", "Good & Gather", "Soup", 3.49, 16, "fl oz", 1, False, True),
        ("Market Pantry Salted Butter Quarters 1 lb", "Market Pantry", "Butter", 4.29, 1, "lb", 4, False, True),
        ("Good & Gather Organic Whole Milk 1/2 Gal", "Good & Gather", "Dairy milk", 3.99, 0.5, "gal", 1, True, True),
        ("Archer Farms Coffee Breakfast Blend 12 oz", "Archer Farms", "Coffee", 9.99, 12, "oz", 1, False, True),
        ("Good & Gather Sliced Sourdough Bread 22 oz", "Good & Gather", "Bread", 3.49, 22, "oz", 1, False, True),
        ("Market Pantry Sour Cream 16 oz", "Market Pantry", "Cheese", 1.99, 16, "oz", 1, False, True),
        ("Good & Gather Sliced Pepperoni 5 oz", "Good & Gather", "Beef", 4.99, 5, "oz", 1, False, True),
        ("Good & Gather Frozen Mango Chunks 16 oz", "Good & Gather", "Frozen pizza", 4.99, 16, "oz", 1, False, True),
        ("Good & Gather Cold Pressed Lemonade 14 fl oz", "Good & Gather", "Orange juice", 2.99, 14, "fl oz", 1, False, True),
        ("Good & Gather Whole Bean Coffee Sumatra 12 oz", "Good & Gather", "Coffee", 8.99, 12, "oz", 1, False, True),
        ("Market Pantry White Sandwich Bread 20 oz", "Market Pantry", "Bread", 1.49, 20, "oz", 1, False, True),
        ("Good & Gather Plain Bagels 6 ct", "Good & Gather", "Bread", 2.99, 6, "ct", 1, False, True),
        ("Good & Gather Almond Flour 16 oz", "Good & Gather", "Pantry", 7.99, 16, "oz", 1, False, True),
        ("Method Hand Wash Sea Minerals 12 fl oz", "Method", "Snacks", 4.99, 12, "fl oz", 1, False, False),
        ("Good & Gather Premium Chicken Breast Strips 8 oz", "Good & Gather", "Chicken", 5.99, 8, "oz", 1, False, True),
        ("Good & Gather Maple Syrup Pure 8 fl oz", "Good & Gather", "Condiments", 6.99, 8, "fl oz", 1, False, True),
        ("Good & Gather Pita Chips Sea Salt 8 oz", "Good & Gather", "Chips", 3.49, 8, "oz", 1, False, True),
        ("Good & Gather Frozen Brussels Sprouts 12 oz", "Good & Gather", "Frozen pizza", 2.99, 12, "oz", 1, False, True),
        ("Market Pantry Cinnamon Toast Cereal 17 oz", "Market Pantry", "Cereal", 2.99, 17, "oz", 1, False, True),
        ("Good & Gather Granola Bars Honey Oat 6 ct", "Good & Gather", "Granola bars", 3.49, 1.4, "oz", 6, False, True),
        ("Good & Gather Sliced Black Olives 3.8 oz Can", "Good & Gather", "Pantry", 1.49, 3.8, "oz", 1, False, True),
        ("Good & Gather Whole Wheat Tortillas 8 ct", "Good & Gather", "Bread", 2.49, 8, "ct", 1, False, True),
    ],
}


# ---------------------------------------------------------------------------
# Filler templates — used to top up each retailer to the 200-row floor.
#
# Each entry describes a product type that *each* retailer can sell under
# its own private-label brand (or a generic national brand). When we build
# a filler row we pick the retailer's PL brand and one of the size variants.
# These rows never belong to a match group: their names include retailer-
# specific phrasing so we don't accidentally collide with concept rows.
# ---------------------------------------------------------------------------
FILLER_TEMPLATES = [
    # (label, category, [size variants], typical price range)
    # Each size variant is (descriptor, amount, unit, pack)
    ("All-Purpose Flour",       "Pantry",       [("5 lb", 5, "lb", 1), ("10 lb", 10, "lb", 1), ("2 lb", 2, "lb", 1)], (3.49, 6.49)),
    ("Granulated Sugar",        "Pantry",       [("4 lb", 4, "lb", 1), ("10 lb", 10, "lb", 1)], (3.99, 8.99)),
    ("Brown Sugar",             "Pantry",       [("16 oz", 16, "oz", 1), ("32 oz", 32, "oz", 1)], (1.99, 4.49)),
    ("Baking Soda",             "Pantry",       [("16 oz", 16, "oz", 1), ("8 oz", 8, "oz", 1)], (0.99, 1.99)),
    ("Baking Powder",           "Pantry",       [("8 oz", 8, "oz", 1), ("10 oz", 10, "oz", 1)], (1.99, 3.99)),
    ("Active Dry Yeast",        "Pantry",       [("0.75 oz 3 ct", 0.25, "oz", 3), ("4 oz", 4, "oz", 1)], (1.99, 5.49)),
    ("Vanilla Extract",         "Pantry",       [("2 fl oz", 2, "fl oz", 1), ("4 fl oz", 4, "fl oz", 1)], (3.99, 9.99)),
    ("Ground Cinnamon",         "Pantry",       [("2 oz", 2, "oz", 1), ("3 oz", 3, "oz", 1)], (2.49, 4.99)),
    ("Paprika",                 "Pantry",       [("2 oz", 2, "oz", 1), ("3.5 oz", 3.5, "oz", 1)], (1.99, 3.99)),
    ("Ground Cumin",            "Pantry",       [("2 oz", 2, "oz", 1), ("3 oz", 3, "oz", 1)], (2.49, 4.99)),
    ("Italian Seasoning",       "Pantry",       [("0.75 oz", 0.75, "oz", 1), ("1.5 oz", 1.5, "oz", 1)], (1.49, 3.99)),
    ("Garlic Powder",           "Pantry",       [("3 oz", 3, "oz", 1), ("5 oz", 5, "oz", 1)], (1.99, 4.49)),
    ("Black Pepper Ground",     "Pantry",       [("3 oz", 3, "oz", 1), ("4 oz", 4, "oz", 1)], (2.99, 5.99)),
    ("Iodized Salt",            "Pantry",       [("26 oz", 26, "oz", 1)], (0.79, 1.99)),
    ("Sea Salt",                "Pantry",       [("17.6 oz", 17.6, "oz", 1)], (1.99, 4.49)),
    ("Soy Sauce",               "Condiments",   [("10 fl oz", 10, "fl oz", 1), ("15 fl oz", 15, "fl oz", 1)], (1.99, 4.99)),
    ("Apple Cider Vinegar",     "Condiments",   [("16 fl oz", 16, "fl oz", 1), ("32 fl oz", 32, "fl oz", 1)], (2.49, 5.99)),
    ("White Vinegar",           "Condiments",   [("32 fl oz", 32, "fl oz", 1), ("64 fl oz", 64, "fl oz", 1)], (2.49, 4.99)),
    ("Yellow Mustard",          "Condiments",   [("8 oz", 8, "oz", 1), ("14 oz", 14, "oz", 1)], (0.99, 2.49)),
    ("Dijon Mustard",           "Condiments",   [("8 oz", 8, "oz", 1), ("12 oz", 12, "oz", 1)], (1.99, 4.49)),
    ("BBQ Sauce Original",      "Condiments",   [("18 oz", 18, "oz", 1), ("28 oz", 28, "oz", 1)], (1.99, 4.49)),
    ("Hot Sauce",               "Condiments",   [("5 fl oz", 5, "fl oz", 1), ("12 fl oz", 12, "fl oz", 1)], (1.99, 5.99)),
    ("Sriracha",                "Condiments",   [("17 fl oz", 17, "fl oz", 1), ("28 fl oz", 28, "fl oz", 1)], (4.49, 7.99)),
    ("Strawberry Jam",          "Condiments",   [("12 oz", 12, "oz", 1), ("18 oz", 18, "oz", 1)], (2.49, 4.99)),
    ("Grape Jelly",             "Condiments",   [("18 oz", 18, "oz", 1), ("32 oz", 32, "oz", 1)], (2.49, 4.99)),
    ("Maple Syrup Pure",        "Condiments",   [("8 fl oz", 8, "fl oz", 1), ("12 fl oz", 12, "fl oz", 1)], (5.99, 12.99)),
    ("Pancake Syrup",           "Condiments",   [("24 fl oz", 24, "fl oz", 1)], (2.99, 5.49)),
    ("Olive Oil",               "Condiments",   [("17 fl oz", 17, "fl oz", 1), ("33.8 fl oz", 33.8, "fl oz", 1)], (5.99, 12.99)),
    ("Canola Oil",              "Pantry",       [("48 fl oz", 48, "fl oz", 1)], (3.49, 5.99)),
    ("Vegetable Oil",           "Pantry",       [("48 fl oz", 48, "fl oz", 1)], (3.49, 5.99)),
    ("Chunk Light Tuna in Water", "Pantry",     [("5 oz can", 5, "oz", 1), ("12 oz can", 12, "oz", 1)], (0.79, 2.99)),
    ("Albacore Tuna in Water",  "Pantry",       [("5 oz can", 5, "oz", 1)], (1.99, 3.99)),
    ("Diced Tomatoes",          "Pantry",       [("14.5 oz can", 14.5, "oz", 1), ("28 oz can", 28, "oz", 1)], (0.99, 2.99)),
    ("Tomato Sauce",            "Pantry",       [("8 oz can", 8, "oz", 1), ("15 oz can", 15, "oz", 1)], (0.59, 1.49)),
    ("Tomato Paste",            "Pantry",       [("6 oz can", 6, "oz", 1)], (0.99, 1.99)),
    ("Sweet Corn Canned",       "Pantry",       [("15 oz can", 15, "oz", 1)], (0.79, 1.79)),
    ("Green Beans Canned",      "Pantry",       [("14.5 oz can", 14.5, "oz", 1)], (0.79, 1.79)),
    ("Pinto Beans Canned",      "Canned beans", [("15 oz can", 15, "oz", 1)], (0.79, 1.79)),
    ("Kidney Beans Canned",     "Canned beans", [("15 oz can", 15, "oz", 1)], (0.79, 1.99)),
    ("Garbanzo Beans Canned",   "Canned beans", [("15 oz can", 15, "oz", 1)], (0.99, 2.49)),
    ("Chicken Noodle Soup",     "Soup",         [("10.5 oz can", 10.5, "oz", 1)], (1.49, 2.99)),
    ("Cream of Mushroom Soup",  "Soup",         [("10.5 oz can", 10.5, "oz", 1)], (1.49, 2.49)),
    ("Vegetable Broth",         "Soup",         [("32 fl oz carton", 32, "fl oz", 1)], (2.49, 4.49)),
    ("Chicken Broth",           "Soup",         [("32 fl oz carton", 32, "fl oz", 1)], (2.49, 4.49)),
    ("Penne Pasta",             "Pasta",        [("16 oz", 16, "oz", 1)], (1.49, 2.49)),
    ("Rotini Pasta",            "Pasta",        [("16 oz", 16, "oz", 1)], (1.49, 2.49)),
    ("Elbow Macaroni",          "Pasta",        [("16 oz", 16, "oz", 1)], (1.49, 2.49)),
    ("Lasagna Noodles",         "Pasta",        [("16 oz", 16, "oz", 1)], (1.99, 3.49)),
    ("Marinara Sauce",          "Pasta",        [("24 oz", 24, "oz", 1)], (1.99, 4.49)),
    ("Alfredo Sauce",           "Pasta",        [("15 oz", 15, "oz", 1)], (2.49, 4.49)),
    ("Jasmine Rice",            "Rice",         [("2 lb", 2, "lb", 1), ("5 lb", 5, "lb", 1)], (3.49, 8.99)),
    ("Basmati Rice",            "Rice",         [("2 lb", 2, "lb", 1), ("5 lb", 5, "lb", 1)], (3.99, 9.99)),
    ("Brown Rice",              "Rice",         [("2 lb", 2, "lb", 1), ("5 lb", 5, "lb", 1)], (2.49, 6.99)),
    ("Quinoa",                  "Pantry",       [("12 oz", 12, "oz", 1), ("16 oz", 16, "oz", 1)], (3.99, 7.99)),
    ("Steel Cut Oats",          "Cereal",       [("24 oz", 24, "oz", 1), ("30 oz", 30, "oz", 1)], (3.99, 6.99)),
    ("Instant Oatmeal Variety Pack", "Cereal",  [("12 ct", 12, "ct", 1), ("8 ct", 8, "ct", 1)], (3.49, 5.99)),
    ("Pancake Mix Original",    "Cereal",       [("32 oz", 32, "oz", 1)], (2.49, 4.99)),
    ("Plain Bagels",            "Bread",        [("6 ct", 6, "ct", 1)], (2.49, 4.99)),
    ("Hamburger Buns",          "Bread",        [("8 ct", 8, "ct", 1)], (1.49, 3.49)),
    ("Hot Dog Buns",            "Bread",        [("8 ct", 8, "ct", 1)], (1.49, 3.49)),
    ("Flour Tortillas",         "Bread",        [("10 ct", 10, "ct", 1)], (1.99, 3.99)),
    ("Corn Tortillas",          "Bread",        [("30 ct", 30, "ct", 1)], (2.49, 4.49)),
    ("Pita Bread",              "Bread",        [("6 ct", 6, "ct", 1)], (2.49, 4.49)),
    ("English Muffins",         "Bread",        [("6 ct", 6, "ct", 1)], (1.99, 3.99)),
    ("Half and Half",           "Dairy milk",   [("1 pt", 16, "fl oz", 1), ("1 qt", 32, "fl oz", 1)], (2.49, 4.99)),
    ("Heavy Whipping Cream",    "Dairy milk",   [("1 pt", 16, "fl oz", 1), ("1 qt", 32, "fl oz", 1)], (3.99, 6.99)),
    ("Sour Cream",              "Cheese",       [("8 oz", 8, "oz", 1), ("16 oz", 16, "oz", 1)], (1.49, 3.49)),
    ("Cottage Cheese",          "Cheese",       [("16 oz", 16, "oz", 1), ("24 oz", 24, "oz", 1)], (2.49, 4.99)),
    ("Ricotta Cheese",          "Cheese",       [("15 oz", 15, "oz", 1)], (2.99, 4.99)),
    ("String Cheese Mozzarella","Cheese",       [("12 ct", 12, "ct", 1), ("24 ct", 24, "ct", 1)], (3.99, 7.99)),
    ("Sliced Swiss Cheese",     "Cheese",       [("8 oz", 8, "oz", 1)], (3.49, 5.99)),
    ("Pepper Jack Shredded",    "Cheese",       [("8 oz", 8, "oz", 1)], (3.49, 5.49)),
    ("Plain Whole Milk Yogurt", "Yogurt",       [("32 oz", 32, "oz", 1)], (3.49, 5.99)),
    ("Strawberry Yogurt",       "Yogurt",       [("6 oz", 6, "oz", 1)], (0.79, 1.49)),
    ("Vanilla Almond Granola",  "Cereal",       [("12 oz", 12, "oz", 1), ("16 oz", 16, "oz", 1)], (3.99, 6.99)),
    ("Honey Nut Granola",       "Cereal",       [("12 oz", 12, "oz", 1)], (3.99, 6.99)),
    ("Granola Bars Chewy",      "Granola bars", [("8 ct", 0.84, "oz", 8), ("12 ct", 0.84, "oz", 12)], (2.99, 5.99)),
    ("Pretzels Mini Twists",    "Snacks",       [("16 oz", 16, "oz", 1)], (1.99, 3.99)),
    ("Pretzel Sticks",          "Snacks",       [("15 oz", 15, "oz", 1)], (1.99, 3.99)),
    ("Microwave Popcorn Butter","Snacks",       [("3 ct", 3.2, "oz", 3), ("6 ct", 3.2, "oz", 6)], (2.49, 4.99)),
    ("Mixed Nuts Salted",       "Snacks",       [("8 oz", 8, "oz", 1), ("16 oz", 16, "oz", 1)], (4.99, 9.99)),
    ("Roasted Cashews",         "Snacks",       [("8 oz", 8, "oz", 1), ("16 oz", 16, "oz", 1)], (5.99, 11.99)),
    ("Dried Cranberries",       "Snacks",       [("6 oz", 6, "oz", 1), ("10 oz", 10, "oz", 1)], (2.49, 4.99)),
    ("Raisins Seedless",        "Snacks",       [("12 oz", 12, "oz", 1), ("20 oz", 20, "oz", 1)], (2.49, 5.99)),
    ("Fruit Snacks Variety",    "Snacks",       [("10 ct", 0.8, "oz", 10)], (2.99, 5.49)),
    ("Apple Sauce Cups",        "Snacks",       [("6 ct", 4, "oz", 6), ("12 ct", 4, "oz", 12)], (2.49, 4.99)),
    ("Frozen Mixed Berries",    "Frozen pizza", [("12 oz", 12, "oz", 1), ("48 oz", 48, "oz", 1)], (3.49, 9.99)),
    ("Frozen Strawberries",     "Frozen pizza", [("16 oz", 16, "oz", 1)], (3.49, 5.99)),
    ("Frozen Broccoli",         "Frozen pizza", [("12 oz", 12, "oz", 1)], (1.49, 2.99)),
    ("Frozen Peas",             "Frozen pizza", [("12 oz", 12, "oz", 1)], (1.49, 2.99)),
    ("Frozen Spinach",          "Frozen pizza", [("12 oz", 12, "oz", 1)], (1.49, 2.99)),
    ("Frozen Waffles",          "Frozen pizza", [("10 ct", 1.25, "oz", 10)], (2.49, 4.99)),
    ("Frozen Cheese Pizza",     "Frozen pizza", [("12 inch", 12, "oz", 1)], (3.49, 5.99)),
    ("Frozen Burritos Bean Cheese", "Frozen pizza", [("8 ct", 4, "oz", 8)], (3.99, 6.99)),
    ("Vanilla Ice Cream",       "Ice cream",    [("48 fl oz", 48, "fl oz", 1)], (2.99, 5.49)),
    ("Chocolate Ice Cream",     "Ice cream",    [("48 fl oz", 48, "fl oz", 1)], (2.99, 5.49)),
    ("Cookies & Cream Ice Cream","Ice cream",   [("48 fl oz", 48, "fl oz", 1)], (3.49, 5.99)),
    ("Apple Juice",             "Orange juice", [("64 fl oz", 64, "fl oz", 1)], (2.49, 4.49)),
    ("Cranberry Juice Cocktail","Orange juice", [("64 fl oz", 64, "fl oz", 1)], (2.49, 5.49)),
    ("Lemonade",                "Orange juice", [("64 fl oz", 64, "fl oz", 1)], (1.99, 4.49)),
    ("Iced Tea Black",          "Coffee",       [("128 fl oz", 128, "fl oz", 1)], (2.49, 4.99)),
    ("Black Tea Bags",          "Coffee",       [("100 ct", 100, "ct", 1)], (3.49, 6.99)),
    ("Green Tea Bags",          "Coffee",       [("40 ct", 40, "ct", 1)], (2.99, 5.99)),
    ("Coffee Filters",          "Coffee",       [("100 ct", 100, "ct", 1)], (1.99, 4.49)),
    ("Ground Coffee Medium Roast", "Coffee",    [("12 oz", 12, "oz", 1), ("24 oz", 24, "oz", 1)], (5.99, 12.99)),
    ("Decaf Ground Coffee",     "Coffee",       [("12 oz", 12, "oz", 1)], (6.99, 11.99)),
    ("K-Cup Coffee Pods",       "Coffee",       [("18 ct", 0.4, "oz", 18), ("24 ct", 0.4, "oz", 24)], (8.99, 16.99)),
]


# Some retailers sometimes don't sell every kind of item. We use a dict of
# "brand to use" per retailer for the filler. Half the time we use the
# retailer's private label, half the time we use a noisy generic name.
def _retailer_filler_brand(retailer):
    """Pick a private-label brand for the retailer to badge a filler item."""
    return random.choice(PRIVATE_LABELS[retailer])


def _retailer_name_style(retailer, brand, base, size_str):
    """Render the filler name in the retailer's typical voice."""
    if retailer == "whole_foods":
        return f"{brand} {base} {size_str}"
    if retailer == "kroger":
        return f"{brand} {base}, {size_str}"
    if retailer == "walmart":
        return f"{brand} {base}, {size_str}"
    # target uses dashes
    return f"{brand} {base} - {size_str}"


def build_filler_row(retailer, template, variant_idx):
    """Build one filler row from a template + which size variant to use."""
    base, category, sizes, price_range = template
    size_str, amount, unit, pack = sizes[variant_idx % len(sizes)]
    brand = _retailer_filler_brand(retailer)
    name = _retailer_name_style(retailer, brand, base, size_str)
    base_price = random.uniform(*price_range)
    return {
        "name": name, "brand": brand, "category": category,
        "amount": amount, "unit": unit, "pack": pack,
        "organic": False, "pl": True, "base_price": base_price,
    }


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------
class IDCounter:
    """Generates retailer-prefixed sequential IDs like wf_0001, kr_0001, ..."""
    def __init__(self):
        self.counters = {r: 0 for r in RETAILERS}

    def next_id(self, retailer):
        self.counters[retailer] += 1
        return f"{RETAILER_PREFIX[retailer]}_{self.counters[retailer]:04d}"


def build_row(pid, retailer, raw_name, brand, category_canonical,
              amount, unit, pack_count, organic, private_label,
              base_price, ts_idx, promo=None,
              ocr_noise=False, force_promo_in_name=False):
    """Construct one CSV row dict."""
    # Resolve retailer-specific category text.
    cat_map = CATEGORY_NAMES.get(category_canonical)
    if cat_map is None:
        category_raw = category_canonical
    else:
        category_raw = cat_map[retailer]

    # Optionally inject promo text directly into raw_name (and keep promo_text).
    if promo is None:
        promo = random.choice(PROMO_POOL_BY_RETAILER[retailer])
    name = raw_name
    if promo and (force_promo_in_name or random.random() < 0.25):
        name = maybe_inject_promo_into_name(name, promo)

    # OCR-style corruption on a minority of rows.
    if ocr_noise:
        name = apply_ocr_noise(name)

    price = random_price(base_price * 0.95, base_price * 1.05, retailer)

    # Unit price field: we leave it blank for many rows (mimics real data).
    if random.random() < 0.55 and isinstance(amount, (int, float)) and amount and price:
        # very rough per-unit price for liquids/weights
        denom = amount * (pack_count or 1)
        if denom:
            unit_price = round(price / denom, 4)
        else:
            unit_price = ""
    else:
        unit_price = ""

    return {
        "product_id": pid,
        "retailer": retailer,
        "raw_name": name,
        "brand": brand,
        "category_raw": category_raw,
        "price": price,
        "unit_price": unit_price,
        "promo_text": promo,
        "amount": amount,
        "unit": unit,
        "pack_count": pack_count,
        "is_organic": bool(organic),
        "is_private_label": bool(private_label),
        "url": make_url(retailer, pid),
        "scraped_at": make_timestamp(ts_idx),
        "estimated_cost": estimated_cost_for(price),
    }


def generate_dataset():
    """Main generation routine: returns rows-by-retailer + label structures."""
    ids = IDCounter()
    rows_by_retailer = {r: [] for r in RETAILERS}
    ground_truth_groups = []     # for ground_truth_matches.json
    concept_to_pids = {}          # concept_id -> {retailer: pid}
    negative_pairs = []
    substitute_pairs = []
    ts_idx = 0

    # OCR noise: target ~5-10% of rows. Pick a fraction up-front.
    total_planned = sum(len(c["retailers"]) for c in CONCEPTS)
    total_planned += sum(len(v) for v in UNIQUE_PRODUCTS.values())
    total_planned += len(NEAR_MISS_RULES)
    ocr_target = int(total_planned * 0.08)
    ocr_picked = 0

    def maybe_ocr():
        nonlocal ocr_picked
        if ocr_picked < ocr_target and random.random() < 0.10:
            ocr_picked += 1
            return True
        return False

    # ----- PASS 1: concepts (positive match groups) -----
    for concept in CONCEPTS:
        cid = concept["concept_id"]
        category = concept["category"]
        match_type = concept["match_type"]
        base_price = concept["base_price"]

        mappings = {}
        for retailer, info in concept["retailers"].items():
            pid = ids.next_id(retailer)
            ts_idx += 1
            row = build_row(
                pid=pid, retailer=retailer,
                raw_name=info["name"], brand=info["brand"],
                category_canonical=category,
                amount=info["amount"], unit=info["unit"],
                pack_count=info["pack"],
                organic=info["organic"], private_label=info["pl"],
                base_price=base_price, ts_idx=ts_idx,
                ocr_noise=maybe_ocr(),
            )
            rows_by_retailer[retailer].append(row)
            mappings[retailer] = pid

        concept_to_pids[cid] = mappings

        # Build the ground-truth group (skip if only one retailer carries it).
        if len(mappings) >= 2:
            ground_truth_groups.append({
                "match_group_id": cid,
                "match_type": match_type,
                "canonical_description": concept_canonical_description(concept),
                "mappings": mappings,
            })

    # ----- PASS 2: near-miss rows -> negative pairs -----
    for source_cid, retailer, mutation, reason in NEAR_MISS_RULES:
        if source_cid not in concept_to_pids:
            continue
        # Find a sample retailer info we can mutate (use any retailer that has
        # the concept; prefer the same retailer we want to add the near-miss
        # row to, so amounts/styles look natural).
        source_concept = next((c for c in CONCEPTS if c["concept_id"] == source_cid), None)
        if source_concept is None:
            continue
        # Pick the retailer's info if present, else first available.
        if retailer in source_concept["retailers"]:
            base_info = source_concept["retailers"][retailer]
        else:
            base_info = next(iter(source_concept["retailers"].values()))

        mutated = mutation(base_info)

        pid = ids.next_id(retailer)
        ts_idx += 1
        row = build_row(
            pid=pid, retailer=retailer,
            raw_name=mutated["name"], brand=mutated["brand"],
            category_canonical=source_concept["category"],
            amount=mutated["amount"], unit=mutated["unit"],
            pack_count=mutated["pack"],
            organic=mutated["organic"], private_label=mutated["pl"],
            base_price=source_concept["base_price"], ts_idx=ts_idx,
            ocr_noise=maybe_ocr(),
        )
        rows_by_retailer[retailer].append(row)

        # Pair this near-miss row against every retailer's row in the source
        # match group → these are non-matches.
        for other_retailer, other_pid in concept_to_pids[source_cid].items():
            if other_pid == pid:
                continue
            negative_pairs.append({
                "product_id_a": pid,
                "product_id_b": other_pid,
                "label": "non_match",
                "reason": reason,
            })

    # ----- PASS 3: hard-negative cross-concept pairs -----
    # Same broad area but different product / brand / variant → hard negatives.
    hard_negative_concept_pairs = [
        ("coke_zero_12pk_12oz", "diet_coke_12pk_12oz",
         "same brand family, different variant (Coke Zero vs Diet Coke)"),
        ("coke_zero_12pk_12oz", "pepsi_12pk_12oz",
         "same category and pack size, different brand (Coke vs Pepsi)"),
        ("lacroix_lime_8pk", "lacroix_grapefruit_8pk",
         "same brand and pack, different flavor"),
        ("chobani_blueberry_5p3oz", "chobani_strawberry_5p3oz",
         "same brand and size, different flavor"),
        ("organic_whole_milk_1gal_pl", "whole_milk_1gal_pl",
         "same size and form factor, organic vs non-organic"),
        ("unsweet_almond_milk_64floz_pl", "oat_milk_64floz_pl",
         "same form factor, different plant base (almond vs oat)"),
        ("oj_no_pulp_52floz_pl", "tropicana_pure_premium_52floz",
         "same product type and size; one is private-label, one national"),
        ("frozen_pepperoni_pizza_pl", "digiorno_pepperoni",
         "same product type, different brand and size"),
        ("cheerios_orig_12oz", "cheerios_honey_nut_10p8oz",
         "same brand, different variant and size"),
        ("driscolls_blueberries_6oz", "driscolls_strawberries_1lb",
         "same brand, different berry"),
        ("lacroix_lime_8pk", "lacroix_lime_12pk",
         "same brand same flavor, different pack size"),
        ("coke_zero_12pk_12oz", "coke_zero_6pk_12oz",
         "same brand same variant, different pack size"),
        ("organic_gala_apples_3lb_pl", "bulk_organic_fuji_apples",
         "both organic apples but different variety and packaging"),
        ("plain_greek_yogurt_32oz_pl", "chobani_blueberry_5p3oz",
         "same category (yogurt) but different brand, flavor, and size"),
        ("creamy_pb_16oz_pl", "skippy_creamy_pb_16p3oz",
         "same product type and size, different brand"),
    ]
    for a_cid, b_cid, reason in hard_negative_concept_pairs:
        if a_cid not in concept_to_pids or b_cid not in concept_to_pids:
            continue
        # Emit several cross-retailer pairs per concept pair (cap at 4) so we
        # accumulate a healthy negative-pair count.
        emitted = 0
        for ra, pa in concept_to_pids[a_cid].items():
            for rb, pb in concept_to_pids[b_cid].items():
                if ra == rb:  # cross-retailer is the most useful test case
                    continue
                negative_pairs.append({
                    "product_id_a": pa,
                    "product_id_b": pb,
                    "label": "non_match",
                    "reason": reason,
                })
                emitted += 1
                if emitted >= 4:
                    break
            if emitted >= 4:
                break

    # ----- PASS 4: substitute pairs -----
    for a_cid, b_cid, reason in SUBSTITUTE_CONCEPT_PAIRS:
        if a_cid not in concept_to_pids or b_cid not in concept_to_pids:
            continue
        # Emit cross-retailer substitute pairs (cap at 4 per concept pair).
        a_items = list(concept_to_pids[a_cid].items())
        b_items = list(concept_to_pids[b_cid].items())
        random.shuffle(a_items)
        random.shuffle(b_items)
        emitted = 0
        for ra, pa in a_items:
            for rb, pb in b_items:
                if pa == pb:
                    continue
                substitute_pairs.append({
                    "product_id_a": pa,
                    "product_id_b": pb,
                    "label": "substitute",
                    "reason": reason,
                })
                emitted += 1
                if emitted >= 4:
                    break
            if emitted >= 4:
                break

    # ----- PASS 5: unique unmatched products -----
    for retailer, items in UNIQUE_PRODUCTS.items():
        for tup in items:
            (name, brand, category, base_price,
             amount, unit, pack, organic, pl) = tup
            pid = ids.next_id(retailer)
            ts_idx += 1
            row = build_row(
                pid=pid, retailer=retailer,
                raw_name=name, brand=brand,
                category_canonical=category,
                amount=amount, unit=unit, pack_count=pack,
                organic=organic, private_label=pl,
                base_price=base_price, ts_idx=ts_idx,
                ocr_noise=maybe_ocr(),
            )
            rows_by_retailer[retailer].append(row)

    # ----- PASS 6: filler unique products to reach the 200-row floor -----
    # We programmatically generate additional unique products per retailer
    # using a fixed pool of item templates × size variants. Each filler row
    # belongs to ONE retailer only — none of them are part of a match group.
    for retailer in RETAILERS:
        target_rows = 215  # comfortably inside [200, 300]
        deficit = max(0, target_rows - len(rows_by_retailer[retailer]))
        for i in range(deficit):
            template = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
            variant = i // len(FILLER_TEMPLATES)  # rotate sizes/flavors
            row_data = build_filler_row(retailer, template, variant)
            pid = ids.next_id(retailer)
            ts_idx += 1
            row = build_row(
                pid=pid, retailer=retailer,
                raw_name=row_data["name"], brand=row_data["brand"],
                category_canonical=row_data["category"],
                amount=row_data["amount"], unit=row_data["unit"],
                pack_count=row_data["pack"],
                organic=row_data["organic"], private_label=row_data["pl"],
                base_price=row_data["base_price"], ts_idx=ts_idx,
                ocr_noise=maybe_ocr(),
            )
            rows_by_retailer[retailer].append(row)

    return {
        "rows_by_retailer": rows_by_retailer,
        "ground_truth_groups": ground_truth_groups,
        "negative_pairs": negative_pairs,
        "substitute_pairs": substitute_pairs,
        "concept_to_pids": concept_to_pids,
    }


def concept_canonical_description(concept):
    """Build a short human-readable canonical description from concept info."""
    sample = next(iter(concept["retailers"].values()))
    bits = []
    if sample.get("organic"):
        bits.append("Organic")
    bits.append(sample["name"].split(",")[0])  # first comma chunk is usually the core name
    bits.append(f"{sample['amount']} {sample['unit']}")
    if sample.get("pack", 1) and sample["pack"] > 1:
        bits.append(f"{sample['pack']}-pack")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Validation & summary
# ---------------------------------------------------------------------------
def validate(result):
    """Sanity-check the dataset before writing anything."""
    rows = result["rows_by_retailer"]
    all_pids = []
    for r in RETAILERS:
        for row in rows[r]:
            all_pids.append(row["product_id"])

    # 1. Globally unique product_ids.
    assert len(all_pids) == len(set(all_pids)), "Duplicate product_id values exist"

    # 2. Each CSV row count in [200, 300].
    for r in RETAILERS:
        n = len(rows[r])
        assert 200 <= n <= 300, f"{r} has {n} rows; must be in [200,300]"

    # 3. All product IDs referenced in labels actually exist.
    pid_set = set(all_pids)
    for grp in result["ground_truth_groups"]:
        assert len(grp["mappings"]) >= 2, f"Group {grp['match_group_id']} has fewer than 2 retailers"
        for retailer, pid in grp["mappings"].items():
            assert pid in pid_set, f"Group {grp['match_group_id']} references missing pid {pid}"

    pos_pair_set = set()
    for grp in result["ground_truth_groups"]:
        pids = list(grp["mappings"].values())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pos_pair_set.add(frozenset((pids[i], pids[j])))

    # 4. Negative pairs do not duplicate positive match pairs.
    for np_ in result["negative_pairs"]:
        assert np_["product_id_a"] in pid_set, f"Missing pid in negative pair: {np_}"
        assert np_["product_id_b"] in pid_set, f"Missing pid in negative pair: {np_}"
        assert frozenset((np_["product_id_a"], np_["product_id_b"])) not in pos_pair_set, \
            f"Negative pair duplicates a positive match: {np_}"

    # 5. Substitute pairs do not duplicate exact positive pairs (private-label
    # equivalents *can* overlap conceptually, but our data structure declares
    # substitute pairs only between concepts that are NOT in the same group).
    for sp_ in result["substitute_pairs"]:
        assert frozenset((sp_["product_id_a"], sp_["product_id_b"])) not in pos_pair_set, \
            f"Substitute pair duplicates a positive match: {sp_}"

    # 6. estimated_cost < price for the majority of rows.
    cheaper = 0
    total = 0
    for r in RETAILERS:
        for row in rows[r]:
            total += 1
            if row["estimated_cost"] < row["price"]:
                cheaper += 1
    assert cheaper / total >= 0.85, \
        f"Only {cheaper}/{total} rows have estimated_cost < price (target >= 85%)"


def print_summary(result):
    rows = result["rows_by_retailer"]
    print("=" * 70)
    print("RetailGraph synthetic dataset — generation summary")
    print("=" * 70)

    # Rows per retailer.
    print("\nRows per retailer:")
    for r in RETAILERS:
        print(f"  {r:13s} {len(rows[r])} rows")

    # Label counts.
    print("\nLabels:")
    print(f"  positive match groups : {len(result['ground_truth_groups'])}")
    print(f"  negative pairs        : {len(result['negative_pairs'])}")
    print(f"  substitute pairs      : {len(result['substitute_pairs'])}")

    # Match-type breakdown.
    by_type = {}
    for g in result["ground_truth_groups"]:
        by_type[g["match_type"]] = by_type.get(g["match_type"], 0) + 1
    print("\nMatch-type breakdown:")
    for t, c in by_type.items():
        print(f"  {t:30s} {c}")

    # Counts of organic / private-label / promo / OCR rows.
    organic = pl = promoed = ocr_noisy = 0
    OCR_MARKERS = ("0", "$", "I")  # rough heuristic
    for r in RETAILERS:
        for row in rows[r]:
            if row["is_organic"]:
                organic += 1
            if row["is_private_label"]:
                pl += 1
            if row["promo_text"]:
                promoed += 1
            # OCR-noisy heuristic: name has digits in places they shouldn't be
            # plus dropped vowels. We don't track this perfectly; report a
            # rough estimate by checking for "0" inside what should be a word.
            name = row["raw_name"]
            if any(m in name for m in OCR_MARKERS) and " " in name:
                # only count when it's NOT a normal numeric like "12 oz"
                # (heuristic: presence of 'C0' or '0r' style markers)
                if "C0" in name or "0r" in name or "I" in name.split(" ")[0]:
                    ocr_noisy += 1

    print("\nRow content:")
    print(f"  organic items         : {organic}")
    print(f"  private-label items   : {pl}")
    print(f"  rows with promo_text  : {promoed}")
    print(f"  approx. OCR-noisy rows: {ocr_noisy}")

    # Category distribution per retailer.
    print("\nCategory distribution per retailer:")
    for r in RETAILERS:
        cats = {}
        for row in rows[r]:
            cats[row["category_raw"]] = cats.get(row["category_raw"], 0) + 1
        print(f"  {r}:")
        for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"      {c:25s} {n}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "product_id", "retailer", "raw_name", "brand", "category_raw",
    "price", "unit_price", "promo_text", "amount", "unit", "pack_count",
    "is_organic", "is_private_label", "url", "scraped_at", "estimated_cost",
]


def write_outputs(result):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for r in RETAILERS:
        path = RAW_DIR / f"{r}.csv"
        df = pd.DataFrame(result["rows_by_retailer"][r], columns=CSV_FIELDS)
        df.to_csv(path, index=False)

    with open(LABELS_DIR / "ground_truth_matches.json", "w", encoding="utf-8") as f:
        json.dump(result["ground_truth_groups"], f, indent=2, ensure_ascii=False)

    with open(LABELS_DIR / "negative_pairs.json", "w", encoding="utf-8") as f:
        json.dump(result["negative_pairs"], f, indent=2, ensure_ascii=False)

    with open(LABELS_DIR / "substitute_pairs.json", "w", encoding="utf-8") as f:
        json.dump(result["substitute_pairs"], f, indent=2, ensure_ascii=False)


README_TEMPLATE = """# RetailGraph Synthetic Dataset

This directory holds the synthetic-but-realistic grocery dataset used by the
RetailGraph entity-resolution pipeline. The files were generated by
`backend/generate_dataset.py` and are reproducible from a fixed random seed.

## Files

```
data/raw/whole_foods.csv
data/raw/kroger.csv
data/raw/walmart.csv
data/raw/target.csv
data/labels/ground_truth_matches.json
data/labels/negative_pairs.json
data/labels/substitute_pairs.json
```

## How it was generated

1. The script defines a list of **canonical product concepts** — the real
   products that exist in the world (e.g., "Coca-Cola Zero Sugar 12-pack of
   12 fl oz cans").
2. For every concept and every retailer that carries it, the script generates
   a *retailer-specific row* with messy distortions: different naming style,
   different unit format (1 gal vs 128 fl oz), different category labels,
   private-label rebranding, optional OCR-style noise, optional promo text.
3. Because the generator knows which rows came from the same concept, it
   writes the ground-truth labels in the same pass — so the answer key is
   always consistent with the data.
4. After concept rows, the script adds:
   - **near-miss rows** (different size / pack / organic-status / flavor) to
     test that the matcher does not over-merge,
   - **substitute pairs** (private-label vs national-brand of the same item),
   - **unique unmatched products** that exist at only one retailer.

## Hard cases included

The dataset is designed to exercise these specific normalization challenges:

- **Same product, different naming.** "Coca Cola Zero Sugar 12 x 12 fl oz" vs
  "Coke Zero 12pk 12 oz cans" vs "Coca-Cola Zero Sugar Soda 12-Pack 12 fl oz".
- **Same product, different unit notation.** "1 gal" vs "128 fl oz", "0.5 gal"
  vs "64 fl oz", "1 lb" vs "16 oz", "Dozen" vs "12 ct", "6 pack" vs "6pk".
- **Private-label equivalence.** 365 / Simple Truth / Great Value / Good &
  Gather all sell organic whole milk, organic eggs, peanut butter, etc. These
  are tagged `private_label_equivalent`, **not** `exact_equivalent`.
- **Produce ambiguity.** Bulk loose apples vs 3-lb bagged apples, loose
  avocados vs 4-count bag, 6 oz blueberries vs 18 oz family pack, lettuce head
  vs chopped lettuce bag.
- **Pack-size differences.** Coke Zero 6pk vs 12pk. LaCroix 8pk vs 12pk.
  Chobani 5.3 oz single vs 4-pack of 5.3 oz cups.
- **Promotion noise.** Some `raw_name` values include things like
  "BOGO 50% off", "2 for $5", "Rollback" mixed with the product description.
- **OCR / noisy text.** A minority of rows (about 5–10%) contain
  intentional character corruption: "C0ca C0la", "Org Whle Milk 1GAL".
- **Substitute but not equivalent.** Private-label almond milk vs Silk;
  private-label sparkling water vs LaCroix; private-label cereal vs Cheerios.
  These live in `substitute_pairs.json` rather than the exact-match labels.

## Why the dataset is useful for testing entity resolution

A real-world grocery normalization pipeline has to:

1. parse messy free-text product names into structured fields (brand, item,
   variant, size, pack count, organic flag),
2. unify units of measure (gal vs fl oz, lb vs oz, dozen vs 12 ct),
3. recognize that a retailer's private label is *equivalent* to another
   retailer's private label when the underlying product is the same, but
4. NOT match a 6-pack to a 12-pack, or organic to non-organic, or unsweetened
   to sweetened.

This dataset gives you the labels to measure all of those at once: you can
compute precision / recall on `ground_truth_matches.json` for exact and
private-label equivalences, and use `negative_pairs.json` for hard negatives
that should never be merged. `substitute_pairs.json` powers the
"substitute but not equivalent" capability that pricing-intelligence systems
use to suggest a comparable product when an exact match isn't available.

## Choosing the base store at runtime

The dataset deliberately does **not** designate any retailer as "my store".
Every retailer row has its own `estimated_cost` field, so the application can
later let the user pick any of the four retailers as the base store and
compute price comparisons against the other three.

## Known limitations

- Prices, costs, and timestamps are synthetic and not based on real retailer
  data. They are realistic in *range* but not in any specific value.
- The OCR-noise simulator is intentionally simple — character substitution
  plus the occasional dropped vowel. Real OCR mistakes can be more varied.
- The catalog is curated for hard cases, so it leans heavily on a few
  product categories (dairy, soda, snacks, produce, condiments). Long-tail
  categories (international foods, baby products, household, beauty) are
  under-represented or absent.
- Promo text is a single fixed string per row; we do not model time-windowed
  promotions or store-specific deals.
- Brand normalization is not perfect inside the dataset itself (e.g., one row
  may say "Coke" and another "Coca-Cola"); that is intentional, since the
  point is to test a normalization pipeline.

## Reproducibility

The script seeds Python's `random` and NumPy with the value 42, so the
dataset is fully reproducible. To regenerate:

```bash
python backend/generate_dataset.py
```
"""


def write_readme():
    README_PATH.write_text(README_TEMPLATE, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Generating RetailGraph dataset...")
    result = generate_dataset()
    validate(result)
    write_outputs(result)
    write_readme()
    print_summary(result)
    print(f"\nWrote files under: {RAW_DIR.parent}")


if __name__ == "__main__":
    main()
