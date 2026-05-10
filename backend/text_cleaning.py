"""Generic raw-text cleanup helpers for RetailGraph normalization.

This module deliberately stays below the taxonomy layer. It does not know
grocery ontology and does not decide brand, product type, category, matching,
or substitution. Its job is only to make noisy raw text more regular so later
deterministic taxonomy code can work with cleaner input.
"""

from __future__ import annotations

import re
import unicodedata


def _normalize_unicode(text: str) -> str:
    """Convert accented Unicode text to a simpler ASCII-friendly form."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def _preserve_numeric_punctuation(text: str) -> str:
    """Normalize punctuation while preserving decimals, percents, and fractions.

    This is intentionally conservative. Quantity-bearing patterns like `5.3`,
    `2%`, and `1/2` should survive cleanup because downstream normalization
    still needs them.
    """
    text = text.replace("&", " and ")

    chars: list[str] = []
    length = len(text)
    for index, char in enumerate(text):
        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < length else ""

        if char == ".":
            if previous_char.isdigit() and next_char.isdigit():
                chars.append(char)
            else:
                chars.append(" ")
            continue

        if char == "%":
            if previous_char.isdigit():
                chars.append(char)
            else:
                chars.append(" ")
            continue

        if char == "/":
            if previous_char.isdigit() and next_char.isdigit():
                chars.append(char)
            else:
                chars.append(" ")
            continue

        if char.isalnum() or char.isspace():
            chars.append(char)
        else:
            chars.append(" ")

    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _replace_ocr_inside_words(text: str) -> str:
    """Repair OCR digit/symbol confusions only inside alphabetic word contexts.

    The key constraint is avoiding global replacements such as `0 -> o` or
    `1 -> l`, which would corrupt real quantities.
    """
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)
    text = re.sub(r"(?<=\b)0(?=[a-z])", "o", text)
    text = re.sub(r"(?<=[a-z])0(?=\b)", "o", text)
    text = re.sub(r"(?<=[a-z])1(?=[a-z])", "i", text)
    text = re.sub(r"(?<=[a-z])5(?=[a-z])", "s", text)
    text = re.sub(r"(?<=[a-z])4(?=[a-z])", "a", text)
    text = re.sub(r"(?<=[a-z])\$(?=[a-z])", "s", text)
    text = re.sub(r"(?<=[a-z])@(?=[a-z])", "a", text)
    return text


def _normalize_unit_ocr(text: str) -> str:
    """Fix common OCR errors only in measurement/unit contexts.

    These rules are narrow on purpose so numeric quantities like `1 gal` or
    `4 pack` remain numeric instead of being rewritten as letters.
    """
    text = re.sub(r"(\b\d+\s+)0z\b", r"\1oz", text)
    text = re.sub(r"(\b\d+\s+)f[1i]\s*oz\b", r"\1fl oz", text)
    text = re.sub(r"(\b\d+\s+)fi\s*oz\b", r"\1fl oz", text)
    text = re.sub(r"(\b\d+\s+)fioz\b", r"\1floz", text)
    text = re.sub(r"(\b\d+\s+)f1oz\b", r"\1floz", text)
    text = re.sub(r"(\b\d+\s+)ga1\b", r"\1gal", text)
    text = re.sub(r"(\b\d+\s+)1b\b", r"\1lb", text)
    return text


def _normalize_pack_patterns(text: str) -> str:
    """Standardize pack/count expressions without removing quantity evidence."""
    text = re.sub(r"\b(\d+)\s*[- ]?\s*pk\b", r"\1 pack", text)
    text = re.sub(r"\b(\d+)\s*-\s*pack\b", r"\1 pack", text)
    text = re.sub(r"\b(\d+)\s+pack\b", r"\1 pack", text)
    text = re.sub(r"\b(\d+)\s*[- ]?\s*ct\b", r"\1 count", text)
    text = re.sub(r"\b(\d+)\s*-\s*count\b", r"\1 count", text)
    text = re.sub(r"\b(\d+)\s*x\s*(\d+(?:\.\d+)?)\b", r"\1 pack \2", text)
    return text


def clean_product_text(text: str | None) -> str:
    """Return conservatively cleaned text suitable for taxonomy lookup."""
    if text is None:
        return ""

    cleaned = _normalize_unicode(str(text))
    cleaned = _preserve_numeric_punctuation(cleaned)
    cleaned = _replace_ocr_inside_words(cleaned)
    cleaned = _normalize_unit_ocr(cleaned)
    cleaned = _normalize_pack_patterns(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


if __name__ == "__main__":
    test_cases = [
        {
            "input": "C0ca C0la Zer0 Sugar 12pk 12 fl 0z Cans",
            "expected_contains": ["coca cola zero sugar 12 pack 12 fl oz cans"],
        },
        {
            "input": "S1lk Unsweetened Almondmilk 64 f1 oz",
            "expected_contains": ["silk unsweetened almondmilk 64 fl oz"],
        },
        {
            "input": "OatIy Original Oatmilk 64 fl oz",
            "expected_contains": ["oatiy", "original", "oatmilk", "64 fl oz"],
            "note": "Uppercase I vs lowercase l ambiguity is not fixed here; fuzzy brand resolution can handle that later.",
        },
        {
            "input": "Organic Gala Apples 3 1b Bag",
            "expected_contains": ["organic gala apples 3 lb bag"],
        },
        {
            "input": "Häagen-Dazs Vanilla Ice Cream 14 fl oz",
            "expected_contains": ["haagen dazs vanilla ice cream 14 fl oz"],
        },
        {
            "input": "2% Reduced Fat Milk 1/2 Gallon",
            "expected_contains": ["2%", "1/2"],
        },
    ]

    for index, case in enumerate(test_cases, start=1):
        cleaned = clean_product_text(case["input"])
        print(f"\nTest case {index}")
        print("input:", case["input"])
        print("cleaned:", cleaned)
        if "note" in case:
            print("note:", case["note"])
        for expected in case["expected_contains"]:
            assert expected in cleaned, f"Expected substring {expected!r} in {cleaned!r}"
