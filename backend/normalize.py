"""Taxonomy-driven normalization for RetailGraph retailer product rows.

This script is the first real pipeline step after raw CSV ingestion. It turns
retailer-specific observations into a structured canonical representation that
later stages can use for blocking, matching, graph construction, pricing, and
review workflows.

The important boundary: this file does not decide whether two products match.
It only extracts deterministic normalized facts from each row and records
confidence/validation metadata. Entity resolution belongs in later modules.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from taxonomy import (
    canonicalize_brand,
    convert_to_canonical_quantity,
    estimate_taxonomy_confidence,
    extract_category_attributes,
    get_family_for_product_type,
    infer_package_type,
    infer_product_type,
    is_private_label,
    normalize_raw_category,
    normalize_text,
    validate_required_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

RAW_INPUT_FILES = [
    RAW_DATA_DIR / "whole_foods.csv",
    RAW_DATA_DIR / "kroger.csv",
    RAW_DATA_DIR / "walmart.csv",
    RAW_DATA_DIR / "target.csv",
]

NORMALIZED_PRODUCTS_PATH = PROCESSED_DATA_DIR / "normalized_products.csv"
NORMALIZATION_REPORT_PATH = OUTPUTS_DIR / "normalization_report.json"

OUTPUT_COLUMNS = [
    "product_id",
    "retailer",
    "raw_name",
    "brand_raw",
    "canonical_brand",
    "category_raw",
    "category_family_from_raw",
    "product_type",
    "category_family",
    "price",
    "unit_price",
    "promo_text",
    "amount",
    "unit",
    "pack_count",
    "size_value",
    "size_unit",
    "unit_quantity",
    "total_quantity",
    "quantity_type",
    "canonical_unit",
    "package_type",
    "is_organic_raw",
    "is_private_label_raw",
    "is_private_label_detected",
    "attributes_json",
    "confidence_json",
    "overall_confidence",
    "needs_review",
    "missing_required_fields_json",
    "warnings_json",
    "url",
    "scraped_at",
    "estimated_cost",
]


def _is_missing(value: Any) -> bool:
    """Treat pandas NaN, None, and blank strings as missing."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _clean_scalar(value: Any) -> Any:
    """Convert pandas missing values to None so JSON/CSV output is predictable."""
    return None if _is_missing(value) else value


def _to_bool(value: Any) -> bool | None:
    """Normalize raw boolean-ish CSV values without assuming one retailer format."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)

    cleaned = normalize_text(str(value))
    if cleaned in {"true", "t", "yes", "y", "1"}:
        return True
    if cleaned in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _json_dumps(value: Any) -> str:
    """Serialize nested metadata with stable key ordering for diff-friendly output."""
    return json.dumps(value, sort_keys=True)


def _coerce_number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_pack_count(value: Any) -> int:
    if _is_missing(value):
        return 1
    try:
        pack_count = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(pack_count, 1)


def _extract_pack_count_from_text(raw_name: str) -> int | None:
    """Find simple pack-count patterns such as 12pk, 12 pack, or 6 ct."""
    cleaned = normalize_text(raw_name)
    match = re.search(r"\b(\d+)\s*(pk|pack|ct|count)\b", cleaned)
    if not match:
        return None
    return max(int(match.group(1)), 1)


def _parse_amount_token(token: str) -> float:
    """Parse simple decimal or fraction amount tokens from product text."""
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return float(numerator) / float(denominator)
    return float(token)


def _extract_quantity_from_text(raw_name: str) -> tuple[float | None, str | None, int | None]:
    """Extract quantity only as a fallback when amount/unit are absent in the CSV.

    The raw CSV should remain the primary source because retailer feeds often
    contain cleaner structured quantity data than product titles. This fallback
    handles common cases like "1 Gallon", "1/2 Gallon", "64 fl oz", and
    "12pk 12 fl oz Cans" without trying to become a full parser.
    """
    cleaned = normalize_text(raw_name)
    pack_count = _extract_pack_count_from_text(cleaned)

    half_match = re.search(r"\bhalf\s+(gallon|gal)\b", cleaned)
    if half_match:
        return 0.5, half_match.group(1), pack_count

    unit_pattern = (
        r"fl\s*oz|floz|fluid\s+ounces?|oz|ounces?|lb|lbs|pounds?|"
        r"gal|gallons?|liter|litre|l|ct|count|each|dozen"
    )
    matches = list(re.finditer(rf"\b(\d+(?:\.\d+)?|\d+/\d+)\s*({unit_pattern})\b", cleaned))
    if not matches:
        return None, None, pack_count

    # In a multipack title, the useful per-item size is usually the last size
    # expression: "12pk 12 fl oz" should use amount=12 and pack_count=12.
    match = matches[-1]
    return _parse_amount_token(match.group(1)), match.group(2), pack_count


def _choose_quantity_inputs(row: pd.Series) -> tuple[float | None, str | None, int]:
    """Prefer structured CSV quantity fields, then fall back to raw-name regex."""
    amount = _coerce_number(row.get("amount"))
    unit = None if _is_missing(row.get("unit")) else str(row.get("unit"))
    pack_count = _coerce_pack_count(row.get("pack_count"))

    if amount is not None and unit:
        return amount, unit, pack_count

    fallback_amount, fallback_unit, fallback_pack_count = _extract_quantity_from_text(str(row.get("raw_name", "")))
    return (
        amount if amount is not None else fallback_amount,
        unit if unit else fallback_unit,
        pack_count if pack_count != 1 else (fallback_pack_count or 1),
    )


def _choose_category_family(product_type: str, category_family_from_raw: str) -> str:
    """Product type is more specific than raw category, so it wins when known."""
    product_type_family = get_family_for_product_type(product_type)
    if product_type_family != "unknown":
        return product_type_family
    if category_family_from_raw != "unknown":
        return category_family_from_raw
    return "unknown"


def normalize_row(row: pd.Series) -> dict[str, Any]:
    """Normalize one retailer row into the canonical product observation shape."""
    raw_name = "" if _is_missing(row.get("raw_name")) else str(row.get("raw_name"))
    brand_raw = _clean_scalar(row.get("brand"))
    category_raw = _clean_scalar(row.get("category_raw"))
    retailer = _clean_scalar(row.get("retailer"))

    canonical_brand = canonicalize_brand(brand_raw)
    category_family_from_raw = normalize_raw_category(category_raw)
    product_type = infer_product_type(raw_name)
    category_family = _choose_category_family(product_type, category_family_from_raw)

    amount, unit, pack_count = _choose_quantity_inputs(row)
    quantity = convert_to_canonical_quantity(amount, unit, pack_count)

    # Package type appears both as a top-level field and inside some category
    # attributes. Keeping it top-level makes blocking/matching easier later.
    package_type = infer_package_type(raw_name)
    attributes = extract_category_attributes(category_family, raw_name)
    if attributes.get("package_type") in (None, "", "unknown") and package_type != "unknown":
        attributes["package_type"] = package_type

    extracted_for_validation = {
        "canonical_brand": canonical_brand,
        "product_type": product_type,
        "category_family": category_family,
        "raw_category_family": category_family_from_raw,
        "package_type": package_type,
        **quantity,
        **attributes,
    }
    missing_required_fields = validate_required_fields(category_family, extracted_for_validation)
    confidence = estimate_taxonomy_confidence(extracted_for_validation)
    warnings = list(confidence.get("warnings", []))
    needs_review = bool(confidence.get("needs_review")) or bool(missing_required_fields)

    return {
        "product_id": _clean_scalar(row.get("product_id")),
        "retailer": retailer,
        "raw_name": raw_name,
        "brand_raw": brand_raw,
        "canonical_brand": canonical_brand,
        "category_raw": category_raw,
        "category_family_from_raw": category_family_from_raw,
        "product_type": product_type,
        "category_family": category_family,
        "price": _clean_scalar(row.get("price")),
        "unit_price": _clean_scalar(row.get("unit_price")),
        "promo_text": _clean_scalar(row.get("promo_text")),
        "amount": amount,
        "unit": unit,
        "pack_count": pack_count,
        "size_value": quantity["size_value"],
        "size_unit": quantity["size_unit"],
        "unit_quantity": quantity["unit_quantity"],
        "total_quantity": quantity["total_quantity"],
        "quantity_type": quantity["quantity_type"],
        "canonical_unit": quantity["canonical_unit"],
        "package_type": package_type,
        "is_organic_raw": _to_bool(row.get("is_organic")),
        "is_private_label_raw": _to_bool(row.get("is_private_label")),
        "is_private_label_detected": is_private_label(str(retailer or ""), str(brand_raw or "")),
        "attributes_json": _json_dumps(attributes),
        "confidence_json": _json_dumps(confidence),
        "overall_confidence": confidence["overall"],
        "needs_review": needs_review,
        "missing_required_fields_json": _json_dumps(missing_required_fields),
        "warnings_json": _json_dumps(warnings),
        "url": _clean_scalar(row.get("url")),
        "scraped_at": _clean_scalar(row.get("scraped_at")),
        "estimated_cost": _clean_scalar(row.get("estimated_cost")),
    }


def load_raw_products() -> pd.DataFrame:
    """Load all supported retailer files and concatenate them uniformly."""
    frames = []
    for path in RAW_INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw input file: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def normalize_products(raw_products: pd.DataFrame) -> pd.DataFrame:
    """Normalize every raw row while preserving the required output column order."""
    records = [normalize_row(row) for _, row in raw_products.iterrows()]
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def build_normalization_report(normalized_products: pd.DataFrame) -> dict[str, Any]:
    """Build operational quality metrics for the normalization run."""
    warning_counter: Counter[str] = Counter()
    missing_field_counter: Counter[str] = Counter()

    for _, row in normalized_products.iterrows():
        for warning in json.loads(row["warnings_json"]):
            warning_counter[warning] += 1
        for field in json.loads(row["missing_required_fields_json"]):
            missing_field_counter[field] += 1

    total_rows = int(len(normalized_products))
    needs_review_count = int(normalized_products["needs_review"].sum())
    average_confidence = (
        float(normalized_products["overall_confidence"].mean()) if total_rows else 0.0
    )

    return {
        "total_rows": total_rows,
        "rows_by_retailer": normalized_products["retailer"].value_counts(dropna=False).to_dict(),
        "rows_by_category_family": normalized_products["category_family"].value_counts(dropna=False).to_dict(),
        "rows_by_product_type": normalized_products["product_type"].value_counts(dropna=False).to_dict(),
        "average_overall_confidence": round(average_confidence, 4),
        "needs_review_count": needs_review_count,
        "needs_review_rate": round(needs_review_count / total_rows, 4) if total_rows else 0.0,
        "top_warning_types": dict(warning_counter.most_common(20)),
        "missing_required_field_counts": dict(missing_field_counter.most_common()),
    }


def write_outputs(normalized_products: pd.DataFrame, report: dict[str, Any]) -> None:
    """Persist normalized products and run-level report artifacts."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    normalized_products.to_csv(NORMALIZED_PRODUCTS_PATH, index=False)
    NORMALIZATION_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_run_summary(normalized_products: pd.DataFrame, report: dict[str, Any]) -> None:
    """Print the exact inspection views requested by normalization.md."""
    display_columns = [
        "product_id",
        "retailer",
        "raw_name",
        "canonical_brand",
        "product_type",
        "category_family",
        "total_quantity",
        "quantity_type",
        "package_type",
        "overall_confidence",
        "needs_review",
    ]

    print("\nFirst 20 normalized rows")
    print(normalized_products[display_columns].head(20).to_string(index=False))

    print("\nNormalization report summary")
    print(json.dumps(report, indent=2, sort_keys=True))

    review_examples = normalized_products[normalized_products["needs_review"]].head(10)
    print("\nFirst 10 needs_review examples")
    if review_examples.empty:
        print("No rows require review.")
    else:
        print(review_examples[display_columns + ["warnings_json", "missing_required_fields_json"]].to_string(index=False))


def main() -> None:
    raw_products = load_raw_products()
    normalized_products = normalize_products(raw_products)
    report = build_normalization_report(normalized_products)
    write_outputs(normalized_products, report)
    print_run_summary(normalized_products, report)


if __name__ == "__main__":
    main()
