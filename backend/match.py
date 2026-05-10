"""Inference-style matcher for RetailGraph candidate product pairs.

This module scores blocked candidate pairs using deterministic, interpretable
features. It does not read evaluation labels and should behave like a real
system at inference time: given normalized observations and plausible
cross-retailer candidate pairs, decide whether the products are equivalent,
substitutable, ambiguous, or non-matches.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is optional
    fuzz = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

CANDIDATE_PAIRS_PATH = PROCESSED_DATA_DIR / "candidate_pairs.csv"
NORMALIZED_PRODUCTS_PATH = PROCESSED_DATA_DIR / "normalized_products.csv"
SCORED_PAIRS_PATH = PROCESSED_DATA_DIR / "scored_pairs.csv"
PREDICTED_MATCHES_PATH = PROCESSED_DATA_DIR / "predicted_matches.csv"
MATCHING_REPORT_PATH = OUTPUTS_DIR / "matching_report.json"

UNKNOWN_VALUES = {"", "unknown", "nan", "none", "null"}
CRITICAL_CONFLICTS_FOR_EQUIVALENCE = {
    "product_type_conflict",
    "quantity_type_conflict",
    "organic_conflict",
    "flavor_conflict",
    "diet_type_conflict",
    "produce_form_conflict",
    "pack_size_conflict",
}
SEVERE_CONFLICTS_FOR_NON_MATCH = {
    "product_type_conflict",
    "quantity_type_conflict",
    "flavor_conflict",
    "diet_type_conflict",
    "produce_form_conflict",
}
ATTRIBUTE_FIELDS = [
    "organic",
    "flavor",
    "diet_type",
    "fat_content",
    "sweetness",
    "produce_form",
    "variety",
    "style",
    "cut",
    "animal",
    "egg_size",
    "cage_claim",
]
PRODUCE_PACKAGE_GROUPS = {
    "loose": "loose",
    "bulk": "loose",
    "each": "loose",
    "bag": "packaged",
    "clamshell": "packaged",
    "box": "packaged",
    "package": "packaged",
    "packaged": "packaged",
}
SCORED_COLUMNS = [
    "candidate_pair_id",
    "product_id_a",
    "retailer_a",
    "raw_name_a",
    "canonical_brand_a",
    "product_type_a",
    "category_family_a",
    "total_quantity_a",
    "quantity_type_a",
    "package_type_a",
    "is_private_label_a",
    "overall_confidence_a",
    "needs_review_a",
    "product_id_b",
    "retailer_b",
    "raw_name_b",
    "canonical_brand_b",
    "product_type_b",
    "category_family_b",
    "total_quantity_b",
    "quantity_type_b",
    "package_type_b",
    "is_private_label_b",
    "overall_confidence_b",
    "needs_review_b",
    "blocking_reasons_json",
    "shared_tokens_json",
    "candidate_priority",
    "title_similarity",
    "brand_score",
    "product_type_score",
    "category_score",
    "quantity_score",
    "package_score",
    "attribute_score",
    "private_label_score",
    "confidence_score",
    "blocking_strength_score",
    "equivalence_score",
    "substitute_score",
    "hard_conflicts_json",
    "match_type",
    "route",
    "match_explanation",
]
VALID_MATCH_TYPES = {
    "exact_equivalent",
    "private_label_equivalent",
    "substitute",
    "ambiguous_review",
    "non_match",
}
VALID_ROUTES = {
    "auto_accept_equivalent",
    "auto_accept_substitute",
    "review",
    "reject",
}


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in UNKNOWN_VALUES


def safe_float(value: Any) -> float | None:
    if is_unknown(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_unknown(value):
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def parse_json_like(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default

    text = str(value).strip()
    if not text:
        return default

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError):
            continue
    return default


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def title_similarity(raw_name_a: Any, raw_name_b: Any) -> float:
    left = str(raw_name_a or "")
    right = str(raw_name_b or "")
    if not left and not right:
        return 0.0
    if fuzz is not None:
        return round(fuzz.token_set_ratio(left, right) / 100.0, 4)
    return round(SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio(), 4)


def quantity_ratio(quantity_a: Any, quantity_b: Any) -> float | None:
    left = safe_float(quantity_a)
    right = safe_float(quantity_b)
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return min(left, right) / max(left, right)


def score_brand(row: dict[str, Any], substitute_like: bool) -> float:
    brand_a = row["canonical_brand_a"]
    brand_b = row["canonical_brand_b"]
    both_private_label = row["is_private_label_a"] and row["is_private_label_b"]

    if not is_unknown(brand_a) and brand_a == brand_b:
        return 1.0
    if both_private_label:
        return 0.85
    if substitute_like:
        return 0.5
    if not is_unknown(brand_a) and not is_unknown(brand_b):
        return 0.0
    return 0.25


def score_product_type(row: dict[str, Any]) -> float:
    product_type_a = row["product_type_a"]
    product_type_b = row["product_type_b"]
    same_category = (
        not is_unknown(row["category_family_a"])
        and row["category_family_a"] == row["category_family_b"]
    )

    if not is_unknown(product_type_a) and product_type_a == product_type_b:
        return 1.0
    if (is_unknown(product_type_a) or is_unknown(product_type_b)) and same_category:
        return 0.5
    return 0.0


def score_category(row: dict[str, Any]) -> float:
    family_a = row["category_family_a"]
    family_b = row["category_family_b"]
    if not is_unknown(family_a) and family_a == family_b:
        return 1.0
    if is_unknown(family_a) or is_unknown(family_b):
        return 0.5
    return 0.0


def score_quantity(row: dict[str, Any], product_type_score: float) -> float:
    quantity_type_a = row["quantity_type_a"]
    quantity_type_b = row["quantity_type_b"]
    total_quantity_a = safe_float(row["total_quantity_a"])
    total_quantity_b = safe_float(row["total_quantity_b"])
    ratio = quantity_ratio(total_quantity_a, total_quantity_b)

    if not is_unknown(quantity_type_a) and not is_unknown(quantity_type_b):
        if quantity_type_a != quantity_type_b:
            return 0.25
        if ratio is None:
            return 0.5 if product_type_score >= 1.0 else 0.3
        if ratio >= 0.98:
            return 1.0
        if ratio >= 0.90:
            return 0.85
        if ratio >= 0.80:
            return 0.65
        if ratio >= 0.60:
            return 0.3
        return 0.0

    if total_quantity_a is None or total_quantity_b is None:
        return 0.5 if product_type_score >= 1.0 else 0.25

    return 0.0


def package_group(package_type: Any) -> str:
    cleaned = str(package_type or "")
    return PRODUCE_PACKAGE_GROUPS.get(cleaned, cleaned)


def score_package(row: dict[str, Any], attributes_a: dict[str, Any], attributes_b: dict[str, Any]) -> float:
    package_type_a = row["package_type_a"]
    package_type_b = row["package_type_b"]
    category_family_a = row["category_family_a"]
    category_family_b = row["category_family_b"]

    if not is_unknown(package_type_a) and package_type_a == package_type_b:
        return 1.0
    if is_unknown(package_type_a) or is_unknown(package_type_b):
        return 0.7

    produce_form_a = attributes_a.get("produce_form")
    produce_form_b = attributes_b.get("produce_form")
    if category_family_a == "produce" and category_family_b == "produce":
        group_a = package_group(produce_form_a or package_type_a)
        group_b = package_group(produce_form_b or package_type_b)
        if group_a != group_b and {group_a, group_b} == {"loose", "packaged"}:
            return 0.1
        return 0.2

    return 0.2


def score_attributes(row: dict[str, Any], attributes_a: dict[str, Any], attributes_b: dict[str, Any]) -> float:
    direct_conflicts = 0
    agreements = 0
    comparable = 0
    missing = 0

    for field in ATTRIBUTE_FIELDS:
        value_a = attributes_a.get(field)
        value_b = attributes_b.get(field)
        if is_unknown(value_a) or is_unknown(value_b):
            if not is_unknown(value_a) or not is_unknown(value_b):
                missing += 1
            continue
        comparable += 1
        if value_a == value_b:
            agreements += 1
        else:
            direct_conflicts += 1

    if direct_conflicts >= 2:
        return 0.0
    if direct_conflicts == 1:
        return 0.4
    if comparable == 0:
        return 0.75 if row["product_type_a"] == row["product_type_b"] else 0.5
    if agreements == comparable and missing == 0:
        return 1.0
    if agreements == comparable:
        return 0.75
    return 0.6


def score_private_label(row: dict[str, Any]) -> float:
    return 1.0 if row["is_private_label_a"] and row["is_private_label_b"] else 0.0


def score_confidence(row: dict[str, Any]) -> float:
    confidence_a = safe_float(row["overall_confidence_a"]) or 0.0
    confidence_b = safe_float(row["overall_confidence_b"]) or 0.0
    return round((confidence_a + confidence_b) / 2, 4)


def score_blocking_strength(row: dict[str, Any]) -> float:
    base = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(str(row["candidate_priority"]), 0.4)
    reasons = parse_json_like(row["blocking_reasons_json"], [])
    bonus = min(max(len(reasons) - 1, 0) * 0.05, 0.2)
    return round(min(base + bonus, 1.0), 4)


def detect_hard_conflicts(row: dict[str, Any], attributes_a: dict[str, Any], attributes_b: dict[str, Any]) -> list[str]:
    conflicts: list[str] = []

    if (
        not is_unknown(row["product_type_a"])
        and not is_unknown(row["product_type_b"])
        and row["product_type_a"] != row["product_type_b"]
    ):
        conflicts.append("product_type_conflict")

    if (
        not is_unknown(row["quantity_type_a"])
        and not is_unknown(row["quantity_type_b"])
        and row["quantity_type_a"] != row["quantity_type_b"]
    ):
        conflicts.append("quantity_type_conflict")

    organic_a = attributes_a.get("organic")
    organic_b = attributes_b.get("organic")
    if isinstance(organic_a, bool) and isinstance(organic_b, bool) and organic_a != organic_b:
        conflicts.append("organic_conflict")

    flavor_a = attributes_a.get("flavor")
    flavor_b = attributes_b.get("flavor")
    if not is_unknown(flavor_a) and not is_unknown(flavor_b) and flavor_a != flavor_b:
        conflicts.append("flavor_conflict")

    diet_type_a = attributes_a.get("diet_type")
    diet_type_b = attributes_b.get("diet_type")
    if not is_unknown(diet_type_a) and not is_unknown(diet_type_b) and diet_type_a != diet_type_b:
        conflicts.append("diet_type_conflict")

    produce_form_a = attributes_a.get("produce_form")
    produce_form_b = attributes_b.get("produce_form")
    if not is_unknown(produce_form_a) and not is_unknown(produce_form_b):
        group_a = package_group(produce_form_a)
        group_b = package_group(produce_form_b)
        if group_a != group_b and {group_a, group_b} == {"loose", "packaged"}:
            conflicts.append("produce_form_conflict")

    ratio = quantity_ratio(row["total_quantity_a"], row["total_quantity_b"])
    if row["product_type_a"] == row["product_type_b"] and ratio is not None and ratio < 0.8:
        conflicts.append("pack_size_conflict")

    brand_a = row["canonical_brand_a"]
    brand_b = row["canonical_brand_b"]
    if (
        not row["is_private_label_a"]
        and not row["is_private_label_b"]
        and not is_unknown(brand_a)
        and not is_unknown(brand_b)
        and brand_a != brand_b
    ):
        conflicts.append("brand_conflict_for_exact")

    return sorted(set(conflicts))


def contains_critical_conflict(conflicts: list[str]) -> bool:
    return any(conflict in CRITICAL_CONFLICTS_FOR_EQUIVALENCE for conflict in conflicts)


def contains_severe_conflict(conflicts: list[str]) -> bool:
    return any(conflict in SEVERE_CONFLICTS_FOR_NON_MATCH for conflict in conflicts)


def classify_match(
    row: dict[str, Any],
    scores: dict[str, float],
    conflicts: list[str],
    attributes_a: dict[str, Any],
    attributes_b: dict[str, Any],
) -> tuple[str, str]:
    same_known_brand = (
        not is_unknown(row["canonical_brand_a"])
        and row["canonical_brand_a"] == row["canonical_brand_b"]
    )
    both_private_label = row["is_private_label_a"] and row["is_private_label_b"]
    critical_conflict = contains_critical_conflict(conflicts)
    severe_conflict = contains_severe_conflict(conflicts)
    key_attributes_compatible = scores["attribute_score"] >= 0.75
    plausible_review = (
        scores["equivalence_score"] >= 0.62
        or scores["substitute_score"] >= 0.60
        or row["needs_review_a"]
        or row["needs_review_b"]
    )

    if (
        scores["equivalence_score"] >= 0.88
        and same_known_brand
        and scores["product_type_score"] == 1.0
        and scores["quantity_score"] >= 0.85
        and not critical_conflict
        and not both_private_label
    ):
        return "exact_equivalent", "auto_accept_equivalent"

    if (
        scores["equivalence_score"] >= 0.84
        and both_private_label
        and scores["product_type_score"] == 1.0
        and scores["quantity_score"] >= 0.85
        and not critical_conflict
        and key_attributes_compatible
    ):
        return "private_label_equivalent", "auto_accept_equivalent"

    if (
        scores["substitute_score"] >= 0.72
        and scores["category_score"] >= 0.5
        and scores["product_type_score"] >= 0.5
        and "product_type_conflict" not in conflicts
    ):
        return "substitute", "auto_accept_substitute"

    if severe_conflict and scores["equivalence_score"] < 0.62 and scores["substitute_score"] < 0.60:
        return "non_match", "reject"

    if plausible_review or conflicts:
        return "ambiguous_review", "review"

    return "non_match", "reject"


def apply_route_safety(
    match_type: str,
    route: str,
    confidence_score: float,
) -> str:
    if confidence_score < 0.65 and match_type != "non_match":
        return "review"
    return route


def explain_match(
    row: dict[str, Any],
    scores: dict[str, float],
    conflicts: list[str],
    match_type: str,
) -> str:
    if match_type == "exact_equivalent":
        return (
            f"Matched as exact_equivalent because both products share brand "
            f"{row['canonical_brand_a']}, product type {row['product_type_a']}, "
            "compatible quantity, and no critical attribute conflicts."
        )
    if match_type == "private_label_equivalent":
        return (
            "Matched as private_label_equivalent because both products are retailer "
            "private labels with the same product type, compatible quantity, and "
            "compatible key attributes."
        )
    if match_type == "substitute":
        return (
            f"Classified as substitute because products share product type "
            f"{row['product_type_a'] if row['product_type_a'] == row['product_type_b'] else 'or category evidence'} "
            "and similar quantity, but brands differ, so they are comparable but "
            "not exact equivalents."
        )
    if match_type == "ambiguous_review":
        return (
            "Routed to review because product type and category are plausible, but "
            "quantity/package evidence is incomplete or conflicting."
        )
    return "Rejected because product types or critical attributes conflict."


def round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CANDIDATE_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing candidate pair file: {CANDIDATE_PAIRS_PATH}")
    if not NORMALIZED_PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing normalized product file: {NORMALIZED_PRODUCTS_PATH}")
    return pd.read_csv(CANDIDATE_PAIRS_PATH), pd.read_csv(NORMALIZED_PRODUCTS_PATH)


def score_pairs(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    scored_records: list[dict[str, Any]] = []

    for row in candidate_pairs.to_dict(orient="records"):
        row["is_private_label_a"] = normalize_bool(row["is_private_label_a"])
        row["is_private_label_b"] = normalize_bool(row["is_private_label_b"])
        row["needs_review_a"] = normalize_bool(row["needs_review_a"])
        row["needs_review_b"] = normalize_bool(row["needs_review_b"])

        attributes_a = parse_json_like(row.get("attributes_json_a"), {})  # fallback if ever present
        attributes_b = parse_json_like(row.get("attributes_json_b"), {})
        if not attributes_a:
            attributes_a = {}
        if not attributes_b:
            attributes_b = {}

        normalized_lookup_attrs = {}
        row["_attributes_a"] = normalized_lookup_attrs.get("a", attributes_a)
        row["_attributes_b"] = normalized_lookup_attrs.get("b", attributes_b)

        row_attrs_a = parse_json_like(row.get("attributes_json_a"), {})
        row_attrs_b = parse_json_like(row.get("attributes_json_b"), {})
        if row_attrs_a:
            attributes_a = row_attrs_a
        if row_attrs_b:
            attributes_b = row_attrs_b

        substitute_like = (
            (not is_unknown(row["category_family_a"]) and row["category_family_a"] == row["category_family_b"])
            or (not is_unknown(row["product_type_a"]) and row["product_type_a"] == row["product_type_b"])
        )

        scores = {
            "title_similarity": title_similarity(row["raw_name_a"], row["raw_name_b"]),
            "product_type_score": score_product_type(row),
            "category_score": score_category(row),
        }
        scores["brand_score"] = score_brand(row, substitute_like)
        scores["quantity_score"] = score_quantity(row, scores["product_type_score"])
        scores["package_score"] = score_package(row, attributes_a, attributes_b)
        scores["attribute_score"] = score_attributes(row, attributes_a, attributes_b)
        scores["private_label_score"] = score_private_label(row)
        scores["confidence_score"] = score_confidence(row)
        scores["blocking_strength_score"] = score_blocking_strength(row)

        conflicts = detect_hard_conflicts(row, attributes_a, attributes_b)

        scores["equivalence_score"] = round_score(
            0.15 * scores["title_similarity"]
            + 0.15 * scores["brand_score"]
            + 0.15 * scores["product_type_score"]
            + 0.10 * scores["category_score"]
            + 0.20 * scores["quantity_score"]
            + 0.10 * scores["package_score"]
            + 0.10 * scores["attribute_score"]
            + 0.05 * scores["blocking_strength_score"]
        )
        scores["substitute_score"] = round_score(
            0.10 * scores["title_similarity"]
            + 0.10 * scores["category_score"]
            + 0.25 * scores["product_type_score"]
            + 0.15 * scores["quantity_score"]
            + 0.10 * scores["package_score"]
            + 0.20 * scores["attribute_score"]
            + 0.10 * scores["blocking_strength_score"]
        )

        match_type, route = classify_match(row, scores, conflicts, attributes_a, attributes_b)
        route = apply_route_safety(match_type, route, scores["confidence_score"])
        if route == "review" and match_type == "substitute":
            match_type = "ambiguous_review"
        explanation = explain_match(row, scores, conflicts, match_type)

        scored_record = {column: row.get(column) for column in SCORED_COLUMNS if column in row}
        scored_record.update(
            {
                "title_similarity": scores["title_similarity"],
                "brand_score": scores["brand_score"],
                "product_type_score": scores["product_type_score"],
                "category_score": scores["category_score"],
                "quantity_score": scores["quantity_score"],
                "package_score": scores["package_score"],
                "attribute_score": scores["attribute_score"],
                "private_label_score": scores["private_label_score"],
                "confidence_score": scores["confidence_score"],
                "blocking_strength_score": scores["blocking_strength_score"],
                "equivalence_score": scores["equivalence_score"],
                "substitute_score": scores["substitute_score"],
                "hard_conflicts_json": json.dumps(conflicts),
                "match_type": match_type,
                "route": route,
                "match_explanation": explanation,
            }
        )
        scored_records.append(scored_record)

    return pd.DataFrame(scored_records, columns=SCORED_COLUMNS)


def build_attribute_maps(normalized_products: pd.DataFrame) -> dict[str, dict[str, Any]]:
    attribute_map: dict[str, dict[str, Any]] = {}
    for row in normalized_products.to_dict(orient="records"):
        attribute_map[str(row["product_id"])] = parse_json_like(row.get("attributes_json"), {})
    return attribute_map


def attach_attributes(candidate_pairs: pd.DataFrame, attribute_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    pairs = candidate_pairs.copy()
    pairs["attributes_json_a"] = pairs["product_id_a"].astype(str).map(
        lambda product_id: json.dumps(attribute_map.get(product_id, {}), sort_keys=True)
    )
    pairs["attributes_json_b"] = pairs["product_id_b"].astype(str).map(
        lambda product_id: json.dumps(attribute_map.get(product_id, {}), sort_keys=True)
    )
    return pairs


def build_matching_report(scored_pairs: pd.DataFrame) -> dict[str, Any]:
    match_type_counts = {
        str(key): int(value)
        for key, value in scored_pairs["match_type"].value_counts(dropna=False).sort_index().items()
    }
    route_counts = {
        str(key): int(value)
        for key, value in scored_pairs["route"].value_counts(dropna=False).sort_index().items()
    }

    conflict_counter: Counter[str] = Counter()
    retailer_pair_counter: Counter[str] = Counter()
    for row in scored_pairs.to_dict(orient="records"):
        for conflict in parse_json_like(row["hard_conflicts_json"], []):
            conflict_counter[conflict] += 1
        retailer_pair = " | ".join(sorted((str(row["retailer_a"]), str(row["retailer_b"]))))
        retailer_pair_counter[retailer_pair] += 1

    def example_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
        subset = frame[
            [
                "candidate_pair_id",
                "product_id_a",
                "product_id_b",
                "raw_name_a",
                "raw_name_b",
                "match_type",
                "route",
                "equivalence_score",
                "substitute_score",
                "match_explanation",
            ]
        ].head(limit)
        return subset.to_dict(orient="records")

    return {
        "total_candidate_pairs": int(len(scored_pairs)),
        "scored_pairs": int(len(scored_pairs)),
        "match_type_counts": match_type_counts,
        "route_counts": route_counts,
        "average_equivalence_score": round(float(scored_pairs["equivalence_score"].mean()), 4),
        "average_substitute_score": round(float(scored_pairs["substitute_score"].mean()), 4),
        "auto_accept_equivalent_count": int(route_counts.get("auto_accept_equivalent", 0)),
        "auto_accept_substitute_count": int(route_counts.get("auto_accept_substitute", 0)),
        "review_count": int(route_counts.get("review", 0)),
        "reject_count": int(route_counts.get("reject", 0)),
        "top_hard_conflict_types": dict(conflict_counter.most_common(20)),
        "pairs_by_retailer_pair": dict(sorted(retailer_pair_counter.items())),
        "high_confidence_examples": example_rows(
            scored_pairs[scored_pairs["route"] == "auto_accept_equivalent"].sort_values(
                ["equivalence_score", "substitute_score"], ascending=False
            ),
            limit=10,
        ),
        "substitute_examples": example_rows(
            scored_pairs[scored_pairs["route"] == "auto_accept_substitute"].sort_values(
                "substitute_score", ascending=False
            ),
            limit=10,
        ),
        "ambiguous_examples": example_rows(
            scored_pairs[scored_pairs["route"] == "review"].sort_values(
                ["equivalence_score", "substitute_score"], ascending=False
            ),
            limit=10,
        ),
        "rejected_examples": example_rows(
            scored_pairs[scored_pairs["route"] == "reject"].sort_values(
                ["equivalence_score", "substitute_score"], ascending=True
            ),
            limit=10,
        ),
    }


def validate_outputs(
    candidate_pairs: pd.DataFrame,
    normalized_products: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    predicted_matches: pd.DataFrame,
) -> None:
    if len(scored_pairs) != len(candidate_pairs):
        raise ValueError("scored_pairs row count must equal candidate_pairs row count.")

    valid_product_ids = set(normalized_products["product_id"].astype(str))
    for row in scored_pairs.to_dict(orient="records"):
        if row["retailer_a"] == row["retailer_b"]:
            raise ValueError(f"Found same-retailer scored pair: {row['candidate_pair_id']}")
        if row["product_id_a"] not in valid_product_ids or row["product_id_b"] not in valid_product_ids:
            raise ValueError(f"Found unknown product IDs in scored pair: {row['candidate_pair_id']}")
        if row["match_type"] not in VALID_MATCH_TYPES:
            raise ValueError(f"Invalid match_type: {row['match_type']}")
        if row["route"] not in VALID_ROUTES:
            raise ValueError(f"Invalid route: {row['route']}")
        if parse_json_like(row["hard_conflicts_json"], None) is None:
            raise ValueError(f"Missing hard_conflicts_json for {row['candidate_pair_id']}")
        for score_field in [
            "title_similarity",
            "brand_score",
            "product_type_score",
            "category_score",
            "quantity_score",
            "package_score",
            "attribute_score",
            "private_label_score",
            "confidence_score",
            "blocking_strength_score",
            "equivalence_score",
            "substitute_score",
        ]:
            score_value = safe_float(row[score_field])
            if score_value is None or not (0.0 <= score_value <= 1.0):
                raise ValueError(f"Score {score_field} out of range for {row['candidate_pair_id']}")

    if any(predicted_matches["route"].astype(str).str.lower() == "reject"):
        raise ValueError("predicted_matches.csv must exclude reject rows.")


def write_outputs(scored_pairs: pd.DataFrame, predicted_matches: pd.DataFrame, report: dict[str, Any]) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scored_pairs.to_csv(SCORED_PAIRS_PATH, index=False)
    predicted_matches.to_csv(PREDICTED_MATCHES_PATH, index=False)
    MATCHING_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_sample(title: str, frame: pd.DataFrame, limit: int = 10) -> None:
    print(f"\n{title}")
    if frame.empty:
        print("No rows.")
        return
    display_columns = [
        "candidate_pair_id",
        "product_id_a",
        "product_id_b",
        "match_type",
        "route",
        "equivalence_score",
        "substitute_score",
        "hard_conflicts_json",
    ]
    print(frame[display_columns].head(limit).to_string(index=False))


def print_summary(scored_pairs: pd.DataFrame, report: dict[str, Any]) -> None:
    print("\nMatching summary")
    print(f"Total candidate pairs: {report['total_candidate_pairs']}")

    print("\nMatch type counts")
    print(json.dumps(report["match_type_counts"], indent=2, sort_keys=True))

    print("\nRoute counts")
    print(json.dumps(report["route_counts"], indent=2, sort_keys=True))

    print("\nTop hard conflicts")
    print(json.dumps(report["top_hard_conflict_types"], indent=2, sort_keys=True))

    print_sample(
        "Sample 10 auto accepted equivalents",
        scored_pairs[scored_pairs["route"] == "auto_accept_equivalent"].sort_values(
            "equivalence_score", ascending=False
        ),
    )
    print_sample(
        "Sample 10 substitutes",
        scored_pairs[scored_pairs["route"] == "auto_accept_substitute"].sort_values(
            "substitute_score", ascending=False
        ),
    )
    print_sample(
        "Sample 10 review pairs",
        scored_pairs[scored_pairs["route"] == "review"].sort_values(
            ["equivalence_score", "substitute_score"], ascending=False
        ),
    )
    print_sample(
        "Sample 10 rejected pairs",
        scored_pairs[scored_pairs["route"] == "reject"].sort_values(
            ["equivalence_score", "substitute_score"], ascending=True
        ),
    )


def main() -> None:
    candidate_pairs, normalized_products = load_inputs()
    attribute_map = build_attribute_maps(normalized_products)
    candidate_pairs_with_attributes = attach_attributes(candidate_pairs, attribute_map)
    scored_pairs = score_pairs(candidate_pairs_with_attributes)
    predicted_matches = scored_pairs[scored_pairs["route"] != "reject"].copy()
    report = build_matching_report(scored_pairs)
    validate_outputs(candidate_pairs, normalized_products, scored_pairs, predicted_matches)
    write_outputs(scored_pairs, predicted_matches, report)
    print_summary(scored_pairs, report)


if __name__ == "__main__":
    main()
