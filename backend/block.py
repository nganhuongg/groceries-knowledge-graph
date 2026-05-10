"""Blocking stage for RetailGraph normalized product observations.

Blocking is designed for high recall, not final precision. Its job is to
generate plausible cross-retailer candidate pairs cheaply so later stages do
not need to compare every product to every other product.

This module does not compute final match probabilities. That belongs in a later
`match.py` stage after blocking has reduced the search space.

Private-label blocking is especially important in grocery because equivalent
products often appear under retailer-owned brands like 365, Simple Truth, Great
Value, and Good & Gather. Brand equality alone would miss those links.

Quantity is attached as candidate metadata rather than enforced as a hard
blocking rule. Missing or noisy quantity should not eliminate a plausible
candidate too early, but quantity compatibility is still useful evidence for
later scoring.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

NORMALIZED_PRODUCTS_PATH = PROCESSED_DATA_DIR / "normalized_products.csv"
CANDIDATE_PAIRS_PATH = PROCESSED_DATA_DIR / "candidate_pairs.csv"
BLOCKING_REPORT_PATH = OUTPUTS_DIR / "blocking_report.json"

UNKNOWN_VALUES = {"", "unknown", "nan", "none", "null"}
IMPORTANT_TOKEN_STOPWORDS = {
    "organic",
    "natural",
    "fresh",
    "premium",
    "original",
    "classic",
    "great",
    "value",
    "good",
    "gather",
    "simple",
    "truth",
    "whole",
    "foods",
    "pack",
    "pk",
    "ct",
    "count",
    "oz",
    "fl",
    "lb",
    "gal",
    "gallon",
    "bottle",
    "can",
    "cans",
    "bag",
    "box",
    "each",
    "the",
    "and",
    "with",
    "for",
}

CANDIDATE_COLUMNS = [
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
    "shared_token_count",
    "same_product_type",
    "same_category_family",
    "same_brand",
    "both_private_label",
    "same_quantity_type",
    "quantity_ratio",
    "quantity_compatible",
    "candidate_priority",
]


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in UNKNOWN_VALUES


def normalize_for_tokens(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def important_tokens(text: Any) -> set[str]:
    cleaned = normalize_for_tokens(text)
    tokens = set()
    for token in cleaned.split():
        if token in IMPORTANT_TOKEN_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
    return tokens


def safe_float(value: Any) -> float | None:
    if is_unknown(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quantity_ratio(q1: Any, q2: Any) -> float | None:
    left = safe_float(q1)
    right = safe_float(q2)
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return min(left, right) / max(left, right)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_unknown(value):
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def sort_pair_ids(product_id_1: str, product_id_2: str) -> tuple[str, str]:
    return tuple(sorted((str(product_id_1), str(product_id_2))))


def family_compatible(family_a: Any, family_b: Any) -> bool:
    if is_unknown(family_a) or is_unknown(family_b):
        return True
    return str(family_a) == str(family_b)


def iter_cross_retailer_pairs(product_ids: list[str], products_by_id: dict[str, dict[str, Any]]):
    for product_id_1, product_id_2 in combinations(sorted(product_ids), 2):
        product_1 = products_by_id[product_id_1]
        product_2 = products_by_id[product_id_2]
        if product_1["retailer"] == product_2["retailer"]:
            continue
        yield product_id_1, product_id_2


def load_normalized_products() -> pd.DataFrame:
    if not NORMALIZED_PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing normalized input file: {NORMALIZED_PRODUCTS_PATH}")
    return pd.read_csv(NORMALIZED_PRODUCTS_PATH)


def prepare_products(normalized_products: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    products: list[dict[str, Any]] = []
    products_by_id: dict[str, dict[str, Any]] = {}

    for row in normalized_products.to_dict(orient="records"):
        product = {
            "product_id": str(row["product_id"]),
            "retailer": str(row["retailer"]),
            "raw_name": "" if is_unknown(row.get("raw_name")) else str(row.get("raw_name")),
            "canonical_brand": "unknown" if is_unknown(row.get("canonical_brand")) else str(row.get("canonical_brand")),
            "product_type": "unknown" if is_unknown(row.get("product_type")) else str(row.get("product_type")),
            "category_family": "unknown" if is_unknown(row.get("category_family")) else str(row.get("category_family")),
            "total_quantity": safe_float(row.get("total_quantity")),
            "quantity_type": "unknown" if is_unknown(row.get("quantity_type")) else str(row.get("quantity_type")),
            "package_type": "unknown" if is_unknown(row.get("package_type")) else str(row.get("package_type")),
            "is_private_label": normalize_bool(row.get("is_private_label_detected")),
            "overall_confidence": safe_float(row.get("overall_confidence")),
            "needs_review": normalize_bool(row.get("needs_review")),
        }
        product["important_tokens"] = important_tokens(product["raw_name"])
        products.append(product)
        products_by_id[product["product_id"]] = product

    return products, products_by_id


def add_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    product_id_1: str,
    product_id_2: str,
    reason: str,
) -> None:
    pair_key = sort_pair_ids(product_id_1, product_id_2)
    left = products_by_id[pair_key[0]]
    right = products_by_id[pair_key[1]]

    if left["retailer"] == right["retailer"]:
        return

    candidate = candidates.setdefault(
        pair_key,
        {
            "product_id_a": pair_key[0],
            "product_id_b": pair_key[1],
            "blocking_reasons": set(),
        },
    )
    candidate["blocking_reasons"].add(reason)


def strategy_same_product_type(
    products: list[dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    by_product_type: defaultdict[str, list[str]] = defaultdict(list)
    for product in products:
        if is_unknown(product["product_type"]):
            continue
        by_product_type[product["product_type"]].append(product["product_id"])

    for product_ids in by_product_type.values():
        for product_id_1, product_id_2 in iter_cross_retailer_pairs(product_ids, products_by_id):
            add_candidate(candidates, products_by_id, product_id_1, product_id_2, "same_product_type")


def strategy_same_brand_category(
    products: list[dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    by_brand: defaultdict[str, list[str]] = defaultdict(list)
    for product in products:
        if is_unknown(product["canonical_brand"]):
            continue
        by_brand[product["canonical_brand"]].append(product["product_id"])

    for product_ids in by_brand.values():
        for product_id_1, product_id_2 in iter_cross_retailer_pairs(product_ids, products_by_id):
            left = products_by_id[product_id_1]
            right = products_by_id[product_id_2]
            if family_compatible(left["category_family"], right["category_family"]):
                add_candidate(candidates, products_by_id, product_id_1, product_id_2, "same_brand_category")


def strategy_private_label_same_product_type(
    products: list[dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    private_label_products = [
        product
        for product in products
        if product["is_private_label"] and not is_unknown(product["product_type"])
    ]

    by_product_type: defaultdict[str, list[str]] = defaultdict(list)
    for product in private_label_products:
        by_product_type[product["product_type"]].append(product["product_id"])

    for product_ids in by_product_type.values():
        for product_id_1, product_id_2 in iter_cross_retailer_pairs(product_ids, products_by_id):
            left = products_by_id[product_id_1]
            right = products_by_id[product_id_2]
            if family_compatible(left["category_family"], right["category_family"]):
                add_candidate(
                    candidates,
                    products_by_id,
                    product_id_1,
                    product_id_2,
                    "private_label_same_product_type",
                )


def build_family_token_pair_counts(
    products: list[dict[str, Any]],
    minimum_shared_tokens: int,
) -> dict[tuple[str, str], int]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    family_to_token_map: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for product in products:
        family = product["category_family"]
        if is_unknown(family):
            continue
        for token in sorted(product["important_tokens"]):
            family_to_token_map[family][token].append(product["product_id"])

    for token_map in family_to_token_map.values():
        for product_ids in token_map.values():
            if len(product_ids) < 2:
                continue
            for product_id_1, product_id_2 in combinations(sorted(product_ids), 2):
                pair_key = sort_pair_ids(product_id_1, product_id_2)
                pair_counts[pair_key] += 1

    return {
        pair_key: count
        for pair_key, count in pair_counts.items()
        if count >= minimum_shared_tokens
    }


def strategy_same_category_shared_token(
    pair_counts: dict[tuple[str, str], int],
    products_by_id: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for pair_key in sorted(pair_counts):
        left = products_by_id[pair_key[0]]
        right = products_by_id[pair_key[1]]
        if left["retailer"] == right["retailer"]:
            continue
        add_candidate(
            candidates,
            products_by_id,
            pair_key[0],
            pair_key[1],
            "same_category_shared_token",
        )


def strategy_fuzzy_token_rescue(
    pair_counts: dict[tuple[str, str], int],
    products_by_id: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for pair_key in sorted(pair_counts):
        left = products_by_id[pair_key[0]]
        right = products_by_id[pair_key[1]]
        if left["retailer"] == right["retailer"]:
            continue
        if not (
            is_unknown(left["product_type"]) or is_unknown(right["product_type"])
        ):
            continue
        add_candidate(
            candidates,
            products_by_id,
            pair_key[0],
            pair_key[1],
            "fuzzy_token_rescue",
        )


def determine_priority(metadata: dict[str, Any]) -> str:
    if metadata["same_product_type"] and metadata["quantity_compatible"]:
        return "HIGH"
    if metadata["same_brand"] and metadata["same_category_family"]:
        return "HIGH"
    if metadata["both_private_label"] and metadata["same_product_type"]:
        return "HIGH"
    if metadata["same_category_family"] and metadata["shared_token_count"] >= 2:
        return "MEDIUM"
    return "LOW"


def build_candidate_records(
    candidates: dict[tuple[str, str], dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for index, pair_key in enumerate(sorted(candidates), start=1):
        candidate = candidates[pair_key]
        left = products_by_id[candidate["product_id_a"]]
        right = products_by_id[candidate["product_id_b"]]

        shared_tokens = sorted(left["important_tokens"] & right["important_tokens"])
        same_quantity_type = (
            not is_unknown(left["quantity_type"])
            and not is_unknown(right["quantity_type"])
            and left["quantity_type"] == right["quantity_type"]
        )
        pair_quantity_ratio = quantity_ratio(left["total_quantity"], right["total_quantity"])
        quantity_compatible = (
            same_quantity_type
            and pair_quantity_ratio is not None
            and pair_quantity_ratio >= 0.80
        )
        same_product_type = (
            not is_unknown(left["product_type"])
            and left["product_type"] == right["product_type"]
        )
        same_category_family = (
            not is_unknown(left["category_family"])
            and left["category_family"] == right["category_family"]
        )
        same_brand = (
            not is_unknown(left["canonical_brand"])
            and left["canonical_brand"] == right["canonical_brand"]
        )
        both_private_label = bool(left["is_private_label"] and right["is_private_label"])

        metadata = {
            "shared_token_count": len(shared_tokens),
            "same_product_type": same_product_type,
            "same_category_family": same_category_family,
            "same_brand": same_brand,
            "both_private_label": both_private_label,
            "same_quantity_type": same_quantity_type,
            "quantity_ratio": round(pair_quantity_ratio, 4) if pair_quantity_ratio is not None else None,
            "quantity_compatible": quantity_compatible,
        }

        records.append(
            {
                "candidate_pair_id": f"cand_{index:06d}",
                "product_id_a": left["product_id"],
                "retailer_a": left["retailer"],
                "raw_name_a": left["raw_name"],
                "canonical_brand_a": left["canonical_brand"],
                "product_type_a": left["product_type"],
                "category_family_a": left["category_family"],
                "total_quantity_a": left["total_quantity"],
                "quantity_type_a": left["quantity_type"],
                "package_type_a": left["package_type"],
                "is_private_label_a": left["is_private_label"],
                "overall_confidence_a": left["overall_confidence"],
                "needs_review_a": left["needs_review"],
                "product_id_b": right["product_id"],
                "retailer_b": right["retailer"],
                "raw_name_b": right["raw_name"],
                "canonical_brand_b": right["canonical_brand"],
                "product_type_b": right["product_type"],
                "category_family_b": right["category_family"],
                "total_quantity_b": right["total_quantity"],
                "quantity_type_b": right["quantity_type"],
                "package_type_b": right["package_type"],
                "is_private_label_b": right["is_private_label"],
                "overall_confidence_b": right["overall_confidence"],
                "needs_review_b": right["needs_review"],
                "blocking_reasons_json": json.dumps(sorted(candidate["blocking_reasons"])),
                "shared_tokens_json": json.dumps(shared_tokens),
                "shared_token_count": metadata["shared_token_count"],
                "same_product_type": metadata["same_product_type"],
                "same_category_family": metadata["same_category_family"],
                "same_brand": metadata["same_brand"],
                "both_private_label": metadata["both_private_label"],
                "same_quantity_type": metadata["same_quantity_type"],
                "quantity_ratio": metadata["quantity_ratio"],
                "quantity_compatible": metadata["quantity_compatible"],
                "candidate_priority": determine_priority(metadata),
            }
        )

    return records


def naive_cross_retailer_pair_count(products_by_retailer: dict[str, int]) -> int:
    total = 0
    retailers = sorted(products_by_retailer)
    for index, retailer_a in enumerate(retailers):
        for retailer_b in retailers[index + 1 :]:
            total += products_by_retailer[retailer_a] * products_by_retailer[retailer_b]
    return total


def build_blocking_report(
    normalized_products: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> dict[str, Any]:
    products_by_retailer = {
        str(key): int(value)
        for key, value in normalized_products["retailer"].value_counts(dropna=False).sort_index().items()
    }
    naive_pair_count = naive_cross_retailer_pair_count(products_by_retailer)
    candidate_pair_count = int(len(candidate_pairs))
    reduction_rate = 1 - (candidate_pair_count / naive_pair_count) if naive_pair_count else 0.0

    reason_counter: Counter[str] = Counter()
    retailer_pair_counter: Counter[str] = Counter()
    for row in candidate_pairs.to_dict(orient="records"):
        for reason in json.loads(row["blocking_reasons_json"]):
            reason_counter[reason] += 1
        retailer_pair = " | ".join(sorted((str(row["retailer_a"]), str(row["retailer_b"]))))
        retailer_pair_counter[retailer_pair] += 1

    average_candidates_per_product = (
        (candidate_pair_count * 2) / len(normalized_products) if len(normalized_products) else 0.0
    )

    return {
        "total_products": int(len(normalized_products)),
        "products_by_retailer": products_by_retailer,
        "naive_all_pairs_cross_retailer_count": int(naive_pair_count),
        "candidate_pair_count": candidate_pair_count,
        "reduction_rate": round(reduction_rate, 4),
        "candidate_pairs_by_priority": {
            str(key): int(value)
            for key, value in candidate_pairs["candidate_priority"].value_counts(dropna=False).sort_index().items()
        },
        "candidate_pairs_by_blocking_reason": dict(sorted(reason_counter.items())),
        "candidate_pairs_by_retailer_pair": dict(sorted(retailer_pair_counter.items())),
        "unknown_product_type_count": int((normalized_products["product_type"] == "unknown").sum()),
        "needs_review_count": int(normalized_products["needs_review"].astype(str).str.lower().eq("true").sum()),
        "average_candidates_per_product": round(average_candidates_per_product, 4),
    }


def validate_candidates(
    normalized_products: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    valid_product_ids = set(normalized_products["product_id"].astype(str))

    if candidate_pairs.empty:
        raise ValueError("Blocking produced zero candidate pairs.")

    pair_keys = set()
    for row in candidate_pairs.to_dict(orient="records"):
        if row["retailer_a"] == row["retailer_b"]:
            raise ValueError(f"Found same-retailer pair: {row['product_id_a']} vs {row['product_id_b']}")
        if row["product_id_a"] not in valid_product_ids or row["product_id_b"] not in valid_product_ids:
            raise ValueError(f"Found candidate pair with unknown product IDs: {row}")
        pair_key = sort_pair_ids(row["product_id_a"], row["product_id_b"])
        if pair_key in pair_keys:
            raise ValueError(f"Found duplicate candidate pair: {pair_key}")
        pair_keys.add(pair_key)
        if not json.loads(row["blocking_reasons_json"]):
            raise ValueError(f"Found candidate pair without blocking reasons: {pair_key}")

    if report["reduction_rate"] <= 0:
        raise ValueError(f"Expected positive reduction rate, got {report['reduction_rate']}")


def write_outputs(candidate_pairs: pd.DataFrame, report: dict[str, Any]) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_pairs.to_csv(CANDIDATE_PAIRS_PATH, index=False)
    BLOCKING_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_summary(candidate_pairs: pd.DataFrame, report: dict[str, Any]) -> None:
    print("\nBlocking summary")
    print(f"Total products: {report['total_products']}")
    print(f"Naive cross-retailer comparisons: {report['naive_all_pairs_cross_retailer_count']}")
    print(f"Candidate pairs after blocking: {report['candidate_pair_count']}")
    print(f"Reduction rate: {report['reduction_rate']:.4f}")

    print("\nCandidate pairs by priority")
    print(json.dumps(report["candidate_pairs_by_priority"], indent=2, sort_keys=True))

    top_reasons = dict(
        sorted(
            report["candidate_pairs_by_blocking_reason"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
    )
    print("\nTop blocking reasons")
    print(json.dumps(top_reasons, indent=2, sort_keys=True))

    sample_columns = [
        "candidate_pair_id",
        "product_id_a",
        "product_id_b",
        "retailer_a",
        "retailer_b",
        "blocking_reasons_json",
        "shared_tokens_json",
        "shared_token_count",
        "candidate_priority",
    ]
    print("\nSample 20 candidate pairs")
    print(candidate_pairs[sample_columns].head(20).to_string(index=False))


def main() -> None:
    normalized_products = load_normalized_products()
    products, products_by_id = prepare_products(normalized_products)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    strategy_same_product_type(products, products_by_id, candidates)
    strategy_same_brand_category(products, products_by_id, candidates)
    strategy_private_label_same_product_type(products, products_by_id, candidates)

    family_pair_counts_min_1 = build_family_token_pair_counts(products, minimum_shared_tokens=1)
    strategy_same_category_shared_token(family_pair_counts_min_1, products_by_id, candidates)

    family_pair_counts_min_2 = {
        pair_key: count for pair_key, count in family_pair_counts_min_1.items() if count >= 2
    }
    strategy_fuzzy_token_rescue(family_pair_counts_min_2, products_by_id, candidates)

    candidate_records = build_candidate_records(candidates, products_by_id)
    candidate_pairs = pd.DataFrame(candidate_records, columns=CANDIDATE_COLUMNS)
    report = build_blocking_report(normalized_products, candidate_pairs)
    validate_candidates(normalized_products, candidate_pairs, report)
    write_outputs(candidate_pairs, report)
    print_summary(candidate_pairs, report)


if __name__ == "__main__":
    main()
