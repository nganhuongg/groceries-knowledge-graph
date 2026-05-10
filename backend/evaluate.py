"""Offline evaluation for RetailGraph matching outputs.

This module compares inference-time matcher outputs against synthetic labels.
It measures equivalent matching, substitute matching, safety, routing
conservatism, and blocking coverage without modifying any model outputs.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SCORED_PAIRS_PATH = PROCESSED_DATA_DIR / "scored_pairs.csv"
PREDICTED_MATCHES_PATH = PROCESSED_DATA_DIR / "predicted_matches.csv"
NORMALIZED_PRODUCTS_PATH = PROCESSED_DATA_DIR / "normalized_products.csv"

GROUND_TRUTH_MATCHES_PATH = LABELS_DIR / "ground_truth_matches.json"
NEGATIVE_PAIRS_PATH = LABELS_DIR / "negative_pairs.json"
SUBSTITUTE_PAIRS_PATH = LABELS_DIR / "substitute_pairs.json"

EVALUATION_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"
ERROR_CASES_PATH = OUTPUTS_DIR / "error_cases.csv"
EVALUATION_BY_MATCH_TYPE_PATH = OUTPUTS_DIR / "evaluation_by_match_type.csv"
EVALUATION_BY_CATEGORY_PATH = OUTPUTS_DIR / "evaluation_by_category.csv"
EVALUATION_BY_ROUTE_PATH = OUTPUTS_DIR / "evaluation_by_route.csv"

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
EQUIVALENT_LABELS = {"exact_equivalent", "private_label_equivalent"}
ERROR_CASE_COLUMNS = [
    "error_type",
    "product_id_a",
    "product_id_b",
    "true_label",
    "predicted_match_type",
    "route",
    "equivalence_score",
    "substitute_score",
    "category_family_a",
    "category_family_b",
    "product_type_a",
    "product_type_b",
    "retailer_a",
    "retailer_b",
    "raw_name_a",
    "raw_name_b",
    "hard_conflicts_json",
    "match_explanation",
    "label_reason_or_description",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def make_pair_key(product_id_a: Any, product_id_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(product_id_a), str(product_id_b))))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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


def expand_ground_truth_groups(groups: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    expanded: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        mappings = group.get("mappings", {})
        product_ids = [str(product_id) for product_id in mappings.values()]
        for product_id_a, product_id_b in combinations(sorted(product_ids), 2):
            expanded[make_pair_key(product_id_a, product_id_b)] = {
                "true_label": str(group.get("match_type")),
                "match_group_id": str(group.get("match_group_id")),
                "canonical_description": str(group.get("canonical_description", "")),
            }
    return expanded


def build_truth_map(
    positive_equivalent_pairs: dict[tuple[str, str], dict[str, Any]],
    negative_pairs: list[dict[str, Any]],
    substitute_pairs: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    truth_map = dict(positive_equivalent_pairs)
    label_conflicts: list[dict[str, Any]] = []
    precedence = {
        "exact_equivalent": 3,
        "private_label_equivalent": 3,
        "substitute": 2,
        "non_match": 1,
    }

    def merge_label(pair_key: tuple[str, str], new_payload: dict[str, Any]) -> None:
        existing = truth_map.get(pair_key)
        if existing is None:
            truth_map[pair_key] = new_payload
            return
        if existing["true_label"] == new_payload["true_label"]:
            return

        label_conflicts.append(
            {
                "pair_key": list(pair_key),
                "existing_label": existing["true_label"],
                "new_label": new_payload["true_label"],
            }
        )
        if precedence[new_payload["true_label"]] > precedence[existing["true_label"]]:
            truth_map[pair_key] = new_payload

    for row in negative_pairs:
        merge_label(
            make_pair_key(row["product_id_a"], row["product_id_b"]),
            {
                "true_label": "non_match",
                "reason": str(row.get("reason", "")),
            },
        )

    for row in substitute_pairs:
        merge_label(
            make_pair_key(row["product_id_a"], row["product_id_b"]),
            {
                "true_label": "substitute",
                "reason": str(row.get("reason", "")),
            },
        )

    return truth_map, label_conflicts


def get_prediction_for_pair(
    scored_pairs_map: dict[tuple[str, str], dict[str, Any]],
    pair_key: tuple[str, str],
) -> dict[str, Any] | None:
    return scored_pairs_map.get(pair_key)


def get_product_metadata(
    product_id: str,
    products_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return products_by_id.get(
        product_id,
        {
            "product_id": product_id,
            "retailer": "",
            "raw_name": "",
            "category_family": "unknown",
            "product_type": "unknown",
        },
    )


def infer_pair_category(
    pair_key: tuple[str, str],
    products_by_id: dict[str, dict[str, Any]],
    prediction: dict[str, Any] | None,
) -> str:
    if prediction is not None:
        category_a = str(prediction.get("category_family_a", "unknown"))
        category_b = str(prediction.get("category_family_b", "unknown"))
    else:
        product_a = get_product_metadata(pair_key[0], products_by_id)
        product_b = get_product_metadata(pair_key[1], products_by_id)
        category_a = str(product_a.get("category_family", "unknown"))
        category_b = str(product_b.get("category_family", "unknown"))

    if category_a == category_b and category_a not in {"", "unknown"}:
        return category_a
    return "mixed_or_unknown"


def build_error_case(
    error_type: str,
    pair_key: tuple[str, str],
    true_payload: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    products_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    product_a_meta = get_product_metadata(pair_key[0], products_by_id)
    product_b_meta = get_product_metadata(pair_key[1], products_by_id)
    label_reason_or_description = ""
    if true_payload is not None:
        label_reason_or_description = str(
            true_payload.get("reason")
            or true_payload.get("canonical_description")
            or true_payload.get("match_group_id", "")
        )

    base = {
        "error_type": error_type,
        "product_id_a": pair_key[0],
        "product_id_b": pair_key[1],
        "true_label": None if true_payload is None else true_payload.get("true_label"),
        "predicted_match_type": None if prediction is None else prediction.get("match_type"),
        "route": None if prediction is None else prediction.get("route"),
        "equivalence_score": None if prediction is None else prediction.get("equivalence_score"),
        "substitute_score": None if prediction is None else prediction.get("substitute_score"),
        "category_family_a": product_a_meta.get("category_family", "unknown"),
        "category_family_b": product_b_meta.get("category_family", "unknown"),
        "product_type_a": product_a_meta.get("product_type", "unknown"),
        "product_type_b": product_b_meta.get("product_type", "unknown"),
        "retailer_a": product_a_meta.get("retailer", ""),
        "retailer_b": product_b_meta.get("retailer", ""),
        "raw_name_a": product_a_meta.get("raw_name", ""),
        "raw_name_b": product_b_meta.get("raw_name", ""),
        "hard_conflicts_json": None if prediction is None else prediction.get("hard_conflicts_json"),
        "match_explanation": None if prediction is None else prediction.get("match_explanation"),
        "label_reason_or_description": label_reason_or_description,
    }

    if prediction is not None:
        base["category_family_a"] = prediction.get("category_family_a", base["category_family_a"])
        base["category_family_b"] = prediction.get("category_family_b", base["category_family_b"])
        base["product_type_a"] = prediction.get("product_type_a", base["product_type_a"])
        base["product_type_b"] = prediction.get("product_type_b", base["product_type_b"])
        base["retailer_a"] = prediction.get("retailer_a", base["retailer_a"])
        base["retailer_b"] = prediction.get("retailer_b", base["retailer_b"])
        base["raw_name_a"] = prediction.get("raw_name_a", base["raw_name_a"])
        base["raw_name_b"] = prediction.get("raw_name_b", base["raw_name_b"])

    return base


def validate_inputs(
    scored_pairs: pd.DataFrame,
    predicted_matches: pd.DataFrame,
    normalized_products: pd.DataFrame,
    ground_truth_matches: list[dict[str, Any]],
    negative_pairs: list[dict[str, Any]],
    substitute_pairs: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    valid_product_ids = set(normalized_products["product_id"].astype(str))

    def note_missing_ids(source_name: str, product_ids: list[str]) -> None:
        missing = [product_id for product_id in product_ids if product_id not in valid_product_ids]
        if missing:
            notes.append(f"{source_name} references missing product IDs: {sorted(set(missing))[:20]}")

    label_product_ids = []
    for group in ground_truth_matches:
        label_product_ids.extend([str(product_id) for product_id in group.get("mappings", {}).values()])
    for row in negative_pairs:
        label_product_ids.extend([str(row["product_id_a"]), str(row["product_id_b"])])
    for row in substitute_pairs:
        label_product_ids.extend([str(row["product_id_a"]), str(row["product_id_b"])])
    note_missing_ids("label files", label_product_ids)

    scored_product_ids = []
    pair_keys = set()
    for row in scored_pairs.to_dict(orient="records"):
        product_id_a = str(row["product_id_a"])
        product_id_b = str(row["product_id_b"])
        scored_product_ids.extend([product_id_a, product_id_b])

        pair_key = make_pair_key(product_id_a, product_id_b)
        if pair_key in pair_keys:
            notes.append(f"Duplicate pair key in scored_pairs.csv: {pair_key}")
        pair_keys.add(pair_key)

        if row["match_type"] not in VALID_MATCH_TYPES:
            notes.append(f"Unknown match_type in scored_pairs.csv: {row['match_type']}")
        if row["route"] not in VALID_ROUTES:
            notes.append(f"Unknown route in scored_pairs.csv: {row['route']}")

    note_missing_ids("scored_pairs.csv", scored_product_ids)

    reject_count_in_predicted = int(predicted_matches["route"].astype(str).str.lower().eq("reject").sum())
    if reject_count_in_predicted > 0:
        notes.append("predicted_matches.csv contains reject rows, which should not happen.")

    return notes


def main() -> None:
    required_files = [
        SCORED_PAIRS_PATH,
        PREDICTED_MATCHES_PATH,
        NORMALIZED_PRODUCTS_PATH,
        GROUND_TRUTH_MATCHES_PATH,
        NEGATIVE_PAIRS_PATH,
        SUBSTITUTE_PAIRS_PATH,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required evaluation inputs: {missing_files}")

    scored_pairs = pd.read_csv(SCORED_PAIRS_PATH)
    predicted_matches = pd.read_csv(PREDICTED_MATCHES_PATH)
    normalized_products = pd.read_csv(NORMALIZED_PRODUCTS_PATH)

    ground_truth_matches = load_json(GROUND_TRUTH_MATCHES_PATH)
    negative_pairs = load_json(NEGATIVE_PAIRS_PATH)
    substitute_pairs = load_json(SUBSTITUTE_PAIRS_PATH)

    positive_equivalent_pairs = expand_ground_truth_groups(ground_truth_matches)
    truth_map, label_conflicts = build_truth_map(
        positive_equivalent_pairs,
        negative_pairs,
        substitute_pairs,
    )

    notes = validate_inputs(
        scored_pairs,
        predicted_matches,
        normalized_products,
        ground_truth_matches,
        negative_pairs,
        substitute_pairs,
    )

    products_by_id = {
        str(row["product_id"]): row for row in normalized_products.to_dict(orient="records")
    }
    scored_pairs_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scored_pairs.to_dict(orient="records"):
        scored_pairs_map[make_pair_key(row["product_id_a"], row["product_id_b"])] = row

    total_true_equivalent_pairs = len(
        [pair_key for pair_key, payload in truth_map.items() if payload["true_label"] in EQUIVALENT_LABELS]
    )
    total_true_substitute_pairs = len(
        [pair_key for pair_key, payload in truth_map.items() if payload["true_label"] == "substitute"]
    )
    total_known_negative_pairs = len(
        [pair_key for pair_key, payload in truth_map.items() if payload["true_label"] == "non_match"]
    )

    equivalent_tp = 0
    equivalent_fp = 0
    equivalent_fn = 0
    true_equivalent_routed_to_review = 0

    substitute_tp = 0
    substitute_fp = 0
    substitute_fn = 0
    true_substitute_routed_to_review = 0
    substitute_predicted_as_equivalent = 0
    equivalent_predicted_as_substitute = 0

    review_count = int(scored_pairs["route"].astype(str).eq("review").sum())
    review_rate = round(safe_divide(review_count, len(scored_pairs)), 4)
    auto_accept_equivalent_count = int(scored_pairs["route"].astype(str).eq("auto_accept_equivalent").sum())
    auto_accept_substitute_count = int(scored_pairs["route"].astype(str).eq("auto_accept_substitute").sum())
    reject_count = int(scored_pairs["route"].astype(str).eq("reject").sum())

    known_negative_auto_accepted_as_equivalent = 0
    known_negative_auto_accepted_as_substitute = 0
    known_negative_rejected = 0
    known_negative_routed_to_review = 0
    total_known_negative_pairs_that_appear_in_scored_pairs = 0

    unknown_auto_accept_equivalent_count = 0
    unknown_auto_accept_substitute_count = 0
    unknown_review_count = 0
    unknown_reject_count = 0

    true_equivalent_pairs_seen_by_blocking = 0
    true_substitute_pairs_seen_by_blocking = 0

    route_rows: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []

    by_match_type_rows = {
        "exact_equivalent": {
            "true_count": 0,
            "auto_accepted_correct": 0,
            "auto_accepted_wrong": 0,
            "routed_to_review": 0,
            "rejected_or_missed": 0,
        },
        "private_label_equivalent": {
            "true_count": 0,
            "auto_accepted_correct": 0,
            "auto_accepted_wrong": 0,
            "routed_to_review": 0,
            "rejected_or_missed": 0,
        },
        "substitute": {
            "true_count": 0,
            "auto_accepted_correct": 0,
            "auto_accepted_wrong": 0,
            "routed_to_review": 0,
            "rejected_or_missed": 0,
        },
        "non_match": {
            "true_count": 0,
            "auto_accepted_correct": 0,
            "auto_accepted_wrong": 0,
            "routed_to_review": 0,
            "rejected_or_missed": 0,
        },
    }

    category_rows: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "true_equivalent_count": 0,
            "predicted_equivalent_tp": 0,
            "predicted_equivalent_fp": 0,
            "safety_violations": 0,
            "review_count": 0,
        }
    )

    pair_keys_by_route = defaultdict(list)
    for pair_key, prediction in scored_pairs_map.items():
        pair_keys_by_route[str(prediction["route"])].append(pair_key)

    for route_name in ["auto_accept_equivalent", "auto_accept_substitute", "review", "reject"]:
        route_pairs = pair_keys_by_route.get(route_name, [])
        true_equivalent_count = 0
        true_substitute_count = 0
        true_non_match_count = 0
        unknown_count = 0
        for pair_key in route_pairs:
            truth = truth_map.get(pair_key)
            if truth is None:
                unknown_count += 1
            elif truth["true_label"] in EQUIVALENT_LABELS:
                true_equivalent_count += 1
            elif truth["true_label"] == "substitute":
                true_substitute_count += 1
            elif truth["true_label"] == "non_match":
                true_non_match_count += 1
        route_rows.append(
            {
                "route": route_name,
                "total_count": len(route_pairs),
                "true_equivalent_count": true_equivalent_count,
                "true_substitute_count": true_substitute_count,
                "true_non_match_count": true_non_match_count,
                "unknown_count": unknown_count,
            }
        )

    for pair_key, truth in truth_map.items():
        true_label = truth["true_label"]
        prediction = get_prediction_for_pair(scored_pairs_map, pair_key)
        category_key = infer_pair_category(pair_key, products_by_id, prediction)

        if true_label in by_match_type_rows:
            by_match_type_rows[true_label]["true_count"] += 1

        if true_label in EQUIVALENT_LABELS:
            category_rows[category_key]["true_equivalent_count"] += 1

        if true_label in EQUIVALENT_LABELS and prediction is not None:
            true_equivalent_pairs_seen_by_blocking += 1
        if true_label == "substitute" and prediction is not None:
            true_substitute_pairs_seen_by_blocking += 1

        if true_label in EQUIVALENT_LABELS:
            if prediction is None:
                equivalent_fn += 1
                by_match_type_rows[true_label]["rejected_or_missed"] += 1
                error_cases.append(
                    build_error_case("false_negative_equivalent", pair_key, truth, prediction, products_by_id)
                )
            elif prediction["route"] == "auto_accept_equivalent" and prediction["match_type"] in EQUIVALENT_LABELS:
                equivalent_tp += 1
                by_match_type_rows[true_label]["auto_accepted_correct"] += 1
                category_rows[category_key]["predicted_equivalent_tp"] += 1
            elif prediction["route"] == "review":
                equivalent_fn += 1
                true_equivalent_routed_to_review += 1
                by_match_type_rows[true_label]["routed_to_review"] += 1
                category_rows[category_key]["review_count"] += 1
                error_cases.append(
                    build_error_case("review_instead_of_auto", pair_key, truth, prediction, products_by_id)
                )
                error_cases.append(
                    build_error_case("false_negative_equivalent", pair_key, truth, prediction, products_by_id)
                )
            else:
                equivalent_fn += 1
                by_match_type_rows[true_label]["rejected_or_missed"] += 1
                error_cases.append(
                    build_error_case("false_negative_equivalent", pair_key, truth, prediction, products_by_id)
                )
                if prediction["route"] == "auto_accept_substitute":
                    equivalent_predicted_as_substitute += 1

        elif true_label == "substitute":
            if prediction is None:
                substitute_fn += 1
                by_match_type_rows["substitute"]["rejected_or_missed"] += 1
                error_cases.append(
                    build_error_case("false_negative_substitute", pair_key, truth, prediction, products_by_id)
                )
            elif prediction["route"] == "auto_accept_substitute" and prediction["match_type"] == "substitute":
                substitute_tp += 1
                by_match_type_rows["substitute"]["auto_accepted_correct"] += 1
            elif prediction["route"] == "review":
                substitute_fn += 1
                true_substitute_routed_to_review += 1
                by_match_type_rows["substitute"]["routed_to_review"] += 1
                error_cases.append(
                    build_error_case("review_instead_of_auto", pair_key, truth, prediction, products_by_id)
                )
                error_cases.append(
                    build_error_case("false_negative_substitute", pair_key, truth, prediction, products_by_id)
                )
            else:
                substitute_fn += 1
                by_match_type_rows["substitute"]["rejected_or_missed"] += 1
                error_cases.append(
                    build_error_case("false_negative_substitute", pair_key, truth, prediction, products_by_id)
                )
                if prediction["route"] == "auto_accept_equivalent":
                    substitute_predicted_as_equivalent += 1

        elif true_label == "non_match":
            if prediction is not None:
                total_known_negative_pairs_that_appear_in_scored_pairs += 1
                if prediction["route"] == "auto_accept_equivalent":
                    known_negative_auto_accepted_as_equivalent += 1
                    by_match_type_rows["non_match"]["auto_accepted_wrong"] += 1
                    category_rows[category_key]["safety_violations"] += 1
                    error_cases.append(
                        build_error_case("safety_violation", pair_key, truth, prediction, products_by_id)
                    )
                elif prediction["route"] == "auto_accept_substitute":
                    known_negative_auto_accepted_as_substitute += 1
                    by_match_type_rows["non_match"]["auto_accepted_wrong"] += 1
                    category_rows[category_key]["safety_violations"] += 1
                    error_cases.append(
                        build_error_case("safety_violation", pair_key, truth, prediction, products_by_id)
                    )
                    error_cases.append(
                        build_error_case("false_positive_substitute", pair_key, truth, prediction, products_by_id)
                    )
                elif prediction["route"] == "review":
                    known_negative_routed_to_review += 1
                    by_match_type_rows["non_match"]["routed_to_review"] += 1
                elif prediction["route"] == "reject":
                    known_negative_rejected += 1
                    by_match_type_rows["non_match"]["auto_accepted_correct"] += 1
            else:
                by_match_type_rows["non_match"]["rejected_or_missed"] += 1

    for pair_key, prediction in scored_pairs_map.items():
        truth = truth_map.get(pair_key)
        true_label = None if truth is None else truth["true_label"]
        category_key = infer_pair_category(pair_key, products_by_id, prediction)

        if truth is None:
            if prediction["route"] == "auto_accept_equivalent":
                unknown_auto_accept_equivalent_count += 1
            elif prediction["route"] == "auto_accept_substitute":
                unknown_auto_accept_substitute_count += 1
            elif prediction["route"] == "review":
                unknown_review_count += 1
            elif prediction["route"] == "reject":
                unknown_reject_count += 1

        if prediction["route"] == "auto_accept_equivalent":
            if true_label not in EQUIVALENT_LABELS:
                if true_label in {"substitute", "non_match"}:
                    equivalent_fp += 1
                    category_rows[category_key]["predicted_equivalent_fp"] += 1
                    error_cases.append(
                        build_error_case("false_positive_equivalent", pair_key, truth, prediction, products_by_id)
                    )
                    if true_label == "substitute":
                        substitute_predicted_as_equivalent += 1
                elif true_label is None:
                    pass
        elif prediction["route"] == "auto_accept_substitute":
            if true_label != "substitute":
                if true_label in EQUIVALENT_LABELS or true_label == "non_match":
                    substitute_fp += 1
                    error_cases.append(
                        build_error_case("false_positive_substitute", pair_key, truth, prediction, products_by_id)
                    )
                    if true_label in EQUIVALENT_LABELS:
                        equivalent_predicted_as_substitute += 1

    equivalent_precision = safe_divide(equivalent_tp, equivalent_tp + equivalent_fp)
    equivalent_recall = safe_divide(equivalent_tp, total_true_equivalent_pairs)
    equivalent_f1 = f1_score(equivalent_precision, equivalent_recall)

    substitute_precision = safe_divide(substitute_tp, substitute_tp + substitute_fp)
    substitute_recall = safe_divide(substitute_tp, total_true_substitute_pairs)
    substitute_f1 = f1_score(substitute_precision, substitute_recall)

    known_negative_auto_accept_rate = safe_divide(
        known_negative_auto_accepted_as_equivalent + known_negative_auto_accepted_as_substitute,
        total_known_negative_pairs_that_appear_in_scored_pairs,
    )

    evaluation_by_match_type_rows = []
    for match_type in ["exact_equivalent", "private_label_equivalent", "substitute", "non_match"]:
        row = by_match_type_rows[match_type]
        if match_type in EQUIVALENT_LABELS:
            precision_value = safe_divide(
                row["auto_accepted_correct"],
                row["auto_accepted_correct"] + row["auto_accepted_wrong"],
            )
            recall_value = safe_divide(row["auto_accepted_correct"], row["true_count"])
        elif match_type == "substitute":
            precision_value = safe_divide(
                row["auto_accepted_correct"],
                row["auto_accepted_correct"] + row["auto_accepted_wrong"],
            )
            recall_value = safe_divide(row["auto_accepted_correct"], row["true_count"])
        else:
            precision_value = safe_divide(
                row["auto_accepted_correct"],
                row["auto_accepted_correct"] + row["auto_accepted_wrong"],
            )
            recall_value = safe_divide(row["auto_accepted_correct"], row["true_count"])

        evaluation_by_match_type_rows.append(
            {
                "match_type": match_type,
                **row,
                "precision_if_applicable": round(precision_value, 4),
                "recall_if_applicable": round(recall_value, 4),
            }
        )

    evaluation_by_category_rows = []
    for category_key in sorted(category_rows):
        row = category_rows[category_key]
        precision_value = safe_divide(
            row["predicted_equivalent_tp"],
            row["predicted_equivalent_tp"] + row["predicted_equivalent_fp"],
        )
        recall_value = safe_divide(row["predicted_equivalent_tp"], row["true_equivalent_count"])
        evaluation_by_category_rows.append(
            {
                "category": category_key,
                **row,
                "equivalent_precision": round(precision_value, 4),
                "equivalent_recall": round(recall_value, 4),
            }
        )

    evaluation_by_route = pd.DataFrame(route_rows)

    top_hard_conflict_types = dict(
        Counter(
            conflict
            for row in scored_pairs.to_dict(orient="records")
            for conflict in parse_json_like(row.get("hard_conflicts_json"), [])
        ).most_common(20)
    )

    report = {
        "overall": {
            "total_scored_pairs": int(len(scored_pairs)),
            "total_predicted_matches": int(len(predicted_matches)),
            "total_true_equivalent_pairs": int(total_true_equivalent_pairs),
            "total_true_substitute_pairs": int(total_true_substitute_pairs),
            "total_known_negative_pairs": int(total_known_negative_pairs),
        },
        "equivalent_metrics": {
            "precision": round(equivalent_precision, 4),
            "recall": round(equivalent_recall, 4),
            "f1": round(equivalent_f1, 4),
            "true_positives": int(equivalent_tp),
            "false_positives": int(equivalent_fp),
            "false_negatives": int(equivalent_fn),
            "true_equivalent_routed_to_review": int(true_equivalent_routed_to_review),
        },
        "substitute_metrics": {
            "precision": round(substitute_precision, 4),
            "recall": round(substitute_recall, 4),
            "f1": round(substitute_f1, 4),
            "true_positives": int(substitute_tp),
            "false_positives": int(substitute_fp),
            "false_negatives": int(substitute_fn),
            "true_substitute_routed_to_review": int(true_substitute_routed_to_review),
            "substitute_predicted_as_equivalent": int(substitute_predicted_as_equivalent),
            "equivalent_predicted_as_substitute": int(equivalent_predicted_as_substitute),
        },
        "safety_metrics": {
            "known_negative_auto_accept_rate": round(known_negative_auto_accept_rate, 4),
            "known_negative_auto_accepted_as_equivalent": int(known_negative_auto_accepted_as_equivalent),
            "known_negative_auto_accepted_as_substitute": int(known_negative_auto_accepted_as_substitute),
            "known_negative_rejected": int(known_negative_rejected),
            "known_negative_routed_to_review": int(known_negative_routed_to_review),
        },
        "routing_metrics": {
            "review_count": int(review_count),
            "review_rate": review_rate,
            "auto_accept_equivalent_count": int(auto_accept_equivalent_count),
            "auto_accept_substitute_count": int(auto_accept_substitute_count),
            "reject_count": int(reject_count),
            "known_negative_routed_to_review": int(known_negative_routed_to_review),
            "unknown_pairs_routed_to_review": int(unknown_review_count),
            "true_equivalent_routed_to_review": int(true_equivalent_routed_to_review),
            "true_substitute_routed_to_review": int(true_substitute_routed_to_review),
        },
        "coverage_metrics": {
            "true_equivalent_pairs_seen_by_blocking": int(true_equivalent_pairs_seen_by_blocking),
            "true_equivalent_pairs_missed_by_blocking": int(total_true_equivalent_pairs - true_equivalent_pairs_seen_by_blocking),
            "true_substitute_pairs_seen_by_blocking": int(true_substitute_pairs_seen_by_blocking),
            "true_substitute_pairs_missed_by_blocking": int(total_true_substitute_pairs - true_substitute_pairs_seen_by_blocking),
        },
        "unknown_pair_metrics": {
            "unknown_auto_accept_equivalent_count": int(unknown_auto_accept_equivalent_count),
            "unknown_auto_accept_substitute_count": int(unknown_auto_accept_substitute_count),
            "unknown_review_count": int(unknown_review_count),
            "unknown_reject_count": int(unknown_reject_count),
        },
        "top_hard_conflict_types": top_hard_conflict_types,
        "label_conflicts": label_conflicts,
        "notes": notes,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    error_cases_df = pd.DataFrame(error_cases, columns=ERROR_CASE_COLUMNS)
    evaluation_by_match_type_df = pd.DataFrame(evaluation_by_match_type_rows)
    evaluation_by_category_df = pd.DataFrame(evaluation_by_category_rows)

    write_json(EVALUATION_REPORT_PATH, report)
    error_cases_df.to_csv(ERROR_CASES_PATH, index=False)
    evaluation_by_match_type_df.to_csv(EVALUATION_BY_MATCH_TYPE_PATH, index=False)
    evaluation_by_category_df.to_csv(EVALUATION_BY_CATEGORY_PATH, index=False)
    evaluation_by_route.to_csv(EVALUATION_BY_ROUTE_PATH, index=False)

    print("\nEvaluation summary")
    print(f"Total scored pairs: {report['overall']['total_scored_pairs']}")
    print(f"Total true equivalent pairs: {report['overall']['total_true_equivalent_pairs']}")
    print(
        "Equivalent precision / recall / F1: "
        f"{report['equivalent_metrics']['precision']:.4f} / "
        f"{report['equivalent_metrics']['recall']:.4f} / "
        f"{report['equivalent_metrics']['f1']:.4f}"
    )
    print(
        "Substitute precision / recall / F1: "
        f"{report['substitute_metrics']['precision']:.4f} / "
        f"{report['substitute_metrics']['recall']:.4f} / "
        f"{report['substitute_metrics']['f1']:.4f}"
    )
    print(
        "Known negative auto-accept rate: "
        f"{report['safety_metrics']['known_negative_auto_accept_rate']:.4f}"
    )
    print(
        "True equivalent pairs missed by blocking: "
        f"{report['coverage_metrics']['true_equivalent_pairs_missed_by_blocking']}"
    )
    print(
        "True substitute pairs missed by blocking: "
        f"{report['coverage_metrics']['true_substitute_pairs_missed_by_blocking']}"
    )
    print(f"Review rate: {report['routing_metrics']['review_rate']:.4f}")
    print(f"Error cases written: {len(error_cases_df)}")


if __name__ == "__main__":
    main()
