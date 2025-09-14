#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jsonl_dataset_validator_aliases.py

Validates a JSONL dataset with flexible key names.

It accepts common aliases, e.g.:
- "entity type", "Entity Type", "entity_type", "entity-type", "entityType" → canonical "entity_type"
- "Examples", "examples" → "examples"
- "Negatives", "negatives" → "negatives"
- "Property", "property" → "property"
- (optional) "Positive", "positive" → "positive"

Checks per row:
- Required keys exist: "property", "entity_type", "examples", "negatives" (and optional "positive" if --expect-positive)
- Types: property/entity_type/positive are strings; examples/negatives are lists of strings
- Counts: examples == N (default 7), negatives == M (default 4)
- Non-empty trimmed strings
- No overlap between examples and negatives (case-insensitive)
- Duplicates within examples/negatives (case-insensitive) flagged unless --allow-duplicates
- Warn on extra/unexpected keys (unless --no-warn-extra-keys)

Outputs:
- Console summary
- CSV + JSON reports with details (only when issues exist)

Usage:
  python jsonl_dataset_validator_aliases.py \
    --input /path/to/augmented_list_of_entities_and_negatives.jsonl \
    --examples 7 \
    --negatives 4
"""

import os
import sys
import csv
import json
import argparse
from typing import Any, Dict, List, Tuple

# ---------- Configurable alias map ----------
# Map any key variant to a canonical field name
KEY_ALIASES = {
    # property
    "property": "property",
    "Property": "property",

    # entity_type (canonical)
    "entity type": "entity_type",
    "Entity type": "entity_type",
    "Entity Type": "entity_type",
    "entity_type": "entity_type",
    "entity-type": "entity_type",
    "entitytype": "entity_type",
    "entityType": "entity_type",

    # examples
    "examples": "examples",
    "Examples": "examples",

    # negatives
    "negatives": "negatives",
    "Negatives": "negatives",

    # optional positive
    "positive": "positive",
    "Positive": "positive",
}

CANONICAL_REQUIRED = {"property", "entity_type", "examples", "negatives"}

def is_str(x: Any) -> bool:
    return isinstance(x, str)

def norm_str(s: str) -> str:
    return s.strip()

def to_canonical_keys(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (canonical, extra_keys_used).

    canonical: dict with canonical keys for any recognized aliases; unrecognized keys are kept as-is.
    extra_keys_used: list of original keys that were not recognized as aliases and not canonical names.
    """
    canonical: Dict[str, Any] = {}
    extras: List[str] = []

    for k, v in obj.items():
        if k in KEY_ALIASES:
            dst = KEY_ALIASES[k]
            # If both an alias and canonical already exist, prefer canonical and ignore alias overwrite
            if dst in canonical:
                # Don't clobber; if you want to catch conflicts, you could log here.
                continue
            canonical[dst] = v
        else:
            # Keep unrecognized keys too; we may warn about them later
            canonical[k] = v
            # track as "extra" only if not itself a canonical name
            if k not in CANONICAL_REQUIRED and k != "positive":
                extras.append(k)

    return canonical, extras

def find_duplicates(items: List[str], case_insensitive: bool = True) -> List[str]:
    seen = set()
    dupes = set()
    for v in items:
        t = v.lower() if case_insensitive else v
        if t in seen:
            dupes.add(t)
        else:
            seen.add(t)
    return sorted(list(dupes))

def validate_row(
    obj: Dict[str, Any],
    index_1based: int,
    expected_examples: int,
    expected_negatives: int,
    expect_positive: bool,
    warn_extra_keys: bool,
    enforce_uniques: bool,
    case_insensitive: bool = True
) -> List[str]:
    """
    Return a list of issue strings for this row (empty list means OK).
    """
    issues: List[str] = []

    # 1) Canonicalize keys (handle "entity type" → "entity_type", etc.)
    obj_can, extra_keys = to_canonical_keys(obj)

    # 2) Required keys present?
    required = set(CANONICAL_REQUIRED)
    if expect_positive:
        required.add("positive")

    missing = [k for k in required if k not in obj_can]
    if missing:
        issues.append(f"Missing required keys: {missing}")

    # 3) Warn about extras (that are not recognized aliases)
    if warn_extra_keys and extra_keys:
        issues.append(f"Unexpected extra keys: {extra_keys}")

    # 4) Type + content checks

    # property
    if "property" in obj_can:
        if not is_str(obj_can["property"]):
            issues.append("property must be a string")
        elif not norm_str(obj_can["property"]):
            issues.append("property is an empty string after trimming")

    # entity_type
    if "entity_type" in obj_can:
        if not is_str(obj_can["entity_type"]):
            issues.append("entity_type must be a string")
        elif not norm_str(obj_can["entity_type"]):
            issues.append("entity_type is an empty string after trimming")

    # positive (optional)
    if expect_positive and "positive" in obj_can:
        if not is_str(obj_can["positive"]):
            issues.append("positive must be a string")
        elif not norm_str(obj_can["positive"]):
            issues.append("positive is an empty string after trimming")

    # examples
    ex_norm: List[str] = []
    if "examples" in obj_can:
        if not isinstance(obj_can["examples"], list):
            issues.append("examples must be a list of strings")
        else:
            # type check and trim
            bad_types = [i for i, v in enumerate(obj_can["examples"]) if not is_str(v)]
            if bad_types:
                issues.append(f"examples contains non-string items at indices {bad_types}")
            ex_norm = [norm_str(v) for v in obj_can["examples"] if is_str(v)]
            empties = [i for i, v in enumerate(ex_norm) if v == ""]
            if empties:
                issues.append(f"examples contains empty strings after trim at indices {empties}")
            # length
            if len(obj_can["examples"]) != expected_examples:
                issues.append(f"examples must contain exactly {expected_examples} items (found {len(obj_can['examples'])})")
            # uniqueness
            if enforce_uniques:
                d = find_duplicates([v for v in ex_norm if v != ""], case_insensitive=case_insensitive)
                if d:
                    issues.append(f"examples contains duplicates (case-insensitive): {d}")

    # negatives
    neg_norm: List[str] = []
    if "negatives" in obj_can:
        if not isinstance(obj_can["negatives"], list):
            issues.append("negatives must be a list of strings")
        else:
            bad_types = [i for i, v in enumerate(obj_can["negatives"]) if not is_str(v)]
            if bad_types:
                issues.append(f"negatives contains non-string items at indices {bad_types}")
            neg_norm = [norm_str(v) for v in obj_can["negatives"] if is_str(v)]
            empties = [i for i, v in enumerate(neg_norm) if v == ""]
            if empties:
                issues.append(f"negatives contains empty strings after trim at indices {empties}")
            if len(obj_can["negatives"]) != expected_negatives:
                issues.append(f"negatives must contain exactly {expected_negatives} items (found {len(obj_can['negatives'])})")
            if enforce_uniques:
                d = find_duplicates([v for v in neg_norm if v != ""], case_insensitive=case_insensitive)
                if d:
                    issues.append(f"negatives contains duplicates (case-insensitive): {d}")

    # Overlap (only check if both lists exist)
    if ex_norm and neg_norm:
        ex_set = set(v.lower() if case_insensitive else v for v in ex_norm if v)
        neg_set = set(v.lower() if case_insensitive else v for v in neg_norm if v)
        inter = sorted(list(ex_set.intersection(neg_set)))
        if inter:
            issues.append(f"overlap between examples and negatives (case-insensitive): {inter}")

    return issues

def validate_file(
    input_path: str,
    expected_examples: int,
    expected_negatives: int,
    expect_positive: bool,
    warn_extra_keys: bool,
    enforce_uniques: bool,
    case_insensitive: bool,
    report_prefix: str | None = None
):
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}", flush=True)
        return 1, []

    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    prefix = report_prefix or f"{base_name}_validation"

    issues_report: List[Dict[str, Any]] = []
    total_lines = 0
    total_ok = 0
    total_fail = 0
    total_json_errors = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            total_lines += 1
            s = line.strip()
            if not s:
                issues_report.append({
                    "line": i,
                    "error_count": 1,
                    "errors": ["blank line"],
                })
                total_fail += 1
                continue

            try:
                obj = json.loads(s)
            except Exception as e:
                issues_report.append({
                    "line": i,
                    "error_count": 1,
                    "errors": [f"invalid JSON: {e}"],
                    "raw": s,
                })
                total_fail += 1
                total_json_errors += 1
                continue

            if not isinstance(obj, dict):
                issues_report.append({
                    "line": i,
                    "error_count": 1,
                    "errors": ["top-level JSON must be an object"],
                    "raw": s,
                })
                total_fail += 1
                continue

            errs = validate_row(
                obj=obj,
                index_1based=i,
                expected_examples=expected_examples,
                expected_negatives=expected_negatives,
                expect_positive=expect_positive,
                warn_extra_keys=warn_extra_keys,
                enforce_uniques=enforce_uniques,
                case_insensitive=case_insensitive,
            )

            if errs:
                issues_report.append({
                    "line": i,
                    "error_count": len(errs),
                    "errors": errs,
                    "raw": s,
                })
                total_fail += 1
            else:
                total_ok += 1

    # Write reports if there are issues
    if issues_report:
        csv_path = os.path.join(base_dir, f"{prefix}.csv")
        json_path = os.path.join(base_dir, f"{prefix}.json")

        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["line", "error_count", "errors"])
            for row in issues_report:
                writer.writerow([row["line"], row["error_count"], " | ".join(row["errors"])])

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "file": input_path,
                "summary": {
                    "total_lines": total_lines,
                    "ok": total_ok,
                    "failed": total_fail,
                    "json_parse_errors": total_json_errors,
                },
                "issues": issues_report,
            }, jf, ensure_ascii=False, indent=2)

        print(f"\nReports written:\n  - {csv_path}\n  - {json_path}")

    # Console summary
    print("\n=== Validation Summary ===")
    print(f"File: {input_path}")
    print(f"Total rows: {total_lines}")
    print(f"Valid rows: {total_ok}")
    print(f"Invalid rows: {total_fail}")
    print(f"JSON parse errors: {total_json_errors}")

    return (0 if total_fail == 0 else 1), issues_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL file")
    parser.add_argument("--examples", type=int, default=7, help="Expected number of examples (default: 7)")
    parser.add_argument("--negatives", type=int, default=4, help="Expected number of negatives (default: 4)")
    parser.add_argument("--expect-positive", action="store_true", help="Require a 'positive' string field")
    parser.add_argument("--no-warn-extra-keys", action="store_true", help="Do not warn about unexpected keys")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicates within examples/negatives")
    parser.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive checks for duplicates/overlap")
    parser.add_argument("--report-prefix", default=None, help="Basename for report files (no extension)")
    args = parser.parse_args()

    code, _ = validate_file(
        input_path=args.input,
        expected_examples=args.examples,
        expected_negatives=args.negatives,
        expect_positive=args.expect_positive,
        warn_extra_keys=(not args.no_warn_extra_keys),
        enforce_uniques=(not args.allow_duplicates),
        case_insensitive=(not args.case_sensitive),
        report_prefix=args.report_prefix,
    )
    sys.exit(code)

if __name__ == "__main__":
    main()
