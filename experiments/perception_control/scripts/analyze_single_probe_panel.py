#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import exact_mcnemar_p_value, wilson_interval


def read_items(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def read_result_dir(path: Path) -> tuple[str, list[dict[str, Any]]]:
    files = sorted(path.glob("**/results_*.json"))
    if len(files) != 1:
        raise ValueError(f"{path}: expected exactly one result JSON, found {len(files)}")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{files[0]} has no result list")
    model = str(payload.get("metadata", {}).get("model", path.name))
    return model, rows


def metric(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    total = len(values)
    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "accuracy": successes / total if total else None,
        "wilson_95_ci": [low, high] if total else [None, None],
    }


def index_results(
    rows: list[dict[str, Any]],
    expected: set[int],
    label: str,
    conservative_max_tokens: bool,
) -> tuple[dict[int, bool], int]:
    indexed: dict[int, bool] = {}
    max_token_rows = 0
    for row in rows:
        index = int(row["index"])
        if index in indexed:
            raise ValueError(f"{label}: duplicate index {index}")
        if row.get("is_correct") is None:
            raise ValueError(f"{label}: inference error at index {index}")
        hit_max_tokens = bool(
            (row.get("generation_metadata") or {}).get("hit_max_tokens", False)
        )
        max_token_rows += hit_max_tokens
        indexed[index] = bool(row["is_correct"]) and not (
            conservative_max_tokens and hit_max_tokens
        )
    if set(indexed) != expected:
        raise ValueError(f"{label}: result indices do not match the manifest")
    return indexed, max_token_rows


def analyze_model(
    controls: list[dict[str, Any]],
    originals: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    model: str,
    conservative_max_tokens: bool = True,
) -> dict[str, Any]:
    original_by_index = {int(item["index"]): item for item in originals}
    expected = {int(item["index"]) for item in controls}
    if expected != set(original_by_index):
        raise ValueError("control and original manifests have different indices")
    for item in controls:
        counterpart = original_by_index[int(item["index"])]
        if item["P"] != counterpart["P"]:
            raise ValueError(f"video mismatch at index {item['index']}")

    control, control_max_token_rows = index_results(
        control_rows, expected, f"{model} controls", conservative_max_tokens
    )
    original, original_max_token_rows = index_results(
        original_rows, expected, f"{model} originals", conservative_max_tokens
    )
    ordered = sorted(expected)
    both_correct = sum(control[i] and original[i] for i in ordered)
    control_only = sum(control[i] and not original[i] for i in ordered)
    original_only = sum(not control[i] and original[i] for i in ordered)
    both_wrong = sum(not control[i] and not original[i] for i in ordered)
    control_pass = [i for i in ordered if control[i]]

    by_type: dict[str, list[bool]] = defaultdict(list)
    by_subtype: dict[str, list[bool]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    for item in controls:
        index = int(item["index"])
        by_type[str(item["probe_type"])].append(control[index])
        by_subtype[str(item["probe_subtype"])].append(control[index])
        category = str(original_by_index[index].get("category_id", "unknown"))
        by_category[category].append(index)

    return {
        "model": model,
        "n": len(ordered),
        "conservative_max_tokens_as_incorrect": conservative_max_tokens,
        "max_token_rows": {
            "control": control_max_token_rows,
            "original": original_max_token_rows,
        },
        "control_accuracy": metric([control[i] for i in ordered]),
        "original_tsi_accuracy": metric([original[i] for i in ordered]),
        "accuracy_gap_control_minus_original": (
            sum(control.values()) - sum(original.values())
        ) / len(ordered),
        "conditional_original_accuracy_given_control_pass": metric(
            [original[i] for i in control_pass]
        ),
        "paired_2x2": {
            "control_correct_original_correct": both_correct,
            "control_correct_original_wrong": control_only,
            "control_wrong_original_correct": original_only,
            "control_wrong_original_wrong": both_wrong,
        },
        "exact_mcnemar_p": exact_mcnemar_p_value(control_only, original_only),
        "control_by_type": {
            key: metric(values) for key, values in sorted(by_type.items())
        },
        "control_by_subtype": {
            key: metric(values) for key, values in sorted(by_subtype.items())
        },
        "by_original_category": {
            key: {
                "n": len(indices),
                "control_accuracy": metric([control[i] for i in indices]),
                "original_accuracy": metric([original[i] for i in indices]),
            }
            for key, indices in sorted(by_category.items())
        },
    }


def parse_model_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or ":" not in parts[1]:
        raise argparse.ArgumentTypeError("model must be LABEL=CONTROL_DIR:ORIGINAL_DIR")
    control, original = parts[1].split(":", 1)
    return parts[0], Path(control), Path(original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired single-probe perception-control results.")
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-max-token-parsed",
        action="store_true",
        help="Sensitivity analysis: score parsed answers even when generation hit the limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controls = read_items(args.controls)
    originals = read_items(args.originals)
    reports = []
    for label, control_dir, original_dir in args.model:
        _, control_rows = read_result_dir(control_dir)
        _, original_rows = read_result_dir(original_dir)
        reports.append(
            analyze_model(
                controls,
                originals,
                control_rows,
                original_rows,
                label,
                conservative_max_tokens=not args.allow_max_token_parsed,
            )
        )
    output = {
        "status": "provisional_unreviewed_control_ground_truth",
        "scoring_mode": (
            "parsed_answer_including_max_token_rows"
            if args.allow_max_token_parsed
            else "conservative_max_token_rows_incorrect"
        ),
        "models": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
