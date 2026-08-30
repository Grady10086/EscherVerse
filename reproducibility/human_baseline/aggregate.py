#!/usr/bin/env python3
"""Aggregate anonymized first-pass human judgments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "correct"}
FALSE_VALUES = {"0", "false", "no", "incorrect"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("human_baseline_summary.json"))
    parser.add_argument("--expected-annotators", type=int, default=11)
    parser.add_argument("--expected-items-per-annotator", type=int, default=8000)
    return parser.parse_args()


def parse_correct(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognized is_correct value: {value!r}")


def main() -> None:
    args = parse_args()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"annotator_id", "is_correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("Input CSV requires annotator_id and is_correct columns")
        for row in reader:
            correct = parse_correct(row["is_correct"])
            counts[row["annotator_id"]][0] += int(correct)
            counts[row["annotator_id"]][1] += 1

    if len(counts) != args.expected_annotators:
        raise SystemExit(f"Expected {args.expected_annotators} annotators, found {len(counts)}")
    wrong_sizes = {key: total for key, (_, total) in counts.items() if total != args.expected_items_per_annotator}
    if wrong_sizes:
        raise SystemExit(f"Unexpected item counts: {wrong_sizes}")

    per_annotator = [
        {"annotator_id": key, "correct": correct, "total": total, "accuracy": correct / total}
        for key, (correct, total) in sorted(counts.items())
    ]
    accuracies = [row["accuracy"] for row in per_annotator]
    summary = {
        "annotators": len(per_annotator),
        "items_per_annotator": args.expected_items_per_annotator,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "minimum_accuracy": min(accuracies),
        "maximum_accuracy": max(accuracies),
        "per_annotator": per_annotator,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
