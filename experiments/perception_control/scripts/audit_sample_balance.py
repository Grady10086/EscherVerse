#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    canonical_category,
    canonical_question_type,
    canonical_scene_type,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    counts = Counter(str(row[key]) for row in rows)
    total = len(rows)
    return {
        value: {"count": count, "rate": count / total if total else None}
        for value, count in sorted(counts.items())
    }


def max_rate_difference(
    full_distribution: dict[str, dict[str, Any]],
    sample_distribution: dict[str, dict[str, Any]],
) -> float:
    values = set(full_distribution) | set(sample_distribution)
    return max(
        (
            abs(
                float(sample_distribution.get(value, {}).get("rate", 0.0))
                - float(full_distribution.get(value, {}).get("rate", 0.0))
            )
            for value in values
        ),
        default=0.0,
    )


def audit(
    benchmark_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
) -> dict[str, Any]:
    full_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        try:
            category_id, _ = canonical_category(str(row.get("C", "")))
        except ValueError:
            continue
        full_by_category[category_id].append(
            {
                "scene_type": canonical_scene_type(str(row.get("scene_type", ""))),
                "question_type": canonical_question_type(
                    str(row.get("question_type", ""))
                ),
            }
        )

    sample_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        sample_by_category[int(row["category_id"])].append(
            {
                "scene_type": canonical_scene_type(row["scene_type"]),
                "question_type": canonical_question_type(row["question_type"]),
            }
        )

    categories = {}
    global_max = 0.0
    for category_id in range(1, 7):
        full_rows = full_by_category[category_id]
        sample_rows = sample_by_category[category_id]
        dimensions = {}
        category_max = 0.0
        for key in ("scene_type", "question_type"):
            full_distribution = distribution(full_rows, key)
            sample_distribution = distribution(sample_rows, key)
            difference = max_rate_difference(full_distribution, sample_distribution)
            category_max = max(category_max, difference)
            dimensions[key] = {
                "full": full_distribution,
                "sample": sample_distribution,
                "max_absolute_rate_difference": difference,
            }
        global_max = max(global_max, category_max)
        categories[str(category_id)] = {
            "full_n": len(full_rows),
            "sample_n": len(sample_rows),
            "max_absolute_rate_difference": category_max,
            "dimensions": dimensions,
        }

    return {
        "benchmark_rows": len(benchmark_rows),
        "canonical_benchmark_rows": sum(map(len, full_by_category.values())),
        "manifest_rows": len(manifest_rows),
        "sampling_rule": "100 unique-video items per canonical category",
        "global_max_absolute_rate_difference": global_max,
        "categories": categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit scene and question-type balance in the frozen sample."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = audit(benchmark_rows, read_manifest(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_rows": report["manifest_rows"],
                "global_max_absolute_rate_difference": report[
                    "global_max_absolute_rate_difference"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
