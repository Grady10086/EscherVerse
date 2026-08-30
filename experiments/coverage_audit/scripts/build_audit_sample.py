#!/usr/bin/env python3
"""Freeze the coverage-audit video sample before independent question authoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260813


def largest_remainder(counts: Counter[tuple[str, str]], total: int) -> dict[tuple[str, str], int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args()

    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category_id"]].append(row)
    if sorted(by_category) != ["1", "2", "3", "4", "5", "6"]:
        raise ValueError("Expected exactly the six benchmark categories")

    rng = random.Random(SEED)
    selected: list[dict[str, str]] = []
    allocation_report: dict[str, dict[str, int]] = {}
    for category_id in sorted(by_category):
        category_rows = by_category[category_id]
        strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in category_rows:
            strata[(row["scene_type"], row["question_type"])].append(row)
        allocations = largest_remainder(Counter({key: len(value) for key, value in strata.items()}), args.per_category)
        allocation_report[category_id] = {
            f"{scene_type}|{question_type}": count
            for (scene_type, question_type), count in sorted(allocations.items())
        }
        for key in sorted(strata):
            candidates = sorted(strata[key], key=lambda row: row["sample_id"])
            rng.shuffle(candidates)
            selected.extend(candidates[: allocations[key]])

    selected.sort(key=lambda row: (int(row["category_id"]), row["sample_id"]))
    if len(selected) != 6 * args.per_category:
        raise ValueError("Unexpected sample size")
    if len({row["video"] for row in selected}) != len(selected):
        raise ValueError("The frozen sample must contain unique videos")

    # Randomize exported IDs so their order does not reveal category strata to
    # independent authors. This occurs before any coverage annotation exists.
    rng.shuffle(selected)

    fieldnames = [
        "audit_item_id",
        "source_sample_id",
        "benchmark_index",
        "video",
        "category_id",
        "category",
        "scene_type",
        "question_type",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "audit_item_id": f"audit-{index:04d}",
                    "source_sample_id": row["sample_id"],
                    "benchmark_index": row["benchmark_index"],
                    "video": row["video"],
                    "category_id": row["category_id"],
                    "category": row["category"],
                    "scene_type": row["scene_type"],
                    "question_type": row["question_type"],
                }
            )

    meta = {
        "schema": "escher-audit-frozen-sample-v1",
        "seed": SEED,
        "source_manifest": str(args.source),
        "source_manifest_sha256": sha256(args.source),
        "sample_rows": len(selected),
        "unique_videos": len({row["video"] for row in selected}),
        "per_category": dict(sorted(Counter(row["category_id"] for row in selected).items())),
        "per_scene_type": dict(sorted(Counter(row["scene_type"] for row in selected).items())),
        "per_question_type": dict(sorted(Counter(row["question_type"] for row in selected).items())),
        "stratum_allocations": allocation_report,
        "selection_fields": ["category_id", "scene_type", "question_type"],
        "selection_excluded_fields": ["question", "ground_truth", "model_output", "coverage_annotation"],
        "blind_id_order": "fixed-seed random permutation after stratified selection",
        "output_sha256": sha256(args.output),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
