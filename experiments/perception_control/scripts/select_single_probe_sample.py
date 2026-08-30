#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


CATEGORY_QUOTAS = {1: 34, 2: 34, 3: 33, 4: 33, 5: 33, 6: 33}
PROBE_QUOTAS = {
    1: {"entity": 12, "action_event": 11, "simple_relation": 11},
    2: {"entity": 11, "action_event": 12, "simple_relation": 11},
    3: {"entity": 11, "action_event": 11, "simple_relation": 11},
    4: {"entity": 11, "action_event": 11, "simple_relation": 11},
    5: {"entity": 11, "action_event": 11, "simple_relation": 11},
    6: {"entity": 11, "action_event": 11, "simple_relation": 11},
}
FIELDNAMES = (
    "probe_sample_id",
    "source_sample_id",
    "benchmark_index",
    "video",
    "original_question",
    "original_ground_truth",
    "category_id",
    "category",
    "scene_type",
    "question_type",
    "assigned_probe_type",
    "assigned_probe_subtype",
    "target_answer_position",
    "source_sample_sha256",
    "selection_sha256",
)


def stable_hash(seed: str, *parts: object) -> str:
    text = "\x1f".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def largest_remainder_quotas(
    counts: dict[tuple[str, str], int], total: int
) -> dict[tuple[str, str], int]:
    population = sum(counts.values())
    ideals = {key: total * count / population for key, count in counts.items()}
    quotas = {key: int(value) for key, value in ideals.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(ideals[key] - quotas[key]), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if any(quotas[key] > counts[key] for key in counts):
        raise ValueError("Stratum allocation exceeds available rows")
    return quotas


def select_rows(
    source_rows: list[dict[str, str]], seed: str
) -> list[dict[str, str]]:
    by_category: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_category[int(row["category_id"])].append(row)
    if set(by_category) != set(CATEGORY_QUOTAS):
        raise ValueError("Source manifest must contain all six categories")

    selected: list[dict[str, str]] = []
    for category_id, target in CATEGORY_QUOTAS.items():
        rows = by_category[category_id]
        strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            strata[(row["scene_type"], row["question_type"])].append(row)
        quotas = largest_remainder_quotas(
            {key: len(values) for key, values in strata.items()}, target
        )
        category_selected: list[dict[str, str]] = []
        for key, values in sorted(strata.items()):
            ordered = sorted(
                values,
                key=lambda row: stable_hash(
                    seed, "sample", category_id, key, row["sample_id"]
                ),
            )
            category_selected.extend(ordered[: quotas[key]])

        ordered = sorted(
            category_selected,
            key=lambda row: stable_hash(seed, "probe-type", row["sample_id"]),
        )
        probe_types = [
            probe_type
            for probe_type, count in PROBE_QUOTAS[category_id].items()
            for _ in range(count)
        ]
        for row, probe_type in zip(ordered, probe_types, strict=True):
            updated = dict(row)
            updated["assigned_probe_type"] = probe_type
            selected.append(updated)

    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_type[row["assigned_probe_type"]].append(row)
    for probe_type, rows in by_type.items():
        rows.sort(key=lambda row: stable_hash(seed, "subtype", row["sample_id"]))
        for position, row in enumerate(rows):
            if probe_type == "entity":
                subtype = "object_actor_presence"
            elif probe_type == "action_event":
                subtype = "action_recognition" if position % 2 == 0 else "temporal_order"
            else:
                subtype = "orientation" if position % 2 == 0 else "simple_spatial_relation"
            row["assigned_probe_subtype"] = subtype

    answer_order = sorted(
        selected,
        key=lambda row: stable_hash(seed, "answer-position", row["sample_id"]),
    )
    for position, row in enumerate(answer_order):
        row["target_answer_position"] = "ABCD"[position % 4]

    selected.sort(key=lambda row: stable_hash(seed, "final-order", row["sample_id"]))
    output = []
    for position, row in enumerate(selected, start=1):
        selection_sha = stable_hash(
            seed,
            row["sample_id"],
            row["sample_sha256"],
            row["assigned_probe_type"],
            row["assigned_probe_subtype"],
            row["target_answer_position"],
        )
        output.append(
            {
                "probe_sample_id": f"probe-{position:04d}",
                "source_sample_id": row["sample_id"],
                "benchmark_index": row["benchmark_index"],
                "video": row["video"],
                "original_question": row["question"],
                "original_ground_truth": row["ground_truth"],
                "category_id": row["category_id"],
                "category": row["category"],
                "scene_type": row["scene_type"],
                "question_type": row["question_type"],
                "assigned_probe_type": row["assigned_probe_type"],
                "assigned_probe_subtype": row["assigned_probe_subtype"],
                "target_answer_position": row["target_answer_position"],
                "source_sample_sha256": row["sample_sha256"],
                "selection_sha256": selection_sha,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_rows = read_rows(args.source_manifest)
    selected = select_rows(source_rows, args.seed)
    write_csv(args.output, selected)
    report = {
        "schema_version": "perception-single-probe-sample-v1",
        "seed": args.seed,
        "source_manifest": {
            "path": str(args.source_manifest),
            "sha256": file_sha256(args.source_manifest),
            "rows": len(source_rows),
        },
        "output_manifest": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
            "rows": len(selected),
        },
        "category_counts": dict(
            sorted(Counter(row["category_id"] for row in selected).items())
        ),
        "probe_type_counts": dict(
            sorted(Counter(row["assigned_probe_type"] for row in selected).items())
        ),
        "probe_subtype_counts": dict(
            sorted(Counter(row["assigned_probe_subtype"] for row in selected).items())
        ),
        "answer_position_counts": dict(
            sorted(Counter(row["target_answer_position"] for row in selected).items())
        ),
        "unique_videos": len({row["video"] for row in selected}),
        "unique_benchmark_indices": len(
            {row["benchmark_index"] for row in selected}
        ),
        "selection_used_model_outcomes": False,
    }
    args.meta.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a balanced 200-video, one-probe-per-video perception-control sample."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--seed", default="20260812-perception-single-probe-200")
    return parser.parse_args(argv)


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
