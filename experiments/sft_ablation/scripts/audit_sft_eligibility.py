#!/usr/bin/env python3
"""Audit SFT-ablation eligibility and benchmark-video overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(value: str) -> str:
    value = " ".join(str(value or "").split())
    aliases = {
        "Egocentric vs. Allocentric Reference Frames": "Category 6: Egocentric vs. Allocentric Reference Frames",
    }
    return aliases.get(value, value)


def video_basename(row: dict[str, object]) -> str:
    videos = row.get("videos") or []
    if not isinstance(videos, list) or len(videos) != 1:
        return ""
    return Path(str(videos[0])).name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_videos = {Path(str(row["P"])).name for row in benchmark}
    rows = []
    parse_errors = []
    with args.sft.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "error": str(exc)})
                continue
            metadata = row.get("metadata") or {}
            row["_line"] = line_number
            row["_video"] = video_basename(row)
            row["_category"] = normalize_category(metadata.get("category", ""))
            row["_scene_type"] = " ".join(str(metadata.get("scene_type", "")).split())
            row["_question_type"] = " ".join(str(metadata.get("question_type", "")).split())
            rows.append(row)

    valid_schema = [
        row for row in rows
        if row["_video"] and row["_category"] and row["_scene_type"]
        and isinstance(row.get("messages"), list) and len(row["messages"]) >= 2
    ]
    no_eval_overlap = [row for row in valid_schema if row["_video"] not in benchmark_videos]

    def intent(row: dict[str, object]) -> bool:
        return row["_category"].startswith("Category 3:") and row["_scene_type"] == "Human-Centric"

    def dynamic_non_intent(row: dict[str, object]) -> bool:
        return row["_category"].startswith("Category 2:") and row["_scene_type"] == "Object-Centric"

    def dynamic_all_scenes(row: dict[str, object]) -> bool:
        return row["_category"].startswith("Category 2:")

    def summarize(group: list[dict[str, object]]) -> dict[str, object]:
        video_counts = Counter(row["_video"] for row in group)
        return {
            "rows": len(group),
            "unique_videos": len(video_counts),
            "rows_per_video": dict(sorted(Counter(video_counts.values()).items())),
            "categories": dict(sorted(Counter(row["_category"] for row in group).items())),
            "scene_types": dict(sorted(Counter(row["_scene_type"] for row in group).items())),
            "question_types": dict(sorted(Counter(row["_question_type"] for row in group).items())),
        }

    categories = Counter(row["_category"] for row in valid_schema)
    expected_categories = {f"Category {index}:" for index in range(1, 7)}
    unknown_categories = {
        key: value
        for key, value in categories.items()
        if not any(key.startswith(prefix) for prefix in expected_categories)
    }
    duplicate_fingerprints = Counter(
        (
            row["_video"],
            json.dumps(row.get("messages"), sort_keys=True, ensure_ascii=False),
        )
        for row in valid_schema
    )
    duplicate_rows = sum(value - 1 for value in duplicate_fingerprints.values() if value > 1)

    report = {
        "schema": "escher-sft-eligibility-audit-v1",
        "sources": {
            "sft": str(args.sft),
            "sft_sha256": sha256(args.sft),
            "benchmark": str(args.benchmark),
            "benchmark_sha256": sha256(args.benchmark),
        },
        "raw_lines": sum(1 for _ in args.sft.open(encoding="utf-8")),
        "parsed_rows": len(rows),
        "parse_errors": parse_errors,
        "valid_schema_rows": len(valid_schema),
        "invalid_schema_rows": len(rows) - len(valid_schema),
        "exact_duplicate_extra_rows": duplicate_rows,
        "all_valid": summarize(valid_schema),
        "benchmark_unique_videos": len(benchmark_videos),
        "sft_rows_sharing_benchmark_video": sum(row["_video"] in benchmark_videos for row in valid_schema),
        "sft_unique_videos_sharing_benchmark_video": len({row["_video"] for row in valid_schema} & benchmark_videos),
        "no_benchmark_video_overlap": summarize(no_eval_overlap),
        "groups_all_valid": {
            "intent_human_category3": summarize([row for row in valid_schema if intent(row)]),
            "dynamic_object_category2": summarize([row for row in valid_schema if dynamic_non_intent(row)]),
            "dynamic_all_scenes_category2_sensitivity": summarize([row for row in valid_schema if dynamic_all_scenes(row)]),
        },
        "groups_no_benchmark_video_overlap": {
            "intent_human_category3": summarize([row for row in no_eval_overlap if intent(row)]),
            "dynamic_object_category2": summarize([row for row in no_eval_overlap if dynamic_non_intent(row)]),
            "dynamic_all_scenes_category2_sensitivity": summarize([row for row in no_eval_overlap if dynamic_all_scenes(row)]),
        },
        "normalized_category_counts": dict(sorted(categories.items())),
        "unknown_category_counts": unknown_categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
