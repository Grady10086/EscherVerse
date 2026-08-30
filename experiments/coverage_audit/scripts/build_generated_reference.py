#!/usr/bin/env python3
"""Build all same-video final questions for the frozen coverage-audit sample."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_CAPABILITY = {
    "1": "occlusion_permanence",
    "2": "dynamic_relation",
    "3": "action_goal",
    "4": "physical_prediction",
    "5": "deformation_state",
    "6": "reference_frame",
}


def category_id(value: str) -> str:
    match = re.search(r"Category\s+([1-6])", value)
    if not match:
        raise ValueError(f"Cannot parse category: {value}")
    return match.group(1)


def bool_int(value: bool) -> int:
    return int(bool(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.sample.open(newline="", encoding="utf-8") as handle:
        sample = list(csv.DictReader(handle))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in benchmark:
        by_video[row["P"]].append(row)

    fields = [
        "audit_item_id", "generated_id", "benchmark_index", "video", "question",
        "answer", "category_id", "primary_capability", "scene_type",
        "question_type", "intent_required", "reference_transform",
        "prediction_or_counterfactual", "temporal_integration",
        "occlusion_tracking", "deformation_tracking", "reasoning_level",
    ]
    output_rows = []
    for sample_row in sample:
        final_rows = sorted(by_video[sample_row["video"]], key=lambda row: int(row["index"]))
        if not final_rows:
            raise ValueError(f"No final question for {sample_row['video']}")
        for ordinal, row in enumerate(final_rows, start=1):
            cid = category_id(str(row["C"]))
            text = str(row["Q"]).lower()
            temporal = (
                cid in {"1", "2", "3", "4", "5"}
                or any(token in text for token in ["before", "after", "between", "throughout", "during", "then"])
            )
            output_rows.append({
                "audit_item_id": sample_row["audit_item_id"],
                "generated_id": f"{sample_row['audit_item_id']}-g{ordinal:02d}",
                "benchmark_index": row["index"],
                "video": row["P"],
                "question": row["Q"],
                "answer": row["A"],
                "category_id": cid,
                "primary_capability": CATEGORY_CAPABILITY[cid],
                "scene_type": row["scene_type"],
                "question_type": row["question_type"],
                "intent_required": bool_int(cid == "3"),
                "reference_transform": bool_int(cid == "6"),
                "prediction_or_counterfactual": bool_int(cid == "4"),
                "temporal_integration": bool_int(temporal),
                "occlusion_tracking": bool_int(cid == "1"),
                "deformation_tracking": bool_int(cid == "5"),
                "reasoning_level": "high" if cid in {"3", "4", "6"} else "intermediate",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    per_video = Counter(row["audit_item_id"] for row in output_rows)
    summary = {
        "sample_videos": len(sample),
        "generated_final_questions": len(output_rows),
        "questions_per_video": dict(sorted(Counter(per_video.values()).items())),
        "videos_with_multiple_questions": sum(value > 1 for value in per_video.values()),
        "capability_distribution": dict(sorted(Counter(row["primary_capability"] for row in output_rows).items())),
        "annotation_note": "Primary capability and cross-cut flags are deterministic category-aligned mappings, not fresh semantic judgments.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
