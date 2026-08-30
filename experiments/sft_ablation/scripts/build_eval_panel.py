#!/usr/bin/env python3
"""Freeze the intent/dynamic capability panel before model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SEED = 20260814


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category_number(row: dict[str, object]) -> int | None:
    match = re.search(r"Category\s*(\d)", str(row.get("C", "")))
    return int(match.group(1)) if match else None


def scene_type(row: dict[str, object]) -> str:
    label = str(row.get("scene_type", "")).lower()
    if "human" in label or "people" in label:
        return "human"
    if "object" in label:
        return "object"
    return "unknown"


def canonical_answer(row: dict[str, object]) -> str:
    answer = str(row.get("A", "")).replace("\u200b", "").strip().upper()
    compact = re.sub(r"\s+", "", answer)
    question_type = row.get("question_type")
    if question_type == "Single-Choice":
        return compact if compact in {"A", "B", "C", "D"} else ""
    if question_type == "True/False":
        normalized = re.sub(r"[^A-Z]", "", compact)
        return normalized if normalized in {"TRUE", "FALSE"} else ""
    if question_type == "Multiple-Select":
        if not re.fullmatch(r"[A-D\s,.;/]+", answer):
            return ""
        return ",".join(sorted(set(re.findall(r"[A-D]", answer))))
    return re.sub(r"[.。]+$", "", compact)


def sample_by_stratum(
    rows: list[dict[str, object]], quotas: dict[tuple[str, str], int], seed: int
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    selected = []
    for (question_type, answer), quota in quotas.items():
        candidates = [
            row for row in rows
            if row.get("question_type") == question_type
            and canonical_answer(row) == answer
        ]
        candidates.sort(key=lambda row: int(row["index"]))
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda row: int(row["index"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    intent = [
        row for row in benchmark
        if category_number(row) == 3 and scene_type(row) == "human"
    ]
    dynamic = [
        row for row in benchmark
        if category_number(row) == 2 and scene_type(row) == "object"
    ]
    intent_counts = Counter((row.get("question_type"), canonical_answer(row)) for row in intent if canonical_answer(row))
    dynamic_counts = Counter((row.get("question_type"), canonical_answer(row)) for row in dynamic if canonical_answer(row))
    quotas = {
        stratum: min(intent_counts[stratum], dynamic_counts[stratum])
        for stratum in sorted(set(intent_counts) & set(dynamic_counts))
    }

    selected_intent = sample_by_stratum(intent, quotas, SEED + 1)
    selected_dynamic = sample_by_stratum(dynamic, quotas, SEED + 2)
    panel = []
    for capability, rows in [("intent", selected_intent), ("dynamic_non_intent", selected_dynamic)]:
        for row in rows:
            panel.append({**row, "target_capability": capability})
    panel.sort(key=lambda row: (row["target_capability"], int(row["index"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "escher-sft-eval-panel-v1",
        "seed": SEED,
        "selection_uses_model_outcomes": False,
        "source_benchmark": str(args.benchmark.resolve()),
        "source_benchmark_sha256": sha256(args.benchmark),
        "operational_definitions": {
            "intent": "Category 3 and Human-Centric",
            "dynamic_non_intent": "Category 2 and Object-Centric",
        },
        "matching_dimensions": ["question_type", "canonical_answer"],
        "matched_question_type_answer_quotas_per_capability": {
            f"{question_type}|{answer}": quota
            for (question_type, answer), quota in quotas.items()
        },
        "items_per_capability": len(selected_intent),
        "total_items": len(panel),
        "unique_videos": {
            "intent": len({row["P"] for row in selected_intent}),
            "dynamic_non_intent": len({row["P"] for row in selected_dynamic}),
            "overall": len({row["P"] for row in panel}),
        },
        "panel_sha256": sha256(args.output),
    }
    args.metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
