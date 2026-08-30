#!/usr/bin/env python3
"""Freeze leakage-free, size- and question-type-matched SFT-ablation subsets."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(value: str) -> str:
    value = " ".join(str(value or "").split())
    if value == "Category 6: Egocentric vs. Allocentric ReferenceFrames":
        return "Category 6: Egocentric vs. Allocentric Reference Frames"
    return value


def infer_question_type(row: dict[str, object]) -> str:
    user_content = str(row["messages"][0].get("content", ""))
    match = re.search(r"\[([^]\n]+)\]", user_content)
    label = match.group(1).strip().lower() if match else ""
    aliases = {
        "single-choice": "single_choice",
        "multiple-choice": "single_choice",
        "true/false": "true_false",
        "fill-in-the-blank": "fill_in_blank",
        "multiple-select": "multiple_select",
    }
    for source, normalized in aliases.items():
        if source in label:
            return normalized
    raise ValueError(f"Unrecognized question type at source line {row['_source_line']}: {label!r}")


def canonical_answer(row: dict[str, object]) -> str:
    assistant = str(row["messages"][1].get("content", ""))
    match = re.search(r"<answer>(.*?)</answer>", assistant, flags=re.I | re.S)
    answer = (match.group(1) if match else assistant).replace("\u200b", "").strip().upper()
    question_type = row["_resolved_question_type"]
    compact = re.sub(r"\s+", "", answer)
    if question_type == "single_choice":
        return compact if compact in {"A", "B", "C", "D"} else ""
    if question_type == "true_false":
        normalized = re.sub(r"[^A-Z]", "", compact)
        return normalized if normalized in {"TRUE", "FALSE"} else ""
    if question_type == "multiple_select":
        if not re.fullmatch(r"[A-D\s,.;/]+", answer):
            return ""
        return ",".join(sorted(set(re.findall(r"[A-D]", answer))))
    return re.sub(r"[.。]+$", "", compact)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            row = json.loads(line)
            metadata = row["metadata"]
            metadata["category"] = normalize_category(metadata.get("category", ""))
            row["_source_line"] = source_line
            row["_video_basename"] = Path(row["videos"][0]).name
            row["_resolved_question_type"] = infer_question_type(row)
            row["_canonical_answer"] = canonical_answer(row)
            rows.append(row)
    return rows


def stratified_sample(
    rows: list[dict[str, object]],
    quotas: dict[tuple[str, str], int],
    rng: random.Random,
) -> list[dict[str, object]]:
    selected = []
    for (question_type, answer), quota in quotas.items():
        candidates = [
            row for row in rows
            if row["_resolved_question_type"] == question_type
            and row["_canonical_answer"] == answer
        ]
        candidates.sort(key=lambda row: int(row["_source_line"]))
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"Insufficient {(question_type, answer)}: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: int(row["_source_line"]))
    return selected


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "unique_videos": len({row["_video_basename"] for row in rows}),
        "question_types": dict(sorted(Counter(row["_resolved_question_type"] for row in rows).items())),
        "question_type_answer_strata": {
            f"{question_type}|{answer}": count
            for (question_type, answer), count in sorted(
                Counter((row["_resolved_question_type"], row["_canonical_answer"]) for row in rows).items()
            )
        },
        "categories": dict(sorted(Counter(row["metadata"].get("category") for row in rows).items())),
        "scene_types": dict(sorted(Counter(row["metadata"].get("scene_type") for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_videos = {Path(row["P"]).name for row in benchmark}
    all_rows = load_rows(args.sft)
    eligible = [row for row in all_rows if row["_video_basename"] not in benchmark_videos]

    intent_pool = [
        row for row in eligible
        if row["metadata"]["category"].startswith("Category 3:")
        and row["metadata"]["scene_type"] == "Human-Centric"
    ]
    dynamic_pool = [
        row for row in eligible
        if row["metadata"]["category"].startswith("Category 2:")
        and row["metadata"]["scene_type"] == "Object-Centric"
    ]
    arm_source_lines = {
        int(row["_source_line"]) for row in intent_pool + dynamic_pool
    }
    random_pool = [
        row for row in eligible
        if int(row["_source_line"]) not in arm_source_lines
    ]

    stratum_counts = [
        Counter((row["_resolved_question_type"], row["_canonical_answer"]) for row in pool if row["_canonical_answer"])
        for pool in (intent_pool, dynamic_pool, random_pool)
    ]
    common_strata = set.intersection(*(set(counts) for counts in stratum_counts))
    quotas = {
        stratum: min(counts[stratum] for counts in stratum_counts)
        for stratum in sorted(common_strata)
    }
    matched_rows = sum(quotas.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subsets = {
        f"intent_matched{matched_rows}": stratified_sample(intent_pool, quotas, random.Random(SEED + 1)),
        f"dynamic_matched{matched_rows}": stratified_sample(dynamic_pool, quotas, random.Random(SEED + 2)),
        f"random_matched{matched_rows}": stratified_sample(random_pool, quotas, random.Random(SEED + 3)),
        "full_no_eval_overlap": eligible,
    }
    files = {}
    for name, rows in subsets.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {
            "path": str(path),
            "sha256": sha256(path),
            **summarize(rows),
        }

    source_lines = {
        name: {int(row["_source_line"]) for row in rows}
        for name, rows in subsets.items()
    }
    report = {
        "schema": "escher-sft-frozen-subsets-v1",
        "seed": SEED,
        "source_sft_sha256": sha256(args.sft),
        "benchmark_sha256": sha256(args.benchmark),
        "benchmark_video_exclusion": True,
        "matching_dimensions": ["resolved_question_type", "canonical_answer"],
        "matched_question_type_answer_quotas": {
            f"{question_type}|{answer}": quota
            for (question_type, answer), quota in quotas.items()
        },
        "matched_rows_per_condition": matched_rows,
        "random_pool_excludes_intent_and_dynamic_arms": True,
        "pairwise_row_overlap": {
            "intent_dynamic": len(source_lines[f"intent_matched{matched_rows}"] & source_lines[f"dynamic_matched{matched_rows}"]),
            "intent_random": len(source_lines[f"intent_matched{matched_rows}"] & source_lines[f"random_matched{matched_rows}"]),
            "dynamic_random": len(source_lines[f"dynamic_matched{matched_rows}"] & source_lines[f"random_matched{matched_rows}"]),
        },
        "files": files,
    }
    meta = args.output_dir / "frozen_subsets.meta.json"
    meta.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
