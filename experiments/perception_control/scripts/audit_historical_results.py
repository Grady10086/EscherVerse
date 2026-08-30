#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def read_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload, payload["results"]
    if isinstance(payload, list):
        return {}, payload
    raise ValueError(f"{path} is not a result list or evaluate.py output object")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Result source must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Result source must use LABEL=PATH")
    return label.strip(), Path(path)


def audit_source(
    manifest: list[dict[str, str]],
    source_path: Path,
    matched_output: Path | None = None,
) -> dict[str, Any]:
    payload, results = read_results(source_path)
    manifest_by_index = {str(row["benchmark_index"]): row for row in manifest}
    matched_by_sample: dict[str, dict[str, Any]] = {}
    mismatch_counts: Counter[str] = Counter()

    for result in results:
        index = str(result.get("index", result.get("benchmark_index", ""))).strip()
        expected = manifest_by_index.get(index)
        if expected is None:
            mismatch_counts["outside_frozen_manifest"] += 1
            continue
        video = str(result.get("video", result.get("P", ""))).strip()
        if video != expected["video"]:
            mismatch_counts["video_mismatch"] += 1
            continue
        question = normalize_text(result.get("question", result.get("Q", "")))
        if question != normalize_text(expected["question"]):
            mismatch_counts["question_mismatch"] += 1
            continue
        sample_id = expected["sample_id"]
        if sample_id in matched_by_sample:
            mismatch_counts["duplicate_strict_match"] += 1
            continue
        matched = dict(result)
        matched["sample_id"] = sample_id
        matched_by_sample[sample_id] = matched

    matched_rows = [
        matched_by_sample[row["sample_id"]]
        for row in manifest
        if row["sample_id"] in matched_by_sample
    ]
    if matched_output is not None:
        matched_output.parent.mkdir(parents=True, exist_ok=True)
        matched_payload = {
            "metadata": {
                "source": str(source_path),
                "strict_match_fields": ["benchmark_index", "video", "question"],
            },
            "results": matched_rows,
        }
        matched_output.write_text(
            json.dumps(matched_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    matched_ids = set(matched_by_sample)
    category_counts = Counter(
        row["category_id"] for row in manifest if row["sample_id"] in matched_ids
    )
    correct_values = [
        bool(row["is_correct"]) for row in matched_rows if "is_correct" in row
    ]
    return {
        "source": str(source_path),
        "reported_total": payload.get("total"),
        "reported_accuracy": payload.get("accuracy"),
        "result_rows": len(results),
        "matched_unique_samples": len(matched_rows),
        "coverage_rate": len(matched_rows) / len(manifest) if manifest else None,
        "matched_by_category": dict(sorted(category_counts.items())),
        "matched_correct": sum(correct_values),
        "matched_accuracy": (
            sum(correct_values) / len(correct_values) if correct_values else None
        ),
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly audit historical result coverage for the frozen sample."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result",
        type=parse_source,
        action="append",
        required=True,
        help="Historical result source in LABEL=PATH form; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    sources = {}
    for label, path in args.result:
        matched_output = (
            args.matched_dir / f"{label}.json" if args.matched_dir is not None else None
        )
        sources[label] = audit_source(manifest, path, matched_output)
    report = {
        "manifest_samples": len(manifest),
        "strict_match_fields": ["benchmark_index", "video", "question"],
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
