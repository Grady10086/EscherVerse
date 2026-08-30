#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_and_audit(
    manifest_path: Path,
    draft_paths: list[Path],
    usage_paths: list[Path],
    output_path: Path,
    input_rate: float,
    output_rate: float,
) -> dict:
    manifest = read_csv(manifest_path)
    manifest_by_id = {row["probe_sample_id"]: row for row in manifest}
    errors: list[str] = []
    merged: dict[str, dict[str, str]] = {}

    for path in draft_paths:
        for row in read_csv(path):
            sample_id = row.get("probe_sample_id", "").strip()
            if sample_id not in manifest_by_id:
                errors.append(f"unknown probe_sample_id {sample_id!r} in {path}")
                continue
            if sample_id in merged:
                errors.append(f"duplicate probe_sample_id {sample_id!r}")
                continue
            expected = manifest_by_id[sample_id]
            checks = {
                "source_sample_id": "source_sample_id",
                "benchmark_index": "benchmark_index",
                "video": "video",
                "probe_type": "assigned_probe_type",
                "probe_subtype": "assigned_probe_subtype",
                "answer": "target_answer_position",
            }
            for draft_field, manifest_field in checks.items():
                if row.get(draft_field, "").strip() != expected[manifest_field].strip():
                    errors.append(
                        f"{sample_id}: {draft_field} does not match frozen manifest"
                    )
            if row.get("validation_status", "").strip() != "draft":
                errors.append(f"{sample_id}: generated row must remain draft")
            merged[sample_id] = row

    missing = sorted(set(manifest_by_id) - set(merged))
    if missing:
        errors.append(f"missing {len(missing)} manifest IDs: {missing}")

    ordered = [merged[row["probe_sample_id"]] for row in manifest if row["probe_sample_id"] in merged]
    if ordered:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(ordered)

    statuses: Counter[str] = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    for path in usage_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            status = str(record.get("status", "unknown"))
            statuses[status] += 1
            if status in {"ok", "content_rejected"}:
                prompt_tokens += int(record.get("prompt_tokens") or 0)
                completion_tokens += int(record.get("completion_tokens") or 0)

    estimated_cost = (
        prompt_tokens * input_rate + completion_tokens * output_rate
    ) / 1_000_000
    answer_counts = Counter(row.get("answer", "") for row in ordered)
    type_counts = Counter(row.get("probe_type", "") for row in ordered)
    subtype_counts = Counter(row.get("probe_subtype", "") for row in ordered)
    return {
        "valid": not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path) if output_path.exists() else None,
        "manifest_rows": len(manifest),
        "merged_unique_drafts": len(ordered),
        "validation_status_counts": dict(Counter(row.get("validation_status", "") for row in ordered)),
        "probe_type_counts": dict(sorted(type_counts.items())),
        "probe_subtype_counts": dict(sorted(subtype_counts.items())),
        "answer_counts": {letter: answer_counts.get(letter, 0) for letter in "ABCD"},
        "usage_status_counts": dict(sorted(statuses.items())),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "provisional_input_cny_per_million": input_rate,
        "provisional_output_cny_per_million": output_rate,
        "provisional_estimated_cost_cny": round(estimated_cost, 6),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and audit perception-control single-probe drafts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--draft", type=Path, action="append", required=True)
    parser.add_argument("--usage-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-cny-per-million", type=float, default=20.0)
    parser.add_argument("--output-cny-per-million", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_and_audit(
        args.manifest,
        args.draft,
        args.usage_log,
        args.output,
        args.input_cny_per_million,
        args.output_cny_per_million,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
