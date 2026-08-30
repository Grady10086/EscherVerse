#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import PROBE_TYPES, exact_mcnemar_p_value, wilson_interval


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_probe_results(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload["results"]
        if isinstance(payload, list):
            return payload
        raise ValueError("Probe JSON must be a result list or evaluate.py output object")

    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def proportion_record(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95_ci": [low, high] if total else [None, None],
    }


def load_original_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        model = str(payload.get("metadata", {}).get("model", "unknown"))
        return model, payload["results"]
    if isinstance(payload, list):
        return "unknown", payload
    raise ValueError("Original results must be a result list or evaluate.py output object")


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "correct"}:
        return True
    if normalized in {"false", "0", "no", "incorrect"}:
        return False
    return None


def analyze(
    manifest: list[dict[str, str]],
    probes: list[dict[str, str]],
    original_results: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    manifest_by_video = {row["video"]: row for row in manifest}
    sample_ids = {row["sample_id"] for row in manifest}
    probe_definitions = {row["probe_id"]: row for row in probes}

    original_by_sample: dict[str, bool] = {}
    for result in original_results:
        sample_id = str(result.get("sample_id", "")).strip()
        if not sample_id:
            video = str(result.get("video", result.get("P", ""))).strip()
            sample_id = manifest_by_video.get(video, {}).get("sample_id", "")
        correct = as_bool(result.get("is_correct"))
        if sample_id in sample_ids and correct is not None:
            original_by_sample[sample_id] = correct

    probe_by_sample: dict[str, dict[str, bool]] = defaultdict(dict)
    for result in probe_results:
        probe_id = str(result.get("probe_id", result.get("index", ""))).strip()
        definition = probe_definitions.get(probe_id)
        if not definition:
            continue
        correct = as_bool(result.get("is_correct"))
        if correct is None:
            continue
        probe_by_sample[definition["sample_id"]][definition["probe_type"]] = correct

    complete_samples = [
        sample_id
        for sample_id in sorted(sample_ids)
        if sample_id in original_by_sample
        and set(probe_by_sample.get(sample_id, {})) == set(PROBE_TYPES)
    ]
    if not complete_samples:
        raise ValueError("No samples have both original and all three probe results")

    probe_type_metrics = {}
    for probe_type in PROBE_TYPES:
        values = [probe_by_sample[sample_id][probe_type] for sample_id in complete_samples]
        probe_type_metrics[probe_type] = proportion_record(sum(values), len(values))

    probe_subtype_values: dict[str, list[bool]] = defaultdict(list)
    for sample_id in complete_samples:
        for probe_type, correct in probe_by_sample[sample_id].items():
            probe_id = f"{sample_id}-{probe_type}"
            subtype = probe_definitions[probe_id].get("probe_subtype", "").strip()
            if subtype:
                probe_subtype_values[subtype].append(correct)
    probe_subtype_metrics = {
        subtype: proportion_record(sum(values), len(values))
        for subtype, values in sorted(probe_subtype_values.items())
    }

    strict_pass = {
        sample_id: all(probe_by_sample[sample_id].values())
        for sample_id in complete_samples
    }
    relaxed_pass = {
        sample_id: sum(probe_by_sample[sample_id].values()) >= 2
        for sample_id in complete_samples
    }
    original_values = [original_by_sample[sample_id] for sample_id in complete_samples]
    strict_ids = [sample_id for sample_id in complete_samples if strict_pass[sample_id]]
    relaxed_ids = [sample_id for sample_id in complete_samples if relaxed_pass[sample_id]]

    b = sum(strict_pass[sample_id] and not original_by_sample[sample_id] for sample_id in complete_samples)
    c = sum(not strict_pass[sample_id] and original_by_sample[sample_id] for sample_id in complete_samples)

    by_category = {}
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    category_ids = sorted({row["category_id"] for row in manifest})
    for category_id in category_ids:
        ids = [
            sample_id
            for sample_id in complete_samples
            if manifest_by_id[sample_id]["category_id"] == category_id
        ]
        category_strict_ids = [sample_id for sample_id in ids if strict_pass[sample_id]]
        by_category[category_id] = {
            "n": len(ids),
            "perception_strict_pass": proportion_record(
                sum(strict_pass[sample_id] for sample_id in ids), len(ids)
            ),
            "original_tsi_accuracy": proportion_record(
                sum(original_by_sample[sample_id] for sample_id in ids), len(ids)
            ),
            "conditional_tsi_accuracy": proportion_record(
                sum(original_by_sample[sample_id] for sample_id in category_strict_ids),
                len(category_strict_ids),
            ),
        }

    return {
        "model": model,
        "complete_samples": len(complete_samples),
        "missing_original_samples": len(sample_ids - set(original_by_sample)),
        "missing_or_incomplete_probe_samples": len(
            sample_ids
            - {
                sample_id
                for sample_id, values in probe_by_sample.items()
                if set(values) == set(PROBE_TYPES)
            }
        ),
        "per_probe_accuracy": probe_type_metrics,
        "per_probe_subtype_accuracy": probe_subtype_metrics,
        "perception_strict_pass": proportion_record(
            sum(strict_pass.values()), len(complete_samples)
        ),
        "perception_relaxed_pass": proportion_record(
            sum(relaxed_pass.values()), len(complete_samples)
        ),
        "original_tsi_accuracy": proportion_record(
            sum(original_values), len(original_values)
        ),
        "conditional_tsi_accuracy_strict": proportion_record(
            sum(original_by_sample[sample_id] for sample_id in strict_ids),
            len(strict_ids),
        ),
        "conditional_tsi_accuracy_relaxed": proportion_record(
            sum(original_by_sample[sample_id] for sample_id in relaxed_ids),
            len(relaxed_ids),
        ),
        "mcnemar_strict_vs_original": {
            "perception_correct_original_wrong": b,
            "perception_wrong_original_correct": c,
            "exact_two_sided_p": exact_mcnemar_p_value(b, c),
        },
        "by_category": by_category,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze perception-control results.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, original_results = load_original_results(args.original_results)
    report = analyze(
        read_csv(args.manifest),
        read_csv(args.probes),
        original_results,
        read_probe_results(args.probe_results),
        model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
