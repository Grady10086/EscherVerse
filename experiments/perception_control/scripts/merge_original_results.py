#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{path} is not a result list or evaluate.py output object")


def normalized_prediction(row: dict[str, Any]) -> str:
    value = row.get("prediction_clean", row.get("model_prediction", ""))
    return " ".join(str(value).strip().split())


def merge_sources(
    sources: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    source_by_sample: dict[str, str] = {}
    consistent_overlaps = 0

    for label, rows in sources:
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{label} contains a row without sample_id")
            existing = merged.get(sample_id)
            if existing is None:
                merged[sample_id] = dict(row)
                source_by_sample[sample_id] = label
                continue
            if (
                bool(existing.get("is_correct")) != bool(row.get("is_correct"))
                or normalized_prediction(existing) != normalized_prediction(row)
            ):
                raise ValueError(
                    f"Conflicting historical results for {sample_id}: "
                    f"{source_by_sample[sample_id]} vs {label}"
                )
            consistent_overlaps += 1

    ordered = [merged[sample_id] for sample_id in sorted(merged)]
    metadata = {
        "sources": [label for label, _ in sources],
        "source_rows": {label: len(rows) for label, rows in sources},
        "merged_unique_samples": len(ordered),
        "consistent_overlap_rows": consistent_overlaps,
        "conflicts": 0,
    }
    return ordered, metadata


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Input source must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Input source must use LABEL=PATH")
    return label.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge strictly matched original-task result subsets."
    )
    parser.add_argument(
        "--input",
        type=parse_source,
        action="append",
        required=True,
        help="Matched result source in LABEL=PATH form; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [(label, read_results(path)) for label, path in args.input]
    rows, metadata = merge_sources(sources)
    payload = {"metadata": metadata, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
