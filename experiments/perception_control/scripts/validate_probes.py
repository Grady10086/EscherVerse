#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from common import OPTION_LETTERS, PROBE_SUBTYPES, PROBE_SUBTYPES_BY_TYPE, PROBE_TYPES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(
    manifest_rows: list[dict[str, str]],
    probe_rows: list[dict[str, str]],
    require_approved: bool,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids = {row["sample_id"] for row in manifest_rows}
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    probes_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_probe_ids: set[str] = set()

    for line_number, row in enumerate(probe_rows, start=2):
        sample_id = row.get("sample_id", "").strip()
        probe_id = row.get("probe_id", "").strip()
        probe_type = row.get("probe_type", "").strip()
        probe_subtype = row.get("probe_subtype", "").strip()

        if sample_id not in sample_ids:
            errors.append(f"line {line_number}: unknown sample_id {sample_id!r}")
            continue
        if probe_id in seen_probe_ids:
            errors.append(f"line {line_number}: duplicate probe_id {probe_id!r}")
        seen_probe_ids.add(probe_id)
        probes_by_sample[sample_id].append(row)

        if probe_type not in PROBE_TYPES:
            errors.append(f"line {line_number}: invalid probe_type {probe_type!r}")
        if probe_subtype not in PROBE_SUBTYPES:
            errors.append(f"line {line_number}: invalid probe_subtype {probe_subtype!r}")
        elif (
            probe_type in PROBE_SUBTYPES_BY_TYPE
            and probe_subtype not in PROBE_SUBTYPES_BY_TYPE[probe_type]
        ):
            errors.append(
                f"line {line_number}: probe_subtype {probe_subtype!r} "
                f"is incompatible with probe_type {probe_type!r}"
            )
        if row.get("video", "").strip() != manifest_by_id[sample_id]["video"].strip():
            errors.append(f"line {line_number}: video does not match manifest")
        if not row.get("question", "").strip():
            errors.append(f"line {line_number}: empty question")

        options = [row.get(f"option_{letter.lower()}", "").strip() for letter in OPTION_LETTERS]
        if any(not option for option in options):
            errors.append(f"line {line_number}: all four options are required")
        if len(set(option.casefold() for option in options if option)) != len(options):
            errors.append(f"line {line_number}: options must be distinct")

        answer = row.get("answer", "").strip().upper()
        if answer not in OPTION_LETTERS:
            errors.append(f"line {line_number}: answer must be one of A-D")
        status = row.get("validation_status", "").strip().lower()
        if require_approved and status != "approved":
            errors.append(f"line {line_number}: validation_status must be approved")
        elif status not in {"draft", "needs_revision", "approved", "rejected"}:
            warnings.append(f"line {line_number}: unusual validation_status {status!r}")

        lower_question = row.get("question", "").casefold()
        if probe_type == "entity" and any(term in lower_question for term in ("intend", "goal", "objective")):
            warnings.append(f"line {line_number}: entity probe may contain intent language")
        if probe_type == "simple_relation" and any(
            term in lower_question for term in ("would", "most likely", "if ", "counterfactual")
        ):
            warnings.append(f"line {line_number}: simple_relation probe may require prediction")

    for sample_id in sorted(sample_ids):
        rows = probes_by_sample.get(sample_id, [])
        counts = Counter(row.get("probe_type", "").strip() for row in rows)
        if set(counts) != set(PROBE_TYPES) or any(counts[kind] != 1 for kind in PROBE_TYPES):
            errors.append(
                f"{sample_id}: expected one probe of each type; found {dict(counts)}"
            )

    answer_counts = Counter(
        row.get("answer", "").strip().upper()
        for row in probe_rows
        if row.get("answer", "").strip().upper() in OPTION_LETTERS
    )
    answer_counts_all = {
        letter: answer_counts.get(letter, 0) for letter in OPTION_LETTERS
    }
    answer_position_spread = max(answer_counts_all.values()) - min(
        answer_counts_all.values()
    )
    balance_tolerance = max(1, math.ceil(len(probe_rows) * 0.02))
    if len(probe_rows) >= len(OPTION_LETTERS) and answer_position_spread > balance_tolerance:
        warnings.append(
            "answer positions are imbalanced: "
            f"{answer_counts_all} (spread {answer_position_spread}, "
            f"tolerance {balance_tolerance})"
        )

    return {
        "valid": not errors,
        "manifest_samples": len(manifest_rows),
        "probe_rows": len(probe_rows),
        "approved_rows": sum(
            row.get("validation_status", "").strip().lower() == "approved"
            for row in probe_rows
        ),
        "probe_subtype_counts": dict(
            sorted(
                Counter(row.get("probe_subtype", "").strip() for row in probe_rows).items()
            )
        ),
        "answer_counts": answer_counts_all,
        "answer_position_spread": answer_position_spread,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate annotated perception probes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(
        read_csv(args.manifest),
        read_csv(args.probes),
        require_approved=args.require_approved,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
