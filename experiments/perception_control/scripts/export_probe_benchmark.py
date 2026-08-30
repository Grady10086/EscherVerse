#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import OPTION_LETTERS
from validate_probes import validate


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_question(row: dict[str, str]) -> str:
    options = " ".join(
        f"{letter}) {row[f'option_{letter.lower()}'].strip()}"
        for letter in OPTION_LETTERS
    )
    return f"[Single-Choice] {row['question'].strip()} [Options] {options}"


def export_rows(
    manifest_rows: list[dict[str, str]], probe_rows: list[dict[str, str]]
) -> list[dict]:
    report = validate(manifest_rows, probe_rows, require_approved=True)
    if not report["valid"]:
        preview = "\n".join(report["errors"][:20])
        raise ValueError(f"Probe validation failed:\n{preview}")

    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    exported = []
    for row in probe_rows:
        manifest = manifest_by_id[row["sample_id"]]
        exported.append(
            {
                "index": row["probe_id"],
                "probe_id": row["probe_id"],
                "sample_id": row["sample_id"],
                "P": row["video"],
                "Q": render_question(row),
                "A": row["answer"].strip().upper(),
                "C": f"Perception Control: {row['probe_subtype']}",
                "scene_type": manifest["scene_type"],
                "question_type": "Single-Choice",
            }
        )
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export approved probes for evaluate.py.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_rows(read_csv(args.manifest), read_csv(args.probes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(exported)} approved probes to {args.output}")


if __name__ == "__main__":
    main()
