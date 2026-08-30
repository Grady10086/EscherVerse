#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from common import OPTION_LETTERS, PROBE_TYPES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_markdown(value: str) -> str:
    return " ".join(value.strip().split())


def build_packet(
    manifest_rows: list[dict[str, str]],
    probe_rows: list[dict[str, str]],
    image_dir: Path,
    output_path: Path,
) -> str:
    probes_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in probe_rows:
        probes_by_sample[row["sample_id"]].append(row)

    lines = [
        "# Perception-control probe review",
        "",
        "Review each item against the full video. The contact sheet is only a",
        "navigation aid. Keep every row as `draft` until a second reviewer has",
        "checked the correct answer, distractors, and separation from the original",
        "reasoning task.",
        "",
    ]
    for manifest in manifest_rows:
        sample_id = manifest["sample_id"]
        rows = probes_by_sample.get(sample_id)
        if not rows:
            continue
        rows.sort(key=lambda row: PROBE_TYPES.index(row["probe_type"]))
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"**Video:** `{manifest['video']}`",
                "",
                f"**Original category:** {clean_markdown(manifest['category'])}",
                "",
                f"**Original question:** {clean_markdown(manifest['question'])}",
                "",
                f"**Original answer:** `{manifest['ground_truth']}`",
                "",
            ]
        )

        image_path = image_dir / f"{sample_id}.jpg"
        if image_path.exists():
            relative_image = Path(
                os.path.relpath(image_path, output_path.parent)
            ).as_posix()
            lines.extend(
                [
                    f"![{sample_id} one-frame-per-second contact sheet]({relative_image})",
                    "",
                ]
            )
        else:
            lines.extend(["_Contact sheet not available._", ""])

        for row in rows:
            lines.extend(
                [
                    f"### {row['probe_id']}",
                    "",
                    f"**Type:** `{row['probe_type']}` / `{row['probe_subtype']}`",
                    "",
                    f"**Question:** {clean_markdown(row['question'])}",
                    "",
                ]
            )
            for letter in OPTION_LETTERS:
                option = clean_markdown(row[f"option_{letter.lower()}"])
                lines.append(f"- {letter}. {option}")
            lines.extend(
                [
                    "",
                    f"**Draft answer:** `{row['answer'].strip().upper()}`",
                    "",
                    "- [ ] Correct answer verified in the full video",
                    "- [ ] Distractors are plausible and visually false",
                    "- [ ] Probe does not reproduce the original reasoning step",
                    "",
                ]
            )
        lines.extend(
            [
                "**Sample decision:** [ ] approve all [ ] revise [ ] reject",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Markdown probe review packet.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered = build_packet(
        read_csv(args.manifest),
        read_csv(args.probes),
        args.image_dir,
        args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote probe review packet to {args.output}")


if __name__ == "__main__":
    main()
