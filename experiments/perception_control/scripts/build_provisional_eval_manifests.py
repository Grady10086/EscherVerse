#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_control_question(row: dict[str, str]) -> str:
    return (
        f"[Single-Choice] {row['question']} [Options] "
        f"A) {row['option_a']} B) {row['option_b']} "
        f"C) {row['option_c']} D) {row['option_d']}"
    )


def build_items(
    manifest: list[dict[str, str]], drafts: list[dict[str, str]]
) -> tuple[list[dict], list[dict]]:
    drafts_by_id = {row["probe_sample_id"]: row for row in drafts}
    if len(drafts_by_id) != len(drafts):
        raise ValueError("draft probe_sample_id values must be unique")
    expected_ids = {row["probe_sample_id"] for row in manifest}
    if set(drafts_by_id) != expected_ids:
        raise ValueError("draft IDs do not exactly cover the frozen manifest")

    controls: list[dict] = []
    originals: list[dict] = []
    for index, source in enumerate(manifest, start=1):
        draft = drafts_by_id[source["probe_sample_id"]]
        for key in ("source_sample_id", "benchmark_index", "video"):
            if draft[key].strip() != source[key].strip():
                raise ValueError(f"{source['probe_sample_id']}: {key} mismatch")
        if draft["answer"].strip() != source["target_answer_position"].strip():
            raise ValueError(f"{source['probe_sample_id']}: answer-position mismatch")
        common = {
            "index": index,
            "P": source["video"],
            "scene_type": source["scene_type"],
            "source_sample_id": source["source_sample_id"],
            "probe_sample_id": source["probe_sample_id"],
            "benchmark_index": int(source["benchmark_index"]),
        }
        controls.append(
            {
                **common,
                "Q": render_control_question(draft),
                "A": draft["answer"],
                "C": f"Perception control: {draft['probe_type']}",
                "question_type": "Single-Choice",
                "probe_type": draft["probe_type"],
                "probe_subtype": draft["probe_subtype"],
                "candidate_validation_status": draft["validation_status"],
            }
        )
        originals.append(
            {
                **common,
                "Q": source["original_question"],
                "A": source["original_ground_truth"],
                "C": source["category"],
                "question_type": source["question_type"],
                "category_id": int(source["category_id"]),
            }
        )
    return controls, originals


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_shards(root: Path, items: list[dict], shard_count: int) -> None:
    write_json(root / "items.json", items)
    for shard in range(shard_count):
        write_json(
            root / f"items_shard_{shard}.json",
            [item for position, item in enumerate(items) if position % shard_count == shard],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional perception-control evaluation manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controls, originals = build_items(read_csv(args.manifest), read_csv(args.drafts))
    control_root = args.output_root / "controls"
    original_root = args.output_root / "originals"
    write_shards(control_root, controls, args.shards)
    write_shards(original_root, originals, args.shards)
    report = {
        "status": "provisional_unreviewed_controls",
        "controls": len(controls),
        "originals": len(originals),
        "unique_videos": len({item["P"] for item in controls}),
        "shared_index_video_mapping": all(
            (a["index"], a["P"]) == (b["index"], b["P"])
            for a, b in zip(controls, originals)
        ),
        "control_manifest_sha256": sha256(control_root / "items.json"),
        "original_manifest_sha256": sha256(original_root / "items.json"),
        "shards": args.shards,
    }
    write_json(args.output_root / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
