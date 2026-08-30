#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (
    PROBE_TYPES,
    canonical_category,
    canonical_question_type,
    canonical_scene_type,
    stable_digest,
)


MANIFEST_FIELDS = (
    "sample_id",
    "benchmark_index",
    "video",
    "question",
    "ground_truth",
    "category_id",
    "category",
    "raw_category",
    "scene_type",
    "question_type",
    "sample_sha256",
)

PROBE_FIELDS = (
    "sample_id",
    "benchmark_index",
    "video",
    "probe_id",
    "probe_type",
    "probe_subtype",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "validation_status",
    "annotator_id",
    "reviewer_id",
    "notes",
)


def normalize_multiline_text(value: object) -> str:
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def load_candidates(path: Path) -> dict[int, list[dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_category: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            category_id, category = canonical_category(row.get("C", ""))
        except ValueError:
            continue
        video = str(row.get("P", "")).strip()
        question = normalize_multiline_text(row.get("Q", ""))
        answer = normalize_multiline_text(row.get("A", ""))
        if not video or not question or not answer:
            continue
        candidate = {
            "benchmark_index": row.get("index"),
            "video": video,
            "question": question,
            "ground_truth": answer,
            "category_id": category_id,
            "category": category,
            "raw_category": str(row.get("C", "")).strip(),
            "scene_type": canonical_scene_type(row.get("scene_type", "")),
            "question_type": canonical_question_type(row.get("question_type", "")),
        }
        by_category[category_id].append(candidate)
    return by_category


def sample_candidates(
    by_category: dict[int, list[dict]], per_category: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    selected: list[dict] = []
    used_videos: set[str] = set()

    for category_id in range(1, 7):
        candidates = list(by_category.get(category_id, []))
        rng.shuffle(candidates)
        category_rows = []
        for row in candidates:
            if row["video"] in used_videos:
                continue
            category_rows.append(row)
            used_videos.add(row["video"])
            if len(category_rows) == per_category:
                break
        if len(category_rows) != per_category:
            raise RuntimeError(
                f"Category {category_id} has only {len(category_rows)} unique-video "
                f"candidates; requested {per_category}"
            )
        selected.extend(category_rows)

    selected.sort(key=lambda row: (row["category_id"], int(row["benchmark_index"])))
    for offset, row in enumerate(selected, start=1):
        row["sample_id"] = f"pc-{offset:04d}"
        row["sample_sha256"] = stable_digest(
            (
                row["benchmark_index"],
                row["video"],
                row["question"],
                row["ground_truth"],
                row["category_id"],
                row["scene_type"],
                row["question_type"],
            )
        )
    return selected


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_probe_template(rows: list[dict]) -> list[dict]:
    probes = []
    for row in rows:
        for probe_type in PROBE_TYPES:
            probes.append(
                {
                    "sample_id": row["sample_id"],
                    "benchmark_index": row["benchmark_index"],
                    "video": row["video"],
                    "probe_id": f"{row['sample_id']}-{probe_type}",
                    "probe_type": probe_type,
                    "probe_subtype": "",
                    "question": "",
                    "option_a": "",
                    "option_b": "",
                    "option_c": "",
                    "option_d": "",
                    "answer": "",
                    "validation_status": "draft",
                    "annotator_id": "",
                    "reviewer_id": "",
                    "notes": "",
                }
            )
    return probes


def build_benchmark_subset(rows: list[dict]) -> list[dict]:
    return [
        {
            "index": row["benchmark_index"],
            "sample_id": row["sample_id"],
            "P": row["video"],
            "Q": row["question"],
            "A": row["ground_truth"],
            "C": row["raw_category"],
            "scene_type": row["scene_type"],
            "question_type": row["question_type"],
        }
        for row in rows
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the perception-control sample.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-template", type=Path, required=True)
    parser.add_argument("--benchmark-subset", type=Path)
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_category = load_candidates(args.benchmark)
    selected = sample_candidates(by_category, args.per_category, args.seed)
    write_csv(args.manifest, selected, MANIFEST_FIELDS)
    write_csv(args.probe_template, build_probe_template(selected), PROBE_FIELDS)
    if args.benchmark_subset:
        args.benchmark_subset.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_subset.write_text(
            json.dumps(build_benchmark_subset(selected), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    digest = stable_digest(row["sample_sha256"] for row in selected)
    metadata_path = args.manifest.with_suffix(".meta.txt")
    metadata_path.write_text(
        "\n".join(
            (
                f"benchmark={args.benchmark}",
                f"benchmark_sha256={file_sha256(args.benchmark)}",
                f"seed={args.seed}",
                f"per_category={args.per_category}",
                f"sample_count={len(selected)}",
                f"unique_videos={len({row['video'] for row in selected})}",
                f"manifest_sha256={digest}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} samples to {args.manifest}")
    print(f"Wrote {len(selected) * len(PROBE_TYPES)} probe rows to {args.probe_template}")
    if args.benchmark_subset:
        print(f"Wrote evaluation subset to {args.benchmark_subset}")
    print(f"Manifest digest: {digest}")


if __name__ == "__main__":
    main()
