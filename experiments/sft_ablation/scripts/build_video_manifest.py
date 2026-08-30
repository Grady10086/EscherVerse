#!/usr/bin/env python3
"""Build the unique video manifest needed by SFT-ablation training and evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-sft", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()

    sft_videos = set()
    with args.full_sft.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sft_videos.add(Path(row["videos"][0]).name)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_videos = {Path(row["P"]).name for row in benchmark}
    videos = sorted(sft_videos | benchmark_videos)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video"])
        writer.writeheader()
        for index, video in enumerate(videos, start=1):
            writer.writerow({"sample_id": f"video-{index:05d}", "video": video})
    meta = {
        "schema": "escher-sft-video-manifest-v1",
        "sft_unique_videos": len(sft_videos),
        "benchmark_unique_videos": len(benchmark_videos),
        "overlap": len(sft_videos & benchmark_videos),
        "union_unique_videos": len(videos),
        "manifest_sha256": sha256(args.output),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
