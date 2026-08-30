#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def contact_sheet_filter(
    duration: float,
    frame_count: int = 20,
    columns: int = 5,
) -> str:
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    if frame_count <= 0 or columns <= 0:
        raise ValueError("frame_count and columns must be positive")
    rows = math.ceil(frame_count / columns)
    fps = frame_count / duration
    return (
        f"fps={fps:.12g},scale=320:-2,"
        "drawtext=text='%{pts\\:hms}':x=8:y=8:fontsize=18:"
        "fontcolor=white:box=1:boxcolor=black@0.65,"
        f"tile={columns}x{rows}:padding=4:margin=4:color=black"
    )


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def render_contact_sheet(
    row: dict[str, str],
    video_dir: Path,
    output_dir: Path,
    overwrite: bool,
    frame_count: int = 20,
    columns: int = 5,
) -> tuple[str, str]:
    source = video_dir / row["video"]
    destination = output_dir / f"{row['sample_id']}.jpg"
    if not source.is_file():
        return row["sample_id"], "missing_video"
    if destination.is_file() and not overwrite:
        return row["sample_id"], "existing"

    duration = video_duration(source)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            contact_sheet_filter(duration, frame_count=frame_count, columns=columns),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )
    return row["sample_id"], "rendered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render uniformly sampled contact sheets for available videos."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-count", type=int, default=20)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_count <= 0 or args.columns <= 0:
        raise ValueError("frame-count and columns must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)

    def render(row: dict[str, str]) -> tuple[str, str]:
        return render_contact_sheet(
            row,
            args.video_dir,
            args.output_dir,
            args.overwrite,
            args.frame_count,
            args.columns,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(render, rows))
    counts = Counter(status for _, status in results)
    print(json.dumps({"total": len(rows), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
