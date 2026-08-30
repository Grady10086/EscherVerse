#!/usr/bin/env python3
"""Reconstruct EscherVerse clips from provenance metadata.

The script can clip retained source videos supplied with ``--source-dir`` or,
when explicitly enabled, retrieve currently available public sources with
yt-dlp. It never bypasses authentication, access controls, or platform rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


SOURCE_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/video_list.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source-cache", type=Path, default=Path("data/source_videos"))
    parser.add_argument("--report", type=Path, default=Path("video_availability.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required executable is not on PATH: {name}")


def locate_source(source_dir: Path | None, video_id: str) -> Path | None:
    if source_dir is None:
        return None
    for suffix in SOURCE_SUFFIXES:
        candidate = source_dir / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(source_dir.glob(f"{video_id}.*"))
    return next((path for path in matches if path.is_file()), None)


def download_source(row: dict[str, str], cache: Path) -> tuple[Path | None, str]:
    require_tool("yt-dlp")
    cache.mkdir(parents=True, exist_ok=True)
    template = str(cache / "%(id)s.%(ext)s")
    completed = run(
        [
            "yt-dlp",
            "--no-playlist",
            "--restrict-filenames",
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "-o",
            template,
            row["youtube_url"],
        ]
    )
    source = locate_source(cache, row["youtube_video_id"])
    detail = (completed.stderr or completed.stdout).strip()[-1000:]
    return source, detail


def probe_duration(path: Path) -> tuple[float | None, str]:
    completed = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(completed.stdout.strip()), ""
    except ValueError:
        return None, (completed.stderr or "ffprobe returned no duration").strip()


def clip(source: Path, target: Path, start: str, end: str) -> tuple[bool, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        return False, (completed.stderr or "ffmpeg produced no clip").strip()[-1000:]
    duration, detail = probe_duration(target)
    if duration is None or duration <= 0:
        return False, detail
    return True, f"duration_seconds={duration:.3f}"


def main() -> None:
    args = parse_args()
    require_tool("ffmpeg")
    require_tool("ffprobe")
    rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("video_list.json must contain a JSON list")
    if args.limit is not None:
        rows = rows[: args.limit]

    report_rows: list[dict[str, str]] = []
    for row in rows:
        target = args.output_dir / row["video_file"]
        if target.is_file() and target.stat().st_size > 0 and not args.overwrite:
            duration, detail = probe_duration(target)
            status = "existing_valid" if duration and duration > 0 else "existing_invalid"
            report_rows.append({"video_file": row["video_file"], "status": status, "detail": detail or str(duration)})
            continue

        source = locate_source(args.source_dir, row["youtube_video_id"])
        retrieval_detail = ""
        if source is None and args.download:
            source, retrieval_detail = download_source(row, args.source_cache)
        if source is None:
            status = "source_unavailable" if args.download else "source_not_supplied"
            report_rows.append({"video_file": row["video_file"], "status": status, "detail": retrieval_detail})
            continue

        ok, detail = clip(source, target, row["clip_start_precise"], row["clip_end_precise"])
        report_rows.append(
            {
                "video_file": row["video_file"],
                "status": "reconstructed" if ok else "clip_failed",
                "detail": detail,
            }
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_file", "status", "detail"])
        writer.writeheader()
        writer.writerows(report_rows)
    counts: dict[str, int] = {}
    for row in report_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"total": len(report_rows), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
