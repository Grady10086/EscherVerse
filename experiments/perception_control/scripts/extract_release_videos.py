#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path


REPORT_FIELDS = (
    "sample_id",
    "video",
    "status",
    "bytes",
    "sha256",
    "archive_member",
    "message",
)


class ConcatenatedReader(io.RawIOBase):
    def __init__(self, paths: list[Path], first_offset: int = 0):
        super().__init__()
        self.paths = paths
        self.first_offset = first_offset
        self.index = -1
        self.handle = None

    def readable(self) -> bool:
        return True

    def _advance(self) -> bool:
        if self.handle is not None:
            self.handle.close()
        self.index += 1
        if self.index >= len(self.paths):
            self.handle = None
            return False
        self.handle = self.paths[self.index].open("rb")
        if self.index == 0 and self.first_offset:
            self.handle.seek(self.first_offset)
        return True

    def readinto(self, buffer: bytearray) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            if self.handle is None and not self._advance():
                break
            count = self.handle.readinto(view[total:])
            if count:
                total += count
                continue
            self._advance()
        return total

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        super().close()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_part_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith("part_index\t")
        )
    except StopIteration as exc:
        raise ValueError(f"No part table found in {path}") from exc
    return list(csv.DictReader(lines[header_index:], delimiter="\t"))


def verify_parts(
    parts_dir: Path,
    start_part_index: int = 0,
    end_part_index: int | None = None,
) -> list[Path]:
    manifest_rows = read_part_manifest(parts_dir / "MANIFEST.tsv")
    manifest_rows = [
        row
        for row in manifest_rows
        if int(row["part_index"]) >= start_part_index
        and (
            end_part_index is None
            or int(row["part_index"]) <= end_part_index
        )
    ]
    if not manifest_rows:
        raise ValueError(
            f"No archive parts selected for range {start_part_index}..{end_part_index}"
        )
    expected_names = [row["part_name"] for row in manifest_rows]
    parts = [parts_dir / name for name in expected_names]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} archive parts: {missing[:3]}")

    for index, (path, row) in enumerate(zip(parts, manifest_rows), start=1):
        expected_size = int(row["bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path.name}: expected {expected_size} bytes, found {actual_size}"
            )
        actual_hash = file_sha256(path)
        if actual_hash != row["sha256"]:
            raise ValueError(
                f"{path.name}: expected sha256 {row['sha256']}, found {actual_hash}"
            )
        print(f"Verified part {index}/{len(parts)}: {path.name}")
    return parts


def list_parts_without_verification(
    parts_dir: Path,
    start_part_index: int = 0,
    end_part_index: int | None = None,
) -> list[Path]:
    parts = sorted(
        path
        for path in parts_dir.glob("*.part-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
        and int(path.name.rsplit("-", 1)[-1]) >= start_part_index
        and (
            end_part_index is None
            or int(path.name.rsplit("-", 1)[-1]) <= end_part_index
        )
    )
    if not parts:
        raise FileNotFoundError(f"No split archive parts found in {parts_dir}")
    return parts


def find_next_video_header(path: Path) -> int:
    path_metadata_types = {
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
    }
    offset = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(tarfile.BLOCKSIZE)
            if len(block) < tarfile.BLOCKSIZE:
                break
            try:
                member = tarfile.TarInfo.frombuf(
                    block,
                    encoding="utf-8",
                    errors="surrogateescape",
                )
            except (tarfile.HeaderError, ValueError):
                member = None
            if (
                member is not None
                and member.size > 0
                and (
                    member.type in path_metadata_types
                    or (
                        member.isfile()
                        and member.name.casefold().endswith(".mp4")
                    )
                )
            ):
                print(
                    f"Resynchronized at byte {offset} in {path.name}: "
                    f"{member.name}"
                )
                return offset
            offset += tarfile.BLOCKSIZE
    raise ValueError(f"No valid video header found in {path}")


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> tuple[int, str]:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Cannot read archive member {member.name}")
    temporary = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with source, temporary.open("wb") as output:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                size += len(block)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return size, digest.hexdigest()


def extract_videos(
    manifest: list[dict[str, str]],
    parts: list[Path],
    video_dir: Path,
    report_path: Path,
    allow_incomplete_archive: bool = False,
    start_at_next_video_header: bool = False,
) -> list[dict[str, object]]:
    video_dir.mkdir(parents=True, exist_ok=True)
    rows_by_video: dict[str, dict[str, object]] = {}
    wanted: set[str] = set()
    for row in manifest:
        video = row["video"]
        report = {
            "sample_id": row["sample_id"],
            "video": video,
            "status": "pending",
            "bytes": 0,
            "sha256": "",
            "archive_member": "",
            "message": "",
        }
        destination = video_dir / video
        if destination.is_file() and destination.stat().st_size > 0:
            report.update(
                {
                    "status": "existing",
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(destination),
                }
            )
        else:
            wanted.add(video)
        rows_by_video[video] = report

    archive_error = ""
    if wanted:
        first_offset = find_next_video_header(parts[0]) if start_at_next_video_header else 0
        reader = ConcatenatedReader(parts, first_offset=first_offset)
        buffered = io.BufferedReader(reader, buffer_size=8 * 1024 * 1024)
        try:
            with buffered, tarfile.open(fileobj=buffered, mode="r|") as archive:
                for member in archive:
                    video = Path(member.name).name
                    if not member.isfile() or video not in wanted:
                        continue
                    destination = video_dir / video
                    try:
                        size, digest = copy_member(archive, member, destination)
                    except (OSError, tarfile.TarError, ValueError) as exc:
                        rows_by_video[video].update(
                            {
                                "status": "failed",
                                "archive_member": member.name,
                                "message": str(exc),
                            }
                        )
                    else:
                        rows_by_video[video].update(
                            {
                                "status": "extracted",
                                "bytes": size,
                                "sha256": digest,
                                "archive_member": member.name,
                            }
                        )
                        wanted.remove(video)
                        completed = len(manifest) - len(wanted)
                        if completed % 25 == 0 or not wanted:
                            print(f"Resolved {completed}/{len(manifest)} videos")
                    write_report(
                        report_path,
                        [rows_by_video[row["video"]] for row in manifest],
                    )
                    if not wanted:
                        break
        except (OSError, tarfile.TarError) as exc:
            if not allow_incomplete_archive:
                raise
            archive_kind = "segment" if start_at_next_video_header else "prefix"
            archive_error = (
                f"Verified archive {archive_kind} ended before the full tar: {exc}"
            )

    for video in wanted:
        rows_by_video[video].update(
            {
                "status": "missing",
                "message": archive_error
                or "Video basename was not found in the release archive",
            }
        )
    report_rows = [rows_by_video[row["video"]] for row in manifest]
    write_report(report_path, report_rows)
    return report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the frozen sample from the split Hugging Face release tar."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-part-index", type=int, default=0)
    parser.add_argument("--end-part-index", type=int)
    parser.add_argument("--allow-incomplete-archive", action="store_true")
    parser.add_argument("--skip-part-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    parts = (
        list_parts_without_verification(
            args.parts_dir,
            args.start_part_index,
            args.end_part_index,
        )
        if args.skip_part_verification
        else verify_parts(
            args.parts_dir,
            args.start_part_index,
            args.end_part_index,
        )
    )
    rows = extract_videos(
        manifest,
        parts,
        args.video_dir,
        args.report,
        allow_incomplete_archive=args.allow_incomplete_archive,
        start_at_next_video_header=args.start_part_index > 0,
    )
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"total": len(rows), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
