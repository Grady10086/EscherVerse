#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shlex
import subprocess
from pathlib import Path
from urllib.parse import quote

from extract_release_videos import read_part_manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_url(repo: str, revision: str, remote_path: str) -> str:
    encoded_path = quote(remote_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"{revision}/{encoded_path}?download=true"
    )


def build_command(url: str, partial: Path, proxy: str | None) -> list[str]:
    command = [
        "curl",
        "-q",
        "-L",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--continue-at",
        "-",
        "--output",
        str(partial),
    ]
    if proxy:
        command.extend(["--proxy", proxy])
    command.append(url)
    return command


def download_part(
    row: dict[str, str],
    parts_dir: Path,
    repo: str,
    revision: str,
    remote_dir: str,
    proxy: str | None,
) -> tuple[str, str]:
    name = row["part_name"]
    expected_size = int(row["bytes"])
    expected_hash = row["sha256"]
    destination = parts_dir / name
    partial = destination.with_name(destination.name + ".partial")

    if destination.is_file():
        if (
            destination.stat().st_size == expected_size
            and file_sha256(destination) == expected_hash
        ):
            return name, "existing_verified"
        raise ValueError(f"{destination} exists but does not match the manifest")

    remote_path = f"{remote_dir.rstrip('/')}/{name}"
    command = build_command(build_url(repo, revision, remote_path), partial, proxy)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"{name}: curl exited {completed.returncode}: {message[-500:]}"
        )
    actual_size = partial.stat().st_size if partial.exists() else 0
    if actual_size != expected_size:
        raise ValueError(
            f"{name}: expected {expected_size} bytes, downloaded {actual_size}"
        )
    actual_hash = file_sha256(partial)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{name}: expected sha256 {expected_hash}, downloaded {actual_hash}"
        )
    partial.replace(destination)
    return name, "downloaded_verified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify split release parts over resumable HTTP."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument(
        "--repo",
        default="Gradygu3u/spatial-escherverse-release",
    )
    parser.add_argument(
        "--revision",
        default="f1db716d2697c1f926e51e41a18d7fadd438988a",
    )
    parser.add_argument(
        "--remote-dir",
        default="packages/escher_actual_videos.tar.parts",
    )
    parser.add_argument("--proxy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_part_manifest(args.manifest)
    rows = [
        row
        for row in rows
        if int(row["part_index"]) >= args.start_index
        and (
            args.end_index is None
            or int(row["part_index"]) <= args.end_index
        )
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_part,
                row,
                args.parts_dir,
                args.repo,
                args.revision,
                args.remote_dir,
                args.proxy,
            ): row["part_name"]
            for row in rows
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"FAILED {name}: {exc}")
            else:
                completed_count += 1
                print(f"{status} {completed_count}/{len(rows)}: {name}")
    if failures:
        commands = [
            shlex.join(
                build_command(
                    build_url(
                        args.repo,
                        args.revision,
                        f"{args.remote_dir.rstrip('/')}/{name}",
                    ),
                    args.parts_dir / f"{name}.partial",
                    args.proxy,
                )
            )
            for name, _ in failures[:3]
        ]
        raise SystemExit(
            f"{len(failures)} parts failed. Example resume commands: {commands}"
        )


if __name__ == "__main__":
    main()
