#!/usr/bin/env python3
"""Build a blinded six-dimension scope-classification packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


SEED = 20260814


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-a", type=Path, required=True)
    parser.add_argument("--author-b", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()

    eligible = []
    excluded = {"rejected": 0, "still_uncertain": 0}
    for author, path in [("author_a", args.author_a), ("author_b", args.author_b)]:
        for row in read_csv(path):
            if row["excluded_from_coverage"] == "1":
                excluded["rejected"] += 1
            elif row["needs_full_video"] == "1":
                excluded["still_uncertain"] += 1
            else:
                eligible.append((author, row))
    rng = random.Random(SEED)
    rng.shuffle(eligible)

    args.packet.parent.mkdir(parents=True, exist_ok=True)
    with args.packet.open("w", newline="", encoding="utf-8") as ph, args.mapping.open(
        "w", newline="", encoding="utf-8"
    ) as mh:
        pw = csv.DictWriter(ph, fieldnames=["scope_item_id", "question_text_cn"])
        mw = csv.DictWriter(
            mh,
            fieldnames=["scope_item_id", "author_id", "audit_item_id", "proposal_no"],
        )
        pw.writeheader()
        mw.writeheader()
        for index, (author, row) in enumerate(eligible, start=1):
            item_id = f"scope-{index:04d}"
            pw.writerow({"scope_item_id": item_id, "question_text_cn": row["question_text_cn"]})
            mw.writerow(
                {
                    "scope_item_id": item_id,
                    "author_id": author,
                    "audit_item_id": row["audit_item_id"],
                    "proposal_no": row["proposal_no"],
                }
            )
    meta = {
        "schema": "escher-audit-six-dimension-scope-packet-v1",
        "seed": SEED,
        "included": len(eligible),
        "excluded": excluded,
        "blind_packet_excludes": [
            "author identity", "video identity", "original capability label",
            "benchmark questions and labels", "coverage-overlap judgments",
        ],
        "packet_sha256": digest(args.packet),
        "mapping_sha256": digest(args.mapping),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
