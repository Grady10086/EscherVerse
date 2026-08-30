#!/usr/bin/env python3
"""Build blinded semantic-coverage packets from the 64-frame-reviewed proposals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-a", type=Path, required=True)
    parser.add_argument("--author-b", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()

    generated_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.generated):
        generated_by_video[row["audit_item_id"]].append(row)

    proposals = []
    excluded = Counter()
    for author_id, path in [("author_a", args.author_a), ("author_b", args.author_b)]:
        for row in read_csv(path):
            status = row.get("review_status", "")
            if row.get("excluded_from_coverage") == "1":
                excluded["rejected"] += 1
                continue
            if status == "still_uncertain" or row.get("needs_full_video") == "1":
                excluded["still_uncertain"] += 1
                continue
            proposals.append((author_id, row))

    proposals.sort(key=lambda item: (item[1]["audit_item_id"], item[0], int(item[1]["proposal_no"])))
    packet_fields = ["blind_proposal_id", "audit_item_id", "reference_question", "generated_questions_json"]
    mapping_fields = ["blind_proposal_id", "author_id", "audit_item_id", "proposal_no", "primary_capability"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as packet_handle, args.mapping.open(
        "w", newline="", encoding="utf-8"
    ) as mapping_handle:
        packet_writer = csv.DictWriter(packet_handle, fieldnames=packet_fields)
        mapping_writer = csv.DictWriter(mapping_handle, fieldnames=mapping_fields)
        packet_writer.writeheader()
        mapping_writer.writeheader()
        for index, (author_id, row) in enumerate(proposals, start=1):
            blind_id = f"judgment-{index:04d}"
            generated = [
                {"generated_id": item["generated_id"], "question": item["question"]}
                for item in generated_by_video[row["audit_item_id"]]
            ]
            packet_writer.writerow(
                {
                    "blind_proposal_id": blind_id,
                    "audit_item_id": row["audit_item_id"],
                    "reference_question": row["question_text_cn"],
                    "generated_questions_json": json.dumps(generated, ensure_ascii=False),
                }
            )
            mapping_writer.writerow(
                {
                    "blind_proposal_id": blind_id,
                    "author_id": author_id,
                    "audit_item_id": row["audit_item_id"],
                    "proposal_no": row["proposal_no"],
                    "primary_capability": row["primary_capability"],
                }
            )

    meta = {
        "schema": "escher-audit-coverage-judgment-packet-v1",
        "included_proposals": len(proposals),
        "excluded": dict(sorted(excluded.items())),
        "unique_videos": len({row["audit_item_id"] for _, row in proposals}),
        "packet_excludes": ["author identity", "capability label", "benchmark answer", "model result"],
        "packet_sha256": sha256(args.output),
        "mapping_sha256": sha256(args.mapping),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
