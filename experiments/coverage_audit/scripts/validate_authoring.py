#!/usr/bin/env python3
"""Validate an independent-authoring CSV against the blind coverage-audit packet."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CAPABILITIES = {
    "occlusion_permanence", "dynamic_relation", "action_goal",
    "physical_prediction", "deformation_state", "reference_frame",
    "orientation_facing", "temporal_order", "entity_attribute",
    "action_event", "static_relation", "other_askable",
}
LEVELS = {"low", "intermediate", "high"}
FORMATS = {"true_false", "multiple_choice", "short_answer"}
BOOL_FIELDS = {
    "intent_required", "reference_transform", "prediction_or_counterfactual",
    "temporal_integration", "occlusion_tracking", "deformation_tracking",
    "human_involved", "needs_full_video",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    with args.blind_packet.open(newline="", encoding="utf-8") as handle:
        ids = [row["audit_item_id"] for row in csv.DictReader(handle)]
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    counts = Counter(row.get("audit_item_id", "") for row in rows)
    if len(rows) != 360:
        errors.append(f"expected 360 rows, found {len(rows)}")
    if set(counts) != set(ids):
        errors.append("annotation IDs do not equal blind-packet IDs")
    if any(counts[item_id] != 3 for item_id in ids):
        errors.append("every ID must have exactly 3 proposals")
    seen = set()
    for line, row in enumerate(rows, start=2):
        key = (row.get("audit_item_id"), row.get("proposal_no"))
        if key in seen:
            errors.append(f"line {line}: duplicate proposal key {key}")
        seen.add(key)
        if not row.get("question_text_cn", "").strip():
            errors.append(f"line {line}: blank question")
        if row.get("primary_capability") not in CAPABILITIES:
            errors.append(f"line {line}: invalid capability")
        if row.get("reasoning_level") not in LEVELS:
            errors.append(f"line {line}: invalid reasoning level")
        if row.get("question_format") not in FORMATS:
            errors.append(f"line {line}: invalid question format")
        for field in BOOL_FIELDS:
            if row.get(field) not in {"0", "1"}:
                errors.append(f"line {line}: {field} must be 0/1")
    if errors:
        raise SystemExit("\n".join(errors[:100]))
    print(f"valid: {len(rows)} proposals, {len(counts)} videos, 3 proposals/video")


if __name__ == "__main__":
    main()
