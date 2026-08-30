#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATEGORY_NAMES = {
    1: "Object Permanence and Occlusion Tracking",
    2: "Dynamic Spatial Relationships",
    3: "Action and Intent-Driven Spatial Reasoning",
    4: "Predictive and Counterfactual Spatial Reasoning",
    5: "Object Deformation and State Transition",
    6: "Egocentric vs. Allocentric Reference Frames",
}

PAIR_TYPES = (
    {
        "name": "dynamic_vs_intent",
        "code": "dvi",
        "arm_a_category": 2,
        "arm_b_category": 3,
    },
    {
        "name": "camera_vs_actor_goal",
        "code": "cag",
        "arm_a_category": 6,
        "arm_b_category": 3,
    },
    {
        "name": "tracking_vs_prediction",
        "code": "tvp",
        "arm_a_category": 2,
        "arm_b_category": 4,
    },
)

CANDIDATE_FIELDS = (
    "candidate_pair_id",
    "packet_id",
    "pair_type",
    "video",
    "arm_a_index",
    "arm_a_category_id",
    "arm_a_category",
    "arm_a_scene_type",
    "arm_a_question_type",
    "arm_a_question",
    "arm_a_answer",
    "arm_b_index",
    "arm_b_category_id",
    "arm_b_category",
    "arm_b_scene_type",
    "arm_b_question_type",
    "arm_b_question",
    "arm_b_answer",
    "question_type_match",
    "scene_type_match",
    "pair_sha256",
)

EXCLUSION_CODES = (
    "video_unavailable_or_corrupt",
    "source_row_mismatch",
    "duplicate_or_wrong_video",
    "other_technical",
)

REVIEW_FIELDS = CANDIDATE_FIELDS + (
    "rater_id",
    "review_status",
    "same_event",
    "same_critical_entities",
    "same_critical_evidence",
    "arm_a_visual_sufficiency",
    "arm_b_visual_sufficiency",
    "arm_a_well_posed",
    "arm_b_well_posed",
    "arm_a_reasoning_level",
    "arm_b_reasoning_level",
    "arm_a_intent_conditioned",
    "arm_b_intent_conditioned",
    "arm_a_perspective",
    "arm_b_perspective",
    "arm_a_evidence_spans",
    "arm_b_evidence_spans",
    "arm_a_critical_entities_text",
    "arm_b_critical_entities_text",
    "arm_a_critical_event_text",
    "arm_b_critical_event_text",
    "arm_a_required_visual_fact",
    "arm_b_required_visual_fact",
    "intended_contrast_valid",
    "packet_best_candidate",
    "exclusion_code",
    "exclusion_reason",
    "notes",
)


def stable_digest(parts: Iterable[Any]) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_digest(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> str:
    digest = hashlib.sha256()
    field_list = tuple(fields)
    for row in rows:
        payload = "\x1f".join(str(row.get(field, "")) for field in field_list)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_cardinalities(candidates: list[dict[str, Any]]) -> dict[str, int]:
    item_ids = {
        str(row[field])
        for row in candidates
        for field in ("arm_a_index", "arm_b_index")
    }
    return {
        "candidate_pairs": len(candidates),
        "packets": len({str(row["packet_id"]) for row in candidates}),
        "unique_items": len(item_ids),
        "videos": len({str(row["video"]) for row in candidates}),
    }


def validate_candidate_freeze(
    candidate_path: Path,
    summary_path: Path,
    expected_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        if fields != list(CANDIDATE_FIELDS):
            raise ValueError("Candidate CSV header does not match CANDIDATE_FIELDS")
        candidates = list(reader)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Candidate summary root must be an object")
    checks = {
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "candidate_rows_sha256": rows_digest(candidates, CANDIDATE_FIELDS),
    }
    if (
        expected_manifest_sha256 is not None
        and checks["candidate_manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("Candidate CSV does not match the external frozen SHA-256")
    for field, observed in checks.items():
        if summary.get(field) != observed:
            raise ValueError(f"Candidate freeze mismatch for {field}")
    observed_cardinalities = candidate_cardinalities(candidates)
    if summary.get("cardinalities") != observed_cardinalities:
        raise ValueError(
            "Candidate freeze cardinalities changed: "
            f"expected={summary.get('cardinalities')}, "
            f"observed={observed_cardinalities}"
        )
    candidate_ids: set[str] = set()
    for row in candidates:
        candidate_id = row["candidate_pair_id"]
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"Duplicate or blank candidate_pair_id: {candidate_id!r}")
        candidate_ids.add(candidate_id)
        expected_pair_digest = stable_digest(
            row[field] for field in CANDIDATE_FIELDS if field != "pair_sha256"
        )
        if row["pair_sha256"] != expected_pair_digest:
            raise ValueError(f"Candidate row digest mismatch: {candidate_id}")
    return candidates, summary


def canonical_category(raw: object) -> tuple[int, str]:
    match = re.match(r"\s*Category\s+([1-6])\s*:", str(raw or ""), re.I)
    if not match:
        raise ValueError(f"Cannot canonicalize category: {raw!r}")
    category_id = int(match.group(1))
    return category_id, CATEGORY_NAMES[category_id]


def normalize_text(value: object) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines())


def canonical_scene_type(raw: object) -> str:
    value = re.sub(r"[^a-z]", "", str(raw or "").lower())
    if value in {"humancentric", "humancentered", "humanoriented"}:
        return "Human-Centric"
    if value in {"objectcentric", "objectcentered"}:
        return "Object-Centric"
    return normalize_text(raw) or "Unknown"


def canonical_question_type(raw: object) -> str:
    value = re.sub(r"[^a-z]", "", str(raw or "").lower())
    mapping = {
        "singlechoice": "Single-Choice",
        "multipleselect": "Multiple-Select",
        "truefalse": "True/False",
        "fillintheblank": "Fill-in-the-Blank",
    }
    return mapping.get(value, normalize_text(raw) or "Unknown")


def canonical_answer_key(answer: str, question_type: str) -> str:
    normalized = " ".join(answer.strip().split())
    if question_type == "Multiple-Select":
        parts = [part.strip().upper() for part in re.split(r"[,，;/]", normalized)]
        if parts and all(part in {"A", "B", "C", "D"} for part in parts):
            return ",".join(sorted(set(parts)))
    if question_type in {"Single-Choice", "True/False"}:
        return normalized.upper()
    return normalized.lower().rstrip(".。")


def load_scoreable_items(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("Benchmark root must be a JSON array")

    exclusions: Counter[str] = Counter()
    duplicate_group_counts: Counter[str] = Counter()
    exclusion_indices: dict[str, list[Any]] = defaultdict(list)
    items: list[dict[str, Any]] = []
    seen_indices: set[int] = set()

    for row in source_rows:
        try:
            category_id, category = canonical_category(row.get("C", ""))
        except ValueError:
            exclusions["invalid_category_field"] += 1
            exclusion_indices["invalid_category_field"].append(row.get("index"))
            continue

        video = normalize_text(row.get("P", ""))
        question = normalize_text(row.get("Q", ""))
        answer = normalize_text(row.get("A", ""))
        try:
            benchmark_index = int(row.get("index"))
        except (TypeError, ValueError):
            exclusions["invalid_benchmark_index"] += 1
            exclusion_indices["invalid_benchmark_index"].append(row.get("index"))
            continue

        missing_reason = ""
        if not video:
            missing_reason = "missing_video"
        elif not question:
            missing_reason = "missing_question"
        elif not answer:
            missing_reason = "blank_answer"
        if missing_reason:
            exclusions[missing_reason] += 1
            exclusion_indices[missing_reason].append(benchmark_index)
            continue
        if benchmark_index in seen_indices:
            raise ValueError(f"Duplicate benchmark index: {benchmark_index}")
        seen_indices.add(benchmark_index)

        question_type = canonical_question_type(row.get("question_type", ""))
        items.append(
            {
                "benchmark_index": benchmark_index,
                "video": video,
                "question": question,
                "answer": answer,
                "answer_key": canonical_answer_key(answer, question_type),
                "category_id": category_id,
                "category": category,
                "scene_type": canonical_scene_type(row.get("scene_type", "")),
                "question_type": question_type,
            }
        )

    pre_dedup_scoreable_rows = len(items)
    by_video_question: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_video_question[(item["video"], item["question"])].append(item)

    deduplicated_items: list[dict[str, Any]] = []
    for duplicate_group in by_video_question.values():
        category_ids = {item["category_id"] for item in duplicate_group}
        answers = {item["answer_key"] for item in duplicate_group}
        if len(category_ids) > 1:
            reason = "duplicate_category_conflict"
            duplicate_group_counts[reason] += 1
            exclusions[reason] += len(duplicate_group)
            exclusion_indices[reason].extend(
                item["benchmark_index"] for item in duplicate_group
            )
            continue
        if len(answers) > 1:
            reason = "duplicate_answer_conflict"
            duplicate_group_counts[reason] += 1
            exclusions[reason] += len(duplicate_group)
            exclusion_indices[reason].extend(
                item["benchmark_index"] for item in duplicate_group
            )
            continue

        duplicate_group.sort(key=lambda item: item["benchmark_index"])
        deduplicated_items.append(duplicate_group[0])
        redundant = duplicate_group[1:]
        if redundant:
            reason = "exact_duplicate_redundant"
            duplicate_group_counts[reason] += 1
            exclusions[reason] += len(redundant)
            exclusion_indices[reason].extend(
                item["benchmark_index"] for item in redundant
            )
    items = sorted(deduplicated_items, key=lambda item: item["benchmark_index"])

    all_videos = [normalize_text(row.get("P", "")) for row in source_rows]
    all_video_counts = Counter(video for video in all_videos if video)
    scoreable_video_counts = Counter(item["video"] for item in items)
    audit = {
        "source_rows": len(source_rows),
        "source_unique_videos": len(all_video_counts),
        "source_multi_item_videos": sum(count > 1 for count in all_video_counts.values()),
        "pre_dedup_scoreable_rows": pre_dedup_scoreable_rows,
        "scoreable_rows": len(items),
        "scoreable_unique_videos": len(scoreable_video_counts),
        "scoreable_multi_item_videos": sum(
            count > 1 for count in scoreable_video_counts.values()
        ),
        "excluded_rows": sum(exclusions.values()),
        "exclusions": dict(sorted(exclusions.items())),
        "duplicate_groups": dict(sorted(duplicate_group_counts.items())),
        "exclusion_indices": {
            key: sorted(values) for key, values in sorted(exclusion_indices.items())
        },
    }
    return items, audit


def group_items(items: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        grouped[item["video"]][item["category_id"]].append(item)
    for category_map in grouped.values():
        for category_items in category_map.values():
            category_items.sort(key=lambda item: item["benchmark_index"])
    return grouped


def build_candidate_pool(
    items: list[dict[str, Any]], *, seed: int, max_videos_per_type: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_items(items)
    candidates: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for pair_type in PAIR_TYPES:
        name = pair_type["name"]
        category_a = pair_type["arm_a_category"]
        category_b = pair_type["arm_b_category"]
        source_videos = [
            video
            for video, categories in grouped.items()
            if category_a in categories and category_b in categories
        ]
        source_videos.sort(
            key=lambda video: (stable_digest((seed, name, video)), video)
        )
        selected_videos = source_videos[:max_videos_per_type]
        source_pairs = sum(
            len(grouped[video][category_a]) * len(grouped[video][category_b])
            for video in source_videos
        )
        selected_pair_count = 0
        same_type_count = 0

        for packet_number, video in enumerate(selected_videos, start=1):
            packet_id = f"pair-{pair_type['code']}-{packet_number:03d}"
            pair_number = 0
            for arm_a in grouped[video][category_a]:
                for arm_b in grouped[video][category_b]:
                    pair_number += 1
                    selected_pair_count += 1
                    question_type_match = arm_a["question_type"] == arm_b["question_type"]
                    same_type_count += int(question_type_match)
                    candidate = {
                        "candidate_pair_id": f"{packet_id}-p{pair_number:02d}",
                        "packet_id": packet_id,
                        "pair_type": name,
                        "video": video,
                        "arm_a_index": arm_a["benchmark_index"],
                        "arm_a_category_id": arm_a["category_id"],
                        "arm_a_category": arm_a["category"],
                        "arm_a_scene_type": arm_a["scene_type"],
                        "arm_a_question_type": arm_a["question_type"],
                        "arm_a_question": arm_a["question"],
                        "arm_a_answer": arm_a["answer"],
                        "arm_b_index": arm_b["benchmark_index"],
                        "arm_b_category_id": arm_b["category_id"],
                        "arm_b_category": arm_b["category"],
                        "arm_b_scene_type": arm_b["scene_type"],
                        "arm_b_question_type": arm_b["question_type"],
                        "arm_b_question": arm_b["question"],
                        "arm_b_answer": arm_b["answer"],
                        "question_type_match": "yes" if question_type_match else "no",
                        "scene_type_match": "yes"
                        if arm_a["scene_type"] == arm_b["scene_type"]
                        else "no",
                    }
                    candidate["pair_sha256"] = stable_digest(
                        candidate[field]
                        for field in CANDIDATE_FIELDS
                        if field != "pair_sha256"
                    )
                    candidates.append(candidate)

        summaries[name] = {
            "retrieval_categories": [category_a, category_b],
            "source_videos": len(source_videos),
            "source_item_pairs": source_pairs,
            "selected_videos": len(selected_videos),
            "selected_item_pairs": selected_pair_count,
            "selected_same_question_type_pairs": same_type_count,
        }

    candidate_ids = [row["candidate_pair_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("Candidate pair IDs are not unique")
    return candidates, summaries


def build_review_template(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    annotation_fields = REVIEW_FIELDS[len(CANDIDATE_FIELDS) :]
    for candidate in candidates:
        row = dict(candidate)
        row.update({field: "" for field in annotation_fields})
        row["review_status"] = "draft"
        rows.append(row)
    return rows


def required_videos(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted({str(candidate["video"]) for candidate in candidates})


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_summary(summary: dict[str, Any]) -> str:
    audit = summary["source_audit"]
    lines = [
        "# Matched-Pair Candidate Pool Audit",
        "",
        f"- Benchmark SHA-256: `{summary['benchmark_sha256']}`",
        f"- Candidate CSV SHA-256: `{summary['candidate_manifest_sha256']}`",
        f"- Candidate rows SHA-256: `{summary['candidate_rows_sha256']}`",
        f"- Pair-content SHA-256: `{summary['pair_content_digest_sha256']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Source rows: {audit['source_rows']:,}",
        f"- Rows passing field checks before duplicate audit: {audit['pre_dedup_scoreable_rows']:,}",
        f"- Scoreable rows: {audit['scoreable_rows']:,}",
        f"- Scoreable videos: {audit['scoreable_unique_videos']:,}",
        f"- Scoreable multi-item videos: {audit['scoreable_multi_item_videos']:,}",
        f"- Excluded rows: {audit['excluded_rows']:,}",
        f"- Source videos: {audit['source_unique_videos']:,}",
        f"- Multi-item videos: {audit['source_multi_item_videos']:,}",
        "",
        "## Pre-pair Attrition",
        "",
        "| Reason | Rows | Duplicate groups |",
        "|---|---:|---:|",
    ]
    for reason, count in audit["exclusions"].items():
        group_count = audit["duplicate_groups"].get(reason)
        lines.append(
            f"| {reason} | {count} | "
            f"{group_count if group_count is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval Pools",
            "",
            "| Pair type | Source videos | Source item pairs | Selected videos | Selected item pairs | Same-format selected pairs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, record in summary["pair_types"].items():
        lines.append(
            f"| {name} | {record['source_videos']} | "
            f"{record['source_item_pairs']} | {record['selected_videos']} | "
            f"{record['selected_item_pairs']} | "
            f"{record['selected_same_question_type_pairs']} |"
        )
    lines.extend(
        [
            "",
            f"Total exported candidate pairs: {summary['exported_candidate_pairs']:,}.",
            "",
            "Category pairs are retrieval proxies only. No reasoning-level or intent label has been assigned.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen matched-pair candidate pool.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--review-template", type=Path, required=True)
    parser.add_argument("--required-videos", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-videos-per-type", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items, source_audit = load_scoreable_items(args.benchmark)
    candidates, pair_summaries = build_candidate_pool(
        items,
        seed=args.seed,
        max_videos_per_type=args.max_videos_per_type,
    )
    write_csv(args.candidate_pairs, candidates, CANDIDATE_FIELDS)
    write_csv(args.review_template, build_review_template(candidates), REVIEW_FIELDS)
    args.required_videos.parent.mkdir(parents=True, exist_ok=True)
    args.required_videos.write_text(
        "\n".join(required_videos(candidates)) + "\n", encoding="utf-8"
    )

    summary = {
        "benchmark": str(args.benchmark),
        "benchmark_sha256": file_sha256(args.benchmark),
        "seed": args.seed,
        "max_videos_per_type": args.max_videos_per_type,
        "source_audit": source_audit,
        "pair_types": pair_summaries,
        "exported_candidate_pairs": len(candidates),
        "pair_content_digest_sha256": stable_digest(
            row["pair_sha256"] for row in candidates
        ),
        "candidate_rows_sha256": rows_digest(candidates, CANDIDATE_FIELDS),
        "candidate_manifest_sha256": file_sha256(args.candidate_pairs),
        "cardinalities": candidate_cardinalities(candidates),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_summary(summary), encoding="utf-8")

    print(f"Wrote {len(candidates)} candidate pairs to {args.candidate_pairs}")
    print(f"Wrote review template to {args.review_template}")
    print(f"Wrote {len(required_videos(candidates))} required videos to {args.required_videos}")
    print(f"Candidate manifest digest: {summary['candidate_manifest_sha256']}")


if __name__ == "__main__":
    main()
