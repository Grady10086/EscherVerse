#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SINGLE_CHOICE = re.compile(r"^\s*(?:option\s+)?([A-D])\s*[\).]?\s*$", re.I)
MULTIPLE_SELECT = re.compile(
    r"^\s*[A-D](?:\s*(?:,|;|\band\b|&)\s*[A-D])*\s*$", re.I
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_MANIFEST_SCHEMA = "pair-model-run-v2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_free_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[\s\"']+|[\s\"'.!?]+$", "", value)
    return " ".join(value.split())


def extract_answer(raw_response: str) -> tuple[str | None, str]:
    matches = ANSWER_TAG.findall(raw_response or "")
    if not matches:
        return None, "missing_answer_tags"
    if len(matches) != 1:
        return None, "multiple_answer_tags"
    content = matches[0].strip()
    if not content:
        return None, "empty_answer"
    return content, "ok"


def parse_choice(value: str, question_type: str) -> tuple[str | None, str]:
    if question_type == "Single-Choice":
        match = SINGLE_CHOICE.fullmatch(value)
        return (match.group(1).upper(), "ok") if match else (None, "invalid_single_choice")
    if question_type == "Multiple-Select":
        if not MULTIPLE_SELECT.fullmatch(value):
            return None, "invalid_multiple_select"
        letters = re.findall(r"\b[A-D]\b", value.upper())
        if len(letters) != len(set(letters)):
            return None, "duplicate_multiple_select_option"
        return ",".join(sorted(letters)), "ok"
    if question_type == "True/False":
        normalized = normalize_free_text(value)
        if normalized not in {"true", "false"}:
            return None, "invalid_true_false"
        return normalized.title(), "ok"
    if question_type == "Fill-in-the-Blank":
        normalized = normalize_free_text(value)
        return (normalized, "ok") if normalized else (None, "empty_fill_blank")
    return None, "unknown_question_type"


def canonical_gold(value: str, question_type: str) -> str:
    parsed, status = parse_choice(value, question_type)
    if status != "ok" or parsed is None:
        raise ValueError(
            f"Gold answer cannot be parsed for {question_type}: {value!r} ({status})"
        )
    return parsed


def score_response(
    raw_response: str, ground_truth: str, question_type: str
) -> dict[str, Any]:
    answer, tag_status = extract_answer(raw_response)
    gold = canonical_gold(ground_truth, question_type)
    if tag_status != "ok" or answer is None:
        return {
            "correct": 0,
            "parse_status": tag_status,
            "parsed_answer": "",
            "ground_truth_canonical": gold,
        }
    parsed, parse_status = parse_choice(answer, question_type)
    return {
        "correct": int(parse_status == "ok" and parsed == gold),
        "parse_status": parse_status,
        "parsed_answer": parsed or "",
        "ground_truth_canonical": gold,
    }


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if not isinstance(payload, list):
        raise ValueError("Raw prediction JSON must be a list or contain results[]")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_video_media_audit(path: Path) -> dict[str, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Video media audit root must be an object")
    if report.get("missing") or report.get("extra") or report.get("probe_errors"):
        raise ValueError("Video media audit contains unresolved inventory errors")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Video media audit requires non-empty records")
    by_video: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Video media audit record must be an object")
        video = str(record.get("video", ""))
        digest = str(record.get("sha256", ""))
        if not video or video in by_video or not SHA256.fullmatch(digest):
            raise ValueError(f"Invalid video media audit digest for {video!r}")
        by_video[video] = digest
    return by_video


def validate_run_manifest(
    manifest: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    benchmark_sha256: str,
    final_pairs_sha256: str,
    raw_predictions_sha256: str,
    video_media_audit_sha256: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"Run manifest schema must be {RUN_MANIFEST_SCHEMA!r}")
    if manifest.get("final_pairs_sha256") != final_pairs_sha256:
        raise ValueError("Run manifest final_pairs_sha256 does not match input")
    if manifest.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("Run manifest benchmark_sha256 does not match input")
    if manifest.get("raw_predictions_sha256") != raw_predictions_sha256:
        raise ValueError("Run manifest raw_predictions_sha256 does not match input")
    if manifest.get("video_media_audit_sha256") != video_media_audit_sha256:
        raise ValueError("Run manifest video_media_audit_sha256 does not match input")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Run manifest requires a non-empty runs list")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for position, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Run manifest entry {position} must be an object")
        model = str(run.get("model", "")).strip()
        run_id = str(run.get("run_id", "")).strip()
        model_artifact = str(run.get("model_artifact", "")).strip()
        prompt_digest = str(run.get("prompt_template_sha256", "")).strip()
        if not model or not run_id or not model_artifact:
            raise ValueError(
                f"Run manifest entry {position} requires model, run_id, and model_artifact"
            )
        if not SHA256.fullmatch(prompt_digest):
            raise ValueError(
                f"Run manifest entry {position} has invalid prompt_template_sha256"
            )
        if not isinstance(run.get("inference_config"), dict):
            raise ValueError(
                f"Run manifest entry {position} requires inference_config object"
            )
        key = (model, run_id)
        if key in by_key:
            raise ValueError(f"Duplicate run manifest entry for {model} / {run_id}")
        by_key[key] = run
    record_keys = {
        (str(record.get("model", "")).strip(), str(record.get("run_id", "")).strip())
        for record in raw_records
    }
    if record_keys != set(by_key):
        raise ValueError(
            "Run manifest model/run_id set differs from raw predictions: "
            f"manifest_only={sorted(set(by_key) - record_keys)}, "
            f"raw_only={sorted(record_keys - set(by_key))}"
        )
    return by_key


def score_records(
    benchmark: list[dict[str, Any]],
    final_pairs: list[dict[str, str]],
    raw_records: list[dict[str, Any]],
    run_manifest_sha256: str,
    run_metadata: dict[tuple[str, str], dict[str, Any]],
    video_sha256_by_name: dict[str, str],
) -> list[dict[str, Any]]:
    benchmark_by_index = {int(row["index"]): row for row in benchmark}
    if len(benchmark_by_index) != len(benchmark):
        raise ValueError("Benchmark contains duplicate index values")
    required = {
        int(row[field])
        for row in final_pairs
        for field in ("arm_a_index", "arm_b_index")
    }
    for pair in final_pairs:
        for arm in ("arm_a", "arm_b"):
            index = int(pair[f"{arm}_index"])
            benchmark_row = benchmark_by_index.get(index)
            if benchmark_row is None:
                raise ValueError(f"Benchmark index not found: {index}")
            expected = {
                "video": str(benchmark_row.get("P", "")),
                f"{arm}_question": str(benchmark_row.get("Q", "")),
                f"{arm}_answer": str(benchmark_row.get("A", "")),
                f"{arm}_question_type": str(benchmark_row.get("question_type", "")),
            }
            for field, value in expected.items():
                if pair.get(field, "") != value:
                    raise ValueError(
                        f"Final pair source mismatch for index {index}: {field}"
                    )
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in raw_records:
        model = str(record.get("model", "")).strip()
        run_id = str(record.get("run_id", "")).strip()
        if not model or not run_id:
            raise ValueError("Every raw prediction requires model and run_id")
        index = int(record["benchmark_index"])
        key = (model, index)
        if key in by_key:
            raise ValueError(f"Duplicate raw prediction for {model} index {index}")
        by_key[key] = record
    models = sorted({model for model, _ in by_key})
    if not models:
        raise ValueError("Raw prediction file contains no models")
    output: list[dict[str, Any]] = []
    for model in models:
        run_ids = {
            str(record["run_id"])
            for (candidate_model, _), record in by_key.items()
            if candidate_model == model
        }
        if len(run_ids) != 1:
            raise ValueError(
                f"{model}: expected one run_id for the frozen item set, found "
                + ", ".join(sorted(run_ids))
            )
        run_id = next(iter(run_ids))
        metadata = run_metadata.get((model, run_id))
        if metadata is None:
            raise ValueError(f"Missing run metadata for {model} / {run_id}")
        available = {index for candidate_model, index in by_key if candidate_model == model}
        missing = sorted(required - available)
        extra = sorted(available - required)
        if missing or extra:
            raise ValueError(
                f"{model}: raw index set differs from final pairs; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for index in sorted(required):
            source = by_key[(model, index)]
            benchmark_row = benchmark_by_index.get(index)
            if benchmark_row is None:
                raise ValueError(f"Benchmark index not found: {index}")
            if str(source.get("video", "")) != str(benchmark_row.get("P", "")):
                raise ValueError(f"Video mismatch for {model} index {index}")
            video = str(benchmark_row.get("P", ""))
            video_digest = video_sha256_by_name.get(video)
            if video_digest is None:
                raise ValueError(f"Video content digest not found: {video}")
            raw_response = str(source.get("raw_response", ""))
            result = score_response(
                raw_response,
                str(benchmark_row.get("A", "")),
                str(benchmark_row.get("question_type", "")),
            )
            output.append(
                {
                    "model": model,
                    "run_id": run_id,
                    "model_artifact": str(metadata["model_artifact"]),
                    "prompt_template_sha256": str(metadata["prompt_template_sha256"]),
                    "benchmark_index": index,
                    "video": video,
                    "video_sha256": video_digest,
                    "question_type": str(benchmark_row.get("question_type", "")),
                    **result,
                    "raw_response_sha256": sha256_bytes(raw_response.encode("utf-8")),
                    "run_manifest_sha256": run_manifest_sha256,
                    "model_input_sha256": sha256_bytes(
                        "\x1f".join(
                            (
                                str(index),
                                video,
                                video_digest,
                                str(benchmark_row.get("Q", "")),
                                str(benchmark_row.get("question_type", "")),
                                str(metadata["prompt_template_sha256"]),
                            )
                        ).encode("utf-8")
                    ),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly rescore raw matched-pair responses")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--final-pairs", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--video-media-audit", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    raw_records = read_json_records(args.raw_predictions)
    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest root must be an object")
    run_metadata = validate_run_manifest(
        manifest,
        raw_records,
        benchmark_sha256=file_sha256(args.benchmark),
        final_pairs_sha256=file_sha256(args.final_pairs),
        raw_predictions_sha256=file_sha256(args.raw_predictions),
        video_media_audit_sha256=file_sha256(args.video_media_audit),
    )
    records = score_records(
        benchmark,
        read_csv(args.final_pairs),
        raw_records,
        file_sha256(args.run_manifest),
        run_metadata,
        load_video_media_audit(args.video_media_audit),
    )
    fields = list(records[0]) if records else []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema_version": "pair-scored-predictions-v1",
        "benchmark_sha256": file_sha256(args.benchmark),
        "final_pairs_sha256": file_sha256(args.final_pairs),
        "raw_predictions_sha256": file_sha256(args.raw_predictions),
        "run_manifest_sha256": file_sha256(args.run_manifest),
        "video_media_audit_sha256": file_sha256(args.video_media_audit),
        "scored_predictions_sha256": file_sha256(args.output_csv),
        "scored_rows": len(records),
        "models": sorted({str(record["model"]) for record in records}),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(records)} model-item responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
