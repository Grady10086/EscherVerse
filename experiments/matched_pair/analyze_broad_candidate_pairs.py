#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from analyze_matched_pairs import analyze_subset
from build_candidate_pool import CANDIDATE_FIELDS, validate_candidate_freeze


PAIR_TYPES = (
    "dynamic_vs_intent",
    "camera_vs_actor_goal",
    "tracking_vs_prediction",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_question(value: object) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").strip().splitlines())


def standard_gold_format(item: dict[str, Any]) -> bool:
    question_type = str(item.get("question_type", ""))
    answer = str(item.get("A", "")).strip()
    if question_type == "Single-Choice":
        return re.fullmatch(r"[A-D]", answer) is not None
    if question_type == "Multiple-Select":
        return re.fullmatch(r"[A-D](?:\s*,\s*[A-D])*", answer) is not None
    if question_type == "True/False":
        return answer.lower() in {"true", "false"}
    return bool(answer)


def read_result_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        metadata = payload.get("metadata")
        return metadata if isinstance(metadata, dict) else {}, payload["results"]
    raise ValueError(f"Result file is not a list or results[] object: {path}")


def validate_candidate_sources(
    candidates: list[dict[str, str]], benchmark_by_index: dict[int, dict[str, Any]]
) -> None:
    for candidate in candidates:
        for arm in ("a", "b"):
            index = int(candidate[f"arm_{arm}_index"])
            source = benchmark_by_index.get(index)
            if source is None:
                raise ValueError(f"Candidate index is absent from benchmark: {index}")
            expected = {
                "video": candidate["video"],
                "question": canonical_question(candidate[f"arm_{arm}_question"]),
                "answer": candidate[f"arm_{arm}_answer"],
                "question_type": candidate[f"arm_{arm}_question_type"],
            }
            observed = {
                "video": str(source.get("P", "")),
                "question": canonical_question(source.get("Q", "")),
                "answer": str(source.get("A", "")),
                "question_type": str(source.get("question_type", "")),
            }
            if expected != observed:
                raise ValueError(
                    f"Candidate source mismatch for index {index}: "
                    f"expected={expected}, observed={observed}"
                )


def correctness(row: dict[str, Any]) -> int:
    value = row.get("is_correct")
    if value is True or str(value).strip().lower() in {"1", "true", "yes"}:
        return 1
    if value is False or value is None or str(value).strip().lower() in {
        "0",
        "false",
        "no",
        "none",
        "",
    }:
        return 0
    raise ValueError(f"Cannot parse is_correct={value!r}")


def merge_predictions(
    result_paths: list[Path],
    benchmark_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[int, int], dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    source_by_index: dict[int, str] = {}
    metadata_rows: list[dict[str, Any]] = []
    overlap_rows = 0
    for path in result_paths:
        metadata, rows = read_result_rows(path)
        metadata_rows.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "metadata": metadata,
            }
        )
        for row in rows:
            index = int(row["index"])
            benchmark = benchmark_by_index.get(index)
            if benchmark is None:
                raise ValueError(f"Result index is absent from benchmark: {index}")
            expected = (
                str(benchmark.get("P", "")),
                canonical_question(benchmark.get("Q", "")),
                str(benchmark.get("A", "")),
                str(benchmark.get("question_type", "")),
            )
            observed = (
                str(row.get("video", "")),
                canonical_question(row.get("question", "")),
                str(row.get("ground_truth", "")),
                str(row.get("question_type", "")),
            )
            if expected != observed:
                raise ValueError(
                    f"Result identity mismatch for index {index} in {path}"
                )
            previous = merged.get(index)
            if previous is not None:
                signatures = (
                    str(previous.get("model_prediction", "")).strip(),
                    correctness(previous),
                ), (
                    str(row.get("model_prediction", "")).strip(),
                    correctness(row),
                )
                if signatures[0] != signatures[1]:
                    raise ValueError(
                        f"Conflicting predictions for index {index}: "
                        f"{source_by_index[index]} vs {path}"
                    )
                overlap_rows += 1
                continue
            merged[index] = row
            source_by_index[index] = str(path)
    return (
        {index: correctness(row) for index, row in merged.items()},
        {
            "sources": metadata_rows,
            "unique_prediction_rows": len(merged),
            "consistent_overlap_rows": overlap_rows,
        },
    )


def read_pair_review(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["candidate_pair_id"]: row for row in rows}


def subset_result(
    requested: list[dict[str, str]],
    predictions: dict[int, int],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    available = [
        row
        for row in requested
        if int(row["arm_a_index"]) in predictions
        and int(row["arm_b_index"]) in predictions
    ]
    result = analyze_subset(
        available,
        predictions,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    result["requested_pairs"] = len(requested)
    result["coverage_rate"] = len(available) / len(requested) if requested else None
    return result


def render_markdown(report: dict[str, Any]) -> str:
    def pct(value: object) -> str:
        return "NA" if value is None else f"{100 * float(value):.1f}%"

    lines = [
        "# Broad Candidate-Pair Analysis",
        "",
        f"- Model: `{report['model']}`",
        f"- Candidate pairs: {report['candidate_pair_count']}",
        f"- Unique items: {report['unique_item_count']}",
        f"- Prediction coverage: {report['prediction_item_coverage']['covered']}/"
        f"{report['prediction_item_coverage']['required']}",
        "- Interpretation: exploratory same-video category-proxy analysis; the full set is not a strict matched-pair sample.",
        "",
        "| Subset | Covered/requested | Arm A | Arm B | B-A | P(B wrong | A correct) | A correct/B wrong | A wrong/B correct | Video-cluster sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "all_candidates",
        "standard_gold_format",
        *PAIR_TYPES,
        "coverage_reference_complete",
        "same_question_type",
        *(f"{pair_type}_same_question_type" for pair_type in PAIR_TYPES),
        "consensus_eligible",
    ]
    for name in order:
        if name not in report["subsets"]:
            continue
        result = report["subsets"][name]
        lines.append(
            f"| `{name}` | {result['n_pairs']}/{result['requested_pairs']} | "
            f"{pct(result['arm_a_accuracy'])} | {pct(result['arm_b_accuracy'])} | "
            f"{pct(result['paired_difference_b_minus_a'])} | "
            f"{pct(result['b_wrong_given_a_correct_rate'])} | "
            f"{result['a_correct_b_wrong']} | {result['a_wrong_b_correct']} | "
            f"{float(result['video_cluster_sign_flip_p']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Arm directions are the category-proxy directions frozen in the candidate manifest. Full-candidate results must not be described as controlling the same event, entities, and critical evidence for every pair.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates, _ = validate_candidate_freeze(
        args.candidate_pairs,
        args.candidate_summary,
        args.expected_candidate_sha256,
    )
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_by_index = {int(row["index"]): row for row in benchmark}
    if len(benchmark_by_index) != len(benchmark):
        raise ValueError("Benchmark contains duplicate indices")
    validate_candidate_sources(candidates, benchmark_by_index)
    predictions, prediction_provenance = merge_predictions(
        args.result, benchmark_by_index
    )
    coverage_reference_indices: set[int] = set()
    coverage_reference_provenance = None
    if args.coverage_reference:
        coverage_reference, coverage_reference_provenance = merge_predictions(
            args.coverage_reference, benchmark_by_index
        )
        coverage_reference_indices = set(coverage_reference)
    required_indices = {
        int(row[field])
        for row in candidates
        for field in ("arm_a_index", "arm_b_index")
    }
    unexpected = set(predictions) - set(benchmark_by_index)
    if unexpected:
        raise ValueError(f"Predictions contain unknown indices: {sorted(unexpected)[:10]}")

    review_a = read_pair_review(args.pair_review_a)
    review_b = read_pair_review(args.pair_review_b)
    candidate_ids = {row["candidate_pair_id"] for row in candidates}
    if set(review_a) != candidate_ids or set(review_b) != candidate_ids:
        raise ValueError("Pair-review candidate sets differ from candidate manifest")
    consensus_eligible = [
        row
        for row in candidates
        if review_a[row["candidate_pair_id"]]["intended_contrast_valid"] == "yes"
        and review_b[row["candidate_pair_id"]]["intended_contrast_valid"] == "yes"
    ]
    subsets: dict[str, list[dict[str, str]]] = {
        "all_candidates": candidates,
        "standard_gold_format": [
            row
            for row in candidates
            if standard_gold_format(benchmark_by_index[int(row["arm_a_index"])])
            and standard_gold_format(benchmark_by_index[int(row["arm_b_index"])])
        ],
        **{
            pair_type: [row for row in candidates if row["pair_type"] == pair_type]
            for pair_type in PAIR_TYPES
        },
        "same_question_type": [
            row for row in candidates if row["question_type_match"] == "yes"
        ],
        **{
            f"{pair_type}_same_question_type": [
                row
                for row in candidates
                if row["pair_type"] == pair_type
                and row["question_type_match"] == "yes"
            ]
            for pair_type in PAIR_TYPES
        },
        "consensus_eligible": consensus_eligible,
    }
    if coverage_reference_indices:
        subsets["coverage_reference_complete"] = [
            row
            for row in candidates
            if int(row["arm_a_index"]) in coverage_reference_indices
            and int(row["arm_b_index"]) in coverage_reference_indices
        ]
    results = {
        name: subset_result(
            rows,
            predictions,
            seed=args.seed + position,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        for position, (name, rows) in enumerate(subsets.items())
    }
    report = {
        "schema_version": "pair-broad-candidate-analysis-v1",
        "model": args.model,
        "analysis_scope": "all packet-closed candidates without pair-eligibility filtering",
        "candidate_pairs_sha256": file_sha256(args.candidate_pairs),
        "benchmark_sha256": file_sha256(args.benchmark),
        "candidate_pair_count": len(candidates),
        "unique_item_count": len(required_indices),
        "prediction_item_coverage": {
            "required": len(required_indices),
            "covered": len(required_indices & set(predictions)),
            "missing": len(required_indices - set(predictions)),
        },
        "prediction_provenance": prediction_provenance,
        "coverage_reference_provenance": coverage_reference_provenance,
        "nonstandard_gold_indices": sorted(
            index
            for index in required_indices
            if not standard_gold_format(benchmark_by_index[index])
        ),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "subsets": results,
        "reporting_guardrail": (
            "Exploratory same-video category-proxy analysis; do not describe all "
            "candidates as strict same-event/entity/evidence matched pairs."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze all frozen matched-pair candidate pairs with complete or partial logs."
    )
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--coverage-reference", type=Path, action="append")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pair-review-a", type=Path, required=True)
    parser.add_argument("--pair-review-b", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    report = run(args)
    coverage = report["prediction_item_coverage"]
    print(
        f"Analyzed {report['candidate_pair_count']} candidates with "
        f"{coverage['covered']}/{coverage['required']} item predictions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
