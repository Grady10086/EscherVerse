#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from analyze_broad_candidate_pairs import (
    PAIR_TYPES,
    file_sha256,
    merge_predictions,
    standard_gold_format,
    subset_result,
)
from build_candidate_pool import validate_candidate_freeze


def canonical_whitespace(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def validate_sources(
    candidates: list[dict[str, str]],
    benchmark_by_index: dict[int, dict[str, Any]],
) -> None:
    for candidate in candidates:
        for arm in ("a", "b"):
            index = int(candidate[f"arm_{arm}_index"])
            item = benchmark_by_index.get(index)
            if item is None:
                raise ValueError(f"Candidate index is absent from benchmark: {index}")
            expected = (
                candidate["video"],
                canonical_whitespace(candidate[f"arm_{arm}_question"]),
                candidate[f"arm_{arm}_answer"],
                candidate[f"arm_{arm}_question_type"],
            )
            observed = (
                str(item.get("P", "")),
                canonical_whitespace(item.get("Q", "")),
                str(item.get("A", "")),
                str(item.get("question_type", "")),
            )
            if expected != observed:
                raise ValueError(f"Candidate source mismatch for index {index}")


def render_markdown(report: dict[str, Any]) -> str:
    def pct(value: object) -> str:
        return "NA" if value is None else f"{100 * float(value):.1f}%"

    lines = [
        f"# {report['title']}",
        "",
        f"- Model: `{report['model']}`",
        f"- Candidate pairs: {report['candidate_pair_count']}",
        f"- Unique items: {report['unique_item_count']}",
        f"- Prediction coverage: {report['prediction_item_coverage']['covered']}/"
        f"{report['prediction_item_coverage']['required']}",
        f"- Interpretation: {report['interpretation']}",
        "",
        "| Subset | Covered/requested | Arm A | Arm B | B-A | P(B wrong | A correct) | A correct/B wrong | A wrong/B correct | Sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "all_candidates",
        "standard_gold_format",
        *PAIR_TYPES,
        "same_question_type",
        *(f"{pair_type}_same_question_type" for pair_type in PAIR_TYPES),
    ]
    for name in order:
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
            report["reporting_guardrail"],
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
    validate_sources(candidates, benchmark_by_index)
    predictions, prediction_provenance = merge_predictions(
        args.result, benchmark_by_index
    )
    required_indices = {
        int(row[field])
        for row in candidates
        for field in ("arm_a_index", "arm_b_index")
    }
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
    }
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
        "schema_version": "same-video-additional-analysis-v1",
        "title": args.title,
        "model": args.model,
        "analysis_scope": args.analysis_scope,
        "interpretation": args.interpretation,
        "candidate_pairs_sha256": file_sha256(args.candidate_pairs),
        "benchmark_sha256": file_sha256(args.benchmark),
        "candidate_pair_count": len(candidates),
        "unique_item_count": len(required_indices),
        "prediction_item_coverage": {
            "required": len(required_indices),
            "covered": len(required_indices & set(predictions)),
            "missing_indices": sorted(required_indices - set(predictions)),
        },
        "prediction_provenance": prediction_provenance,
        "nonstandard_gold_indices": sorted(
            index
            for index in required_indices
            if not standard_gold_format(benchmark_by_index[index])
        ),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "subsets": results,
        "reporting_guardrail": args.reporting_guardrail,
    }
    if (
        not args.allow_partial_coverage
        and report["prediction_item_coverage"]["covered"] != len(required_indices)
    ):
        raise ValueError("Model output does not cover all required items")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze outcome-blind same-video category-proxy candidates."
    )
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--title", default="Additional Same-Video Candidate-Pair Analysis"
    )
    parser.add_argument(
        "--analysis-scope",
        default="outcome-blind additional same-video category-proxy pairs",
    )
    parser.add_argument(
        "--interpretation",
        default=(
            "outcome-blind same-video category-proxy sample; not strict "
            "same-event/entity/evidence matched pairs."
        ),
    )
    parser.add_argument(
        "--reporting-guardrail",
        default=(
            "Arm A/B are frozen category-proxy directions. Selection used neither "
            "model predictions nor correctness; no candidate may be removed after "
            "viewing these results."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-iterations", type=int, default=100000)
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Allow a labeled diagnostic report when historical outputs are incomplete",
    )
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
