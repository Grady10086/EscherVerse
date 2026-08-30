#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_candidate_pool import CANDIDATE_FIELDS, file_sha256
from finalize_adjudicated_pairs import rows_digest


PAIR_TYPE_NAMES = (
    "dynamic_vs_intent",
    "camera_vs_actor_goal",
    "tracking_vs_prediction",
)


def exact_mcnemar_p_value(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(a_correct_b_wrong, a_wrong_b_correct) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        current = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, current)
        adjusted[name] = running_max
    return adjusted


def cluster_bootstrap_interval(
    records: list[dict[str, Any]], *, seed: int, iterations: int
) -> list[float]:
    by_video: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_video[record["video"]].append(float(record["difference"]))
    videos = sorted(by_video)
    if not videos:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_indices = rng.integers(0, len(videos), size=len(videos))
        differences = [
            difference
            for index in sampled_indices
            for difference in by_video[videos[int(index)]]
        ]
        estimates[iteration] = float(np.mean(differences))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def cluster_sign_flip_p_value(
    records: list[dict[str, Any]], *, seed: int, iterations: int
) -> float:
    by_video: dict[str, float] = defaultdict(float)
    for record in records:
        by_video[record["video"]] += float(record["difference"])
    contributions = np.array([by_video[video] for video in sorted(by_video)])
    observed = abs(float(contributions.sum()))
    if not len(contributions) or observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    batch_size = 10000
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(contributions)))
        statistics = np.abs(signs @ contributions)
        extreme += int(np.count_nonzero(statistics >= observed))
        completed += size
    return (extreme + 1) / (iterations + 1)


def parse_correct(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "correct"}:
        return 1
    if normalized in {"0", "false", "no", "incorrect"}:
        return 0
    raise ValueError(f"Cannot parse correctness value: {value!r}")


def validate_pairs(
    pairs: list[dict[str, str]],
    candidates: list[dict[str, str]] | None = None,
    *,
    max_pairs_per_type: int = 100,
) -> None:
    required = {
        "final_pair_id",
        "pair_type",
        "video",
        "arm_a_index",
        "arm_b_index",
        "question_type_match",
    }
    seen_ids: set[str] = set()
    seen_packets: set[str] = set()
    pair_type_counts: dict[str, int] = defaultdict(int)
    candidates_by_id: dict[str, dict[str, str]] = {}
    if candidates is not None:
        candidates_by_id = {row["candidate_pair_id"]: row for row in candidates}
        if len(candidates_by_id) != len(candidates):
            raise ValueError("Candidate manifest has duplicate candidate_pair_id values")
        required.update(
            {
                "candidate_pair_id",
                "packet_id",
                "pair_sha256",
                "selection_rank_sha256",
            }
        )
    for row in pairs:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Final-pair row missing fields: {sorted(missing)}")
        pair_id = row["final_pair_id"]
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate final_pair_id: {pair_id}")
        seen_ids.add(pair_id)
        if row["pair_type"] not in PAIR_TYPE_NAMES:
            raise ValueError(f"Unknown pair_type: {row['pair_type']}")
        pair_type_counts[row["pair_type"]] += 1
        packet_id = row.get("packet_id", pair_id)
        packet_key = f"{row['pair_type']}\x1f{packet_id}"
        if packet_key in seen_packets:
            raise ValueError(f"More than one final pair for packet: {packet_id}")
        seen_packets.add(packet_key)
        if candidates is not None:
            candidate_id = row["candidate_pair_id"]
            if candidate_id not in candidates_by_id:
                raise ValueError(f"Final pair references unknown candidate: {candidate_id}")
            candidate = candidates_by_id[candidate_id]
            for field in CANDIDATE_FIELDS:
                if row.get(field, "") != candidate.get(field, ""):
                    raise ValueError(
                        f"Final pair {pair_id} changed candidate field {field}"
                    )
            selection_rank = row["selection_rank_sha256"]
            if len(selection_rank) != 64 or any(
                character not in "0123456789abcdef" for character in selection_rank
            ):
                raise ValueError(f"Invalid selection_rank_sha256 for {pair_id}")
    for pair_type, count in pair_type_counts.items():
        if count > max_pairs_per_type:
            raise ValueError(
                f"Pair type {pair_type} exceeds cap: {count}>{max_pairs_per_type}"
            )


def prediction_lookup(
    predictions: list[dict[str, str]],
    *,
    require_provenance: bool = False,
) -> dict[str, dict[int, int]]:
    lookup: dict[str, dict[int, int]] = defaultdict(dict)
    provenance_by_model: dict[str, tuple[str, ...]] = {}
    for row in predictions:
        if require_provenance:
            required_provenance = (
                "run_id",
                "model_artifact",
                "prompt_template_sha256",
                "parse_status",
                "raw_response_sha256",
                "run_manifest_sha256",
                "video_sha256",
                "model_input_sha256",
            )
            missing = [field for field in required_provenance if not row.get(field, "").strip()]
            if missing:
                raise ValueError(
                    "Prediction row is missing provenance fields: " + ", ".join(missing)
                )
            for field in (
                "prompt_template_sha256",
                "raw_response_sha256",
                "run_manifest_sha256",
                "video_sha256",
                "model_input_sha256",
            ):
                value = row[field].strip()
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"Prediction row has invalid {field}")
            if row["parse_status"].strip() != "ok" and parse_correct(row["correct"]) != 0:
                raise ValueError("Prediction parse failures must be scored incorrect")
        model = row["model"].strip()
        if not model:
            raise ValueError("Prediction row requires model")
        if require_provenance:
            signature = tuple(
                row[field].strip()
                for field in (
                    "run_id",
                    "model_artifact",
                    "prompt_template_sha256",
                    "run_manifest_sha256",
                )
            )
            previous = provenance_by_model.setdefault(model, signature)
            if previous != signature:
                raise ValueError(
                    f"{model}: predictions mix run or artifact provenance"
                )
        benchmark_index = int(row["benchmark_index"])
        if benchmark_index in lookup[model]:
            raise ValueError(f"Duplicate prediction for {model} index {benchmark_index}")
        lookup[model][benchmark_index] = parse_correct(row["correct"])
    if not lookup:
        raise ValueError("Prediction CSV contains no model rows")
    return lookup


def analyze_subset(
    pairs: list[dict[str, str]],
    model_predictions: dict[int, int],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    records = []
    missing_indices: set[int] = set()
    for pair in pairs:
        index_a = int(pair["arm_a_index"])
        index_b = int(pair["arm_b_index"])
        if index_a not in model_predictions:
            missing_indices.add(index_a)
        if index_b not in model_predictions:
            missing_indices.add(index_b)
        if index_a not in model_predictions or index_b not in model_predictions:
            continue
        correct_a = model_predictions[index_a]
        correct_b = model_predictions[index_b]
        records.append(
            {
                "video": pair["video"],
                "correct_a": correct_a,
                "correct_b": correct_b,
                "difference": correct_b - correct_a,
            }
        )
    if missing_indices:
        raise ValueError(
            f"Missing predictions for {len(missing_indices)} benchmark indices: "
            f"{sorted(missing_indices)[:10]}"
        )
    if not records:
        return {
            "n_pairs": 0,
            "arm_a_accuracy": None,
            "arm_b_accuracy": None,
            "paired_difference_b_minus_a": None,
            "video_cluster_bootstrap_interval": [None, None],
            "a_correct_b_wrong": 0,
            "a_correct_b_wrong_rate": None,
            "a_correct_count": 0,
            "b_wrong_given_a_correct_rate": None,
            "a_wrong_b_correct": 0,
            "a_wrong_b_correct_rate": None,
            "exact_mcnemar_p": 1.0,
            "video_cluster_sign_flip_p": 1.0,
        }
    accuracy_a = float(np.mean([record["correct_a"] for record in records]))
    accuracy_b = float(np.mean([record["correct_b"] for record in records]))
    a_correct_b_wrong = sum(
        record["correct_a"] == 1 and record["correct_b"] == 0 for record in records
    )
    a_wrong_b_correct = sum(
        record["correct_a"] == 0 and record["correct_b"] == 1 for record in records
    )
    a_correct_count = sum(record["correct_a"] == 1 for record in records)
    return {
        "n_pairs": len(records),
        "arm_a_accuracy": accuracy_a,
        "arm_b_accuracy": accuracy_b,
        "paired_difference_b_minus_a": accuracy_b - accuracy_a,
        "video_cluster_bootstrap_interval": cluster_bootstrap_interval(
            records, seed=seed, iterations=bootstrap_iterations
        ),
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_correct_b_wrong_rate": a_correct_b_wrong / len(records),
        "a_correct_count": a_correct_count,
        "b_wrong_given_a_correct_rate": (
            a_correct_b_wrong / a_correct_count if a_correct_count else None
        ),
        "a_wrong_b_correct": a_wrong_b_correct,
        "a_wrong_b_correct_rate": a_wrong_b_correct / len(records),
        "exact_mcnemar_p": exact_mcnemar_p_value(
            a_correct_b_wrong, a_wrong_b_correct
        ),
        "video_cluster_sign_flip_p": cluster_sign_flip_p_value(
            records, seed=seed + 1_000_000, iterations=bootstrap_iterations
        ),
    }


def analyze(
    pairs: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    seed: int,
    bootstrap_iterations: int,
    candidates: list[dict[str, str]] | None = None,
    max_pairs_per_type: int = 100,
    require_prediction_provenance: bool = False,
) -> dict[str, Any]:
    validate_pairs(
        pairs, candidates, max_pairs_per_type=max_pairs_per_type
    )
    by_model = prediction_lookup(
        predictions, require_provenance=require_prediction_provenance
    )
    required_indices = {
        int(row[field])
        for row in pairs
        for field in ("arm_a_index", "arm_b_index")
    }
    for model, model_predictions in by_model.items():
        observed = set(model_predictions)
        if observed != required_indices:
            raise ValueError(
                f"{model}: scored index set differs from final pairs; "
                f"missing={sorted(required_indices - observed)[:10]}, "
                f"extra={sorted(observed - required_indices)[:10]}"
            )
    results: dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "n_final_pairs": len(pairs),
            "n_unique_videos": len({row["video"] for row in pairs}),
        },
        "models": {},
    }
    for model, model_predictions in sorted(by_model.items()):
        model_result: dict[str, Any] = {"pair_types": {}}
        model_result["multiplicity_family"] = (
            "three_same_question_type_pair_types_within_this_model"
        )
        raw_p_values: dict[str, float] = {}
        for pair_type_index, pair_type in enumerate(PAIR_TYPE_NAMES):
            type_pairs = [row for row in pairs if row["pair_type"] == pair_type]
            all_formats = analyze_subset(
                type_pairs,
                model_predictions,
                seed=seed + pair_type_index,
                bootstrap_iterations=bootstrap_iterations,
            )
            primary = analyze_subset(
                [row for row in type_pairs if row["question_type_match"] == "yes"],
                model_predictions,
                seed=seed + 100 + pair_type_index,
                bootstrap_iterations=bootstrap_iterations,
            )
            model_result["pair_types"][pair_type] = {
                "primary": primary,
                "primary_definition": "same_question_type",
                "all_formats_exploratory": all_formats,
            }
            raw_p_values[pair_type] = primary["exact_mcnemar_p"]
        adjusted = holm_adjust(raw_p_values)
        for pair_type, adjusted_p in adjusted.items():
            model_result["pair_types"][pair_type]["primary"][
                "holm_adjusted_p"
            ] = adjusted_p
        pooled = analyze_subset(
            pairs,
            model_predictions,
            seed=seed + 1000,
            bootstrap_iterations=bootstrap_iterations,
        )
        pooled.pop("exact_mcnemar_p", None)
        pooled["inferential_test"] = (
            "not_reported_because_videos_can_repeat_across_pair_types"
        )
        model_result["pooled_all_formats_exploratory"] = pooled
        results["models"][model] = model_result
    return results


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_markdown(results: dict[str, Any]) -> str:
    def percentage(value: float | None, suffix: str = "%") -> str:
        return "NA" if value is None else f"{100 * value:.1f}{suffix}"

    lines = [
        "# Matched-Pair Results",
        "",
        f"Final pairs: {results['metadata']['n_final_pairs']}; "
        f"unique videos: {results['metadata']['n_unique_videos']}.",
        "",
    ]
    for model, model_result in results["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                "Confirmatory primary: same-question-type pairs only.",
                "",
                "| Pair type | n | Arm A acc. | Arm B acc. | B-A | 95% video-cluster interval | A correct/B wrong | P(B wrong | A correct) | A wrong/B correct | McNemar p | Holm p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pair_type in PAIR_TYPE_NAMES:
            record = model_result["pair_types"][pair_type]["primary"]
            low, high = record["video_cluster_bootstrap_interval"]
            interval = (
                "NA"
                if low is None or high is None
                else f"[{100 * low:.1f}, {100 * high:.1f}]"
            )
            lines.append(
                f"| {pair_type} | {record['n_pairs']} | "
                f"{percentage(record['arm_a_accuracy'])} | "
                f"{percentage(record['arm_b_accuracy'])} | "
                f"{percentage(record['paired_difference_b_minus_a'], ' pp')} | "
                f"{interval} | "
                f"{record['a_correct_b_wrong']} "
                f"({percentage(record['a_correct_b_wrong_rate'])}) | "
                f"{percentage(record['b_wrong_given_a_correct_rate'])} | "
                f"{record['a_wrong_b_correct']} "
                f"({percentage(record['a_wrong_b_correct_rate'])}) | "
                f"{record['exact_mcnemar_p']:.4f} | {record['holm_adjusted_p']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Exploratory: all adjudicated formats.",
                "",
                "| Pair type | n | Arm A acc. | Arm B acc. | B-A | 95% video-cluster interval |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for pair_type in PAIR_TYPE_NAMES:
            record = model_result["pair_types"][pair_type][
                "all_formats_exploratory"
            ]
            low, high = record["video_cluster_bootstrap_interval"]
            interval = (
                "NA"
                if low is None or high is None
                else f"[{100 * low:.1f}, {100 * high:.1f}]"
            )
            lines.append(
                f"| {pair_type} | {record['n_pairs']} | "
                f"{percentage(record['arm_a_accuracy'])} | "
                f"{percentage(record['arm_b_accuracy'])} | "
                f"{percentage(record['paired_difference_b_minus_a'], ' pp')} | "
                f"{interval} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen matched pairs.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--finalization-summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scoring-summary", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--max-pairs-per-type", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_csv(args.pairs)
    candidates = read_csv(args.candidate_pairs)
    freeze_summary = json.loads(args.finalization_summary.read_text(encoding="utf-8"))
    expected_candidate_digest = freeze_summary.get("candidate_manifest_sha256")
    if expected_candidate_digest != args.expected_candidate_sha256:
        raise ValueError("Finalization summary does not match external candidate SHA-256")
    if expected_candidate_digest != file_sha256(args.candidate_pairs):
        raise ValueError("Candidate manifest SHA-256 does not match finalization summary")
    output_fields = freeze_summary.get("output_fields")
    if not isinstance(output_fields, list) or not output_fields:
        raise ValueError("Finalization summary is missing output_fields")
    final_digest = rows_digest(pairs, [str(field) for field in output_fields])
    if final_digest != freeze_summary.get("final_manifest_sha256"):
        raise ValueError("Final-pair manifest digest does not match finalization summary")
    scoring_summary = json.loads(args.scoring_summary.read_text(encoding="utf-8"))
    if scoring_summary.get("schema_version") != "pair-scored-predictions-v1":
        raise ValueError("Unsupported scoring-summary schema")
    scoring_checks = {
        "scored_predictions_sha256": file_sha256(args.predictions),
        "final_pairs_sha256": file_sha256(args.pairs),
        "run_manifest_sha256": file_sha256(args.run_manifest),
    }
    for field, observed in scoring_checks.items():
        if scoring_summary.get(field) != observed:
            raise ValueError(f"Scoring-summary mismatch for {field}")
    run_manifest_sha256 = file_sha256(args.run_manifest)
    predictions = read_csv(args.predictions)
    if any(
        row.get("run_manifest_sha256", "") != run_manifest_sha256
        for row in predictions
    ):
        raise ValueError("Prediction rows do not match the supplied run manifest")
    results = analyze(
        pairs,
        predictions,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap,
        candidates=candidates,
        max_pairs_per_type=args.max_pairs_per_type,
        require_prediction_provenance=True,
    )
    results["metadata"]["candidate_manifest_sha256"] = expected_candidate_digest
    results["metadata"]["final_manifest_sha256"] = final_digest
    results["metadata"]["scored_predictions_sha256"] = scoring_checks[
        "scored_predictions_sha256"
    ]
    results["metadata"]["multiplicity_scope"] = (
        "Holm correction is applied within each model; models are independent "
        "replications and are not pooled into a single cross-model claim."
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(results), encoding="utf-8")
    print(render_markdown(results))


if __name__ == "__main__":
    main()
