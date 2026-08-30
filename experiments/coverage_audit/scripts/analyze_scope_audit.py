#!/usr/bin/env python3
"""Analyze dual-rater six-dimension scope coverage with video clustering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = {
    "directly_in_scope",
    "supporting_operation",
    "outside_scope",
    "ambiguous",
}
DIMENSIONS = {
    "C1_occlusion_permanence",
    "C2_dynamic_spatial",
    "C3_action_intent",
    "C4_predictive_counterfactual",
    "C5_deformation_state",
    "C6_reference_frame",
}
COVERED = {"directly_in_scope", "supporting_operation"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(
    packet: list[dict[str, str]],
    mapping: list[dict[str, str]],
    raters: dict[str, list[dict[str, str]]],
) -> None:
    expected = [row["scope_item_id"] for row in packet]
    if [row["scope_item_id"] for row in mapping] != expected:
        raise ValueError("Private mapping does not match the blinded packet")
    for name, rows in raters.items():
        if [row["scope_item_id"] for row in rows] != expected:
            raise ValueError(f"{name} does not match the blinded packet")
        for row in rows:
            status = row["scope_status"]
            primary = row["primary_dimension"]
            secondary = set(filter(None, row["secondary_dimensions"].split("|")))
            if status not in STATUSES:
                raise ValueError(f"Invalid status in {name}: {status}")
            if primary not in DIMENSIONS | {"NONE"}:
                raise ValueError(f"Invalid primary dimension in {name}: {primary}")
            if (status in {"outside_scope", "ambiguous"}) != (primary == "NONE"):
                raise ValueError(f"Status/dimension mismatch in {name}: {row}")
            if not secondary <= DIMENSIONS - {primary}:
                raise ValueError(f"Invalid secondary dimension in {name}: {row}")


def nominal_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[label] * right[label] for label in labels) / len(pairs) ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def rate(rows: list[dict[str, str]], accepted: set[str]) -> tuple[float, int, int]:
    eligible = [row for row in rows if row["scope_status"] != "ambiguous"]
    numerator = sum(row["scope_status"] in accepted for row in eligible)
    return numerator / len(eligible), numerator, len(eligible)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def cluster_bootstrap(
    rows: list[dict[str, str]],
    id_to_video: dict[str, str],
    accepted: set[str],
    iterations: int,
    seed: int,
) -> list[float]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[id_to_video[row["scope_item_id"]]].append(row)
    videos = sorted(by_video)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(videos) for _ in videos]
        draw = [row for video in sampled for row in by_video[video]]
        estimates.append(rate(draw, accepted)[0])
    return estimates


def cluster_bootstrap_binary(
    flags: dict[str, bool],
    id_to_video: dict[str, str],
    iterations: int,
    seed: int,
) -> list[float]:
    by_video: dict[str, list[str]] = defaultdict(list)
    for item in flags:
        by_video[id_to_video[item]].append(item)
    videos = sorted(by_video)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(videos) for _ in videos]
        draw = [item for video in sampled for item in by_video[video]]
        estimates.append(sum(flags[item] for item in draw) / len(draw))
    return estimates


def summarize_rater(
    rows: list[dict[str, str]],
    id_to_video: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    direct, direct_n, denominator = rate(rows, {"directly_in_scope"})
    absorbed, absorbed_n, _ = rate(rows, COVERED)
    direct_draws = cluster_bootstrap(
        rows, id_to_video, {"directly_in_scope"}, iterations, seed
    )
    absorbed_draws = cluster_bootstrap(rows, id_to_video, COVERED, iterations, seed + 1)
    primary = Counter(row["primary_dimension"] for row in rows)
    incidence = Counter()
    for row in rows:
        if row["primary_dimension"] != "NONE":
            incidence[row["primary_dimension"]] += 1
        incidence.update(filter(None, row["secondary_dimensions"].split("|")))
    return {
        "status_counts": dict(sorted(Counter(row["scope_status"] for row in rows).items())),
        "direct": {
            "rate": direct,
            "numerator": direct_n,
            "denominator": denominator,
            "video_cluster_bootstrap_95ci": [
                percentile(direct_draws, 0.025),
                percentile(direct_draws, 0.975),
            ],
        },
        "absorbed": {
            "definition": "directly_in_scope + supporting_operation",
            "rate": absorbed,
            "numerator": absorbed_n,
            "denominator": denominator,
            "video_cluster_bootstrap_95ci": [
                percentile(absorbed_draws, 0.025),
                percentile(absorbed_draws, 0.975),
            ],
        },
        "primary_dimension_counts": dict(sorted(primary.items())),
        "multilabel_dimension_incidence": dict(sorted(incidence.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    packet = read_csv(args.packet)
    mapping = read_csv(args.mapping)
    raters = {"rater_1": read_csv(args.rater_1), "rater_2": read_csv(args.rater_2)}
    validate(packet, mapping, raters)
    id_to_video = {row["scope_item_id"]: row["audit_item_id"] for row in mapping}
    id_to_author = {row["scope_item_id"]: row["author_id"] for row in mapping}
    lookups = {
        name: {row["scope_item_id"]: row for row in rows}
        for name, rows in raters.items()
    }

    status_pairs = [
        (lookups["rater_1"][item]["scope_status"], lookups["rater_2"][item]["scope_status"])
        for item in id_to_video
    ]
    absorbed_pairs = [
        (str(a in COVERED), str(b in COVERED)) for a, b in status_pairs
    ]
    primary_pairs = [
        (
            lookups["rater_1"][item]["primary_dimension"],
            lookups["rater_2"][item]["primary_dimension"],
        )
        for item in id_to_video
    ]
    eligible_pair_ids = [
        item for item in id_to_video
        if lookups["rater_1"][item]["scope_status"] != "ambiguous"
        and lookups["rater_2"][item]["scope_status"] != "ambiguous"
    ]
    consensus_direct = sum(
        all(lookups[name][item]["scope_status"] == "directly_in_scope" for name in lookups)
        for item in eligible_pair_ids
    )
    consensus_absorbed = sum(
        all(lookups[name][item]["scope_status"] in COVERED for name in lookups)
        for item in eligible_pair_ids
    )
    union_direct = sum(
        any(lookups[name][item]["scope_status"] == "directly_in_scope" for name in lookups)
        for item in eligible_pair_ids
    )
    union_absorbed = sum(
        any(lookups[name][item]["scope_status"] in COVERED for name in lookups)
        for item in eligible_pair_ids
    )
    consensus_direct_flags = {
        item: all(
            lookups[name][item]["scope_status"] == "directly_in_scope"
            for name in lookups
        )
        for item in eligible_pair_ids
    }
    consensus_absorbed_flags = {
        item: all(
            lookups[name][item]["scope_status"] in COVERED for name in lookups
        )
        for item in eligible_pair_ids
    }
    union_direct_flags = {
        item: any(
            lookups[name][item]["scope_status"] == "directly_in_scope"
            for name in lookups
        )
        for item in eligible_pair_ids
    }
    union_absorbed_flags = {
        item: any(
            lookups[name][item]["scope_status"] in COVERED for name in lookups
        )
        for item in eligible_pair_ids
    }
    consensus_direct_draws = cluster_bootstrap_binary(
        consensus_direct_flags, id_to_video, args.bootstrap, args.seed + 20
    )
    consensus_absorbed_draws = cluster_bootstrap_binary(
        consensus_absorbed_flags, id_to_video, args.bootstrap, args.seed + 21
    )
    union_direct_draws = cluster_bootstrap_binary(
        union_direct_flags, id_to_video, args.bootstrap, args.seed + 22
    )
    union_absorbed_draws = cluster_bootstrap_binary(
        union_absorbed_flags, id_to_video, args.bootstrap, args.seed + 23
    )
    transition_counts = Counter(status_pairs)

    by_author = {}
    for author in sorted(set(id_to_author.values())):
        ids = {item for item, value in id_to_author.items() if value == author}
        by_author[author] = {
            name: summarize_rater(
                [row for row in rows if row["scope_item_id"] in ids],
                id_to_video,
                args.bootstrap,
                args.seed + offset,
            )
            for offset, (name, rows) in enumerate(raters.items())
        }

    report = {
        "schema": "escher-audit-six-dimension-scope-audit-v1",
        "items": len(packet),
        "videos": len(set(id_to_video.values())),
        "rater_results": {
            name: summarize_rater(rows, id_to_video, args.bootstrap, args.seed + offset)
            for offset, (name, rows) in enumerate(raters.items())
        },
        "agreement": {
            "exact_scope_status": sum(a == b for a, b in status_pairs) / len(status_pairs),
            "scope_status_cohen_kappa": nominal_kappa(status_pairs),
            "binary_absorbed_agreement": sum(a == b for a, b in absorbed_pairs) / len(absorbed_pairs),
            "binary_absorbed_cohen_kappa": nominal_kappa(absorbed_pairs),
            "exact_primary_dimension": sum(a == b for a, b in primary_pairs) / len(primary_pairs),
            "primary_dimension_cohen_kappa": nominal_kappa(primary_pairs),
            "scope_status_disagreements": sum(a != b for a, b in status_pairs),
            "binary_absorbed_disagreements": sum(a != b for a, b in absorbed_pairs),
            "scope_status_transition_counts": {
                f"{left} -> {right}": count
                for (left, right), count in sorted(transition_counts.items())
            },
        },
        "dual_rater_sensitivity": {
            "pairwise_nonambiguous_denominator": len(eligible_pair_ids),
            "consensus_direct_rate": consensus_direct / len(eligible_pair_ids),
            "consensus_direct_video_cluster_bootstrap_95ci": [
                percentile(consensus_direct_draws, 0.025),
                percentile(consensus_direct_draws, 0.975),
            ],
            "at_least_one_rater_direct_rate": union_direct / len(eligible_pair_ids),
            "at_least_one_rater_direct_video_cluster_bootstrap_95ci": [
                percentile(union_direct_draws, 0.025),
                percentile(union_direct_draws, 0.975),
            ],
            "consensus_absorbed_rate": consensus_absorbed / len(eligible_pair_ids),
            "consensus_absorbed_video_cluster_bootstrap_95ci": [
                percentile(consensus_absorbed_draws, 0.025),
                percentile(consensus_absorbed_draws, 0.975),
            ],
            "at_least_one_rater_absorbed_rate": union_absorbed / len(eligible_pair_ids),
            "at_least_one_rater_absorbed_video_cluster_bootstrap_95ci": [
                percentile(union_absorbed_draws, 0.025),
                percentile(union_absorbed_draws, 0.975),
            ],
            "consensus_counts": {
                "both_direct": consensus_direct,
                "both_absorbed": consensus_absorbed,
                "both_outside_scope": sum(
                    all(lookups[name][item]["scope_status"] == "outside_scope" for name in lookups)
                    for item in eligible_pair_ids
                ),
                "binary_coverage_contested": sum(a != b for a, b in absorbed_pairs),
            },
        },
        "results_by_independent_author": by_author,
        "bootstrap": {
            "iterations": args.bootstrap,
            "cluster": "source video",
            "seed": args.seed,
        },
        "limitations": [
            "Both independent question authors and both scope raters are reviewer-assisted, not human-only raters.",
            "The audit evaluates whether the six dimensions can organize questions, not whether the final benchmark realizes every concrete question.",
            "Supporting operations are reported separately and are not direct six-dimension coverage.",
            "The sample contains 120 videos and cannot establish exhaustive long-tail coverage.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
