#!/usr/bin/env python3
"""Analyze capability-specific SFT effects with video-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CAPABILITIES = ("intent", "dynamic_non_intent")


def load_result(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    integrity = payload.get("metadata", {}).get("run_integrity", {})
    if not integrity.get("complete"):
        raise ValueError(f"Incomplete evaluation: {path}: {integrity}")
    return {int(row["index"]): row for row in payload["results"]}


def mean_by_capability(rows: dict[int, dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        grouped[str(row["target_capability"])].append(float(row["score"]))
    return {key: float(np.mean(grouped[key])) for key in CAPABILITIES}


def bootstrap_did(
    intent_runs: list[dict[int, dict[str, object]]],
    dynamic_runs: list[dict[int, dict[str, object]]],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    common = set.intersection(*(set(run) for run in intent_runs + dynamic_runs))
    template = intent_runs[0]
    by_cap_video: dict[str, dict[str, list[int]]] = {
        capability: defaultdict(list) for capability in CAPABILITIES
    }
    for index in common:
        row = template[index]
        by_cap_video[str(row["target_capability"])][str(row["video"])].append(index)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        means: dict[tuple[str, str], float] = {}
        for capability in CAPABILITIES:
            videos = list(by_cap_video[capability])
            sampled = rng.choice(videos, size=len(videos), replace=True)
            indices = [index for video in sampled for index in by_cap_video[capability][video]]
            for condition, runs in [("intent", intent_runs), ("dynamic", dynamic_runs)]:
                scores = [np.mean([float(run[index]["score"]) for run in runs]) for index in indices]
                means[(condition, capability)] = float(np.mean(scores))
        did = (
            means[("intent", "intent")] - means[("dynamic", "intent")]
            - means[("intent", "dynamic_non_intent")] + means[("dynamic", "dynamic_non_intent")]
        )
        draws.append(did)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "iterations": iterations,
        "cluster": "video within capability",
        "estimate": float(np.mean(draws)),
        "ci95": [float(low), float(high)],
        "p_two_sided_bootstrap": float(2 * min(np.mean(np.array(draws) <= 0), np.mean(np.array(draws) >= 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--intent", type=Path, nargs="+", required=True)
    parser.add_argument("--dynamic", type=Path, nargs="+", required=True)
    parser.add_argument("--random", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    loaded = {
        "base": [load_result(args.base)],
        "intent": [load_result(path) for path in args.intent],
        "dynamic": [load_result(path) for path in args.dynamic],
        "random": [load_result(path) for path in args.random],
    }
    index_sets = [set(run) for runs in loaded.values() for run in runs]
    if len({frozenset(indices) for indices in index_sets}) != 1:
        raise ValueError("Evaluation runs do not contain identical item indices")

    per_run = {
        condition: [mean_by_capability(run) for run in runs]
        for condition, runs in loaded.items()
    }
    aggregate = {}
    base = per_run["base"][0]
    for condition, runs in per_run.items():
        aggregate[condition] = {}
        for capability in CAPABILITIES:
            values = [run[capability] for run in runs]
            aggregate[condition][capability] = {
                "mean_accuracy": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "gain_over_base": float(np.mean(values) - base[capability]),
            }
        aggregate[condition]["intent_minus_dynamic"] = float(
            np.mean([run["intent"] - run["dynamic_non_intent"] for run in runs])
        )

    did_point = (
        aggregate["intent"]["intent"]["mean_accuracy"]
        - aggregate["dynamic"]["intent"]["mean_accuracy"]
        - aggregate["intent"]["dynamic_non_intent"]["mean_accuracy"]
        + aggregate["dynamic"]["dynamic_non_intent"]["mean_accuracy"]
    )
    result = {
        "schema": "escher-sft-ablation-analysis-v1",
        "conditions": {key: len(value) for key, value in loaded.items()},
        "items": len(next(iter(index_sets))),
        "per_run_accuracy": per_run,
        "aggregate": aggregate,
        "difference_in_differences": {
            "definition": "(IntentSFT_intent - DynamicSFT_intent) - (IntentSFT_dynamic - DynamicSFT_dynamic)",
            "point_estimate": did_point,
            "bootstrap": bootstrap_did(loaded["intent"], loaded["dynamic"], args.bootstrap, args.seed),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
