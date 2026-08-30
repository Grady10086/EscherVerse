#!/usr/bin/env python3
"""Replace Base max-token rows with 4096-token retries and rescore."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_calculate_statistics(evaluator: Path):
    spec = importlib.util.spec_from_file_location("escher_evaluator", evaluator)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {evaluator}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.base.read_text(encoding="utf-8"))
    original_hits = {
        int(row["index"])
        for row in payload["results"]
        if row.get("generation_metadata", {}).get("hit_max_tokens") is True
    }
    retries = {}
    for path in args.retry:
        shard = json.loads(path.read_text(encoding="utf-8"))
        if not shard["metadata"]["run_integrity"]["complete"]:
            raise ValueError(f"Incomplete retry shard: {path}")
        for row in shard["results"]:
            index = int(row["index"])
            if index in retries:
                raise ValueError(f"Duplicate retry index: {index}")
            retries[index] = row
    if set(retries) != original_hits:
        raise ValueError(f"Retry mismatch: expected {len(original_hits)}, got {len(retries)}")

    unresolved = []
    corrected = []
    for row in payload["results"]:
        index = int(row["index"])
        if index in retries:
            row = retries[index]
            if row.get("generation_metadata", {}).get("hit_max_tokens") is True and row.get("answer_tag_count", 0) == 0:
                row = {
                    **row,
                    "is_correct": False,
                    "score": 0.0,
                    "prediction_clean": "[NO_FINAL_ANSWER]",
                    "eval_method": "max_tokens_without_answer_tag_forced_wrong",
                }
                unresolved.append(index)
        corrected.append(row)

    calculate_statistics = load_calculate_statistics(args.evaluator)
    payload["results"] = corrected
    payload["statistics"] = calculate_statistics(corrected)
    payload["metadata"] = {
        **payload["metadata"],
        "max_token_correction": {
            "initial_limit": 256,
            "retry_limit": 4096,
            "retried_items": len(retries),
            "retry_shards": len(args.retry),
            "remaining_without_final_answer": len(unresolved),
            "remaining_indices": sorted(unresolved),
            "policy": "4096-token retry; hit-max rows without an answer tag are wrong",
        },
        "run_integrity": {
            "input_items": len(corrected),
            "output_items": len(corrected),
            "unique_output_indices": len({int(row["index"]) for row in corrected}),
            "inference_errors": sum(row.get("is_correct") is None for row in corrected),
            "complete": len(corrected) == len({int(row["index"]) for row in corrected}),
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accuracy": payload["statistics"]["accuracy"],
        "retried": len(retries),
        "remaining_without_final_answer": len(unresolved),
    }, indent=2))


if __name__ == "__main__":
    main()
