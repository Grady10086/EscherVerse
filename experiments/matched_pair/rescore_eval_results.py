#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = REPO_ROOT / "eval" / "evaluate.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "escher_benchmark_evaluate", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Expected a result object with results[]: {path}")
    return payload


def rescore_file(
    path: Path,
    output_dir: Path,
    evaluator: Any,
    *,
    force_hit_max_incorrect: bool = False,
) -> tuple[Path, dict[str, Any]]:
    payload = load_result(path)
    rescored_rows = []
    changed = []
    for row in payload["results"]:
        prediction = str(row.get("model_prediction", ""))
        if "raw_response" in row:
            prediction = evaluator.extract_answer_from_tags(
                str(row.get("raw_response", ""))
            )
        rescored = evaluator.evaluate_answer(
            prediction,
            str(row.get("ground_truth", "")),
            str(row.get("question_type", "Unknown")),
        )
        if force_hit_max_incorrect and bool(
            (row.get("generation_metadata") or {}).get("hit_max_tokens")
        ):
            rescored["is_correct"] = False
            rescored["score"] = 0.0
            rescored["eval_method"] = "forced_incorrect_max_tokens_sensitivity"
        previous = {
            "model_prediction": row.get("model_prediction"),
            "is_correct": row.get("is_correct"),
            "score": row.get("score"),
            "prediction_clean": row.get("prediction_clean"),
            "ground_truth_clean": row.get("ground_truth_clean"),
            "eval_method": row.get("eval_method"),
        }
        updated = dict(row)
        updated["model_prediction"] = prediction
        updated.update(rescored)
        rescored_rows.append(updated)
        if any(previous.get(key) != updated.get(key) for key in previous):
            changed.append(
                {
                    "index": row.get("index"),
                    "before": previous,
                    "after": {key: updated.get(key) for key in previous},
                }
            )

    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "rescored_at": datetime.now(timezone.utc).isoformat(),
            "rescore_protocol": "anchored-option-label-v2",
            "rescore_evaluator_path": str(EVALUATOR_PATH),
            "rescore_evaluator_sha256": sha256(EVALUATOR_PATH),
            "source_result_path": str(path.resolve()),
            "source_result_sha256": sha256(path),
            "changed_row_count": len(changed),
            "force_hit_max_incorrect": force_hit_max_incorrect,
        }
    )
    output_payload = {
        **payload,
        "metadata": metadata,
        "statistics": evaluator.calculate_statistics(rescored_rows),
        "results": rescored_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.parent.name}_{path.stem}_rescored.json"
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    record = {
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "row_count": len(rescored_rows),
        "changed_rows": changed,
    }
    return output_path, record


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = load_evaluator()
    records = []
    seen_indices = set()
    for path in args.input:
        output_path, record = rescore_file(
            path,
            args.output_dir,
            evaluator,
            force_hit_max_incorrect=args.force_hit_max_incorrect,
        )
        rows = load_result(output_path)["results"]
        indices = [row.get("index") for row in rows]
        overlap = seen_indices.intersection(indices)
        if overlap:
            raise ValueError(f"Duplicate indices across inputs: {sorted(overlap)[:10]}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indices within input: {path}")
        seen_indices.update(indices)
        records.append(record)

    manifest = {
        "schema_version": "eval-rescore-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "anchored-option-label-v2",
        "force_hit_max_incorrect": args.force_hit_max_incorrect,
        "evaluator_sha256": sha256(EVALUATOR_PATH),
        "source_file_count": len(args.input),
        "unique_index_count": len(seen_indices),
        "changed_row_count": sum(len(record["changed_rows"]) for record in records),
        "files": records,
    }
    manifest_path = args.output_dir / "rescore_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore evaluation outputs without modifying source files."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-hit-max-incorrect", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    manifest = run(parse_args())
    print(
        f"Rescored {manifest['unique_index_count']} unique rows; "
        f"changed {manifest['changed_row_count']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
