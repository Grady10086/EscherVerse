from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproducibilityReleaseTests(unittest.TestCase):
    def assert_hash_manifest(self, manifest: Path) -> None:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = (manifest.parent / relative).resolve()
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, expected, str(path))

    def test_model_registry_has_all_reported_rows(self) -> None:
        path = ROOT / "reproducibility/models/model_registry.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({row["display_name"] for row in rows}), 27)
        self.assertEqual(
            {row["reproduction_route"] for row in rows},
            {"fresh inference", "configuration reference", "aggregate score only"},
        )

    def test_video_source_lookup(self) -> None:
        module = load_module(
            "reconstruct_clips",
            ROOT / "reproducibility/video/reconstruct_clips.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "example-id.mp4"
            source.write_bytes(b"placeholder")
            self.assertEqual(module.locate_source(Path(directory), "example-id"), source)
            self.assertIsNone(module.locate_source(Path(directory), "missing"))

    def test_human_aggregation_contract(self) -> None:
        script = ROOT / "reproducibility/human_baseline/aggregate.py"
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.json"
            input_path.write_text(
                "annotator_id,is_correct\na,1\na,0\nb,true\nb,correct\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--expected-annotators",
                    "2",
                    "--expected-items-per-annotator",
                    "2",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["annotators"], 2)
            self.assertAlmostEqual(summary["mean_accuracy"], 0.75)

    def test_frozen_artifact_hashes(self) -> None:
        self.assert_hash_manifest(
            ROOT
            / "experiments/coverage_audit/results/SCOPE_AUDIT_ARTIFACT_SHA256.txt"
        )
        self.assert_hash_manifest(
            ROOT
            / "experiments/sft_ablation/results/ARTIFACT_SHA256.txt"
        )


if __name__ == "__main__":
    unittest.main()
