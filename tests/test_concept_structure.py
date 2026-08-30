from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "experiments" / "concept_structure" / "scripts"
DATA_PATH = (
    ROOT
    / "experiments"
    / "concept_structure"
    / "data"
    / "model_category_scores.csv"
)
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_concept_structure import (  # noqa: E402
    analyze,
    cohort_rows,
    leave_one_family_out_pca,
    pair_group_scores,
    load_scores,
    model_pair_group_records,
    named_model_family,
    pca,
    score_matrix,
    spearman_matrix,
)


class ConceptStructureTest(unittest.TestCase):
    def test_locked_table_has_expected_panel_and_anchor_values(self):
        rows = load_scores(DATA_PATH)
        self.assertEqual(len(rows), 27)
        by_model = {row["model"]: row for row in rows}
        self.assertEqual(by_model["Gemini-2.5-pro"]["overall"], 57.26)
        self.assertEqual(by_model["Escher-8B-Instruct"]["c3_action_intent"], 63.60)
        self.assertEqual(by_model["Qwen3-VL-32B-Thinking"]["c2_dynamic_spatial"], 49.80)

    def test_cohorts_match_frozen_sizes(self):
        cohorts = cohort_rows(load_scores(DATA_PATH))
        self.assertEqual(
            {name: len(rows) for name, rows in cohorts.items()},
            {
                "all_27": 27,
                "non_sft_21": 21,
                "overall_ge_20_24": 24,
                "non_sft_overall_ge_20_18": 18,
            },
        )

    def test_spearman_is_symmetric_with_unit_diagonal(self):
        correlation = spearman_matrix(score_matrix(load_scores(DATA_PATH)))
        np.testing.assert_allclose(correlation, correlation.T)
        np.testing.assert_allclose(np.diag(correlation), np.ones(6))

    def test_pair_groups_reproduce_submitted_figure_pairs(self):
        values = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        groups = pair_group_scores(values)
        np.testing.assert_allclose(groups, [[3.0, 3.0, 4.5]])

    def test_full_panel_pair_group_table_preserves_models_and_overall(self):
        rows = load_scores(DATA_PATH)
        records = model_pair_group_records(rows)
        self.assertEqual(len(records), 27)
        by_model = {record["model"]: record for record in records}
        self.assertAlmostEqual(
            by_model["Gemini-2.5-pro"]["group_1_c1_c5_equal_category_weight"],
            (56.17 + 73.11) / 2,
        )
        self.assertEqual(by_model["Gemini-2.5-pro"]["overall"], 57.26)

    def test_pca_detects_one_common_synthetic_dimension(self):
        base = np.linspace(-2, 2, 30)
        values = np.column_stack(
            [base * scale + offset for scale, offset in zip(range(1, 7), range(6))]
        )
        result = pca(values)
        self.assertGreater(result["explained_ratio"][0], 0.999)
        self.assertTrue(np.all(result["loadings"][:, 0] > 0))

    def test_named_family_diagnostic_covers_full_panel(self):
        rows = load_scores(DATA_PATH)
        self.assertEqual(named_model_family("Qwen3-VL-8B-Instruct"), "Qwen")
        self.assertEqual(named_model_family("Escher-8B-Instruct"), "Escher")
        diagnostics = leave_one_family_out_pca(rows)
        self.assertEqual(len(diagnostics), 11)
        self.assertTrue(
            all(0.0 < record["pc1_explained_ratio"] <= 1.0 for record in diagnostics)
        )

    def test_analysis_is_deterministic(self):
        rows = load_scores(DATA_PATH)
        first = analyze(
            rows,
            seed=17,
            bootstrap_iterations=25,
            parallel_iterations=25,
        )
        second = analyze(
            rows,
            seed=17,
            bootstrap_iterations=25,
            parallel_iterations=25,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
