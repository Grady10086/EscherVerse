import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "eval" / "evaluate.py"
spec = importlib.util.spec_from_file_location("benchmark_evaluate", SCRIPT_PATH)
benchmark_evaluate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark_evaluate
spec.loader.exec_module(benchmark_evaluate)


class EvalAnswerParserTests(unittest.TestCase):
    def test_multiple_answer_tags_are_combined_before_parsing(self):
        prediction = benchmark_evaluate.extract_answer_from_tags(
            "<answer>A</answer><answer>C</answer>"
        )
        self.assertEqual(prediction, "A\nC")
        self.assertEqual(
            benchmark_evaluate.extract_option_letters(
                prediction, "Multiple-Select"
            ),
            "A,C",
        )
        self.assertEqual(
            benchmark_evaluate.extract_option_letters(prediction, "Single-Choice"),
            "A,C",
        )

    def test_multiple_select_does_not_parse_letters_inside_words(self):
        response = (
            "A) The vertical distance decreases. "
            "C) The horizontal distance decreases."
        )
        self.assertEqual(
            benchmark_evaluate.extract_option_letters(response, "Multiple-Select"),
            "A,C",
        )

    def test_multiple_select_accepts_common_explicit_formats(self):
        for response in ["A, C", "A and C", "A.C", "Options are A and C"]:
            with self.subTest(response=response):
                self.assertEqual(
                    benchmark_evaluate.extract_option_letters(
                        response, "Multiple-Select"
                    ),
                    "A,C",
                )

    def test_single_choice_rejects_ambiguous_multiple_labels(self):
        self.assertEqual(
            benchmark_evaluate.extract_option_letters(
                "A) first possibility; B) second possibility", "Single-Choice"
            ),
            "A,B",
        )

    def test_single_choice_accepts_answer_phrase(self):
        self.assertEqual(
            benchmark_evaluate.extract_option_letters(
                "The answer is B because the ridge is centered.", "Single-Choice"
            ),
            "B",
        )

    def test_free_text_gold_is_not_coerced_to_option_letter(self):
        self.assertEqual(
            benchmark_evaluate.extract_option_letters("blue", "Single-Choice"),
            "blue",
        )


if __name__ == "__main__":
    unittest.main()
