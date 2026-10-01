import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.generate_report import STRATA, auto_selection, choose_cases


ROOT = Path(__file__).resolve().parents[1]


class StaticReportSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))["cases"]

    def test_manifest_images_exist(self):
        self.assertEqual(len(self.cases), 16)
        for case in self.cases:
            self.assertTrue((ROOT / "evaluation" / case["image"]).is_file(), case["id"])

    def test_one_case_is_selected_from_each_stratum(self):
        selected = choose_cases(self.cases, seed=42)
        self.assertEqual([case["stratum"] for case in selected], STRATA)
        self.assertEqual(len({case["id"] for case in selected}), 4)

    def test_automatic_selection_avoids_previous_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            first = choose_cases(self.cases, seed=1)
            result_dir = outputs / "previous"
            result_dir.mkdir()
            (result_dir / "results.json").write_text(
                json.dumps({"cases": [{"id": case["id"]} for case in first]}), encoding="utf-8"
            )
            with patch("evaluation.generate_report.secrets.randbits", side_effect=[1, 2]):
                seed, selected = auto_selection(self.cases, outputs)
            self.assertEqual(seed, 2)
            self.assertNotEqual([case["id"] for case in selected], [case["id"] for case in first])


if __name__ == "__main__":
    unittest.main()
