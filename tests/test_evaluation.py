"""CPU checks for leakage controls, scoring, paired reports, and safe resume."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation.report import donor_map, guard_run, parse_choice, report, select_rows

try:
    import torch
except ImportError:
    torch = None


class EvaluationTests(unittest.TestCase):
    def test_scoring_does_not_guess_from_prose(self):
        for text in ("C. hail", "c) hail", "Answer: C", "The answer is (C)."):
            self.assertEqual(parse_choice(text), "C")
        for text in ("", "I hear a dog", "hail", "E. none"):
            self.assertIsNone(parse_choice(text))

    def test_selection_rejects_missing_cases_and_duplicate_names(self):
        row = dict(name="avqa-1", task="avqa", reference="A. road", video="v1.mp4")
        with self.assertRaises(ValueError):
            select_rows(dict(cases=[row]), ["missing"])
        with self.assertRaises(ValueError):
            select_rows(dict(cases=[row, row]))
        with self.assertRaises(ValueError):
            select_rows(dict(cases=[dict(row, name="../escape")]))

    def test_donors_exclude_same_source_and_need_distinct_videos(self):
        rows = [dict(name="a", video="a.mp4", source_video_id="same"),
                dict(name="b", video="b.mp4", source_video_id="same"),
                dict(name="c", video="c.mp4", source_video_id="different")]
        self.assertEqual(donor_map(rows), {"a": "c", "b": "c", "c": "a"})
        with self.assertRaises(ValueError):
            donor_map(rows[:2])

    def test_resume_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            guard_run(folder, {"seed": 1})
            guard_run(folder, {"seed": 1})
            with self.assertRaises(ValueError):
                guard_run(folder, {"seed": 2})

    def test_dry_run_plans_external_inputs_without_model_dependencies_or_output(self):
        from scripts.evaluation.run import main

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "finetuned").mkdir()
            (root / "finetuned/latest.json").write_text('{"step":9000}')
            (root / "inference-config.json").write_text('{"seed":1337}')
            rows = []
            for name in ("a", "b"):
                (root / f"{name}.safetensors").write_bytes(b"planning-only")
                rows.append(dict(name=name, task="avqa", reference="C. hail",
                                 input=f"{name}.safetensors", video=f"{name}.mp4"))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=rows)))
            output = root / "output"
            with patch("sys.argv", ["evaluate", "--bundle", str(root), "--manifest",
                                    str(manifest), "--data-root", str(root), "--output",
                                    str(output), "--dry-run"]), contextlib.redirect_stdout(io.StringIO()) as stream:
                main()
            self.assertEqual(json.loads(stream.getvalue())["rollouts"], 6)
            self.assertFalse(output.exists())

    def test_reports_pair_only_completed_cases_and_count_invalid_as_wrong(self):
        def result(name, condition, text):
            return dict(name=f"{name}--{condition}",
                        evaluation=dict(case_id=name, condition=condition, donor_id=None),
                        reference_text="A. road", generated_text=text,
                        audio_seconds=10, wall_seconds=40, visual_tokens=12, events=[])
        with tempfile.TemporaryDirectory() as folder:
            summary = report(folder, [result("a", "correct", "A. road"),
                                      result("a", "padding", "unclear"),
                                      result("b", "correct", "B. street")], 6)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["conditions"]["correct"]["accuracy"], 0.5)
            self.assertEqual(summary["conditions"]["padding"]["invalid"], 1)
            self.assertEqual(summary["paired"]["padding"]["completed_pairs"], 1)
            self.assertEqual(summary["paired"]["padding"]["accuracy_difference"], 1)
            self.assertEqual(json.loads((Path(folder) / "evaluation.json").read_text())["completed"], 3)


@unittest.skipIf(torch is None, "PyTorch is required for tensor ablation checks")
class VisualAblationTests(unittest.TestCase):
    def test_padding_and_mismatch_remove_all_target_deepstack_information(self):
        from scripts.evaluation.inputs import ablate_visual

        target = dict(vision_mask=torch.tensor([[True, False, True, True]]),
                      vision_features=torch.full((1, 4, 2), 7.0),
                      vision_deepstack=torch.full((3, 1, 4, 2), 9.0))
        donor = dict(vision_mask=torch.tensor([[True, True, False]]),
                     vision_features=torch.full((1, 3, 2), 2.0),
                     vision_deepstack=torch.full((3, 1, 3, 2), 3.0))
        padded = ablate_visual(target, "padding")
        self.assertTrue(all(not bool(value.any()) for value in padded.values()))
        swapped = ablate_visual(target, "mismatched", donor)
        self.assertEqual(swapped["vision_mask"].tolist(), [[True, False, False, False]])
        self.assertEqual(swapped["vision_features"][0, :, 0].tolist(), [2, 0, 0, 0])
        self.assertEqual(swapped["vision_deepstack"][0, 0, :, 0].tolist(), [3, 0, 0, 0])
        self.assertEqual(float(target["vision_features"][0, 0, 0]), 7)
        correct = ablate_visual(target, "correct")
        torch.testing.assert_close(correct["vision_features"], target["vision_features"])


if __name__ == "__main__":
    unittest.main()
