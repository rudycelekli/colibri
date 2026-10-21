"""HellaSwag examples must use the reference harness's WikiHow cleanup."""
from pathlib import Path
import runpy
import unittest

FORMAT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "tools" / "fetch_benchmarks.py"))["f_hellaswag"]


class HellaSwagFormatterTests(unittest.TestCase):
    # Hand-derived expected text from the reference task preprocessing:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/utils.py
    def row(self, **changes):
        row = dict(activity_label="Making tea", ctx_a="Gather a cup",
                   ctx_b="pour water", endings=["Stir the tea.", "Wait briefly."], label="1")
        row.update(changes)
        return row

    def test_wikihow_context_title_and_brackets_are_preprocessed(self):
        row = self.row(ctx_a="Gather a cup [title] Add tea [instruction]",
                       ctx_b="pour  water")
        context, _, gold = FORMAT(row)
        self.assertEqual(context, "Making tea: Gather a cup. Add tea Pour water")
        self.assertEqual(gold, 1)

    def test_each_ending_uses_reference_cleanup_before_separator(self):
        _, choices, _ = FORMAT(self.row(endings=[
            "Stir [instruction] the  tea.", "Wait [title] Serve it."]))
        self.assertEqual(choices, [" Stir the tea.", " Wait. Serve it."])

    def test_plain_examples_keep_their_context_choices_and_label(self):
        self.assertEqual(FORMAT(self.row()),
                         ("Making tea: Gather a cup Pour water",
                          [" Stir the tea.", " Wait briefly."], 1))


if __name__ == "__main__":
    unittest.main()
