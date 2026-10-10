"""Tests for Section 17: Milestones and Completion Gates (M0 - M7)."""

from __future__ import annotations

from pathlib import Path
import unittest

from app.eval.milestone_gates import verify_all_milestones


class TestMilestoneGates(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_all_milestones_pass_completion_gates(self):
        """All milestones M0 through M7 must satisfy their defined exit gates."""
        results = verify_all_milestones(self.repo_root)
        self.assertEqual(len(results), 8)

        for m in results:
            self.assertTrue(
                m.passed,
                f"Milestone {m.milestone_id} ({m.name}) exit gate not satisfied: {m.exit_gate_description}",
            )


if __name__ == "__main__":
    unittest.main()
