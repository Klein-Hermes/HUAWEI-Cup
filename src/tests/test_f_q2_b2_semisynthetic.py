from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src import f_q2_b2_semisynthetic as b2


FROZEN_PARAMS = {"E": 1.689797562928721, "A": 0.353980320664689, "alpha": 0.33997658189675806, "B": 1.240305583540225, "beta": 0.27987812854769506}


class CerebrasB2GeneratorTests(unittest.TestCase):
    def test_official_anchor_support_and_compute_optimal_token_ratio(self):
        self.assertEqual(len(b2.CEREBRAS_ANCHORS), 7)
        expected_sizes = [0.111, 0.256, 0.590, 1.3, 2.7, 6.7, 13.0]
        self.assertEqual([anchor.N_params_B for anchor in b2.CEREBRAS_ANCHORS], expected_sizes)
        for anchor in b2.CEREBRAS_ANCHORS:
            self.assertAlmostEqual(anchor.D_end_tokens_B / anchor.N_params_B, 20.0, delta=0.5)

    def test_b1_freeze_is_hash_closed_in_current_project(self):
        evidence = b2.verify_b1_frozen(Path(__file__).resolve().parents[2])
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["b1_fit_scope"], "B1 only")
        self.assertEqual(evidence["b1_gate0_status"], "PASS")

    def test_generation_is_reproducible_and_keeps_original_endpoint_anchors(self):
        left, left_anchors = b2.generate_trajectories(FROZEN_PARAMS, seed=17, checkpoints=15)
        right, right_anchors = b2.generate_trajectories(FROZEN_PARAMS, seed=17, checkpoints=15)
        self.assertTrue(left.equals(right))
        self.assertEqual(len(left), 7 * len(b2.STRESS_SCENARIOS) * 15)
        preserved = left_anchors.loc[~left_anchors["scenario"].str.startswith("family_offset_")]
        self.assertLess(float(preserved["endpoint_difference"].abs().max()), 1e-12)
        shifted = left_anchors.loc[left_anchors["scenario"].str.startswith("family_offset_")]
        self.assertTrue(shifted["terminal_is_calibration_anchor_position"].all())
        self.assertFalse(shifted["terminal_matches_published_anchor"].any())
        self.assertTrue(shifted["terminal_excluded_from_stress_score"].all())
        terminal_rows = left.loc[left["is_terminal_checkpoint"]]
        self.assertEqual(len(terminal_rows), 7 * len(b2.STRESS_SCENARIOS))
        self.assertEqual(int(terminal_rows["matches_published_anchor"].sum()), 7 * (len(b2.STRESS_SCENARIOS) - 2))

    def test_all_paths_use_monotone_positive_exposure_and_finite_losses(self):
        data, _ = b2.generate_trajectories(FROZEN_PARAMS, seed=9, checkpoints=15)
        self.assertTrue(np.isfinite(data["target_loss"]).all())
        self.assertTrue((data["D_tokens_B"] > 0).all())
        self.assertTrue((data["target_loss"] > 0).all())
        for _, group in data.groupby("run_id"):
            self.assertTrue(group["D_tokens_B"].is_monotonic_increasing)
            self.assertAlmostEqual(float(group["tokens_per_parameter"].iloc[-1]), 20.0, delta=0.5)

    def test_stress_scores_exclude_every_calibration_endpoint(self):
        data, _ = b2.generate_trajectories(FROZEN_PARAMS, seed=9, checkpoints=15)
        metrics = b2.score_interior_stress(data)
        self.assertEqual(len(metrics), len(b2.STRESS_SCENARIOS))
        self.assertTrue(metrics["endpoint_used_for_calibration_excluded"].all())
        self.assertEqual(int(data["is_terminal_checkpoint"].sum()), 7 * len(b2.STRESS_SCENARIOS))
        self.assertTrue((metrics["n_interior_points"] == 7 * 14).all())
        self.assertTrue((metrics["n_trajectory_clusters"] == 7).all())

    def test_invalid_exposure_and_invalid_checkpoint_count_fail_closed(self):
        with self.assertRaises(ValueError):
            b2.predict_scaling(FROZEN_PARAMS, 0.0, 10.0)
        with self.assertRaises(ValueError):
            b2.generate_trajectories(FROZEN_PARAMS, checkpoints=2)


if __name__ == "__main__":
    unittest.main()
