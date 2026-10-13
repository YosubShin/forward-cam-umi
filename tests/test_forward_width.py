"""Metric scale, perspective/depth, missing-frame and hold-leak regressions."""
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import forward_width as W
import forward_width_check as C
from labels_lib import marker_obj

K = np.array([[1900., 0, 1920], [0, 1900, 1080], [0, 0, 1]])
DIST = np.array([-.12, .04, .002, -.001, .005])


def model():
    return W.PairModel(44, 46, 12.18, 12.14, 10., 64.6,
                       np.array([.023, .006, -.003]),
                       np.array([.030, .001, .0005]), np.array([.05, .04, .01]))


def observation(gap, rng=None, n=0):
    m = model()
    rv = np.array([.25 + .18*np.sin(n), -.2 + .15*np.cos(n*.7), .06])
    tv = np.array([.02*np.sin(n*.2), .01*np.cos(n), .35 + .04*np.sin(n*.3)])
    points = np.r_[marker_obj(m.base_edge_mm),
                   marker_obj(m.tail_edge_mm) @ Rotation.from_rotvec(m.rotation).as_matrix().T
                   + m.origin_m + m.travel_m * ((gap - m.minimum_gap_mm) / m.span_mm)]
    projected = cv2.projectPoints(points, rv, tv, K, DIST)[0].reshape(8, 2)
    if rng is not None:
        projected += rng.normal(0, .08, projected.shape)
    return W.Observation(rv, tv, projected[:4], projected[4:])


class ForwardWidthTests(unittest.TestCase):
    def test_nearest_reference_requires_evidence_at_every_training_aperture(self):
        corners = observation(50).tail_corners
        frames, labels = [], []
        for gap in (0, 32.3, 64.6):
            for k in range(40):
                seen = {44: corners, 46: corners}
                if gap == 0 and k == 0:
                    seen[41] = corners
                frames.append(W.Frame(len(frames), len(frames), np.zeros(3), np.zeros(3), seen))
                labels.append(gap)
        nearer, supported = np.eye(4), np.eye(4)
        nearer[2, 3] = .01; supported[2, 3] = .02
        extra = {41: (nearer, 11.5), 46: (supported, 11.5)}
        with patch.object(cv2, "solvePnP", return_value=(True, np.zeros(3), np.zeros(3))):
            candidates = C.nearest_candidates(frames, labels, {44: 12.18}, extra, K, DIST)
        self.assertEqual(candidates[0]["base"], 46)
        self.assertNotIn("rejected", candidates[0])
        self.assertEqual(candidates[0]["reference_evidence"]["41"]["paired_frames_per_gap"][32.3], 0)
        # Preserve the failure reasons if no candidate can be trained.
        rejected = [{"tail": 44, "rejected": "insufficient reference coverage"}]
        with patch.object(C, "nearest_candidates", return_value=rejected):
            with self.assertRaises(C.TrainingSelectionError) as error:
                C.train_models(frames, [(0, 0, 39), (32.3, 40, 79), (64.6, 80, 119)],
                               {44: 12.18}, extra, K, DIST)
        self.assertEqual(error.exception.candidates, rejected)

    def test_measured_size_revision_keeps_recorded_snapshot_immutable(self):
        original = {"unit": "yuan", "tails": {"38": 5.6, "39": 5.7, "44": 12.18},
                    "bases": {"46": 12.14}, "tips": {"38": 5.6, "39": 5.7},
                    "tips_dict": "4x4", "ball_faces": {"25": "f0"}, "ball_edge_mm": 15.18}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "data/raw").mkdir(parents=True)
            snapshot = root / "data/raw/take_marker_map.json"
            snapshot.write_text(json.dumps(original)); before = snapshot.read_bytes()
            revised = {**original, "unit": "yosub", "ball_edge_mm": 14.6167,
                       "bases": {"46": 11.5}, "measurement_revision": {"source": "caliper"}}
            path = root / "measured.json"; path.write_text(json.dumps(revised))
            analysis, recorded = C.review_marker_maps(root, ["take"], "yosub", path)
            self.assertEqual(analysis["take"], path.resolve())
            self.assertEqual(recorded["take"], snapshot)
            self.assertEqual(snapshot.read_bytes(), before)
            # Physical-unit correction requires an explicit revision.
            with self.assertRaisesRegex(ValueError, "does not match"):
                C.review_marker_maps(root, ["take"], "yosub")
            revised["tails"] = {"38": 5.6, "39": 5.7, "45": 12.18}
            path.write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, "identities"):
                C.review_marker_maps(root, ["take"], "yosub", path)

    def test_uniform_metric_scale_preserves_projected_ball_geometry(self):
        rv = np.array([.2, -.4, .1]); tv = np.array([.03, -.02, .6])
        R = Rotation.from_rotvec([.4, .2, -.1]).as_matrix()
        origin = np.array([.01, .02, -.03])
        points = marker_obj(15.18) @ R.T + origin
        scale = (sum([14.4, 14.9, 14.9, 14.7, 14.4, 14.4])/6)/15.18
        corrected = marker_obj(15.18*scale) @ R.T + origin*scale
        before = cv2.projectPoints(points, rv, tv, K, DIST)[0]
        after = cv2.projectPoints(corrected, rv, tv*scale, K, DIST)[0]
        np.testing.assert_allclose(before, after, atol=1e-9, rtol=0)

    def test_training_ranked_fusion_does_not_double_vote_for_one_reference(self):
        o = observation(50)
        f = W.Frame(0, 10, o.rvec, o.tvec,
                    {44: o.tail_corners, 48: o.tail_corners, 45: o.tail_corners,
                     46: o.base_corners, 47: o.base_corners})
        models = [replace(model(), holdout_p95_mm=3),
                  replace(model(), tail=48, holdout_p95_mm=1),
                  replace(model(), tail=45, base=47, holdout_p95_mm=2)]
        with patch.object(W, "predict_pair", side_effect=[
                W.PairPrediction(80, .2, 1), W.PairPrediction(60, .2, 1),
                W.PairPrediction(40, .2, 1)]):
            width, available = W.predict_width(f, models, {46: np.eye(4), 47: np.eye(4)}, K, DIST)
        self.assertEqual(width, 50)
        self.assertEqual(len(available), 3)  # discarded votes remain inspectable

    def test_legacy_unranked_model_retains_its_frozen_fusion(self):
        o = observation(50)
        f = W.Frame(0, 10, o.rvec, o.tvec,
                    {44: o.tail_corners, 48: o.tail_corners, 45: o.tail_corners,
                     46: o.base_corners, 47: o.base_corners})
        models = [model(), replace(model(), tail=48), replace(model(), tail=45, base=47)]
        with patch.object(W, "predict_pair", side_effect=[
                W.PairPrediction(80, .2, 1), W.PairPrediction(60, .2, 1),
                W.PairPrediction(40, .2, 1)]):
            width, _ = W.predict_width(f, models, {46: np.eye(4), 47: np.eye(4)}, K, DIST)
        self.assertEqual(width, 60)
        historical = model().to_dict()
        historical.pop("holdout_p95_mm")
        self.assertIsNone(W.PairModel.from_dict(historical).holdout_p95_mm)

    def test_ranked_fusion_falls_back_when_best_face_is_missing(self):
        o = observation(50)
        f = W.Frame(0, 10, o.rvec, o.tvec,
                    {44: o.tail_corners, 45: o.tail_corners,
                     46: o.base_corners, 47: o.base_corners})
        models = [replace(model(), holdout_p95_mm=3),
                  replace(model(), tail=48, holdout_p95_mm=1),
                  replace(model(), tail=45, base=47, holdout_p95_mm=2)]
        with patch.object(W, "predict_pair", side_effect=[
                W.PairPrediction(80, .2, 1), W.PairPrediction(40, .2, 1)]):
            width, available = W.predict_width(f, models, {46: np.eye(4), 47: np.eye(4)}, K, DIST)
        self.assertEqual(width, 60)
        self.assertEqual(set(available), {"pair:44-46", "pair:45-47"})

    def test_degenerate_cold_seed_does_not_prevent_other_seed_recovery(self):
        m = model()
        true = observation(67.9, n=3)
        _, rotations, translations, *_ = cv2.solvePnPGeneric(
            marker_obj(m.base_edge_mm), true.base_corners, K, DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        wrong = W.Observation(rotations[1].ravel(), translations[1].ravel(),
                              true.base_corners, true.tail_corners)
        original = cv2.solvePnP
        calls = 0

        def fail_first_seed(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise cv2.error("degenerate SQPnP seed")
            return original(*args, **kwargs)

        with patch.object(cv2, "solvePnP", side_effect=fail_first_seed):
            prediction = W.predict_pair(wrong, m, K, DIST)
        self.assertIsNotNone(prediction)
        self.assertGreater(calls, 1)
        self.assertAlmostEqual(prediction.gap_mm, 67.9, places=4)

    def test_cold_start_recovers_from_wrong_planar_pose_branch(self):
        m = model()
        true = observation(67.9, n=3)
        _, rotations, translations, *_ = cv2.solvePnPGeneric(
            marker_obj(m.base_edge_mm), true.base_corners, K, DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        wrong = W.Observation(rotations[1].ravel(), translations[1].ravel(),
                              true.base_corners, true.tail_corners)
        prediction = W.predict_pair(wrong, m, K, DIST)
        self.assertIsNotNone(prediction)
        self.assertTrue(prediction.used_cold_start)
        self.assertGreater(prediction.initial_rms_px, 1)
        self.assertAlmostEqual(prediction.gap_mm, 67.9, places=4)
        self.assertLess(prediction.rms_px, 1e-5)

    def test_joint_image_constraint_preserves_mm_across_perspectives_without_clipping(self):
        # Known widths only construct synthetic images, never enter inference.
        for gap in (-5., 10., 42.3, 67.9, 90.):
            for n in (0, 3, 8):
                prediction = W.predict_pair(observation(gap, n=n), model(), K, DIST)
                self.assertIsNotNone(prediction)
                self.assertAlmostEqual(prediction.gap_mm, gap, places=4)
                self.assertGreater(prediction.std_mm_per_pixel, 0)

    def test_training_reconstructs_motion_and_generalizes_to_unused_aperture(self):
        rng = np.random.default_rng(8)
        labels, observations = [], []
        for gap in (10., 42.3, 74.6):
            for n in range(18):
                observations.append(observation(gap, rng, n))
                labels.append(gap)
        m = W.fit_pair(observations, labels, K, DIST, tail=44, base=46,
                       tail_edge_mm=12.18, base_edge_mm=12.14,
                       minimum_gap_mm=10, span_mm=64.6)
        for n in (1, 5, 10):
            p = W.predict_pair(observation(67.9, n=n), m, K, DIST)
            self.assertIsNotNone(p)
            self.assertLess(abs(p.gap_mm - 67.9), .35)
        restored = W.PairModel.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_array_equal(restored.parameters, m.parameters)

    def test_single_hold_cannot_produce_a_spurious_width_model(self):
        with self.assertRaisesRegex(ValueError, "One hold"):
            W.fit_pair([observation(42.3)] * 5, [42.3] * 5, K, DIST,
                       tail=44, base=46, tail_edge_mm=12.18, base_edge_mm=12.14,
                       minimum_gap_mm=10, span_mm=64.6)

    def test_ambiguous_markers_and_missing_pairs_never_generate_width(self):
        o = observation(67.9)
        corners = W.unique_corners([44, 44, 46],
                                  [o.tail_corners, o.tail_corners, o.base_corners])
        self.assertEqual(set(corners), {46})
        f = W.Frame(0, 10, o.rvec, o.tvec, corners)
        width, available = W.predict_width(f, [model()], {46: np.eye(4)}, K, DIST)
        self.assertIsNone(width)
        self.assertEqual(available, {})

    def test_base_frame_conversion_does_not_modify_calibrated_geometry(self):
        o = observation(67.9)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
        T[:3, 3] = [.05, -.03, .1]
        original = T.copy()
        local_R = cv2.Rodrigues(o.rvec)[0]
        body_R = local_R @ T[:3, :3].T
        body_t = o.tvec - body_R @ T[:3, 3]
        f = W.Frame(0, 10, cv2.Rodrigues(body_R)[0].ravel(), body_t,
                    {44: o.tail_corners, 46: o.base_corners})
        local = W.local_observation(f, 44, 46, T)
        prediction = W.predict_pair(local, model(), K, DIST)
        self.assertAlmostEqual(prediction.gap_mm, 67.9, places=4)
        np.testing.assert_array_equal(T, original)

    def test_repeated_hold_does_not_label_the_transit_between_holds(self):
        log = {"units": "mm", "events": [
            {"event": "e8_meas", "gap_mm": g, "t_wall": t}
            for g, t in [(0, 0), (0, .5), (32.3, 3), (32.3, 3.5), (0, 6), (0, 6.5)]]}
        holds = C.measurement_holds(log)
        self.assertEqual(holds, [(0, 0, .5), (32.3, 3, 3.5), (0, 6, 6.5)])
        self.assertIsNone(C.gap_at(1, holds))
        self.assertIsNone(C.gap_at(5, holds))

    def test_empty_verification_cannot_pass_vacuously(self):
        with self.assertRaisesRegex(ValueError, "No complete measured hold"):
            C.measurement_holds({"units": "mm", "events": []})

    def test_rejected_frames_stay_in_coverage_denominator(self):
        values = np.r_[np.full(5, 57.9), np.full(5, np.nan)]
        report = C.summarize_holds(np.arange(10), values, [(57.9, 0, 9)])[0]
        self.assertEqual(report["recorded_frames"], 10)
        self.assertEqual(report["width_coverage"], .5)
        self.assertFalse(report["passed"])
        self.assertTrue(report["checks"]["p95_absolute_error_at_most_2mm"])
        # Paired NaNs also preserve the right reference labels in CV statistics.
        summary = W.error_summary([0, np.nan, 64.6], [0, 32.3, 64.6])
        self.assertEqual(summary["absolute_error_p95_mm"], 0)

    def test_projective_center_removes_perspective_mean_bias(self):
        corners = cv2.projectPoints(marker_obj(30), np.array([.5, .7, .1]),
                                     np.array([.08, .02, .2]), K, DIST)[0].reshape(4, 2)
        # Distortion must be removed before applying a projective construction
        # when checking exact geometry; production works in image pixels.
        undistorted = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, DIST, P=K).reshape(4, 2)
        expected = cv2.projectPoints(np.zeros((1, 3)), np.array([.5, .7, .1]),
                                     np.array([.08, .02, .2]), K, np.zeros(5))[0].ravel()
        np.testing.assert_allclose(W.image_center(undistorted), expected, atol=1e-6)
        self.assertGreater(np.linalg.norm(undistorted.mean(axis=0)-expected), .1)


if __name__ == "__main__":
    unittest.main()
