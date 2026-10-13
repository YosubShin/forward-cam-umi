"""Workspace scale propagation and unchanged legacy layout geometry."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tiles_pose as T

K = np.array([[1800., 0, 1920], [0, 1800., 1080], [0, 0, 1]])
DIST = np.array([-.08, .02, .001, -.001, 0.])
RV = np.array([3.05, .1, .05])
TV = np.array([-.25, .15, .8])
POSITIONS = {2: [0, .35, 0], 3: [.5, .35, 0], 4: [.5, 0, 0], 5: [0, 0, 0]}
EDGES = {"2": 60.15, "3": 59.25, "4": 59.35, "5": 59.55}


def scene(measured=True):
    layout = {"edge_mm": 60.14,
              "tiles": {str(i): {"xyz_mm": (np.array(p)*1000).tolist(), "rvec": [0, 0, 0]}
                        for i, p in POSITIONS.items()}}
    if measured:
        layout["marker_edges_mm"] = EDGES
    corners = {}
    for i, p in POSITIONS.items():
        edge = EDGES[str(i)] if measured else layout["edge_mm"]
        points = T.tag_obj(edge/1000) + np.array(p)
        corners[i] = cv2.projectPoints(points, RV, TV, K, DIST)[0].reshape(4, 2)
    return layout, corners


class TilesPoseTests(unittest.TestCase):
    def test_single_rotated_translated_tile_uses_world_coordinates(self):
        # The old IPPE_SQUARE branch assumed centred local object points and
        # could return extreme reprojection errors for these world corners.
        position = np.array([.42, .24, .015])
        rotation = np.array([.025, -.03, 1.2])
        layout = {"edge_mm": 60.14, "marker_edges_mm": {"5": 59.55},
                  "tiles": {"5": {"xyz_mm": (position*1000).tolist(),
                                  "rvec": rotation.tolist()}}}
        points = T.tag_obj(.05955) @ cv2.Rodrigues(rotation)[0].T + position
        corners = cv2.projectPoints(points, RV, TV, K, DIST)[0].reshape(4, 2)
        transform, residual, count = T.solve_frame({5: corners}, layout, K, DIST)
        np.testing.assert_allclose(transform, T.to_T(RV, TV), atol=1e-6, rtol=0)
        self.assertLess(residual, 1e-6)
        self.assertEqual(count, 1)

    def test_nonfinite_and_negative_depth_camera_poses_are_rejected(self):
        layout, corners = scene()
        for rv, tv in [(np.full((3, 1), np.nan), TV.reshape(3, 1)),
                       (RV.reshape(3, 1), np.array([[0.], [0.], [-10.]]))]:
            with patch.object(cv2, "solvePnP", return_value=(True, rv, tv)):
                self.assertIsNone(T.solve_frame({5: corners[5]}, layout, K, DIST))

    def test_per_id_edges_recover_camera_pose_without_uniform_scale_bias(self):
        layout, corners = scene()
        result = T.solve_frame(corners, layout, K, DIST)
        self.assertIsNotNone(result)
        transform, residual, count = result
        np.testing.assert_allclose(transform, T.to_T(RV, TV), atol=1e-6, rtol=0)
        self.assertLess(residual, 1e-6)
        self.assertEqual(count, 4)

    def test_legacy_uniform_layout_retains_camera_pose(self):
        layout, corners = scene(measured=False)
        transform, residual, count = T.solve_frame(corners, layout, K, DIST)
        np.testing.assert_allclose(transform, T.to_T(RV, TV), atol=1e-6, rtol=0)
        self.assertLess(residual, 1e-6)
        self.assertEqual(count, 4)

    def test_invalid_metric_edge_is_rejected(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                T.tag_edge_mm({"edge_mm": 60.14, "marker_edges_mm": {"5": value}}, 5)

    def test_layout_preserves_per_id_edges_for_subsequent_solve(self):
        _, corners = scene()
        ids = np.array(list(corners), dtype=np.int32).reshape(-1, 1)
        pixels = [c.reshape(1, 4, 2).astype(np.float64) for c in corners.values()]
        detector = SimpleNamespace(detectMarkers=lambda _: (pixels, ids, None))
        revision = {"raw_black_edges_mm": {"5": {"horizontal": 60., "vertical": 59.1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"calib").mkdir()
            intr, tags = root/"intr.json", root/"tags.json"
            intr.write_text(json.dumps({"camera_matrix": K.tolist(), "dist_coeffs": DIST.tolist()}))
            tags.write_text(json.dumps({"ids": list(corners), "edge_mm": 60.14,
                                        "marker_edges_mm": EDGES, "measurement_revision": revision}))
            args = SimpleNamespace(intr=str(intr), tags=str(tags), stem="synthetic",
                                   stride=1, origin=5, xref=4, name="measured")
            with patch.object(T, "ROOT", root), patch.object(T, "detector", return_value=detector), \
                 patch.object(T, "sample_frames", return_value=iter([(0, None), (1, None)])), \
                 redirect_stdout(io.StringIO()):
                T.cmd_layout(args)
            output = json.loads((root/"calib/workspace_tiles_measured.json").read_text())
            self.assertEqual(output["marker_edges_mm"], EDGES)
            self.assertEqual(output["measurement_revision"], revision)
            self.assertAlmostEqual(np.linalg.norm(np.array(output["tiles"]["4"]["xyz_mm"])), 500., places=1)


if __name__ == "__main__":
    unittest.main()
