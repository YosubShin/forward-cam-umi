"""Calibration routing and real OpenCV fitting from synthetic corner views."""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_uvc as calibration
from camera_config import CameraSpec, save_config


def projected_views():
    board = cv2.aruco.CharucoBoard(
        (10, 7), .0244, .01795,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250))
    K = np.array([[2500., 0, 1920], [0, 2480., 1080], [0, 0, 1.]])
    ids = np.arange(54, dtype=np.int32).reshape(-1, 1)
    rng = np.random.default_rng(13)
    views = []
    for i in range(18):
        rv = rng.uniform(-.5, .5, 3)
        tv = np.array([(-.16 if i % 2 else .04), -.08, rng.uniform(.5, .9)])
        corners, _ = cv2.projectPoints(board.getChessboardCorners(), rv, tv, K, np.zeros(5))
        views.append((corners.astype(np.float32), ids.copy()))
    return K, views


class CalibrationTests(unittest.TestCase):
    def config(self, directory):
        path = Path(directory) / "cameras.local.json"
        save_config(path, [
            CameraSpec("b0587", "forward-id", "Twin", 3840, 2160, 25),
            CameraSpec("b0587b", "head-id", "Twin", 3840, 2160, 25, 180),
            CameraSpec("fish", "wrist-id", "Fish", 1920, 1080, 30),
        ])
        return path

    def test_mac_uses_role_identity_and_rejects_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(calibration.sys, "platform", "darwin"):
            config = self.config(directory)
            for role, name in [("forward", "B0587"), ("head", "B0587b"), ("wrist", "fisheye")]:
                args = calibration.parse_args([role, "--config", str(config)])
                spec = calibration.prepare_camera(args)
                self.assertEqual(args.name, name)
                self.assertEqual((args.width, args.height), (spec.width, spec.height))
                self.assertEqual(spec.device_id, f"{role}-id")
            with patch.object(calibration, "open_capture") as capture:
                with self.assertRaisesRegex(ValueError, "saved recording mode"):
                    calibration.main(["head", "--config", str(config), "--width", "1920"])
            capture.assert_not_called()

    def test_linux_keeps_original_mode_warmup_and_controls(self):
        cap = Mock()
        cap.get.side_effect = [3840, 2160]
        with patch.object(calibration.sys, "platform", "linux"), \
                patch.object(calibration, "device_name", return_value="Arducam-B0587-4K-Low-Light"), \
                patch.object(calibration.cv2, "VideoCapture", return_value=cap) as factory, \
                patch.object(calibration, "apply_ritual") as controls:
            args = calibration.parse_args([])
            spec = calibration.prepare_camera(args)
            self.assertEqual(calibration.open_capture(args, spec), (cap, 3840, 2160))
            controls.assert_called_once_with("/dev/video0")
        factory.assert_called_once_with("/dev/video0")
        self.assertEqual(cap.read.call_count, 5)
        cap.set.assert_any_call(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        self.assertEqual(args.name, "B0587")

    def test_failed_linux_open_releases_camera(self):
        cap = Mock()
        cap.isOpened.return_value = False
        with patch.object(calibration.cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(RuntimeError, "cannot open"):
                calibration.open_capture(SimpleNamespace(dev="/dev/video0", width=3840, height=2160), None)
        cap.release.assert_called_once()

    def test_capture_and_refit_preserve_board_orientation_and_fit(self):
        K, views = projected_views()
        # The native adapter has already rotated the image; the calibration
        # loop must pass that exact image through to the detector unchanged.
        canonical = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
        cap = Mock()
        cap.read.return_value = (True, canonical)
        native = SimpleNamespace(OpenCVPreview=Mock(return_value=cap))
        detector = Mock()
        detector.detectBoard.side_effect = [(c, i, None, None) for c, i in views]
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch.object(calibration.sys, "platform", "darwin"), \
                patch.dict(sys.modules, {"camera_macos": native}), \
                patch.object(calibration.cv2.aruco, "CharucoDetector", return_value=detector), \
                patch.object(calibration.time, "time", side_effect=range(1, 100)), \
                redirect_stdout(io.StringIO()):
            config = self.config(directory)
            calibration.main(["head", "--config", str(config), "--n", "18", "--holdout", "4",
                              "--square-mm", "24.4", "--marker-mm", "17.95"])
            out = Path("calib/B0587b_intrinsics.json")
            result = json.loads(out.read_text())
            np.testing.assert_allclose(result["camera_matrix"], K, atol=.03)
            self.assertLess(result["holdout_mean_px"], .001)
            self.assertEqual(result["capture"]["rotation_degrees"], 180)
            self.assertEqual(result["capture"]["device_id"], "head-id")
            self.assertFalse(result["capture"]["exposure_gain_controlled"])
            self.assertEqual(result["board"]["marker_mm"], 17.95)
            self.assertTrue(all(call.args[0] is canonical for call in detector.detectBoard.call_args_list))
            self.assertEqual(native.OpenCVPreview.call_args.args[0].rotation_degrees, 180)
            cap.release.assert_called_once()

            # A refit must work even after the physical devices/config vanish.
            config.unlink()
            with patch.object(calibration, "open_capture") as capture:
                calibration.main(["--refit", "calib/B0587b_charuco_views.npz", "--holdout", "4"])
            capture.assert_not_called()
            refit = json.loads(out.read_text())
            np.testing.assert_allclose(refit["camera_matrix"], result["camera_matrix"])
            self.assertEqual(refit["capture"], result["capture"])
            self.assertEqual(refit["board"], result["board"])
            backups = list(Path("calib").glob("B0587b_intrinsics.json.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), result)

    def test_interruption_keeps_existing_outputs_and_releases_capture(self):
        cap = Mock()
        cap.read.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch.object(calibration.sys, "platform", "darwin"), \
                patch.object(calibration, "open_capture", return_value=(cap, 3840, 2160)), \
                redirect_stdout(io.StringIO()):
            config = self.config(directory)
            Path("calib").mkdir()
            outputs = [Path("calib/B0587_intrinsics.json"), Path("calib/B0587_charuco_views.npz")]
            for path in outputs:
                path.write_bytes(b"previous result")
            with self.assertRaises(KeyboardInterrupt):
                calibration.main(["forward", "--config", str(config)])
            self.assertTrue(all(p.read_bytes() == b"previous result" for p in outputs))
        cap.release.assert_called_once()

    def test_legacy_views_can_refit_without_capture_metadata(self):
        K, views = projected_views()
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
                patch.object(calibration, "open_capture") as capture, \
                redirect_stdout(io.StringIO()):
            np.savez_compressed("legacy_charuco_views.npz", n=len(views), width=3840, height=2160,
                                **{f"c{i}": v[0] for i, v in enumerate(views)},
                                **{f"i{i}": v[1] for i, v in enumerate(views)})
            calibration.main(["--refit", "legacy_charuco_views.npz", "--holdout", "4",
                              "--square-mm", "24.4", "--marker-mm", "17.95"])
            result = json.loads(Path("calib/legacy_intrinsics.json").read_text())
            np.testing.assert_allclose(result["camera_matrix"], K, atol=.03)
            self.assertNotIn("capture", result)
            self.assertIn("not recorded", result["note"])
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
