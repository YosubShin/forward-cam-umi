"""Physical marker identity, metric scale and session hand-off regressions."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import detect_parallel
import labels_lib as L
import pack_session
import record_session
import session_check
from marker_config import WristMarkers, load_marker_map, resolve_marker_map, save_snapshot


def marker_map(family="4x4"):
    return {"unit": "test", "tips_dict": family,
            "tips": {"38": 12.18, "39": 12.18} if family == "4x4" else {"2": 6.14, "3": 6.14},
            "tails": {"38": 12.18, "39": 12.18}, "bases": {"40": 12.14}}


def image_of(family, ids):
    code = cv2.aruco.DICT_4X4_50 if family == "4x4" else cv2.aruco.DICT_APRILTAG_16h5
    dictionary = cv2.aruco.getPredefinedDictionary(code)
    image = np.full((240, 220 * len(ids) + 20), 255, np.uint8)
    for n, marker_id in enumerate(ids):
        x = 30 + 220 * n
        image[40:200, x:x + 160] = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class MarkerConfigurationTests(unittest.TestCase):
    def test_both_families_filter_unrelated_and_ambiguous_ids(self):
        for family, pair, unrelated in (("4x4", (38, 39), 40), ("16h5", (2, 3), 4)):
            config = WristMarkers.from_map(marker_map(family))
            for visible, expected in ((pair, 2), ((pair[0], unrelated), 1),
                                      ((pair[0], pair[0], pair[1]), 1), ((unrelated,), 0)):
                with self.subTest(family=family, visible=visible):
                    _, count, black, contrast = record_session._frame_health(
                        "take_fish", image_of(family, visible), config)
                    self.assertEqual(count, expected)
                    self.assertFalse(record_session._washed(black, contrast))
                    self.assertEqual(record_session._required_markers("take_fish", config), 2)
        # The two dictionaries are not interchangeable, even when IDs look similar.
        self.assertEqual(record_session._frame_health("take_fish", image_of("4x4", (38, 39)))[1], 0)
        self.assertEqual(record_session._frame_health("take_fish", image_of("16h5", (2, 3)))[1], 2)
        self.assertEqual(record_session._required_markers("take_fish", None), 1)
        self.assertEqual(record_session._frame_health("fish_trial_b0587", image_of("4x4", (40,)))[1], 1)

    def test_invalid_identity_or_scale_fails_before_capture(self):
        for changes in ({"tips_dict": "36h11"}, {"tips": {"38": 12.18}},
                        {"tips": {"38": 6.14, "39": 12.18}},
                        {"tips": {"38": float("nan"), "39": 12.18}},
                        {"tips": {"38": True, "39": 12.18}},
                        {"tips": {"38": -1, "39": 12.18}},
                        {"tips": {"38": 12.18, "50": 12.18}},
                        {"ball_faces": {"38": "f0"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                WristMarkers.from_map(marker_map() | changes)

    def test_snapshot_freezes_config_and_is_shipped_instead_of_legacy_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calib").mkdir()
            legacy = root / "calib/marker_map_glove_test.json"
            legacy.write_text(json.dumps(marker_map("16h5")))
            local = root / "calib/marker_map_glove_test.local.json"
            local.write_text(json.dumps(marker_map()))
            self.assertEqual(resolve_marker_map(root, "take", unit="test"), legacy)
            saved = save_snapshot(root, "take", load_marker_map(local))
            self.assertEqual(resolve_marker_map(root, "take", unit="test"), saved)
            self.assertEqual(resolve_marker_map(root, "take", path=local), local)
            local.write_text(json.dumps(marker_map("16h5")))
            with self.assertRaisesRegex(ValueError, "new dataset"):
                resolve_marker_map(root, "take", path=local)
            with self.assertRaisesRegex(ValueError, "new dataset"):
                save_snapshot(root, "take", marker_map("16h5"))
            with self.assertRaisesRegex(ValueError, "unit"):
                resolve_marker_map(root, "take", unit="different")
            self.assertEqual(load_marker_map(saved), marker_map())
            with patch.object(pack_session, "ROOT", root), patch.object(sys, "argv", [
                    "pack_session.py", "take", "--unit", "test"]), patch("builtins.print"):
                pack_session.main()
            manifest = (root / "data/raw/handoff_test.txt").read_text().splitlines()
            self.assertIn("data/raw/take_marker_map.json", manifest)
            self.assertNotIn("calib/marker_map_glove_test.json", manifest)
            self.assertNotIn("calib/marker_map_glove_test.local.json", manifest)

    def test_existing_detection_worker_covers_both_families(self):
        for family, pair in (("4x4", (38, 39)), ("16h5", (2, 3))):
            cap = Mock()
            cap.grab.side_effect = [True, False]
            cap.retrieve.return_value = (True, image_of(family, pair))
            cap.read.return_value = (False, None)
            with patch.object(cv2, "VideoCapture", return_value=cap):
                _, frames, aruco, april = detect_parallel.work(("fake.mkv", 0, 1, True, True))
            self.assertEqual(frames, 1)
            result = aruco if family == "4x4" else april
            self.assertEqual(set(result[0][0]), set(pair))

    def test_recording_gates_on_full_pair_and_passes_config_to_shared_preflight(self):
        # Exercise the real CLI/preflight flow with no hardware, on the Linux
        # path too: marker configuration is not a macOS-specific exception.
        for visible in ((38,), (38, 39)):
            with self.subTest(visible=visible), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "data/raw").mkdir(parents=True)
                local = root / "markers.local.json"
                local.write_text(json.dumps(marker_map()))
                device = root / "video0"
                device.touch()
                process = Mock()
                process.poll.return_value = None
                process.wait.return_value = 0
                with ExitStack() as stack:
                    stack.enter_context(patch.object(record_session, "PILOT", root))
                    stack.enter_context(patch.object(record_session, "CAMERAS", [
                        ("fish", str(device), None, "1920x1080", "30")]))
                    stack.enter_context(patch.object(sys, "platform", "linux"))
                    stack.enter_context(patch.object(sys, "argv", ["record_session.py", "--cameras_only",
                        "--dataset", "take", "--cameras", "fish", "--duration", "0.01", "--no_preview",
                        "--marker-map", str(local)]))
                    stack.enter_context(patch.object(sys.stdin, "isatty", return_value=False))
                    stack.enter_context(patch.object(record_session.atexit, "register"))
                    stack.enter_context(patch.object(record_session.time, "sleep"))
                    stack.enter_context(patch("shutil.disk_usage", return_value=Mock(free=100e9)))
                    stack.enter_context(patch.object(record_session.subprocess, "run", return_value=Mock(stdout="")))
                    stack.enter_context(patch.object(record_session.subprocess, "Popen", return_value=process))
                    stack.enter_context(patch.object(record_session, "_tail_frame", return_value=image_of("4x4", visible)))
                    stack.enter_context(patch("builtins.print"))
                    if len(visible) == 1:
                        with self.assertRaisesRegex(SystemExit, "1/2 required markers"):
                            record_session.main()
                        # In production atexit finalizes this startup failure.
                        for call in record_session.subprocess.Popen.call_args_list:
                            call.kwargs["stdout"].close()
                    else:
                        record_session.main()
                self.assertEqual(load_marker_map(root / "data/raw/take_marker_map.json"), marker_map())
                process.send_signal.assert_called()

    def test_metric_separation_uses_correct_cache_ids_and_black_edge_size(self):
        for family in ("4x4", "16h5"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "calib").mkdir()
                (root / "data/raw").mkdir(parents=True)
                config = WristMarkers.from_map(marker_map(family))
                K = np.array([[800., 0, 480], [0, 800, 270], [0, 0, 1]])
                intrinsics = root / "calib/intrinsics.json"
                intrinsics.write_text(json.dumps({"camera_matrix": K.tolist(), "dist_coeffs": [0.] * 5}))
                ids, corners = [], []
                for (marker_id, edge), x in zip(config.markers, (-.04, .04)):
                    c, _ = cv2.projectPoints(L.marker_obj(edge), np.array([.2, -.15, .05]),
                                             np.array([x, 0, .3]), K, np.zeros(5))
                    ids.append(marker_id)
                    corners.append(c.reshape(4, 2))
                # Full pair, only one jaw, duplicate ambiguous ID.
                payload = {"frame_indices": [0, 1, 2], "i0": ids, "c0": corners,
                           "i1": ids[:1], "c1": corners[:1],
                           "i2": [ids[0], ids[0], ids[1]],
                           "c2": [corners[0], corners[0], corners[1]]}
                np.savez(root / f"calib/take_fish_{config.cache_tag}_detections.npz", **payload)
                (root / "data/raw/take_fish_ts.json").write_text(json.dumps([{"wall": 100 + i} for i in range(3)]))
                times, separation = session_check.wrist_series(root, "take", config, intrinsics)
                np.testing.assert_equal(times, [100, 101, 102])
                self.assertAlmostEqual(separation[0], 80, places=5)
                self.assertTrue(np.isnan(separation[1:]).all())

    def test_session_check_fails_when_only_wrong_family_cache_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calib").mkdir()
            (root / "data/raw").mkdir(parents=True)
            save_snapshot(root, "take", marker_map())
            (root / "data/raw/take_glove_episodes.json").write_text(json.dumps({"episodes": [
                {"episode_index": 0, "t_start": 100, "t_end": 100.08, "duration_s": .08}]}))
            for role in ("b0587", "fish"):
                (root / f"data/raw/take_{role}_ts.json").write_text(json.dumps([
                    {"wall": 100 + i * .04} for i in range(3)]))
            np.savez(root / "calib/take_b0587_dodeca_detections.npz", frame_indices=[])
            np.savez(root / "calib/take_fish_tips16h5_detections.npz", frame_indices=[])
            (root / "calib/tags36h11_test.json").write_text('{"ids": [1, 2]}')
            cap = Mock()
            cap.get.return_value = 0
            cap.grab.return_value = False
            cap.read.return_value = (False, None)
            with ExitStack() as stack:
                stack.enter_context(patch.object(session_check, "ROOT", root))
                stack.enter_context(patch.object(sys, "argv", ["session_check.py", "take", "--unit", "test"]))
                stack.enter_context(patch.object(L, "load_bundle", return_value=({}, None)))
                stack.enter_context(patch.object(L, "load_intrinsics", return_value=(np.eye(3), np.zeros(5))))
                stack.enter_context(patch.object(L, "body_frames", return_value=[{"t_wall": 100 + i*.04} for i in range(3)]))
                stack.enter_context(patch.object(cv2, "VideoCapture", return_value=cap))
                output = stack.enter_context(patch("builtins.print"))
                with self.assertRaises(SystemExit) as stopped:
                    session_check.main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertIn("wrist width not checked", str(output.call_args_list))


if __name__ == "__main__":
    unittest.main()
