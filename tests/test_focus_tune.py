"""Shared focus analysis and Linux/macOS capture routing, without hardware."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_config import CameraSpec, save_config
import focus_tune as focus


def scene():
    image = np.full((720, 1280, 3), 230, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for marker_id, x in ((2, 200), (3, 800)):
        tag = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
        image[200:360, x:x + 160] = tag[:, :, None]
    return image


class FocusTests(unittest.TestCase):
    def test_blur_measurement_still_uses_native_pixels(self):
        image = scene()
        results = [focus.measure(image if sigma == 0 else cv2.GaussianBlur(image, (0, 0), sigma),
                                 (280, 280), (300, 260)) for sigma in (0, 1, 2)]
        self.assertTrue(all(r[0].shape == image.shape[:2] and len(r[3]) == 2 for r in results))
        self.assertTrue(results[0][2] < results[1][2] < results[2][2])
        self.assertTrue(results[0][1] > results[1][1] > results[2][1])

    def test_native_refinement_does_not_measure_neighbor_marker(self):
        detector = Mock()
        quad = np.array([[[10, 10], [40, 10], [40, 40], [10, 40]]], np.float32)
        right = quad + 12
        detector.detectMarkers.side_effect = [([quad], np.array([[2]]), []),
                                               ([quad + 1, right], np.array([[3], [2]]), [])]
        empty = Mock()
        empty.detectMarkers.return_value = ([], None, [])
        with patch.object(focus, 'DETS', [detector, empty]):
            _, _, _, markers = focus.measure(np.full((200, 200, 3), 120, np.uint8), (100, 100), (40, 40))
        np.testing.assert_array_equal(markers[0][0], right.reshape(4, 2))

    def test_linux_capture_keeps_mjpg_mode_controls_and_warmup(self):
        cap = Mock()
        with patch.object(focus.sys, 'platform', 'linux'), \
                patch.object(focus.cv2, 'VideoCapture', return_value=cap) as factory, \
                patch.object(focus.subprocess, 'run') as controls:
            self.assertIs(focus.open_cam('/dev/video9', 3840, 2160), cap)
            parsed = focus.parse_args([])
        factory.assert_called_once_with('/dev/video9')
        self.assertEqual(parsed.size, '3840x2160')
        self.assertEqual(parsed.dev, '/dev/video0')
        self.assertEqual(cap.read.call_count, 8)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.assertEqual([c.args[0][-1] for c in controls.call_args_list], focus.BASE.split(','))

    def test_mac_reads_latest_once_while_linux_drains_capture_queue(self):
        for platform, expected in [('darwin', 1), ('linux', 3)]:
            cap = Mock()
            cap.read.return_value = (True, scene())
            with patch.object(focus.sys, 'platform', platform), \
                    patch.object(focus, 'open_cam', return_value=cap):
                tuner = focus.Tuner('camera', 'THIS', 1280, 720)
                self.assertIsNotNone(tuner.tick((300, 260)))
            self.assertEqual(cap.read.call_count, expected)

    def test_mac_rejects_mode_mismatch_before_opening_any_camera(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            save_config(path, [CameraSpec('b0587b', 'head-id', 'Twin', 3840, 2160, 25, 180)])
            with patch.object(focus.sys, 'platform', 'darwin'), \
                    patch.object(focus, 'open_cam') as capture:
                with self.assertRaisesRegex(ValueError, '--size'):
                    focus.main(['head', '--size', '1920x1080', '--config', str(path)])
            capture.assert_not_called()

    def test_reference_failure_releases_first_camera(self):
        cap = Mock()
        with patch.object(focus.sys, 'platform', 'linux'), \
                patch.object(focus, 'open_cam', side_effect=[cap, RuntimeError('reference missing')]):
            with self.assertRaisesRegex(RuntimeError, 'reference missing'):
                focus.main(['/dev/video0', '--ref', '/dev/video1'])
        cap.release.assert_called_once()

    def test_shared_loop_sends_composite_to_player_and_quits_from_terminal(self):
        cap, sink = Mock(), Mock()
        cap.read.return_value = (True, scene())
        sink.poll.return_value = None
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'q')
        os.close(write_fd)
        with os.fdopen(read_fd) as terminal, \
                patch.object(focus.sys, 'platform', 'linux'), \
                patch.object(focus.sys, 'stdin', terminal), \
                patch.object(focus, 'open_cam', return_value=cap), \
                patch.object(focus, 'open_sink', return_value=sink) as player, \
                patch.object(focus, 'close_sink') as close, redirect_stdout(io.StringIO()):
            focus.main(['/dev/video0', '--size', '1280x720'])
        player.assert_called_once_with('focus_tune', 1280, 1080)
        self.assertEqual(len(sink.stdin.write.call_args.args[0]), 1280 * 1080 * 3)
        cap.release.assert_called_once()
        close.assert_called_once_with(sink)


if __name__ == '__main__':
    unittest.main()
