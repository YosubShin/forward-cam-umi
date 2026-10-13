"""Coordinate and identity regressions; no cameras or macOS needed."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from camera_config import (CameraSpec, load_config, normalize_image, save_config,
                           supported_frame_rate, validate_specs)


class CameraConfigTests(unittest.TestCase):
    def setUp(self):
        self.front = CameraSpec("b0587", "test-front", "Identical camera", 3840, 2160, 25)
        self.head = CameraSpec("b0587b", "test-head", "Identical camera", 3840, 2160, 25, 180)

    def test_same_name_different_ids_and_rotation_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cameras.json"
            save_config(path, [self.front, self.head])
            self.assertEqual(load_config(path), [self.front, self.head])
            with self.assertRaises(FileExistsError):
                save_config(path, [self.front])
            self.assertEqual(load_config(path), [self.front, self.head])

    def test_one_device_cannot_be_assigned_twice(self):
        with self.assertRaisesRegex(ValueError, "duplicate device_id"):
            validate_specs([self.front, replace(self.head, device_id=self.front.device_id)])
        with self.assertRaisesRegex(ValueError, "duplicate role"):
            validate_specs([self.front, replace(self.head, role=self.front.role)])

    def test_invalid_modes_are_rejected(self):
        for patch in ({"rotation_degrees": 90}, {"rotation_degrees": False},
                      {"width": 0}, {"height": 1.5}, {"fps": float("nan")},
                      {"fps": 0}, {"device_id": ""}, {"role": "unknown"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                replace(self.front, **patch)

    def test_uvc_frame_interval_rounding_is_not_a_silent_fps_fallback(self):
        actual = 30.00003000003
        self.assertEqual(supported_frame_rate(30, actual, actual), actual)
        self.assertEqual(supported_frame_rate(25, 5, 30), 25)
        self.assertIsNone(supported_frame_rate(30, 29.97, 29.97))
        self.assertIsNone(supported_frame_rate(30, 5, 25))

    def test_legacy_config_is_not_silently_misread(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cameras.json"
            path.write_text(json.dumps([["b0587", "/dev/video0", "name", "3840x2160", "25"]]))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_rotation_preserves_pixels_channels_and_dimensions(self):
        source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        before = source.copy()
        rotated = normalize_image(source, 180)
        self.assertEqual(rotated.shape, source.shape)
        self.assertTrue(rotated.flags.c_contiguous)
        np.testing.assert_array_equal(rotated[0, 0], source[-1, -1])
        np.testing.assert_array_equal(rotated[-1, 0], source[0, -1])
        np.testing.assert_array_equal(normalize_image(rotated, 180), source)
        np.testing.assert_array_equal(source, before)
        self.assertIs(normalize_image(source, 0), source)


if __name__ == "__main__":
    unittest.main()
