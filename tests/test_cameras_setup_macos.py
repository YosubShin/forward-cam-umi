"""Terminal setup preserves bindings on failure and verifies capture modes."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_config import CameraSpec, load_config, save_config, resolve_spec
import cameras_setup_macos as setup


class CameraSetupTests(unittest.TestCase):
    def setUp(self):
        self.inventory = [
            {'device_id': 'front-id', 'name': 'Twin camera',
             'modes': [(3840, 2160, 25, 25), (1280, 720, 15, 30)]},
            {'device_id': 'head-id', 'name': 'Twin camera',
             'modes': [(3840, 2160, 25, 25), (1280, 720, 15, 30)]},
            {'device_id': 'wrist-id', 'name': 'Wrist camera',
             'modes': [(1920, 1080, 30.00003, 30.00003), (640, 480, 5, 30)]},
        ]
        self.specs = [CameraSpec('b0587', 'front-id', 'Twin camera', 3840, 2160, 25),
                      CameraSpec('b0587b', 'head-id', 'Twin camera', 3840, 2160, 25, 180),
                      CameraSpec('fish', 'wrist-id', 'Wrist camera', 1920, 1080, 30)]

    def run_wizard(self, path, answers, preview_effect=None):
        args = SimpleNamespace(config=path, seconds=1, require_head=True)
        with patch.object(setup.sys.stdin, 'isatty', return_value=True), \
                patch('builtins.input', side_effect=answers), \
                patch.object(setup, 'preview', side_effect=preview_effect) as preview, \
                redirect_stdout(io.StringIO()):
            setup.wizard(self.inventory, args)
        return preview

    def test_per_device_assignment_keeps_native_modes_and_head_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            preview = self.run_wizard(path, ['b0587', '', '', 'y',
                                             'b0587b', '180', '', 'y',
                                             'fish', '', '', 'y'])
            self.assertEqual(load_config(path), self.specs)
        shown = [c.args[0] for c in preview.call_args_list]
        self.assertEqual([s.width for s in shown], [1280, 3840, 1280, 3840, 640, 1920])
        self.assertEqual(shown[3].rotation_degrees, 180)

    def test_successful_rerun_backs_up_existing_config_and_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            save_config(path, self.specs)
            before = path.read_bytes()
            self.run_wizard(path, ['b0587', '', '', 'y', 'b0587b', '', '', 'y', 'fish', '', '', 'y'])
            self.assertEqual(load_config(path), self.specs)
            backups = list(Path(d).glob('*.backup-*.local.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before)

    def test_cancel_or_missing_required_role_preserves_existing_file(self):
        for answers, error in [(['b0587', KeyboardInterrupt()], KeyboardInterrupt),
                               (['-', '-', '-'], ValueError)]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'cameras.local.json'
                save_config(path, self.specs)
                before = path.read_bytes()
                with self.assertRaises(error):
                    self.run_wizard(path, answers)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(Path(d).iterdir()), [path])

    def test_unavailable_preview_cannot_be_assigned(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            with self.assertRaisesRegex(ValueError, 'required roles'):
                self.run_wizard(path, ['b0587', '-', '-', '-'],
                                preview_effect=RuntimeError('no frames'))
            self.assertFalse(path.exists())

    def test_unsupported_recommendation_requires_explicit_supported_mode(self):
        # A role is an identity, not a sensor-resolution constraint.
        with patch('builtins.input', side_effect=['0', '', '1920x1080@30']), \
                redirect_stdout(io.StringIO()):
            spec = setup.capture_spec('b0587', self.inventory[2], {})
        self.assertEqual((spec.device_id, spec.width, spec.height, spec.fps),
                         ('wrist-id', 1920, 1080, 30))

    def test_role_and_exact_id_resolution_do_not_use_same_model_name(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            save_config(path, self.specs)
            self.assertEqual(resolve_spec('head', path), self.specs[1])
            self.assertEqual(resolve_spec('front-id', path), self.specs[0])
            with self.assertRaises(ValueError):
                resolve_spec('Twin camera', path)
        for bad_inventory in ([], [self.inventory[0], self.inventory[0]]):
            with self.assertRaises(ValueError):
                setup.validate_device(self.specs[0], bad_inventory)

    @unittest.skipUnless(sys.platform == 'darwin', 'native enumeration requires macOS')
    def test_preview_checks_only_the_selected_role(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cameras.local.json'
            save_config(path, self.specs)
            with patch('camera_macos.devices', return_value=['head-device']), \
                    patch('camera_macos.device_info', return_value=self.inventory[1]), \
                    patch.object(setup, 'preview') as preview:
                setup.main(['--preview', 'head', '--config', str(path)])
            preview.assert_called_once_with(self.specs[1], 8)


if __name__ == '__main__':
    unittest.main()
