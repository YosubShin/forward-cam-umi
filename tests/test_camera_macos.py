"""Native buffer/identity tests without opening a camera or requesting access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if sys.platform == "darwin":
    import CoreMedia as CM
    import Quartz as CV
    import camera_macos as native
    from camera_config import CameraSpec


@unittest.skipUnless(sys.platform == "darwin", "AVFoundation requires macOS")
class NativeCameraTests(unittest.TestCase):
    def test_opencv_adapter_returns_normalized_pixels_without_rotating_again(self):
        canonical = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        spec = CameraSpec("b0587b", "head-id", "Twin", 5, 3, 25, 180)
        with patch.object(native, "PreviewCapture") as factory:
            capture = factory.return_value
            capture.read.return_value = SimpleNamespace(image=canonical)
            adapter = native.OpenCVPreview(spec)
            ok, pixels = adapter.read()
            adapter.release()
        factory.assert_called_once_with(spec)
        capture.__enter__.assert_called_once()
        self.assertTrue(ok)
        self.assertIs(pixels, canonical)
        capture.close.assert_called_once()

    def test_start_configures_device_without_setting_input_priority_preset(self):
        device = Mock()
        device.localizedName.return_value = "test camera"
        rate = Mock()
        rate.minFrameRate.return_value = rate.maxFrameRate.return_value = 25
        rate.maxFrameDuration.return_value = CM.CMTimeMake(1, 25)
        fmt = Mock()
        fmt.videoSupportedFrameRateRanges.return_value = [rate]
        device.formats.return_value = [fmt]
        device.lockForConfiguration_.return_value = (True, None)
        session, output, capture_input = Mock(), Mock(), Mock()
        # Reproduce the platform's rejection; startup must never call it.
        session.setSessionPreset_.side_effect = RuntimeError("unsupported preset")
        calls = Mock()
        calls.attach_mock(session, "session")
        calls.attach_mock(device, "device")
        cap = native.PreviewCapture(CameraSpec("b0587", "test", "test camera", 5, 3, 25))
        with ExitStack() as stack:
            stack.enter_context(patch.object(native, "request_camera_access"))
            stack.enter_context(patch.object(native, "resolve_device", return_value=device))
            stack.enter_context(patch.object(native.CM, "CMVideoFormatDescriptionGetDimensions",
                                            return_value=SimpleNamespace(width=5, height=3)))
            session_class = stack.enter_context(patch.object(native.AV, "AVCaptureSession"))
            session_class.alloc.return_value.init.return_value = session
            output_class = stack.enter_context(patch.object(native.AV, "AVCaptureVideoDataOutput"))
            output_class.alloc.return_value.init.return_value = output
            input_class = stack.enter_context(patch.object(native.AV, "AVCaptureDeviceInput"))
            input_class.deviceInputWithDevice_error_.return_value = (capture_input, None)
            with cap:
                session.setSessionPreset_.assert_not_called()
                device.setActiveFormat_.assert_called_once_with(fmt)
                device.setActiveVideoMinFrameDuration_.assert_called_once_with(CM.CMTimeMake(1, 25))
                device.setActiveVideoMaxFrameDuration_.assert_called_once_with(CM.CMTimeMake(1, 25))
                output.setVideoSettings_.assert_called_once_with({
                    CV.kCVPixelBufferPixelFormatTypeKey: CV.kCVPixelFormatType_32BGRA,
                    CV.kCVPixelBufferWidthKey: 5,
                    CV.kCVPixelBufferHeightKey: 3,
                })
                names = [c[0] for c in calls.mock_calls]
                self.assertLess(names.index("session.addOutput_"), names.index("device.setActiveFormat_"))
                self.assertLess(names.index("device.unlockForConfiguration"), names.index("session.commitConfiguration"))
                self.assertLess(names.index("session.commitConfiguration"), names.index("session.startRunning"))
                source_checks = [i for i, name in enumerate(names) if name == "device.activeFormat"]
                self.assertEqual(len(source_checks), 2)
                self.assertLess(names.index("session.commitConfiguration"), source_checks[0])
                self.assertLess(source_checks[0], names.index("session.startRunning"))
                self.assertLess(names.index("session.startRunning"), source_checks[1])
            session.stopRunning.assert_called_once()
            output.setSampleBufferDelegate_queue_.assert_called_with(None, None)

    def test_rejects_source_fallback_even_when_output_size_is_requested(self):
        cap = native.PreviewCapture(CameraSpec("b0587", "test", "test", 3840, 2160, 25))
        device = Mock()
        with patch.object(native.CM, "CMVideoFormatDescriptionGetDimensions",
                          return_value=SimpleNamespace(width=1920, height=1080)):
            with self.assertRaisesRegex(RuntimeError, "camera source after startup is 1920x1080"):
                cap._check_source_dimensions(device, "after startup")

    def test_startup_source_failure_closes_capture(self):
        # A startup validation error must not leave the camera/delegate active.
        cap = native.PreviewCapture(CameraSpec("b0587", "test", "test", 3840, 2160, 25))
        cap.session, cap.output = Mock(), Mock()
        session, output = cap.session, cap.output
        with patch.object(cap, "start", side_effect=RuntimeError("source mismatch")):
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                with cap:
                    self.fail("invalid source must not become usable")
        self.assertTrue(cap.closed)
        session.stopRunning.assert_called_once()
        output.setSampleBufferDelegate_queue_.assert_called_once_with(None, None)

    def make_sample(self, pts=1234.25):
        err, pixel = CV.CVPixelBufferCreate(None, 5, 3, CV.kCVPixelFormatType_32BGRA, None, None)
        self.assertEqual(err, 0)
        CV.CVPixelBufferLockBaseAddress(pixel, 0)
        try:
            stride = CV.CVPixelBufferGetBytesPerRow(pixel)
            rows = np.frombuffer(CV.CVPixelBufferGetBaseAddress(pixel).as_buffer(stride * 3),
                                 dtype=np.uint8).reshape(3, stride)
            rows[:] = 255  # Non-image padding must not appear in output.
            expected = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)
            rows[:, :20] = expected.reshape(3, 20)
        finally:
            CV.CVPixelBufferUnlockBaseAddress(pixel, 0)
        err, fmt = CM.CMVideoFormatDescriptionCreateForImageBuffer(None, pixel, None)
        self.assertEqual(err, 0)
        timing = CM.CMSampleTimingInfo(CM.CMTimeMake(1, 25),
                                     CM.CMTimeMakeWithSeconds(pts, 1000), CM.kCMTimeInvalid)
        err, sample = CM.CMSampleBufferCreateReadyWithImageBuffer(None, pixel, fmt, timing, None)
        self.assertEqual(err, 0)
        return pixel, sample, expected[:, :, :3]

    def test_rotated_buffer_owns_pixels_and_preserves_sample_timestamp(self):
        pixel, sample, expected = self.make_sample()
        cap = native.PreviewCapture(CameraSpec("b0587b", "test", "test", 5, 3, 25, 180))
        sink = native._FrameSink.alloc().init()
        sink.owner = cap
        sink.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample, None)
        frame = cap.read(timeout=0.01)
        self.assertEqual(frame.presentation_seconds, 1234.25)
        np.testing.assert_array_equal(frame.image, expected[::-1, ::-1])
        # Reusing native memory must not mutate a frame already handed off.
        CV.CVPixelBufferLockBaseAddress(pixel, 0)
        try:
            stride = CV.CVPixelBufferGetBytesPerRow(pixel)
            np.frombuffer(CV.CVPixelBufferGetBaseAddress(pixel).as_buffer(stride * 3),
                          dtype=np.uint8)[:] = 0
        finally:
            CV.CVPixelBufferUnlockBaseAddress(pixel, 0)
        np.testing.assert_array_equal(frame.image, expected[::-1, ::-1])
        sink.owner = None
        cap.close()

    def test_preview_keeps_latest_frame_and_rejects_wrong_dimensions(self):
        cap = native.PreviewCapture(CameraSpec("fish", "test", "test", 5, 3, 30))
        sink = native._FrameSink.alloc().init()
        sink.owner = cap
        for pts in (1, 2):
            _, sample, _ = self.make_sample(pts)
            sink.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample, None)
        self.assertEqual(cap.read(timeout=0.01).presentation_seconds, 2)
        cap.spec = CameraSpec("fish", "test", "test", 6, 3, 30)
        sink.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample, None)
        with self.assertRaisesRegex(RuntimeError, "expected 6x3"):
            cap.read(timeout=0.01)
        cap.close()
        sink.owner = None

    def test_missing_id_never_falls_back_to_identical_name(self):
        class Device:
            def __init__(self, uid):
                self.uid = uid

            def uniqueID(self):
                return self.uid

            def localizedName(self):
                return "identical name"

        first, second = Device("first"), Device("second")
        with patch.object(native, "devices", return_value=[first, second]):
            self.assertIs(native.resolve_device("second"), second)
            with self.assertRaisesRegex(RuntimeError, "no name/index fallback"):
                native.resolve_device("missing")
        with patch.object(native, "devices", return_value=[first, first]):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                native.resolve_device("first")


if __name__ == "__main__":
    unittest.main()
