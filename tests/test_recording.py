"""Real MJPEG round-trip, timestamps with gaps, and native delivery failures."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from camera_config import CameraSpec, normalize_image
from recording_io import (MJPEGWriter, timing_stats, validate_stem, write_json,
                          frame_queue_capacity, RECORDING_PIXEL_BUDGET_BYTES)
from recording_check import check_stream
import record_session


class RecordingContractTests(unittest.TestCase):
    def test_burst_capacity_is_bounded_by_capture_time_and_pixel_memory(self):
        self.assertEqual(frame_queue_capacity(3840, 2160, 25), 21)
        self.assertEqual(frame_queue_capacity(1920, 1080, 30.00003), 31)
        self.assertEqual(frame_queue_capacity(160, 96, 5), 5)
        for width, height, fps in ((3840, 2160, 25), (1920, 1080, 30.00003),
                                   (7680, 4320, 60)):
            capacity = frame_queue_capacity(width, height, fps)
            self.assertLessEqual(capacity * width * height * 3, RECORDING_PIXEL_BUDGET_BYTES)
            self.assertGreaterEqual(capacity, 1)
        for width, height, fps in ((0, 10, 25), (10, -1, 25), (10, 10, 0),
                                   (10, 10, float("nan")), (100000, 100000, 25)):
            with self.assertRaises(ValueError):
                frame_queue_capacity(width, height, fps)

    def test_partial_pipe_progress_cannot_extend_frame_write_timeout_forever(self):
        with tempfile.TemporaryDirectory() as d:
            writer = MJPEGWriter(d, "slow", 160, 96, 25)
            writer.process = Mock()
            writer.process.poll.return_value = None
            writer.process.wait.return_value = 0
            try:
                with patch("recording_io.time.monotonic", side_effect=[0, 0, 6]), \
                     patch("recording_io.select.select", return_value=([], [writer.process.stdin], [])), \
                     patch("recording_io.os.write", return_value=1) as write:
                    with self.assertRaisesRegex(RuntimeError, "ffmpeg input stalled"):
                        writer.write(b"ab", 1)
                write.assert_called_once()
                self.assertEqual(writer.walls, [])
            finally:
                writer.close()

    def test_disk_budget_scales_with_duration_and_camera_modes(self):
        def budget(cameras, duration=10, is_macos=True):
            return record_session._disk_budget_gb(
                cameras, is_macos=is_macos, duration=duration,
                num_episodes=1, episode_time=60)[0]
        cameras = record_session.CAMERAS
        pair = cameras[:2]
        self.assertLess(budget(pair), 20)  # a short pair test fits below the old 30 GB floor
        self.assertGreater(budget(cameras), 20)  # selecting only the pair matters
        self.assertGreater(budget(pair, duration=60), 30)
        self.assertLess(budget(cameras[2:]), budget(cameras[:1]))
        self.assertEqual(budget(pair, duration=None), 30)  # interactive recording keeps its floor
        self.assertEqual(budget(pair, is_macos=False), 30)  # Linux policy unchanged

    def test_zero_frames_never_starts_muxer_or_creates_fake_recording(self):
        with tempfile.TemporaryDirectory() as d, patch("recording_io.subprocess.Popen") as spawn:
            writer = MJPEGWriter(d, "no_frames", 160, 96, 25)
            with self.assertRaisesRegex(RuntimeError, "no frames captured"):
                writer.finalize()
            writer.close()
            spawn.assert_not_called()
            self.assertFalse(writer.out.exists())
            self.assertFalse(writer.ts_path.exists())
            self.assertTrue(writer.journal.closed)
            write_json(Path(d) / "no_frames_meta.json", {"status": "failed", "received_frames": 0,
                                                       "error": "camera timeout"})
            with self.assertRaisesRegex(RuntimeError, "received 0 camera frames; camera timeout"):
                check_stream(d, "no_frames")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_prepared_muxer_with_no_camera_frames_is_cancelled_and_reaped(self):
        with tempfile.TemporaryDirectory() as d:
            writer = MJPEGWriter(d, "camera_failed", 160, 96, 25)
            try:
                writer.prepare()
                process = writer.process
                writer.prepare()
                self.assertIs(writer.process, process)
                self.assertEqual(writer.walls, [])
                with self.assertRaisesRegex(RuntimeError, "no frames captured"):
                    writer.finalize()
                self.assertIsNotNone(process.returncode)
                self.assertTrue(process.stdin.closed)
                self.assertTrue(writer.journal.closed)
                self.assertFalse(writer.out.exists())
                self.assertFalse(writer.ts_path.exists())
                with self.assertRaisesRegex(RuntimeError, "writer is closed"):
                    writer.prepare()
            finally:
                writer.close()

    def test_recorder_exit_reports_actual_log_error_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "take_b0587_recorder.log"
            log.write_text("starting camera\nRECORDING FAILED: frame queue full\n\n")
            message = record_session._recorder_failure("take_b0587", 1, log)
            self.assertIn("RECORDING FAILED: frame queue full", message)
            self.assertIn(str(log), message)
            self.assertNotIn("device busy", message)
            log.unlink()
            self.assertIn("no recorder output", record_session._recorder_failure("take_b0587", 1, log))

    def test_check_keeps_startup_gaps_as_failures_and_reports_later_cadence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "startup.mkv").touch()
            write_json(root / "startup_meta.json", {"capture": {"fps": 25, "width": 160, "height": 96},
                                                   "status": "finalized", "native_drops": 0})
            walls = [100, 100.8] + [100.84 + i / 25 for i in range(100)]
            write_json(root / "startup_ts.json", [{"wall": t} for t in walls])
            with patch("recording_check.probe_video", return_value={"nb_read_packets": len(walls),
                    "width": 160, "height": 96, "codec_name": "mjpeg"}):
                report = check_stream(root, "startup")
            self.assertEqual(report["timing"]["gap_count"], 1)
            self.assertEqual(report["after_first_2s"]["gap_count"], 0)
            self.assertAlmostEqual(report["after_first_2s"]["measured_fps"], 25)
            self.assertTrue(any("timestamp gaps 1" in p for p in report["problems"]))

    def test_missing_camera_has_no_all_camera_overlap(self):
        import recording_check
        def stream(root, stem):
            if stem.endswith("_b0587"):
                raise RuntimeError("received 0 camera frames")
            return {"stem": stem, "capture": {"rotation_degrees": 0},
                    "timing": timing_stats([100 + i / 25 for i in range(300)], 25),
                    "after_first_2s": None, "problems": []}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data/raw").mkdir(parents=True)
            with patch.object(recording_check, "ROOT", root), \
                 patch.object(recording_check, "check_stream", side_effect=stream), \
                 patch.object(sys, "argv", ["recording_check.py", "missing"]), \
                 patch("builtins.print"):
                self.assertEqual(recording_check.main(), 1)
            report = json.loads((root / "data/raw/missing_recording_check.json").read_text())
            self.assertIsNone(report["common_span_s"])
            self.assertFalse(report["transport_passed"])

    def test_rejects_unsafe_stems_and_bad_timestamps(self):
        for stem in ("", "../take", "-take", "x'", "a/b"):
            with self.assertRaises(ValueError):
                validate_stem(stem)
        for walls in ([], [float("nan")], [1, 1], [2, 1]):
            with self.assertRaises(ValueError):
                timing_stats(walls, 25)
        stats = timing_stats([1, 1.04, 1.12], 25)
        self.assertEqual(stats["gap_count"], 1)
        self.assertEqual(stats["estimated_missing_frames"], 1)

    def test_cleanup_waits_for_all_children_and_reports_failure(self):
        first, second, log1, log2 = Mock(), Mock(), Mock(), Mock()
        first.poll.return_value = None
        second.poll.return_value = 1
        first.wait.return_value = 0
        second.wait.return_value = 1
        log1.closed = log2.closed = False
        self.assertEqual(record_session._stop_recorders([("a", first, log1), ("b", second, log2)]), ["b"])
        first.send_signal.assert_called_once()
        second.send_signal.assert_not_called()
        first.wait.assert_called_once()
        second.wait.assert_called_once()
        log1.close.assert_called_once()
        log2.close.assert_called_once()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg")
    def test_segmented_video_preserves_frame_order_pixels_and_capture_gaps(self):
        with tempfile.TemporaryDirectory(prefix="umi recording '") as directory:
            root = Path(directory)
            spec = CameraSpec("b0587b", "test-head", "test", 160, 96, 25, 180)
            writer = MJPEGWriter(root, "test_b0587b", 160, 96, 25, 0.16)
            writer.prepare()
            # 12 asymmetric frames spanning three segments, with one missing
            # source frame. Exactly 12 decoded frames must survive remux.
            walls = [1791600000 + (i + (i >= 5)) / 25 for i in range(12)]
            for i, wall in enumerate(walls):
                image = np.full((96, 160, 3), 20 + i * 10, np.uint8)
                image[:30, :50] = (0, 0, 255)
                canonical = normalize_image(image, 180)
                ok, jpeg = cv2.imencode(".jpg", canonical, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.assertTrue(ok)
                writer.write(jpeg, wall)
            writer.close()
            self.assertGreaterEqual(len(list(root.glob(".*_seg*.mkv"))), 3)
            stats = writer.finalize()
            self.assertEqual(stats["frames"], 12)
            self.assertEqual(stats["gap_count"], 1)
            self.assertEqual([v["wall"] for v in json.loads(writer.ts_path.read_text())], walls)
            cap = cv2.VideoCapture(str(writer.out))
            levels = []
            try:
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    self.assertGreater(float(image[-15:, -25:, 2].mean()), 240)  # physically rotated pixels
                    self.assertLess(float(image[-15:, -25:, 0].mean()), 12)
                    levels.append(float(image[5:15, 5:15].mean()))
            finally:
                cap.release()
            self.assertEqual(len(levels), 12)
            self.assertTrue(all(b > a + 5 for a, b in zip(levels, levels[1:])))
            from dataclasses import asdict
            write_json(root / "test_b0587b_meta.json", {"capture": asdict(spec), "status": "finalized", "native_drops": 0})
            report = check_stream(root, "test_b0587b")
            self.assertTrue(any("timestamp gaps 1" in p for p in report["problems"]))
            self.assertFalse(any("frame count" in p for p in report["problems"]))
            with self.assertRaises(FileExistsError):
                MJPEGWriter(root, "test_b0587b", 160, 96, 25)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_failed_mux_keeps_journal_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            writer = MJPEGWriter(d, "failure", 160, 96, 25)
            _, jpeg = cv2.imencode(".jpg", np.zeros((96, 160, 3), np.uint8))
            writer.write(jpeg, 1)
            writer.process.kill()
            writer.process.wait()
            with self.assertRaises(RuntimeError):
                writer.write(b"not sent", 2)
            with self.assertRaises(RuntimeError):
                writer.finalize()
            self.assertTrue(writer.journal_path.exists())
            self.assertFalse(writer.out.exists())
            with self.assertRaises(FileExistsError):
                MJPEGWriter(d, "failure", 160, 96, 25)


@unittest.skipUnless(sys.platform == "darwin", "native macOS capture")
class NativeRecordingTests(unittest.TestCase):
    def test_startup_burst_survives_writer_pause_with_all_frames_and_original_pts(self):
        import recording_macos as native
        from camera_macos import PreviewFrame
        cap = native.RecordingCapture(CameraSpec("fish", "test", "test", 5, 3, 30))
        entered, release = threading.Event(), threading.Event()
        writer = Mock(walls=[])

        def write(jpeg, wall):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test writer was not released")
            writer.walls.append(wall)

        writer.write.side_effect = write
        consumer = native.RecordingWriter(cap, writer)
        # Reproduce delayed delivery of roughly half a second of source
        # frames during startup. A running writer is briefly busy on frame
        # one; the old eight-frame buffer failed on the tenth arrival.
        source_pts = [100 + i / 30 for i in range(16)]
        try:
            consumer.start()
            cap._publish(PreviewFrame(np.zeros((3, 5, 3), np.uint8), source_pts[0]))
            self.assertTrue(entered.wait(2))
            for pts in source_pts[1:]:
                cap._publish(PreviewFrame(np.zeros((3, 5, 3), np.uint8), pts))
            self.assertGreater(cap.queue_peak, 8)
        finally:
            release.set()
            cap.close()
            consumer.finish()
        self.assertIsNone(consumer.error)
        self.assertEqual(writer.walls, [pts + cap.clock.offset for pts in source_pts])
        self.assertEqual(consumer.stats["frames"], len(source_pts))
        self.assertEqual(cap.received, len(source_pts))
        self.assertFalse(consumer.thread.is_alive())

    def test_writer_drains_frames_before_blocking_camera_start_returns_and_at_shutdown(self):
        import recording_macos as native
        from camera_macos import PreviewFrame
        spec = CameraSpec("b0587", "test", "test", 5, 3, 25)
        consumed = threading.Event()
        handlers = {}
        class StartupCapture(native.RecordingCapture):
            def start(self):
                # FFmpeg initialization must overlap camera startup, not
                # block the writer when the first callback arrives.
                writer.prepare.assert_called_once_with()
                image = np.zeros((3, 5, 3), np.uint8)
                # Simulate callbacks during startRunning(), which has not
                # returned to main. A consumer started afterwards deadlocks.
                for i in range(12):
                    consumed.clear()
                    self._publish(PreviewFrame(image, 100 + i / 25))
                    if not consumed.wait(2):
                        raise RuntimeError("writer did not consume during camera startup")
                handlers[native.signal.SIGTERM](None, None)

            def close(self):
                # Simulate the last accepted delegate frame while stopping.
                if not self.closed:
                    self._publish(PreviewFrame(np.zeros((3, 5, 3), np.uint8), 100.48))
                super().close()

        cap = StartupCapture(spec)
        writer = Mock(walls=[])
        def write(jpeg, wall):
            writer.walls.append(wall)
            consumed.set()
        writer.write.side_effect = write
        def finalize():
            self.assertTrue(cap.closed)
            self.assertEqual(len(writer.walls), 13)
            return timing_stats(writer.walls, 25)
        writer.finalize.side_effect = finalize
        with patch.object(native, "resolve_spec", return_value=spec), \
             patch.object(native, "RecordingCapture", return_value=cap), \
             patch.object(native, "MJPEGWriter", return_value=writer), \
             patch.object(native, "write_json") as metadata, \
             patch.object(native.signal, "signal", side_effect=lambda sig, fn: handlers.update({sig: fn})), \
             patch.dict(native.os.environ, {"PREVIEW": "0"}), patch("builtins.print"):
            self.assertEqual(native.main(["startup_test", "b0587", "5x3", "25"]), 0)
        meta = metadata.call_args.args[1]
        self.assertEqual(meta["status"], "finalized")
        self.assertGreater(meta["startup"]["frames_written_when_start_returned"], 0)
        self.assertEqual(meta["writer_performance"]["frames"], 13)
        self.assertIsNone(cap.error)
        self.assertLess(cap.queue_peak, 8)
        self.assertFalse(any(t.name == "umi.recording.writer" for t in threading.enumerate()))

    def test_writer_failure_is_reported_and_worker_stops(self):
        import recording_macos as native
        cap = native.RecordingCapture(CameraSpec("b0587", "test", "test", 5, 3, 25))
        cap.frames.put(native.RecordedFrame(np.zeros((3, 5, 3), np.uint8), 100))
        writer = Mock()
        writer.write.side_effect = OSError("disk write failed")
        worker = native.RecordingWriter(cap, writer)
        worker.start()
        cap.close()
        worker.finish()
        self.assertIsInstance(worker.error, OSError)
        self.assertEqual(str(worker.error), "disk write failed")
        self.assertEqual(worker.stats["frames"], 0)
        self.assertFalse(worker.thread.is_alive())

    def test_timed_pair_disk_gate_runs_before_opening_cameras(self):
        bindings = [CameraSpec("b0587", "forward-id", "Twin", 3840, 2160, 25),
                    CameraSpec("b0587b", "head-id", "Twin", 3840, 2160, 25, 180),
                    CameraSpec("fish", "wrist-id", "Fish", 1920, 1080, 30)]
        for free_gb in (10, 20):
            with self.subTest(free_gb=free_gb), ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["record_session.py", "--cameras_only",
                    "--dataset", "disk_test", "--duration", "10", "--cameras", "b0587,b0587b"]))
                stack.enter_context(patch("camera_config.load_config", return_value=bindings))
                lookup = stack.enter_context(patch("camera_macos.resolve_device",
                    side_effect=RuntimeError("device lookup reached")))
                spawn = stack.enter_context(patch.object(record_session.subprocess, "Popen"))
                stack.enter_context(patch.object(record_session.sys.stdin, "isatty", return_value=False))
                stack.enter_context(patch.object(record_session.atexit, "register"))
                stack.enter_context(patch("shutil.disk_usage", return_value=Mock(free=free_gb * 1e9)))
                stack.enter_context(patch("builtins.print"))
                if free_gb == 10:
                    with self.assertRaisesRegex(SystemExit, "required budget 18.59 GB"):
                        record_session.main()
                    lookup.assert_not_called()
                else:
                    with self.assertRaisesRegex(RuntimeError, "device lookup reached"):
                        record_session.main()
                    lookup.assert_called_once_with("forward-id")
                spawn.assert_not_called()  # this test never opens a camera

    def test_session_uses_exact_bindings_shared_glove_controls_and_no_linux_ritual(self):
        from camera_config import save_config
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data/raw").mkdir(parents=True)
            bindings = [CameraSpec("b0587", "forward-id", "Twin", 3840, 2160, 25),
                        CameraSpec("b0587b", "head-id", "Twin", 3840, 2160, 25, 180),
                        CameraSpec("fish", "wrist-id", "Fish", 1920, 1080, 30)]
            save_config(root / "calib/cameras.macos.local.json", bindings)
            processes = []

            def start(*args, **kwargs):
                proc = Mock()
                proc.poll.return_value = None
                def wait(**kwargs):
                    proc.poll.return_value = 0
                    return 0
                proc.wait.side_effect = wait
                processes.append(proc)
                return proc

            def device(uid):
                result = Mock()
                result.localizedName.return_value = "Fish" if uid == "wrist-id" else "Twin"
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.object(record_session, "PILOT", root))
                stack.enter_context(patch.object(record_session.sys, "argv", ["record_session.py", "--glove",
                    "--dataset", "trial", "--num_episodes", "1", "--no_preview", "--config",
                    str(root / "calib/cameras.macos.local.json")]))
                stack.enter_context(patch("camera_macos.resolve_device", side_effect=device))
                stack.enter_context(patch.object(record_session.sys.stdin, "isatty", return_value=False))
                stack.enter_context(patch.object(record_session.atexit, "register"))
                stack.enter_context(patch.object(record_session.time, "sleep"))
                stack.enter_context(patch("shutil.disk_usage", return_value=Mock(free=100e9)))
                stack.enter_context(patch("builtins.input", side_effect=["", ""]))
                stack.enter_context(patch("builtins.print"))
                stack.enter_context(patch.object(record_session, "_tail_frame", return_value=object()))
                stack.enter_context(patch.object(record_session, "_frame_health", return_value=(120, 3, 30, 190)))
                stack.enter_context(patch.object(record_session.subprocess, "run", return_value=Mock(stdout="")))
                spawn = stack.enter_context(patch.object(record_session.subprocess, "Popen", side_effect=start))
                ritual = stack.enter_context(patch.dict(sys.modules, {"uvc_ritual": Mock()}))
                record_session.main()
                sys.modules["uvc_ritual"].apply.assert_not_called()
            calls = spawn.call_args_list
            self.assertEqual([c.args[0][3] for c in calls], [s.device_id for s in bindings])
            self.assertTrue(all(c.kwargs["env"]["PREVIEW"] == "0" for c in calls))
            marks = json.loads((root / "data/raw/trial_glove_episodes.json").read_text())["episodes"]
            self.assertEqual(len(marks), 1)
            self.assertGreaterEqual(marks[0]["t_end"], marks[0]["t_start"])
            self.assertTrue(all(p.poll() == 0 for p in processes))

    def test_queue_never_replaces_frames_when_full(self):
        import recording_macos as native
        from camera_macos import PreviewFrame
        cap = native.RecordingCapture(CameraSpec("b0587b", "test", "test", 5, 3, 25, 180))
        image = np.zeros((3, 5, 3), np.uint8)
        capacity = cap.frames.maxsize
        for i in range(capacity):
            cap._publish(PreviewFrame(image, i + 1))
        with self.assertRaisesRegex(RuntimeError, "queue full"):
            cap._publish(PreviewFrame(image, capacity + 1))
        self.assertEqual(cap.frames.qsize(), capacity)
        self.assertEqual([cap.read().wall for _ in range(capacity)],
                         [cap.clock.offset + i + 1 for i in range(capacity)])
        self.assertEqual(cap.queue_peak, capacity)
        self.assertEqual(cap.received, capacity)

    def test_clock_uses_capture_pts_not_callback_arrival(self):
        import recording_macos as native
        import CoreMedia as CM
        cap = native.RecordingCapture(CameraSpec("b0587b", "test", "test", 5, 3, 25, 180))
        cap.session = Mock()
        cap.session.synchronizationClock.return_value = cap.clock.host
        pts = CM.CMTimeMakeWithSeconds(cap.clock.host_seconds() - 0.4, 1_000_000_000)
        with patch.object(CM, "CMSampleBufferGetPresentationTimeStamp", return_value=pts):
            host_pts = cap._timestamp(object())
        self.assertAlmostEqual(CM.CMTimeGetSeconds(host_pts), CM.CMTimeGetSeconds(pts), places=6)
        cap.clock.check()
        with patch.object(native.time, "time", return_value=cap.clock.host_seconds() + cap.clock.offset + 1):
            with self.assertRaisesRegex(RuntimeError, "wall clock changed"):
                cap.clock.check()


if __name__ == "__main__":
    unittest.main()
