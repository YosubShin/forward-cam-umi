"""Native macOS capture behind uvc_record's existing command-line interface.

Image orientation is identical to calibrate_uvc. Capture PTS -> host clock ->
Unix wall time, never encoder/preview arrival time. No camera-index fallback.
"""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import cv2
import CoreMedia as CM
import objc

from camera_config import DEFAULT_CONFIG, resolve_spec
from camera_macos import NativeCapture
from preview_sink import open_sink, close_sink
from recording_io import (MJPEGWriter, write_json, frame_queue_capacity,
                          RECORDING_BURST_SECONDS, RECORDING_PIXEL_BUDGET_BYTES)


class CaptureClock:
    """Bridge CoreMedia host time and time.time() used by episode marks.

    Apple: AVCaptureSession.synchronizationClock is the sample PTS timebase;
    CMSyncConvertTime compensates clock rate differences. No optical sync
    guarantee: independent UVC sensors still expose at independent instants.
    """
    def __init__(self):
        self.host = CM.CMClockGetHostTimeClock()
        candidates = []
        for _ in range(10):
            before = self.host_seconds()
            wall = time.time()
            after = self.host_seconds()
            candidates.append((after - before, wall - (before + after) / 2))
        uncertainty, self.offset = min(candidates)
        self.metadata = {"method": "session PTS -> CoreMedia host clock -> Unix anchor",
                         "wall_minus_host_s": self.offset,
                         "anchor_uncertainty_s": uncertainty / 2,
                         "optical_sync_validated": False}

    def host_seconds(self):
        return CM.CMTimeGetSeconds(CM.CMClockGetTime(self.host))

    def check(self):
        if abs(time.time() - (self.host_seconds() + self.offset)) > 0.020:
            raise RuntimeError("system wall clock changed during capture; episode timing is unsafe")


@dataclass(frozen=True)
class RecordedFrame:
    image: object
    wall: float
    arrival: float | None = None


class RecordingCapture(NativeCapture):
    """Keep every delivered frame in order; overflow is a fatal error.

    Native delivery can release several previously captured frames together,
    including when another camera starts. Budget one second of queued BGR
    frames, bounded by pixel memory, rather than treating eight frames as a
    universal burst limit. Timestamp gaps and native drops remain visible.
    """
    def __init__(self, spec):
        super().__init__(spec)
        self.frames = queue.Queue(maxsize=frame_queue_capacity(spec.width, spec.height, spec.fps))
        self.clock = CaptureClock()
        self.received = 0
        self.queue_peak = 0
        self.first_arrival = self.last_arrival = None

    def _timestamp(self, sample):
        clock = self.session.synchronizationClock()
        if clock is None:
            raise RuntimeError("capture synchronization clock is unavailable")
        pts = CM.CMSampleBufferGetPresentationTimeStamp(sample)
        return CM.CMSyncConvertTime(pts, clock, self.clock.host)

    def _publish(self, frame):
        arrival = time.monotonic()
        wall = frame.presentation_seconds + self.clock.offset
        if not math.isfinite(wall):
            raise RuntimeError("invalid host timestamp")
        try:
            self.frames.put_nowait(RecordedFrame(frame.image, wall, arrival))
        except queue.Full as exc:
            raise RuntimeError(f"recording cannot keep up: frame queue full "
                               f"({self.frames.maxsize} frames; bounded burst/pixel budget); "
                               "take is incomplete") from exc
        self.received += 1
        self.queue_peak = max(self.queue_peak, self.frames.qsize())
        if self.first_arrival is None:
            self.first_arrival = arrival
        self.last_arrival = arrival


class RecordingWriter:
    """Drain capture on a dedicated thread, including during startRunning().

    AVFoundation can deliver callbacks before its blocking startup returns.
    The consumer must already be running then; the burst buffer cannot replace
    correct startup ordering. Only this thread writes JPEGs. The caller stops
    capture, joins this writer, then remuxes its output.
    """
    def __init__(self, capture, writer, preview=None):
        self.capture, self.writer, self.preview = capture, writer, preview
        self.input_closed = threading.Event()
        self.error = None
        self.stats = {"frames": 0, "first_encode_ms": None, "first_mux_write_ms": None,
                      "max_encode_ms": 0.0, "max_mux_write_ms": 0.0,
                      "max_queue_wait_ms": 0.0,
                      "total_encode_ms": 0.0, "total_mux_write_ms": 0.0}
        self.thread = threading.Thread(target=self._run, name="umi.recording.writer")

    def start(self):
        self.thread.start()

    def _run(self):
        last_preview = 0
        try:
            while not self.input_closed.is_set() or not self.capture.frames.empty():
                try:
                    frame = self.capture.frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                before = time.monotonic()
                if frame.arrival is not None:
                    self.stats["max_queue_wait_ms"] = max(
                        self.stats["max_queue_wait_ms"], (before - frame.arrival) * 1000)
                ok, jpeg = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("JPEG encoder failed")
                encoded = time.monotonic()
                self.writer.write(jpeg, frame.wall)
                after = time.monotonic()
                encode_ms, mux_ms = (encoded - before) * 1000, (after - encoded) * 1000
                if not self.stats["frames"]:
                    self.stats.update(first_encode_ms=encode_ms, first_mux_write_ms=mux_ms)
                self.stats["frames"] += 1
                self.stats["max_encode_ms"] = max(self.stats["max_encode_ms"], encode_ms)
                self.stats["max_mux_write_ms"] = max(self.stats["max_mux_write_ms"], mux_ms)
                self.stats["total_encode_ms"] += encode_ms
                self.stats["total_mux_write_ms"] += mux_ms
                if self.preview is not None and after - last_preview >= 0.12:
                    self.preview.publish(frame.image)
                    last_preview = after
        except Exception as exc:
            self.error = exc

    def finish(self):
        # Call only after capture.close() has drained the delegate queue: no
        # producer may append between the empty check and thread shutdown.
        self.input_closed.set()
        if self.thread.ident is not None:
            self.thread.join()


class RecorderPreview:
    """Optional external player; this latest-frame queue cannot block recording."""
    def __init__(self, name):
        self.name = name
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.sink = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def publish(self, image):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(image)

    def _run(self):
        dictionary = (cv2.aruco.DICT_APRILTAG_16h5 if self.name.endswith("_fish")
                      else cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
        try:
            while not self.stop.is_set():
                try:
                    image = self.frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                h, w = image.shape[:2]
                image = cv2.resize(image, (960, round(h * 960 / w)))
                if os.environ.get("OVERLAY", "1") == "1":
                    corners, ids, _ = detector.detectMarkers(image)
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(image, corners, ids)
                rotation = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                            "180": cv2.ROTATE_180}.get(os.environ.get("ROTATE"))
                if rotation is not None:
                    image = cv2.rotate(image, rotation)  # additional PREVIEW-only rotation
                if self.sink is None:
                    self.sink = open_sink(self.name, image.shape[1], image.shape[0])
                if self.sink is None or self.stop.is_set():
                    break
                self.sink.stdin.write(image.tobytes())
                self.sink.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # closing the viewer must not end recording
        finally:
            close_sink(self.sink)

    def close(self):
        self.stop.set()
        # Terminate first to unblock a stalled pipe without touching camera I/O.
        if self.sink is not None and self.sink.poll() is None:
            self.sink.terminate()
        self.thread.join(timeout=3)


def main(argv=None):
    # Each camera already has its own process. Eight OpenCV workers per
    # process oversubscribe an M-series laptop when all cameras are running.
    cv2.setNumThreads(1)
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        raise SystemExit("macOS: uvc_record.py <stem> <saved device ID or role> <WxH> <fps>")
    name, selector, resolution, fps = args
    spec = resolve_spec(selector, os.environ.get("UMI_CAMERA_CONFIG", DEFAULT_CONFIG))
    if resolution != f"{spec.width}x{spec.height}" or not math.isclose(float(fps), spec.fps, rel_tol=1e-5):
        raise SystemExit("recording mode differs from saved camera setup; refusing mismatch")
    root = Path("data/raw")
    capture = RecordingCapture(spec)
    writer = MJPEGWriter(root, name, spec.width, spec.height, spec.fps,
                         float(os.environ.get("SEG_SECS", "60")))
    meta_path = root / f"{name}_meta.json"
    meta = {"device": spec.device_id, "backend": "avfoundation", "capture": asdict(spec),
            "codec": "mjpeg", "jpeg_quality": 95, "source_passthrough": False,
            "orientation": "canonical pixels; rotation applied before encoding",
            "container_timing": "nominal fps; actual per-frame capture time is in _ts.json",
            "clock": capture.clock.metadata, "exposure_gain_controlled": False,
            "controls": ["Exposure/gain not controlled on macOS"], "status": "recording",
            "buffering": {"burst_seconds": RECORDING_BURST_SECONDS,
                          "pixel_budget_bytes": RECORDING_PIXEL_BUDGET_BYTES,
                          "capacity_frames": capture.frames.maxsize,
                          "max_queued_pixel_bytes": (capture.frames.maxsize * spec.width
                                                     * spec.height * 3)}}
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    preview = None
    consumer = None
    error = None
    last_report = 0

    print(f"{spec.role}: {spec.device_id} -> {name}.mkv; MJPEG quality 95 re-encode, "
          f"rotation {spec.rotation_degrees} degrees before encoding", flush=True)
    print("Exposure/gain uncontrolled. Stop with Ctrl-C; wait for finalization.", flush=True)
    try:
        write_json(meta_path, meta)
        # Let the muxer initialize while AVFoundation starts, rather than
        # cold-starting a subprocess on the time-critical first frame.
        writer.prepare()
        if os.environ.get("PREVIEW", "0") == "1":
            preview = RecorderPreview(name)
        consumer = RecordingWriter(capture, writer, preview)
        consumer.start()
        start = time.monotonic()
        with objc.autorelease_pool():
            capture.__enter__()
        started = time.monotonic()
        meta["startup"] = {"capture_start_s": started - start,
                           "frames_received_when_start_returned": capture.received,
                           "frames_written_when_start_returned": consumer.stats["frames"]}
        print(f"capture start returned after {started-start:.3f}s; "
              f"received {capture.received}, written {consumer.stats['frames']} frames", flush=True)
        while not stop.wait(0.05):
            if capture.error is not None:
                raise RuntimeError(f"capture failed: {capture.error}")
            if consumer.error is not None:
                raise RuntimeError(f"recording writer failed: {consumer.error}")
            now = time.monotonic()
            if now - (capture.last_arrival if capture.last_arrival is not None else started) > 10:
                raise RuntimeError(f"no camera frame received for 10s at {spec.width}x{spec.height} "
                                   f"@ {spec.fps:g} fps; check other camera apps and USB connection")
            if now - last_report >= 1:
                capture.clock.check()
                print(f"{len(writer.walls)} frames; queue {capture.frames.qsize()}/{capture.frames.maxsize}; "
                      f"native drops {capture.native_drops}", flush=True)
                last_report = now
    except Exception as exc:
        error = str(exc)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        capture.close()
        if consumer is not None:
            consumer.finish()  # Drain every accepted frame before finalization.
            error = error or (str(consumer.error) if consumer.error is not None else None)
            meta["writer_performance"] = consumer.stats
        if preview is not None:
            preview.close()
            preview = None
        try:
            if capture.error is not None:
                error = error or str(capture.error)
            print("finalizing video and checking frame/timestamp counts…", flush=True)
            meta["timing"] = writer.finalize()
            meta["status"] = "incomplete" if error else "finalized"
        except Exception as exc:
            error = error or str(exc)
            meta["status"] = "failed"
        finally:
            try:
                writer.close()
            except Exception as exc:
                error = error or str(exc)
                meta["status"] = "failed"
        meta.update(native_drops=capture.native_drops, queue_peak=capture.queue_peak,
                    error=error, received_frames=capture.received)
        arrival_span = ((capture.last_arrival - capture.first_arrival)
                        if capture.first_arrival is not None else 0)
        meta["callback_timing"] = {"span_s": arrival_span,
                                   "measured_fps": ((capture.received - 1) / arrival_span
                                                    if arrival_span > 0 else None)}
        write_json(meta_path, meta)
    if error:
        print(f"RECORDING FAILED: {error}; inspect {meta_path}", flush=True)
        return 1
    print(f"saved {len(writer.walls)} frames; {meta['timing']}; native drops {capture.native_drops}", flush=True)
    return 0
