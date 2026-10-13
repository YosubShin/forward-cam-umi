"""AVFoundation identity, native capture, and latest-frame preview delivery.

Native imports live here so Linux tools do not acquire macOS dependencies.
PreviewCapture intentionally drops superseded frames: it is NOT a recorder.
The session recorder must separately validate timestamps, buffering and codec.
"""
from dataclasses import dataclass
import math
import queue
import threading

import AVFoundation as AV
import CoreMedia as CM
import Foundation
import Quartz as CV
import libdispatch
import numpy as np
import objc
import cv2

from camera_config import normalize_image, supported_frame_rate


def devices():
    # Include video devices only, not microphones. IDs, never enumeration
    # indices or names, are the binding key. No name-based fallback on replug.
    return list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))


def device_info(device):
    modes = set()
    for fmt in device.formats():
        size = CM.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
        for rate in fmt.videoSupportedFrameRateRanges():
            modes.add((size.width, size.height, float(rate.minFrameRate()),
                       float(rate.maxFrameRate())))
    return {"device_id": str(device.uniqueID()), "name": str(device.localizedName()),
            "modes": sorted(modes)}


def resolve_device(device_id):
    matches = [d for d in devices() if str(d.uniqueID()) == device_id]
    if len(matches) != 1:
        raise RuntimeError(f"device ID {device_id!r} is missing or ambiguous; "
                           "rerun camera setup (no name/index fallback)")
    return matches[0]


def request_camera_access():
    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == AV.AVAuthorizationStatusNotDetermined:
        done = threading.Event()
        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AV.AVMediaTypeVideo, lambda granted: done.set())
        if not done.wait(30):
            raise RuntimeError("camera permission prompt is pending; allow access and rerun")
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status != AV.AVAuthorizationStatusAuthorized:
        raise RuntimeError("camera access unavailable; check macOS Privacy & Security > Camera")


@dataclass(frozen=True)
class PreviewFrame:
    image: np.ndarray
    presentation_seconds: float  # AVFoundation clock; NOT Unix wall time


class _FrameSink(Foundation.NSObject,
                 protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        owner = self.owner
        if owner is None or owner.closed or owner.error is not None:
            return
        try:
            with objc.autorelease_pool():
                pixel = CM.CMSampleBufferGetImageBuffer(sample)
                if pixel is None:
                    raise RuntimeError("AVFoundation delivered no pixel buffer")
                if CV.CVPixelBufferLockBaseAddress(pixel, CV.kCVPixelBufferLock_ReadOnly) != 0:
                    raise RuntimeError("cannot lock AVFoundation pixel buffer")
                try:
                    width = CV.CVPixelBufferGetWidth(pixel)
                    height = CV.CVPixelBufferGetHeight(pixel)
                    stride = CV.CVPixelBufferGetBytesPerRow(pixel)
                    if (width, height) != (owner.spec.width, owner.spec.height):
                        raise RuntimeError(f"capture negotiated {width}x{height}, "
                                           f"expected {owner.spec.width}x{owner.spec.height}")
                    if CV.CVPixelBufferGetPixelFormatType(pixel) != CV.kCVPixelFormatType_32BGRA:
                        raise RuntimeError("capture did not deliver requested BGRA pixels")
                    base = CV.CVPixelBufferGetBaseAddress(pixel)
                    raw = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8)
                    # Honor row padding; copy before unlocking the native buffer.
                    bgra = raw.reshape(height, stride)[:, :width * 4].reshape(height, width, 4)
                    # OpenCV's SIMD conversion makes an owned BGR buffer;
                    # NumPy's strided channel-copy is too slow for 4K25.
                    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                finally:
                    CV.CVPixelBufferUnlockBaseAddress(pixel, CV.kCVPixelBufferLock_ReadOnly)
                pts = CM.CMTimeGetSeconds(owner._timestamp(sample))
                if not math.isfinite(pts):
                    raise RuntimeError("invalid sample presentation timestamp")
                frame = PreviewFrame(normalize_image(bgr, owner.spec.rotation_degrees), pts)
                owner._publish(frame)
        except Exception as exc:
            owner.error = exc  # Report on the consumer thread, never escape ObjC callback.

    def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sample, connection):
        if self.owner is not None and not self.owner.closed:
            self.owner.native_drops += 1


class NativeCapture:
    """Exact device/mode and owned, canonical BGR buffers; subclass owns delivery.

Native pixel buffers live only inside the delegate. Delivered arrays own their
memory. Capture callbacks use one serial native queue; consumers own encoding
or preview independently.
    """
    def __init__(self, spec):
        self.spec = spec
        self.native_drops = 0
        self.error = None
        self.closed = False
        self.session = None
        self.output = None
        self.sink = None
        self.callback_queue = None
        self.negotiated_fps = None

    def _timestamp(self, sample):
        return CM.CMSampleBufferGetPresentationTimeStamp(sample)

    def _publish(self, frame):
        raise NotImplementedError

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def start(self):
        request_camera_access()
        device = resolve_device(self.spec.device_id)
        if str(device.localizedName()) != self.spec.name:
            raise RuntimeError("device name changed for saved ID; rerun camera setup")
        matching = []
        for fmt in device.formats():
            size = CM.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            if (size.width, size.height) == (self.spec.width, self.spec.height):
                for rate in fmt.videoSupportedFrameRateRanges():
                    actual = supported_frame_rate(self.spec.fps, rate.minFrameRate(), rate.maxFrameRate())
                    if actual is None:
                        continue
                    if actual == rate.minFrameRate():
                        duration = rate.maxFrameDuration()
                    elif actual == rate.maxFrameRate():
                        duration = rate.minFrameDuration()
                    else:
                        duration = CM.CMTimeMakeWithSeconds(1.0 / actual, 1_000_000_000)
                    matching.append((fmt, duration, actual))
        if not matching:
            raise RuntimeError(f"unsupported mode {self.spec.width}x{self.spec.height} "
                               f"@ {self.spec.fps}; use --list --formats")
        capture_input, error = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if capture_input is None:
            raise RuntimeError(f"cannot open camera: {error}")
        self.session = AV.AVCaptureSession.alloc().init()
        self.session.beginConfiguration()
        try:
            if not self.session.canAddInput_(capture_input):
                raise RuntimeError("cannot add camera input")
            self.session.addInput_(capture_input)
            self.output = AV.AVCaptureVideoDataOutput.alloc().init()
            # On macOS the output dimensions are negotiated separately from
            # activeFormat. A pixel-format-only dictionary can keep the
            # session's default 1080p output even with a 4K source selected.
            self.output.setVideoSettings_({
                CV.kCVPixelBufferPixelFormatTypeKey: CV.kCVPixelFormatType_32BGRA,
                CV.kCVPixelBufferWidthKey: self.spec.width,
                CV.kCVPixelBufferHeightKey: self.spec.height,
            })
            self.output.setAlwaysDiscardsLateVideoFrames_(True)
            self.sink = _FrameSink.alloc().init()
            self.sink.owner = self
            self.callback_queue = libdispatch.dispatch_queue_create(b"umi.camera.capture", None)
            self.output.setSampleBufferDelegate_queue_(self.sink, self.callback_queue)
            if not self.session.canAddOutput_(self.output):
                raise RuntimeError("cannot add video output")
            self.session.addOutput_(self.output)
            connection = self.output.connectionWithMediaType_(AV.AVMediaTypeVideo)
            if connection.isVideoMirroringSupported():
                connection.setAutomaticallyAdjustsVideoMirroring_(False)
                connection.setVideoMirrored_(False)
            # Configure the device directly, after attaching both input/output.
            # InputPriority is not a settable preset on macOS. Explicitly
            # selecting it raises NSInvalidArgumentException even when this
            # camera supports the requested format. Do not fall back to a
            # quality preset that can silently change calibrated dimensions.
            locked, error = device.lockForConfiguration_(None)
            if not locked:
                raise RuntimeError(f"cannot configure camera: {error}")
            try:
                fmt, duration, self.negotiated_fps = matching[0]
                device.setActiveFormat_(fmt)
                device.setActiveVideoMinFrameDuration_(duration)
                device.setActiveVideoMaxFrameDuration_(duration)
            finally:
                device.unlockForConfiguration()
        finally:
            self.session.commitConfiguration()
        self._check_source_dimensions(device, "after configuration")
        self.session.startRunning()
        self._check_source_dimensions(device, "after startup")

    def _check_source_dimensions(self, device, stage):
        # Checking output buffers alone could accept scaled-up frames. Require
        # the device source itself to retain the requested native dimensions.
        size = CM.CMVideoFormatDescriptionGetDimensions(device.activeFormat().formatDescription())
        if (size.width, size.height) != (self.spec.width, self.spec.height):
            raise RuntimeError(f"camera source {stage} is {size.width}x{size.height}, "
                               f"expected {self.spec.width}x{self.spec.height}; "
                               "refusing scaled or mismatched capture")

    def read(self, timeout=10):
        if self.closed:
            raise RuntimeError("capture is closed")
        if self.error is not None:
            raise RuntimeError(f"capture failed: {self.error}") from self.error
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty as exc:
            if self.error is not None:
                raise RuntimeError(f"capture failed: {self.error}") from self.error
            raise RuntimeError(f"no camera frame received at {self.spec.width}x{self.spec.height} "
                               f"@ {self.spec.fps:g} fps; close other camera apps, "
                               "check the USB connection, and retry") from exc

    def close(self):
        self.closed = True
        if self.output is not None:
            self.output.setSampleBufferDelegate_queue_(None, None)
        if self.session is not None:
            self.session.stopRunning()
        if self.callback_queue is not None:
            # close() runs on the consumer thread, never in the delegate.
            # Drain an in-flight callback before releasing it or its owner.
            libdispatch.dispatch_sync(self.callback_queue, lambda: None)
        if self.sink is not None:
            self.sink.owner = None
        self.sink = self.output = self.session = self.callback_queue = None

    def __exit__(self, *exc):
        self.close()


class PreviewCapture(NativeCapture):
    """Inspection only: discard superseded frames instead of building latency."""
    def __init__(self, spec):
        super().__init__(spec)
        self.frames = queue.Queue(maxsize=1)

    def _publish(self, frame):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(frame)


class OpenCVPreview:
    """Small read/release adapter for the shared Linux-style diagnostic loop.

    Frames are already normalized by PreviewCapture. Its latest-frame queue
    supports previews and still calibration views, not continuous recording
    or timestamped dataset collection.
    """
    def __init__(self, spec):
        self.capture = PreviewCapture(spec)
        self.capture.__enter__()

    def read(self):
        return True, self.capture.read().image

    def release(self):
        self.capture.close()
