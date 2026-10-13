"""MJPEG file transport for macOS. Capture wall clocks remain authoritative.

The container runs at the nominal frame rate; *_ts.json maps every video
frame to its real capture time, including gaps. Never synthesize/duplicate
frames to fill those gaps. Linux's downstream tools use this same index map.
"""
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import time


RECORDING_BURST_SECONDS = 1.0
RECORDING_PIXEL_BUDGET_BYTES = 512 * 1024 * 1024


def frame_queue_capacity(width, height, fps):
    """Absorb up to one second of burst delivery, capped at 512 MiB of BGR.

    This bounds queued pixel storage, not total process memory: the delegate,
    encoder and AVFoundation also own buffers. A full queue remains fatal;
    this budget cannot compensate for sustained encoding below capture rate.
    """
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid frame dimensions or rate")
    frame_bytes = width * height * 3
    memory_capacity = RECORDING_PIXEL_BUDGET_BYTES // frame_bytes
    if memory_capacity < 1:
        raise ValueError("one BGR frame exceeds the recording pixel budget")
    return min(math.ceil(fps * RECORDING_BURST_SECONDS), memory_capacity)


def validate_stem(stem):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stem):
        raise ValueError("dataset/stem must use letters, digits, '.', '_' or '-'")
    return stem


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=codec_name,width,height,nb_read_packets",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"{path}: expected one video stream")
    return streams[0]


def timing_stats(walls, fps):
    if not walls or any(not math.isfinite(t) for t in walls):
        raise ValueError("no valid capture timestamps")
    gaps = [b - a for a, b in zip(walls, walls[1:])]
    if any(dt <= 0 for dt in gaps):
        raise ValueError("capture timestamps must increase strictly")
    return {"frames": len(walls), "first_wall": walls[0], "last_wall": walls[-1],
            "measured_fps": (len(walls) - 1) / (walls[-1] - walls[0]) if gaps else 0,
            "max_gap_s": max(gaps, default=0),
            "gap_count": sum(dt > 1.5 / fps for dt in gaps),
            "estimated_missing_frames": sum(max(0, round(dt * fps) - 1) for dt in gaps)}


class MJPEGWriter:
    """Bounded segments + durable timestamp journal; never overwrite a take.

    JPEG encoding belongs to the capture caller. ffmpeg only muxes those JPEGs,
    preserving order. Finalization verifies frame counts before publishing the
    normal .mkv / _ts.json pair. Failed takes retain segments and the journal.
    """
    def __init__(self, root, stem, width, height, fps, segment_seconds=60):
        self.root, self.stem = Path(root), validate_stem(stem)
        self.width, self.height, self.fps = width, height, fps
        self.root.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(segment_seconds) or segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        self.out = self.root / f"{stem}.mkv"
        self.ts_path = self.root / f"{stem}_ts.json"
        self.journal_path = self.root / f".{stem}_timestamps.jsonl"
        self.part = self.root / f".{stem}_finalizing.mkv"
        if any(p.exists() for p in (self.out, self.ts_path, self.journal_path, self.part)) \
                or list(self.root.glob(f".{stem}_seg*.mkv")):
            raise FileExistsError(f"{stem}: existing recording/unfinished take; use a new dataset name")
        self.walls = []
        self.closed = False
        self.process = None
        self.segment_seconds = segment_seconds
        self.journal = self.journal_path.open("x", buffering=1)

    def prepare(self):
        """Launch the muxer before camera startup, without sending any pixels.

        Process/library initialization must not stall the first capture frame
        while the bounded recording FIFO fills. FFmpeg waits on stdin for the
        first real JPEG; close() cancels it if capture never produces a frame.
        Call before starting the producer/consumer threads.
        """
        if self.closed:
            raise RuntimeError("writer is closed")
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"ffmpeg stopped with code {self.process.returncode}")
            return
        try:
            self.process = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-f", "mjpeg", "-framerate", str(self.fps), "-i", "pipe:0",
                 "-map", "0:v:0", "-c:v", "copy", "-f", "segment",
                 "-segment_time", str(self.segment_seconds), "-reset_timestamps", "1",
                 "-segment_format", "matroska", "-n",
                 str(self.root / f".{self.stem}_seg%05d.mkv")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
        except BaseException:
            if self.process is not None:
                self.process.kill()
                self.process.wait()
                self.process.stdin.close()
                self.process = None
            raise

    def write(self, jpeg, wall):
        if self.closed:
            raise RuntimeError("writer is closed")
        if not math.isfinite(wall) or (self.walls and wall <= self.walls[-1]):
            raise ValueError("capture timestamps must increase strictly")
        self.prepare()
        remaining = memoryview(jpeg).cast("B")
        deadline = time.monotonic() + 5
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select([], [self.process.stdin], [], timeout)[1]:
                raise RuntimeError("ffmpeg input stalled; recording cannot keep up")
            try:
                written = os.write(self.process.stdin.fileno(), remaining)
            except BlockingIOError:
                continue
            remaining = remaining[written:]
        self.journal.write(json.dumps({"wall": wall}) + "\n")
        self.walls.append(wall)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            self.journal.close()
            return
        if not self.walls:
            # No completed frame: cancel rather than send EOF to an empty
            # JPEG demuxer (which would report "dimensions not set"). There
            # is nothing to flush; retain any partial segment/journal.
            try:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
            finally:
                self.process.stdin.close()
                self.journal.close()
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            self.journal.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("ffmpeg did not finish; segments and timestamps preserved")
        if code:
            raise RuntimeError(f"ffmpeg failed ({code}); segments and timestamps preserved")

    def finalize(self):
        self.close()
        if not self.walls:
            raise RuntimeError("no frames captured")
        segments = sorted(self.root.glob(f".{self.stem}_seg*.mkv"))
        count = sum(int(probe_video(p)["nb_read_packets"]) for p in segments)
        if count != len(self.walls):
            raise RuntimeError(f"{count} encoded frames != {len(self.walls)} timestamps; take preserved")
        listing = self.root / f".{self.stem}_concat.txt"
        # Relative validated stems avoid ffconcat quoting problems in user paths.
        listing.write_text("".join(f"file '{p.name}'\n" for p in segments))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat",
                        "-safe", "0", "-i", str(listing), "-c", "copy",
                        "-avoid_negative_ts", "make_zero", "-n", str(self.part)], check=True)
        info = probe_video(self.part)
        if (info["codec_name"], info["width"], info["height"], int(info["nb_read_packets"])) \
                != ("mjpeg", self.width, self.height, len(self.walls)):
            raise RuntimeError(f"final video does not match capture: {info}; take preserved")
        write_json(self.ts_path, [{"wall": t} for t in self.walls])
        self.part.rename(self.out)
        for p in [*segments, listing, self.journal_path]:
            p.unlink()
        return timing_stats(self.walls, self.fps)
