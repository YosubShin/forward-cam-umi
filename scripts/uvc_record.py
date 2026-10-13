#!/usr/bin/env python3
"""Record a UVC camera's OWN MJPEG stream untouched (no re-encode) + wall-clock
per-frame timestamps, via ffmpeg stream copy.

The previous version decoded and re-encoded with mp4v (inter-frame), which
crushed moving regions at default bitrate — detection collapsed above
~0.25 m/s while static frames stayed pristine. Stream copy preserves the
camera's per-frame MJPEG exactly as the live viewer sees it.

Writes data/raw/<name>.avi (MJPEG passthrough) + <name>_ts.json.
Usage: python scripts/uvc_record.py <name> [dev] [WxH] [fps]
  name  output stem: data/raw/<name>.mkv + <name>_ts.json (+ _meta.json)
  dev   V4L2 node or /dev/v4l/by-path link   (default /dev/video0)
  WxH   capture size, e.g. 3840x2160          (default 1920x1080)
  fps   capture rate the camera advertises     (default 30)
Normally launched by record_session.py, one process per camera; run it by
hand only to test a single camera. Stop with Ctrl-C (or SIGTERM) and WAIT
for the remux to finish — a 4K take needs about a minute.
PREVIEW=1 additionally opens a live window (downscaled, ~12fps): the same
ffmpeg tees a decoded rawvideo branch over stdout; this process overlays
marker detections (headless cv2 draws fine, it just can't display) and
pipes frames to ffplay for the window (env cv2 is GUI-less and ffmpeg's own
sdl2 device renders black on this box). OVERLAY=0 disables the marker
overlay. The recording stays an untouched stream copy either way, and a
dying/closed viewer never stops the recording.
Ctrl-C stops cleanly.

On macOS the same entry point uses saved AVFoundation IDs and canonical
image rotation. It re-encodes high-quality MJPEG (not sensor passthrough);
actual capture times are in _ts.json. See scripts/README.md for limitations.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

if len(sys.argv) < 2 or any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip())
    sys.exit(0 if len(sys.argv) >= 2 else 2)
if sys.platform == "darwin":
    from recording_macos import main as record_macos
    sys.exit(record_macos())
name = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "/dev/video0"
RES = sys.argv[3] if len(sys.argv) > 3 else "1920x1080"
FPS_IN = sys.argv[4] if len(sys.argv) > 4 else "30"
out = f"data/raw/{name}.mkv"
# epoch-PTS temp is written as bounded SEGMENTS (finalized on rollover, so a
# crash loses <=1 segment, and a session janitor can prune idle footage);
# survivors are concatenated + remuxed to the zero-based deliverable at stop
SEG_SECS = int(os.environ.get("SEG_SECS", "60"))
segpat = f"data/raw/.{name}_seg%05d.mkv"

# -use_wallclock_as_timestamps: PTS = wall clock at capture -> per-frame times
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info",
       "-f", "v4l2", "-input_format", "mjpeg",
       "-video_size", RES, "-framerate", FPS_IN,
       "-use_wallclock_as_timestamps", "1",
       "-i", dev, "-map", "0:v", "-c", "copy", "-copyts",
       "-f", "segment", "-segment_time", str(SEG_SECS),
       "-segment_atclocktime", "1",  # epoch PTS breaks the default boundary
       "-segment_format", "matroska", "-reset_timestamps", "0",  # logic
       "-y", segpat]
PREVIEW = os.environ.get("PREVIEW", "0") == "1"
ZOOM = os.environ.get("ZOOM", "0") == "1"        # EEF-tracking zoom window
ZOOM_CROP = int(os.environ.get("ZOOM_CROP", "1200"))  # native px, square
if PREVIEW:
    W_IN, H_IN = (int(x) for x in RES.split("x"))
    # pipe at HALF-res for 4K cameras: full-res raw (300 MB/s/cam) plus
    # inline processing backpressured ffmpeg and lagged both the preview
    # and the capture. Half-res still gives the zoom 2x the main window's
    # detail, and detection runs on it directly.
    SSCALE = 2 if W_IN >= 2000 else 1     # sensor px per piped px
    PW_P, PH_P = W_IN // SSCALE, H_IN // SSCALE
    cmd += ["-map", "0:v", "-vf", f"scale={PW_P}:{PH_P},fps=12",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
# snapshot camera controls into a sidecar — captures are self-documenting,
# and a loud warning fires if auto-exposure is on (settings reset on replug)
ctl = subprocess.run(
    ["v4l2-ctl", "-d", dev, "-C",
     "auto_exposure,exposure_time_absolute,gain,sharpness,focus_absolute,"
     "focus_automatic_continuous,white_balance_automatic"],
    capture_output=True, text=True).stdout
json.dump({"device": dev, "controls": ctl.strip().splitlines()},
          open(f"data/raw/{name}_meta.json", "w"), indent=1)
print(ctl.strip())
if "auto_exposure: 3" in ctl or "focus_automatic_continuous: 1" in ctl:
    print("*** WARNING: AUTO exposure/focus is ON — the lock ritual was lost "
          "(replug resets controls). Re-apply v4l2-ctl before recording. ***")

signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
print(f"{dev} -> {out} (MJPEG stream copy)   Ctrl-C to stop", flush=True)
p = subprocess.Popen(cmd, stderr=subprocess.PIPE,
                     stdout=subprocess.PIPE if PREVIEW else None, text=False)

# A busy device (a recorder from an earlier run still holds it) makes ffmpeg
# exit within a second. That MUST be fatal here: otherwise this wrapper
# carries on, the pre-flight reads the other process's segments, and at
# stop time we remux footage we never captured (2026-10-06).
import time as _t0
_t0.sleep(2.0)
if p.poll() is not None:
    err = p.stderr.read().decode(errors="replace")[-600:] if p.stderr else ""
    print(f"!! ffmpeg exited immediately (code {p.returncode}) on {dev} — "
          f"device busy or wrong node?\n{err}", flush=True)
    sys.exit(3)

def _late_ritual():
    import time as _t
    _t.sleep(1.5)  # stream must be running before controls stick (ISP latch)
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from uvc_ritual import apply as _apply
    _apply(dev)

import threading
threading.Thread(target=_late_ritual, daemon=True).start()


def _pump_stderr():
    while True:
        chunk = p.stderr.read(4096)
        if not chunk:
            break
        ms = re.findall(rb"frame=\s*(\d+)", chunk)
        if ms:
            print(f"\r{ms[-1].decode()} frames", end="", flush=True)


try:
    if PREVIEW:
        threading.Thread(target=_pump_stderr, daemon=True).start()
        # cv2 here is HEADLESS (detection + drawing only); display goes
        # through ffplay — env cv2 has no GUI and ffmpeg's own sdl2 output
        # renders black on this box
        import cv2
        import numpy as np
        overlay = os.environ.get("OVERLAY", "1") == "1"
        if overlay:
            # fast preview-res detection (no APRILTAG refinement) — a visual
            # aid only; offline stats come from the full-res pipeline
            # the wrist fisheye sees the AprilTag 16h5 tip markers, every
            # other stream the ArUco 4x4 ball/tail/base set — count the
            # family that stream is supposed to see, or the overlay reads
            # "0 markers" on a perfectly healthy wrist stream
            _dic = (cv2.aruco.DICT_APRILTAG_16h5 if "fish" in name
                    else cv2.aruco.DICT_4X4_50)
            adet = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(_dic),
                cv2.aruco.DetectorParameters())
        # ROTATE=cw|ccw|180 rotates the PREVIEW only (portrait-mounted
        # camera); the recording stays in sensor orientation, which the
        # offline pipeline doesn't care about
        ROT = {"cw": cv2.ROTATE_90_CLOCKWISE,
               "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
               "180": cv2.ROTATE_180}.get(os.environ.get("ROTATE", ""))
        FW, FH = (PH_P, PW_P) if ROT in (cv2.ROTATE_90_CLOCKWISE,
                                         cv2.ROTATE_90_COUNTERCLOCKWISE) \
            else (PW_P, PH_P)                     # pipe-res after rotation
        DW = 960
        DH = (DW * FH // FW) // 2 * 2             # main window size
        CROP = min(ZOOM_CROP // SSCALE, FW, FH)   # zoom crop, pipe px
        ZD = CROP                                 # shown 1:1

        def _sink(title, w, h):
            # mpv first: it renders via XVideo, immune to the broken-GLX
            # state this box gets after NVIDIA driver auto-updates (ffplay's
            # SDL insists on a GL context and dies instantly there, in every
            # renderer mode). Declared fps 60 >> real ~12fps supply so the
            # consumer always outruns production — no standing lag.
            cmds = [
                ["mpv", "--no-config", "--really-quiet", f"--title={title}",
                 # live pipe: no cache, no readahead, render on arrival —
                 # mpv's default demuxer cache buffered seconds of feed
                 "--profile=low-latency", "--no-cache", "--untimed",
                 "--demuxer-readahead-secs=0",
                 "--demuxer=rawvideo", "--demuxer-rawvideo-mp-format=bgr24",
                 f"--demuxer-rawvideo-w={w}", f"--demuxer-rawvideo-h={h}",
                 "--demuxer-rawvideo-fps=60", "-"],
                ["ffplay", "-loglevel", "error", "-window_title", title,
                 "-fflags", "nobuffer", "-flags", "low_delay",
                 "-f", "rawvideo", "-pixel_format", "bgr24",
                 "-video_size", f"{w}x{h}", "-framerate", "60",
                 "-i", "pipe:0"],
            ]
            for cmd in cmds:
                try:
                    return subprocess.Popen(
                        cmd, stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:
                    continue
            print(f"\n(no preview window for {title})", flush=True)
            return None

        sink = _sink(name, DW, DH)
        zsink = _sink(f"{name} EEF zoom", ZD, ZD) if ZOOM else None

        # zoom-center tracking: project the fixed body-frame target point
        # (between the gripper tips, calib/eef_target.json) through the
        # camera using a live dodeca pose. Falls back to the marker-corner
        # centroid when calibration files are missing or the pose fails.
        track = None
        if zsink is not None:
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import labels_lib as LL
                Kc, distc = LL.load_intrinsics(
                    os.environ.get("ZOOM_INTR", "calib/B0587_intrinsics.json"))
                Tb, dobj_b = LL.load_bundle("calib/dodeca_bundle_ideal.json")
                tgt = (np.array(json.load(open("calib/eef_target.json"))
                                ["p_body_mm"]).reshape(1, 3) / 1000.0)
                track = (Kc, distc, Tb, dobj_b, tgt)
                print("(zoom: tracking the between-tips target via dodeca "
                      "pose)", flush=True)
            except Exception as e:
                print(f"(zoom: marker-centroid fallback — {e})", flush=True)
        last_rt = None

        def rot_pt(x, y):
            """pipe coords -> rotated-preview coords"""
            if ROT == cv2.ROTATE_90_CLOCKWISE:
                return PH_P - 1 - y, x
            if ROT == cv2.ROTATE_90_COUNTERCLOCKWISE:
                return y, PW_P - 1 - x
            if ROT == cv2.ROTATE_180:
                return PW_P - 1 - x, PH_P - 1 - y
            return x, y

        # drain thread: the pipe is ALWAYS read at full speed into a
        # latest-frame slot; processing works on the freshest frame only.
        # Under CPU pressure frames DROP instead of queuing — the preview
        # can never lag behind, and ffmpeg (and the capture) is never
        # backpressured by slow processing.
        nbytes = PW_P * PH_P * 3
        latest = [None, 0, False]                 # buf, seq, eof

        def _drain():
            while True:
                b = p.stdout.read(nbytes)
                if not b or len(b) < nbytes:
                    latest[2] = True
                    return
                latest[0] = b
                latest[1] += 1
        threading.Thread(target=_drain, daemon=True).start()

        cx, cy = FW / 2, FH / 2                   # EEF crop center (EMA)
        seen = 0
        while True:
            if latest[2]:
                break
            if latest[1] == seen:
                time.sleep(0.01)
                continue
            buf, seen = latest[0], latest[1]
            if sink is None and zsink is None:
                continue
            raw = np.frombuffer(buf, np.uint8).reshape(PH_P, PW_P, 3)
            ids = None
            if overlay or zsink is not None:
                # detect UNROTATED (PnP needs sensor coords, corners scale
                # by SSCALE); the half-res pipe is already detection-sized
                detf, dscale = raw, float(SSCALE)
                cs, ids, _ = adet.detectMarkers(detf)
            full = cv2.rotate(raw, ROT) if ROT is not None else raw
            small = cv2.resize(full, (DW, DH), interpolation=cv2.INTER_AREA)
            if os.environ.get("SQUARE", "0") == "1":
                # operation-range aid: the policy sees a center square crop —
                # dim everything outside it so the operator stays in range
                s = min(DW, DH)
                x0s, y0s = (DW - s) // 2, (DH - s) // 2
                small[:, :x0s] //= 2
                small[:, x0s + s:] //= 2
                small[:y0s, x0s:x0s + s] //= 2
                small[y0s + s:, x0s:x0s + s] //= 2
                cv2.rectangle(small, (x0s, y0s), (x0s + s - 1, y0s + s - 1),
                              (255, 255, 255), 1)
            if overlay:
                if ids is not None:
                    boxes = []
                    for c in cs:
                        pts = c.reshape(-1, 2)        # pipe coords
                        pts = np.array([rot_pt(x, y) for x, y in pts])
                        boxes.append((pts * (DW / FW)).astype(np.float32)
                                     .reshape(1, 4, 2))
                    cv2.aruco.drawDetectedMarkers(small, boxes, ids)
                n = 0 if ids is None else len(ids)
                cv2.putText(small, f"{n} markers", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if n else (0, 0, 255), 2)
            if zsink is not None:
                cen = None
                if ids is not None and track is not None:
                    Kc, distc, Tb, dobj_b, tgt = track
                    op, ip = [], []
                    for i, c in zip(ids.ravel(), cs):
                        i = int(i)
                        if i in Tb:
                            P = (Tb[i] @ np.hstack(
                                [dobj_b, np.ones((4, 1))]).T).T[:, :3]
                            op.append(P)
                            ip.append(c.reshape(4, 2).astype(np.float64)
                                      * dscale)
                    if op:
                        kw = dict(useExtrinsicGuess=True,
                                  rvec=last_rt[0].copy(),
                                  tvec=last_rt[1].copy()) \
                            if last_rt is not None else {}
                        okp, rv_, tv_ = cv2.solvePnP(
                            np.concatenate(op), np.concatenate(ip), Kc, distc,
                            flags=cv2.SOLVEPNP_ITERATIVE, **kw)
                        if okp:
                            last_rt = (rv_, tv_)
                            px, _ = cv2.projectPoints(tgt, rv_, tv_, Kc, distc)
                            cen = px.ravel() / dscale  # sensor -> pipe px
                if cen is None and ids is not None:
                    cen = np.concatenate([c.reshape(-1, 2) for c in cs]) \
                        .mean(axis=0)                  # already pipe px
                if cen is not None:
                    rx, ry = rot_pt(float(cen[0]), float(cen[1]))
                    cx += 0.25 * (rx - cx)        # EMA; holds position
                    cy += 0.25 * (ry - cy)        # when tracking drops
                x0 = int(min(max(cx - CROP / 2, 0), FW - CROP))
                y0 = int(min(max(cy - CROP / 2, 0), FH - CROP))
                crop = cv2.resize(full[y0:y0 + CROP, x0:x0 + CROP],
                                  (ZD, ZD), interpolation=cv2.INTER_AREA)
                try:
                    zsink.stdin.write(crop.tobytes())
                except (BrokenPipeError, OSError):
                    zsink = None
            if sink is not None:
                try:
                    sink.stdin.write(small.tobytes())
                except (BrokenPipeError, OSError):
                    print("\n(preview window closed — recording continues)",
                          flush=True)
                    sink = None  # viewer death must never stop the recording
    else:
        _pump_stderr()
except KeyboardInterrupt:
    pass
finally:
    # a second stop signal must not abort finalization (ts extraction +
    # concat/remux) — it crashed mid-finalize when TERM landed here
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    p.send_signal(signal.SIGINT)
    if PREVIEW:
        try:
            p.stdout.close()  # unblock ffmpeg if the pipe backed up
        except Exception:
            pass
    p.wait()
    if PREVIEW:
        try:
            if sink is not None:
                sink.stdin.close()
                sink.terminate()
        except Exception:
            pass

# extract per-frame wall-clock timestamps from the surviving epoch-PTS
# segments (a session janitor may have pruned idle ones), then concatenate +
# remux the deliverable to zero-based time so players can seek it. ts.json is
# built from the SAME segments in the SAME order, so frame index k of the
# deliverable is ts[k] — even across pruned gaps.
import glob

segs = sorted(glob.glob(f"data/raw/.{name}_seg*.mkv"))
if not segs:
    print("\nno segments captured (camera never delivered frames?) — "
          "nothing to save", flush=True)
    raise SystemExit(1)
ts = []
for s in segs:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", s],
        capture_output=True, text=True)
    ts += [{"wall": float(x)} for x in probe.stdout.split() if x]
json.dump(ts, open(f"data/raw/{name}_ts.json", "w"))
lst = f"data/raw/.{name}_concat.txt"
open(lst, "w").write("".join(f"file '{os.path.abspath(s)}'\n" for s in segs))
print(f"\nconcatenating {len(segs)} segments + remuxing "
      "(large 4K files take a minute — do NOT Ctrl-C)…", flush=True)
try:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy",
                    "-avoid_negative_ts", "make_zero", "-y", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(lst)
except BaseException:
    print(f"\nremux interrupted — segments preserved "
          f"(data/raw/.{name}_seg*.mkv); recover:\n"
          f"  ffmpeg -f concat -safe 0 -i {lst} -c copy "
          f"-avoid_negative_ts make_zero -y {out}", flush=True)
    raise
print(f"\nsaved {len(ts)} frames -> {out} + _ts.json "
      f"(wall {ts[0]['wall']:.2f}..{ts[-1]['wall']:.2f})" if ts else "no frames?!")
