#!/usr/bin/env python3
"""Live focus tuner for BARREL-focused cameras (B0587: the lens focuses
mechanically only — the UVC focus_absolute control is accepted but INERT).

Opens a window with the live image so you see what the barrel does:
  top     the whole frame, downscaled, with the 1:1 crop box and any decoded
          marker outlined
  bottom  the crop at 1:1 — ACTUAL sensor pixels, no scaling — so blur is
          visible as blur, next to the numbers. The crop follows the largest
          decoded marker on its own (a tile, ideally); 'n' in the terminal
          steps it to the next marker, 'c' parks it on the frame centre.
The window is an mpv/ffplay sink fed raw frames (the env's cv2 is GUI-less,
like the recorder's previews); keys are read from the TERMINAL, not the
window, so keep the terminal focused.

Two numbers, both measured at the camera's RECORDING resolution (4K):
  edge px   width of the black/white transition on decoded marker edges
            (36h11 tiles, 4x4 ball/tail faces). Resolution-independent and
            comparable across cameras: a sharp B0587 reads 2-3 px on a tile
            at 1 m. A head-worn unit at 7 px decoded a second ball face in 5%
            of frames; refocused to 4 px (2026-10-07) it decoded two faces in
            92% and fused the ball pose in 99% of frames — focus, not
            geometry, was the limit. Minimise this number; needs a marker.
  sharp     centre-crop Laplacian variance, with a session peak-hold. Scene-
            dependent (texture, light), so only compare against itself while
            you turn — dial past the peak and come back to it.

Why 4K: at 1080p the ISP's downscale hides a 6 px blur as a 3 px one and
the meter happily peaks on a lens that is soft at the resolution we record.

Pass --ref <dev> to open a second (known-good) camera: its 1:1 crop and edge
px are shown beside yours, so "as sharp as the forward cam" is a comparison
you make by eye and by number, on the same target, at the same moment.

Keys (terminal): q quit, r reset peak, n next marker, c centre.  Ctrl-C quits.

Usage:
  python scripts/focus_tune.py /dev/v4l/by-path/<head>
  python scripts/focus_tune.py head                 # macOS: saved role
  python scripts/focus_tune.py /dev/v4l/by-path/<head> --ref /dev/v4l/by-path/<forward>
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from preview_sink import open_sink, close_sink


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dev", nargs="?", default="forward" if sys.platform == "darwin" else "/dev/video0")
    ap.add_argument("--ref", help="second camera shown beside the first for comparison")
    ap.add_argument("--size", default=None if sys.platform == "darwin" else "3840x2160",
                    help="capture size; macOS defaults to the saved recording size")
    if sys.platform == "darwin":
        from camera_config import DEFAULT_CONFIG
        ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        ap.add_argument("--role", dest="role", choices=("forward", "head", "wrist", "b0587", "b0587b", "fish"),
                        help=argparse.SUPPRESS)  # previous macOS command spelling
    return ap.parse_args(argv)


BASE = ("auto_exposure=1,exposure_time_absolute=2,gain=0,sharpness=0,"
        "backlight_compensation=0")


def open_cam(dev, width, height, config=None):
    if sys.platform == "darwin":
        from camera_config import resolve_spec
        from camera_macos import OpenCVPreview
        spec = resolve_spec(dev, config)
        if (spec.width, spec.height) != (width, height):
            raise ValueError("capture size differs from the saved macOS mode; rerun camera setup")
        return OpenCVPreview(spec)
    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {dev}")
    for _ in range(8):
        cap.read()
    for kv in BASE.split(","):        # controls latch only mid-stream
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", kv], capture_output=True)
    return cap


p36 = cv2.aruco.DetectorParameters()
p36.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
p4 = cv2.aruco.DetectorParameters()
p4.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
p4.errorCorrectionRate = 1.0
DETS = [cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p36),
        cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p4)]


def edge_px(g, quad):
    """Median 10%->90% transition width (px) across a marker's four sides,
    from intensity profiles sampled perpendicular to each side."""
    q = quad.reshape(4, 2)
    widths = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        nv = b - a
        L = np.linalg.norm(nv)
        if L < 12:
            continue
        nrm = np.array([-nv[1], nv[0]]) / L
        for t in np.linspace(0.25, 0.75, 5):
            c = a + t * nv
            ts = np.arange(-8, 9)
            pts = c[None, :] + ts[:, None] * nrm[None, :]
            x, y = pts[:, 0], pts[:, 1]
            if (x.min() < 1 or y.min() < 1 or x.max() >= g.shape[1] - 1
                    or y.max() >= g.shape[0] - 1):
                continue
            prof = cv2.remap(g, x.astype(np.float32)[None, :],
                             y.astype(np.float32)[None, :],
                             cv2.INTER_LINEAR)[0].astype(float)
            lo, hi = prof.min(), prof.max()
            if hi - lo < 40:
                continue
            nz = (prof - lo) / (hi - lo)
            widths.append(int(((nz > 0.1) & (nz < 0.9)).sum()))
    return float(np.median(widths)) if widths else float("nan")


def measure(img, crop_xy, crop_wh):
    """-> (gray, sharp, edge, markers[(quad, edge)]).  Markers are FOUND at
    half resolution (a full-res 4K pass is ~1.5 s, too slow to turn a barrel
    against) and then RE-DETECTED inside a full-res window around each, so
    the corners used for the edge width are full-res. 'edge' is the median
    over 36h11 tiles when any are decoded — big and static — else over the
    4x4 faces (small, often moving)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy = crop_xy
    cw, ch = crop_wh
    c = g[max(0, cy - ch // 2): cy + ch // 2, max(0, cx - cw // 2): cx + cw // 2]
    sharp = float(cv2.Laplacian(c, cv2.CV_64F).var())
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    markers, by_dict = [], []
    for det in DETS:
        corners, ids, _ = det.detectMarkers(small)
        found = []
        if ids is not None:
            for q, marker_id in zip(corners, ids.ravel()):
                q2 = q.reshape(4, 2) * 2.0
                x0, y0 = q2.min(0)
                x1, y1 = q2.max(0)
                m = 0.4 * max(x1 - x0, y1 - y0)
                xa, ya = int(max(0, x0 - m)), int(max(0, y0 - m))
                xb, yb = int(min(g.shape[1], x1 + m)), int(min(g.shape[0], y1 + m))
                cr, ci, _ = det.detectMarkers(g[ya:yb, xa:xb])
                if ci is None:
                    continue
                matches = np.flatnonzero(ci.ravel() == marker_id)
                if len(matches) != 1:
                    continue
                q2 = cr[matches[0]].reshape(4, 2) + np.array([xa, ya], np.float32)
                e = edge_px(g, q2)
                markers.append((q2, e))
                if not np.isnan(e):
                    found.append(e)
        by_dict.append(found)
    edges = by_dict[0] or by_dict[1]
    edge = float(np.median(edges)) if edges else float("nan")
    return g, sharp, edge, markers


def panel(img, g, markers, crop_xy, crop_wh, label, sharp, peak, edge,
          view_w):
    """Downscaled full view (with overlays) stacked over the 1:1 crop."""
    h, w = g.shape
    s = view_w / w
    view = cv2.resize(img, (view_w, int(h * s)), interpolation=cv2.INTER_AREA)
    for q, e in markers:
        pts = (q * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [pts], True, (0, 255, 0), 1)
        if not np.isnan(e):
            cv2.putText(view, f"{e:.0f}px", tuple((q[0] * s).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cx, cy = crop_xy
    cw, ch = crop_wh
    x0, y0 = max(0, cx - cw // 2), max(0, cy - ch // 2)
    cv2.rectangle(view, (int(x0 * s), int(y0 * s)),
                  (int((x0 + cw) * s), int((y0 + ch) * s)), (0, 200, 255), 1)
    crop = img[y0: y0 + ch, x0: x0 + cw]
    if crop.shape[1] != cw or crop.shape[0] != ch:
        pad = np.zeros((ch, cw, 3), np.uint8)
        pad[: crop.shape[0], : crop.shape[1]] = crop
        crop = pad
    crop = crop.copy()
    etxt = "edge n/a (no marker)" if np.isnan(edge) else f"edge {edge:.1f} px"
    cv2.putText(crop, f"{label}: {etxt}   sharp {sharp:.0f} (peak {peak:.0f})",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(crop, f"{label}: {etxt}   sharp {sharp:.0f} (peak {peak:.0f})",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(crop, "1:1 pixels", (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1)
    return np.vstack([view, crop])


class Tuner:
    def __init__(self, dev, label, width, height, config=None):
        self.dev, self.label = dev, label
        self.width, self.height = width, height
        self.cap = open_cam(dev, width, height, config)
        self.crop_xy = [width // 2, height // 2]
        self.follow = 0            # index into the size-sorted marker list; -1 = centre
        self.peak = 0.0

    def tick(self, crop_wh):
        ok = False
        # AVFoundation already retains only the latest frame. Linux needs draining.
        for _ in range(1 if sys.platform == "darwin" else 3):
            ok, img = self.cap.read()
        if not ok:
            return None
        g, sharp, edge, markers = measure(img, self.crop_xy, crop_wh)
        # crop follows a decoded marker: tiles first (big, static), then by size
        order = sorted(markers, key=lambda m: -np.linalg.norm(m[0][0] - m[0][2]))
        if self.follow >= 0 and order:
            q = order[self.follow % len(order)][0]
            self.crop_xy = [int(q[:, 0].mean()), int(q[:, 1].mean())]
        elif self.follow < 0:
            self.crop_xy = [self.width // 2, self.height // 2]
        self.peak = max(self.peak, sharp)
        return img, g, sharp, edge, markers


def main(argv=None):
    args = parse_args(argv)
    config = getattr(args, "config", None)
    if sys.platform == "darwin":
        from camera_config import resolve_spec
        args.dev = args.role or args.dev
        spec = resolve_spec(args.dev, config)
        W, H = spec.width, spec.height
        if args.size and tuple(int(x) for x in args.size.lower().split("x")) != (W, H):
            raise ValueError("--size must match the saved macOS recording mode")
        if args.ref:
            reference = resolve_spec(args.ref, config)
            if (reference.width, reference.height) != (W, H):
                raise ValueError("--ref must have the same capture size as the main camera")
        print(f"{spec.role}: {spec.device_id}, {W}x{H} @ {spec.fps:g} fps, rotation {spec.rotation_degrees}°")
        print("macOS: exposure/gain controls are not applied; inspect a stationary, well-lit target.")
    else:
        W, H = (int(x) for x in args.size.lower().split("x"))
    VIEW_W = 960 if args.ref else 1280           # per-camera column width
    CROP = (VIEW_W, 360)                           # 1:1 crop size (w, h)
    PANEL_H = int(H * VIEW_W / W) + CROP[1]
    main = Tuner(args.dev, "THIS", W, H, config)
    ref = sink = None
    try:
        ref = Tuner(args.ref, "REF", W, H, config) if args.ref else None
        ncol = 2 if ref else 1
        sink = open_sink("focus_tune", VIEW_W * ncol, PANEL_H)
    except BaseException:
        main.cap.release()
        if ref:
            ref.cap.release()
        raise

    print(f"{args.dev}: turn the focus barrel by hand. Minimise 'edge px' on a "
          "marker (2-3 px = sharp B0587 at 1 m); 'sharp' peaks at best focus.\n"
          "keys here in the terminal: q quit, r reset peak, n next marker, "
          "c centre crop.\n")
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_tty = None
    try:
        old_tty = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, OSError):
        pass
    try:
        while True:
            cols, line = [], []
            for t in (main, ref) if ref else (main,):
                r = t.tick(CROP)
                if r is None:
                    cols.append(np.zeros((PANEL_H, VIEW_W, 3), np.uint8))
                    line.append(f"{t.label} no frame")
                    continue
                img, g, sharp, edge, markers = r
                cols.append(panel(img, g, markers, t.crop_xy, CROP, t.label,
                                  sharp, t.peak, edge, VIEW_W))
                etxt = "  n/a" if np.isnan(edge) else f"{edge:5.1f}"
                line.append(f"{t.label} edge {etxt}px sharp {sharp:7.1f} "
                            f"(peak {t.peak:7.1f})")
            frame = np.hstack(cols)
            if sink is not None and sink.poll() is None:
                try:
                    sink.stdin.write(frame.tobytes())
                    sink.stdin.flush()
                except (BrokenPipeError, OSError):
                    close_sink(sink)
                    sink = None
            print("\r  " + " | ".join(line) + " " * 8, end="", flush=True)
            if select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                if k == "r":
                    main.peak = 0.0
                    if ref:
                        ref.peak = 0.0
                if k == "n":
                    for t in (main, ref) if ref else (main,):
                        t.follow = max(0, t.follow) + 1
                if k == "c":
                    for t in (main, ref) if ref else (main,):
                        t.follow = -1
    except KeyboardInterrupt:
        pass
    finally:
        if old_tty is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_tty)
        main.cap.release()
        if ref:
            ref.cap.release()
        close_sink(sink)
    print(f"\n\npeak sharp {main.peak:.1f} — park the barrel there, add a witness "
          "mark, then calibrate:")
    if sys.platform == "darwin":
        print("Next: intrinsics calibration. Live macOS calibration still needs adaptation.")
    else:
        print(f"  python scripts/calibrate_uvc.py {args.dev} --name <camera>")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Focus tuning: {exc}")
