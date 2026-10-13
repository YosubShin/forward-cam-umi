#!/usr/bin/env python3
"""Live ChArUco intrinsics calibration for a UVC camera.

Wave the 10x7 board (25.0mm square / 18.0mm marker, DICT_5X5_250) through the
view: near/far, corners, tilted. Frames with enough corners, spaced in time
and board position, are collected automatically. After --n frames it
calibrates, validates on a holdout, and writes calib/<name>_intrinsics.json
(same schema the analysis scripts read).

The output name is derived from the camera's sysfs device name (B0587 /
B0591 / fisheye) — pass --name only for an unknown camera. An existing
output file is backed up, never silently clobbered (a default-name overwrite
destroyed the B0591 calibration on 2026-09-03).

Collected views are saved to calib/<name>_charuco_views.npz; re-fit later
without re-waving via --refit calib/<name>_charuco_views.npz.

--model rational uses the 8-coefficient rational distortion model — needed
for wide/fisheye lenses where the 5-coefficient fit fails (fisheye pinhole
holdout was 7.2px; rational stays cv2.solvePnP-compatible downstream, unlike
cv2.fisheye's equidistant coefficients).

Lock focus/exposure BEFORE calibrating — intrinsics are focus-dependent.

macOS uses the saved camera role and recording mode, including image rotation.
Exposure/gain are not controlled by the macOS adapter; hold the board still
between moves under steady light. Collection uses the same terminal counter
and fitting logic on both platforms (no preview window).

Usage: python scripts/calibrate_uvc.py [/dev/video0] [--model rational]
       python scripts/calibrate_uvc.py forward --square-mm 24.4 --marker-mm 17.95
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from uvc_ritual import apply as apply_ritual, _name as device_name

# sysfs device-name fragment -> calib file stem
NAME_MAP = {"B0587": "B0587", "1080P-HDR": "B0591",
            "1080P Low Light": "fisheye"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dev", nargs="?", default="forward" if sys.platform == "darwin" else "/dev/video0")
    ap.add_argument("--name", default=None,
                    help="calib file stem; default: derived from device name")
    ap.add_argument("--model", choices=["pinhole", "rational"], default="pinhole")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--square-mm", type=float, default=None,
                    help="printed square edge; default: saved value for refit, otherwise 25.0")
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="printed black marker edge; default: saved value for refit, "
                         "otherwise OUR board's 18.06 mm — measure your own print")
    ap.add_argument("--holdout", type=int, default=10)
    ap.add_argument("--width", type=int, default=None,
                    help="default: saved mode on macOS; 3840 for 4K-named cameras on Linux, else 1920")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--refit", help="charuco_views.npz from a previous run — "
                                    "recalibrate without capturing")
    if sys.platform == "darwin":
        from camera_config import DEFAULT_CONFIG
        ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return ap.parse_args(argv)


def prepare_camera(args):
    """Resolve identity and calibration dimensions before opening hardware."""
    if sys.platform == "darwin":
        from camera_config import resolve_spec
        spec = resolve_spec(args.dev, args.config)
        if ((args.width is not None and args.width != spec.width) or
                (args.height is not None and args.height != spec.height)):
            raise ValueError("calibration dimensions must match the saved recording mode")
        args.width, args.height = spec.width, spec.height
        args.name = args.name or {"b0587": "B0587", "b0587b": "B0587b",
                                  "fish": "fisheye"}[spec.role]
        return spec
    dn = device_name(args.dev)
    args.name = args.name or next((v for k, v in NAME_MAP.items() if k in dn), None)
    if args.name is None:
        raise ValueError(f"unknown camera '{dn}' — pass --name explicitly")
    args.width = args.width or (3840 if "4K" in dn else 1920)
    args.height = args.height or (2160 if "4K" in dn else 1080)
    return None


def open_capture(args, spec):
    if spec is not None:
        from camera_macos import OpenCVPreview
        print(f"macOS: {spec.role} · {spec.device_id} · {spec.fps:g} fps · "
              f"rotation {spec.rotation_degrees}° before detection", flush=True)
        print("Exposure/gain are not controlled on macOS; use steady light "
              "and pause the board between moves.", flush=True)
        return OpenCVPreview(spec), spec.width, spec.height
    cap = cv2.VideoCapture(args.dev)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # UNSET fps negotiates a broken mode on the fisheye (10s+ stalls).
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {args.dev}")
        for _ in range(5):
            cap.read()  # stream must be running before controls stick
        apply_ritual(args.dev)
        return cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except BaseException:
        cap.release()
        raise


def backup_existing(path):
    if os.path.exists(path):
        bak = f"{path}.bak-{datetime.now():%Y%m%d-%H%M%S-%f}"
        os.rename(path, bak)
        print(f"existing {path} backed up to {bak}")


def main(argv=None):
    args = parse_args(argv)
    metadata = {}
    spec = None
    if args.refit:
        # Refitting must not access cameras or lose the original orientation
        # and board measurements. Older NPZ files without metadata still work.
        with np.load(args.refit, allow_pickle=False) as z:
            views = [(z[f"c{i}"], z[f"i{i}"]) for i in range(int(z["n"]))]
            W, H = int(z["width"]), int(z["height"])
            if "metadata_json" in z:
                metadata = json.loads(str(z["metadata_json"].item()))
        if ((args.width is not None and args.width != W) or
                (args.height is not None and args.height != H)):
            raise ValueError("refit dimensions must match the saved views")
        args.width, args.height = W, H
        args.name = args.name or Path(args.refit).stem.replace("_charuco_views", "")
    else:
        spec = prepare_camera(args)
        metadata = {"device": args.dev, "note": "focus/exposure locked at calibration"}
        if spec is not None:
            metadata["capture"] = {"backend": "avfoundation", "role": spec.role,
                                   "device_id": spec.device_id, "width": spec.width,
                                   "height": spec.height, "fps": spec.fps,
                                   "rotation_degrees": spec.rotation_degrees,
                                   "exposure_gain_controlled": False}
            metadata["note"] = ("Frames normalized before calibration; macOS "
                                "exposure/gain not controlled. Keep physical focus unchanged.")
    if args.holdout < 1 or (len(views) if args.refit else args.n) - args.holdout < 3:
        raise ValueError("need at least 1 held-out view and 3 training views")
    saved_board = metadata.get("board", {})
    args.square_mm = args.square_mm if args.square_mm is not None else saved_board.get("square_mm", 25.0)
    args.marker_mm = args.marker_mm if args.marker_mm is not None else saved_board.get("marker_mm", 18.06)
    if not (np.isfinite(args.square_mm) and 0 < args.marker_mm < args.square_mm):
        raise ValueError("board sizes must satisfy 0 < marker-mm < square-mm")
    metadata["board"] = {"squares_x": 10, "squares_y": 7, "dictionary": "DICT_5X5_250",
                         "square_mm": args.square_mm, "marker_mm": args.marker_mm}
    name = args.name
    print(f"calibration target: {name}  (model: {args.model}, "
          f"{args.width}x{args.height})")
    print(f"ChArUco: 10x7 · squares {args.square_mm:g} mm · markers {args.marker_mm:g} mm")
    SQUARE_M, MARKER_M = args.square_mm / 1000.0, args.marker_mm / 1000.0
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((10, 7), SQUARE_M, MARKER_M, d)
    cdet = cv2.aruco.CharucoDetector(board)

    if args.refit:
        print(f"refitting {len(views)} saved views @ {W}x{H}")
    else:
        cap, W, H = open_capture(args, spec)
        print(f"{args.dev} @ {W}x{H} — show the board; collecting {args.n} diverse views")

        print("No preview window. Move, pause, and watch the counter; Ctrl-C cancels.", flush=True)
        views = []          # (charucoCorners, charucoIds)
        last_t = 0.0
        last_c = None
        try:
            while len(views) < args.n:
                ok, img = cap.read()
                if not ok:
                    continue
                cc, ci, _, _ = cdet.detectBoard(img)
                if cc is None or ci is None or len(cc) < 20:
                    continue
                now = time.time()
                cen = cc.reshape(-1, 2).mean(axis=0)
                if now - last_t < 0.4:
                    continue
                if last_c is not None and np.linalg.norm(cen - last_c) < 40:
                    continue  # board hasn't moved enough for a new view
                views.append((cc.copy(), ci.copy()))
                last_t, last_c = now, cen
                print(f"\r  collected {len(views)}/{args.n} "
                      f"({len(cc)} corners, center {cen[0]:.0f},{cen[1]:.0f})   ",
                      end="", flush=True)
        finally:
            cap.release()
        print()
        vout = f"calib/{name}_charuco_views.npz"
        Path("calib").mkdir(exist_ok=True)
        backup_existing(vout)
        np.savez_compressed(vout, n=len(views), width=W, height=H,
                            metadata_json=json.dumps(metadata),
                            **{f"c{i}": v[0] for i, v in enumerate(views)},
                            **{f"i{i}": v[1] for i, v in enumerate(views)})
        print(f"saved views to {vout} (refit anytime with --refit)")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(views))
    hold = set(order[: args.holdout].tolist())
    obj_all, img_all = [], []
    for i, (cc, ci) in enumerate(views):
        if i in hold:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        if op is not None and len(op) >= 6:
            obj_all.append(op); img_all.append(ip)
    print(f"calibrating on {len(obj_all)} views (holdout {len(hold)})…")
    flags = cv2.CALIB_RATIONAL_MODEL if args.model == "rational" else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_all, img_all, (W, H), None, None, flags=flags)
    # A view grabbed while the board was still moving carries corners that are
    # smeared or mis-refined by several px; a handful of them pull fx by percent
    # (B0587b 2026-10-06: 7 of 35 views at 4-5 px dragged fx 2497 -> 2607 and
    # RMS 1.0 -> 1.9). Drop views whose mean residual is far above the pack and
    # refit once; the dropped count is printed so a sloppy capture is visible.
    per_view = []
    for op, ip, rv, tv in zip(obj_all, img_all, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        # OpenCV versions differ: matchImagePoints may return Nx2 or Nx1x2.
        # Subtract point pairs, never broadcast them into an NxN error grid.
        per_view.append(float(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1).mean()))
    per_view = np.array(per_view)
    cut = max(2.0, 3.0 * np.median(per_view))
    keep = per_view <= cut
    if (~keep).any() and keep.sum() >= 10:
        print(f"  dropping {(~keep).sum()} view(s) with mean residual > {cut:.2f}px "
              f"(board moving during capture): "
              f"{np.round(per_view[~keep], 1).tolist()}")
        obj_all = [o for o, k in zip(obj_all, keep) if k]
        img_all = [i for i, k in zip(img_all, keep) if k]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_all, img_all, (W, H), None, None, flags=flags)
    print(f"calibration RMS {rms:.3f}px on {len(obj_all)} views  fx {K[0,0]:.1f} "
          f"fy {K[1,1]:.1f} ppx {K[0,2]:.1f} ppy {K[1,2]:.1f}")

    errs = []
    for i in sorted(hold):
        cc, ci = views[i]
        op, ip = board.matchImagePoints(cc, ci)
        if op is None or len(op) < 6:
            continue
        ok, rv, tv = cv2.solvePnP(op, ip, K, dist)
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        errs.append(float(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1).mean()))
    if errs:
        print(f"held-out reprojection: mean {np.mean(errs):.3f}px  "
              f"worst {np.max(errs):.3f}px  ({len(errs)} frames)")
        clean = [e for e in errs if e <= cut]
        if len(clean) < len(errs):
            print(f"  ({len(errs) - len(clean)} held-out view(s) above the "
                  f"{cut:.2f}px moving-board cut; clean held-out mean "
                  f"{np.mean(clean):.3f}px)")

    out = f"calib/{name}_intrinsics.json"
    Path("calib").mkdir(exist_ok=True)
    backup_existing(out)
    with open(out, "w") as f:
        json.dump({"camera_matrix": K.tolist(), "dist_coeffs": dist.ravel().tolist(),
                   "width": W, "height": H, "rms_px": rms, "model": args.model,
                   "holdout_mean_px": float(np.mean(errs)) if errs else None,
                   "device": metadata.get("device", args.dev),
                   "note": metadata.get("note", "Refit from saved views; capture settings not recorded"),
                   **{k: metadata[k] for k in ("board", "capture") if k in metadata}},
                  f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nCalibration interrupted.")
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Calibration: {exc}")
