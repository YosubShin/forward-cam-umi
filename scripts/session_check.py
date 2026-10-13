#!/usr/bin/env python3
"""One-command health report for a recorded glove session, after detection.

Answers "is this session good enough to ship/build?" before anyone builds a
dataset from it. Per episode it reports what the dataset build gates on, and
it checks the width holds the way the build will use them.

  streams    every expected stream present, frame counts, fps, wall-clock span
  episodes   per episode: duration, forward-cam ball pose coverage (fused
             solve, ball faces + calibrated back markers), wrist-cam tip
             coverage (both tips decoded), head-cam ball coverage if present;
             flags anything under the build's 70% gate
  holds      per caliper hold: both-tip decode rate and the tip separation
             (must be flat; separation - gap should be constant for this print)
  tiles      forward cam: which workspace tiles decode and how often

Usage:
  python scripts/session_check.py <dataset> --unit <unit> [--forward-intr ...] [--head-intr ...]
e.g.
  python scripts/session_check.py glove_yuan_01 --unit yuan
Reads data/raw/<dataset>_{b0587,b0587b,fish}.*, calib/<stem>_*_detections.npz,
calib/glove_<unit>_{dodeca,gripper}_bundle.json, calib/marker_map_glove_<unit>.json,
calib/tags36h11_<unit>.json. Exit code 1 if any episode or hold fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels_lib as L  # noqa: E402
from marker_config import WristMarkers, load_marker_map, resolve_marker_map

ROOT = Path(__file__).parent.parent.resolve()   # the tree the script was invoked from, not the symlink target
GATE = 0.70


def ts_of(stem):
    p = ROOT / f"data/raw/{stem}_ts.json"
    return np.array([x["wall"] for x in json.load(open(p))]) if p.exists() else None


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def wrist_series(root, dataset, wrist_markers, intrinsics):
    """Marker-center separation in mm, not yet the calibrated jaw gap.

    Read the configured family's cache. Missing/ambiguous members remain
    NaN and count against width coverage.
    """
    root = Path(root)
    cache = root / "calib" / f"{dataset}_fish_{wrist_markers.cache_tag}_detections.npz"
    K, dist = L.load_intrinsics(root / intrinsics)
    with (root / "data/raw" / f"{dataset}_fish_ts.json").open() as stream:
        ts = json.load(stream)
    with np.load(cache) as detections:
        rows = L.body_frames(detections, ts, K, dist, None, None, wrist_markers.sizes_mm)
    left, right = sorted(wrist_markers.sizes_mm)
    return (np.array([r["t_wall"] for r in rows]),
            np.array([np.linalg.norm(r["pb"][left] - r["pb"][right])
                      if left in r["pb"] and right in r["pb"] else np.nan for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--marker-map", default=None,
                    help="map for legacy recordings; default session snapshot, then the unit map")
    ap.add_argument("--forward-intr", default="calib/B0587_intrinsics.json")
    ap.add_argument("--head-intr", default="calib/B0587b_intrinsics.json")
    ap.add_argument("--wrist-intr", default=None, help="default calib/fisheye_glove_<unit>_intrinsics.json")
    a = ap.parse_args()
    unit = a.unit
    wrist_intr = a.wrist_intr or f"calib/fisheye_glove_{unit}_intrinsics.json"
    bad = []
    try:
        marker_path = resolve_marker_map(ROOT, a.dataset, unit=unit, path=a.marker_map)
        mm = load_marker_map(marker_path)
        wrist_markers = WristMarkers.from_map(mm)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))

    # ---- streams
    print(f"== {a.dataset}")
    print(f"  marker map: {marker_path}")
    print(f"  wrist: {wrist_markers.description}")
    tw = {}
    for cam, fps_nom in (("b0587", 25), ("b0587b", 25), ("fish", 30)):
        t = ts_of(f"{a.dataset}_{cam}")
        if t is None:
            print(f"  {cam:7s} MISSING" + ("" if cam == "b0587b" else "  <-- required"))
            if cam != "b0587b":
                bad.append(f"stream {cam} missing")
            continue
        tw[cam] = t
        fps = len(t) / max(t[-1] - t[0], 1e-6)
        print(f"  {cam:7s} {len(t):6d} frames  {hms(t[0])} -> {hms(t[-1])}  {fps:5.2f} fps")
    marks_p = ROOT / f"data/raw/{a.dataset}_glove_episodes.json"
    if not marks_p.exists():
        sys.exit("  no episode marks file — was this recorded with --glove?")
    eps = json.load(open(marks_p))["episodes"]
    print(f"  {len(eps)} marked episodes, durations {min(e['duration_s'] for e in eps):.1f}-{max(e['duration_s'] for e in eps):.1f} s")

    # ---- label stacks
    L.TAILS = {int(k): v for k, v in mm["tails"].items()}
    L.BASES = {int(k): v for k, v in mm["bases"].items()}
    Tb, dobj = L.load_bundle(ROOT / f"calib/glove_{unit}_dodeca_bundle.json")
    extra = L.load_gripper_bundle(ROOT / f"calib/glove_{unit}_gripper_bundle.json")

    def fused_times(stem, intr):
        det_p = ROOT / f"calib/{stem}_dodeca_detections.npz"
        if not det_p.exists():
            return None
        K, dist = L.load_intrinsics(ROOT / intr)
        ts = json.load(open(ROOT / f"data/raw/{stem}_ts.json"))
        rows = L.body_frames(np.load(det_p), ts, K, dist, Tb, dobj, {}, extra=extra)
        return np.array([r["t_wall"] for r in rows])

    tP = fused_times(f"{a.dataset}_b0587", a.forward_intr)
    if tP is None:
        sys.exit("  no forward detections — run detect_parallel.py on the b0587 stream first")
    tH = fused_times(f"{a.dataset}_b0587b", a.head_intr) if "b0587b" in tw else None

    # wrist tips
    tips_t, sep = None, None
    if "fish" in tw:
        try:
            tips_t, sep = wrist_series(ROOT, a.dataset, wrist_markers, wrist_intr)
        except FileNotFoundError as exc:
            print(f"  wrist width inputs missing: {exc.filename}")
            bad.append("wrist width not checked: missing detections, timestamps or intrinsics")

    # ---- episodes
    print("\n  ep   start     dur   ball-pose  tips   head-pose")
    for e in eps:
        s0, s1 = e["t_start"], e["t_end"]
        n = ((tw["b0587"] >= s0) & (tw["b0587"] <= s1)).sum()
        cov = ((tP >= s0) & (tP <= s1)).sum() / max(n, 1)
        if sep is not None:
            m = (tips_t >= s0) & (tips_t <= s1)
            nw = ((tw["fish"] >= s0) & (tw["fish"] <= s1)).sum()
            tcov = np.isfinite(sep[m]).sum() / max(nw, 1)
        else:
            tcov = float("nan")
        if tH is not None:
            nh = ((tw["b0587b"] >= s0) & (tw["b0587b"] <= s1)).sum()
            hcov = ((tH >= s0) & (tH <= s1)).sum() / max(nh, 1)
        else:
            hcov = float("nan")
        flag = ""
        if cov < GATE:
            flag += f"  <-- ball pose under {GATE:.0%} (build would skip)"
            bad.append(f"ep{e['episode_index']} pose {cov:.0%}")
        if sep is not None and tcov < GATE:
            flag += f"  <-- tips under {GATE:.0%}"
            bad.append(f"ep{e['episode_index']} tips {tcov:.0%}")
        print(f"  {e['episode_index']:3d}  {hms(s0)}  {e['duration_s']:5.1f}  {cov:8.0%}  {tcov:5.0%}  {hcov:8.0%}{flag}")

    # ---- holds
    log_p = ROOT / f"data/raw/{a.dataset}_glove_log.json"
    if log_p.exists() and sep is not None:
        holds = {}
        for ev in json.load(open(log_p))["events"]:
            if ev["event"] == "e8_meas":
                holds.setdefault(ev["gap_mm"], []).append(ev["t_wall"])
        print(f"\n  holds ({len(holds)}): gap -> both-tip decode, tip separation p5..p95 (flat?), separation - gap")
        offs = []
        for g, t in sorted(holds.items()):
            m = (tips_t >= min(t) - 0.3) & (tips_t <= max(t) + 0.3)
            v = sep[m]
            ok = np.isfinite(v).mean() if m.any() else 0.0
            if np.isfinite(v).sum() < 10:
                print(f"    {g:6.2f} mm: both tips in {ok:4.0%} of {m.sum()} frames  <-- too few; hold again with both tips in the wrist view")
                bad.append(f"hold {g} mm unusable")
                continue
            p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
            offs.append(p50 - g)
            # the wide holds put the tips near the fisheye's edge where the
            # per-frame solve is noisier (glove4: 5 mm p5..p95 at 85 mm, fit
            # RMS still 0.9 mm) — flag only a spread the fit cannot absorb
            lim = max(6.0, 0.08 * p50)
            flat = "" if (p95 - p5) < lim else "  <-- not flat: gap changed during the hold"
            print(f"    {g:6.2f} mm: both tips {ok:4.0%} of {m.sum()} frames; separation {p5:.1f}..{p95:.1f} mm; offset {p50 - g:+.1f}{flat}")
            if (p95 - p5) >= lim:
                bad.append(f"hold {g} mm not flat")
        if len(offs) >= 2 and (max(offs) - min(offs)) > 4.0:
            print(f"    offsets disagree by {max(offs) - min(offs):.1f} mm — a caliper value or a spacer is off")
            bad.append("hold offsets inconsistent")
        if len(holds) < 3:
            print("    fewer than 3 holds: the width fit needs at least closed / mid / open")
            bad.append("fewer than 3 holds")
    elif not log_p.exists():
        print("\n  (no hold log for this session — fine if the unit's holds live in another session)")

    # ---- tiles (forward cam, sampled)
    tags = json.load(open(ROOT / f"calib/tags36h11_{unit}.json"))
    p = cv2.aruco.DetectorParameters(); p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
    cap = cv2.VideoCapture(str(ROOT / f"data/raw/{a.dataset}_b0587.mkv"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); seen = {int(i): 0 for i in tags["ids"]}; ns = 0
    step, k, misses = max(n // 60, 1), 0, 0
    while misses < 50:                      # sequential: seeking a 4K MJPEG is slow
        if not cap.grab():                  # corrupt packet: resync via read()
            ok, _ = cap.read(); misses += 1
            if not ok and misses >= 5:
                break
            k += 1
            continue
        misses = 0
        if k % step == 0:
            ok, im = cap.retrieve()
            if ok:
                ns += 1
                _, ids, _ = det.detectMarkers(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                for i in (ids.ravel() if ids is not None else []):
                    if int(i) in seen:
                        seen[int(i)] += 1
        k += 1
    cap.release()
    print("\n  workspace tiles in the forward cam (sampled frames): " +
          "  ".join(f"tile{i}:{100 * c / max(ns, 1):.0f}%" for i, c in sorted(seen.items())))
    if sum(c > 0 for c in seen.values()) < 2:
        print("    fewer than 2 tiles visible — the camera->world solve needs at least 2; re-aim or move tiles into view")
        bad.append("fewer than 2 tiles visible")

    print("\n== " + ("OK — ship it" if not bad else "PROBLEMS: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
