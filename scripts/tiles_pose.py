#!/usr/bin/env python3
"""Camera pose from the workspace tiles (AprilTag 36h11, ids 1-5).

The tiles define a WORLD frame shared between rigs: a glove session recorded
in another lab, with its camera wherever its mount put it, lands in the same
frame as ours as long as the tiles sit at the same places relative to the
task. On our side one hand-eye gives world -> robot base; on any side this
script gives camera -> world.

Two commands:

  layout  — export the tile layout from a static camera's footage. Per-tile
            PnP on sampled frames, medians, then a world frame anchored on
            the tiles themselves (origin = tile ORIGIN's centre, x toward
            tile XREF's centre, z = mean tile normal pointing up). Writes
            calib/workspace_tiles_<name>.json with every tile's pose in the
            world frame, in mm. Reproduce it elsewhere with a tape measure:
            the file is the drawing.
  solve   — camera -> world for a stream, from the layout: multi-tile PnP on
            every sampled frame (all corners of all decoded tiles against
            their world coordinates). For a static camera the per-frame
            spread is the precision; the median is the answer. Writes
            calib/<stem>_cam_world.json (T_cam_world = world expressed in the
            camera frame, 4x4, metres) and prints the residuals.

Optional marker_edges_mm maps string IDs to measured square-edge lengths.
These override the uniform edge_mm fallback and are preserved in new layouts.
Raw horizontal/vertical readings may accompany measurement_revision; using
their mean remains a square approximation, not a rectangular shape calibration.

Usage:
  python scripts/tiles_pose.py layout <stem> --intr calib/B0587_intrinsics.json \
      --tags calib/tags36h11_<unit>.json --name pilot [--origin 5 --xref 4] [--stride 50]
  python scripts/tiles_pose.py solve  <stem> --intr calib/B0587_intrinsics.json \
      --tags calib/tags36h11_<unit>.json --layout calib/workspace_tiles_pilot.json [--stride 50]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels_lib as L  # noqa: E402

ROOT = Path(__file__).parent.parent.resolve()   # the tree the script was invoked from, not the symlink target


def tag_obj(edge_m):
    h = edge_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def tag_edge_mm(config, marker_id):
    """Resolve one measured square edge, retaining legacy uniform layouts."""
    edges = config.get("marker_edges_mm", {})
    edge = float(edges[str(marker_id)] if str(marker_id) in edges else config["edge_mm"])
    if not np.isfinite(edge) or edge <= 0:
        raise ValueError(f"Tile {marker_id}: black-edge length must be positive and finite")
    return edge


def detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def sample_frames(stem, stride):
    """Sequential grab() with decode every `stride` frames: seeking a 4K
    MJPEG container by frame index re-scans it each time (minutes per
    pass); grab() just advances the demuxer."""
    cap = cv2.VideoCapture(str(ROOT / f"data/raw/{stem}.mkv"))
    k, misses = 0, 0
    while misses < 50:
        if not cap.grab():
            # a corrupt packet (seen at some stream starts) fails grab();
            # read() resyncs the decoder past it. Give up after 50 in a row.
            ok, _ = cap.read()
            misses += 1
            if not ok and misses >= 5:
                break
            k += 1
            continue
        misses = 0
        if k % stride == 0:
            ok, im = cap.retrieve()
            if ok:
                yield k, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k += 1
    cap.release()


def to_T(rv, tv):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
    T[:3, 3] = np.asarray(tv, float).ravel()
    return T


def cmd_layout(a):
    K, dist = L.load_intrinsics(a.intr)
    with Path(a.tags).open() as stream:
        tags = json.load(stream)
    ids_ok = set(int(i) for i in tags["ids"])
    det = detector()
    poses = {}
    n_frames = 0
    for k, g in sample_frames(a.stem, a.stride):
        n_frames += 1
        cs, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        for i, c in zip(ids.ravel(), cs):
            i = int(i)
            if i not in ids_ok:
                continue
            obj = tag_obj(tag_edge_mm(tags, i) / 1000.0)
            ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2).astype(float), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                poses.setdefault(i, []).append(to_T(rv, tv))
    if not poses:
        sys.exit("no tiles decoded")
    med = {}
    for i, Ts in poses.items():
        Ts = np.array(Ts)
        t = np.median(Ts[:, :3, 3], 0)
        # rotation: median rotation vector relative to the first (tiles are
        # static; spread is noise)
        rvs = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in Ts])
        R, _ = cv2.Rodrigues(np.median(rvs, 0).reshape(3, 1))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        dev = np.linalg.norm(Ts[:, :3, 3] - t, axis=1) * 1000
        med[i] = (T, len(Ts), float(np.median(dev)))
        print(f"tile {i}: {len(Ts)}/{n_frames} frames, position spread median {np.median(dev):.1f} mm")
    if a.origin not in med or a.xref not in med:
        sys.exit(f"origin tile {a.origin} or x-ref tile {a.xref} not seen; pick others")
    # world frame in the camera: origin at tile ORIGIN, x toward XREF, z = mean tile normal (+z of the tag frame)
    o = med[a.origin][0][:3, 3]
    x = med[a.xref][0][:3, 3] - o
    x /= np.linalg.norm(x)
    z = np.mean([T[:3, 2] for T, _, _ in med.values()], 0)
    z -= (z @ x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    T_cam_world = np.eye(4)
    T_cam_world[:3, 0], T_cam_world[:3, 1], T_cam_world[:3, 2], T_cam_world[:3, 3] = x, y, z, o
    T_world_cam = np.linalg.inv(T_cam_world)
    layout = {"name": a.name, "dict": "36h11", "edge_mm": tags["edge_mm"],
              "frame": f"origin = centre of tile {a.origin}; +x toward centre of tile {a.xref}; "
                       f"+z = tile normal (up); +y = z cross x. Units mm.",
              "source": {"stem": a.stem, "intrinsics": a.intr, "frames_sampled": n_frames},
              "tiles": {}}
    # Carry measured edges into the layout: solve_frame must project the same
    # per-ID geometry that was used to estimate the tile positions.
    if "marker_edges_mm" in tags:
        layout["marker_edges_mm"] = {str(i): tag_edge_mm(tags, i) for i in med}
    if "measurement_revision" in tags:
        layout["measurement_revision"] = tags["measurement_revision"]
    for i, (T, n, spread) in sorted(med.items()):
        Tw = T_world_cam @ T
        layout["tiles"][str(i)] = {"xyz_mm": (Tw[:3, 3] * 1000).round(1).tolist(),
                                   "rvec": cv2.Rodrigues(Tw[:3, :3])[0].ravel().round(4).tolist(),
                                   "frames": n, "spread_mm": round(spread, 1)}
    layout["T_cam_world_source"] = T_cam_world.tolist()
    out = ROOT / f"calib/workspace_tiles_{a.name}.json"
    with out.open("w") as stream:
        json.dump(layout, stream, indent=1)
    print(f"\nlayout (mm, world frame):")
    for i, t in layout["tiles"].items():
        print(f"  tile {i}: x {t['xyz_mm'][0]:7.1f}  y {t['xyz_mm'][1]:7.1f}  z {t['xyz_mm'][2]:6.1f}")
    ids = sorted(layout["tiles"])
    print("tile-to-tile centre distances (tape-measure these):")
    for p in range(len(ids)):
        for q in range(p + 1, len(ids)):
            d = np.linalg.norm(np.array(layout["tiles"][ids[p]]["xyz_mm"]) - np.array(layout["tiles"][ids[q]]["xyz_mm"]))
            print(f"  {ids[p]}-{ids[q]}: {d:6.1f} mm")
    print(f"wrote {out}")


def solve_frame(corners_by_id, layout, K, dist):
    """Multi-tile PnP: all corners of all decoded tiles vs world coordinates."""
    O, I = [], []
    for i, c in corners_by_id.items():
        t = layout["tiles"].get(str(i))
        if t is None:
            continue
        obj_t = tag_obj(tag_edge_mm(layout, i) / 1000.0)
        Tw = to_T(t["rvec"], np.array(t["xyz_mm"]) / 1000.0)
        O.append((Tw[:3, :3] @ obj_t.T).T + Tw[:3, 3])
        I.append(c.reshape(4, 2).astype(float))
    if not O:
        return None

    def fit(Os, Is):
        Oc, Ic = np.concatenate(Os), np.concatenate(Is)
        # These are WORLD corners, including a single translated/rotated tile.
        # IPPE_SQUARE accepts only its centred canonical square; SQPnP supports
        # these world coordinates without violating that object-point contract.
        ok, rv, tv = cv2.solvePnP(Oc, Ic, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        if len(Oc) > 4:
            refined, rv2, tv2 = cv2.solvePnP(Oc, Ic, K, dist, rvec=rv.copy(), tvec=tv.copy(),
                                           useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if refined:
                rv, tv = rv2, tv2
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            return None
        camera_points = Oc @ cv2.Rodrigues(rv)[0].T + tv.ravel()
        if np.any(camera_points[:, 2] <= 0):
            return None
        proj, _ = cv2.projectPoints(Oc, rv, tv, K, dist)
        if not np.isfinite(proj).all():
            return None
        per_tile = np.sqrt(((proj.reshape(-1, 2) - Ic) ** 2).sum(1)).reshape(-1, 4).mean(1)
        return rv, tv, per_tile

    r = fit(O, I)
    if r is None:
        return None
    rv, tv, per_tile = r
    # a tile that was bumped after the layout was taken disagrees with the
    # others by many px: drop the worst and refit while >= 3 tiles remain and
    # it reprojects far worse than the rest (taped tiles never trip this)
    while len(O) >= 4 and per_tile.max() > max(8.0, 4 * np.median(per_tile)):
        j = int(np.argmax(per_tile))
        O.pop(j); I.pop(j)
        r = fit(O, I)
        if r is None:
            return None
        rv, tv, per_tile = r
    return to_T(rv, tv), float(np.sqrt((per_tile ** 2).mean())), len(O)


def cmd_solve(a):
    K, dist = L.load_intrinsics(a.intr)
    with Path(a.layout).open() as stream:
        layout = json.load(stream)
    with Path(a.tags).open() as stream:
        tags = json.load(stream)
    ids_ok = set(int(i) for i in tags["ids"])
    det = detector()
    Ts, rms, nt, ks = [], [], [], []
    for k, g in sample_frames(a.stem, a.stride):
        cs, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        by = {int(i): c for i, c in zip(ids.ravel(), cs) if int(i) in ids_ok}
        r = solve_frame(by, layout, K, dist)
        if r is None:
            continue
        T, e, n = r
        Ts.append(T); rms.append(e); nt.append(n); ks.append(k)
    if not Ts:
        sys.exit("no frames solved")
    Ts = np.array(Ts)
    t_med = np.median(Ts[:, :3, 3], 0)
    rv_med = np.median([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in Ts], 0)
    T_med = to_T(rv_med, t_med)
    dev = np.linalg.norm(Ts[:, :3, 3] - t_med, axis=1) * 1000
    print(f"{a.stem}: {len(Ts)} frames solved; tiles per frame median {np.median(nt):.0f}; "
          f"reprojection RMS median {np.median(rms):.2f} px; camera position spread "
          f"median {np.median(dev):.1f} mm, p90 {np.percentile(dev, 90):.1f} mm")
    print(f"  camera in world (mm): {np.round(np.linalg.inv(T_med)[:3, 3] * 1000, 1)}")
    out = ROOT / f"calib/{a.stem}_cam_world.json"
    with out.open("w") as stream:
        json.dump({"stem": a.stem, "layout": a.layout, "T_cam_world": T_med.tolist(),
                   "n_frames": len(Ts), "rms_px_median": float(np.median(rms)),
                   "pos_spread_mm_median": float(np.median(dev)),
                   "per_frame": [{"k": int(k), "T_cam_world": T.tolist(), "rms": float(e), "tiles": int(n)}
                                 for k, T, e, n in zip(ks, Ts, rms, nt)] if a.per_frame else None},
                  stream, indent=1)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("layout")
    p1.add_argument("stem"); p1.add_argument("--intr", required=True); p1.add_argument("--tags", required=True)
    p1.add_argument("--name", required=True); p1.add_argument("--origin", type=int, default=5)
    p1.add_argument("--xref", type=int, default=4); p1.add_argument("--stride", type=int, default=50)
    p2 = sub.add_parser("solve")
    p2.add_argument("stem"); p2.add_argument("--intr", required=True); p2.add_argument("--tags", required=True)
    p2.add_argument("--layout", required=True); p2.add_argument("--stride", type=int, default=50)
    p2.add_argument("--per-frame", action="store_true", help="keep every sampled frame's pose (moving camera)")
    a = ap.parse_args()
    cmd_layout(a) if a.cmd == "layout" else cmd_solve(a)


if __name__ == "__main__":
    main()
