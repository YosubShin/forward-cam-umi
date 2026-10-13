#!/usr/bin/env python3
"""Hand-off manifest for recorded sessions: lists exactly which files to send,
their sizes, and prints an rsync line. Nothing is copied or compressed — 4K
MJPEG sessions are 7-14 GB each, so they travel by rsync (or a drive), not
as a tarball.

What a session needs on the receiving side (the builds run there):
  data/raw/<dataset>_{b0587,b0587b,fish}.mkv + _ts.json + _meta.json
  data/raw/<dataset>_glove_episodes.json           (episode boundaries)
  data/raw/<dataset>_glove_log.json                (width holds, if any)
  calib/<dataset>_*_detections.npz                 (optional; saves an hour)
  data/raw/<dataset>_marker_map.json               (physical print snapshot)
and, once per unit / rig:
  calib/marker_map_glove_<unit>.json (legacy fallback), tags36h11_<unit>.json
  calib/glove_<unit>_dodeca_bundle.json, glove_<unit>_gripper_bundle.json
  calib/<camera>_intrinsics.json for every camera (+ _charuco_views.npz)
  calib/cameras.json
  calib/workspace_tiles_*.json, calib/<stem>_cam_world.json

Usage: python scripts/pack_session.py <dataset> [<dataset> ...] --unit <unit> [--dest user@host:/path]
"""
import argparse
import glob
import os
from pathlib import Path

from marker_config import WristMarkers, load_marker_map, resolve_marker_map

ROOT = Path(__file__).parent.parent.resolve()   # the tree the script was invoked from, not the symlink target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--marker-map", default=None,
                    help="map for legacy recordings; default session snapshot, then unit map")
    ap.add_argument("--dest", default="<user@host:/path/to/forward-cam-umi>",
                    help="rsync destination (the receiving repo root)")
    ap.add_argument("--out", default=None, help="write the manifest here (default data/raw/handoff_<unit>.txt)")
    a = ap.parse_args()

    files = []
    for ds in a.datasets:
        try:
            marker_path = resolve_marker_map(ROOT, ds, unit=a.unit, path=a.marker_map)
        except FileNotFoundError:
            marker_path = ROOT / (a.marker_map or f"calib/marker_map_glove_{a.unit}.json")
        except ValueError as exc:
            ap.error(str(exc))
        files.append(str(marker_path))
        print(f"{ds}: effective marker map {os.path.relpath(marker_path, ROOT)}")
        if marker_path.exists():
            print(f"  wrist: {WristMarkers.from_map(load_marker_map(marker_path)).description}")
        files += sorted(glob.glob(str(ROOT / f"data/raw/{ds}_*")))
        files += sorted(glob.glob(str(ROOT / f"calib/{ds}_*_detections.*")))
        files += sorted(glob.glob(str(ROOT / f"calib/{ds}_*_cam_world.json")))
    files += [str(ROOT / f) for f in (f"calib/tags36h11_{a.unit}.json",
                                      f"calib/glove_{a.unit}_dodeca_bundle.json", f"calib/glove_{a.unit}_gripper_bundle.json",
                                      "calib/cameras.json")]
    files += sorted(glob.glob(str(ROOT / "calib/*_intrinsics.json"))) + sorted(glob.glob(str(ROOT / "calib/*_charuco_views.npz")))
    files += sorted(glob.glob(str(ROOT / "calib/workspace_tiles_*.json")))
    files = sorted(set(files))
    print("Receiver: use each session's effective map (tips_dict, tips IDs and sizes) for width labels.")

    total = 0
    missing = []
    lines = []
    for f in files:
        rel = os.path.relpath(f, ROOT)
        if os.path.exists(f):
            s = os.path.getsize(f); total += s
            lines.append(f"{s / 1e9:8.2f} GB  {rel}")
        else:
            missing.append(rel)
    print("\n".join(lines))
    print(f"\n{len(lines)} files, {total / 1e9:.1f} GB")
    if missing:
        print("MISSING (expected for a complete hand-off):\n  " + "\n  ".join(missing))

    out = Path(a.out) if a.out else ROOT / f"data/raw/handoff_{a.unit}.txt"
    out.write_text("\n".join(os.path.relpath(f, ROOT) for f in files if os.path.exists(f)) + "\n")
    print(f"\nmanifest -> {out}")
    print("send with (run from the repo root; keeps the data/raw and calib layout):")
    print(f"  rsync -avP --files-from={os.path.relpath(out, ROOT)} . {a.dest}/")


if __name__ == "__main__":
    main()
