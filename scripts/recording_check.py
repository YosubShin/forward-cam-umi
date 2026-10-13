#!/usr/bin/env python3
"""Check raw recording transport before geometry fitting or session_check.

Validates encoded frame count against capture timestamps, saved native mode,
cadence/gaps, and overlap. Does NOT certify optical sync or marker coverage.
"""
import argparse
import json
from pathlib import Path
import subprocess

from recording_io import probe_video, timing_stats, validate_stem

ROOT = Path(__file__).parent.parent.resolve()


def check_stream(root, stem):
    root = Path(root)
    meta = json.loads((root / f"{stem}_meta.json").read_text())
    if not all((root / f"{stem}{ext}").exists() for ext in (".mkv", "_ts.json")):
        raise RuntimeError(f"no finalized video/timestamps; received {meta.get('received_frames', 'unknown')} "
                           f"camera frames; {meta.get('error') or meta.get('status', 'unknown status')}")
    walls = [row["wall"] for row in json.loads((root / f"{stem}_ts.json").read_text())]
    spec = meta["capture"]
    stats = timing_stats(walls, spec["fps"])
    info = probe_video(root / f"{stem}.mkv")
    problems = []
    if int(info["nb_read_packets"]) != len(walls):
        problems.append("video frame count differs from timestamp count")
    if (info["width"], info["height"], info["codec_name"]) != (spec["width"], spec["height"], "mjpeg"):
        problems.append("encoded resolution/codec differs from capture contract")
    if meta.get("status") != "finalized" or meta.get("error"):
        problems.append(f"recorder status: {meta.get('status')}; {meta.get('error')}")
    if meta.get("native_drops", 0) or stats["gap_count"]:
        problems.append(f"native drops {meta.get('native_drops', 0)}, timestamp gaps {stats['gap_count']}")
    if abs(stats["measured_fps"] / spec["fps"] - 1) > 0.05:
        problems.append("measured frame rate differs from requested rate by >5%")
    # Diagnostic only: preserve the full-take failure criteria above. Startup
    # gaps must not be mistaken for sustained overload or silently discarded.
    later = [t for t in walls if t >= walls[0] + 2]
    post_startup = timing_stats(later, spec["fps"]) if len(later) >= 2 else None
    return {"stem": stem, "capture": spec, "timing": stats,
            "after_first_2s": post_startup, "problems": problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=validate_stem)
    ap.add_argument("--cameras", default="b0587,b0587b,fish")
    ap.add_argument("--min-overlap", type=float, default=10,
                    help="minimum common recording span, in seconds (default 10)")
    args = ap.parse_args()
    from camera_config import ROLES
    roles = [r.strip() for r in args.cameras.split(",")]
    if not roles or any(r not in ROLES for r in roles) or len(set(roles)) != len(roles):
        ap.error(f"--cameras must list distinct roles from {ROLES}")
    if not 0 < args.min_overlap < float("inf"):
        ap.error("--min-overlap must be positive and finite")
    rows, problems = [], []
    for role in roles:
        stem = f"{args.dataset}_{role}"
        try:
            row = check_stream(ROOT / "data/raw", stem)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
            problems.append(f"{stem}: {exc}")
            continue
        rows.append(row)
        stats = row["timing"]
        print(f"{role}: {stats['frames']} frames, {stats['measured_fps']:.2f} fps, "
              f"max gap {stats['max_gap_s']*1000:.1f} ms, "
              f"rotation {row['capture']['rotation_degrees']} degrees")
        if later := row["after_first_2s"]:
            print(f"  after first 2s (diagnostic only): {later['measured_fps']:.2f} fps, "
                  f"{later['gap_count']} gaps, max gap {later['max_gap_s']*1000:.1f} ms")
        problems += [f"{stem}: {p}" for p in row["problems"]]
    overlap = (min(r["timing"]["last_wall"] for r in rows)
               - max(r["timing"]["first_wall"] for r in rows)) if len(rows) == len(roles) else None
    if overlap is None:
        problems.append("common capture span unavailable: an expected stream is missing/invalid")
    elif overlap < args.min_overlap:
        problems.append(f"common capture span {overlap:.2f}s < {args.min_overlap:g}s")
    report = {"dataset": args.dataset, "streams": rows, "common_span_s": overlap,
              "problems": problems, "transport_passed": not problems,
              "optical_sync_validated": False, "exposure_gain_controlled": False}
    from recording_io import write_json
    out = ROOT / "data/raw" / f"{args.dataset}_recording_check.json"
    write_json(out, report)
    print(f"Common capture span: {overlap:.2f}s" if overlap is not None else
          "Common capture span: unavailable (missing/invalid stream)")
    print("Stored timestamps checked; optical synchronization and moving-marker quality still require review.")
    print("\n".join(problems) if problems else "PASS: raw video/timestamps/cadence")
    print(f"Report: {out}")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
