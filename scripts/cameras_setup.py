#!/usr/bin/env python3
"""Map the cameras on THIS machine to the roles the pipeline expects, and
write calib/cameras.json.

Why a step for this: the recorder pins cameras by USB port (/dev/v4l/by-path),
and the two B0587 4K units share one serial, so nothing but a human looking at
a live picture can say which one is the forward (label) camera and which is
the head-worn one. If the pins are wrong the recorder does not crash — it
records the wrong stream under the wrong name.

Usage:
  python scripts/cameras_setup.py            # list, preview each, assign, write
  python scripts/cameras_setup.py --list     # just list what is plugged in

For every capture device it prints the by-path link, the kernel name and the
serial, opens a live preview window (mpv/ffplay, like the recorder's), and
asks which role it is: b0587 (forward, fixed over the workspace), b0587b
(head-worn / second B0587), fish (wrist fisheye), or - to skip. Then it
writes calib/cameras.json in the recorder's format:
  [["b0587",  "/dev/v4l/by-path/...-video-index0", "B0587",          "3840x2160", "25"],
   ["b0587b", "/dev/v4l/by-path/...-video-index0", null,             "3840x2160", "25"],
   ["fish",   "/dev/v4l/by-path/...-video-index0", "1080P Low Light", "1920x1080", "30"]]
Re-run whenever a camera moves to a different USB port.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROLES = {
    "b0587": ("B0587", "3840x2160", "25", "forward: fixed over the workspace (label camera)"),
    "b0587b": (None, "3840x2160", "25", "second B0587: head-worn (optional)"),
    "fish": ("1080P Low Light", "1920x1080", "30", "wrist fisheye on the glove"),
}
ROOT = Path(__file__).parent.parent.resolve()   # the tree the script was invoked from, not the symlink target
OUT = ROOT / "calib/cameras.json"


def devices():
    out = []
    for bp in sorted(glob.glob("/dev/v4l/by-path/*-video-index0")):
        node = os.path.basename(os.path.realpath(bp))
        try:
            name = open(f"/sys/class/video4linux/{node}/name").read().strip()
        except OSError:
            continue
        caps = subprocess.run(["v4l2-ctl", "-d", bp, "--list-formats"],
                              capture_output=True, text=True).stdout
        if "MJPG" not in caps and "YUYV" not in caps:
            continue
        props = subprocess.run(["udevadm", "info", "-q", "property", "-n", bp],
                               capture_output=True, text=True).stdout
        serial = next((l.split("=", 1)[1] for l in props.splitlines()
                       if l.startswith("ID_SERIAL_SHORT=")), "")
        out.append((bp, name, serial))
    return out


def preview(dev, seconds=8):
    """Live window for a few seconds (mpv, else ffplay). Returns False if no
    frames could be read."""
    for cmd in (["mpv", "--no-config", "--really-quiet", f"--title={dev}",
                 "--profile=low-latency", "--untimed", f"--length={seconds}",
                 f"--demuxer-lavf-o=video_size=1280x720", f"av://v4l2:{dev}"],
                ["ffplay", "-loglevel", "error", "-window_title", dev,
                 "-f", "v4l2", "-video_size", "1280x720", "-t", str(seconds),
                 "-i", dev]):
        try:
            r = subprocess.run(cmd, timeout=seconds + 5)
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            continue
    print("  (no mpv/ffplay — cannot preview; identify by name/port instead)")
    return True


def main():
    devs = devices()
    if not devs:
        sys.exit("no UVC capture devices found")
    print("capture devices on this machine:")
    for i, (bp, name, serial) in enumerate(devs):
        print(f"  [{i}] {bp}\n      name {name!r}  serial {serial!r}")
    if "--list" in sys.argv:
        return
    assigned = {}
    print("\nfor each device a preview opens for ~8 s: wave a hand in front of the "
          "camera you think it is, then answer.\n"
          "roles: " + "  ".join(f"{k} = {v[3]}" for k, v in ROLES.items()) + "\n")
    for bp, name, serial in devs:
        print(f"-- {bp}  ({name})")
        preview(bp)
        while True:
            ans = input("   role [b0587 / b0587b / fish / - to skip / r to replay]: ").strip()
            if ans == "r":
                preview(bp)
                continue
            if ans in ("-", ""):
                break
            if ans in ROLES and ans not in assigned:
                assigned[ans] = bp
                break
            print("   unknown or already assigned")
    missing = [r for r in ("b0587", "fish") if r not in assigned]
    if missing:
        sys.exit(f"required roles not assigned: {missing} — rerun")
    table = []
    for role in ("b0587", "b0587b", "fish"):
        if role in assigned:
            frag, res, fps, _ = ROLES[role]
            table.append([role, assigned[role], frag, res, fps])
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        bak = OUT.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        OUT.rename(bak)
        print(f"existing {OUT.name} backed up to {bak.name}")
    OUT.write_text(json.dumps(table, indent=1))
    print(f"\nwrote {OUT}:")
    for row in table:
        print("  ", row)
    print("\nnext: python scripts/record_session.py --glove --cameras_only   "
          "(rehearsal: previews only, confirms the mapping before recording)")


if __name__ == "__main__":
    if sys.platform == "darwin":
        from cameras_setup_macos import main as macos_main
        macos_main()
    else:
        main()
