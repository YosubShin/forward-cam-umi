# Tracking software: calibrate, record, solve

The Python side of the forward-camera rig. Everything runs from the
**repository root** (`python scripts/...`) and reads/writes `calib/` and
`data/raw/` relative to it — keep that layout.

Status: the robot-side path (same code, robot-mounted markers) is validated
against robot forward kinematics — 7.7 mm / 1.7° median, see
[yam-umi/pos-tracking/dynamic-accuracy.md](https://github.com/YosubShin/yam-umi/blob/main/pos-tracking/dynamic-accuracy.md).
The glove-session path (steps 4–6) has been run end to end on four glove
sessions (Oct 2026, ~70 episodes): recorded, solved, built into training
datasets and trained. The gotchas below are the ones it surfaced.

## 0. Install

```
pip install -r scripts/requirements.txt      # opencv-contrib, numpy, scipy
sudo apt install ffmpeg v4l-utils            # ffmpeg/ffprobe + v4l2-ctl
```
Linux with the cameras on USB (UVC). Tested with Python 3.12.

### macOS adapter (experimental)

macOS uses the same terminal interaction and **mpv/ffplay** preview as Linux.
There is no separate Python GUI or Qt dependency. Install with Python 3.12:

```sh
brew install ffmpeg                       # includes ffplay; mpv is also supported
python -m pip install -r scripts/requirements-macos.txt
python scripts/cameras_setup.py --list --formats
python scripts/cameras_setup.py --require-head
```

Like Linux setup, each device gets a preview, then the terminal asks for its
role: `b0587` (forward), `b0587b` (head), `fish` (wrist), or `-` to skip.
Use `r` to replay. Choose image rotation and recording mode, verify that mode,
and confirm the physical camera. The identification preview uses a temporary
small resolution; the saved recording mode defaults to the pilot settings.
Existing bindings are backed up only after all required roles are verified;
cancelling or failing setup keeps the old config.

The macOS differences are limited to camera I/O and local configuration:
AVFoundation device IDs replace V4L2 paths; capture validates native dimensions
and normalizes 0°/180° rotation before analysis. Identical names are never used
as identity fallbacks. Changing USB ports can change IDs; rerun setup and
verify the physical cameras. Bindings remain in ignored
`calib/cameras.macos.local.json`; measurements in `calib/*.local.json` stay local.
Existing macOS configs are compatible and need no migration.

```sh
python scripts/cameras_setup.py --check-config --require-head
python scripts/cameras_setup.py --preview head --seconds 15
python scripts/focus_tune.py forward
python scripts/focus_tune.py head --ref forward
python scripts/focus_tune.py wrist
```

`focus_tune.py` shares the Linux measurement, layout and terminal controls:
**q** quit, **r** reset peak, **n** next marker, **c** centre crop. Keep the
**terminal focused**, not the player window. `--ref` opens both cameras at
once and needs enough USB bandwidth. macOS selects each role's saved native
resolution/FPS; `--size`, if supplied, must match it. Never rotate the normalized
head-camera images again or reuse calibration from the opposite orientation.
As in Linux, `edge px` aggregates workspace-tag edges when available, otherwise
glove-marker edges; it is not necessarily the edge width of the selected crop.
The crop uses source pixels; the external player's scaling controls its final
on-screen size. Compare a stationary target under steady lighting.

Close other camera apps before capture. Camera permission belongs to the
terminal/app launching Python; allow it under System Settings → Privacy &
Security → Camera. `--list` and `--check-config` do not open cameras.

Intrinsics use the same `calibrate_uvc.py` collection and fitting loop as Linux.
Pass a saved role instead of a V4L2 path and your own board measurements:

```sh
python scripts/calibrate_uvc.py forward --name B0587 --square-mm <measured> --marker-mm <measured>
python scripts/calibrate_uvc.py head --name B0587b --square-mm <measured> --marker-mm <measured>
python scripts/calibrate_uvc.py wrist --name fisheye_glove_<unit> --model rational --square-mm <measured> --marker-mm <measured>
```

Close the focus preview first. There is no calibration preview window: move
the flat board through near/far, centre/corners and tilted poses, pause between
moves, and watch the terminal's `collected N/45` counter. Ctrl-C during collection
cancels without replacing previous results. Ten views are held out for validation. Camera
mode/rotation and board measurements are saved with the JSON and NPZ outputs;
`--refit` reuses that metadata without opening a camera (explicit board-size
flags can correct an earlier measurement). Older NPZs without metadata still
need their original board-size flags. Head-camera coordinates are already
rotated before corner detection. Results/backups stay local under `calib/`.

Currently adapted: **camera setup, focus inspection, live intrinsics, and
glove/rehearsal recording**. Robot/teleop recording is still Linux-only.
Exposure/gain control is not ported: keep light steady, check moving markers
in a short take, and do not assume the Linux 0.2 ms exposure ritual ran.

Start with a short three-camera rehearsal (close QuickTime/focus/calibration
first). Hold the glove so forward/head see its ball and wrist sees its tips;
the original recorded-frame marker pre-flight still applies. The 20 seconds
start **after** pre-flight, so the total recording will be longer:

```sh
python scripts/record_session.py --cameras_only --dataset mac_smoke_01 --duration 20 --num_episodes 1 --no_preview
python scripts/recording_check.py mac_smoke_01
```

This first test isolates capture/encoding throughput. Repeat with a new dataset
name and `--preview_all` instead of `--no_preview` to validate the live viewers.
Then use the same `--glove --dataset D --num_episodes N` commands and ENTER /
`d` / `q` / optional `--width_holds` interaction as Linux. `--no_preview` turns
off external mpv/ffplay windows; the macOS recording preview shows a main view
and marker overlay (the Linux EEF zoom panel is not ported). `--config PATH`
selects another local binding file; all cameras must have distinct exact IDs.

The raw interface remains `<dataset>_<role>.mkv`, `_ts.json` (one `wall` value
per decoded frame), `_meta.json`, and the unchanged glove episode/hold files.
macOS has the following explicit differences:

- AVFoundation delivers decoded pixels. They are rotated to the calibrated
  orientation **before** encoding, then encoded as MJPEG at JPEG quality 95.
  This is a lossy re-encode, not Linux's untouched camera JPEG stream copy.
  Metadata records the device, dimensions, rate, rotation and codec path.
- The native session PTS is converted to the CoreMedia host clock and anchored
  to Unix time, matching the episode marks' `time.time()` clock. The video
  container uses nominal fps; **`_ts.json` is the capture time source of truth**,
  including gaps. No frames are synthesized to fill gaps. An observed system
  clock jump aborts the take instead of silently misaligning episode marks.
- Recording has a bounded FIFO separate from preview's latest-frame queue.
  Its capacity absorbs up to one second of burst delivery, capped at 512 MiB
  of queued BGR pixels per camera (21 frames at 4K, 31 at the saved 1080p
  30.00003 fps mode). This is a queue payload limit, not a process-memory cap.
  Starting another UVC stream can interrupt and then burst frame delivery to
  an already running stream; a fixed eight-frame FIFO did not absorb this.
  FFmpeg and the JPEG/mux consumer start before the blocking AVFoundation
  startup call, so process initialization does not block the first frame and
  frames can be drained while the session is still starting. No warm-up pixels
  are injected; a muxer that receives no frames is cancelled. Shutdown
  stops capture, drains the writer, then remuxes. Metadata includes startup
  duration, buffer capacity, queue wait, encode/mux latency and callback
  cadence for diagnosis. All received frames retain their original capture
  timestamps; no startup frames are removed to make the queue look healthy.
  FIFO overflow is fatal; native drops and timing gaps are reported. Closing
  a viewer never stops capture. Each stream retains 60-second segments and a
  timestamp journal until remux and frame-count checks succeed. Failed or
  unfinished takes are preserved and cannot be silently reused. The session
  stops if a recorder exits or remaining disk space cannot accommodate the
  final remux copy plus a 2 GB reserve.
- On macOS, timed rehearsals (`--duration`) budget disk space from the selected
  cameras' resolution and fps: 0.5 compressed bytes/pixel, the requested
  duration plus 30 seconds for startup, two video copies for remux, and a
  2 GB reserve. This is an estimate; MJPEG size varies with the scene and the
  runtime space check still applies. Untimed sessions retain the 30 GB floor.
- `recording_check.py` checks raw transport, dimensions, frame/timestamp
  counts, frame cadence and common time span, **not** optical synchronization,
  marker coverage, or label accuracy. For hardware validation, show all views
  a common visible event and compare its capture times (allowing frame-period
  quantization); also inspect moving-marker detection. Run the later
  `session_check.py` after geometry and offline detection are available.

Daily bindings and generated capture/calibration data remain local. The
macOS adapter contains no user-specific device IDs or physical measurements.

## 1. Measurements — one caliper reading per printed group

Printers rescale by a few percent and the error does not announce itself:
a 3% size error is a 3% range error in every pose. **Caliper one black square
per group** and declare it in the config. Nominal is not enough.

| what | file | fields |
|---|---|---|
| glove markers (sheet `markers/glove_markers_v4.pdf`) | `calib/marker_map_glove_<unit>.json` | `tails`, `bases`, `tips` sizes in mm; `ball_edge_mm` |
| workspace tags (`markers/apriltag_36h11_60mm_marker_75mm_tile_final.pdf`) | `calib/tags36h11_<unit>.json` | `edge_mm`; **use tags 1–5 only** |
| ChArUco board (`markers/charuco_letter_8.5x11.pdf`) | flags to `calibrate_uvc.py` | `--square-mm`, `--marker-mm` |

Two built units are recorded, both from the same sheet and both complete:
`calib/marker_map_glove_{yuan,yosub}.json` with `calib/tags36h11_{yuan,yosub}.json`.
They differ by ~1% in every group — the normal print-to-print variance — which
is why a new unit always gets its **own** file and its own bundle solves
(step 6): copy one of them, replace every size with your calipers, rename.

Two ID traps, already handled if you follow the commands below:
`4x4-27` is a real ball face but aliases to AprilTag `16h5-1` (a robot tip
marker) — detect glove footage with `--keep-27`; workspace tag `36h11-0`
aliases to `4x4-37` (a glove tail) — never place tag 0.

## 2. Cameras: pin by USB port, then lock focus and exposure

Copy `calib/cameras.example.json` to `calib/cameras.json` and edit it for
your machine. Each entry is `[suffix, by_path, name_fragment_or_null,
"WxH", "fps"]`; find by-path links with `ls -l /dev/v4l/by-path/`. Suffixes
the pipeline understands: `b0587` (fixed forward 4K), `b0587b` (second
B0587 — fixed twin or head-worn), `fish` (wrist fisheye). The two B0587s
share a serial, so `b0587b` has no name fallback; if its port changes the
recorder resolves it as "the B0587 that is not the forward one".

Focus is mechanical on the B0587 and intrinsics are focus-dependent:
lock the barrel with the live sharpness meter, mark it, and only then
calibrate.
```
python scripts/focus_tune.py /dev/v4l/by-path/<forward>
python scripts/focus_tune.py /dev/v4l/by-path/<head> --ref /dev/v4l/by-path/<forward>
```
It opens a window: the whole frame on top and a 1:1 crop of actual sensor
pixels below (the crop follows the largest decoded marker; `n` in the
terminal steps to the next one), so you see the blur you are turning out. Aim at a marker (a workspace tile at ~1 m is ideal) and minimise
**edge px**, the black/white transition width on the marker's edges —
2–3 px is a sharp B0587 at 4K; the "sharp" number is a scene-dependent
Laplacian meter with peak-hold, useful only against itself while you turn.
`--ref` puts a known-good camera beside yours on the same target, which is
the only honest "as sharp as the forward cam" check. It matters more than it
looks: a head-worn B0587 at 7 px edges decoded a second ball face in 5% of
frames (no pose); the same unit at 4 px decoded two faces in 92% and fused a
pose in 99% — a calibration RMS of 0.6 px had said nothing about it. The meter captures at
the recording size (4K): a 1080p preview hides half the blur.
The exposure/gain ritual (`scripts/uvc_ritual.py`) is applied automatically
by the recorder and the calibrator: manual exposure, `exposure_time_absolute=2`
(0.2 ms), gain 0. Verify behaviourally — a hand waved in front of the lens
must be frozen, not smeared; readback alone lies on these modules.

## 3. Intrinsics (once per camera, or after any barrel movement)

Wave the ChArUco board through the view — near/far, corners, tilted — until
it has collected enough frames; it calibrates, validates on a holdout and
writes `calib/<name>_intrinsics.json`, naming the camera from its device
name (B0587 → `B0587_intrinsics.json`; pass `--name B0587b` for the second
unit). Existing files are backed up, never overwritten.
```
python scripts/calibrate_uvc.py /dev/v4l/by-path/<forward> --square-mm 25.0 --marker-mm <caliper>
python scripts/calibrate_uvc.py /dev/v4l/by-path/<second>  --name B0587b --square-mm 25.0 --marker-mm <caliper>
python scripts/calibrate_uvc.py /dev/v4l/by-path/<fish>    --model rational --square-mm 25.0 --marker-mm <caliper>
```
Expect ~0.7 px RMS at 4K on the holdout; the 1080p fisheye lands near 2.7 px
with `--model rational` (required for it). **Calibrate the glove's wrist camera
under its own `--name`** (e.g. `--name fisheye_glove_<unit>`): the wrist-observation
arm reads gripper width from the 16h5 tip markers that camera sees, which is
metric PnP, and the auto-derived name `fisheye` would overwrite another unit's
file. Only a wrist stream used purely as observations can skip calibration.

## 4. Record a glove session

No robot, no host. Cameras roll continuously; you mark episode boundaries.
Hold the glove in the workspace when it starts — the pre-flight decodes a
live frame from every camera and aborts on zero markers (lens smudge,
exposure not latched, wrong device).
```
python scripts/record_session.py --glove --dataset glove_yuan_01 --num_episodes 30 --task "<task>"
```
`ENTER` starts an episode, `ENTER` ends it, `d` discards the one you just
did, `q` finishes. Outputs in `data/raw/`: `<dataset>_<cam>.mkv` + `_ts.json`
per camera and `<dataset>_glove_episodes.json` (boundaries).

**Different physical wrist markers (Linux and macOS).** Copy the full unit
marker map to `calib/marker_map_glove_<unit>.local.json` (git-ignored). Set
`tips_dict` to `16h5` or `4x4`, and `tips` to the two jaw-marker IDs and their
measured **black-edge lengths in mm**. For a 4x4 marker also listed in `tails`
or `bases`, the sizes must agree. Do not reuse the old tips' size for a
different print. Pass the map when recording:

```sh
python scripts/record_session.py --glove --dataset glove_test_01 --num_episodes 5 \
  --marker-map calib/marker_map_glove_<unit>.local.json
```

This requires **both configured wrist IDs** at pre-flight, rejecting unrelated
and ambiguous duplicate IDs. It saves the full map to
`data/raw/<dataset>_marker_map.json`; retries reuse this snapshot. Changing
the map requires a new dataset name. Without a map or snapshot, recording
keeps the legacy any-16h5 wrist check.

`detect_parallel.py` already writes both families; run it as usual.
`session_check.py --unit <unit>` reads the snapshot automatically: 4x4 wrist
markers use `_dodeca_detections.npz`, 16h5 uses `_tips16h5_detections.npz`.
Missing wrist inputs fail the check. `pack_session.py` includes the effective
map and identifies it for the receiver; any downstream dataset builder must
honor that map's family, IDs and sizes. On recordings without a snapshot,
both commands accept `--marker-map PATH`, otherwise use the legacy unit map.

Changing wrist markers requires fresh **width holds** to verify visibility
and metric accuracy across apertures: marker-center separation is not the
jaw gap. It does not require new camera intrinsics if the lens and capture
settings stay unchanged.

**Width holds — once per unit, not per session.** The first session on a new
glove unit (and any session after you move or reprint the tip/back markers)
runs with `--width_holds`. At the first prompt, clamp the jaws on a rigid
**spacer** of caliper-measured width (the gap must not change during the
hold; fingers can't hold an exact opening while you move, a spacer can),
type the value (`37.2`, or `m 37.2`) and ENTER; the recorder logs that gap
until you press ENTER again. During the ~15–20 s hold rotate the whole glove
slowly so the forward camera sees **both jaws'** tail cubes and the base
markers from a few angles — every marker face only gets a width estimator if
it was seen at two or more holds, and a one-sided hold leaves the other jaw
blind. Five apertures (closed, ~20, ~40, ~60, ~80 mm) beat three: the open
end extrapolates badly otherwise. The wrist camera must decode **both** tip
markers at every hold (check with `tag_contrast.py` first). Holds land in
`<dataset>_glove_log.json`, which every later build of that unit reuses via
`--log`. Without the flag the prompt never asks. Caliper values are logged
as raw mm unless `calib/gap_to_actuator.json` exists (that file comes from a
robot bench measurement and maps mm to the robot's gripper units — we
supply it).

**Offline metric forward-width review.** `forward_width_check.py` uses existing
forward detection caches and raw-mm width holds. It trains only on the first
recording, freezes a separate model, then replays a different aperture from the
second recording. Intrinsics, ball/base geometry, marker maps and capture
settings remain unchanged. For the Yuan recordings:

```bash
.venv/bin/python scripts/forward_width_check.py \
  glove_yuan_width_02 glove_yuan_width_verify_01 --unit yuan \
  --out tmp/forward_width_yuan_v4.local.json
```

The learned model jointly fits moving/fixed tag corners rather than subtracting
two independently estimated tag depths. Candidate references must be within
50 mm at the closed hold and observed at three widths; a time-blocked training
holdout applies physical-mm error gates. The final independent gates remain
coverage >=70%, absolute median error <=1 mm, and p95 absolute error <=2 mm.
Missing detections, rejected solves and missing body poses remain in the
coverage denominator. Exit status 2 means this accuracy review failed, with the
report and frozen model still saved under `tmp/`. Predictions are not clipped or
smoothed. Static-hold success does not verify moving-width accuracy. This is a
physical-mm offline path; it does not replace `labels_lib.py`'s normalized robot
actuator mapping or install a model into the robot pipeline.

Joint solves above the 1 px Huber residual scale retry SQPnP initializations at
three positions along the trained stroke, choosing the lowest-cost converged
positive-depth solution. These are starting points, not width bounds. New models
store per-pair training-blocked p95 scores: the best visible candidate per fixed
reference contributes one vote before taking a median across references. Missing
faces use the next ranked candidate; historical models without scores retain
their flat-median fusion. Reports record solver policy and implementation hashes.
The current 57.9 mm replay reaches p95 2.084 mm against the unchanged 2 mm gate;
it remains a failed review. See [progress](../docs/forward-width-progress.md).

The subsequent measured-size Yosub v3 baseline reaches 2.167 mm P95 on the same
independent 57.9 mm hold. On 2026-10-13 the user selected that preserved baseline
and accepted a **2.2 mm static P95 limit**, retaining 70% coverage and 1 mm median
bias limits. `forward_width_accept.py` replays an existing frozen model under an
explicit policy without refitting or modifying historical 2 mm reports:

```bash
.venv/bin/python scripts/forward_width_accept.py glove_yuan_width_verify_01 \
  --model tmp/forward_width_measured_yosub_v3.local.model.json \
  --p95-max-mm 2.2 \
  --out tmp/forward_width_measured_yosub_v3_acceptance_repeat.local.json
```

Use a fresh output name for each review. Historical dataset names remain
`glove_yuan_*`, while the frozen model records physical unit `yosub`. This static
acceptance is separate from the unchanged training-candidate gates and does not
qualify the recorded larger-turn failures. The next step is one short normal
glove episode to inspect moving-marker coverage and width continuity, using a
fresh dataset name. This acceptance does not install the metric model into the
legacy normalized actuator pipeline.

For moving recordings, `forward_width_quality_check.py` layers an explicit
numerical quality policy over the frozen solver. It rejects ill-conditioned
pair candidates **before** selecting the best training-ranked face for each
fixed reference, so a usable face can still provide a fallback. The score is
`max(corner_noise_floor_px, rms_px) * std_mm_per_pixel`; it is a numerical
diagnostic, not an absolute error bound. Derive policy limits from training
diagnostics, then freeze the policy and model hashes before reviewing motion.

```sh
python scripts/forward_width_quality_check.py <dataset> \
  --model tmp/<frozen-model>.json \
  --quality-policy calib/<quality-policy>.local.json \
  --out tmp/<fresh-motion-review>.json
```

Width-hold recordings additionally require an explicit `--p95-max-mm`. The
report retains raw and qualified arrays, per-pair rejection reasons, coverage
and adjacent valid-frame steps. Missing/rejected frames remain NaN; no temporal
averaging, interpolation, clipping or refitting occurs. Receiver replay resolves
the frozen geometry within its own checkout and verifies the recorded hashes;
historical absolute provenance paths are not active fallback inputs. See the
[moving-width review](../docs/forward-width-progress.md) for measured coverage
tradeoffs and the remaining view-dependent bias. Local models, policies,
calibration and recordings must be transferred separately from Git.

Also record one **calibration take** per unit (`--num_episodes 1`): a single
long episode slowly rotating the ball through every orientation at working
distance, with the gripper-back markers visible part of the time. Step 6
solves the ball's geometry from it.

## 4b. Collecting for the pilot on another machine

The hand-off sequence — camera mapping, calibration, geometry, holds,
recording, detection, a one-command session check, camera→tiles, and what
to ship — is in [`docs/pilot-task.md`](../docs/pilot-task.md), with the task
definition, reference frames and the tile layout that puts your labels in
the same frame as ours.

## 5. Detection

```
python scripts/detect_parallel.py data/raw/<dataset>_b0587.mkv --keep-27     # and _b0587b, _fish
```
Writes `calib/<stem>_dodeca_detections.npz` (ArUco 4x4) and
`calib/<stem>_tips16h5_detections.npz`. Uses all cores; `--workers N` to
limit.

## 6. Solve the unit's geometry (once per physical glove)

Face-to-ID layout on the ball is not prescribed; it is recovered from the
calibration take. Three passes, outputs are specific to the physical unit:
```
python scripts/bundle_dodeca.py --npz=calib/glove_cal_b0587_dodeca_detections.npz \
    --intr=calib/B0587_intrinsics.json --marker-mm=<ball caliper> --ball-ids=25-35 --max-vel=600 --out=calib/tmp_bundle_<unit>.json
python scripts/bundle_dodeca_ideal.py --npz=calib/glove_cal_b0587_dodeca_detections.npz \
    --intr=calib/B0587_intrinsics.json --init=calib/tmp_bundle_<unit>.json --marker-mm=<ball caliper> \
    --max-vel=600 --out=calib/glove_<unit>_dodeca_bundle.json
python scripts/calibrate_gripper_bundle.py glove_cal_b0587 --bundle=calib/glove_<unit>_dodeca_bundle.json \
    --marker-map=calib/marker_map_glove_<unit>.json --out=calib/glove_<unit>_gripper_bundle.json
```
`--max-vel=600` admits moving frames: the default 150 px/s stillness gate
was tuned for a slow-shutter camera, and on a rotation take at the 0.2 ms
shutter it discards most frames and drops the rarely-shown faces from the
bundle entirely (a first solve at the default kept 9/11 faces). Expect all
11 faces at ~0.5 px RMS — a built glove unit solved at 0.52 px over 597
frames; the robot's ball, included as a format example
(`calib/*_robot.example.json`), sits at 0.71 px on 465 frames.

## What to send back for dataset building

`data/raw/<dataset>_*` (all cameras, mkv + ts.json), the two
`_glove_*.json` sidecars, `calib/<name>_intrinsics.json` for each camera,
`calib/marker_map_glove_<unit>.json`, `calib/tags36h11_<unit>.json`, and the
calibration take. Dataset construction (labels → joint actions via IK) needs
the robot's hand-eye and kinematics, so it runs on the robot side.

## Command index

Every command prints its full documentation with `--help`. Run from the
repository root.

| command | does | step |
|---|---|---|
| `cameras_setup.py [--list]` | preview every camera, assign forward / head / wrist roles, write `calib/cameras.json` | 2 |
| `focus_tune.py <dev> [--ref <dev>]` | live window + marker edge-width meter to lock a barrel-focused lens; `--ref` shows a known-good camera beside it | 2 |
| `calibrate_uvc.py <dev> [--name] [--model rational] --square-mm --marker-mm` | live ChArUco intrinsics → `calib/<name>_intrinsics.json` | 3 |
| `record_session.py --glove --dataset D --num_episodes N` | multi-camera session: pre-flight, ritual, episode marks, width holds | 4 |
| `uvc_record.py <name> [dev] [WxH] [fps]` | one camera's raw MJPEG + wall clocks (launched by record_session) | 4 |
| `detect_parallel.py <video.mkv> [--keep-27] [--workers N]` | ArUco 4x4 + AprilTag 16h5 detection caches | 5 |
| `bundle_dodeca.py --npz= --intr= --marker-mm= --ball-ids=` | pass 2: free bundle of the ball's faces | 6 |
| `bundle_dodeca_ideal.py --npz= --init= --intr= --marker-mm= --out=` | pass 3: dodecahedron-prior refinement → the bundle the pipeline uses | 6 |
| `calibrate_gripper_bundle.py <stem> --bundle= --marker-map= --out=` | gripper-back markers in the ball's body frame | 6 |
| `session_check.py <dataset> --unit <unit>` | per-episode pose/width coverage, hold sanity, tile visibility → OK / problems | 4b |
| `tiles_pose.py layout\|solve …` | export the workspace-tile layout from a static camera; camera→world pose from the tiles | 4b |
| `pack_session.py <dataset> … --unit <unit> [--dest]` | hand-off manifest + rsync line (sessions are 7–14 GB) | 4b |

Libraries (not commands): `labels_lib.py` — the label stack (fused ball
pose, body-frame aperture estimators); `forward_width.py` — physical-mm
tail/base image constraints for the offline width review; `uvc_ritual.py` — per-camera
exposure/gain/focus controls, selected by device name.

## Where to edit

This directory is the single source of truth for these scripts and for the
measured unit configs in `calib/`. The private experiment tree (`pilot`)
links to them with relative symlinks rather than keeping copies, so a change
committed here is immediately what the robot-side pipeline runs. Robot-only
tooling (hand-eye, gap bench, dataset construction, training) is not here.

## Troubleshooting (things that actually happened)

- **Wrist fisheye washed out / saturated, tags not decoding, controls look
  fine.** The Arducam 1080P Low Light module's ISP can get stuck in a state
  where exposure/brightness writes are accepted, read back correctly, and
  make no difference to the image (blacks lifted to ~180 while the scene
  still has real blacks). **Replug the camera.** Seen repeatedly on two
  units; nothing in software reaches it. `scripts/tag_contrast.py <dev>`
  shows per-tag black/white levels live: healthy tags read blacks ≲40,
  contrast ≳150 under lab light.
- **A recorder from an earlier run is still alive.** If a session is killed
  rather than finished, its `uvc_record` processes can outlive it, keep the
  cameras, and keep writing `data/raw/.<stem>_seg*.mkv`; the next run with
  the same dataset name then reads *those* segments as if they were its own
  (pre-flights pass, deliverables are stitched from the wrong process).
  `record_session.py` now refuses to start while a recorder for the stem is
  alive (`kill -TERM <pid>` first — it finishes its remux). Check with
  `pgrep -af uvc_record.py` if anything looks odd.
- **A camera dropped mid-session.** In glove mode a watchdog now beeps if a
  stream stops producing segments for 90 s. Episodes recorded after the drop
  have no stream from that camera; the build gates them per stream.
- **Episode marks across runs.** `<dataset>_glove_episodes.json` is
  cumulative: a second run with the same name continues the numbering and
  never overwrites. (Before 2026-10-06 it overwrote — 30 episodes' marks
  were lost and had to be reconstructed from marker visibility.)
- **`input()` never responds at the prompt.** The terminal is in raw mode
  from an earlier process; the glove loop runs `stty sane` first and prints
  the tty flags it found. Also: the preview windows take keyboard focus —
  click the terminal.
