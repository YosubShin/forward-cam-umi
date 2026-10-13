# Forward width progress — 2026-10-13

The user's physical Yosub glove recordings retain historical `glove_yuan_*`
filenames. The initial reviews below held their original local geometry fixed;
the later measured-size revision and fresh-recording review are recorded
separately. The offline estimator jointly fits a moving/fixed marker pair's
image corners, instead of subtracting independently estimated tag depths.

Training: `glove_yuan_width_02`, measured holds 0, 32.3, and 64.6 mm.
Verification: `glove_yuan_width_verify_01`, a separate 57.9 mm hold.
Only training data selects and fits estimators. The frozen model is reloaded
before verification. No prediction clipping or temporal smoothing is applied.

| Verification metric | Original estimator | Joint image estimator | Gate |
| --- | ---: | ---: | ---: |
| Signed median error | -1.23 mm | +0.08 mm | absolute value <=1 mm |
| P95 absolute error | 11.23 mm | 2.36 mm | <=2 mm |
| Width coverage | 91.7% | 83.9% (689/821 frames) | >=70% |

Coverage and median error pass. P95 still fails by 0.36 mm; this checkpoint does
not certify forward-width accuracy. Missing detections and rejected solves stay
in the coverage denominator. Moving-width accuracy remains untested.

Selected local pairs: 44–46, 45–47, 48–46, and 49–47. Candidate checks require
three apertures, at least 40 paired observations per aperture, a nearby base
reference, optimizer convergence, and physical-mm time-blocked holdout gates.
Single-marker and cross-jaw estimators are excluded.

Reproduce with the original local recordings and fixed calibration files:

```bash
.venv/bin/python scripts/forward_width_check.py \
  glove_yuan_width_02 glove_yuan_width_verify_01 --unit yuan \
  --out tmp/forward_width_yuan_v2.local.json
```

Exit status 2 denotes an accuracy-gate failure; it still saves the report,
frozen model, and per-frame predictions in `tmp/`. Those device-specific artifacts
and videos are kept locally. The legacy normalized actuator estimator is unchanged.

Regression checks:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_forward_width.py' -v
```

All nine regressions pass, including metric scale, perspective changes, unseen
apertures, missing/ambiguous IDs, repeated hold boundaries, and coverage accounting.

## P95 investigation after the checkpoint

The initial checkpoint was committed and pushed as `4a11c3b` before this review.
Per-frame analysis found two error modes. Some converged joint solves remain in
a poor local pose minimum despite a large image residual. Other frames have
only 0.3–0.4 px image residual but 2–3 mm width error, so a residual threshold
alone cannot certify metric accuracy.

The solver now retries SQPnP initializations at normalized trained-stroke
positions 0, 0.5, and 1 if the warm solve fails or exceeds the 1 px Huber scale.
It selects the lowest Huber cost among converged positive-depth solutions.
The width is still free to leave the training range. At frame 512, this changes
the fused estimate from 61.99 to 58.65 mm for the 57.9 mm hold; frame 711 changes
from 54.37 to 56.21 mm. The frozen motion parameters need no change for recovery.

New model artifacts rank candidate faces by training time-blocked p95. Faces
sharing a fixed tag have correlated pose errors, so only the best available face
per reference votes in the fused median; another face remains a fallback. The
training ranking favors 44 over 48 for reference 46 and 49 over 45 for reference
47. Scores are frozen before the verification replay. Old unranked artifacts
retain flat-median fusion.

| 57.9 mm replay | Checkpoint | Solver recovery | Recovery + reference ranking | Gate |
| --- | ---: | ---: | ---: | ---: |
| P95 absolute error | 2.3602 mm | 2.2016 mm | 2.0836 mm | <=2 mm |
| Signed median error | +0.0824 mm | -0.1613 mm | -0.2039 mm | absolute value <=1 mm |
| Error RMS | 1.3896 mm | 1.3362 mm | 1.2587 mm | diagnostic |
| Coverage | 689/821 (83.9%) | 691/821 (84.2%) | 691/821 (84.2%) | >=70% |

P95 remains 0.0836 mm above the gate. No extra frame rejection, clipping,
temporal smoothing, or changed calibration/capture settings explain the gain.
The 64.6 mm training-hold replay also has p95 2.1949 mm, indicating a remaining
tail-error problem beyond the one verification aperture.

Other diagnostic attempts were not adopted:

- Body-pose rotation priors at 1, 3, and 5 degrees, chosen by training-blocked
  RMS: the selected 1-degree prior yields verification p95 2.2155 mm.
- Taking a median within each reference before fusing references: p95 2.2029 mm.
- First-order image-row motion timing correction, fitted on training data:
  verification p95 worsens from 2.2019 to 2.2356 mm on the same 670 frames with
  neighboring observations. The fitted coefficient is not a camera readout-time
  measurement.

Reference ranking is an engineering candidate, not a new blind qualification.
A cached training-block comparison had p95 2.8414 mm for flat fusion versus
2.8841 mm for the ranked-reference policy, so the validation improvement is not
uniform across datasets. Although model parameters and ranking scores use only
training data, the 57.9 mm recording has now been inspected repeatedly while
developing inference policies. Fresh unseen holds are required for qualification.

Reproduce the latest review:

```bash
.venv/bin/python scripts/forward_width_check.py \
  glove_yuan_width_02 glove_yuan_width_verify_01 --unit yuan \
  --out tmp/forward_width_yuan_v4.local.json
```

Reports include solver policy and implementation hashes in addition to frozen
model and input hashes. All 14 width regressions pass, including wrong planar
pose recovery, a degenerate cold seed, correlated-reference fusion, missing-face
fallback, and historical model compatibility. Existing marker-map regressions
also pass (7 tests), for 21 passing local regressions in total.

Next useful experiments are corner localization/pose-conditioning diagnostics
and testing the straight-travel/constant-marker-orientation assumption at more
intermediate apertures and viewing angles. Three measured apertures provide
limited evidence for that motion model. Changes should preserve coverage and
the 70% / 1 mm / 2 mm gates, then be assessed on fresh unseen holds; this review
does not justify treating the remaining tail error as solved.

## Physical marker measurements — 2026-10-13

The user confirmed that the physical glove is Yosub's; the local recordings
retain historical `glove_yuan_*` filenames. The 2 mm verification P95 gate is
a local engineering diagnostic, not an accuracy requirement established by
Yosub. The user accepts the previous 2.0836 mm result, but that acceptance does
not validate metric geometry inherited from the wrong physical unit.

Black-square dimensions include the black border and exclude white margins:

| Marker | Horizontal | Vertical | Square-model edge |
| --- | ---: | ---: | ---: |
| Base 46 | 11.6 mm | 11.6 mm | 11.50 mm base group mean |
| Base 47 | 11.5 mm | 11.3 mm | 11.50 mm base group mean |
| Tail 44 | 11.7 mm | 11.9 mm | 11.80 mm |
| Tail 45 | 11.7 mm | 12.0 mm | 11.85 mm |
| Tail 48 | 11.9 mm | 11.7 mm | 11.80 mm |
| Tail 49 | 11.5 mm | 11.6 mm | 11.55 mm |

Six ball-edge readings (14.4, 14.9, 14.9, 14.7, 14.4, 14.4 mm) give a group
mean of 14.6167 mm. Individual ball face IDs were not supplied. Base 40/41 use
the assumed base group mean; they were not measured. Other tail sizes remain
inherited. The existing square-marker model uses means and retains all raw
readings; these readings do not establish whether each sticker is rectangular.
Actual wrist IDs 38/39 remain 5.6/5.7 mm in the 4x4 dictionary.

The separate `marker_map_glove_yosub.local.json` records explicit measurement
provenance. The measured ball bundle uniformly rescales the previous local fit
by 14.6167/15.18, preserving its pixel projections under the corresponding
camera-translation scale. The base bundle is refitted against that ball from
the existing calibration take (165 views); its converged coordinate RMS is
1.0018 px. This is local measured geometry, not an imported author bundle.
Three other recordings have base/ball corner P95 errors of 2.1664, 1.1198,
and 1.8199 px after per-frame camera-pose fitting; those pixel checks do not
certify absolute width accuracy.

All nine backed-up original maps, bundles, camera intrinsics/configuration, and
capture snapshots remain byte-identical. The offline CLI's `--measurement-map`
option checks identities and records both the revised map and immutable capture
snapshots. Model selection now checks paired observations at every training
aperture before selecting the nearest local base; unsupported base 40/41 no
longer mask supported 46/47. The original 40-frame and 50 mm gates remain.
Failed selection saves candidate reasons and explicitly skips verification.

After measuring all four tails, all four tail/base models pass blocked training
selection. Their holdout P95 errors are 2.70, 2.84, 3.71, and 2.55 mm for
44–46, 45–47, 48–46, and 49–47 respectively. These training figures are distinct
from the separate 57.9 mm replay. Width and marker-map regressions pass (17 + 7).

| 57.9 mm diagnostic replay | Previous local geometry | Measured Yosub geometry |
| --- | ---: | ---: |
| Signed median error | -0.2039 mm | +0.0676 mm |
| Error RMS | 1.2587 mm | 1.1994 mm |
| P95 absolute error | 2.0836 mm | 2.1668 mm |
| Coverage | 691/821 (84.2%) | 691/821 (84.2%) |

Both reviews use the same 691 valid hold frames. In the measured replay,
57 frames exceed 2 mm absolute error, one exceeds 3 mm, and none exceeds 5 mm;
maximum absolute error is 3.4292 mm. Median absolute error is 0.9656 mm, which
is distinct from the near-zero signed median bias. The 0, 32.3, and 64.6 mm
training replays have P95 errors 1.0433, 1.2316, and 2.4941 mm respectively.
The original 2 mm diagnostic gate remains unmet; the report exits with code 2
to communicate this numeric result, not an execution failure.

Corrected dimensions improve systematic bias and RMS but do not remove the
remaining tail errors. Retain the measured dimensions rather than selecting
known-wrong geometry for its slightly better P95. No new frame rejection or
verification-label fitting was used. The corrected files remain separate
offline artifacts; existing capture settings and maps have not been replaced.

Reproduce the measured-size replay:

```bash
.venv/bin/python scripts/forward_width_check.py \
  glove_yuan_width_02 glove_yuan_width_verify_01 --unit yosub \
  --measurement-map calib/marker_map_glove_yosub.local.json \
  --out tmp/forward_width_measured_yosub_v3.local.json
```

Revisiting this previously inspected verification recording is a diagnostic;
fresh widths and moving-width recordings remain necessary for qualification.

## Fresh recording with frozen model — 2026-10-13

`glove_yosub_width_verify_02` records three 32.2 mm spacer holds and one fully
closed 0 mm hold. The user confirmed the nominal order straight, left, right,
closed. Replay uses the already frozen `forward_width_measured_yosub_v3` model
without training, model selection, clipping, smoothing, or inference changes.
The four inference implementation hashes and all used geometry hashes match
the frozen artifact. The new snapshot agrees with the measured marker map.
Only the fish USB ID changed before recording; physical roles were confirmed
from live frames and USB topology. Camera modes and intrinsics were preserved.

| Recorded segment | Signed median error | Absolute error P95 | Width coverage |
| --- | ---: | ---: | ---: |
| 32.2 mm, nominal straight | -0.9358 mm | 4.3648 mm | 681/701 (97.1%) |
| 32.2 mm, nominal left | -0.7297 mm | 3.0366 mm | 500/504 (99.2%) |
| 32.2 mm, nominal right | +2.1423 mm | 3.5574 mm | 335/384 (87.2%) |
| 0 mm, fully closed | -1.2780 mm | 2.4069 mm | 378/384 (98.4%) |

All four fail the original 2 mm diagnostic P95 gate. The right and closed
segments also fail the 1 mm absolute median-bias gate. Their missing and rejected
frames remain in the denominator. This fresh recording demonstrates larger
view-dependent errors than the previous single 57.9 mm view. Because 32.2 mm
is near trained 32.3 mm and 0 mm is a trained aperture, it does not qualify
unseen aperture accuracy.

Visual review of frames at 10%, 50%, and 90% of each segment confirms substantial
within-hold rotation. The nominal pose labels describe the user's sequence;
they are not fixed pose measurements. Estimated average body orientations for
the left and right segments differ by about 106 and 73 degrees from the first
segment, including all rotation axes. These are estimator-derived diagnostics,
not externally measured yaw angles. The left midpoint loses the ball below
the image boundary. Keeping these segments reveals fixed-width robustness to
changing pose; it is not the intended mild-angle static precision protocol.

Reference-only diagnostics do not justify an added rejection gate. If both base
references were required, coverage would fall to 31.1%, 6.9%, 63.3%, and 34.4%
respectively. On the right segment, even both-reference predictions have median
error +2.1724 mm and P95 3.6146 mm. Thus requiring more visible tags alone does
not solve the view-dependent bias. The baseline report remains unchanged.

All streams have zero native drops. The whole-recording check flags timestamp
gaps (5 forward, 9 head, 2 wrist), but none exceeds 1.5 frame periods inside the
four measured holds. This does not certify optical synchronization. Wrist
38/39 pair visibility is 99.75%, 99.83%, 98.17%, and 99.77%; that is visibility,
not a new wrist-width accuracy measurement. Head-camera two-ball-face visibility
is 46.1%, 47.1%, 85.2%, and 55.5%, reported separately from forward inference.

Artifacts:

- `tmp/glove_yosub_width_verify_02_frozen_review.local.json` and `.npz`:
  frozen-model predictions, per-pair errors, input hashes, and per-hold gates.
- `tmp/verify02_reference_diagnostics.local.json`: one/two-reference diagnostics.
- `tmp/verify02_pose_stability.local.json`: estimated within-hold pose variability.
- `tmp/verify02_within_hold_views.local.jpg`: actual early/middle/late views.
- `tmp/verify02_head_marker_coverage.local.json`: supplementary head detections.

Reproduce the diagnostic without fitting:

```bash
.venv/bin/python tmp/replay_frozen_forward_width.local.py \
  glove_yosub_width_verify_02 \
  --model tmp/forward_width_measured_yosub_v3.local.model.json \
  --out tmp/glove_yosub_width_verify_02_frozen_review.local.json
```

Before interpreting the remaining error as a static calibration defect, record
a short fixed-pose 32.2 mm and fully closed control with the arm supported.
Set the angle before starting each hold; stop the hold before changing it.
Keep this rotating recording as a robustness diagnostic. Further software work
should investigate per-marker dimensions and non-square-marker assumptions,
then assess changes without refitting on this recording's verification labels.

## Fixed-pose control after reusing the dataset name — 2026-10-13

The next recording reused `glove_yosub_width_verify_02` and contains a 0 mm hold
followed by a 32.2 mm hold. The preceding rotating videos, timestamps, metadata,
and detection caches were automatically moved to per-camera `.take1` names.
Their timestamps and available detection hashes match the previous report.
The recorder overwrites the dataset-level width log on reuse: the previous
dense event log is no longer available, although the exact hold intervals and
labels remain in the original frozen report. `verify02_take1_archive_link`
records this distinction. The new log, snapshot and recording-check report are
backed up separately before analysis. Use a new dataset name for future takes.

This control uses the same frozen measured-Yosub model, same inference code,
and unchanged camera geometry. The new result is saved separately as
`tmp/glove_yosub_width_verify_02_static_frozen_review.local.json` and `.npz`.
The preceding rotating report and its predictions have not been overwritten.

| Fixed-pose control | Signed median error | Absolute error P95 | Coverage |
| --- | ---: | ---: | ---: |
| 0 mm | +0.4231 mm | 0.5281 mm | 344/344 (100%) |
| 32.2 mm | +0.3462 mm | 0.5099 mm | 397/397 (100%) |

Both controls pass the existing coverage, median-bias and P95 gates. Maximum
absolute errors are 0.6094 and 0.6157 mm respectively. Estimated within-hold
orientation deviation P95 is 0.138 and 1.204 degrees, and translation deviation
P95 is 0.91 and 3.33 mm. Early/middle/late images agree with the fixed-pose intent.
Forward marker pairs are visible in every hold frame; wrist pair visibility is
100% and 98.68%. Forward and wrist hold windows contain no timestamp gaps over
1.5 frame periods; head capture has two such gaps during the closed hold.
Head marker detection was not repeated because this is a forward-width control.

Passing fused static results do not establish per-reference accuracy. At 0 mm,
44–46 predicts a median +2.0351 mm while 49–47 predicts -1.2066 mm. At 32.2 mm,
their signed biases are +1.9398 and -1.2754 mm. Both references are available
throughout, so fusion cancels part of these opposite biases. This is consistent
with instability when a reference disappears, although the earlier right-view
both-reference bias shows that missing tags are not the sole cause.

The supported conclusion is good fixed-view fused accuracy at these two
apertures, not general moving-view accuracy or unseen-aperture qualification.
Do not subtract a verification-derived zero offset or refit on these controls.
A useful next independent test is an approximately 45 mm rigid spacer in the
same fixed pose, using its actual caliper width and a fresh dataset name such
as `glove_yosub_width_verify_03`.

Reproduce this control with a fresh output name:

```bash
.venv/bin/python tmp/replay_frozen_width_take.local.py \
  glove_yosub_width_verify_02 --take 2 \
  --model tmp/forward_width_measured_yosub_v3.local.model.json \
  --out tmp/glove_yosub_width_verify_02_static_repeat.local.json
```

## Independent 59.1 mm fixed-view hold — 2026-10-13

The following take again reused the dataset name. It records a 59.1 mm hold,
distinct from the training apertures 0, 32.3, and 64.6 mm. The preceding static
control streams and caches were archived as `.take2`. Its exact width log,
marker snapshot and recording-check report were recovered from the verified
pre-analysis backup and saved as dataset-level `.take2` files. An archived-take
replay reproduces all preceding prediction arrays within 1e-8; the old static
and rotating reports remain separate. The new 59.1 mm log is also backed up.

Frozen-model result:

| Metric | 59.1 mm hold |
| --- | ---: |
| Median prediction | 59.0121 mm |
| Signed median error | -0.0879 mm |
| P95 absolute error | 0.2879 mm |
| Error RMS | 0.1570 mm |
| Maximum absolute error | 0.4331 mm |
| Width coverage | 463/463 (100%) |

All existing numerical gates pass. The model, four inference implementation
files and geometry inputs match their frozen hashes; no training, offset
correction, clipping or smoothing is performed. Visual samples show a stable
table-supported gripper. Estimated pose deviations P95 are 0.100 degrees and
0.278 mm. All three measured hold windows have zero timestamp gaps over 1.5
frame periods; wrist 38/39 are both visible in all 529 wrist hold frames.

Reference 46 is never decoded, so 44–46 and 48–46 cannot vote. Both 45–47 and
49–47 are available throughout, and the frozen training ranking selects 49–47.
Its signed median error is -0.0879 mm; the unused 45–47 candidate has +1.2283 mm
median error and 1.4112 mm P95. Thus this passing result uses one trained-ranked
reference and does not depend on cancellation between base 46 and base 47.
Do not infer that two references are always required or that single-reference
predictions are generally accurate at other viewing angles.

The 0, 32.2 and 59.1 mm fixed-view controls now pass, while the earlier rotating
recording still fails. A useful next test is to retain the 59.1 mm spacer and
record separate stable holds after small left/right orientation changes. Set
each angle before starting the hold; do not rotate during it. Use a fresh name
such as `glove_yosub_width_verify_03` to preserve independent width logs.

Artifacts: `tmp/glove_yosub_width_verify_02_59p1_frozen_review.local.json` and
`.npz`, `tmp/verify02_59p1_pose_stability.local.json`,
`tmp/verify02_59p1_pose_views.local.jpg`, and
`tmp/verify02_take2_replay_equality.local.json`.

Reproduce this take before another same-name recording, using a fresh output:

```bash
.venv/bin/python tmp/replay_frozen_59p1_width.local.py \
  glove_yosub_width_verify_02 \
  --model tmp/forward_width_measured_yosub_v3.local.model.json \
  --out tmp/glove_yosub_width_verify_02_59p1_repeat.local.json
```

## Two static right-turn 59.1 mm holds — 2026-10-13

The fresh `glove_yosub_width_verify_03` recording contains two 59.1 mm holds.
The user corrected the initial left/right description: the first is a larger
right turn and the second a smaller right turn. This correction is recorded in
`tmp/verify03_pose_diagnostics.local.json`. The original numerical replay stays
unchanged, with its generic `unspecified` pose labels; the diagnostic file is
the authoritative pose annotation. No left-view or new baseline hold was
recorded in this take.

The same frozen measured-Yosub model and all its inference/geometry hashes were
verified. Width labels were used only for scoring and never supplied to inference.
No fitting, offset adjustment, smoothing, clipping, or capture-setting change
was made. Input sidecars were backed up before detection and replay.

| User-confirmed pose | Predicted median | Signed median error | Absolute error P95 | Width coverage |
| --- | ---: | ---: | ---: | ---: |
| Larger right turn | 63.9752 mm | +4.8752 mm | 5.0807 mm | 414/423 (97.87%) |
| Smaller right turn | 61.2144 mm | +2.1144 mm | 2.3014 mm | 330/331 (99.70%) |

Both holds pass the original 70% coverage diagnostic and fail the original
1 mm median-bias and 2 mm P95 diagnostics. These remain local engineering
targets, not an author-specified accuracy guarantee. The larger-turn error is
substantial even if the user accepts slightly more than 2 mm P95.

Both holds use only 45–47: 46 and 49 are not decoded in either hold. Raw 45–47
visibility is 97.87% and 100%; one second-hold frame has no valid body pose.
The ball has at least two decoded faces in every hold frame. Wrist 38/39 are
visible together in every wrist hold frame (484/484 and 378/378). Head marker
detection was not repeated for this forward-width check.

Estimated full three-axis orientation differences from the prior 59.1 mm
baseline are 93.13 and 40.49 degrees. These are model-derived orientation
differences, not externally measured yaw angles; neither qualifies the proposed
approximately 10-degree test. Early/middle/late images agree with stable poses.
Within-hold orientation-deviation P95 is 0.169 and 0.259 degrees; translation
deviation P95 is 1.668 and 0.716 mm. All three camera hold windows contain zero
timestamp intervals over 1.5 frame periods, so the whole-recording startup and
transition gap warnings do not explain the steady hold bias.

The central 90% prediction ranges are 63.769–64.181 and 61.028–61.401 mm,
respectively: the errors are sustained offsets rather than isolated outliers.
The earlier baseline selected 49–47 and achieved 0.288 mm P95. Its unused
45–47 candidate already had +1.228 mm median bias and 1.411 mm P95; the same
45–47 candidate now has larger biases at the changed poses. Both viewing angle
and selected marker face matter. These observations do not identify which
geometry or corner-detection assumption causes the bias.

Further analysis should examine 45–47's view-dependent geometry and corner
residuals using existing recordings. Removing a constant offset or changing a
training ranking on these verification labels would contaminate independent
validation. New baseline and approximately 10-degree right/left stable holds
would separately test the intended small-angle range; this recording supports
two larger right-turn views only.

Artifacts: `tmp/glove_yosub_width_verify_03_frozen_review.local.json` and
`.npz`, `tmp/verify03_pose_diagnostics.local.json`, and
`tmp/verify03_pose_views.local.jpg`. Reproduce with a new output name:

```bash
.venv/bin/python tmp/replay_frozen_width_take.local.py \
  glove_yosub_width_verify_03 \
  --model tmp/forward_width_measured_yosub_v3.local.model.json \
  --out tmp/glove_yosub_width_verify_03_repeat.local.json
```

## View-dependent 45–47 bias investigation — 2026-10-13

All camera/marker calibration JSON files, capture configuration, the four frozen
inference implementation files, recorder entry point and v3 model remain
byte-identical. This investigation creates read-only diagnostics and separately
named experimental models under `tmp`; it does not replace production inference.
The preceding baseline detection cache is now archived as `.take3`, and its hash
matches the original 59.1 mm replay report. The `.take1`, `.take2` and fresh
verify03 detection hashes were also checked against their original reports.

The 45/47 crop inspection confirms the intended physical IDs. Single-frame
SUBPIX and unrefined corner probes do not improve the observed bias; this does
not rule out systematic corner errors. The moving and fixed tags span roughly
50–55 pixels per side in these views.

An 18-frame-per-hold diagnostic compares several image-only initializations.
An initial probe incorrectly combined the already fitted width with the
unfitted body pose instead of including the actual production analytical seed.
It found a truth-closer but worse-image-fit solution; this is not an inference
improvement. `solver_diagnosis.json` is the corrected result, and the initial
artifact is explicitly superseded in `conclusion.json`. Full hold replays of
both planar-branch enumeration and always-on global stroke seeds reproduce the
two static right-view biases; some older rotating holds worsen slightly.

The corrected sampled coordinate RMS is about 1.099 pixels in the large-right
view and 0.634 pixels in the smaller-right view. Fixing width to the independently
known 59.1 mm and optimizing only nuisance pose raises those RMS values to about
1.475 and 0.752 pixels. Thus a biased width can fit the image better under the
current geometry. Low image residual is not a certified metric accuracy bound.
Forcing the unchanged fused body pose or adding rigid ball corners produces
larger width biases, so these constraints are not adopted.

The two separately learned faces on base 47 have travel vectors differing by
1.328 mm over the trained 64.6 mm width span, despite being faces of one moving
holder. A shared-travel candidate jointly fits 45/49 using original training
holds only. Its blocked training holdout has 100% coverage, 1.330 mm RMS and
3.035 mm P95. After the full training-only refit it improves the smaller-right
P95 but substantially regresses the previously accurate baseline:

| Existing hold | Frozen P95 | Shared-travel P95 | Center-path P95 / coverage |
| --- | ---: | ---: | ---: |
| Baseline 59.1 mm | 0.288 mm | 2.632 mm | 0.288 mm / 100% |
| Larger right turn 59.1 mm | 5.081 mm | 5.451 mm | No valid estimates / 0% |
| Smaller right turn 59.1 mm | 2.301 mm | 1.969 mm | 1.171 mm / 99.40% |
| Static 0 mm | 0.528 mm | 0.381 mm | 0.528 mm / 100% |
| Static 32.2 mm | 0.510 mm | 0.612 mm | 0.510 mm / 100% |
| Original 57.9 mm | 2.167 mm | 1.934 mm | 2.187 mm / 81.49% |

The center-path candidate obtains base pose from its own square corners, choosing
the planar branch closest to the unchanged fused body pose. It learns a metric
tail-center travel path only from the original 0/32.3/64.6 mm training holds,
without assuming the moving face's orientation. Diagonal intersection is applied
after corner undistortion. The frozen 49–47 preference and base-46 fusion are
retained; only fallback 45 is replaced. Its blocked training holdout has 99.76%
coverage, 1.358 mm RMS and 2.948 mm P95. At the smaller right view its signed
median error is +0.874 mm (329/331 frames). All 423 larger-right frames lack
an accepted estimate: that is a failed coverage check, not a passing accuracy
result. On the first two earlier rotating holds, fused P95 worsens from
4.365/3.037 to 5.441/4.726 mm and coverage falls to 77.03%/64.29%. This prevents
adopting it as a general replacement even though the static smaller-right result
passes all original diagnostics. Thirty synthetic prediction cases plus four
unused-aperture cases verify its projection/path mechanics with changing tail
orientation, moving-marker rectangularity, distortion and extrapolated widths.

Existing caliper readings show 45 is 11.7 by 12.0 mm and 47 is 11.5 by 11.3 mm,
while the frozen model uses square means (47 uses the base group mean). A
mechanism-only simulation projects those rectangles without writing calibration
files; both canonical horizontal/vertical assignments are tested. It produces
0.224–0.796 mm width bias in the three tested poses, insufficient by itself to
explain the real larger-right bias. This does not establish the true shapes,
corner locations, or cause of the remaining error.

No candidate improves all reviewed operating conditions. Preserve the frozen
model and original report. The evidence favors further examination of learned
tail/base geometry, print/shape/corner effects and mounting rigidity. Existing
images alone cannot uniquely attribute the error to a specific caliper reading
or physical assembly fault. These are development/regression results on already
inspected recordings, not a new independent accuracy certification.

Full comparison and input hashes: `tmp/angle_bias_45_47.local/conclusion.json`.
Mechanism probes and marker crops are in that directory. Shared-travel and
center-path artifacts are separately in `tmp/angle_bias_shared_stage.local/`
and `tmp/angle_bias_center_path.local/`; all missing frames remain in coverage
denominators. Experimental reproduction uses:

```bash
.venv/bin/python tmp/diagnose_45_47_angle.local.py
.venv/bin/python tmp/replay_width_branch_candidate.local.py
.venv/bin/python tmp/replay_width_global_candidate.local.py
.venv/bin/python tmp/prototype_width_shared_stage.local.py
.venv/bin/python tmp/prototype_width_center_path.local.py
```

These local research scripts regenerate their own experimental outputs and
expect the recorded/cache archival names checked above. They do not rewrite
the original frozen replay, raw labels or calibration inputs.

## User-selected v3 baseline and explicit static acceptance — 2026-10-13

After reporting no apparent physical installation problem, the user selected
the preceding measured-size v3 result (48 = 11.80 mm, 49 = 11.55 mm) and requested
acceptance of its 2.167 mm P95 to proceed. The experimental candidates were never
installed, so no calibration or estimator rollback is needed. The selected
frozen model SHA remains
`8432d7869c43a77975b0af29f4c0859fea49b9b137bed9f022f86ffae1d22d0d`.
Its four original inference implementation hashes are unchanged.

`scripts/forward_width_accept.py` independently replays the selected frozen
model, with an explicit **2.2 mm P95 acceptance limit**, original coverage >=70%
and absolute median bias <=1 mm. No fitting or changed predictions are involved.
All four output arrays reproduce the preceding verification arrays within 1e-8,
including NaN coverage positions. Boundary checks confirm 2.2 passes, 2.2001
fails, rejected frames remain in the denominator, and invalid limits fail.

| Independent 57.9 mm result | Value |
| --- | ---: |
| Signed median error | +0.06756 mm |
| Error RMS | 1.19944 mm |
| P95 absolute error | 2.16676 mm |
| Coverage | 691/821 (84.17%) |
| Explicit 2.2 mm static acceptance | PASS |

Original 2 mm diagnostic failures and the larger-view failures remain intact.
The 64.6 mm training hold's 2.494 mm P95 is also retained; the static acceptance
decision does not declare all apertures or viewing angles qualified. Measurement
uncertainty may contribute but has not been uniquely identified from the images.

Acceptance report: `tmp/forward_width_measured_yosub_v3_acceptance_2p2.local.json`
and `.npz`. Decision record: `tmp/forward_width_baseline_selection.local.json`.
Calibration/marker/capture inputs remain unchanged. Proceed with one 10–15 second
normal grasp/move/release episode using fresh name `glove_yosub_task_smoke_01`;
then review recording integrity, moving-marker coverage and width continuity
before increasing collection volume. This is an end-to-end capture test, not
an automatic installation of the metric model into the robot actuator pipeline.

## First normal 12.24 second grasp episode — 2026-10-13

`glove_yosub_task_smoke_01` records one normal episode, distinct from measured
width holds. All three streams finalize normally. Whole-recording transport
still reports startup/wait-period gaps, including the wrist's 900 ms maximum,
but **none of those gaps overlaps the episode's internal frame intervals**.
The recording-check report is preserved rather than overridden.

| Actual episode window | Forward | Head | Wrist |
| --- | ---: | ---: | ---: |
| Recorded frames | 319 | 320 | 365 |
| Measured frame rate | 26.272 fps | 26.271 fps | 30.003 fps |
| Maximum timestamp interval | 39.03 ms | 38.53 ms | 33.57 ms |
| Intervals over 1.5 nominal frame periods | 0 | 0 | 0 |
| Native drops in complete recording | 0 | 0 | 0 |

Last recorded frames are about 94/67/72 ms before the episode's end mark. This
small tail difference is reported explicitly. Timestamp health does not certify
optical synchronization. Queue peaks are below configured capacities; no queue
overflow or recorder failure is present.

The selected unchanged v3 forward estimator and existing frozen wrist affine
mapping are replayed without fitting. The wrist uses the historically named
`calib/fisheye_glove_yuan_intrinsics.json`, with its hash verified against the
earlier independent wrist review; physical unit and marker snapshot are Yosub,
4x4 IDs38/39 at 5.6/5.7 mm. Raw marker-center separation is converted using the
already frozen slope/intercept, never treated directly as jaw gap.

| Episode coverage | Result |
| --- | ---: |
| Forward body pose | 319/319 (100%) |
| Forward metric width | 315/319 (98.75%) |
| Head body pose | 294/320 (91.88%) |
| Wrist unique pair and metric width | 343/365 (93.97%) |

All exceed the existing 70% build coverage gate. Wrist pair coverage is below
the earlier 95% static wrist-validation target; missing frames are retained as
NaNs and no interpolation is applied. The forward 45–47 pair is visible in
99.06% of episode frames, 44–46 in 67.40%; the frozen training-ranked fallback
remains unchanged. The early/middle/late contact sheet confirms the normal
grasp/move/place/release sequence.

Using nearest recorded wrist timestamps within 1/60 second gives 294 paired
predictions, maximum timestamp difference 16.27 ms, signed median forward-minus-
wrist difference -0.486 mm and absolute-difference P95 4.487 mm. These are
cross-view consistency measurements, not dynamic ground-truth errors. The
width curves share their main opening/closing movements, with view-dependent
differences while the object is held. The largest adjacent forward change is
8.219 mm in 38 ms; the large changes overlap actual open/close phases and often
missing wrist tags, so they are not automatically classed as solver outliers.
No 2.2 mm static accuracy pass is inferred for this moving episode.

`session_check.py` exits 1 solely for fewer than two workspace 36h11 tags among
IDs1–5. Its separate direct tile detection finds none in sampled forward frames;
the missing tile result is not caused by the glove detector's dictionary choice.
This recording is useful as a capture/coverage smoke test, while world-coordinate
alignment remains incomplete. Before producing world/robot-aligned training
data, configure the workspace reference markers and their layout without
altering the existing camera or gripper calibration.

Artifacts: `tmp/glove_yosub_task_smoke_01_review.local.json` and `.npz`,
`tmp/task_smoke_01_summary.local.json`, `tmp/task_smoke_01_action_views.local.jpg`,
`tmp/task_smoke_01_width_curves.local.png`, and the session-check log. All source
and calibration hashes were checked unchanged after review. Under standing disk
cleanup authorization, four inactive August Xcode intermediate directories were
removed, freeing 4.037 GB and bringing free space to 33.605 GB; every raw recording
and calibration file was preserved. Cleanup manifest:
`tmp/build_cache_cleanup_task_smoke_01.local.json`.

## Moving-width quality gate and data handoff — 2026-10-13

The three `glove_yosub_task_smoke_02` episodes last 13.62, 12.13 and 14.17 seconds.
Their forward/wrist episode windows contain no long capture intervals; the head
has one approximately 76 ms interval in each of the latter two episodes. Whole-
recording checks still report startup gaps and the forward camera's approximately
26.27 fps versus the requested 25 fps. These warnings remain in the original
reports; timestamp sidecars, not container nominal fps, determine alignment.
All streams have zero reported native drops and recording queues stay below
their bounded capacities. Optical synchronization remains unvalidated.

The unchanged v3 solver admits several large 45–47 width spikes despite low
pixel residuals. `forward_width_quality.py` rejects candidates with excessive
residual-scaled local sensitivity before training-ranked reference fusion.
The local v1 limit is 3.34794 mm, derived from three times the training-hold
P95 sensitivity across 1,973 finite training frames, with a 0.3 px noise floor.
Verification labels and moving recordings were not used to derive this limit.
This sensitivity is a numerical quality proxy, not an absolute error bound.

| Episode | Raw width coverage | Qualified coverage | Raw maximum adjacent step | Qualified maximum step |
| --- | ---: | ---: | ---: | ---: |
| 0 | 98.04% | 94.41% | 22.16 mm | 5.55 mm |
| 1 | 99.06% | 94.03% | 7.70 mm | 7.70 mm |
| 2 | 96.49% | 88.38% | 21.78 mm | 9.10 mm |

The four previously inspected spikes are rejected. Remaining large steps overlap
visibly rapid closing, but wrist tags are missing at their peaks, so dynamic
absolute accuracy is not certified. Rejected estimates remain NaN, with missing
runs up to eight frames (267 ms between the first and last invalid timestamps).
Step statistics never bridge a missing frame. The gate does not correct the
existing 3–5 mm cross-view differences or larger-angle systematic bias.

The independent 57.9 mm static predictions remain unchanged: P95 2.16676 mm,
RMS 1.19944 mm, signed median error +0.06756 mm and 691/821 coverage. This passes
the explicitly selected 2.2 mm static limit; the original 2 mm diagnostic and
other-view failures remain preserved. Original replay arrays agree within
1e-8, and camera, marker, geometry and capture inputs were not changed.

The handoff exports ball-to-local-table-world transforms (translation in metres),
physical jaw widths (millimetres), capture timestamps and validity masks. The
local world uses tag 5 as origin, +x toward tag 4 and +z upward. Its measured
5–4 distance is 481 mm, versus approximately 728 mm in the upstream workspace;
the receiver must supply a compatible world-to-robot transform, ball-to-tool
transform and width-to-actuator mapping. These are not finalized robot actions.
Cached receiver replay was exercised in an isolated relocated directory on the
sender Mac; a real receiver/robot build remains outstanding.

The confirmed formal task is to grasp one of three objects and put it in the
target bin/frame. Current smoke clips show table-top diagnostic motions, not
that complete setup. A one-episode sample contains three native-resolution
MJPEG cuts with encoded-packet hashes verified against the originals. A separate
MP4/H.264 review copy retains all frames but is lossy; use the MJPEG originals
for pixel-identical processing. Generated packages, unit calibration and frozen
models remain local and are transferred separately from this code review.
