#!/usr/bin/env python3
"""Train metric forward-width models and independently replay existing caches."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

import labels_lib as L
import forward_width as W
from marker_config import load_marker_map, resolve_marker_map

ROOT = Path(__file__).resolve().parents[1]
MIN_PAIR_FRAMES_PER_APERTURE = 40


class TrainingSelectionError(ValueError):
    """Keep candidate diagnostics when no model is eligible for verification."""
    def __init__(self, candidates):
        super().__init__("No forward-width candidate passed training checks")
        self.candidates = candidates


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_marker_maps(root, datasets, unit, measurement_map=None):
    """Resolve an explicit physical-size revision without rewriting recordings.

    Capture snapshots remain immutable. Offline reprocessing can replace only
    declared dimensions and the corrected physical-unit name, never identities.
    """
    recorded = {ds: resolve_marker_map(root, ds, unit=None if measurement_map else unit)
                for ds in datasets}
    if measurement_map is None:
        return recorded, recorded
    revised = (root / measurement_map).resolve()
    data = load_marker_map(revised)
    if data.get("unit") != unit or not data.get("measurement_revision"):
        raise ValueError("Measurement revision must identify the requested physical unit and provenance")
    for path in recorded.values():
        original = load_marker_map(path)
        if (original["ball_faces"] != data["ball_faces"]
                or original["tips_dict"] != data["tips_dict"]
                or any(set(original[group]) != set(data[group])
                       for group in ("tails", "bases", "tips"))):
            raise ValueError("Measurement revision may change dimensions, not recorded marker identities")
    return {ds: revised for ds in datasets}, recorded


def measurement_holds(log):
    if log.get("units") != "mm":
        raise ValueError("Physical raw millimetre holds required")
    events = sorted((e for e in log["events"] if e["event"] == "e8_meas"),
                    key=lambda e: e["t_wall"])
    groups = []
    for e in events:
        if not np.isfinite([e["gap_mm"], e["t_wall"]]).all():
            raise ValueError("Nonfinite hold measurement")
        if (not groups or e["gap_mm"] != groups[-1][-1]["gap_mm"]
                or e["t_wall"] - groups[-1][-1]["t_wall"] > 1.5):
            groups.append([])
        groups[-1].append(e)
    # Never interpolate across spacer changes, even when a width is repeated.
    holds = [(g[0]["gap_mm"], g[0]["t_wall"], g[-1]["t_wall"])
             for g in groups if len(g) >= 2]
    if not holds:
        raise ValueError("No complete measured hold in recording")
    return holds


def gap_at(wall, holds):
    return next((g for g, start, end in holds if start <= wall <= end), None)


def load_frames(stem, K, dist, body, obj, extras):
    timestamps = read_json(ROOT / f"data/raw/{stem}_ts.json")
    times = np.array([t["wall"] for t in timestamps])
    clean = {"frame_indices": []}
    corners = {}
    with np.load(ROOT / f"calib/{stem}_dodeca_detections.npz") as detections:
        for k in detections["frame_indices"]:
            k = int(k)
            if not 0 <= k < len(times):
                raise ValueError("Detection frame index outside timestamp array")
            cs = W.unique_corners(detections[f"i{k}"], detections[f"c{k}"])
            corners[k] = cs
            clean["frame_indices"].append(k)
            clean[f"i{k}"] = np.array(list(cs), dtype=int)
            clean[f"c{k}"] = np.array(list(cs.values()))
    rows = L.body_frames(clean, timestamps, K, dist, body, obj, {}, extra=extras)
    frames = [W.Frame(r["k"], r["t_wall"], r["rvec"].ravel(),
                      r["tvec"].ravel(), corners[r["k"]]) for r in rows]
    return times, frames


def nearest_candidates(frames, labels, tails, extras, K, dist):
    minimum = min(labels)
    gaps = sorted(set(labels))
    candidates = []
    for tail, edge in sorted(tails.items()):
        points = []
        for frame, gap in zip(frames, labels):
            if gap != minimum or tail not in frame.corners:
                continue
            ok, _, tv = cv2.solvePnP(L.marker_obj(edge), frame.corners[tail],
                                     K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R = cv2.Rodrigues(frame.rvec)[0]
                points.append(R.T @ (tv.ravel() - frame.tvec))
        if not points:
            candidates.append({"tail": tail, "rejected": "no closed-hold observations"})
            continue
        center = np.median(points, axis=0)
        distances = {b: float(np.linalg.norm(center - T[:3, 3]) * 1000)
                     for b, (T, _) in extras.items()}
        counts = {b: {g: sum(tail in f.corners and b in f.corners
                              for f, value in zip(frames, labels) if value == g)
                      for g in gaps} for b in extras}
        # A nearby reference with no paired training views cannot calibrate a
        # model. Apply the existing evidence gate BEFORE choosing a reference,
        # so an unsupported closest tag cannot mask a supported local tag.
        eligible = [b for b in extras
                    if min(counts[b].values()) >= MIN_PAIR_FRAMES_PER_APERTURE]
        base = min(eligible or distances, key=distances.get)
        item = {"tail": tail, "base": base, "closed_distance_mm": distances[base],
                "reference_evidence": {str(b): {"closed_distance_mm": distances[b],
                                                "paired_frames_per_gap": counts[b]}
                                       for b in extras}}
        # A local reference cancels global pose errors; do not fit arbitrary
        # cross-jaw pairs simply because they co-occur in many frames.
        if not eligible:
            item["rejected"] = "no base has enough paired observations across all apertures"
        elif distances[base] > 50:
            item["rejected"] = "nearest base farther than 50 mm"
        candidates.append(item)
    return candidates


def train_models(frames, holds, tails, extras, K, dist):
    labeled = [(f, gap_at(f.wall, holds)) for f in frames]
    labeled = [(f, g) for f, g in labeled if g is not None]
    if not labeled:
        raise ValueError("No body poses inside measured training holds")
    labels = [g for _, g in labeled]
    gaps = sorted(set(labels))
    minimum, span = min(gaps), max(gaps) - min(gaps)
    if len(gaps) < 3 or span < 20:
        raise ValueError("Three measured apertures spanning at least 20 mm required")
    candidates = nearest_candidates([f for f, _ in labeled], labels, tails,
                                    extras, K, dist)
    models = []
    for item in candidates:
        if "rejected" in item:
            continue
        tail, base = item["tail"], item["base"]
        T, base_edge = extras[base]
        groups = []
        for gap in gaps:
            obs = [W.local_observation(f, tail, base, T)
                   for f, g in labeled if g == gap]
            groups.append([o for o in obs if o is not None])
        counts = [len(group) for group in groups]
        item["training_frames_per_gap"] = dict(zip(map(str, gaps), counts))
        if min(counts) < MIN_PAIR_FRAMES_PER_APERTURE or sum(counts) < 80:
            item["rejected"] = "too few paired observations across all apertures"
            continue
        train_obs, train_g, test_obs, test_g = [], [], [], []
        for gap, group in zip(gaps, groups):
            cut = len(group) // 2
            first, second = group[:cut:3], group[cut::2]
            train_obs.extend(first); train_g.extend([gap] * len(first))
            test_obs.extend(second); test_g.extend([gap] * len(second))
        kwargs = dict(tail=tail, base=base, tail_edge_mm=tails[tail],
                      base_edge_mm=base_edge, minimum_gap_mm=minimum, span_mm=span)
        try:
            model = W.fit_pair(train_obs, train_g, K, dist, **kwargs)
            predictions = [W.predict_pair(o, model, K, dist) for o in test_obs]
            values = [p.gap_mm if p is not None else np.nan for p in predictions]
            quality = W.error_summary(values, test_g)
            quality["test_frames"] = len(test_g)
            quality["coverage"] = quality["valid_frames"] / len(test_g)
            item["blocked_training_holdout"] = quality
            # Absolute mm gates replace the old 15%-of-stroke noise gate.
            # Candidate gates are distinct from final independent accuracy.
            if (quality["coverage"] < .70
                    or abs(quality.get("signed_median_error_mm", np.inf)) > 1
                    or quality.get("error_rms_mm", np.inf) > 2
                    or quality.get("absolute_error_p95_mm", np.inf) > 4):
                item["rejected"] = "blocked training holdout failed"
                continue
            all_obs, all_g = [], []
            for gap, group in zip(gaps, groups):
                sampled = group[::3]
                all_obs.extend(sampled); all_g.extend([gap] * len(sampled))
            # Seed the full-data refit from the already validated basin.
            model = W.fit_pair(all_obs, all_g, K, dist, seed=model, **kwargs)
        except ValueError as error:
            item["rejected"] = str(error)
            continue
        item["accepted"] = True
        model.holdout_p95_mm = quality["absolute_error_p95_mm"]
        models.append(model)
        print(f"{model.key}: accepted; blocked holdout p95 "
              f"{quality['absolute_error_p95_mm']:.2f} mm", flush=True)
    if not models:
        raise TrainingSelectionError(candidates)
    return models, candidates


def replay(times, frames, models, extras, K, dist):
    values = np.full(len(times), np.nan)
    counts = np.zeros(len(times), dtype=int)
    sensitivities = np.full(len(times), np.nan)
    transforms = {i: T for i, (T, _) in extras.items()}
    for frame in frames:
        value, available = W.predict_width(frame, models, transforms, K, dist)
        if value is not None:
            values[frame.index] = value
            counts[frame.index] = len(available)
            sensitivities[frame.index] = max(
                max(.3, p.rms_px) * p.std_mm_per_pixel for p in available.values())
    return values, counts, sensitivities


def summarize_holds(times, values, holds):
    result = []
    for gap, start, end in holds:
        mask = (times >= start) & (times <= end)
        n = int(mask.sum())
        if not n:
            raise ValueError("Measured hold has no recorded frames")
        summary = W.error_summary(values[mask], gap)
        coverage = summary["valid_frames"] / n
        checks = {"coverage_at_least_70pct": coverage >= .70,
                  "absolute_median_error_at_most_1mm":
                      abs(summary.get("signed_median_error_mm", np.inf)) <= 1,
                  "p95_absolute_error_at_most_2mm":
                      summary.get("absolute_error_p95_mm", np.inf) <= 2}
        result.append({"actual_gap_mm": gap, "recorded_frames": n,
                       "width_coverage": coverage, **summary,
                       "checks": checks, "passed": all(checks.values())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dataset")
    parser.add_argument("verification_dataset")
    parser.add_argument("--unit", required=True)
    parser.add_argument("--camera", default="b0587")
    parser.add_argument("--intr", type=Path, default=ROOT / "calib/B0587_intrinsics.json")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--gripper-bundle", type=Path)
    parser.add_argument("--measurement-map", type=Path,
                        help="Explicit measured-size revision for offline replay; preserves recorded snapshots")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.training_dataset == args.verification_dataset:
        parser.error("Verification must be a separate recording")
    args.bundle = args.bundle or ROOT / f"calib/glove_{args.unit}_dodeca_bundle.json"
    args.gripper_bundle = (args.gripper_bundle
                           or ROOT / f"calib/glove_{args.unit}_gripper_bundle.json")
    output = args.out.resolve()
    # Derived artifacts belong outside calibration/raw inputs.
    if not output.is_relative_to(ROOT / "tmp"):
        parser.error("--out must be inside this checkout's tmp directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = output.with_suffix(".model.json")
    maps, recorded_maps = review_marker_maps(
        ROOT, (args.training_dataset, args.verification_dataset), args.unit, args.measurement_map)
    marker_map = load_marker_map(maps[args.training_dataset])
    verification_map = load_marker_map(maps[args.verification_dataset])
    for field in ("tails", "bases", "ball_faces", "ball_edge_mm"):
        if marker_map[field] != verification_map[field]:
            raise ValueError("Marker identities or physical geometry differ between recordings")
    fixed = {str(p.relative_to(ROOT)): digest(p)
             for p in sorted((ROOT / "calib").glob("*.json"))}
    fixed.update({str(p.relative_to(ROOT)): digest(p) for p in maps.values()})
    fixed.update({str(p.relative_to(ROOT)): digest(p) for p in recorded_maps.values()})
    for path in (args.intr, args.bundle, args.gripper_bundle):
        fixed[str(path.resolve())] = digest(path)
    source = {}
    for ds in maps:
        for path in (ROOT / f"data/raw/{ds}_glove_log.json",
                     ROOT / f"data/raw/{ds}_{args.camera}_ts.json",
                     ROOT / f"calib/{ds}_{args.camera}_dodeca_detections.npz"):
            source[str(path.relative_to(ROOT))] = digest(path)
    K, dist = L.load_intrinsics(args.intr)
    body, obj = L.load_bundle(args.bundle)
    if read_json(args.bundle)["marker_edge_mm"] != marker_map["ball_edge_mm"]:
        raise ValueError("Ball bundle marker size disagrees with the analysis marker map")
    extras = L.load_gripper_bundle(args.gripper_bundle)
    tails = {int(i): edge for i, edge in marker_map["tails"].items()}
    if not extras:
        raise ValueError("Calibrated rigid base markers required")
    for i, (_, edge) in extras.items():
        if float(marker_map["bases"][str(i)]) != edge:
            raise ValueError("Snapshot base size disagrees with fixed geometry")
    training_holds = measurement_holds(read_json(ROOT / f"data/raw/{args.training_dataset}_glove_log.json"))
    train_times, train_frames = load_frames(args.training_dataset + "_" + args.camera,
                                          K, dist, body, obj, extras)
    try:
        models, candidates = train_models(train_frames, training_holds, tails, extras, K, dist)
    except TrainingSelectionError as error:
        for path, expected in (fixed | source).items():
            if digest(ROOT / path) != expected:
                raise RuntimeError(f"Input changed during analysis: {path}")
        report = {"reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "physical_unit": args.unit, "stage": "training_selection",
                  "training_dataset": args.training_dataset,
                  "verification_dataset": args.verification_dataset,
                  "analysis_marker_maps": {ds: str(p) for ds, p in maps.items()},
                  "recorded_marker_maps": {ds: str(p) for ds, p in recorded_maps.items()},
                  "fixed_inputs_sha256": fixed, "source_sha256": source,
                  "fixed_inputs_unchanged": True, "candidates": error.candidates,
                  "passed": False, "verification_run": False,
                  "reason": str(error)}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        print(json.dumps({"passed": False, "verification_run": False,
                          "reason": str(error), "report": str(output)}, ensure_ascii=False), flush=True)
        return 2
    solver = {"version": W.SOLVER_VERSION,
              "fusion_policy": W.FUSION_POLICY,
              "cold_start_trigger_rms_px": W.HUBER_SCALE_PX,
              "cold_start_normalized_strokes": list(W.COLD_START_STROKES),
              "selection": "Lowest Huber cost among converged positive-depth solutions"}
    implementation = {name: digest(ROOT / "scripts" / name) for name in
                      ("forward_width.py", "forward_width_check.py", "labels_lib.py", "marker_config.py")}
    frozen = {"schema_version": 1, "units": "mm", "training_dataset": args.training_dataset,
              "physical_unit": args.unit,
              "analysis_marker_maps": {ds: str(p) for ds, p in maps.items()},
              "recorded_marker_maps": {ds: str(p) for ds, p in recorded_maps.items()},
              "method": "Joint tail/base corner reprojection with linear tail travel",
              "inference_solver": solver, "implementation_sha256": implementation,
              "fixed_inputs_sha256": fixed, "source_sha256": {
                  p: h for p, h in source.items() if args.training_dataset + "_" in p},
              "models": [m.to_dict() for m in models], "candidates": candidates}
    model_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    frozen_hash = digest(model_path)
    # Reload the serialized frozen model before independent data/labels enter.
    models = [W.PairModel.from_dict(m) for m in read_json(model_path)["models"]]
    print(f"Frozen model: {model_path}", flush=True)
    training_values, training_counts, training_std = replay(train_times, train_frames, models, extras, K, dist)
    verification_holds = measurement_holds(read_json(ROOT / f"data/raw/{args.verification_dataset}_glove_log.json"))
    if {h[0] for h in verification_holds} & {h[0] for h in training_holds}:
        raise ValueError("Independent aperture must differ from training apertures")
    verify_times, verify_frames = load_frames(args.verification_dataset + "_" + args.camera,
                                            K, dist, body, obj, extras)
    verification_values, verification_counts, verification_std = replay(
        verify_times, verify_frames, models, extras, K, dist)
    training_summary = summarize_holds(train_times, training_values, training_holds)
    verification_summary = summarize_holds(verify_times, verification_values, verification_holds)
    if digest(model_path) != frozen_hash:
        raise RuntimeError("Frozen model changed during verification")
    for path, expected in (fixed | source).items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}")
    for ds, times, values, counts, std in (
            (args.training_dataset, train_times, training_values, training_counts, training_std),
            (args.verification_dataset, verify_times, verification_values, verification_counts, verification_std)):
        np.savez_compressed(output.with_name(output.stem + f".{ds}.npz"),
                            t_wall=times, gap_mm=values, pair_count=counts,
                            local_sensitivity_mm=std)
    report = {"reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "physical_unit": args.unit,
              "analysis_marker_maps": {ds: str(p) for ds, p in maps.items()},
              "recorded_marker_maps": {ds: str(p) for ds, p in recorded_maps.items()},
              "training_dataset": args.training_dataset,
              "verification_dataset": args.verification_dataset,
              "model": str(model_path), "frozen_model_sha256": frozen_hash,
              "fixed_inputs_sha256": fixed, "source_sha256": source,
              "fixed_inputs_unchanged": True, "candidates": candidates,
              "inference_solver": solver, "implementation_sha256": implementation,
              "training_holds": training_summary, "verification_holds": verification_summary,
              "passed": all(h["passed"] for h in verification_summary),
              "acceptance_gates": {"coverage_min": .70, "absolute_median_error_mm_max": 1,
                                   "p95_absolute_error_mm_max": 2},
              "candidate_gates": {"minimum_distinct_apertures": 3,
                                  "minimum_pair_frames_per_aperture": 40,
                                  "maximum_closed_base_distance_mm": 50,
                                  "blocked_holdout_coverage_min": .70,
                                  "blocked_holdout_absolute_median_error_mm_max": 1,
                                  "blocked_holdout_error_rms_mm_max": 2,
                                  "blocked_holdout_p95_absolute_error_mm_max": 4},
              "scope": "Static width holds; missing/rejected frames remain in coverage denominators. No prediction clipping, temporal smoothing, independent-label refit, or input/capture changes during review. Explicit measured-size revisions, if supplied, are recorded separately from immutable capture snapshots. Moving-width accuracy is untested. New models remain derived offline artifacts; legacy robot actuator path is unchanged."}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"verification_holds": verification_summary,
                      "passed": report["passed"], "report": str(output)}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
