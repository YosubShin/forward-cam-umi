#!/usr/bin/env python3
"""Replay a frozen metric model under an explicit acceptance policy, without fitting."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import forward_width as W
import forward_width_check as C
import labels_lib as L
from marker_config import load_marker_map

ROOT = Path(__file__).resolve().parents[1]


def evaluate_holds(times, values, holds, p95_max_mm):
    if not np.isfinite(p95_max_mm) or p95_max_mm <= 0:
        raise ValueError("P95 acceptance limit must be positive and finite")
    summaries = C.summarize_holds(times, values, holds)
    for summary in summaries:
        summary["original_2mm_diagnostic_passed"] = summary["passed"]
        checks = summary["checks"]
        checks.pop("p95_absolute_error_at_most_2mm")
        checks["p95_absolute_error_within_acceptance_limit"] = (
            summary.get("absolute_error_p95_mm", np.inf) <= p95_max_mm)
        summary["passed"] = all(checks.values())
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification_dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--p95-max-mm", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.p95_max_mm) or args.p95_max_mm <= 0:
        parser.error("--p95-max-mm must be positive and finite")
    output = args.out.resolve()
    if not output.is_relative_to(ROOT / "tmp"):
        parser.error("--out must be inside this checkout's tmp directory")
    if output.exists() or output.with_suffix(".npz").exists():
        parser.error("Choose a fresh output name; preceding reviews must be preserved")
    frozen = C.read_json(args.model)
    if frozen.get("schema_version") != 1 or frozen.get("units") != "mm":
        raise ValueError("Unsupported frozen metric model")
    if args.verification_dataset == frozen["training_dataset"]:
        raise ValueError("Acceptance requires a separate recording")
    for name, expected in frozen["implementation_sha256"].items():
        if C.digest(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    unit = frozen["physical_unit"]
    intr = ROOT / "calib/B0587_intrinsics.json"
    ball = ROOT / f"calib/glove_{unit}_dodeca_bundle.json"
    bases = ROOT / f"calib/glove_{unit}_gripper_bundle.json"
    marker_map = Path(frozen["analysis_marker_maps"][frozen["training_dataset"]])
    for path in (intr, ball, bases, marker_map):
        expected = frozen["fixed_inputs_sha256"].get(str(path))
        expected = expected or frozen["fixed_inputs_sha256"].get(str(path.relative_to(ROOT)))
        if expected is None or C.digest(path) != expected:
            raise ValueError(f"Frozen geometry changed: {path}")
    # Use exactly the original declared offline measured-size revision.
    maps, recorded = C.review_marker_maps(
        ROOT, (frozen["training_dataset"], args.verification_dataset), unit, marker_map)
    for path in maps.values():
        if load_marker_map(path) != load_marker_map(marker_map):
            raise ValueError("Analysis marker geometry differs")
    stem = args.verification_dataset + "_b0587"
    log = ROOT / f"data/raw/{args.verification_dataset}_glove_log.json"
    source_paths = [args.model.resolve(), intr, ball, bases, marker_map, log,
                    ROOT / f"data/raw/{stem}_ts.json",
                    ROOT / f"data/raw/{stem}_meta.json",
                    ROOT / f"calib/{stem}_dodeca_detections.npz",
                    *recorded.values()]
    hashes = {str(path): C.digest(path) for path in source_paths}
    calibration_hashes = {str(path): C.digest(path) for path in (ROOT / "calib").glob("*.json")}
    K, dist = L.load_intrinsics(intr)
    body, obj = L.load_bundle(ball)
    extras = L.load_gripper_bundle(bases)
    models = [W.PairModel.from_dict(m) for m in frozen["models"]]
    times, frames = C.load_frames(stem, K, dist, body, obj, extras)
    values, counts, sensitivity = C.replay(times, frames, models, extras, K, dist)
    # Measurements enter scoring only after inference is complete.
    holds = C.measurement_holds(C.read_json(log))
    summary = evaluate_holds(times, values, holds, args.p95_max_mm)
    for path, expected in hashes.items() | calibration_hashes.items():
        if C.digest(path) != expected:
            raise RuntimeError(f"Input changed during acceptance: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), t_wall=times, gap_mm=values,
                        pair_count=counts, local_sensitivity_mm=sensitivity)
    report = {"reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "physical_unit": unit, "verification_dataset": args.verification_dataset,
              "model": str(args.model.resolve()), "frozen_model_sha256": hashes[str(args.model.resolve())],
              "training_run": False, "inputs_unchanged": True,
              "source_sha256": hashes, "calibration_sha256": calibration_hashes,
              "acceptance_gates": {"coverage_min": .70, "absolute_median_error_mm_max": 1,
                                   "p95_absolute_error_mm_max": args.p95_max_mm},
              "verification_holds": summary, "passed": all(h["passed"] for h in summary),
              "acceptance_script_sha256": C.digest(__file__),
              "scope": "Explicit static independent-width acceptance. Original model, predictions and 2mm diagnostic reports are preserved. No fitting, offsets, clipping, smoothing or capture/calibration edits. This acceptance does not qualify other viewing angles, motion, or robot actuator units."}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"verification_holds": summary, "passed": report["passed"],
                      "report": str(output)}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
