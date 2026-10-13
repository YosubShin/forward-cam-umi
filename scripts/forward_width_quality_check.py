#!/usr/bin/env python3
"""Replay a frozen forward-width model with an explicit quality policy.

Writes separate raw/qualified arrays and rejection diagnostics. Calibration,
model files, prior reviews and recorded frame times are read-only. Unusable
frames remain NaN; no interpolation, smoothing or width clipping is performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import forward_width as W
import forward_width_check as C
import forward_width_quality as Q
from forward_width_accept import evaluate_holds
import labels_lib as L

ROOT = Path(__file__).resolve().parents[1]
REASONS = {None: 0, "nonfinite_prediction": 1, "invalid_sensitivity": 2,
           "excess_local_sensitivity": 3}


def frozen_calibration_input(root, frozen, relative_path):
    """Rebase provenance paths without weakening the frozen content check.

    Historical models retain their original absolute input names. A receiver
    uses the same relative calib file in its own checkout, verified against
    the model's recorded hash; never fall back to a sender's filesystem.
    """
    relative = Path(relative_path)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != 'calib':
        raise ValueError('Frozen calibration input must be one file under calib')
    expected = {sha for name,sha in frozen['fixed_inputs_sha256'].items()
                if tuple(Path(name).parts[-2:]) == relative.parts}
    path = (Path(root)/relative).resolve()
    if (len(expected) != 1 or not path.is_relative_to(Path(root).resolve())
            or not path.is_file() or C.digest(path) not in expected):
        raise ValueError(f'Frozen geometry changed or ambiguous: {relative}')
    return path


def continuity(times, values, indices):
    """Never bridge rejected or missing frames when measuring raw steps."""
    left, right = indices[:-1], indices[1:]
    valid = (right == left + 1) & np.isfinite(values[left]) & np.isfinite(values[right])
    left, right = left[valid], right[valid]
    changes = np.abs(values[right]-values[left])
    worst = np.argsort(changes)[::-1][:5]
    return {"adjacent_valid_pairs": len(changes),
            "absolute_step_p95_mm": float(np.percentile(changes, 95)) if len(changes) else None,
            "maximum_absolute_step_mm": float(changes.max()) if len(changes) else None,
            "largest_steps": [{"left_frame": int(left[j]), "right_frame": int(right[j]),
                               "signed_step_mm": float(values[right[j]]-values[left[j]]),
                               "dt_ms": float((times[right[j]]-times[left[j]])*1000)} for j in worst]}


def replay(times, frames, models, extras, K, dist, policy):
    n, p = len(times), len(models)
    arrays = {"raw_gap_mm": np.full(n, np.nan), "gap_mm": np.full(n, np.nan),
              "body_valid": np.zeros(n, bool), "raw_pair_count": np.zeros(n, int),
              "accepted_pair_count": np.zeros(n, int), "raw_pair_gap_mm": np.full((n,p), np.nan),
              "pair_rms_px": np.full((n,p), np.nan), "pair_std_mm_per_pixel": np.full((n,p), np.nan),
              "pair_local_sensitivity_mm": np.full((n,p), np.nan),
              "pair_rejection_reason": np.zeros((n,p), np.uint8),
              "pair_selected": np.zeros((n,p), bool)}
    transforms = {i: T for i,(T,_) in extras.items()}
    for frame in frames:
        k = frame.index
        raw, available, decision = Q.predict_width(frame, models, transforms, K, dist, policy)
        arrays["body_valid"][k] = True
        if raw is not None:
            arrays["raw_gap_mm"][k] = raw
        if decision.gap_mm is not None:
            arrays["gap_mm"][k] = decision.gap_mm
        arrays["raw_pair_count"][k] = len(available)
        arrays["accepted_pair_count"][k] = len(decision.accepted)
        for j, model in enumerate(models):
            prediction = available.get(model.key)
            if prediction is None:
                continue
            arrays["raw_pair_gap_mm"][k,j] = prediction.gap_mm
            arrays["pair_rms_px"][k,j] = prediction.rms_px
            arrays["pair_std_mm_per_pixel"][k,j] = prediction.std_mm_per_pixel
            arrays["pair_local_sensitivity_mm"][k,j] = policy.sensitivity_mm(prediction)
            arrays["pair_rejection_reason"][k,j] = REASONS[decision.rejected.get(model.key)]
            arrays["pair_selected"][k,j] = model.key in decision.selected_keys
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality-policy", type=Path, required=True)
    parser.add_argument("--p95-max-mm", type=float,
                        help="explicit static acceptance limit; required for a width-hold recording")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/"tmp") or out.exists() or out.with_suffix('.npz').exists():
        parser.error("Choose a fresh --out inside this checkout's tmp directory")
    frozen = C.read_json(args.model)
    declaration = C.read_json(args.quality_policy)
    policy = Q.QualityPolicy.from_dict(declaration)
    if frozen.get("schema_version") != 1 or frozen.get("units") != "mm":
        raise ValueError("Unsupported frozen metric model")
    if declaration.get("model_sha256") != C.digest(args.model):
        raise ValueError("Quality policy belongs to another frozen model")
    for name, expected in frozen['implementation_sha256'].items():
        if C.digest(ROOT/'scripts'/name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    unit = frozen['physical_unit']
    intr = frozen_calibration_input(ROOT,frozen,'calib/B0587_intrinsics.json')
    ball = frozen_calibration_input(ROOT,frozen,f'calib/glove_{unit}_dodeca_bundle.json')
    bases = frozen_calibration_input(ROOT,frozen,f'calib/glove_{unit}_gripper_bundle.json')
    declared_map = Path(frozen['analysis_marker_maps'][frozen['training_dataset']])
    if declared_map.parent.name != 'calib':
        raise ValueError('Unsupported frozen marker-map location')
    marker_map = frozen_calibration_input(ROOT,frozen,Path('calib')/declared_map.name)
    _, recorded = C.review_marker_maps(ROOT, (frozen['training_dataset'],args.dataset),unit,marker_map)
    stem = args.dataset+'_b0587'
    log = ROOT/f'data/raw/{args.dataset}_glove_log.json'
    holds = C.measurement_holds(C.read_json(log)) if log.exists() else []
    if holds and args.p95_max_mm is None:
        parser.error("Width-hold review requires an explicit --p95-max-mm")
    if args.p95_max_mm is not None and (not np.isfinite(args.p95_max_mm) or args.p95_max_mm <= 0):
        parser.error("--p95-max-mm must be positive and finite")
    episodes_path = ROOT/f'data/raw/{args.dataset}_glove_episodes.json'
    # Older hold-only recordings retain their measurement log without an
    # episode sidecar. Normal task recordings still require episode marks.
    if episodes_path.exists():
        episodes = C.read_json(episodes_path)['episodes']
    elif holds:
        episodes = []
    else:
        raise ValueError("Task review requires recorded episode boundaries")
    source = [args.model.resolve(),args.quality_policy.resolve(),intr,ball,bases,marker_map,
              ROOT/f'calib/{stem}_dodeca_detections.npz',ROOT/f'data/raw/{stem}_ts.json',
              ROOT/f'data/raw/{stem}_meta.json',*recorded.values()]
    if episodes_path.exists():
        source.append(episodes_path)
    if log.exists():
        source.append(log)
    hashes = {str(path):C.digest(path) for path in source}
    protected = {str(p):C.digest(p) for p in (ROOT/'calib').glob('*.json')}
    K,dist = L.load_intrinsics(intr)
    body,obj = L.load_bundle(ball)
    extras = L.load_gripper_bundle(bases)
    models = [W.PairModel.from_dict(m) for m in frozen['models']]
    times,frames = C.load_frames(stem,K,dist,body,obj,extras)
    arrays = replay(times,frames,models,extras,K,dist,policy)
    results = []
    for e in episodes:
        ids = np.flatnonzero((times>=e['t_start']) & (times<=e['t_end']))
        raw,qualified = arrays['raw_gap_mm'][ids],arrays['gap_mm'][ids]
        both = np.isfinite(raw) & np.isfinite(qualified)
        results.append({'episode':e,'recorded_frames':len(ids),
                        'raw_width_coverage':float(np.isfinite(raw).mean()) if len(ids) else 0.,
                        'qualified_width_coverage':float(np.isfinite(qualified).mean()) if len(ids) else 0.,
                        'newly_rejected_frames':int((np.isfinite(raw) & ~np.isfinite(qualified)).sum()),
                        'changed_fused_frames':int((abs(raw[both]-qualified[both])>1e-8).sum()),
                        'raw_continuity':continuity(times,arrays['raw_gap_mm'],ids),
                        'qualified_continuity':continuity(times,arrays['gap_mm'],ids)})
    raw_holds = evaluate_holds(times,arrays['raw_gap_mm'],holds,args.p95_max_mm) if holds else []
    qualified_holds = evaluate_holds(times,arrays['gap_mm'],holds,args.p95_max_mm) if holds else []
    rejections = {}
    for j,m in enumerate(models):
        count = Counter(int(code) for code in arrays['pair_rejection_reason'][:,j] if code)
        rejections[m.key] = {reason:count[code] for reason,code in REASONS.items() if reason and count[code]}
    for path,expected in hashes.items() | protected.items():
        if C.digest(path) != expected:
            raise RuntimeError(f'Input changed during quality review: {path}')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),t_wall=times,
                        pair_keys=np.array([m.key for m in models]),**arrays)
    report = {'dataset':args.dataset,'model':str(args.model.resolve()),'model_sha256':C.digest(args.model),
              'policy':declaration,'policy_sha256':C.digest(args.quality_policy),'training_run':False,
              'inputs_unchanged':True,'source_sha256':hashes,'episodes':results,'pair_rejections':rejections,
              'pair_rejection_codes':{str(code):reason for reason,code in REASONS.items() if reason},
              'raw_holds':raw_holds,'qualified_holds':qualified_holds,
              'static_acceptance_passed':all(x['passed'] for x in qualified_holds) if qualified_holds else None,
              'static_acceptance_is_independent':args.dataset!=frozen['training_dataset'] if holds else None,
              'implementation_sha256':{name:C.digest(ROOT/'scripts'/name) for name in ('forward_width_quality.py','forward_width_quality_check.py')},
              'scope':'Stateless candidate quality rejection and training-ranked fallback. Rejected frames remain NaN. No smoothing, interpolation, clipping, refit or new dynamic accuracy certification. A continuity statistic never bridges invalid frames.'}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'episodes':results,'qualified_holds':qualified_holds,'pair_rejections':rejections,'report':str(out)},ensure_ascii=False,indent=2),flush=True)
    if report['static_acceptance_is_independent'] and not report['static_acceptance_passed']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
