"""Metric forward-camera width from learned tail/base image constraints.

Camera intrinsics and existing rigid-body geometry are read-only inputs. A new
width model learns each tail's straight travel path relative to a nearby fixed
base tag, in metres per normalized measured stroke. Inference jointly fits the
base camera pose and one width to all eight tag corners. Unlike independent
single-tag PnP subtraction, the moving tag's depth cannot wander freely.

This module is separate from labels_lib's normalized robot-actuator estimators:
its public outputs are physical millimetres, with no clipping or temporal hold
assumption. Independent verification labels must never enter fit_pair.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from labels_lib import marker_obj

SOLVER_VERSION = 3
HUBER_SCALE_PX = 1.0
COLD_START_STROKES = (0.0, 0.5, 1.0)
FUSION_POLICY = "Best training-blocked p95 candidate per fixed reference"


@dataclass
class Frame:
    index: int
    wall: float
    rvec: np.ndarray
    tvec: np.ndarray
    corners: dict


@dataclass
class Observation:
    rvec: np.ndarray
    tvec: np.ndarray
    base_corners: np.ndarray
    tail_corners: np.ndarray


@dataclass
class PairModel:
    tail: int
    base: int
    tail_edge_mm: float
    base_edge_mm: float
    minimum_gap_mm: float
    span_mm: float
    origin_m: np.ndarray
    travel_m: np.ndarray
    rotation: np.ndarray
    holdout_p95_mm: float | None = None

    @property
    def key(self):
        return f"pair:{self.tail}-{self.base}"

    @property
    def parameters(self):
        return np.r_[self.origin_m, self.travel_m, self.rotation]

    def to_dict(self):
        return {**vars(self), "origin_m": self.origin_m.tolist(),
                "travel_m": self.travel_m.tolist(),
                "rotation": self.rotation.tolist()}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("origin_m", "travel_m", "rotation"):
            data[key] = np.asarray(data[key], dtype=float)
        return cls(**data)


@dataclass
class PairPrediction:
    gap_mm: float
    rms_px: float
    # Local linearized sensitivity, not a certified accuracy bound.
    std_mm_per_pixel: float
    used_cold_start: bool = False
    initial_rms_px: float = 0.0


def unique_corners(ids, corners):
    """An ambiguous marker ID is unusable, including for the rigid pose solve."""
    ids = np.asarray(ids)
    return {int(i): np.asarray(c, dtype=float).reshape(4, 2)
            for i, c in zip(ids, corners) if np.count_nonzero(ids == i) == 1}


def local_observation(frame, tail, base, base_transform):
    if tail not in frame.corners or base not in frame.corners:
        return None
    R = cv2.Rodrigues(frame.rvec)[0]
    local_R = R @ base_transform[:3, :3]
    local_t = R @ base_transform[:3, 3] + frame.tvec.ravel()
    return Observation(cv2.Rodrigues(local_R)[0].ravel(), local_t,
                       frame.corners[base], frame.corners[tail])


def image_center(corners):
    """Projective square centre: intersect diagonals, not mean screen corners."""
    homogeneous = np.c_[corners, np.ones(4)]
    point = np.cross(np.cross(homogeneous[0], homogeneous[2]),
                     np.cross(homogeneous[1], homogeneous[3]))
    if abs(point[2]) < 1e-12:
        raise ValueError("Degenerate marker quadrilateral")
    return point[:2] / point[2]


def _initial(observations, labels, K, dist, edge, minimum, span):
    R = Rotation.from_rotvec(np.array([o.rvec for o in observations])).as_matrix()
    t = np.array([o.tvec for o in observations])
    stroke = (np.asarray(labels) - minimum) / span
    uv = np.array([image_center(o.tail_corners) for o in observations])
    xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    matrix = R[:, :2] - xy[:, :, None] * R[:, 2:3]
    rhs = xy * t[:, 2, None] - t[:, :2]
    design = np.concatenate([matrix, matrix * stroke[:, None, None]], axis=2)
    x = np.linalg.lstsq(design.reshape(-1, 6), rhs.ravel(), rcond=None)[0]

    def residual(value):
        points = value[:3] + stroke[:, None] * value[3:]
        pc = np.einsum("nij,nj->ni", R, points) + t
        projected = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, dist)[0]
        return (projected.reshape(-1, 2) - uv).ravel()

    initial = least_squares(residual, x, loss="soft_l1", f_scale=1,
                            max_nfev=100)
    rotations = []
    for o, base_R in zip(observations, R):
        ok, rv, _ = cv2.solvePnP(marker_obj(edge), o.tail_corners, K, dist,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            rotations.append(base_R.T @ cv2.Rodrigues(rv)[0])
    if not rotations:
        raise ValueError("No tail orientation initialization")
    rot = Rotation.from_matrix(rotations).mean().as_rotvec()
    return np.r_[initial.x, rot]


def fit_pair(observations, labels_mm, K, dist, *, tail, base, tail_edge_mm,
             base_edge_mm, minimum_gap_mm, span_mm, seed=None):
    """Train only on measured hold frames; jointly refine nuisance poses.

    The base tag's measured edge is fixed. Nuisance camera poses are analysis
    variables, never written back to any calibration or recorded timestamp.
    """
    labels = np.asarray(labels_mm, dtype=float)
    if len(observations) != len(labels) or len(labels) < 3:
        raise ValueError("Matched training observations and labels required")
    if not np.isfinite(labels).all() or span_mm <= 0:
        raise ValueError("Finite physical labels and positive span required")
    if len(np.unique(labels)) < 2:
        raise ValueError("One hold cannot identify a width travel path")
    parameters = (_initial(observations, labels, K, dist, tail_edge_mm,
                           minimum_gap_mm, span_mm)
                  if seed is None else seed.parameters)
    n = len(observations)
    x = np.r_[np.concatenate([np.r_[o.rvec, o.tvec] for o in observations]),
              parameters]
    images = np.array([np.r_[o.base_corners, o.tail_corners]
                       for o in observations])
    stroke = (labels - minimum_gap_mm) / span_mm
    base_obj, tail_obj = marker_obj(base_edge_mm), marker_obj(tail_edge_mm)
    sparsity = lil_matrix((16 * n, len(x)), dtype=int)
    for k in range(n):
        sparsity[16*k:16*k+16, 6*k:6*k+6] = 1
        sparsity[16*k+8:16*k+16, -9:] = 1

    def residual(value):
        pose = value[:-9].reshape(n, 6)
        R = Rotation.from_rotvec(pose[:, :3]).as_matrix()
        model = value[-9:]
        tail_points = (tail_obj @ Rotation.from_rotvec(model[6:]).as_matrix().T
                       + model[:3] + stroke[:, None, None] * model[3:6])
        points = np.concatenate([np.tile(base_obj, (n, 1, 1)), tail_points], axis=1)
        pc = np.einsum("nij,npj->npi", R, points) + pose[:, None, 3:]
        projected = cv2.projectPoints(pc.reshape(-1, 3), np.zeros(3),
                                      np.zeros(3), K, dist)[0]
        return (projected.reshape(n, 8, 2) - images).ravel()

    result = least_squares(
        residual, x, jac_sparsity=sparsity.tocsr(), loss="huber", f_scale=1,
        x_scale="jac", max_nfev=1200, ftol=1e-5,
        tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 300},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Width model optimization did not converge")
    model = result.x[-9:]
    if np.linalg.norm(model[3:6]) * 1000 < 20:
        raise ValueError("Tail motion does not span 20 mm over the measured stroke")
    return PairModel(tail, base, tail_edge_mm, base_edge_mm, minimum_gap_mm,
                     span_mm, model[:3], model[3:6], model[6:])


def predict_pair(observation, model, K, dist, max_rms_px=5.0):
    """Fit width without a known hold value, clipping, or temporal averaging."""
    base_obj = marker_obj(model.base_edge_mm)
    tail_obj = (marker_obj(model.tail_edge_mm)
                @ Rotation.from_rotvec(model.rotation).as_matrix().T
                + model.origin_m)
    objects = np.r_[base_obj, tail_obj]
    move = np.r_[np.zeros((4, 3)), np.tile(model.travel_m, (4, 1))]
    images = np.r_[observation.base_corners, observation.tail_corners]
    R = cv2.Rodrigues(observation.rvec)[0]
    xy = cv2.undistortPoints(image_center(observation.tail_corners)
                            .reshape(1, 1, 2), K, dist).ravel()
    a, b = R @ model.origin_m + observation.tvec, R @ model.travel_m
    v = b[:2] - xy * b[2]
    if v @ v < 1e-16:
        return None
    fraction = float(v @ (xy * a[2] - a[:2]) / (v @ v))
    x = np.r_[observation.rvec, observation.tvec, fraction]

    def residual(value):
        projected = cv2.projectPoints(objects + move * value[6], value[:3],
                                      value[3:6], K, dist)[0].reshape(-1, 2)
        return (projected - images).ravel()

    def solve(seed):
        candidate = least_squares(residual, seed, loss="huber", f_scale=HUBER_SCALE_PX,
                                  x_scale="jac", max_nfev=80)
        if not candidate.success or not np.isfinite(candidate.x).all():
            return None
        camera_points = ((objects + move * candidate.x[6])
                         @ cv2.Rodrigues(candidate.x[:3])[0].T + candidate.x[3:6])
        return candidate if np.all(camera_points[:, 2] > 0) else None

    warm = solve(x)
    initial_rms = (float(np.sqrt(np.mean(residual(warm.x) ** 2)))
                   if warm is not None else float("inf"))
    result = warm
    # A body-pose warm start is only an initialization. Planar pose ambiguity
    # can strand a converged joint solve in a poor local minimum. At residuals
    # above the 1 px Huber scale, also initialize globally at three positions
    # of the TRAINED stroke. These are seeds, not bounds or known hold values;
    # the final width remains free to leave the measured range.
    if initial_rms > HUBER_SCALE_PX:
        for stroke in COLD_START_STROKES:
            try:
                ok, rv, tv = cv2.solvePnP(objects + move * stroke, images, K, dist,
                                         flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                continue  # a degenerate seed must not kill another valid solve
            if not ok:
                continue
            candidate = solve(np.r_[rv.ravel(), tv.ravel(), stroke])
            if candidate is not None and (result is None or candidate.cost < result.cost):
                result = candidate
    if result is None:
        return None
    rms = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    if rms > max_rms_px:
        return None
    _, singular, vt = np.linalg.svd(result.jac, full_matrices=False)
    if singular[-1] <= singular[0] * 1e-10:
        return None  # an unobservable width must not get zero uncertainty
    covariance = (vt.T / singular**2) @ vt
    return PairPrediction(
        model.minimum_gap_mm + result.x[6] * model.span_mm, rms,
        float(np.sqrt(max(covariance[6, 6], 0)) * model.span_mm),
        used_cold_start=result is not warm,
        initial_rms_px=initial_rms,
    )


def predict_width(frame, models, base_transforms, K, dist):
    available = {}
    for model in models:
        observation = local_observation(frame, model.tail, model.base,
                                        base_transforms[model.base])
        if observation is not None:
            result = predict_pair(observation, model, K, dist)
            if result is not None:
                available[model.key] = result
    if not available:
        return None, available
    if any(m.holdout_p95_mm is None for m in models):
        # Preserve old frozen models' fusion until a complete training-only
        # ranking exists; do not invent scores from verification outcomes.
        return float(np.median([p.gap_mm for p in available.values()])), available
    by_reference = {}
    for model in models:
        if model.key in available:
            by_reference.setdefault(model.base, []).append(model)
    selected = [min(group, key=lambda m: (m.holdout_p95_mm, m.tail))
                for group in by_reference.values()]
    # Faces sharing one fixed tag have correlated pose errors. Select their
    # best TRAINING-ranked available face before fusing distinct references;
    # a second face remains a fallback instead of getting a duplicate vote.
    return float(np.median([available[m.key].gap_mm for m in selected])), available


def error_summary(predictions, actual_mm):
    values = np.asarray(predictions, dtype=float)
    mask = np.isfinite(values)
    actual = np.broadcast_to(np.asarray(actual_mm), values.shape)[mask]
    finite = values[mask]
    if not len(finite):
        return {"valid_frames": 0}
    errors = finite - actual
    return {"valid_frames": len(finite),
            "median_prediction_mm": float(np.median(finite)),
            "signed_median_error_mm": float(np.median(errors)),
            "absolute_error_p95_mm": float(np.percentile(abs(errors), 95)),
            "error_rms_mm": float(np.sqrt(np.mean(errors ** 2)))}
