"""Stateless quality selection layered over unchanged metric-width inference.

The residual-scaled local sensitivity is a numerical diagnostic, not a
certified error bound. Reject unreliable candidates before training-ranked
selection so another well-conditioned face can remain a fallback. No width
clipping, temporal averaging, interpolation or changed geometry is involved.
"""
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np

import forward_width as W


@dataclass(frozen=True)
class QualityPolicy:
    max_local_sensitivity_mm: float
    corner_noise_floor_px: float = 0.3

    def __post_init__(self):
        for value in (self.max_local_sensitivity_mm, self.corner_noise_floor_px):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError("Quality limits must be positive and finite")

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported width-quality policy schema")
        return cls(data["max_local_sensitivity_mm"], data["corner_noise_floor_px"])

    def sensitivity_mm(self, prediction):
        return max(self.corner_noise_floor_px, prediction.rms_px) * prediction.std_mm_per_pixel

    def rejection_reason(self, prediction):
        if not np.isfinite([prediction.gap_mm, prediction.rms_px,
                            prediction.std_mm_per_pixel]).all():
            return "nonfinite_prediction"
        if prediction.rms_px < 0 or prediction.std_mm_per_pixel <= 0:
            return "invalid_sensitivity"
        if self.sensitivity_mm(prediction) > self.max_local_sensitivity_mm:
            return "excess_local_sensitivity"
        return None


def derive_policy_limit(training_scores, *, percentile=95., multiplier=3.):
    """Derive a numerical envelope from training diagnostics, never labels."""
    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("Training percentile must lie strictly between 0 and 100")
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Training envelope multiplier must be finite and at least one")
    scores = np.asarray(training_scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if len(finite) < 40 or np.any(finite <= 0):
        raise ValueError("At least 40 positive finite training diagnostics required")
    return float(np.percentile(finite, percentile) * multiplier)


@dataclass(frozen=True)
class QualityDecision:
    gap_mm: float | None
    accepted: Mapping[str, W.PairPrediction]
    rejected: Mapping[str, str]
    selected_keys: tuple[str, ...]


def select_predictions(models: Sequence[W.PairModel], available, policy):
    """Select only usable faces; preserve one vote per fixed reference."""
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys) or set(available) - set(keys):
        raise ValueError("Prediction keys must match unique model identities")
    accepted, rejected = {}, {}
    for key, prediction in available.items():
        reason = policy.rejection_reason(prediction)
        if reason is None:
            accepted[key] = prediction
        else:
            rejected[key] = reason
    if not accepted:
        return QualityDecision(None, accepted, rejected, ())
    if any(m.holdout_p95_mm is None for m in models):
        # Existing unranked models fuse all usable candidates.
        selected = list(accepted)
    else:
        by_reference = {}
        for model in models:
            if model.key in accepted:
                by_reference.setdefault(model.base, []).append(model)
        selected = [min(group, key=lambda m: (m.holdout_p95_mm, m.tail)).key
                    for group in by_reference.values()]
    gap = float(np.median([accepted[key].gap_mm for key in selected]))
    return QualityDecision(gap, accepted, rejected, tuple(selected))


def predict_width(frame, models, base_transforms, K, dist, policy):
    """Expose the frozen result, every candidate, and the quality decision."""
    raw_width, available = W.predict_width(frame, models, base_transforms, K, dist)
    return raw_width, available, select_predictions(models, available, policy)
