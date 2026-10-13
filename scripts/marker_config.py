"""Physical wrist-marker configuration shared by capture and session checks.

Keep the existing marker-map schema and detection cache names. A session
snapshot freezes the physical print used for that recording, independently
of later edits to a unit's local configuration.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class WristMarkers:
    family: str
    markers: tuple[tuple[int, float], ...]

    @classmethod
    def from_map(cls, data):
        family = data.get("tips_dict")
        limits = {"4x4": 50, "16h5": 30}
        if family not in limits:
            raise ValueError("tips_dict must be '4x4' or '16h5'")
        tips = data.get("tips")
        if not isinstance(tips, dict) or len(tips) != 2:
            raise ValueError("tips must contain exactly two distinct marker IDs and sizes in mm")
        markers = []
        for key, size in tips.items():
            try:
                marker_id = int(key)
                edge = float(size)
            except (TypeError, ValueError) as exc:
                raise ValueError("tip IDs must be integers and sizes must be positive finite mm") from exc
            if str(marker_id) != str(key) or not 0 <= marker_id < limits[family]:
                raise ValueError(f"invalid {family} tip ID: {key}")
            if isinstance(size, bool) or not math.isfinite(edge) or edge <= 0:
                raise ValueError(f"invalid black-edge size for tip {key}: {size}")
            # A physical 4x4 marker can serve both forward and wrist views,
            # but its metric size cannot depend on which view uses it.
            if family == "4x4":
                for group in ("tails", "bases"):
                    other = data.get(group, {}).get(key)
                    if other is not None and not math.isclose(edge, float(other), abs_tol=1e-9):
                        raise ValueError(f"tip {key} size disagrees with {group}")
                if key in data.get("ball_faces", {}):
                    raise ValueError(f"tip {key} is a rigid ball face, not a jaw marker")
            markers.append((marker_id, edge))
        return cls(family, tuple(sorted(markers)))

    @property
    def sizes_mm(self):
        return dict(self.markers)

    @property
    def cache_tag(self):
        return "dodeca" if self.family == "4x4" else "tips16h5"

    @property
    def description(self):
        return f"{self.family} " + ", ".join(f"ID {i} ({size:g} mm)" for i, size in self.markers)


def load_marker_map(path):
    with Path(path).open() as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("marker map must be a JSON object")
    WristMarkers.from_map(data)
    return data


def snapshot_path(root, dataset):
    return Path(root) / "data/raw" / f"{dataset}_marker_map.json"


def resolve_marker_map(root, dataset, *, unit=None, path=None):
    """Explicit map, otherwise recorded snapshot, otherwise legacy unit map.

    An explicit map must agree with an existing snapshot. Never reinterpret
    an older recording with silently changed marker identities or sizes.
    Without a unit or snapshot, None retains the legacy recording behavior.
    """
    root = Path(root)
    recorded = snapshot_path(root, dataset)
    if path is not None:
        chosen = root / path
        data = load_marker_map(chosen)
        if recorded.exists() and data != load_marker_map(recorded):
            raise ValueError(f"marker map differs from {recorded}; use a new dataset name")
    elif recorded.exists():
        chosen = recorded
    elif unit is not None:
        chosen = root / "calib" / f"marker_map_glove_{unit}.json"
    else:
        return None
    data = load_marker_map(chosen)
    if unit is not None and data.get("unit", unit) != unit:
        raise ValueError(f"marker map unit {data['unit']!r} does not match {unit!r}")
    return chosen


def save_snapshot(root, dataset, data):
    """Create once; retain the same map across retries and archived takes."""
    WristMarkers.from_map(data)
    path = snapshot_path(root, dataset)
    if path.exists():
        if load_marker_map(path) != data:
            raise ValueError(f"marker map differs from {path}; use a new dataset name")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return path
