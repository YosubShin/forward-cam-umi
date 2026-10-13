"""Local camera identity and image-coordinate contract for native capture.

Kept separate from the legacy Linux cameras.json because AVFoundation IDs
are platform-specific. Names are never used as identity fallback.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

ROLES = ("b0587", "b0587b", "fish")
ROLE_ALIASES = {"forward": "b0587", "head": "b0587b", "wrist": "fish"}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "calib/cameras.macos.local.json"


def supported_frame_rate(requested, minimum, maximum):
    """Accept UVC interval rounding, but not a different mode such as 29.97.

Some cameras advertise nominal 30 fps as 30.00003000003 (333333 x 100 ns).
Return the actual supported boundary so native configuration uses its exact
frame duration instead of requesting an interval just outside the range.
"""
    if minimum <= requested <= maximum:
        return requested
    for boundary in (minimum, maximum):
        if math.isclose(requested, boundary, rel_tol=1e-5):
            return boundary
    return None


@dataclass(frozen=True)
class CaptureSource:
    """Physical capture request, usable before any semantic role is assigned."""
    device_id: str
    name: str
    width: int
    height: int
    fps: float
    rotation_degrees: int = 0

    def __post_init__(self):
        if not isinstance(self.device_id, str) or not self.device_id.strip():
            raise ValueError("device_id must be a nonempty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name must be a nonempty string")
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("width and height must be positive integers")
        if type(self.fps) not in (int, float) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if type(self.rotation_degrees) is not int or self.rotation_degrees not in (0, 180):
            raise ValueError("rotation_degrees must be 0 or 180")


@dataclass(frozen=True)
class CameraSpec:
    """Complete capture configuration; preserves the version 1 on-disk schema."""
    role: str
    device_id: str
    name: str
    width: int
    height: int
    fps: float
    rotation_degrees: int = 0

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown camera role: {self.role}")
        self.source()

    def source(self):
        return CaptureSource(self.device_id, self.name, self.width, self.height,
                             self.fps, self.rotation_degrees)


def validate_specs(specs):
    if not specs:
        raise ValueError("camera configuration is empty")
    for field in ("role", "device_id"):
        values = [getattr(s, field) for s in specs]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field}: each physical camera needs one role")
    return specs


def load_config(path):
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or type(data.get("version")) is not int
            or data["version"] != 1 or data.get("backend") != "avfoundation"):
        raise ValueError("expected version 1 avfoundation camera configuration")
    if not isinstance(data.get("cameras"), list):
        raise ValueError("cameras must be a list")
    try:
        return validate_specs([CameraSpec(**row) for row in data["cameras"]])
    except TypeError as exc:
        raise ValueError(f"invalid camera entry: {exc}") from exc


def save_config(path, specs):
    validate_specs(specs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing binding silently. Rerunning setup writes a new
    # path or the operator explicitly removes/archives the previous config.
    with path.open("x") as f:
        json.dump({"version": 1, "backend": "avfoundation",
                   "cameras": [asdict(s) for s in specs]}, f, indent=2)
        f.write("\n")


def resolve_spec(selector, path=DEFAULT_CONFIG):
    """Resolve a saved role or exact ID; never guess from names/list positions."""
    role = ROLE_ALIASES.get(selector, selector)
    specs = load_config(path)
    matches = [s for s in specs if s.role == role or s.device_id == selector]
    if len(matches) != 1:
        raise ValueError(f"camera {selector!r} is not uniquely configured; run cameras_setup.py")
    return matches[0]


def normalize_image(image, rotation_degrees):
    """Rotate pixels once, before any image-coordinate consumer sees them.

Does not mirror or modify the source. 180 degrees preserves image dimensions
and needs no interpolation. Capture timestamps are independent of this step.
"""
    if rotation_degrees == 0:
        return image
    if rotation_degrees == 180:
        import cv2
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("rotation_degrees must be 0 or 180")
