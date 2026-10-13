"""Linux-style terminal setup, with AVFoundation IDs instead of V4L2 paths.

Preview each device in mpv/ffplay, assign its role, then write local bindings.
The Linux setup workflow and role descriptions live in cameras_setup.py.
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

import cv2

from camera_config import (CameraSpec, CaptureSource, DEFAULT_CONFIG, load_config,
                           resolve_spec, save_config, supported_frame_rate)
from cameras_setup import ROLES
from preview_sink import open_sink, close_sink


def identity_source(info):
    """Use a supported small preview mode before the operator assigns a role."""
    modes = info['modes']
    if not modes:
        raise ValueError('device advertises no video modes')
    small = [m for m in modes if m[0] <= 1280 and m[1] <= 720]
    width, height, low, high = sorted(small or modes,
                                    key=lambda m: m[0] * m[1], reverse=bool(small))[0]
    fps = supported_frame_rate(30, low, high)
    if fps is None:
        fps = min(max(30, low), high)
    return CaptureSource(info['device_id'], info['name'], width, height, fps)


def preview(source, seconds):
    from camera_macos import PreviewCapture
    width = min(1280, source.width)
    height = round(source.height * width / source.width)
    sink = open_sink(f'{source.name} | {source.device_id} | rotation {source.rotation_degrees}', width, height)
    if sink is None:
        raise RuntimeError('install mpv or ffplay to verify the physical camera')
    count = 0
    try:
        with PreviewCapture(source) as capture:
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                if sink.poll() is not None:
                    break
                image = capture.read().image
                frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                sink.stdin.write(frame.tobytes())
                sink.stdin.flush()
                count += 1
    finally:
        close_sink(sink)
    if count == 0:
        raise RuntimeError('preview received no frames')
    print(f'  {count} frames sent to preview; recording FPS and synchronization not validated')


def validate_device(spec, inventory):
    matches = [d for d in inventory if d['device_id'] == spec.device_id]
    if len(matches) != 1 or matches[0]['name'] != spec.name:
        raise ValueError(f'{spec.role}: saved device is missing or ambiguous; rerun setup')
    if not any(w == spec.width and h == spec.height
               and supported_frame_rate(spec.fps, low, high) is not None
               for w, h, low, high in matches[0]['modes']):
        raise ValueError(f'{spec.role}: unsupported mode {spec.width}x{spec.height} @ {spec.fps:g}')


def capture_spec(role, info, previous):
    """Keep capture settings explicit, independent of the identification preview."""
    _, resolution, fps, _ = ROLES[role]
    width, height = map(int, resolution.split('x'))
    old = previous.get(role)
    if old:
        width, height, fps = old.width, old.height, old.fps
    rotation = old.rotation_degrees if old else 0
    while True:
        try:
            value = input(f'   image rotation [0 / 180, Enter = {rotation}]: ').strip()
            rotation = int(value) if value else rotation
            if rotation not in (0, 180):
                raise ValueError('rotation must be 0 or 180')
            break
        except ValueError as exc:
            print(f'   {exc}')
    default = f'{width}x{height}@{float(fps):g}'
    while True:
        try:
            value = input(f'   recording mode WxH@fps [Enter = {default}]: ').strip() or default
            size, rate = value.split('@')
            width, height = map(int, size.lower().split('x'))
            spec = CameraSpec(role, info['device_id'], info['name'], width, height, float(rate), rotation)
            validate_device(spec, [info])
            return spec
        except ValueError as exc:
            print(f'   {exc}; advertised modes:')
            for w, h, low, high in info['modes']:
                print(f'     {w}x{h} @ {low:g}–{high:g} fps')


def write_bindings(path, specs):
    """Like Linux setup, archive the existing file only after setup succeeds."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    pending = path.with_name(f'{path.stem}.pending-{stamp}.local.json')
    try:
        save_config(pending, specs)
        if path.exists():
            backup = path.with_name(f'{path.stem}.backup-{stamp}.local.json')
            shutil.copy2(path, backup)
            print(f'existing bindings backed up to {backup.name}')
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def wizard(inventory, args):
    if not sys.stdin.isatty():
        raise ValueError('interactive setup requires a terminal; use --list to inspect devices')
    previous = {s.role: s for s in load_config(args.config)} if args.config.exists() else {}
    assigned = {}
    print('\nFor each device a preview opens. Cover its lens to identify it, then answer.\n'
          'roles: ' + '  '.join(f'{r} = {v[3]}' for r, v in ROLES.items()))
    for info in inventory:
        print(f"\n-- {info['device_id']} ({info['name']})")
        source = identity_source(info)
        verified = False
        while True:
            try:
                preview(source, args.seconds)
                verified = True
            except (RuntimeError, OSError) as exc:
                print(f'   preview failed: {exc}')
            answer = input('   role [b0587 / b0587b / fish / - to skip / r to replay]: ').strip()
            if answer == 'r':
                continue
            if answer in ('', '-'):
                break
            if answer not in ROLES or answer in assigned:
                print('   unknown or already assigned role')
                continue
            if not verified:
                print('   verify a working preview before assigning this camera')
                continue
            spec = capture_spec(answer, info, previous)
            # Validate the selected native mode and normalized direction once.
            try:
                preview(spec, args.seconds)
            except (RuntimeError, OSError) as exc:
                print(f'   capture mode failed: {exc}; device remains unassigned')
                continue
            if input(f'   Correct {ROLES[answer][3]} and upright view? [y/N]: ').strip().lower() != 'y':
                continue
            assigned[answer] = spec
            break
    required = ('b0587', 'b0587b', 'fish') if args.require_head else ('b0587', 'fish')
    missing = [role for role in required if role not in assigned]
    if missing:
        raise ValueError(f'required roles not assigned: {missing}; existing bindings kept')
    write_bindings(args.config, [assigned[r] for r in ROLES if r in assigned])
    print(f'\nwrote {args.config}')
    print('next: python scripts/focus_tune.py forward')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--list', action='store_true', help='list devices without opening cameras')
    action.add_argument('--preview', metavar='ROLE', help='preview a saved role or exact ID')
    action.add_argument('--check-config', action='store_true', help='check saved IDs and modes without capture')
    parser.add_argument('--formats', action='store_true', help='list supported resolutions and frame rates')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--require-head', action='store_true', help='require all three camera roles')
    parser.add_argument('--seconds', type=float, default=8, help='duration of each preview')
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 300:
        parser.error('--seconds must be between 0 and 300')
    try:
        from camera_macos import devices, device_info
        inventory = [device_info(d) for d in devices()]
        if args.preview:
            spec = resolve_spec(args.preview, args.config)
            validate_device(spec, inventory)
            preview(spec, args.seconds)
        elif args.check_config:
            specs = load_config(args.config)
            if args.require_head and 'b0587b' not in {s.role for s in specs}:
                raise ValueError('head camera is required but not configured')
            for spec in specs:
                validate_device(spec, inventory)
            print('Device IDs, names and requested modes match (no camera opened).')
        else:
            if not inventory:
                raise RuntimeError('no video devices found')
            for info in inventory:
                print(f"{info['device_id']}  {info['name']}")
                if args.formats:
                    for w, h, low, high in info['modes']:
                        print(f'  {w}x{h}: {low:g}–{high:g} fps')
            print('Recheck the physical camera after changing USB ports; names are not identities.')
            if not args.list:
                wizard(inventory, args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Camera setup: {exc}\n')
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, '\nCamera setup cancelled; existing bindings kept.\n')


if __name__ == '__main__':
    main()
