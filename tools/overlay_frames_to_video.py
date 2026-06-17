"""Encode overlay val frames into one MP4 per variant (default 10 fps)."""

import argparse
import sys
from pathlib import Path

import cv2

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import MODEL_NAME, OVERLAYS_DIR
from overlay_val_models import VARIANTS

DEFAULT_FPS = 10
IMAGE_SUFFIXES = ('*.jpg', '*.jpeg', '*.png')


def _sorted_frames(frame_dir: Path):
    paths = []
    for pattern in IMAGE_SUFFIXES:
        paths.extend(frame_dir.glob(pattern))
    return sorted(paths, key=lambda p: p.name)


def write_video(frame_paths, output_path: Path, fps: float):
    if not frame_paths:
        raise FileNotFoundError('No overlay frames to encode')

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f'Failed to read first frame: {frame_paths[0]}')

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open video writer for {output_path}')

    written = 0
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    writer.release()
    if written == 0:
        raise RuntimeError(f'No frames written to {output_path}')
    return written


def main():
    parser = argparse.ArgumentParser(
        description='Build MP4 videos from overlay val frames (one video per variant).')
    parser.add_argument(
        '--variant',
        nargs='+',
        choices=list(VARIANTS) + ['all'],
        default=['all'],
        help='Which overlay folder(s) to encode.',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=DEFAULT_FPS,
        help=f'Output frame rate (default: {DEFAULT_FPS}).',
    )
    parser.add_argument(
        '--input-root',
        type=Path,
        default=OVERLAYS_DIR,
        help='Root directory containing per-variant overlay folders.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for MP4 files (default: <input-root>/videos/).',
    )
    args = parser.parse_args()

    variants = list(VARIANTS) if 'all' in args.variant else args.variant
    output_dir = args.output_dir or (args.input_root / 'videos')

    if not args.input_root.exists():
        print(f'ERROR: Overlay root not found: {args.input_root}', file=sys.stderr)
        sys.exit(1)

    for variant in variants:
        frame_dir = args.input_root / variant
        if not frame_dir.is_dir():
            print(f'ERROR: Missing overlay directory: {frame_dir}', file=sys.stderr)
            sys.exit(1)

        frame_paths = _sorted_frames(frame_dir)
        output_path = output_dir / f'{variant}.mp4'
        print(f'Encoding {variant}: {len(frame_paths)} frames @ {args.fps} fps -> {output_path}')
        written = write_video(frame_paths, output_path, args.fps)
        print(f'SUCCESS: Wrote {written} frames to {output_path}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
