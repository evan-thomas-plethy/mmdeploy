"""Render validation overlay MP4 from a predictions JSON.

Reads val images, draws precomputed predictions in memory, writes a 10 fps MP4.
Run tflite_export_val_predictions.py or coreml_export_val_predictions.py first.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import OVERLAYS_DIR
from overlay_draw import VARIANT_COLORS_BGR, draw_banner, draw_bbox_xywh, draw_pose
from pose_eval_common import prediction_label_from_path

# Kept for overlay_frames_to_video.py (legacy multi-variant layout).
VARIANTS = ('coreml_fp32', 'coreml_int8', 'tflite_fp32', 'tflite_int8')

PRECISION_COLORS_BGR = {
    'fp32': VARIANT_COLORS_BGR['coreml_fp32'],
    'fp16': (0, 200, 255),
    'int8': VARIANT_COLORS_BGR['coreml_int8'],
}

DEFAULT_OVERLAY_FPS = 10


def overlay_output_video_path(predictions_path, overlays_dir=None):
    """Map predictions/foo.keypoints.json -> overlays/foo.keypoints.mp4."""
    predictions_path = Path(predictions_path)
    overlays_dir = Path(overlays_dir or OVERLAYS_DIR)
    filename = predictions_path.name
    if filename.endswith('.json'):
        filename = f'{filename[:-5]}.mp4'
    elif not filename.endswith('.mp4'):
        filename = f'{filename}.mp4'
    return overlays_dir / filename


def _load_coco(ann_file):
    try:
        from xtcocotools.coco import COCO
    except ImportError:
        from pycocotools.coco import COCO
    return COCO(str(ann_file))


def _load_predictions_by_image(predictions_path):
    with open(predictions_path) as f:
        predictions = json.load(f)
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred['image_id']].append(pred)
    return by_image


def _parse_flat_keypoints(flat_keypoints):
    kpts = np.asarray(flat_keypoints, dtype=np.float32).reshape(-1, 3)
    return kpts[:, :2], kpts[:, 2]


def _default_label(predictions_path):
    return prediction_label_from_path(predictions_path)


def _color_for_path(path):
    # Color hint from filename tags when present; default green otherwise.
    from pose_eval_common import infer_precision_from_path
    return PRECISION_COLORS_BGR.get(infer_precision_from_path(path), (0, 255, 0))


class _VideoWriter:
    def __init__(self, output_path, fps):
        self._output_path = Path(output_path)
        self._fps = fps
        self._writer = None
        self._size = None  # (width, height) from first frame
        self._written = 0

    def write(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self._output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                self._fps,
                self._size,
            )
            if not self._writer.isOpened():
                raise RuntimeError(f'Failed to open video writer for {self._output_path}')
        elif (w, h) != self._size:
            frame_bgr = cv2.resize(frame_bgr, self._size, interpolation=cv2.INTER_LINEAR)
        self._writer.write(frame_bgr)
        self._written += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
        if self._written == 0:
            raise RuntimeError(f'No frames written to {self._output_path}')
        print(f'SUCCESS: Wrote {self._written} frames to {self._output_path}')
        return self._written


def render_overlay_video_from_predictions(
    predictions_path,
    coco,
    images_dir,
    output_video,
    label=None,
    color=None,
    max_images=None,
    score_thr=0.2,
    fps=DEFAULT_OVERLAY_FPS,
):
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f'Predictions not found: {predictions_path}. '
            'Run tflite_export_val_predictions.py or coreml_export_val_predictions.py first.')

    preds_by_image = _load_predictions_by_image(predictions_path)
    label = label or _default_label(predictions_path)
    color = color or _color_for_path(predictions_path)
    writer = _VideoWriter(output_video, fps)

    img_ids = sorted(preds_by_image)
    if max_images is not None:
        img_ids = img_ids[:max_images]

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(images_dir) / img_info['file_name']
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        canvas = image_bgr.copy()
        draw_banner(canvas, label, color)

        for pred in preds_by_image[img_id]:
            if 'bbox' in pred:
                draw_bbox_xywh(canvas, pred['bbox'], color=(128, 128, 128), thickness=1)
            keypoints_xy, keypoint_scores = _parse_flat_keypoints(pred['keypoints'])
            draw_pose(canvas, keypoints_xy, keypoint_scores, color, score_thr=score_thr)

        writer.write(canvas)

    return writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Overlay val images with predictions and write an MP4 (no frame JPGs).')
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        metavar='PATH',
        help='Precomputed keypoints JSON.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        metavar='PATH',
        help='COCO val annotations JSON (maps image_id to file_name).',
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        metavar='DIR',
        help='Val images directory.',
    )
    parser.add_argument(
        '--label',
        default=None,
        help='Banner text (default: derived from predictions filename).',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Limit number of val images (default: all).',
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.2,
        help='Minimum keypoint score to draw.',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=DEFAULT_OVERLAY_FPS,
        help=f'Output video frame rate (default: {DEFAULT_OVERLAY_FPS}).',
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f'ERROR: Predictions not found: {args.predictions}', file=sys.stderr)
        sys.exit(1)
    if not args.ann_file.exists():
        print(f'ERROR: Annotations not found: {args.ann_file}', file=sys.stderr)
        sys.exit(1)
    if not args.images_dir.is_dir():
        print(f'ERROR: Images directory not found: {args.images_dir}', file=sys.stderr)
        sys.exit(1)

    output_video = overlay_output_video_path(args.predictions)

    coco = _load_coco(args.ann_file)
    print(f'Annotation file: {args.ann_file}')
    print(f'Images dir: {args.images_dir}')
    print(f'Predictions: {args.predictions}')
    print(f'Output video: {output_video}')

    render_overlay_video_from_predictions(
        predictions_path=args.predictions,
        coco=coco,
        images_dir=args.images_dir,
        output_video=output_video,
        label=args.label,
        max_images=args.max_images,
        score_thr=args.score_thr,
        fps=args.fps,
    )


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f'ERROR: Overlay failed - {exc}', file=sys.stderr)
        sys.exit(1)
