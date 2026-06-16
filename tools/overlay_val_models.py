"""Render validation overlays for converted RTMPose models (instance-only).

CoreML variants use prediction JSONs rsync'd from Mac (step 4a).
TFLite variants run live inference on the instance.
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

from model_paths import (
    ANN_FILE,
    FP32_PREDICTIONS,
    FP32_TFLITE,
    IMG_PREFIX,
    INT8_PREDICTIONS,
    INT8_TFLITE,
    MODEL_NAME,
    OVERLAYS_DIR,
)
from overlay_draw import VARIANT_COLORS_BGR, draw_banner, draw_bbox_xywh, draw_pose

VARIANTS = ('coreml_fp32', 'coreml_int8', 'tflite_fp32', 'tflite_int8')

COREML_PREDICTION_PATHS = {
    'coreml_fp32': FP32_PREDICTIONS,
    'coreml_int8': INT8_PREDICTIONS,
}

TFLITE_MODEL_PATHS = {
    'tflite_fp32': FP32_TFLITE,
    'tflite_int8': INT8_TFLITE,
}


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


class TFLiteRunner:
    def __init__(self, model_path):
        from tflite_compare_ap import (
            _format_input_tensor,
            _load_interpreter,
            _pick_simcc_outputs,
            _run_inference,
            _setup_mmpose_imports,
            decode_keypoints,
            preprocess_instance,
        )

        self._get_simcc_maximum, _, self._bbox_xyxy2cs, self._get_warp_matrix = (
            _setup_mmpose_imports())
        self._preprocess_instance = preprocess_instance
        self._decode_keypoints = decode_keypoints
        self._format_input_tensor = _format_input_tensor
        self._run_inference = _run_inference

        self._interpreter = _load_interpreter(model_path)
        self._input_detail = self._interpreter.get_input_details()[0]
        self._simcc_x_detail, self._simcc_y_detail = _pick_simcc_outputs(
            self._interpreter.get_output_details())

    def predict(self, img_rgb, bbox_xywh):
        x, y, w, h = bbox_xywh
        bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
        normalized, center, scale = self._preprocess_instance(
            img_rgb, bbox_xyxy, self._bbox_xyxy2cs, self._get_warp_matrix)
        input_tensor = self._format_input_tensor(normalized, self._input_detail)
        simcc_x, simcc_y = self._run_inference(
            self._interpreter, self._input_detail,
            self._simcc_x_detail, self._simcc_y_detail, input_tensor)
        keypoints, scores = self._decode_keypoints(
            simcc_x, simcc_y, center, scale, self._get_simcc_maximum)
        return keypoints, scores


def _bbox_xywh_from_ann(ann, img_w, img_h):
    x, y, w, h = ann['bbox']
    x1 = np.clip(x, 0, img_w - 1)
    y1 = np.clip(y, 0, img_h - 1)
    x2 = np.clip(x + w, 0, img_w - 1)
    y2 = np.clip(y + h, 0, img_h - 1)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def render_coreml_overlays_from_json(
    variant,
    predictions_path,
    coco,
    img_prefix,
    output_dir,
    max_images=None,
    score_thr=0.2,
):
    if not predictions_path.exists():
        raise FileNotFoundError(
            f'Predictions not found: {predictions_path}. '
            'Run coreml_export_val_predictions.py on Mac (step 4a) and rsync JSONs here.')

    preds_by_image = _load_predictions_by_image(predictions_path)
    color = VARIANT_COLORS_BGR[variant]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = sorted(preds_by_image)
    if max_images is not None:
        img_ids = img_ids[:max_images]

    saved = 0
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(img_prefix) / img_info['file_name']
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        canvas = image_bgr.copy()
        draw_banner(canvas, f'{variant} | {MODEL_NAME} (from Mac preds)', color)

        for pred in preds_by_image[img_id]:
            if 'bbox' in pred:
                draw_bbox_xywh(canvas, pred['bbox'], color=(128, 128, 128), thickness=1)
            keypoints_xy, keypoint_scores = _parse_flat_keypoints(pred['keypoints'])
            draw_pose(canvas, keypoints_xy, keypoint_scores, color, score_thr=score_thr)

        out_name = f'{Path(img_info["file_name"]).stem}_overlay.jpg'
        cv2.imwrite(str(output_dir / out_name), canvas)
        saved += 1

    print(f'SUCCESS: Wrote {saved} overlays to {output_dir}')
    return saved


def render_tflite_overlays(
    variant,
    model_path,
    coco,
    img_prefix,
    output_dir,
    max_images=None,
    score_thr=0.2,
):
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    runner = TFLiteRunner(model_path)
    color = VARIANT_COLORS_BGR[variant]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    saved = 0
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(img_prefix) / img_info['file_name']
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        canvas = image_bgr.copy()
        draw_banner(canvas, f'{variant} | {MODEL_NAME}', color)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)[0]
            if 'bbox' not in ann or 'keypoints' not in ann:
                continue

            bbox_xywh = _bbox_xywh_from_ann(ann, img_w, img_h)
            draw_bbox_xywh(canvas, bbox_xywh, color=(128, 128, 128), thickness=1)
            keypoints_xy, keypoint_scores = runner.predict(img_rgb, bbox_xywh)
            draw_pose(canvas, keypoints_xy, keypoint_scores, color, score_thr=score_thr)

        out_name = f'{Path(img_info["file_name"]).stem}_overlay.jpg'
        cv2.imwrite(str(output_dir / out_name), canvas)
        saved += 1

    print(f'SUCCESS: Wrote {saved} overlays to {output_dir}')
    return saved


def main():
    parser = argparse.ArgumentParser(
        description='Overlay converted RTMPose predictions on val images (instance only).')
    parser.add_argument(
        '--variant',
        nargs='+',
        choices=list(VARIANTS) + ['all'],
        default=['all'],
        help='Which converted model(s) to visualize.',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Limit number of val images (default: all).',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=OVERLAYS_DIR,
        help='Root output directory (default: overlays/{MODEL_NAME}/).',
    )
    args = parser.parse_args()

    variants = list(VARIANTS) if 'all' in args.variant else args.variant

    if not ANN_FILE.exists() or not IMG_PREFIX.exists():
        print(f'ERROR: Val data not found under {ANN_FILE.parent.parent}', file=sys.stderr)
        sys.exit(1)

    coco = _load_coco(ANN_FILE)

    for variant in variants:
        out_dir = args.output_root / variant
        print(f'Rendering {variant} -> {out_dir}')
        if variant in COREML_PREDICTION_PATHS:
            render_coreml_overlays_from_json(
                variant=variant,
                predictions_path=COREML_PREDICTION_PATHS[variant],
                coco=coco,
                img_prefix=IMG_PREFIX,
                output_dir=out_dir,
                max_images=args.max_images,
            )
        else:
            render_tflite_overlays(
                variant=variant,
                model_path=TFLITE_MODEL_PATHS[variant],
                coco=coco,
                img_prefix=IMG_PREFIX,
                output_dir=out_dir,
                max_images=args.max_images,
            )


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f'ERROR: Overlay failed - {exc}', file=sys.stderr)
        sys.exit(1)
