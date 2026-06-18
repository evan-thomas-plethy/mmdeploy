"""Export COCO val predictions from CoreML models on macOS (no mmpose deps)."""

import argparse
import json
import platform
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import coremltools as ct
import cv2
import numpy as np

from rtmpose_coreml_utils import (
    apply_nms,
    crop_expanded_bbox,
    extract_simcc,
    postprocess_rtmpose,
    preprocess_image_rtmpose,
    resolve_io_names,
    run_coreml_predict,
)

from model_paths import (
    ANN_FILE,
    FP16_MLMODEL,
    FP16_PREDICTIONS,
    FP32_MLMODEL,
    FP32_PREDICTIONS,
    IMG_PREFIX,
    INT8_MLMODEL,
    INT8_PREDICTIONS,
    PREDICTIONS_DIR,
    VAL_DATA_ROOT,
)
from pose_eval_common import infer_precision_from_path


def _bbox_xywh_from_xyxy(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _default_output_path(model_path, predictions_dir):
    """Match model_paths naming: fp32 => {stem}_fp32.keypoints.json."""
    model_path = Path(model_path)
    out_dir = Path(predictions_dir)
    if infer_precision_from_path(model_path) == 'fp32':
        return out_dir / f'{model_path.stem}_fp32.keypoints.json'
    return out_dir / f'{model_path.stem}.keypoints.json'


def _load_coco_dataset(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    return data['annotations'], images


def export_predictions(model_path, ann_file, img_prefix, output_path, max_samples=None):
    annotations, images = _load_coco_dataset(ann_file)
    if max_samples is not None:
        annotations = annotations[:max_samples]

    model = ct.models.MLModel(str(model_path))
    input_name, simcc_x_name, simcc_y_name = resolve_io_names(model)

    raw_instances = []
    image_cache = {}

    for ann in annotations:
        if 'bbox' not in ann or 'keypoints' not in ann:
            continue

        img_id = ann['image_id']
        if img_id not in image_cache:
            img_info = images[img_id]
            img_path = Path(img_prefix) / img_info['file_name']
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f'Could not read image: {img_path}')
            image_cache[img_id] = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                img_info['width'],
                img_info['height'],
            )

        img_rgb, img_w, img_h = image_cache[img_id]
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)
        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        if 'area' in ann:
            area = float(ann['area'])
        else:
            area = float(np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None))

        crop, offset_x, offset_y = crop_expanded_bbox(img_rgb, bbox_xywh)
        pil_image, scale_x, scale_y, dx, dy = preprocess_image_rtmpose(crop)
        prediction = run_coreml_predict(model, pil_image, input_name)
        simcc_x, simcc_y = extract_simcc(prediction, simcc_x_name, simcc_y_name)
        keypoints = postprocess_rtmpose(
            simcc_x, simcc_y, scale_x, scale_y, dx, dy,
            offset_x, offset_y, img_w, img_h)

        raw_instances.append({
            'img_id': img_id,
            'category_id': ann.get('category_id', 1),
            'keypoints': keypoints[:, :2].astype(np.float32),
            'keypoint_scores': keypoints[:, 2].astype(np.float32),
            'bbox': _bbox_xywh_from_xyxy(x1, y1, x2, y2),
            'area': area,
        })

    coco_predictions = apply_nms(raw_instances)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_predictions, f)
    print(f'SUCCESS: Wrote {len(coco_predictions)} predictions to {output_path}')
    return coco_predictions


def main():
    if platform.system() != 'Darwin':
        print(
            'ERROR: CoreML model.predict() requires macOS. '
            'Run this script on a Mac, then compare AP with tools/compare_coco_ap.py.',
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Export COCO val predictions from CoreML models.')
    parser.add_argument(
        '--models',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='CoreML .mlmodel paths. Defaults to model_paths fp32/fp16/int8.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit val annotations.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ANN_FILE,
        help='COCO val annotations JSON.',
    )
    parser.add_argument(
        '--img-prefix',
        type=Path,
        default=IMG_PREFIX,
        help='Val images directory.',
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=PREDICTIONS_DIR,
        help='Directory for exported prediction JSONs.',
    )
    parser.add_argument(
        '--output',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='Output JSON per model (same order as --models).',
    )
    args = parser.parse_args()

    if not args.ann_file.exists() or not args.img_prefix.exists():
        print(f'ERROR: Val data not found under {VAL_DATA_ROOT}', file=sys.stderr)
        sys.exit(1)

    if args.models:
        model_paths = list(args.models)
        if args.output and len(args.output) != len(model_paths):
            print('ERROR: --output count must match --models', file=sys.stderr)
            sys.exit(1)
        output_paths = (
            list(args.output)
            if args.output
            else [_default_output_path(p, args.predictions_dir) for p in model_paths]
        )
    else:
        model_paths = [FP32_MLMODEL, FP16_MLMODEL, INT8_MLMODEL]
        output_paths = [FP32_PREDICTIONS, FP16_PREDICTIONS, INT8_PREDICTIONS]

    if args.max_samples is not None:
        print(f'Running on first {args.max_samples} annotations only')

    for model_path, output_path in zip(model_paths, output_paths):
        if not model_path.exists():
            print(f'ERROR: Model not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        print(f'Exporting predictions for {model_path.name} -> {output_path}')
        export_predictions(
            model_path,
            args.ann_file,
            args.img_prefix,
            output_path,
            max_samples=args.max_samples,
        )


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Export failed - {e}', file=sys.stderr)
        sys.exit(1)
