"""Export COCO val predictions from CoreML models on macOS (no mmpose deps)."""

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

# Must match MODEL_NAME in rtmpose_torch_to_coreml.py / coreml_weight_quantize.py
MODEL_NAME = (
    'rtmpose-m_merged_gbe_v6_various_datasets_sweep_lyingperson_unfreeze_full_ld0.5'
)
FP32_MODEL = f'{MODEL_NAME}.mlmodel'
INT8_MODEL = f'{MODEL_NAME}_int8.mlmodel'

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_DATA_ROOT = REPO_ROOT.parent / 'mmpose/data/coco_ground_based_exercises_v6'
ANN_FILE = VAL_DATA_ROOT / 'annotations/person_keypoints_val2017.json'
IMG_PREFIX = VAL_DATA_ROOT / 'val2017'
PREDICTIONS_DIR = REPO_ROOT / 'predictions'
FP32_PREDICTIONS = PREDICTIONS_DIR / f'{MODEL_NAME}_fp32.keypoints.json'
INT8_PREDICTIONS = PREDICTIONS_DIR / f'{MODEL_NAME}_int8.keypoints.json'


def _bbox_xywh_from_xyxy(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


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

    if not ANN_FILE.exists() or not IMG_PREFIX.exists():
        print(f'ERROR: Val data not found under {VAL_DATA_ROOT}', file=sys.stderr)
        sys.exit(1)

    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if max_samples is not None:
        print(f'Running on first {max_samples} annotations only')

    models = [
        (REPO_ROOT / FP32_MODEL, FP32_PREDICTIONS),
        (REPO_ROOT / INT8_MODEL, INT8_PREDICTIONS),
    ]
    for model_path, output_path in models:
        if not model_path.exists():
            print(f'ERROR: Model not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        print(f'Exporting predictions for {model_path.name}...')
        export_predictions(
            model_path, ANN_FILE, IMG_PREFIX, output_path, max_samples=max_samples)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Export failed - {e}', file=sys.stderr)
        sys.exit(1)
