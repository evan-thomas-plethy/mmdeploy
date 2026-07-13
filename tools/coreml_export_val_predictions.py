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

from model_paths import PREDICTIONS_DIR
from pose_eval_common import predictions_output_path
from rtmpose_coreml_utils import (
    apply_nms,
    extract_simcc,
    postprocess_topdown,
    preprocess_topdown_pil,
    resolve_io_names,
    run_coreml_predict,
)


def _bbox_xywh_from_xyxy(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _load_coco_dataset(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    return data['annotations'], images


def export_predictions(
    model_path,
    ann_file,
    img_prefix,
    output_path,
    max_samples=None,
    force_rerun=False,
    no_bbox=False,
):
    """Run CoreML val inference with mmpose-aligned topdown preprocess.

    Per annotation: GetBBoxCenterScale(1.25) + TopdownAffine(192x256) + SimCC
    decode mapped back to image space (same geometry as mmpose val_pipeline).
    With no_bbox=True, use the full image [0, 0, W, H] as the bbox.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force_rerun:
        print(f'Using existing predictions: {output_path}')
        with open(output_path) as f:
            return json.load(f)

    annotations, images = _load_coco_dataset(ann_file)
    if max_samples is not None:
        annotations = annotations[:max_samples]

    model = ct.models.MLModel(str(model_path))
    input_name, simcc_x_name, simcc_y_name = resolve_io_names(model)

    raw_instances = []
    image_cache = {}

    for ann in annotations:
        if 'keypoints' not in ann:
            continue
        if not no_bbox and 'bbox' not in ann:
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
        if no_bbox:
            x1, y1, x2, y2 = 0.0, 0.0, float(img_w), float(img_h)
        else:
            x, y, w, h = ann['bbox']
            x1 = np.clip(x, 0, img_w - 1)
            y1 = np.clip(y, 0, img_h - 1)
            x2 = np.clip(x + w, 0, img_w - 1)
            y2 = np.clip(y + h, 0, img_h - 1)

        if 'area' in ann:
            area = float(ann['area'])
        else:
            area = float(np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None))

        pil_image, center, scale = preprocess_topdown_pil(
            img_rgb, bbox_xyxy=[x1, y1, x2, y2])
        prediction = run_coreml_predict(model, pil_image, input_name)
        simcc_x, simcc_y = extract_simcc(prediction, simcc_x_name, simcc_y_name)
        keypoints = postprocess_topdown(simcc_x, simcc_y, center, scale)

        raw_instances.append({
            'img_id': img_id,
            'category_id': ann.get('category_id', 1),
            'keypoints': keypoints[:, :2].astype(np.float32),
            'keypoint_scores': keypoints[:, 2].astype(np.float32),
            'bbox': _bbox_xywh_from_xyxy(x1, y1, x2, y2),
            'area': area,
        })

    coco_predictions = apply_nms(raw_instances)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_predictions, f)
    print(f'SUCCESS: Wrote {len(coco_predictions)} predictions to {output_path}')
    return coco_predictions


def main():
    if platform.system() != 'Darwin':
        print(
            'ERROR: CoreML model.predict() requires macOS. '
            'Run this script on a Mac, then compare AP with tools/coreml_compare_ap.py.',
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
        required=True,
        help='CoreML .mlmodel paths.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        help='COCO val annotations JSON.',
    )
    parser.add_argument(
        '--img-prefix',
        type=Path,
        required=True,
        help='Val images directory.',
    )
    parser.add_argument(
        '--dataset-name',
        default=None,
        help='Dataset tag for output filename (default: parent of annotations/).',
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=PREDICTIONS_DIR,
        help='Parent directory for prediction JSONs (default: predictions/).',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit val annotations.',
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run inference even if prediction JSON exists.',
    )
    parser.add_argument(
        '--no-bbox',
        action='store_true',
        help='Ignore COCO ann bbox; use the full image [0,0,W,H] as bbox for topdown preprocess.',
    )
    args = parser.parse_args()

    if not args.ann_file.exists():
        print(f'ERROR: Annotations not found: {args.ann_file}', file=sys.stderr)
        sys.exit(1)
    if not args.img_prefix.exists():
        print(f'ERROR: Image prefix not found: {args.img_prefix}', file=sys.stderr)
        sys.exit(1)

    if args.max_samples is not None:
        print(f'Running on first {args.max_samples} annotations only')
    if args.no_bbox:
        print('Using full-image bbox (--no-bbox)')

    for model_path in args.models:
        if not model_path.exists():
            print(f'ERROR: Model not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        output_path = predictions_output_path(
            'coreml',
            model_path,
            args.ann_file,
            predictions_dir=args.predictions_dir,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            no_bbox=args.no_bbox,
        )
        print(f'Exporting predictions for {model_path.name} -> {output_path}')
        export_predictions(
            model_path,
            args.ann_file,
            args.img_prefix,
            output_path,
            max_samples=args.max_samples,
            force_rerun=args.force_rerun,
            no_bbox=args.no_bbox,
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
