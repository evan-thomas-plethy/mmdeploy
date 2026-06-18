import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import (
    ANN_FILE,
    FP16_TFLITE,
    FP32_TFLITE,
    IMG_PREFIX,
    INT8_TFLITE,
    MMPOSE_ROOT,
    PREDICTIONS_DIR,
    TFLITE_FP16_PREDICTIONS,
    TFLITE_FP32_PREDICTIONS,
    TFLITE_INT8_PREDICTIONS,
)
from pose_eval_common import compute_ap_from_predictions, print_ap_comparison_table

INPUT_SIZE = (192, 256)  # (w, h), matches training codec input_size
SIMCC_SPLIT_RATIO = 2.0
PADDING = 1.25
KEYPOINT_SCORE_THR = 0.2
NMS_THR = 0.9
NUM_KEYPOINTS = 17
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)


def _setup_mmpose_imports():
    if not MMPOSE_ROOT.exists():
        raise FileNotFoundError(
            f'mmpose repo not found at {MMPOSE_ROOT}. '
            'Expected a sibling mmpose checkout.'
        )
    mmpose_path = str(MMPOSE_ROOT)
    if mmpose_path not in sys.path:
        sys.path.insert(0, mmpose_path)

    from mmpose.codecs.utils import get_simcc_maximum
    from mmpose.evaluation.functional import oks_nms
    from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix

    return get_simcc_maximum, oks_nms, bbox_xyxy2cs, get_warp_matrix


def _fix_aspect_ratio(bbox_scale, aspect_ratio):
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def preprocess_instance(img_rgb, bbox_xyxy, bbox_xyxy2cs, get_warp_matrix):
    """Match mmpose val pipeline: GetBBoxCenterScale + TopdownAffine + normalize."""
    center, scale = bbox_xyxy2cs(bbox_xyxy, padding=PADDING)
    w, h = INPUT_SIZE
    scale = _fix_aspect_ratio(scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(center, scale, 0, output_size=(w, h))
    warped = cv2.warpAffine(
        img_rgb, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    normalized = (warped.astype(np.float32) - MEAN) / STD
    return normalized, center, scale


def _format_input_tensor(normalized_hwc, input_detail):
    shape = input_detail['shape']
    if len(shape) == 4 and shape[1] == 3:
        # NCHW
        nchw = np.transpose(normalized_hwc, (2, 0, 1))
        return np.expand_dims(nchw, axis=0).astype(input_detail['dtype'])
    # NHWC
    return np.expand_dims(normalized_hwc, axis=0).astype(input_detail['dtype'])


def decode_keypoints(simcc_x, simcc_y, center, scale, get_simcc_maximum):
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints.reshape(1, NUM_KEYPOINTS, 2)
    scores = scores.reshape(1, NUM_KEYPOINTS)
    keypoints = keypoints / SIMCC_SPLIT_RATIO

    w, h = INPUT_SIZE
    keypoints[..., 0] = keypoints[..., 0] / w * scale[0] + center[0] - scale[0] * 0.5
    keypoints[..., 1] = keypoints[..., 1] / h * scale[1] + center[1] - scale[1] * 0.5
    return keypoints[0], scores[0]


def _load_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _pick_simcc_outputs(output_details):
    preferred_pairs = [
        ('simcc_x', 'simcc_y'),
        ('Identity', 'Identity_1'),
        ('output_0', 'output_1'),
        ('PartitionedCall:0', 'PartitionedCall:1'),
    ]
    by_name = {detail['name']: detail for detail in output_details}
    for x_name, y_name in preferred_pairs:
        if x_name in by_name and y_name in by_name:
            return by_name[x_name], by_name[y_name]
    if len(output_details) >= 2:
        return output_details[0], output_details[1]
    raise ValueError('Could not determine TFLite simcc output tensors')


def _run_inference(interpreter, input_detail, simcc_x_detail, simcc_y_detail, input_tensor):
    interpreter.set_tensor(input_detail['index'], input_tensor)
    interpreter.invoke()
    simcc_x = interpreter.get_tensor(simcc_x_detail['index'])
    simcc_y = interpreter.get_tensor(simcc_y_detail['index'])
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]
    return simcc_x, simcc_y


def _bbox_xywh_from_xyxy(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _score_prediction(keypoint_scores, bbox_score=1.0):
    mean_kpt_score = 0.0
    valid_num = 0
    for score in keypoint_scores:
        if score > KEYPOINT_SCORE_THR:
            mean_kpt_score += score
            valid_num += 1
    if valid_num:
        mean_kpt_score /= valid_num
    return float(bbox_score * mean_kpt_score)


def _format_coco_predictions(instances_by_image):
    results = []
    for img_instances in instances_by_image.values():
        for inst in img_instances:
            flat = inst['keypoints'].reshape(-1).tolist()
            results.append({
                'image_id': inst['img_id'],
                'category_id': inst['category_id'],
                'keypoints': flat,
                'score': inst['score'],
                'bbox': inst['bbox'],
            })
    return results


def tflite_predictions_path(model_path, ann_file, max_samples=None, predictions_dir=None):
    """Stable cache path under predictions/ for a TFLite model + val set."""
    out_dir = Path(predictions_dir or PREDICTIONS_DIR)
    ann_tag = Path(ann_file).stem
    model_tag = Path(model_path).stem.replace('.', '_')
    name = f'tflite_{model_tag}_{ann_tag}'
    if max_samples is not None:
        name += f'_n{max_samples}'
    return out_dir / f'{name}.keypoints.json'


def evaluate_model(model_path, ann_file, img_prefix, get_simcc_maximum, oks_nms,
                   bbox_xyxy2cs, get_warp_matrix, max_samples=None,
                   predictions_path=None, force_rerun=False):
    if predictions_path is None:
        predictions_path = tflite_predictions_path(
            model_path, ann_file, max_samples=max_samples)
    else:
        predictions_path = Path(predictions_path)

    if predictions_path.exists() and not force_rerun:
        print(f'Using cached predictions: {predictions_path}')
        with open(predictions_path) as f:
            coco_predictions = json.load(f)
        return compute_ap_from_predictions(ann_file, coco_predictions)

    try:
        from xtcocotools.coco import COCO
    except ImportError:
        from pycocotools.coco import COCO

    interpreter = _load_interpreter(model_path)
    input_detail = interpreter.get_input_details()[0]
    simcc_x_detail, simcc_y_detail = _pick_simcc_outputs(
        interpreter.get_output_details())
    coco = COCO(str(ann_file))

    raw_instances = []
    image_cache = {}

    ann_ids = coco.getAnnIds()
    if max_samples is not None:
        ann_ids = ann_ids[:max_samples]

    for ann_id in ann_ids:
        ann = coco.loadAnns(ann_id)[0]
        if 'bbox' not in ann or 'keypoints' not in ann:
            continue

        img_id = ann['image_id']
        if img_id not in image_cache:
            img_info = coco.loadImgs(img_id)[0]
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
        bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
        if 'area' in ann:
            area = float(ann['area'])
        else:
            area = float(np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None))

        normalized, center, scale = preprocess_instance(
            img_rgb, bbox_xyxy, bbox_xyxy2cs, get_warp_matrix)
        input_tensor = _format_input_tensor(normalized, input_detail)
        simcc_x, simcc_y = _run_inference(
            interpreter, input_detail, simcc_x_detail, simcc_y_detail, input_tensor)
        keypoints, keypoint_scores = decode_keypoints(
            simcc_x, simcc_y, center, scale, get_simcc_maximum)

        raw_instances.append({
            'id': ann_id,
            'img_id': img_id,
            'category_id': ann.get('category_id', 1),
            'keypoints': keypoints.astype(np.float32),
            'keypoint_scores': keypoint_scores.astype(np.float32),
            'bbox_score': 1.0,
            'bbox': _bbox_xywh_from_xyxy(bbox_xyxy),
            'area': area,
        })

    instances_by_image = defaultdict(list)
    for inst in raw_instances:
        inst['score'] = _score_prediction(
            inst['keypoint_scores'], inst['bbox_score'])
        instances_by_image[inst['img_id']].append(inst)

    filtered_by_image = defaultdict(list)
    for img_id, instances in instances_by_image.items():
        nms_input = []
        for inst in instances:
            nms_input.append({
                'id': inst['id'],
                'img_id': inst['img_id'],
                'category_id': inst['category_id'],
                'keypoints': np.concatenate([
                    inst['keypoints'], inst['keypoint_scores'][:, None]
                ], axis=-1),
                'score': inst['score'],
                'bbox': inst['bbox'],
                'area': inst['area'],
            })
        keep = oks_nms(nms_input, NMS_THR, sigmas=COCO_SIGMAS)
        for idx in keep:
            filtered_by_image[img_id].append(nms_input[int(idx)])

    coco_predictions = _format_coco_predictions(filtered_by_image)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, 'w') as f:
        json.dump(coco_predictions, f)
    print(f'Saved predictions to {predictions_path}')

    return compute_ap_from_predictions(ann_file, coco_predictions)


def main():
    parser = argparse.ArgumentParser(
        description='Run TFLite val inference and compare COCO AP across models.')
    parser.add_argument(
        '--models',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='TFLite model paths (first = baseline). Defaults to model_paths fp32/fp16/int8.',
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
        help='Directory for cached prediction JSONs.',
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run inference even if prediction JSON exists.',
    )
    args = parser.parse_args()

    if args.models:
        model_paths = list(args.models)
        pred_paths = [
            tflite_predictions_path(
                p, args.ann_file, max_samples=args.max_samples,
                predictions_dir=args.predictions_dir)
            for p in model_paths
        ]
    else:
        model_paths = [FP32_TFLITE, FP16_TFLITE, INT8_TFLITE]
        pred_paths = [
            TFLITE_FP32_PREDICTIONS,
            TFLITE_FP16_PREDICTIONS,
            TFLITE_INT8_PREDICTIONS,
        ]

    required_paths = (*model_paths, args.ann_file, args.img_prefix)
    for path in required_paths:
        if not path.exists():
            print(f'ERROR: Required path not found: {path}', file=sys.stderr)
            sys.exit(1)

    if args.max_samples is not None:
        print(f'Running on first {args.max_samples} annotations only')

    from pose_eval_common import column_labels_for_paths

    labels = column_labels_for_paths(model_paths)
    get_simcc_maximum, oks_nms, bbox_xyxy2cs, get_warp_matrix = _setup_mmpose_imports()

    print(f'Annotation file: {args.ann_file}')
    print(f'Image prefix: {args.img_prefix}')
    print()

    eval_kwargs = dict(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        get_simcc_maximum=get_simcc_maximum,
        oks_nms=oks_nms,
        bbox_xyxy2cs=bbox_xyxy2cs,
        get_warp_matrix=get_warp_matrix,
        max_samples=args.max_samples,
        force_rerun=args.force_rerun,
    )

    metrics = {}
    for model_path, pred_path in zip(model_paths, pred_paths):
        label = labels[model_path]
        print(f'Evaluating {label} model: {model_path}')
        metrics[label] = evaluate_model(
            model_path, predictions_path=pred_path, **eval_kwargs)
        print()

    baseline_path = model_paths[0]
    baseline_label = labels[baseline_path]
    variant_metrics = {
        label: metrics[label]
        for path, label in labels.items()
        if path != baseline_path
    }

    print('=' * 96)
    print('COCO AP comparison (same metric as training save_best=coco/AP)')
    print('=' * 96)
    print_ap_comparison_table(
        metrics[baseline_label],
        variant_metrics,
        baseline_label=baseline_label,
    )


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Evaluation failed - {e}', file=sys.stderr)
        sys.exit(1)
