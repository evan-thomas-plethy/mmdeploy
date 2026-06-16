import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import ANN_FILE, FP32_TFLITE, IMG_PREFIX, INT8_TFLITE, MMPOSE_ROOT

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


def evaluate_model(model_path, ann_file, img_prefix, get_simcc_maximum, oks_nms,
                   bbox_xyxy2cs, get_warp_matrix, max_samples=None):
    try:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_predictions, f)
        pred_file = f.name

    coco_dt = coco.loadRes(pred_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints', COCO_SIGMAS, True)
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    Path(pred_file).unlink(missing_ok=True)

    stats_names = [
        'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
        'AR .75', 'AR (M)', 'AR (L)'
    ]
    return {name: float(value) for name, value in zip(stats_names, coco_eval.stats)}


def main():
    get_simcc_maximum, oks_nms, bbox_xyxy2cs, get_warp_matrix = _setup_mmpose_imports()

    fp32_path = FP32_TFLITE
    int8_path = INT8_TFLITE

    for path in (fp32_path, int8_path, ANN_FILE, IMG_PREFIX):
        if not path.exists():
            print(f'ERROR: Required path not found: {path}', file=sys.stderr)
            sys.exit(1)

    max_samples = None
    if len(sys.argv) > 1:
        max_samples = int(sys.argv[1])
        print(f'Running on first {max_samples} annotations only')

    print(f'Annotation file: {ANN_FILE}')
    print(f'Image prefix: {IMG_PREFIX}')
    print()

    eval_kwargs = dict(
        ann_file=ANN_FILE,
        img_prefix=IMG_PREFIX,
        get_simcc_maximum=get_simcc_maximum,
        oks_nms=oks_nms,
        bbox_xyxy2cs=bbox_xyxy2cs,
        get_warp_matrix=get_warp_matrix,
        max_samples=max_samples,
    )

    print(f'Evaluating fp32 model: {fp32_path}')
    fp32_metrics = evaluate_model(fp32_path, **eval_kwargs)
    print()

    print(f'Evaluating int8 model: {int8_path}')
    int8_metrics = evaluate_model(int8_path, **eval_kwargs)
    print()

    print('=' * 72)
    print('COCO AP comparison (same metric as training save_best=coco/AP)')
    print('=' * 72)
    print(f'{"Metric":<12} {"fp32":>12} {"int8":>12} {"delta":>12}')
    print('-' * 72)
    for metric in fp32_metrics:
        fp32_val = fp32_metrics[metric]
        int8_val = int8_metrics[metric]
        delta = int8_val - fp32_val
        print(f'{metric:<12} {fp32_val:>12.4f} {int8_val:>12.4f} {delta:>+12.4f}')

    ap_delta = int8_metrics['AP'] - fp32_metrics['AP']
    print('-' * 72)
    print(f'Primary coco/AP delta (int8 - fp32): {ap_delta:+.4f}')


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Evaluation failed - {e}', file=sys.stderr)
        sys.exit(1)
