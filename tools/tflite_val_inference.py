"""TFLite val inference shared by tflite_export_val_predictions.py.

Pre/post matches mmpose val_pipeline geometry (portable to Swift/Java):
  GetBBoxCenterScale(1.25) -> TopdownAffine(192x256) -> SimCC decode
  -> map model coords back to image space.
"""

import json
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

from rtmpose_coreml_utils import (
    apply_nms,
    postprocess_topdown,
    preprocess_topdown,
)

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _bbox_xywh_from_xyxy(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def normalize_for_tflite(warped_rgb):
    """Normalize a warped uint8 RGB crop (H, W, 3) for TFLite input.

    (pixel - MEAN) / STD in RGB order, matching the mmpose data preprocessor
    (bgr_to_rgb=True + ImageNet mean/std).
    """
    arr = np.asarray(warped_rgb, dtype=np.float32)
    return (arr - MEAN) / STD


def _format_input_tensor(normalized_hwc, input_detail):
    shape = input_detail['shape']
    if len(shape) == 4 and shape[1] == 3:
        nchw = np.transpose(normalized_hwc, (2, 0, 1))
        return np.expand_dims(nchw, axis=0).astype(input_detail['dtype'])
    return np.expand_dims(normalized_hwc, axis=0).astype(input_detail['dtype'])


def _load_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _pick_simcc_outputs(output_details):
    """Resolve simcc_x (384) and simcc_y (512), handling swapped output order."""
    if len(output_details) < 2:
        raise ValueError('Expected at least two TFLite output tensors')

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

    out0, out1 = output_details[0], output_details[1]
    last0 = int(out0['shape'][-1])
    last1 = int(out1['shape'][-1])
    if last0 == 384 and last1 == 512:
        return out0, out1
    if last0 == 512 and last1 == 384:
        return out1, out0
    return out0, out1


def _run_inference(interpreter, input_detail, simcc_x_detail, simcc_y_detail, input_tensor):
    interpreter.set_tensor(input_detail['index'], input_tensor)
    interpreter.invoke()
    simcc_x = interpreter.get_tensor(simcc_x_detail['index'])
    simcc_y = interpreter.get_tensor(simcc_y_detail['index'])
    return simcc_x, simcc_y


def export_tflite_predictions(
    model_path,
    ann_file,
    img_prefix,
    output_path,
    max_samples=None,
    force_rerun=False,
    no_bbox=False,
):
    """Run TFLite val inference and write COCO keypoints JSON.

    Default: each COCO annotation bbox with mmpose-aligned topdown preprocess
    (GetBBoxCenterScale + TopdownAffine). With no_bbox=True, use the full image
    [0, 0, W, H] as the bbox and apply the same preprocess.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    if output_path.exists() and not force_rerun:
        print(f'Using existing predictions: {output_path}')
        with open(output_path) as f:
            return json.load(f)

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
        if 'keypoints' not in ann:
            continue
        if not no_bbox and 'bbox' not in ann:
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

        if no_bbox:
            x1, y1, x2, y2 = 0.0, 0.0, float(img_w), float(img_h)
        else:
            x, y, w, h = ann['bbox']
            x1 = np.clip(x, 0, img_w - 1)
            y1 = np.clip(y, 0, img_h - 1)
            x2 = np.clip(x + w, 0, img_w - 1)
            y2 = np.clip(y + h, 0, img_h - 1)
        bbox_xywh = _bbox_xywh_from_xyxy([x1, y1, x2, y2])

        if 'area' in ann:
            area = float(ann['area'])
        else:
            area = float(np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None))

        warped, center, scale = preprocess_topdown(
            img_rgb, bbox_xyxy=[x1, y1, x2, y2])
        normalized = normalize_for_tflite(warped)
        input_tensor = _format_input_tensor(normalized, input_detail)
        simcc_x, simcc_y = _run_inference(
            interpreter, input_detail, simcc_x_detail, simcc_y_detail, input_tensor)
        keypoints = postprocess_topdown(simcc_x, simcc_y, center, scale)

        raw_instances.append({
            'img_id': img_id,
            'category_id': ann.get('category_id', 1),
            'keypoints': keypoints[:, :2].astype(np.float32),
            'keypoint_scores': keypoints[:, 2].astype(np.float32),
            'bbox': bbox_xywh,
            'area': area,
        })

    coco_predictions = apply_nms(raw_instances)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_predictions, f)
    print(f'SUCCESS: Wrote {len(coco_predictions)} predictions to {output_path}')
    return coco_predictions
