"""RTMPose CoreML pre/post-processing (production iOS pipeline)."""

from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

TARGET_WIDTH = 192
TARGET_HEIGHT = 256
NUM_KEYPOINTS = 17
KEYPOINT_SCORE_THR = 0.2
NMS_THR = 0.9
BBOX_PADDING = 1.25

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)

SIMCC_X_KEYS = ('linear_3', 'simcc_x')
SIMCC_Y_KEYS = ('linear_4', 'simcc_y')
INPUT_KEYS = ('inputs_1', 'input', 'x')


def _softmax(values):
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def crop_expanded_bbox(image_rgb, bbox_xywh, padding=BBOX_PADDING):
    """Crop person region with padding; returns crop and top-left offset in image."""
    img_h, img_w = image_rgb.shape[:2]
    x, y, w, h = bbox_xywh
    cx = x + w * 0.5
    cy = y + h * 0.5
    w_p = w * padding
    h_p = h * padding
    x1 = int(max(0, cx - w_p * 0.5))
    y1 = int(max(0, cy - h_p * 0.5))
    x2 = int(min(img_w, cx + w_p * 0.5))
    y2 = int(min(img_h, cy + h_p * 0.5))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return image_rgb[y1:y2, x1:x2], x1, y1


def preprocess_image_rtmpose(image_rgb):
    """Letterbox to 192x256 with black padding."""
    h, w = image_rgb.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    scaled_w = max(1, int(w * scale))
    scaled_h = max(1, int(h * scale))
    resized = cv2.resize(
        image_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    dx = (TARGET_WIDTH - scaled_w) // 2
    dy = (TARGET_HEIGHT - scaled_h) // 2
    output = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    output[dy:dy + scaled_h, dx:dx + scaled_w] = resized

    scale_x = w / scaled_w
    scale_y = h / scaled_h
    return Image.fromarray(output), scale_x, scale_y, float(dx), float(dy)


def postprocess_rtmpose(simcc_x, simcc_y, scale_x, scale_y, dx, dy,
                        offset_x, offset_y, img_w, img_h):
    """Decode SIMCC outputs into image-space keypoints."""
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    x_bins = simcc_x.shape[1]
    y_bins = simcc_y.shape[1]
    keypoints = []

    for kp in range(NUM_KEYPOINTS):
        raw_x = simcc_x[kp]
        raw_y = simcc_y[kp]
        prob_x = _softmax(raw_x)
        prob_y = _softmax(raw_y)
        max_idx_x = int(prob_x.argmax())
        max_idx_y = int(prob_y.argmax())

        avg_raw_logit = (raw_x[max_idx_x] + raw_y[max_idx_y]) / 2.0
        confidence = float(1.0 / (1.0 + np.exp(-avg_raw_logit)))

        norm_x = max_idx_x / (x_bins - 1)
        norm_y = max_idx_y / (y_bins - 1)
        model_x = norm_x * TARGET_WIDTH
        model_y = norm_y * TARGET_HEIGHT

        x_crop = (model_x - dx) * scale_x
        y_crop = (model_y - dy) * scale_y
        x_orig = min(max(x_crop + offset_x, 0.0), img_w - 1)
        y_orig = min(max(y_crop + offset_y, 0.0), img_h - 1)
        keypoints.append([x_orig, y_orig, confidence])

    return np.asarray(keypoints, dtype=np.float32)


def resolve_io_names(model):
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    output_by_name = {out.name: out.name for out in spec.description.output}
    output_names = list(output_by_name.keys())

    simcc_x_name = next((k for k in SIMCC_X_KEYS if k in output_by_name), None)
    simcc_y_name = next((k for k in SIMCC_Y_KEYS if k in output_by_name), None)
    if simcc_x_name is None or simcc_y_name is None:
        if len(output_names) >= 2:
            simcc_x_name, simcc_y_name = output_names[0], output_names[1]
        else:
            raise KeyError(f'Could not resolve simcc outputs from {output_names}')

    predict_input = next((k for k in INPUT_KEYS if k == input_name), input_name)
    return predict_input, simcc_x_name, simcc_y_name


def run_coreml_predict(model, pil_image, input_name):
    return model.predict({input_name: pil_image})


def extract_simcc(prediction, simcc_x_name, simcc_y_name):
    if simcc_x_name not in prediction or simcc_y_name not in prediction:
        raise KeyError(
            f'Missing simcc outputs {simcc_x_name}/{simcc_y_name} in {list(prediction)}'
        )
    return prediction[simcc_x_name], prediction[simcc_y_name]


def score_prediction(keypoint_scores, bbox_score=1.0):
    mean_kpt_score = 0.0
    valid_num = 0
    for score in keypoint_scores:
        if score > KEYPOINT_SCORE_THR:
            mean_kpt_score += score
            valid_num += 1
    if valid_num:
        mean_kpt_score /= valid_num
    return float(bbox_score * mean_kpt_score)


def oks_iou(g, d, area_g, areas_d, sigmas=COCO_SIGMAS):
    vars_ = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars_ / ((area_g + areas_d[n_d]) / 2 + np.spacing(1)) / 2
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) else 0.0
    return ious


def oks_nms(instances, thr=NMS_THR, sigmas=COCO_SIGMAS):
    if not instances:
        return []

    scores = np.array([inst['score'] for inst in instances])
    kpts = np.array([inst['keypoints'].reshape(-1) for inst in instances])
    areas = np.array([inst['area'] for inst in instances])
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas)
        inds = np.where(ious <= thr)[0]
        order = order[inds + 1]
    return keep


def format_coco_predictions(instances_by_image):
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


def apply_nms(raw_instances):
    instances_by_image = defaultdict(list)
    for inst in raw_instances:
        inst['score'] = score_prediction(inst['keypoint_scores'])
        instances_by_image[inst['img_id']].append(inst)

    filtered = defaultdict(list)
    for img_id, instances in instances_by_image.items():
        nms_input = []
        for inst in instances:
            nms_input.append({
                'img_id': inst['img_id'],
                'category_id': inst['category_id'],
                'keypoints': np.concatenate([
                    inst['keypoints'], inst['keypoint_scores'][:, None]
                ], axis=-1),
                'score': inst['score'],
                'bbox': inst['bbox'],
                'area': inst['area'],
            })
        for idx in oks_nms(nms_input):
            filtered[img_id].append(nms_input[int(idx)])
    return format_coco_predictions(filtered)
