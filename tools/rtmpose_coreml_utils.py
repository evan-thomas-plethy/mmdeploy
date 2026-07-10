"""RTMPose exported-model pre/post-processing.

Default geometry mirrors the iOS production pipeline:
  optional applyDetectionCrop (1.25x) -> preprocessImageRTMPose letterbox
  -> decodeSIMCC -> crop-offset shift back to full image.

Written with plain numpy + cv2 so the same steps port to Swift/Kotlin.

Legacy top-down affine helpers (GetBBoxCenterScale + TopdownAffine) remain for
mmpose-aligned experiments but are not used by the val export scripts.
"""

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

TARGET_WIDTH = 192
TARGET_HEIGHT = 256
INPUT_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)  # (w, h), mmpose convention
ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT
NUM_KEYPOINTS = 17
KEYPOINT_SCORE_THR = 0.2
NMS_THR = 0.9
BBOX_PADDING = 1.25
SIMCC_SPLIT_RATIO = 2.0

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)

SIMCC_X_KEYS = ('linear_3', 'simcc_x')
SIMCC_Y_KEYS = ('linear_4', 'simcc_y')
INPUT_KEYS = ('inputs_1', 'input', 'x')

DETECTION_CONF_THR = 0.3


@dataclass(frozen=True)
class LetterboxParams:
    """Letterbox metadata for inverse mapping (matches preprocessImageRTMPose)."""

    scale_x: float
    scale_y: float
    dx: float
    dy: float
    target_width: int
    target_height: int
    orig_width: int
    orig_height: int


# ---------------------------------------------------------------------------
# Mobile production geometry (mirrors iOS applyDetectionCrop / preprocessImageRTMPose
# / decodeSIMCC / runRtmposeInference crop-offset shift).
# ---------------------------------------------------------------------------

def apply_detection_crop(image_rgb, bbox_xyxy, margin_scale=BBOX_PADDING,
                         min_conf=DETECTION_CONF_THR, bbox_conf=1.0):
    """Crop around bbox expanded by margin_scale. Matches iOS applyDetectionCrop.

    Returns (cropped_rgb, offset_x, offset_y) or None if crop is invalid.
    """
    if bbox_conf <= min_conf:
        return None

    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    h_img, w_img = image_rgb.shape[:2]

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0.0 or box_h <= 0.0:
        return None

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_w = box_w * margin_scale
    new_h = box_h * margin_scale

    new_x1 = max(0.0, min(cx - new_w * 0.5, float(w_img)))
    new_y1 = max(0.0, min(cy - new_h * 0.5, float(h_img)))
    new_x2 = max(0.0, min(cx + new_w * 0.5, float(w_img)))
    new_y2 = max(0.0, min(cy + new_h * 0.5, float(h_img)))

    left = max(0, int(np.floor(new_x1)))
    top = max(0, int(np.floor(new_y1)))
    right = min(int(w_img), int(np.ceil(new_x2)))
    bottom = min(int(h_img), int(np.ceil(new_y2)))

    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    if crop_w <= 1 or crop_h <= 1:
        return None

    cropped = image_rgb[top:bottom, left:right].copy()
    return cropped, float(left), float(top)


def preprocess_letterbox(image_rgb, output_size=INPUT_SIZE):
    """Letterbox RGB image to output_size (w, h) with black padding.

    Matches iOS preprocessImageRTMPose:
      scale = min(tw/ow, th/oh)
      scaledW/H = round(orig * scale)
      dx/dy = (target - scaled) / 2
      draw scaled image into black canvas at (dx, dy)
      scaleX/Y = orig / scaled  (for inverse mapping)

    Uses warpAffine so placement honors fractional dx/dy like CoreGraphics.
    Returns (canvas_uint8_rgb, LetterboxParams).
    """
    target_w, target_h = int(output_size[0]), int(output_size[1])
    orig_h, orig_w = image_rgb.shape[:2]

    scale = min(target_w / float(orig_w), target_h / float(orig_h))
    scaled_w = float(round(orig_w * scale))
    scaled_h = float(round(orig_h * scale))
    if scaled_w < 1.0:
        scaled_w = 1.0
    if scaled_h < 1.0:
        scaled_h = 1.0

    dx = (target_w - scaled_w) / 2.0
    dy = (target_h - scaled_h) / 2.0

    # x' = x * (scaled_w/orig_w) + dx ; y' = y * (scaled_h/orig_h) + dy
    # Equivalent to resize then draw-at-(dx,dy) with fractional offsets.
    sx = scaled_w / float(orig_w)
    sy = scaled_h / float(orig_h)
    warp = np.array([[sx, 0.0, dx], [0.0, sy, dy]], dtype=np.float64)
    canvas = cv2.warpAffine(
        image_rgb,
        warp,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    params = LetterboxParams(
        scale_x=float(orig_w) / scaled_w,
        scale_y=float(orig_h) / scaled_h,
        dx=dx,
        dy=dy,
        target_width=target_w,
        target_height=target_h,
        orig_width=int(orig_w),
        orig_height=int(orig_h),
    )
    return canvas, params


def preprocess_letterbox_pil(image_rgb, output_size=INPUT_SIZE):
    """As preprocess_letterbox but returns a PIL image for CoreML ImageType inputs."""
    canvas, params = preprocess_letterbox(image_rgb, output_size=output_size)
    return Image.fromarray(canvas), params


def preprocess_rtmpose_mobile(image_rgb, bbox_xyxy=None, output_size=INPUT_SIZE,
                              margin_scale=BBOX_PADDING, bbox_conf=1.0):
    """Full mobile preprocess matching runRtmposeInference geometry.

    Optional detection crop (applyDetectionCrop), then letterbox
    (preprocessImageRTMPose). Returns
    (input_rgb_uint8, params, crop_offset_x, crop_offset_y).
    """
    pose_input = image_rgb
    offset_x = 0.0
    offset_y = 0.0

    if bbox_xyxy is not None:
        crop_result = apply_detection_crop(
            image_rgb, bbox_xyxy, margin_scale=margin_scale, bbox_conf=bbox_conf)
        if crop_result is not None:
            pose_input, offset_x, offset_y = crop_result

    canvas, params = preprocess_letterbox(pose_input, output_size=output_size)
    return canvas, params, offset_x, offset_y


def preprocess_rtmpose_mobile_pil(image_rgb, bbox_xyxy=None, output_size=INPUT_SIZE,
                                  margin_scale=BBOX_PADDING, bbox_conf=1.0):
    """As preprocess_rtmpose_mobile but returns PIL image for CoreML."""
    canvas, params, offset_x, offset_y = preprocess_rtmpose_mobile(
        image_rgb, bbox_xyxy=bbox_xyxy, output_size=output_size,
        margin_scale=margin_scale, bbox_conf=bbox_conf)
    return Image.fromarray(canvas), params, offset_x, offset_y


def decode_simcc_mobile(simcc_x, simcc_y, params: LetterboxParams):
    """Decode SimCC into poseInput pixel coords. Matches iOS decodeSIMCC.

    For each keypoint:
      xIndex/yIndex = argmax(logits)  # same as argmax(softmax(logits))
      confidence = sigmoid((xPeak + yPeak) / 2)
      modelX = (xIndex / (xBins-1)) * inputWidth
      adjX = (modelX - dx) * scaleX ; clip to [0, originalWidth]
    """
    simcc_x = np.asarray(simcc_x, dtype=np.float32)
    simcc_y = np.asarray(simcc_y, dtype=np.float32)
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    num_kpts = int(simcc_x.shape[0])
    simcc_x_len = int(simcc_x.shape[-1])
    simcc_y_len = int(simcc_y.shape[-1])
    x_den = max(1, simcc_x_len - 1)
    y_den = max(1, simcc_y_len - 1)

    keypoints = np.zeros((num_kpts, 3), dtype=np.float32)
    for kp in range(num_kpts):
        x_logits = simcc_x[kp]
        y_logits = simcc_y[kp]
        x_index = int(np.argmax(x_logits))
        y_index = int(np.argmax(y_logits))

        x_peak_logit = float(x_logits[x_index])
        y_peak_logit = float(y_logits[y_index])
        avg_logit = (x_peak_logit + y_peak_logit) / 2.0
        confidence = 1.0 / (1.0 + np.exp(-avg_logit))

        norm_x = x_index / float(x_den)
        norm_y = y_index / float(y_den)
        model_x = norm_x * float(params.target_width)
        model_y = norm_y * float(params.target_height)

        adj_x = (model_x - params.dx) * params.scale_x
        adj_y = (model_y - params.dy) * params.scale_y

        keypoints[kp, 0] = min(max(adj_x, 0.0), float(params.orig_width))
        keypoints[kp, 1] = min(max(adj_y, 0.0), float(params.orig_height))
        keypoints[kp, 2] = confidence

    return keypoints


# Back-compat alias used by earlier val scripts.
decode_simcc_letterbox = decode_simcc_mobile


def postprocess_letterbox(simcc_x, simcc_y, params: LetterboxParams,
                          crop_offset_x=0.0, crop_offset_y=0.0):
    """decodeSIMCC in poseInput space, then shift to full-image if cropped.

    Mirrors runRtmposeInference: decode on poseInput, then
    keypoints += (cropOffsetX, cropOffsetY) when a detection crop was applied.
    """
    keypoints = decode_simcc_mobile(simcc_x, simcc_y, params)
    if crop_offset_x != 0.0 or crop_offset_y != 0.0:
        keypoints = keypoints.copy()
        keypoints[:, 0] += crop_offset_x
        keypoints[:, 1] += crop_offset_y
    return keypoints


# ---------------------------------------------------------------------------
# Top-down affine preprocessing.
# Step-for-step port of mmpose GetBBoxCenterScale + TopdownAffine (use_udp=False,
# rotation=0 at inference). Every step below has a trivial mobile equivalent.
# ---------------------------------------------------------------------------

def bbox_xyxy_to_center_scale(bbox_xyxy, padding=BBOX_PADDING):
    """[x1, y1, x2, y2] -> (center(2,), scale(2,)). Mirrors GetBBoxCenterScale.

    center = bbox midpoint; scale = bbox (w, h) * padding (1.25).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def fix_aspect_ratio(scale, aspect_ratio=ASPECT_RATIO):
    """Grow scale (w, h) to the model aspect ratio w/h. Mirrors _fix_aspect_ratio.

    if w > h * aspect: h = w / aspect  else: w = h * aspect
    """
    w, h = float(scale[0]), float(scale[1])
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    return np.array([w, h], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    """Rotate 2D point about the origin (CCW)."""
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs],
                    dtype=np.float32)


def _third_point(a, b):
    """Third correspondence point: rotate vector (a - b) by 90deg CCW around b."""
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def get_warp_matrix(center, scale, rot_deg, output_size=INPUT_SIZE):
    """2x3 affine mapping the padded/aspect-fixed bbox onto output_size (w, h).

    Faithful reimplementation of mmpose get_warp_matrix (use_udp=False) using
    three source/destination point correspondences, then solving the affine.
    Mobile equivalents (build the SAME 3 src/dst points):
      - Android: android.graphics.Matrix.setPolyToPoly(src, 0, dst, 0, 3)
      - iOS:     solve the 2x3 from the 3 pairs (or use vImage affine warp)
    """
    dst_w, dst_h = float(output_size[0]), float(output_size[1])
    src_w = float(scale[0])
    rot_rad = np.deg2rad(rot_deg)

    src_dir = _rotate_point(np.array([src_w * -0.5, 0.0], dtype=np.float32),
                            rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.0], dtype=np.float32)

    center = np.asarray(center, dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _third_point(src[0], src[1])

    dst_mid = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = dst_mid
    dst[1] = dst_mid + dst_dir
    dst[2] = _third_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def topdown_affine(image_rgb, center, scale, output_size=INPUT_SIZE):
    """Warp the person region into output_size (w, h). Returns uint8 RGB (h, w, 3).

    No letterbox / black bars: the padded, aspect-fixed bbox is affine-warped to
    fill the whole output canvas (bilinear), exactly like mmpose TopdownAffine.
    """
    out_w, out_h = int(output_size[0]), int(output_size[1])
    warp_mat = get_warp_matrix(center, scale, 0.0, output_size)
    return cv2.warpAffine(
        image_rgb, warp_mat, (out_w, out_h), flags=cv2.INTER_LINEAR)


def preprocess_topdown(image_rgb, bbox_xyxy, output_size=INPUT_SIZE,
                       padding=BBOX_PADDING):
    """Full geometric preprocess: bbox -> warped uint8 crop + (center, scale).

    Returns (warped_uint8_rgb, center, scale). Keep center/scale for decoding.
    """
    center, scale = bbox_xyxy_to_center_scale(bbox_xyxy, padding=padding)
    scale = fix_aspect_ratio(scale, output_size[0] / output_size[1])
    warped = topdown_affine(image_rgb, center, scale, output_size)
    return warped, center, scale


def preprocess_topdown_pil(image_rgb, bbox_xyxy, output_size=INPUT_SIZE,
                           padding=BBOX_PADDING):
    """As preprocess_topdown but returns the warped crop as a PIL image.

    Used for CoreML ImageType inputs (normalization is baked into the model).
    """
    warped, center, scale = preprocess_topdown(
        image_rgb, bbox_xyxy, output_size=output_size, padding=padding)
    return Image.fromarray(warped), center, scale


# ---------------------------------------------------------------------------
# SimCC decode + input-space -> image-space mapping.
# Mirrors mmpose get_simcc_maximum(apply_softmax=False) + SimCCLabel.decode +
# TopdownPoseEstimator.add_pred_to_datasample.
# ---------------------------------------------------------------------------

def _simcc_argmax(simcc):
    """simcc (K, W) -> (locs(K,), vals(K,)) via per-row argmax / max."""
    locs = np.argmax(simcc, axis=1).astype(np.float32)
    vals = np.amax(simcc, axis=1).astype(np.float32)
    return locs, vals


def decode_simcc(simcc_x, simcc_y, split_ratio=SIMCC_SPLIT_RATIO):
    """SimCC 1-D outputs -> model-input keypoints (K,2) + scores (K,).

    Steps (no softmax, matching mmpose val decode):
      x_bin = argmax(simcc_x); y_bin = argmax(simcc_y)
      score = min(max(simcc_x), max(simcc_y))
      coords with score <= 0 -> -1 (invalid)
      coords /= split_ratio  # simcc bins -> model input pixels (w=192, h=256)
    """
    simcc_x = np.asarray(simcc_x, dtype=np.float32)
    simcc_y = np.asarray(simcc_y, dtype=np.float32)
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    x_locs, x_vals = _simcc_argmax(simcc_x)
    y_locs, y_vals = _simcc_argmax(simcc_y)

    coords = np.stack([x_locs, y_locs], axis=-1)  # (K, 2) in simcc-bin units
    scores = np.minimum(x_vals, y_vals)
    coords[scores <= 0.0] = -1.0
    coords = coords / split_ratio
    return coords.astype(np.float32), scores.astype(np.float32)


def model_coords_to_image(coords_model, center, scale, output_size=INPUT_SIZE):
    """Map model-input keypoints back to original image pixels (inverse warp, rot=0).

    Matches mmpose TopdownPoseEstimator.add_pred_to_datasample:
      img = model / input_size * scale + center - 0.5 * scale
    """
    input_w, input_h = float(output_size[0]), float(output_size[1])
    scale_w, scale_h = float(scale[0]), float(scale[1])
    out = np.empty_like(coords_model, dtype=np.float32)
    out[..., 0] = coords_model[..., 0] / input_w * scale_w + center[0] - 0.5 * scale_w
    out[..., 1] = coords_model[..., 1] / input_h * scale_h + center[1] - 0.5 * scale_h
    return out


def postprocess_topdown(simcc_x, simcc_y, center, scale, output_size=INPUT_SIZE):
    """Decode SimCC and map to image space. Returns (K, 3): x, y, score."""
    coords_model, scores = decode_simcc(simcc_x, simcc_y)
    coords_img = model_coords_to_image(coords_model, center, scale, output_size)
    return np.concatenate(
        [coords_img, scores[:, None]], axis=-1).astype(np.float32)


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
