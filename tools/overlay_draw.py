"""Draw COCO-17 poses on BGR images for validation overlays."""

import cv2
import numpy as np

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

VARIANT_COLORS_BGR = {
    'coreml_fp32': (0, 255, 0),
    'coreml_int8': (0, 165, 255),
    'tflite_fp32': (255, 255, 0),
    'tflite_int8': (255, 0, 255),
}


def draw_bbox_xywh(image, bbox_xywh, color=(128, 128, 128), thickness=1):
    x, y, w, h = bbox_xywh
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_pose(image, keypoints_xy, keypoint_scores, color, score_thr=0.2):
    """Draw skeleton + joints. keypoints_xy: (N, 2), scores: (N,)."""
    pts = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
    visible = scores > score_thr

    for i, j in COCO_SKELETON:
        if i < len(visible) and j < len(visible) and visible[i] and visible[j]:
            p1 = (int(pts[i, 0]), int(pts[i, 1]))
            p2 = (int(pts[j, 0]), int(pts[j, 1]))
            cv2.line(image, p1, p2, color, 2, lineType=cv2.LINE_AA)

    for i, ((x, y), vis) in enumerate(zip(pts, visible)):
        if not vis:
            continue
        cv2.circle(image, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_banner(image, text, color):
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        lineType=cv2.LINE_AA)
