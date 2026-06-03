import json
import sys
import tempfile
from pathlib import Path

import numpy as np

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)

STATS_NAMES = [
    'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
    'AR .75', 'AR (M)', 'AR (L)'
]


def load_coco_eval_modules():
    try:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


def compute_ap_from_predictions(ann_file, predictions):
    """Run COCO keypoint AP on a list of prediction dicts."""
    COCO, COCOeval = load_coco_eval_modules()
    coco = COCO(str(ann_file))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name

    coco_dt = coco.loadRes(pred_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints', COCO_SIGMAS, True)
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    Path(pred_file).unlink(missing_ok=True)
    return {name: float(value) for name, value in zip(STATS_NAMES, coco_eval.stats)}


def print_ap_comparison(metrics_a, metrics_b, label_a='fp32', label_b='int8'):
    print('=' * 72)
    print('COCO AP comparison (same metric as training save_best=coco/AP)')
    print('=' * 72)
    print(f'{"Metric":<12} {label_a:>12} {label_b:>12} {"delta":>12}')
    print('-' * 72)
    for metric in metrics_a:
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]
        delta = val_b - val_a
        print(f'{metric:<12} {val_a:>12.4f} {val_b:>12.4f} {delta:>+12.4f}')

    ap_delta = metrics_b['AP'] - metrics_a['AP']
    print('-' * 72)
    print(f'Primary coco/AP delta ({label_b} - {label_a}): {ap_delta:+.4f}')
