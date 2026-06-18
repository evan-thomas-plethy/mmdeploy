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


def infer_precision_from_path(path):
    """Infer fp32 / fp16 / int8 from model filename (no suffix => fp32)."""
    stem = Path(path).stem.lower()
    if '_int8' in stem or stem.endswith('int8'):
        return 'int8'
    if any(tag in stem for tag in ('_fp16', '_float16', 'float_16', 'float16')):
        return 'fp16'
    if any(tag in stem for tag in ('_float32', 'float_32', 'float32')):
        return 'fp32'
    return 'fp32'


def column_labels_for_paths(paths):
    """Assign unique table column labels from model/prediction paths."""
    paths = [Path(p) for p in paths]
    precisions = [infer_precision_from_path(p) for p in paths]
    precision_counts = {}
    for prec in precisions:
        precision_counts[prec] = precision_counts.get(prec, 0) + 1

    labels = {}
    for path, prec in zip(paths, precisions):
        labels[path] = prec if precision_counts[prec] == 1 else path.stem
    return labels


def infer_precision_from_meta_label(meta_value):
    """Map meta.json checkpoint name to precision (no suffix => fp32)."""
    return infer_precision_from_path(meta_value)


def version_column_label(version_key, meta_value):
    """Column label e.g. v1_int8 from meta.json version key + checkpoint name."""
    return f'{version_key}_{infer_precision_from_meta_label(meta_value)}'


def print_ap_comparison_table(
        baseline_metrics,
        variant_metrics_by_label,
        baseline_label='fp32'):
    """Print AP table: one baseline column plus variant columns and deltas."""
    labels = list(variant_metrics_by_label)
    header = f'{"Metric":<12} {baseline_label:>12}'
    for label in labels:
        header += f' {label:>12}'
    for label in labels:
        header += f' {"d" + label:>12}'
    print(header)
    print('-' * len(header))

    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        row = f'{metric:<12} {baseline_val:>12.4f}'
        for label in labels:
            row += f' {variant_metrics_by_label[label][metric]:>12.4f}'
        for label in labels:
            delta = variant_metrics_by_label[label][metric] - baseline_val
            row += f' {delta:>+12.4f}'
        print(row)

    print('-' * len(header))
    for label in labels:
        ap_delta = variant_metrics_by_label[label]['AP'] - baseline_metrics['AP']
        print(f'Primary coco/AP delta ({label} - {baseline_label}): {ap_delta:+.4f}')


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
