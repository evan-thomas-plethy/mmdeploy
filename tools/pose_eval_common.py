import json
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
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


def sanitize_path_tag(name):
    """Normalize a path component for use in prediction filenames."""
    return str(name).replace('.', '_').replace(' ', '_')


def infer_dataset_name(ann_file, dataset_name=None):
    """Derive dataset tag from --dataset-name or val annotation layout."""
    if dataset_name:
        return sanitize_path_tag(dataset_name)
    ann_path = Path(ann_file)
    if ann_path.parent.name == 'annotations':
        return sanitize_path_tag(ann_path.parent.parent.name)
    return sanitize_path_tag(ann_path.stem)


def predictions_output_path(
    backend,
    model_path,
    ann_file,
    predictions_dir=None,
    dataset_name=None,
    max_samples=None,
    no_bbox=False,
):
    """Build predictions/{backend}_{model}_{dataset}[_nobbox].keypoints.json."""
    backend = backend.lower()
    if backend not in ('coreml', 'tflite'):
        raise ValueError(f'backend must be coreml or tflite, got {backend!r}')
    from model_paths import PREDICTIONS_DIR

    model_tag = sanitize_path_tag(Path(model_path).stem)
    dataset_tag = infer_dataset_name(ann_file, dataset_name)
    filename = f'{backend}_{model_tag}_{dataset_tag}'
    if no_bbox:
        filename += '_nobbox'
    if max_samples is not None:
        filename += f'_n{max_samples}'
    out_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
    return out_dir / f'{filename}.keypoints.json'


def prediction_label_from_path(path):
    """Human-readable label from a prediction JSON path (no precision claims)."""
    name = Path(path).name
    if name.endswith('.keypoints.json'):
        return name[: -len('.keypoints.json')]
    if name.endswith('.json'):
        return name[: -len('.json')]
    return Path(path).stem


def infer_precision_from_path(path):
    """Infer fp32 / fp16 / int8 from model filename (no suffix => fp32).

    Kept for overlay color heuristics; AP reports use prediction_label_from_path.
    """
    stem = Path(path).stem.lower()
    if '_int8' in stem or stem.endswith('int8'):
        return 'int8'
    if any(tag in stem for tag in ('_fp16', '_float16', 'float_16', 'float16')):
        return 'fp16'
    if any(tag in stem for tag in ('_float32', 'float_32', 'float32')):
        return 'fp32'
    return 'fp32'


def column_labels_for_paths(paths):
    """Assign unique table column labels from prediction/model filenames."""
    paths = [Path(p) for p in paths]
    bases = [prediction_label_from_path(p) for p in paths]
    counts = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1

    labels = {}
    seen = {}
    for path, base in zip(paths, bases):
        if counts[base] == 1:
            labels[path] = base
        else:
            seen[base] = seen.get(base, 0) + 1
            labels[path] = f'{base}_{seen[base]}'
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
        baseline_label='baseline'):
    """Print AP table: one baseline column plus variant columns and deltas."""
    labels = list(variant_metrics_by_label)
    all_labels = [baseline_label] + labels
    col_w = max(12, max(len(lbl) for lbl in all_labels), max(len('d' + lbl) for lbl in labels) if labels else 12)

    header = f'{"Metric":<12}'
    for lbl in all_labels:
        header += f' {lbl:>{col_w}}'
    for lbl in labels:
        header += f' {"d" + lbl:>{col_w}}'
    print(header)
    print('-' * len(header))

    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        row = f'{metric:<12} {baseline_val:>{col_w}.4f}'
        for label in labels:
            row += f' {variant_metrics_by_label[label][metric]:>{col_w}.4f}'
        for label in labels:
            delta = variant_metrics_by_label[label][metric] - baseline_val
            row += f' {delta:>+{col_w}.4f}'
        print(row)

    print('-' * len(header))
    for label in labels:
        ap_delta = variant_metrics_by_label[label]['AP'] - baseline_metrics['AP']
        print(f'Primary coco/AP delta ({label} - {baseline_label}): {ap_delta:+.4f}')


def print_ap_comparison(metrics_a, metrics_b, label_a='a', label_b='b'):
    col_w = max(12, len(label_a), len(label_b), len('delta'))
    print('=' * 72)
    print('COCO AP comparison (same metric as training save_best=coco/AP)')
    print('=' * 72)
    print(f'{"Metric":<12} {label_a:>{col_w}} {label_b:>{col_w}} {"delta":>{col_w}}')
    print('-' * 72)
    for metric in metrics_a:
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]
        delta = val_b - val_a
        print(f'{metric:<12} {val_a:>{col_w}.4f} {val_b:>{col_w}.4f} {delta:>+{col_w}.4f}')

    ap_delta = metrics_b['AP'] - metrics_a['AP']
    print('-' * 72)
    print(f'Primary coco/AP delta ({label_b} - {label_a}): {ap_delta:+.4f}')


def compare_two_predictions_to_report(
    ann_file,
    prediction_paths,
    report_path,
    export_script_hint,
):
    """Compare exactly two prediction JSONs; print and overwrite report_path."""
    prediction_paths = [Path(p) for p in prediction_paths]
    if len(prediction_paths) != 2:
        raise ValueError('Exactly two --predictions paths are required')

    for path in prediction_paths:
        if not path.exists():
            raise FileNotFoundError(
                f'Predictions not found: {path}. Run {export_script_hint} first.')

    labels = column_labels_for_paths(prediction_paths)
    buffer = StringIO()
    with redirect_stdout(buffer):
        print(f'Annotation file: {ann_file}')
        for path in prediction_paths:
            print(f'{labels[path]} predictions: {path}')
        print()

        metrics = {}
        for path in prediction_paths:
            label = labels[path]
            print(f'Computing AP for {label} predictions...')
            with open(path) as f:
                preds = json.load(f)
            metrics[label] = compute_ap_from_predictions(ann_file, preds)
            print()

        baseline_path = prediction_paths[0]
        baseline_label = labels[baseline_path]
        variant_label = labels[prediction_paths[1]]

        print('=' * 96)
        print('COCO AP comparison (same metric as training save_best=coco/AP)')
        print('=' * 96)
        print_ap_comparison_table(
            metrics[baseline_label],
            {variant_label: metrics[variant_label]},
            baseline_label=baseline_label,
        )

    report = buffer.getvalue()
    sys.stdout.write(report)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f'Wrote report to {report_path}', file=sys.stderr)
