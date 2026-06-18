"""Compare COCO AP across exported models or precomputed prediction JSONs.

TFLite: runs inference via tflite_compare_ap.evaluate_model (predictions cached
under predictions/ unless --force-rerun). Pass any local .tflite paths.

CoreML / other: pass .keypoints.json paths (e.g. from coreml_export_val_predictions.py
on Mac). Paths can live anywhere after export — no meta.json or frontend-models layout.

Column labels are inferred from filenames (_float16 / _fp16 => fp16, _int8 => int8,
no precision suffix => fp32). Duplicate precision labels fall back to the file stem.
"""

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import ANN_FILE, IMG_PREFIX, PREDICTIONS_DIR
from pose_eval_common import (
    column_labels_for_paths,
    compute_ap_from_predictions,
    infer_precision_from_path,
    print_ap_comparison_table,
)
from tflite_compare_ap import (
    _setup_mmpose_imports,
    evaluate_model,
    tflite_predictions_path,
)

TFLITE_EXT = '.tflite'


def _comparison_title(paths, labels):
    parts = [f'{labels[p]} ({infer_precision_from_path(p)})' for p in paths]
    return ' vs '.join(parts)


def compare_tflite_models(model_paths, ann_file, img_prefix, max_samples=None,
                          force_rerun=False, predictions_dir=None):
    model_paths = [Path(p) for p in model_paths]
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f'Model not found: {path}')
        if path.suffix != TFLITE_EXT:
            raise ValueError(f'Expected {TFLITE_EXT} model, got: {path}')

    labels = column_labels_for_paths(model_paths)
    get_simcc_maximum, oks_nms, bbox_xyxy2cs, get_warp_matrix = _setup_mmpose_imports()
    eval_kwargs = dict(
        ann_file=ann_file,
        img_prefix=img_prefix,
        get_simcc_maximum=get_simcc_maximum,
        oks_nms=oks_nms,
        bbox_xyxy2cs=bbox_xyxy2cs,
        get_warp_matrix=get_warp_matrix,
        max_samples=max_samples,
        force_rerun=force_rerun,
    )

    print(f'Annotation file: {ann_file}')
    print(f'Image prefix: {img_prefix}')
    print()

    metrics = {}
    for path in model_paths:
        label = labels[path]
        pred_path = tflite_predictions_path(
            path, ann_file, max_samples=max_samples, predictions_dir=predictions_dir)
        print(f'Evaluating {label}: {path}')
        metrics[label] = evaluate_model(
            path, predictions_path=pred_path, **eval_kwargs)
        print()

    baseline_path = model_paths[0]
    baseline_label = labels[baseline_path]
    variant_metrics = {
        label: metrics[label]
        for path, label in labels.items()
        if path != baseline_path
    }

    print('=' * 96)
    print(f'TFLite AP: {_comparison_title(model_paths, labels)}')
    print('=' * 96)
    print_ap_comparison_table(
        metrics[baseline_label],
        variant_metrics,
        baseline_label=baseline_label,
    )


def compare_predictions(prediction_paths, ann_file):
    prediction_paths = [Path(p) for p in prediction_paths]
    for path in prediction_paths:
        if not path.exists():
            raise FileNotFoundError(f'Predictions not found: {path}')

    labels = column_labels_for_paths(prediction_paths)
    print(f'Annotation file: {ann_file}')
    for path in prediction_paths:
        print(f'{labels[path]} predictions: {path}')
    print()

    metrics = {}
    for path in prediction_paths:
        label = labels[path]
        with open(path) as f:
            preds = json.load(f)
        metrics[label] = compute_ap_from_predictions(ann_file, preds)

    baseline_path = prediction_paths[0]
    baseline_label = labels[baseline_path]
    variant_metrics = {
        label: metrics[label]
        for path, label in labels.items()
        if path != baseline_path
    }

    print('=' * 96)
    print(f'COCO AP: {_comparison_title(prediction_paths, labels)}')
    print('=' * 96)
    print_ap_comparison_table(
        metrics[baseline_label],
        variant_metrics,
        baseline_label=baseline_label,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Compare COCO AP across exported TFLite models or prediction JSONs.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--models',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='TFLite model paths (first = baseline). Any local path after export.',
    )
    group.add_argument(
        '--predictions',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='Pre-exported keypoints JSON paths (first = baseline).',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit val annotations (TFLite inference only).',
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
        help='Val images directory (TFLite inference only).',
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=PREDICTIONS_DIR,
        help='Directory for cached TFLite prediction JSONs.',
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run TFLite inference even if prediction JSON exists.',
    )
    args = parser.parse_args()

    if args.models:
        compare_tflite_models(
            args.models,
            ann_file=args.ann_file,
            img_prefix=args.img_prefix,
            max_samples=args.max_samples,
            force_rerun=args.force_rerun,
            predictions_dir=args.predictions_dir,
        )
    else:
        compare_predictions(args.predictions, ann_file=args.ann_file)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
