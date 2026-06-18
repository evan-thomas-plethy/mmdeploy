"""Compare COCO AP from exported keypoints prediction JSON files (runs on Linux/macOS)."""

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import (
    ANN_FILE,
    FP16_PREDICTIONS,
    FP32_PREDICTIONS,
    INT8_PREDICTIONS,
)
from pose_eval_common import (
    column_labels_for_paths,
    compute_ap_from_predictions,
    print_ap_comparison_table,
)


def load_predictions(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Compare COCO AP from exported keypoints prediction JSONs.')
    parser.add_argument(
        '--predictions',
        nargs='+',
        type=Path,
        metavar='PATH',
        help='Prediction JSON paths (first = baseline). Defaults to model_paths fp32/fp16/int8.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ANN_FILE,
        help='COCO val annotations JSON.',
    )
    args = parser.parse_args()

    prediction_paths = (
        list(args.predictions)
        if args.predictions
        else [FP32_PREDICTIONS, FP16_PREDICTIONS, INT8_PREDICTIONS]
    )
    for path in prediction_paths:
        if not path.exists():
            print(f'ERROR: Required path not found: {path}', file=sys.stderr)
            sys.exit(1)

    labels = column_labels_for_paths(prediction_paths)
    print(f'Annotation file: {args.ann_file}')
    for path in prediction_paths:
        print(f'{labels[path]} predictions: {path}')
    print()

    metrics = {}
    for path in prediction_paths:
        label = labels[path]
        print(f'Computing AP for {label} predictions...')
        metrics[label] = compute_ap_from_predictions(
            args.ann_file, load_predictions(path))
        print()

    baseline_path = prediction_paths[0]
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
        print(f'ERROR: Comparison failed - {e}', file=sys.stderr)
        sys.exit(1)
