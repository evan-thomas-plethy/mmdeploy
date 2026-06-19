"""Compare COCO AP across precomputed prediction JSONs.

Pass keypoints JSON paths from tflite_export_val_predictions.py or
coreml_export_val_predictions.py. Column labels are inferred from filenames.
"""

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import ANN_FILE
from pose_eval_common import (
    column_labels_for_paths,
    compute_ap_from_predictions,
    infer_precision_from_path,
    print_ap_comparison_table,
)


def _comparison_title(paths, labels):
    parts = [f'{labels[p]} ({infer_precision_from_path(p)})' for p in paths]
    return ' vs '.join(parts)


def compare_predictions(prediction_paths, ann_file):
    prediction_paths = [Path(p) for p in prediction_paths]
    for path in prediction_paths:
        if not path.exists():
            raise FileNotFoundError(
                f'Predictions not found: {path}. '
                'Run tflite_export_val_predictions.py or coreml_export_val_predictions.py first.')

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
        description='Compare COCO AP across precomputed prediction JSONs.')
    parser.add_argument(
        '--predictions',
        nargs='+',
        type=Path,
        metavar='PATH',
        required=True,
        help='Keypoints JSON paths (first = baseline).',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ANN_FILE,
        help='COCO val annotations JSON.',
    )
    args = parser.parse_args()

    compare_predictions(args.predictions, ann_file=args.ann_file)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
