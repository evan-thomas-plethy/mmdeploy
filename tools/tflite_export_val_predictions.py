"""Export COCO val predictions from TFLite models (instance / mmpose env)."""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import PREDICTIONS_DIR
from pose_eval_common import predictions_output_path
from tflite_val_inference import export_tflite_predictions


def main():
    parser = argparse.ArgumentParser(
        description='Export COCO val predictions from TFLite models.')
    parser.add_argument(
        '--models',
        nargs='+',
        type=Path,
        metavar='PATH',
        required=True,
        help='TFLite model paths.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        help='COCO val annotations JSON.',
    )
    parser.add_argument(
        '--img-prefix',
        type=Path,
        required=True,
        help='Val images directory.',
    )
    parser.add_argument(
        '--dataset-name',
        default=None,
        help='Dataset tag for output filename (default: parent of annotations/).',
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=PREDICTIONS_DIR,
        help='Parent directory for prediction JSONs (default: predictions/).',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit val annotations.',
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run inference even if prediction JSON exists.',
    )
    args = parser.parse_args()

    if not args.ann_file.exists():
        print(f'ERROR: Annotations not found: {args.ann_file}', file=sys.stderr)
        sys.exit(1)
    if not args.img_prefix.exists():
        print(f'ERROR: Image prefix not found: {args.img_prefix}', file=sys.stderr)
        sys.exit(1)

    if args.max_samples is not None:
        print(f'Running on first {args.max_samples} annotations only')

    for model_path in args.models:
        if not model_path.exists():
            print(f'ERROR: Model not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        output_path = predictions_output_path(
            'tflite',
            model_path,
            args.ann_file,
            predictions_dir=args.predictions_dir,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
        )
        print(f'Exporting predictions for {model_path.name} -> {output_path}')
        export_tflite_predictions(
            model_path,
            args.ann_file,
            args.img_prefix,
            output_path,
            max_samples=args.max_samples,
            force_rerun=args.force_rerun,
        )


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Export failed - {e}', file=sys.stderr)
        sys.exit(1)
