"""Compare COCO AP from two exported CoreML keypoints prediction JSON files."""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import ANN_FILE, AP_COREML_REPORT
from pose_eval_common import compare_two_predictions_to_report


def main():
    parser = argparse.ArgumentParser(
        description='Compare COCO AP from two exported CoreML prediction JSONs.')
    parser.add_argument(
        '--predictions',
        nargs=2,
        type=Path,
        metavar='PATH',
        required=True,
        help='Two prediction JSON paths (first = baseline).',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ANN_FILE,
        help='COCO val annotations JSON.',
    )
    args = parser.parse_args()

    compare_two_predictions_to_report(
        args.ann_file,
        args.predictions,
        AP_COREML_REPORT,
        'tools/coreml_export_val_predictions.py',
    )


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Comparison failed - {e}', file=sys.stderr)
        sys.exit(1)
