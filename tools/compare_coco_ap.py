"""Compare COCO AP from exported keypoints prediction JSON files (runs on Linux/macOS)."""

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_paths import ANN_FILE, FP32_PREDICTIONS, INT8_PREDICTIONS
from pose_eval_common import compute_ap_from_predictions, print_ap_comparison


def load_predictions(path):
    with open(path) as f:
        return json.load(f)


def main():
    for path in (ANN_FILE, FP32_PREDICTIONS, INT8_PREDICTIONS):
        if not path.exists():
            print(f'ERROR: Required path not found: {path}', file=sys.stderr)
            sys.exit(1)

    print(f'Annotation file: {ANN_FILE}')
    print(f'fp32 predictions: {FP32_PREDICTIONS}')
    print(f'int8 predictions: {INT8_PREDICTIONS}')
    print()

    fp32_preds = load_predictions(FP32_PREDICTIONS)
    int8_preds = load_predictions(INT8_PREDICTIONS)

    print('Computing AP for fp32 predictions...')
    fp32_metrics = compute_ap_from_predictions(ANN_FILE, fp32_preds)
    print()

    print('Computing AP for int8 predictions...')
    int8_metrics = compute_ap_from_predictions(ANN_FILE, int8_preds)
    print()

    print_ap_comparison(fp32_metrics, int8_metrics, label_a='fp32', label_b='int8')


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: Comparison failed - {e}', file=sys.stderr)
        sys.exit(1)
