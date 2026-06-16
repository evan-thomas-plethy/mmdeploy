import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from model_paths import FP32_MLMODEL, INT8_MLMODEL

NBITS = 8

try:
    model_fp32 = ct.models.MLModel(str(FP32_MLMODEL))
    model_quantized = quantization_utils.quantize_weights(model_fp32, nbits=NBITS)
    model_quantized.save(str(INT8_MLMODEL))
    print(f"SUCCESS: Quantized model saved to '{INT8_MLMODEL}' (nbits={NBITS})")
except FileNotFoundError as e:
    print(f"ERROR: Input file not found - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Quantization failed - {e}", file=sys.stderr)
    sys.exit(1)
