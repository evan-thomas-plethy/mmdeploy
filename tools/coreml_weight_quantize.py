import sys

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# Must match MODEL_NAME in rtmpose_torch_to_coreml.py
INPUT_MODEL = "rtmpose-m_merged_gbe_v6_various_datasets_sweep_lyingperson_unfreeze_full_ld0.5.mlmodel"
OUTPUT_MODEL = "rtmpose-m_merged_gbe_v6_various_datasets_sweep_lyingperson_unfreeze_full_ld0.5_int8.mlmodel"
NBITS = 8

try:
    model_fp32 = ct.models.MLModel(INPUT_MODEL)
    model_quantized = quantization_utils.quantize_weights(model_fp32, nbits=NBITS)
    model_quantized.save(OUTPUT_MODEL)
    print(f"SUCCESS: Quantized model saved to '{OUTPUT_MODEL}' (nbits={NBITS})")
except FileNotFoundError as e:
    print(f"ERROR: Input file not found - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Quantization failed - {e}", file=sys.stderr)
    sys.exit(1)
