import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import torch

from rtmdet_model_paths import END2END_PT, INPUT_HEIGHT, INPUT_WIDTH, ONNX_MODEL

try:
    scripted_model = torch.jit.load(str(END2END_PT))
    scripted_model.eval()

    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)

    torch.onnx.export(
        scripted_model,
        dummy_input,
        str(ONNX_MODEL),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"SUCCESS: Model converted and saved to '{ONNX_MODEL}'")

except FileNotFoundError as e:
    print(f"ERROR: Input file not found - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Conversion failed - {e}", file=sys.stderr)
    sys.exit(1)
