import torch
import sys

INPUT_MODEL = "work_dir/rtmdet-nano_pretrained/end2end.pt"
OUTPUT_MODEL = "work_dir/rtmdet-nano_pretrained/rtmdet-nano_pretrained.onnx"

try:
    # Load the TorchScript model
    scripted_model = torch.jit.load(INPUT_MODEL)
    scripted_model.eval()  # set to evaluation mode

    # ENSURE TO CHANGE FOR RTMDET VS RTMPOSE
    dummy_input = torch.randn(1, 3, 320, 320)

    torch.onnx.export(
        scripted_model,              # TorchScript model
        dummy_input,                 # example input
        OUTPUT_MODEL,                # output ONNX file
        export_params=True,          # store trained weights
        opset_version=17,            # ONNX opset
        do_constant_folding=True,    # optimize constants
    )
    print(f"SUCCESS: Model converted and saved to '{OUTPUT_MODEL}'")

except FileNotFoundError as e:
    print(f"ERROR: Input file not found - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Conversion failed - {e}", file=sys.stderr)
    sys.exit(1)