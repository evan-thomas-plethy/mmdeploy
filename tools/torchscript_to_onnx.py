import torch

# Load the TorchScript model
scripted_model = torch.jit.load("work_dir/rtmpose-m/end2end.pt")
scripted_model.eval()  # set to evaluation mode

# Example for an RTMPose / YOLO pose model
dummy_input = torch.randn(1, 3, 256, 192)

torch.onnx.export(
    scripted_model,              # TorchScript model
    dummy_input,                 # example input
    "work_dir/rtmpose-m/rtmpose-m.onnx",                # output ONNX file
    export_params=True,          # store trained weights
    opset_version=17,            # ONNX opset
    do_constant_folding=True,    # optimize constants
)