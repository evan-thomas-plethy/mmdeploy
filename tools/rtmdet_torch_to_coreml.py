import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import coremltools as ct
import torch

from rtmdet_model_paths import END2END_PT, FP32_MLMODEL, INPUT_HEIGHT, INPUT_WIDTH

traced_model = torch.jit.load(str(END2END_PT))
traced_model.eval()

# # RTMDet RGB preprocessing
# mean = [123.675, 116.28, 103.53]
# std = [58.395, 57.12, 57.375]

# RTMDet BGR preprocessing
mean=[103.53, 116.28, 123.675]
std=[57.375, 57.12, 58.395]

global_std = sum(std)/len(std)        
scale = 1.0 / global_std                

bias = [- m / s for m, s in zip(mean, std)]

coreml_model = ct.convert(
    traced_model,
    convert_to="neuralnetwork",
    inputs=[
        ct.ImageType(
            shape=(1, 3, INPUT_HEIGHT, INPUT_WIDTH),
            scale=scale,
            bias=bias,
            color_layout=ct.colorlayout.BGR
        )
    ]
)

# Save the converted CoreML model
coreml_model.save(str(FP32_MLMODEL))
print(f"Model successfully converted and saved as '{FP32_MLMODEL}'")