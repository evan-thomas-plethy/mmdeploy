import torch
import coremltools as ct

# Load your TorchScript model
model_path = "work_dir/rtmdet/end2end.pt"
traced_model = torch.jit.load(model_path)
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
            shape=(1, 3, 320, 320),
            scale=scale,
            bias=bias,
            color_layout=ct.colorlayout.BGR
        )
    ]
)

# Save the converted CoreML model
coreml_model.save("rtmdet-nano_pretrained.mlmodel")
print("Model successfully converted and saved as 'rtmdet-nano_pretrained.mlmodel'")