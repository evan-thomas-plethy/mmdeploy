import tensorflow as tf
import numpy as np

model_path = "saved_models/rtmdet-nano_pretrained_saved_model"

# Load the model
model = tf.saved_model.load(model_path)

# Check if model already has a serving_default signature
if 'serving_default' not in model.signatures:
    print("No serving_default signature found. Adding one...")
    # Define a serving function with an input signature for RTMPose (256x192 input)
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 320, 320], dtype=tf.float32)])
    def serving_fn(input_tensor):
        return model(input_tensor)

    # Save the model with the serving function and signature
    saved_model_path = "saved_models/rtmdet-nano_pretrained_saved_model_with_signature"
    tf.saved_model.save(model, saved_model_path, signatures={'serving_default': serving_fn})
    model_path = saved_model_path  # Use the new model path for conversion
else:
    print("Model already has serving_default signature. Using existing model.")

# def representative_data_gen():
#     rep_data_dir = "representative_data/"
#     for filename in os.listdir(rep_data_dir):
#         filepath = os.path.join(rep_data_dir, filename)
#         try:
#             image = cv2.imread(filepath)
#             if image is None:
#                 print(f"Could not load file: {filepath}")
#                 continue
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = cv2.resize(image, (640, 640))
#             image_array = image.astype(np.float32) / 255.0
#             image_array = np.expand_dims(image_array, axis=0)
#             yield [image_array]
#         except Exception as e:
#             print(f"Error processing file {filepath}: {e}")


converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec = tf.lite.TargetSpec(experimental_supported_backends="GPU")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# converter.inference_input_type = tf.float32
# converter.inference_output_type = tf.float32

# converter.experimental_default_to_static_shapes = True
# converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()

# Save the TFLite model
with open("rtmdet-nano_pretrained_int8.tflite", "wb") as f:
    f.write(tflite_model_quant)