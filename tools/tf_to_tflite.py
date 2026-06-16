import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tensorflow as tf

from model_paths import INT8_TFLITE, SAVED_MODEL_DIR, SAVED_MODEL_WITH_SIG_DIR

try:
    model_path = str(SAVED_MODEL_DIR)
    model = tf.saved_model.load(model_path)

    if 'serving_default' not in model.signatures:
        print("No serving_default signature found. Adding one...")

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 256, 192], dtype=tf.float32)])
        def serving_fn(input_tensor):
            return model(input_tensor)

        tf.saved_model.save(
            model, str(SAVED_MODEL_WITH_SIG_DIR),
            signatures={'serving_default': serving_fn})
        model_path = str(SAVED_MODEL_WITH_SIG_DIR)
    else:
        print("Model already has serving_default signature. Using existing model.")

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec = tf.lite.TargetSpec(experimental_supported_backends="GPU")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model_quant = converter.convert()

    with open(INT8_TFLITE, "wb") as f:
        f.write(tflite_model_quant)
    print(f"SUCCESS: TFLite conversion finished; wrote {INT8_TFLITE}")
except FileNotFoundError as e:
    print(f"ERROR: SavedModel or output path not found: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: TFLite conversion failed - {e}", file=sys.stderr)
    sys.exit(1)
