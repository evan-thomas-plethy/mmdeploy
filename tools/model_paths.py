"""Active model paths for conversion/validation pipelines. Update when switching checkpoints."""

from pathlib import Path

MODEL_NAME = (
    'rtmpose-m_merged_gbe_v9_gbe_app_vids_various_versions_sweep_'
    'merged_gbe_v9_app_vids_v4_lyingperson_full_infiniteform_seed4_n736_unfreeze_full_ld0.5_lr1e-3'
)

MMDEPLOY_WORK_DIR = f'work_dir/{MODEL_NAME}'

# Dataset dirname under mmpose/data/
VAL_DATA_DIRNAME = (
    'merged_gbe_v9_gbe_app_vids_v4_lyingperson_full_infiniteform_seed4_n736'
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MMPOSE_ROOT = REPO_ROOT.parent / 'mmpose'
VAL_DATA_ROOT = MMPOSE_ROOT / 'data' / VAL_DATA_DIRNAME

END2END_PT = REPO_ROOT / MMDEPLOY_WORK_DIR / 'end2end.pt'
ONNX_MODEL = REPO_ROOT / MMDEPLOY_WORK_DIR / f'{MODEL_NAME}.onnx'
FP32_MLMODEL = REPO_ROOT / f'{MODEL_NAME}.mlmodel'
INT8_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_int8.mlmodel'
SAVED_MODEL_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model'
SAVED_MODEL_WITH_SIG_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model_with_signature'
FP32_TFLITE = SAVED_MODEL_DIR / f'{MODEL_NAME}_float32.tflite'
INT8_TFLITE = REPO_ROOT / f'{MODEL_NAME}_int8.tflite'

ANN_FILE = VAL_DATA_ROOT / 'annotations/person_keypoints_val2017.json'
IMG_PREFIX = VAL_DATA_ROOT / 'val2017'
PREDICTIONS_DIR = REPO_ROOT / 'predictions'
FP32_PREDICTIONS = PREDICTIONS_DIR / f'{MODEL_NAME}_fp32.keypoints.json'
INT8_PREDICTIONS = PREDICTIONS_DIR / f'{MODEL_NAME}_int8.keypoints.json'
OVERLAYS_DIR = REPO_ROOT / 'overlays' / MODEL_NAME
