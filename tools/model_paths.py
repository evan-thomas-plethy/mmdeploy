"""Active model paths for conversion/validation pipelines. Update when switching checkpoints."""

from pathlib import Path

MODEL_NAME = (
    'rtmpose-m_merged_gbe_v9_gbe_app_vids_various_versions_sweep_'
    'merged_gbe_v9_app_vids_v4_lyingperson_full_infiniteform_seed4_n736_unfreeze_full_ld0.5_lr1e-3'
)
VAL_DATA_DIRNAME = 'merged_gbe_v9_gbe_app_vids_v4'

# Pretrained example:
# MODEL_NAME = 'rtmpose-m_pretrained'
# VAL_DATA_DIRNAME = 'merged_gbe_v9_gbe_app_vids_v4'

MMDEPLOY_WORK_DIR = f'work_dir/{MODEL_NAME}'

REPO_ROOT = Path(__file__).resolve().parent.parent
MMPOSE_ROOT = REPO_ROOT.parent / 'mmpose'
VAL_DATA_ROOT = MMPOSE_ROOT / 'data' / VAL_DATA_DIRNAME

# mmdeploy/mmpose/ — local copy used by deploy.py (not ../mmpose checkout)
MMPOSE_DEPLOY_ROOT = REPO_ROOT / 'mmpose'
CHECKPOINT = MMPOSE_DEPLOY_ROOT / 'checkpoints' / f'{MODEL_NAME}.pth'
MMPOSE_MODEL_CONFIG = MMPOSE_DEPLOY_ROOT / 'configs/rtmpose-m_8xb256-420e_aic-coco-256x192.py'
DEMO_IMAGE = MMPOSE_ROOT / 'tests/data/coco/000000197388.jpg'

END2END_PT = REPO_ROOT / MMDEPLOY_WORK_DIR / 'end2end.pt'
ONNX_MODEL = REPO_ROOT / MMDEPLOY_WORK_DIR / f'{MODEL_NAME}.onnx'
FP32_MLMODEL = REPO_ROOT / f'{MODEL_NAME}.mlmodel'
FP16_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_float16.mlmodel'
INT8_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_int8.mlmodel'
SAVED_MODEL_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model'
SAVED_MODEL_WITH_SIG_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model_with_signature'
FP32_TFLITE = SAVED_MODEL_DIR / f'{MODEL_NAME}_float32.tflite'
FP16_TFLITE = SAVED_MODEL_DIR / f'{MODEL_NAME}_float16.tflite'
INT8_TFLITE = REPO_ROOT / f'{MODEL_NAME}_int8.tflite'

PREDICTIONS_DIR = REPO_ROOT / 'predictions'
TFLITE_FP32_PREDICTIONS = (
    PREDICTIONS_DIR / f'tflite_{MODEL_NAME}_float32_{VAL_DATA_DIRNAME}.keypoints.json')
TFLITE_FP16_PREDICTIONS = (
    PREDICTIONS_DIR / f'tflite_{MODEL_NAME}_float16_{VAL_DATA_DIRNAME}.keypoints.json')
TFLITE_INT8_PREDICTIONS = (
    PREDICTIONS_DIR / f'tflite_{MODEL_NAME}_int8_{VAL_DATA_DIRNAME}.keypoints.json')

ANN_FILE = VAL_DATA_ROOT / 'annotations/person_keypoints_val2017.json'
IMG_PREFIX = VAL_DATA_ROOT / 'val2017'
FP32_PREDICTIONS = (
    PREDICTIONS_DIR / f'coreml_{MODEL_NAME}_{VAL_DATA_DIRNAME}.keypoints.json')
FP16_PREDICTIONS = (
    PREDICTIONS_DIR / f'coreml_{MODEL_NAME}_float16_{VAL_DATA_DIRNAME}.keypoints.json')
INT8_PREDICTIONS = (
    PREDICTIONS_DIR / f'coreml_{MODEL_NAME}_int8_{VAL_DATA_DIRNAME}.keypoints.json')
AP_COREML_REPORT = REPO_ROOT / 'reports' / 'ap_coreml.txt'
AP_TFLITE_REPORT = REPO_ROOT / 'reports' / 'ap_tflite.txt'
OVERLAYS_DIR = REPO_ROOT / 'overlays'
