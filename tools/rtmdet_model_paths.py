"""Active RTMDet model paths for conversion pipelines. Update when switching checkpoints."""

from pathlib import Path

MODEL_NAME = 'rtmdet-nano_various_datasets_epoch40'
VAL_DATA_DIRNAME = 'merged_gbe_v10_gbe_app_vids_v5'

REPO_ROOT = Path(__file__).resolve().parent.parent
MMDETECTION_ROOT = REPO_ROOT.parent / 'mmdetection'

CHECKPOINT = (
    MMDETECTION_ROOT / 'work_dirs/rtmdet-nano_various_datasets/epoch_40.pth')
MMDET_CONFIG = (
    MMDETECTION_ROOT / 'configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py')
DEMO_IMAGE = (
    MMDETECTION_ROOT / 'data/merged_gbe_v10_gbe_app_vids_v5/val2017/'
    '90-90_Bridging_On_Ball_With_Hamstring_Curls_Large_Ball_frame_0000.jpg')

INPUT_HEIGHT = 320
INPUT_WIDTH = 320

MMDEPLOY_WORK_DIR = REPO_ROOT / f'work_dir/{MODEL_NAME}'
END2END_PT = MMDEPLOY_WORK_DIR / 'end2end.pt'
ONNX_MODEL = MMDEPLOY_WORK_DIR / f'{MODEL_NAME}.onnx'

FP32_MLMODEL = REPO_ROOT / f'{MODEL_NAME}.mlmodel'
FP16_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_float16.mlmodel'
INT8_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_int8.mlmodel'

SAVED_MODEL_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model'
SAVED_MODEL_WITH_SIG_DIR = (
    REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model_with_signature')
FP32_TFLITE = SAVED_MODEL_DIR / f'{MODEL_NAME}_float32.tflite'
FP16_TFLITE = SAVED_MODEL_DIR / f'{MODEL_NAME}_float16.tflite'
INT8_TFLITE = REPO_ROOT / f'{MODEL_NAME}_int8.tflite'

VAL_DATA_ROOT = MMDETECTION_ROOT / 'data' / VAL_DATA_DIRNAME
ANN_FILE = VAL_DATA_ROOT / 'annotations/person_keypoints_val2017.json'
IMG_PREFIX = VAL_DATA_ROOT / 'val2017'

PREDICTIONS_DIR = MMDETECTION_ROOT / 'predictions'
OVERLAYS_DIR = MMDETECTION_ROOT / 'overlays'
REPORTS_DIR = MMDETECTION_ROOT / 'reports'
AP_EXPORTED_REPORT = REPORTS_DIR / 'ap_exported.txt'
