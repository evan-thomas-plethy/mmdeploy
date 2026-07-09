"""Active RTMDet model paths for conversion/validation pipelines.

Update when switching checkpoints. RTMPose paths live in model_paths.py.
"""

from pathlib import Path

# --- Active: rtmdet-nano_pretrained ---
MODEL_NAME = 'rtmdet-nano_pretrained'
VAL_DATA_DIRNAME = 'merged_gbe_v9_gbe_app_vids_v4_lyingperson_full_infiniteform_seed4_n736'

# Finetuned gbe_v9 best checkpoint (switch MODEL_NAME / CHECKPOINT / VAL_DATA_DIRNAME):
# MODEL_NAME = (
#     'rtmdet-nano_various_datasets_merged_gbe_v9_gbe_app_vids_v4_'
#     'lyingperson_full_infiniteform_seed4_n736'
# )
# VAL_DATA_DIRNAME = (
#     'merged_gbe_v9_gbe_app_vids_v4_lyingperson_full_infiniteform_seed4_n736'
# )
# CHECKPOINT = (
#     MMDETECTION_ROOT
#     / 'work_dirs/rtmdet-nano_various_datasets/best_coco_bbox_mAP_epoch_49.pth'
# )
# DEMO_IMAGE = (
#     MMDETECTION_ROOT / 'data' / VAL_DATA_DIRNAME / 'val2017'
#     / '90-90_Bridging_On_Ball_With_Hamstring_Curls_Large_Ball_frame_0000.jpg'
# )

REPO_ROOT = Path(__file__).resolve().parent.parent
MMDETECTION_ROOT = REPO_ROOT.parent / 'mmdetection'

CHECKPOINT = REPO_ROOT / 'mmdet/checkpoints/rtmdet-nano_pretrained.pth'
MMDET_CONFIG = REPO_ROOT / 'mmdet/configs/rtmdet_nano_320_8xb32_coco_finetune.py'
DEMO_IMAGE = MMDETECTION_ROOT / 'demo/demo.jpg'

INPUT_HEIGHT = 320
INPUT_WIDTH = 320
TFLITE_INPUT_SHAPE = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)

MMDEPLOY_WORK_DIR = REPO_ROOT / f'work_dir/{MODEL_NAME}'
END2END_PT = MMDEPLOY_WORK_DIR / 'end2end.pt'
ONNX_MODEL = MMDEPLOY_WORK_DIR / f'{MODEL_NAME}.onnx'

FP32_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_float32.mlmodel'
FP16_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_float16.mlmodel'
INT8_MLMODEL = REPO_ROOT / f'{MODEL_NAME}_int8.mlmodel'

SAVED_MODEL_DIR = REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model'
SAVED_MODEL_WITH_SIG_DIR = (
    REPO_ROOT / 'saved_models' / f'{MODEL_NAME}_saved_model_with_signature'
)
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

# frontend-models (shipped to app via S3)
FRONTEND_MODEL_DIRNAME = 'rtmdet-nano_ground_based_exercises'
FRONTEND_MODELS_ROOT = REPO_ROOT / 'frontend-models'
FRONTEND_ANDROID_DIR = (
    FRONTEND_MODELS_ROOT / 'Android' / FRONTEND_MODEL_DIRNAME
)
FRONTEND_IOS_DIR = FRONTEND_MODELS_ROOT / 'iOS' / FRONTEND_MODEL_DIRNAME
FRONTEND_ANDROID_META = FRONTEND_ANDROID_DIR / 'meta.json'
FRONTEND_IOS_META = FRONTEND_IOS_DIR / 'meta.json'
FRONTEND_ANDROID_V1_TFLITE = (
    FRONTEND_ANDROID_DIR / 'v1' / f'{FRONTEND_MODEL_DIRNAME}.v1.tflite'
)
FRONTEND_IOS_V1_MLMODEL = (
    FRONTEND_IOS_DIR / 'v1' / f'{FRONTEND_MODEL_DIRNAME}.v1.mlmodel'
)
# meta.json v1 value: export stem for the fp16 artifact
FRONTEND_META_V1_STEM = f'{MODEL_NAME}_float16'
