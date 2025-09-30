_base_ = ['../../_base_/torchscript_config.py']

ir_config = dict(output_names=['dets', 'labels'])
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=None,
    )
