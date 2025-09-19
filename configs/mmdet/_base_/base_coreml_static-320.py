_base_ = ['./base_torchscript.py', '../../_base_/backends/coreml.py']

ir_config = dict(input_shape=(320, 320))
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 320, 320],
                max_shape=[1, 3, 320, 320],
                default_shape=[1, 3, 320, 320])))
])
