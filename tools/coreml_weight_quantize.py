import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from model_paths import FP32_MLMODEL

# Weight-only quantization from fp32 neuralnetwork MLModel. ImageType pre/post
# processing is unchanged (same pipeline as fp32 and int8).
VARIANT_SPECS = {
    'fp16': (16, 'fp16'),
    'int8': (8, 'int8'),
}


def _model_base_stem(fp32_path):
    """Strip _float32 from fp32 artifact stem so fp16/int8 names stay parallel."""
    stem = Path(fp32_path).stem
    if stem.endswith('_float32'):
        return stem[: -len('_float32')]
    return stem


def _output_path(fp32_path, variant):
    fp32_path = Path(fp32_path)
    base = _model_base_stem(fp32_path)
    suffix = fp32_path.suffix
    if variant == 'fp16':
        return fp32_path.with_name(f'{base}_float16{suffix}')
    return fp32_path.with_name(f'{base}_int8{suffix}')


def main():
    parser = argparse.ArgumentParser(
        description='Weight-quantize a fp32 CoreML model to fp16 and/or int8.')
    parser.add_argument(
        '--fp32',
        type=Path,
        default=FP32_MLMODEL,
        help='Input fp32 .mlmodel (default: model_paths.FP32_MLMODEL).',
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=tuple(VARIANT_SPECS),
        default=('fp16', 'int8'),
        help='Which quantized models to write (default: fp16 int8).',
    )
    parser.add_argument(
        '--fp16',
        type=Path,
        default=None,
        help='Output fp16 path (default: {fp32_stem}_float16.mlmodel).',
    )
    parser.add_argument(
        '--int8',
        type=Path,
        default=None,
        help='Output int8 path (default: {fp32_stem}_int8.mlmodel).',
    )
    args = parser.parse_args()

    fp32_path = Path(args.fp32)
    output_paths = {
        'fp16': Path(args.fp16) if args.fp16 else _output_path(fp32_path, 'fp16'),
        'int8': Path(args.int8) if args.int8 else _output_path(fp32_path, 'int8'),
    }

    if not fp32_path.exists():
        print(f'ERROR: Input file not found: {fp32_path}', file=sys.stderr)
        sys.exit(1)

    try:
        model_fp32 = ct.models.MLModel(str(fp32_path))
        for variant in args.variants:
            nbits, label = VARIANT_SPECS[variant]
            output_path = output_paths[variant]
            model_quantized = quantization_utils.quantize_weights(model_fp32, nbits=nbits)
            model_quantized.save(str(output_path))
            print(f'SUCCESS: {label} model saved to {output_path!r} (nbits={nbits})')
    except Exception as e:
        print(f'ERROR: Quantization failed - {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
