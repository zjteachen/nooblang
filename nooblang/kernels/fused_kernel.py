import os

import torch
from torch.utils.cpp_extension import load

from nooblang.inference.quantization import QuantizedTensor

_CUDA_DIR = os.path.join(os.path.dirname(__file__), "cuda")

_ext = load(
    name="fused_dequant_matmul",
    sources=[os.path.join(_CUDA_DIR, "fused_dequant_matmul.cu")], 
    verbose=True,
)


def fused_matmul_quant_unquant(
    qt: QuantizedTensor, B: torch.Tensor, transpose_b: bool = False
) -> torch.Tensor:
    """
    Dequantize qt on the fly and multiply by B, fused, on GPU.
    qt wraps A's packed INT4 codes + (starts, leaps); see QuantizedTensor /
    RTNQuantizer in quantization.py for what each field holds and how they
    were produced.

    If transpose_b is True, B is passed in its native (N, K) layout (e.g. an
    activation tensor as-is, no .T needed) and the kernel reads it transposed
    internally - no materializing .T().contiguous() copy required.
    """
    starts, leaps = qt.get_dequant_data()
    A = qt.get_quantized_tensor()
    return _ext.fused_matmul_quant_unquant(
        A, B, starts, leaps, qt.quantizer.pack_size, transpose_b
    )
