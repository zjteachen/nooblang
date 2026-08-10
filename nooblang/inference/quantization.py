import torch
from abc import ABC

"""
The output of some quantizer on some tensor.
Stores the quantized tensor and additional
information needed to dequantize.
"""

class Quantizer(ABC):
    def quantize(self, t: torch.Tensor):
        raise NotImplementedError

    def dequantize(self, t: torch.Tensor, dequant_data, dt: torch.dtype):
        raise NotImplementedError

class QuantizedTensor:
    def __init__(self, t: torch.Tensor, q: Quantizer):
        self.quantized_tensor, self.dequant_data = q.quantize(t)
        self.quantizer = q
        self.data_type = t.dtype

    def dequantize(self) -> torch.Tensor:
        """
        Pair quantized and dequant data and dequantize 
        """
        return self.quantizer.dequantize(self.quantized_tensor, self.dequant_data, self.data_type)

    def get_quantized_tensor(self) -> torch.Tensor:
        return self.quantized_tensor


# Implement RTN quantization.

class RTNQuantizer(Quantizer):
    def __init__(self, bits=4, pack_size=64):
        self.bits = bits
        self.pack_size = pack_size

    def quantize(self, t: torch.Tensor):
        # split tensor in chunks of pack_size
        chunks = torch.split(t, self.pack_size, dim=-1)
        starts = []
        leaps = []
        q_chunks = []
        for chunk in chunks:
            # get center and spread
            # get data type of tensor
            minval = chunk.min()
            maxval = chunk.max()
            leap = (maxval - minval) / ((1<<self.bits) - 1)

            # quantize tensor
            q_chunks.append(((chunk - minval) / leap).round())
            starts.append(minval)
            leaps.append(leap)

        # reshape q_chunks and reformat starts & leaps -> tensors
        quantized_tensor = torch.cat(q_chunks, dim=-1)
        starts_t = torch.stack(starts, dim=-1)
        leaps_t = torch.stack(leaps, dim=-1)
        return quantized_tensor, (starts_t, leaps_t)

    def dequantize(self, t: torch.Tensor, dequant_data, dt: torch.dtype):
        starts_t, leaps_t = dequant_data
        q_chunks = torch.split(t, self.pack_size, dim=-1)
        starts = torch.unbind(starts_t, dim=-1)
        leaps = torch.unbind(leaps_t, dim=-1)
        chunks = []
        for q_chunk, start, leap in zip(q_chunks, starts, leaps):
            chunks.append(start + (q_chunk * leap))
        return torch.cat(chunks, dim=-1).to(dt)



