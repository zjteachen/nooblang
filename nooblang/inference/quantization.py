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
"""
Current problem: 
CUDA kernel assumes ceil(|t|/pack_size) different values for the dequant data (start, leap).
This code splits each chunk along the -1 dim, so the -1 dim is split into pack size 16.
This means that each chunk takes the entire slice across the rest of the dimensions,
so the chunk size is not what is expected.
"""
class RTNQuantizer(Quantizer):
    def __init__(self, bits=4, pack_size=64):
        self.bits = bits
        self.pack_size = pack_size

    def quantize(self, t: torch.Tensor):
        """
        In this quantizer, we assign every "chunk" of pack_size values 
        1 pair of quant/dequant values.
        """
        chunk_batches = torch.split(t, self.pack_size, dim=-1)
        starts = []
        all_leaps = []
        q_chunks = []

        for chunk_batch in chunk_batches:
            # get center and spread
            # get data type of tensor
            minvals = chunk_batch.min(dim=-1).values
            maxvals = chunk_batch.max(dim=-1).values
            leaps = (maxvals - minvals) / ((1<<self.bits) - 1)

            # quantize tensor
            adj = lambda a: a.unsqueeze(dim=-1) # match dequant value dim with chunk_batch
            q_chunks.append(((chunk_batch - adj(minvals)) / adj(leaps)).round())
            starts.append(minvals)
            all_leaps.append(leaps)
        
        # result:
        # q_chunks is a list of tensors of shape (..., t.dim(-1)/pack_size), except
        # for the last one which may have a smaller last dim.
        # leaps and starts shape are 1 less dimension than t. |leaps| = ceil(t.dim(-1) / pack_size)
        # reshape q_chunks and reformat starts & leaps -> tensors
        quantized_tensor = torch.cat(q_chunks, dim=-1)
        starts_t = torch.stack(starts, dim=-1)
        leaps_t = torch.stack(all_leaps, dim=-1)
        packed_tensor = self._pack(quantized_tensor)
        return packed_tensor, (starts_t, leaps_t)

    def dequantize(self, t: torch.Tensor, dequant_data, dt: torch.dtype):
        starts_t, leaps_t = dequant_data
        codes = self._unpack(t)
        q_chunks = torch.split(codes, self.pack_size, dim=-1)
        starts = torch.unbind(starts_t, dim=-1)
        leaps = torch.unbind(leaps_t, dim=-1)
        chunks = []
        for q_chunk, start, leap in zip(q_chunks, starts, leaps):
            chunks.append(start.unsqueeze(dim=-1) + (q_chunk * leap.unsqueeze(dim=-1)))
        return torch.cat(chunks, dim=-1).to(dt)

    def _pack(self, codes: torch.Tensor) -> torch.Tensor:
        """Pack values_per_byte codes (0..2**bits-1) into each uint8 along the last dim."""
        values_per_byte = 8 // self.bits
        codes = codes.to(torch.uint8)
        grouped = codes.reshape(*codes.shape[:-1], -1, values_per_byte)
        shifts = (torch.arange(values_per_byte, device=codes.device) * self.bits).to(torch.uint8)
        return (grouped << shifts).sum(dim=-1, dtype=torch.int64).to(torch.uint8)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Inverse of _pack: expand each uint8 back into values_per_byte codes."""
        values_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        shifts = (torch.arange(values_per_byte, device=packed.device) * self.bits).to(torch.uint8)
        expanded = (packed.unsqueeze(-1) >> shifts) & mask
        return expanded.reshape(*packed.shape[:-1], -1).to(torch.float32)



