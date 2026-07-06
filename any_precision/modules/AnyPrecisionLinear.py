import torch
import torch.nn as nn
import os

try:
    import ap_gemv
except:
    ap_gemv = None

@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, output:torch.Tensor, bitwidth:int) -> None:
    ap_gemv.anyprec_gemv(x, output, q_weight, lut, bitwidth)

@anyprec_gemv.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None

class AnyPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None):
        super().__init__()
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        output_device = device if device is not None else 'cuda'
        self.output = torch.zeros((1, 1, self.out_features), dtype=torch.float16, device=output_device)
        self._dequant_weight_cache = {}

    def _should_cache_dequant_weight(self):
        value = os.environ.get("GUIDEDQUANT_CACHE_DEQUANT", "0").lower()
        return value in {"1", "true", "yes", "on"}

    def _dequant_cache_key(self, w_bits):
        device = self.qweight.device
        return (w_bits, device.type, device.index)

    def _dequant_weight_fallback(self, w_bits, use_cache=None):
        from any_precision.quantization.pack import unpack_single_weight

        if use_cache is None:
            use_cache = self._should_cache_dequant_weight()

        cache_key = self._dequant_cache_key(w_bits)
        if use_cache and cache_key in self._dequant_weight_cache:
            return self._dequant_weight_cache[cache_key]

        parent_precision = self.qweight.shape[0]
        indices = unpack_single_weight(self.qweight.detach().cpu(), parent_precision).squeeze(1)
        if parent_precision != w_bits:
            indices = indices >> (parent_precision - w_bits)

        indices = indices.to(self.qweight.device, non_blocking=True).long()
        lut = self._buffers[f'lut{w_bits}'].to(torch.float16)
        weight = torch.gather(lut, 1, indices).contiguous()

        if use_cache:
            self._dequant_weight_cache[cache_key] = weight

        return weight

    def cache_dequantized_weight(self, precision=None):
        if precision is None:
            precision = self.precision
        self._dequant_weight_fallback(precision, use_cache=True)

    def clear_dequantized_weight_cache(self):
        self._dequant_weight_cache.clear()

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')

    def forward(self, x, **kwargs):
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision

        if ap_gemv is None or (x.numel() // x.shape[-1] > 1 and self._should_cache_dequant_weight()):
            weight = self._dequant_weight_fallback(w_bits).to(x.dtype)
            x = torch.matmul(x, weight.T)
        elif x.numel() // x.shape[-1] > 1:
            weight = ap_gemv.anyprec_dequant(self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), w_bits).to(x.dtype)
            x = torch.matmul(x, weight.T)
        else:
            anyprec_gemv(x.to(torch.float16), self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), self.output, w_bits)
            x = self.output.to(x.dtype)

        if self.bias is not None:
            x += self.bias

        return x.clamp_(torch.finfo(x.dtype).min * (1.0 - 5e-3), torch.finfo(x.dtype).max * (1.0 - 5e-3))
        # return x

    def set_precision(self, precision):
        if precision not in self.precisions:
            raise RuntimeError(f"{self.precisions}-bit precisions are supported but {precision}-bit was specified.")

        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
