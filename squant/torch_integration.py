import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .codec import Compressor, decompress


class SQuantLinear(nn.Module):
    def __init__(self, tensor, bias=None, cache_decoded=False, dtype=torch.float32):
        super().__init__()
        tensor.validate()
        if len(tensor.shape) != 2:
            raise ValueError("Linear weight must be 2D")
        self.out_features, self.in_features = tensor.shape
        self.config = tensor.config
        self.cache_decoded = cache_decoded
        self.register_buffer("seeds", torch.from_numpy(tensor.seeds.astype(np.int64)))
        self.register_buffer("counts", torch.from_numpy(tensor.counts.astype(np.int64)))
        self.register_buffer("offsets", torch.from_numpy(np.r_[0, np.cumsum(tensor.counts, dtype=np.int64)]))
        self.register_buffer("coefficients", torch.from_numpy(tensor.coefficients.copy()))
        self.register_buffer("scales", torch.from_numpy(tensor.scales.copy()))
        # A nonpersistent anchor follows .to(dtype/device); integer storage stays integer.
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype), persistent=False)
        self.register_buffer("_decoded", None, persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach().clone().to(dtype))

    def _apply(self, fn, recurse=True):
        self._decoded = None
        # Module.bfloat16() must not round the stored FP16 quantizer scales.
        original_scales = self.scales
        result = super()._apply(fn, recurse)
        self.scales = original_scales.to(device=self._anchor.device, dtype=torch.float16)
        return result

    @torch.no_grad()
    def decode_weight(self):
        if self.cache_decoded and self._decoded is not None:
            return self._decoded
        c = self.config
        n = len(self.seeds)
        chunks = []
        mid = 2**(c.seed_bits-1)
        for start in range(0, n, c.block_chunk):
            stop = min(n, start+c.block_chunk)
            state = self.seeds[start:stop].clone()
            counts = self.counts[start:stop]
            block_ids = torch.arange(start, stop, device=state.device)
            kmax = int(counts.max().item())
            out = torch.zeros((stop-start, c.block_size), device=state.device, dtype=torch.float32)
            for k in range(kmax):
                valid = counts > k
                indices = (self.offsets[start:stop]+k).clamp(max=len(self.coefficients)-1)
                coeff = self.coefficients[indices].float()*self.scales[block_ids//c.group_size].float()*valid
                for b in range(c.block_size):
                    out[:, b] += ((state.float()-mid)/(mid-1))*coeff
                    state = (state >> 1) ^ ((state & 1)*c.mask)
            chunks.append(out.flatten())
        weight = torch.cat(chunks)[:self.out_features*self.in_features].reshape(self.out_features, self.in_features).to(self._anchor.dtype)
        if self.cache_decoded:
            self._decoded = weight
        return weight

    def forward(self, x):
        return F.linear(x, self.decode_weight(), self.bias)


@torch.no_grad()
def quantize_model(model, config, include=None, exclude=("lm_head",), cache_decoded=False,
                   materialize=False, progress=None):
    """Replace torch.nn.Linear weights; keep embeddings, norms, and lm_head.

    materialize=True makes a dense reconstructed evaluation model. Neither mode
    claims compressed GPU acceleration. Search never sees activations or tokens.
    """
    import re
    compressor = Compressor(config)
    report = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or any(x in name for x in exclude):
            continue
        if include is not None and re.search(include, name) is None:
            continue
        tensor = compressor.compress(module.weight.detach().float().cpu().numpy())
        report[name] = tensor.metrics
        if materialize:
            module.weight.copy_(torch.from_numpy(decompress(tensor)).to(module.weight))
        else:
            replacement = SQuantLinear(tensor, module.bias, cache_decoded, module.weight.dtype).to(module.weight.device)
            if not name:
                raise ValueError("Wrap a root Linear in nn.Sequential before quantize_model")
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, replacement)
        if progress:
            progress(name, tensor.metrics)
    if not report:
        raise ValueError("No eligible Linear layers matched")
    return report
