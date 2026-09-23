from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import warnings
import zipfile
import numpy as np
from .config import QuantConfig
from .lfsr import bases, candidate_seeds


@dataclass
class CompressedTensor:
    shape: tuple
    config: QuantConfig
    seeds: np.ndarray
    counts: np.ndarray
    coefficients: np.ndarray  # concatenated, no per-block zero padding
    scales: np.ndarray  # FP16, one per G blocks
    metrics: dict = field(default_factory=dict)

    @property
    def numel(self):
        return math.prod(self.shape)

    @property
    def count_bits(self):
        return self.config.max_bases.bit_length()

    def storage(self):
        n = len(self.seeds)
        seed_bits = n * self.config.seed_bits
        coeff_bits = self.coefficients.size * 8
        scale_bits = self.scales.size * 16
        count_bits = n * self.count_bits
        payload_bytes = (seed_bits + 7) // 8 + (count_bits + 7) // 8 + self.coefficients.nbytes + self.scales.nbytes
        return {"num_weights": self.numel, "num_blocks": n,
                "mean_bases": float(self.counts.mean()),
                "paper_bits_per_weight": (seed_bits + coeff_bits + scale_bits) / self.numel,
                "logical_bits_per_weight": (seed_bits + coeff_bits + scale_bits + count_bits) / self.numel,
                "payload_bits_per_weight": payload_bytes * 8 / self.numel,
                "payload_bytes": payload_bytes,
                "seed_search": "exhaustive" if self.config.seed_budget is None or self.config.seed_budget == 2**self.config.seed_bits-1 else "sampled"}

    def validate(self):
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("Invalid shape")
        n = math.ceil(self.numel / self.config.block_size)
        if self.seeds.shape != (n,) or self.counts.shape != (n,):
            raise ValueError("Block count does not match shape")
        if np.any(self.seeds < 1) or np.any(self.seeds >= 2**self.config.seed_bits):
            raise ValueError("Invalid seeds")
        if np.any(self.counts < self.config.min_bases) or np.any(self.counts > self.config.max_bases):
            raise ValueError("Invalid basis counts")
        if self.coefficients.dtype != np.int8 or self.coefficients.shape != (int(self.counts.sum()),):
            raise ValueError("Invalid coefficient buffer")
        if self.scales.dtype != np.float16 or self.scales.shape != (math.ceil(n / self.config.group_size),):
            raise ValueError("Invalid scale buffer")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("Scales must be positive and finite")


class Compressor:
    """Reusable bounded projector cache; use one instance across model layers."""
    def __init__(self, config):
        self.config = config
        self.seeds = candidate_seeds(config)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.torch = None
        if config.device != "numpy":
            import torch
            if config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.torch = torch

    def _projector(self, k, start):
        key = (k, start)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        seed = self.seeds[start:start+self.config.seed_chunk]
        u = bases(seed, self.config, k)
        if self.torch is None:
            inverse = np.linalg.pinv(u, rcond=1e-10)
            size = u.nbytes + inverse.nbytes
        else:
            t = self.torch
            u = t.as_tensor(u, dtype=t.float64, device=self.config.device)
            inverse = t.linalg.pinv(u, rtol=1e-10)
            size = (u.numel() + inverse.numel()) * u.element_size()
        value = (u, inverse)
        budget = self.config.cache_mb * 1024**2
        if size <= budget:
            while self.cache and self.cache_bytes + size > budget:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.cache_bytes -= old_size
            self.cache[key] = (value, size)
            self.cache_bytes += size
        return value

    def _best(self, blocks, k):
        c = self.config
        n = len(blocks)
        errors = np.full(n, np.inf)
        best_seeds = np.ones(n, np.uint32)
        best_coeff = np.zeros((n, k))
        t = self.torch
        w = blocks if t is None else t.as_tensor(blocks, dtype=t.float64, device=c.device)
        for start in range(0, len(self.seeds), c.seed_chunk):
            u, inverse = self._projector(k, start)
            if t is None:
                a = np.einsum("skb,nb->snk", inverse, w, optimize=True)
                residual = np.einsum("sbk,snk->snb", u, a, optimize=True) - w[None]
                err = np.einsum("snb,snb->sn", residual, residual)
                idx = err.argmin(axis=0)
                e = err[idx, np.arange(n)]
                coeff = a[idx, np.arange(n)]
            else:
                a = t.einsum("skb,nb->snk", inverse, w)
                residual = t.einsum("sbk,snk->snb", u, a) - w[None]
                err = (residual * residual).sum(-1)
                e, idx_t = err.min(dim=0)
                coeff = a[idx_t, t.arange(n, device=c.device)].cpu().numpy()
                e, idx = e.cpu().numpy(), idx_t.cpu().numpy()
            better = e < errors
            errors[better] = e[better]
            best_seeds[better] = self.seeds[start + idx[better]]
            best_coeff[better] = coeff[better]
        return errors, best_seeds, best_coeff

    def compress(self, weight):
        c = self.config
        original = np.asarray(weight)
        if original.size == 0 or original.ndim == 0 or not np.issubdtype(original.dtype, np.floating):
            raise ValueError("Weights must be a nonempty floating-point tensor")
        if not np.isfinite(original).all():
            raise ValueError("Weights must be finite")
        n = math.ceil(original.size / c.block_size)
        flat = np.pad(original.astype(np.float64).ravel(), (0, n*c.block_size-original.size))
        blocks = flat.reshape(n, c.block_size)
        counts = np.zeros(n, np.uint16)
        seeds = np.ones(n, np.uint32)
        all_coeff = np.zeros((n, c.max_bases), np.float64)
        pre_error = np.zeros(n)
        energy = np.einsum("nb,nb->n", blocks, blocks)
        ks = [c.fixed_bases] if c.fixed_bases is not None else range(c.min_bases, c.max_bases+1)
        for first in range(0, n, c.block_chunk):
            active = np.arange(first, min(n, first+c.block_chunk))
            for k in ks:
                err, seed, coeff = self._best(blocks[active], k)
                ratio = np.ones(len(active))
                nonzero = energy[active] > 0
                ratio[nonzero] -= err[nonzero] / energy[active][nonzero]
                accept = (ratio >= c.energy_threshold - 1e-10) | (k == c.max_bases) | (c.fixed_bases is not None)
                chosen = active[accept]
                seeds[chosen], counts[chosen] = seed[accept], k
                all_coeff[chosen, :k] = coeff[accept]
                pre_error[chosen] = err[accept]
                active = active[~accept]
                if not len(active):
                    break
        pre_ratio = np.ones(n)
        np.divide(pre_error, energy, out=pre_ratio, where=energy > 0)
        pre_ratio = np.where(energy > 0, 1-pre_ratio, 1)
        failed = int(np.count_nonzero(pre_ratio < c.energy_threshold - 1e-9))
        if failed and c.fixed_bases is None:
            message = f"{failed}/{n} blocks missed pre-quantization threshold at max_bases={c.max_bases}"
            if c.strict_threshold:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
        q = np.zeros_like(all_coeff, dtype=np.int8)
        scales = np.ones(math.ceil(n/c.group_size), np.float16)
        tiny = float(np.nextafter(np.float16(0), np.float16(1)))
        for g, start in enumerate(range(0, n, c.group_size)):
            coeff = all_coeff[start:start+c.group_size]
            peak = float(np.abs(coeff).max())
            scale = max(peak/127, tiny) if peak else 1.0
            if scale > np.finfo(np.float16).max:
                raise OverflowError("Coefficient scale does not fit FP16; change block/seed configuration")
            scales[g] = scale
            q[start:start+c.group_size] = np.clip(np.rint(coeff / float(scales[g])), -127, 127).astype(np.int8)
        packed_coeff = np.concatenate([row[:int(k)] for row, k in zip(q, counts)])
        result = CompressedTensor(tuple(original.shape), c, seeds, counts, packed_coeff, scales)
        result.validate()
        restored = decompress(result).astype(np.float64)
        residual = original.astype(np.float64) - restored
        post_blocks = np.pad(restored.ravel(), (0, flat.size-original.size)).reshape(n, c.block_size)
        post_error = ((blocks-post_blocks)**2).sum(axis=1)
        post_ratio = 1 - np.divide(post_error, energy, out=np.zeros(n), where=energy > 0)
        result.metrics = {"mse": float(np.mean(residual**2)),
                          "block_mean_squared_frobenius": float(np.sum(residual**2)/n),
                          "relative_squared_error": float(np.sum(residual**2)/max(float(energy.sum()), 1e-300)),
                          "pre_quant_min_energy_ratio": float(pre_ratio.min()),
                          "pre_quant_threshold_misses": failed,
                          "post_quant_min_energy_ratio": float(post_ratio.min()),
                          "post_quant_threshold_misses": int(np.count_nonzero(post_ratio < c.energy_threshold-1e-9)),
                          **result.storage()}
        return result


def compress(weight, config=None):
    return Compressor(config or QuantConfig()).compress(weight)


def decompress(tensor):
    tensor.validate()
    c = tensor.config
    n = len(tensor.seeds)
    offsets = np.concatenate(([0], np.cumsum(tensor.counts, dtype=np.int64)))
    output = np.empty((n, c.block_size), np.float64)
    for start in range(0, n, c.block_chunk):
        stop = min(start+c.block_chunk, n)
        kmax = int(tensor.counts[start:stop].max())
        coeff = np.zeros((stop-start, kmax))
        for local, index in enumerate(range(start, stop)):
            coeff[local, :tensor.counts[index]] = tensor.coefficients[offsets[index]:offsets[index+1]] * float(tensor.scales[index//c.group_size])
        u = bases(tensor.seeds[start:stop], c, kmax)
        output[start:stop] = np.einsum("nbk,nk->nb", u, coeff)
    return output.ravel()[:tensor.numel].reshape(tensor.shape).astype(np.float32)


def pack_unsigned(values, width):
    values = np.asarray(values, dtype=np.uint32).ravel()
    if width < 1 or width > 32 or np.any(values.astype(np.uint64) >= 2**width):
        raise ValueError("Value does not fit packed width")
    # Small temporary chunks avoid allocating num_weights * seed_bits at once.
    chunks = []
    for start in range(0, len(values), 8192):  # divisible by 8: no inter-chunk bit padding
        v = values[start:start+8192]
        bits = ((v[:, None] >> np.arange(width, dtype=np.uint32)) & 1).astype(np.uint8)
        chunks.append(np.packbits(bits.ravel(), bitorder="little").tobytes())
    return b"".join(chunks)


def unpack_unsigned(data, width, count):
    if not 1 <= width <= 32 or count < 0 or len(data) != (count*width+7)//8:
        raise ValueError("Invalid packed buffer length")
    result = np.empty(count, np.uint32)
    for start in range(0, count, 8192):
        end = min(start+8192, count)
        chunk = data[start*width//8:(end*width+7)//8]
        bits = np.unpackbits(np.frombuffer(chunk, np.uint8), bitorder="little")[:(end-start)*width]
        result[start:end] = (bits.reshape(-1, width).astype(np.uint32) << np.arange(width, dtype=np.uint32)).sum(1, dtype=np.uint32)
    return result


def save(tensor, path):
    tensor.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffers = {"seeds.bin": pack_unsigned(tensor.seeds, tensor.config.seed_bits),
               "counts.bin": pack_unsigned(tensor.counts, tensor.count_bits),
               "coefficients.bin": tensor.coefficients.tobytes(),
               "scales.bin": tensor.scales.astype("<f2").tobytes()}
    manifest = {"format": "squant", "version": 1, "shape": list(tensor.shape),
                "config": tensor.config.to_dict(), "metrics": tensor.metrics,
                "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in buffers.items()}}
    temp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, allow_nan=False))
        for name, data in buffers.items():
            archive.writestr(name, data)
    temp.replace(path)
    return {**tensor.storage(), "file_bytes": path.stat().st_size,
            "file_bits_per_weight": path.stat().st_size*8/tensor.numel}


def load(path):
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("manifest.json"))
        if meta.get("format") != "squant" or meta.get("version") != 1:
            raise ValueError("Unsupported S-Quant format")
        c = QuantConfig(**meta["config"])
        shape = tuple(meta["shape"])
        if not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("Invalid shape")
        n = math.ceil(math.prod(shape)/c.block_size)
        buffers = {name: archive.read(name) for name in ("seeds.bin", "counts.bin", "coefficients.bin", "scales.bin")}
        for name, data in buffers.items():
            if hashlib.sha256(data).hexdigest() != meta["sha256"][name]:
                raise ValueError(f"Checksum mismatch: {name}")
    result = CompressedTensor(shape, c,
        unpack_unsigned(buffers["seeds.bin"], c.seed_bits, n),
        unpack_unsigned(buffers["counts.bin"], c.max_bases.bit_length(), n).astype(np.uint16),
        np.frombuffer(buffers["coefficients.bin"], np.int8).copy(),
        np.frombuffer(buffers["scales.bin"], "<f2").astype(np.float16), meta.get("metrics", {}))
    result.validate()
    return result
