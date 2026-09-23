import numpy as np
from .config import QuantConfig
from .codec import compress, decompress


def gpu_check(device="cuda:0"):
    import torch
    from .torch_integration import SQuantLinear
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Check nvidia-smi and install a CUDA-enabled PyTorch build.")
    torch.cuda.set_device(target)
    config = QuantConfig(block_size=4, seed_bits=4, max_bases=4, group_size=4,
                         device=str(target), strict_threshold=True)
    weights = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32)
    packed = compress(weights, config)
    layer = SQuantLinear(packed).to(target)
    reference = torch.from_numpy(decompress(packed)).to(target)
    with torch.inference_mode():
        torch.testing.assert_close(layer.decode_weight(), reference, atol=2e-6, rtol=2e-5)
        x = torch.arange(16, device=target, dtype=torch.float32).reshape(2, 8)/16
        torch.testing.assert_close(layer(x), x@reference.T, atol=2e-6, rtol=2e-5)
    torch.cuda.synchronize(target)
    return {"status": "passed", "device": str(target),
            "gpu": torch.cuda.get_device_name(target), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "total_memory_gib": round(torch.cuda.get_device_properties(target).total_memory/1024**3, 2),
            "checks": ["CUDA candidate search", "seed reconstruction", "linear forward"],
            "scope": "small tensor smoke test; not a full-model accuracy or throughput benchmark"}
