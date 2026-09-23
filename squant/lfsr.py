import numpy as np
from .config import QuantConfig


def states(seeds, length, bits, mask):
    """Shape (seeds, length); zero is forbidden (the absorbing LFSR state)."""
    state = np.asarray(seeds, dtype=np.uint32).reshape(-1).copy()
    if length < 0 or np.any(state == 0) or np.any(state >= 2**bits):
        raise ValueError("Invalid seed or sequence length")
    result = np.empty((state.size, length), dtype=np.uint32)
    for i in range(length):
        result[:, i] = state
        state = (state >> 1) ^ ((state & 1) * np.uint32(mask))
    return result


def bases(seeds, config: QuantConfig, k):
    """Consecutive length-B segments become columns of U: (seeds, B, k)."""
    z = states(seeds, config.block_size * k, config.seed_bits, config.mask)
    mid = 2 ** (config.seed_bits - 1)
    normalized = (z.astype(np.float64) - mid) / (mid - 1)
    return normalized.reshape(-1, k, config.block_size).transpose(0, 2, 1)


def candidate_seeds(config):
    count = 2**config.seed_bits - 1
    if config.seed_budget is None or config.seed_budget == count:
        return np.arange(1, count + 1, dtype=np.uint32)
    # Explicitly approximate mode; reproducible and without replacement.
    rng = np.random.default_rng(config.random_seed)
    return np.sort(rng.choice(count, size=config.seed_budget, replace=False) + 1).astype(np.uint32)

