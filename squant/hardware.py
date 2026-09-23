"""Integer golden models for the included functional RTL, not paper PPA."""
import numpy as np
from .lfsr import states


def reconstruct_numerators(seed, coefficients, block_size, seed_bits, mask):
    q = np.asarray(coefficients, dtype=np.int64)
    if q.ndim != 1 or not len(q) or np.any(q < -128) or np.any(q > 127):
        raise ValueError("Expected nonempty int8 coefficients")
    z = states([seed], block_size*len(q), seed_bits, mask)[0].astype(np.int64)
    return ((z.reshape(len(q), block_size)-2**(seed_bits-1))*q[:, None]).sum(0)


def generator_trace(seed, coefficients, block_size, seed_bits, mask, ready_pattern=(True,)):
    """Cycle trace from accepted start through done; period must allow progress."""
    if not ready_pattern or not any(ready_pattern):
        raise ValueError("Output must eventually become ready")
    q = np.asarray(coefficients, dtype=np.int64)
    expected = reconstruct_numerators(seed, q, block_size, seed_bits, mask)
    trace, state = [], int(seed)
    accum = np.zeros(block_size, dtype=np.int64)
    cycle = 0
    for k, a in enumerate(q):
        for b in range(block_size):
            cycle += 1
            accum[b] += int(a)*(state-2**(seed_bits-1))
            trace.append({"cycle": cycle, "phase": "compute", "basis": k, "element": b,
                          "state": state, "accumulator": int(accum[b])})
            state = (state >> 1) ^ ((state & 1)*mask)
    b = 0
    while b < block_size:
        ready = bool(ready_pattern[cycle % len(ready_pattern)])
        cycle += 1
        trace.append({"cycle": cycle, "phase": "emit", "element": b,
                      "numerator": int(accum[b]), "ready": ready, "done": ready and b == block_size-1})
        if ready:
            b += 1
    np.testing.assert_array_equal(accum, expected)
    return {"architecture": "included serial integer reference RTL", "cycles_after_start": cycle,
            "compute_cycles": block_size*len(q), "trace": trace, "numerators": accum.tolist()}


def systolic_multiply(a, b):
    """Cycle-by-cycle output-stationary N x N tile with skewed edge streams."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[0] != b.shape[1]:
        raise ValueError("Expected (N,K) and (K,N)")
    n, k = a.shape
    aa, bb = np.zeros((n, n), np.int64), np.zeros((n, n), np.int64)
    av, bv = np.zeros((n, n), bool), np.zeros((n, n), bool)
    sums = np.zeros((n, n), np.int64)
    for cycle in range(k+2*(n-1)):
        next_a, next_b = np.zeros_like(aa), np.zeros_like(bb)
        next_av, next_bv = np.zeros_like(av), np.zeros_like(bv)
        for i in range(n):
            for j in range(n):
                ax = a[i, cycle-i] if j == 0 and 0 <= cycle-i < k else (aa[i,j-1] if j else 0)
                bx = b[cycle-j, j] if i == 0 and 0 <= cycle-j < k else (bb[i-1,j] if i else 0)
                axv = (0 <= cycle-i < k) if j == 0 else av[i,j-1]
                bxv = (0 <= cycle-j < k) if i == 0 else bv[i-1,j]
                if axv and bxv:
                    sums[i,j] += ax*bx
                next_a[i,j], next_b[i,j] = ax, bx
                next_av[i,j], next_bv[i,j] = axv, bxv
        aa, bb, av, bv = next_a, next_b, next_av, next_bv
    return {"result": sums.tolist(), "cycles": k+2*(n-1)}
