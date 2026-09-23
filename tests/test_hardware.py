import numpy as np
import pytest
from squant.hardware import reconstruct_numerators, generator_trace, systolic_multiply
from squant import QuantConfig, compress, decompress


def test_integer_reconstruction_matches_codec():
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4, group_size=2)
    w = np.random.default_rng(0).normal(size=16).astype(np.float32)
    packed = compress(w, c)
    offset = 0
    blocks = []
    for i, k in enumerate(packed.counts):
        q = packed.coefficients[offset:offset+k]
        offset += k
        num = reconstruct_numerators(int(packed.seeds[i]), q, 4, 4, c.mask)
        blocks.append(num*float(packed.scales[i//c.group_size])/7)
    np.testing.assert_allclose(np.concatenate(blocks), decompress(packed), atol=1e-6)


def test_stalls_preserve_outputs_and_cycles():
    trace = generator_trace(1, [-3, 7, 0], 8, 8, 0xB8, [True, False, False])
    accepted = [row["numerator"] for row in trace["trace"] if row["phase"] == "emit" and row["ready"]]
    assert accepted == trace["numerators"]
    assert trace["compute_cycles"] == 24
    assert trace["cycles_after_start"] > 32
    assert trace["trace"][-1]["done"]
    with pytest.raises(ValueError):
        generator_trace(1, [1], 8, 8, 0xB8, [False])


@pytest.mark.parametrize("n,k", [(1, 1), (2, 3), (4, 7)])
def test_systolic_integer_reference(n, k):
    rng = np.random.default_rng(n)
    a = rng.integers(-127, 128, size=(n,k))
    b = rng.integers(-127, 128, size=(k,n))
    result = systolic_multiply(a,b)
    np.testing.assert_array_equal(result["result"], a@b)
    assert result["cycles"] == k+2*(n-1)
