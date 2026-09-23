from dataclasses import replace
import io
import json
import zipfile
import numpy as np
import pytest
from squant import QuantConfig, compress, decompress, save, load
from squant.codec import Compressor, pack_unsigned, unpack_unsigned
from squant.config import MASKS
from squant.lfsr import bases, states, candidate_seeds


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8, 10, 12, 16])
def test_lfsr_maximal_period(bits):
    seq = states([1], 2**bits, bits, MASKS[bits])[0]
    assert len(np.unique(seq[:-1])) == 2**bits-1
    assert seq[-1] == 1
    assert seq.min() == 1


def test_golden_state_and_basis_order():
    np.testing.assert_array_equal(states([1], 5, 4, 0xC), [[1, 12, 6, 3, 13]])
    c = QuantConfig(block_size=2, seed_bits=4, max_bases=2)
    np.testing.assert_allclose(bases([1], c, 2)[0], np.array([[-7, -2], [4, -5]])/7)


@pytest.mark.parametrize("width", [1, 3, 8, 16, 24, 32])
@pytest.mark.parametrize("length", [1, 13, 8193])
def test_bitpacking(width, length):
    rng = np.random.default_rng(1)
    values = rng.integers(0, 2**width, length, dtype=np.uint32)
    data = pack_unsigned(values, width)
    assert len(data) == (length*width+7)//8
    np.testing.assert_array_equal(unpack_unsigned(data, width, length), values)


def test_search_agrees_with_independent_lstsq():
    c = QuantConfig(block_size=4, seed_bits=3, max_bases=4, seed_chunk=2)
    w = np.random.default_rng(2).normal(size=(3, 4))
    errors, seeds, coefficients = Compressor(c)._best(w, 2)
    for i, block in enumerate(w):
        candidates = []
        for seed in range(1, 8):
            u = bases([seed], c, 2)[0]
            a = np.linalg.lstsq(u, block, rcond=1e-10)[0]
            candidates.append(np.sum((u@a-block)**2))
        np.testing.assert_allclose(errors[i], min(candidates), atol=1e-12)
        np.testing.assert_allclose(bases([seeds[i]], c, 2)[0]@coefficients[i],
                                   bases([seeds[i]], c, 2)[0]@np.linalg.lstsq(bases([seeds[i]], c, 2)[0], block, rcond=1e-10)[0])


def test_adaptive_threshold_and_disk_roundtrip(tmp_path):
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4, group_size=3, strict_threshold=True)
    weight = np.random.default_rng(3).normal(size=(7, 5)).astype(np.float32)
    packed = compress(weight, c)
    assert packed.metrics["pre_quant_threshold_misses"] == 0
    assert packed.metrics["pre_quant_min_energy_ratio"] >= .9-1e-9
    stat = save(packed, tmp_path/"test.sqz")
    restored = load(tmp_path/"test.sqz")
    np.testing.assert_array_equal(decompress(restored), decompress(packed))
    assert decompress(restored).shape == (7, 5)
    assert stat["logical_bits_per_weight"] > stat["paper_bits_per_weight"]
    with zipfile.ZipFile(tmp_path/"test.sqz") as z:
        assert sum(z.getinfo(x).file_size for x in z.namelist() if x.endswith(".bin")) == stat["payload_bytes"]


def test_adaptive_minimum_k():
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4, energy_threshold=.95)
    w = np.random.default_rng(5).normal(size=32).astype(np.float32)
    result = compress(w, c)
    for index, k in enumerate(result.counts):
        if k > c.min_bases:
            err, _, _ = Compressor(c)._best(w.reshape(-1, 4)[index:index+1], int(k)-1)
            ratio = 1-err[0]/np.sum(w.reshape(-1, 4)[index].astype(float)**2)
            assert ratio < c.energy_threshold


def test_zero_tiny_large_and_invalid_weights():
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4)
    for w in [np.zeros(7), np.full(5, 1e-20), np.full(9, 1000.)]:
        r = compress(w, c)
        assert np.isfinite(decompress(r)).all()
        if not np.any(w):
            assert not np.any(decompress(r))
            assert np.all(r.counts == c.min_bases)
    for w in [np.array([np.nan]), np.array([np.inf]), np.array([]), np.array([1, 2])]:
        with pytest.raises(ValueError):
            compress(w, c)


def test_threshold_failure_is_explicit():
    c = QuantConfig(block_size=16, seed_bits=2, max_bases=2, energy_threshold=.999, strict_threshold=True)
    with pytest.raises(ValueError, match="missed"):
        compress(np.random.default_rng(0).normal(size=16), c)


def test_sampled_search_reproducible():
    c = QuantConfig(seed_bits=8, seed_budget=17)
    np.testing.assert_array_equal(candidate_seeds(c), candidate_seeds(c))
    assert len(np.unique(candidate_seeds(c))) == 17


def test_corrupted_archive_rejected(tmp_path):
    path = tmp_path/"bad.sqz"
    save(compress(np.ones(8), QuantConfig(block_size=4, seed_bits=4, max_bases=4)), path)
    with zipfile.ZipFile(path) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    entries["coefficients.bin"] = b"x"+entries["coefficients.bin"][1:]
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    with pytest.raises(ValueError, match="Checksum"):
        load(path)


def test_cpu_torch_projection_matches_numpy():
    pytest.importorskip("torch")
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4)
    w = np.random.default_rng(10).normal(size=32)
    a, b = compress(w, c), compress(w, replace(c, device="cpu"))
    np.testing.assert_allclose(decompress(a), decompress(b), atol=1e-6)

