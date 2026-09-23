import numpy as np
import pytest
from squant.search import hypervolume, pareto_indices, search
from squant import QuantConfig


def test_known_hypervolume_and_dominance():
    y = [[1, 3], [2, 2], [3, 1], [3, 3], [1, 3]]
    assert hypervolume(y, [4, 4]) == 6
    assert pareto_indices(y) == [0, 1, 2, 4]
    assert hypervolume([], [4, 4]) == 0
    assert hypervolume([[5, 0]], [4, 4]) == 0


def test_search_budget_determinism_and_unreachable_target():
    pytest.importorskip("sklearn")
    from dataclasses import replace
    c = QuantConfig(block_size=4, seed_bits=3, max_bases=4)
    grid = [replace(c, energy_threshold=t) for t in [.5, .7, .8, .9, .99]]
    w = np.random.default_rng(0).normal(size=24)
    a = search(w, grid, trials=4, initial=2, mc_samples=8, target_bpw=.01)
    b = search(w, grid, trials=4, initial=2, mc_samples=8, target_bpw=.01)
    assert a == b
    assert len(a["observations"]) == 4
    assert a["selected_config"] is None
    assert not a["target_feasible"]

