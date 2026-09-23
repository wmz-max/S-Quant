import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA server")
def test_cuda_search_decode_and_forward():
    from squant.server import gpu_check
    result = gpu_check("cuda:0")
    assert result["status"] == "passed"
    assert result["total_memory_gib"] > 0
