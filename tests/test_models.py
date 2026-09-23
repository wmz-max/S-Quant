import math
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from squant import QuantConfig, compress, decompress
from squant.torch_integration import SQuantLinear
from squant.evaluation import perplexity


def test_linear_decoder_and_forward():
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4, group_size=3, block_chunk=3)
    w = np.random.default_rng(0).normal(size=(5, 7)).astype(np.float32)
    r = compress(w, c)
    layer = SQuantLinear(r, bias=torch.arange(5).float(), cache_decoded=True)
    expected = torch.from_numpy(decompress(r))
    torch.testing.assert_close(layer.decode_weight(), expected)
    x = torch.randn(2, 3, 7)
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, expected, layer.bias))
    layer.to(dtype=torch.float64)
    assert layer._decoded is None
    assert layer.decode_weight().dtype == torch.float64
    torch.testing.assert_close(layer(x.double()), torch.nn.functional.linear(x.double(), expected.double(), layer.bias), rtol=1e-5, atol=1e-6)


class UniformLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 7, device=input_ids.device))


@pytest.mark.parametrize("length", [2, 7, 8, 9, 17, 25])
@pytest.mark.parametrize("stride", [1, 3, 7])
def test_ppl_scores_every_token_once(length, stride):
    model = UniformLM().train()
    result = perplexity(model, torch.zeros(length, dtype=torch.long), context_length=8, stride=stride)
    assert result["scored_tokens"] == length-1
    assert result["perplexity"] == pytest.approx(7, rel=1e-6)
    assert model.training


def test_ppl_invalid_window():
    with pytest.raises(ValueError):
        perplexity(UniformLM(), torch.zeros(12, dtype=torch.long), context_length=8, stride=8)


def test_bfloat16_conversion_preserves_scale_bits():
    c = QuantConfig(block_size=4, seed_bits=4, max_bases=4)
    r = compress(np.random.default_rng(3).normal(size=(4,4)), c)
    layer = SQuantLinear(r)
    before = layer.scales.clone()
    layer.bfloat16()
    assert layer.scales.dtype == torch.float16
    torch.testing.assert_close(layer.scales, before, rtol=0, atol=0)


def test_tiny_llama_checkpoint_and_generation(tmp_path):
    pytest.importorskip("transformers")
    from squant.experiments import tiny_llm
    report = tiny_llm(tmp_path/"llama")
    assert report["compressed_layers"] == 7
    assert report["on_demand_vs_export_max_logit_difference"] < 2e-6
    assert report["baseline"]["scored_tokens"] == 66
    assert len(report["generated_token_ids"][0]) == 8
    from squant.evaluation import evaluate_hf
    evaluation = evaluate_hf(str(tmp_path/"llama/restored_model"),
        text_file=tmp_path/"llama/sample.txt", context_length=32)
    assert evaluation["scored_tokens"] == 66
    assert math.isfinite(evaluation["perplexity"])


def test_incomplete_checkpoint_rejected(tmp_path):
    import json
    from squant.checkpoint import restore_checkpoint
    (tmp_path/"manifest.json").write_text(json.dumps({"format": "squant-hf", "version": 1, "complete": False}))
    with pytest.raises(ValueError, match="Incomplete"):
        restore_checkpoint(tmp_path, tmp_path/"export")
