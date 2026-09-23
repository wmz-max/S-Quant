"""Offline smoke experiments and controlled tensor ablations."""
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from .codec import compress, decompress, save, load
from .config import QuantConfig


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def demo(output, config=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.02, (32, 32)).astype(np.float32)
    start = time.perf_counter()
    tensor = compress(weight, config or QuantConfig(block_size=8, seed_bits=6, max_bases=8))
    elapsed = time.perf_counter()-start
    storage = save(tensor, output/"weight.sqz")
    restored = decompress(load(output/"weight.sqz"))
    np.testing.assert_array_equal(restored, decompress(tensor))
    np.save(output/"original.npy", weight)
    np.save(output/"reconstructed.npy", restored)
    x = rng.normal(size=(4, 32)).astype(np.float32)
    report = {"experiment": "seeded synthetic tensor; not a pretrained LLM benchmark",
              "config": tensor.config.to_dict(), "compression_seconds": elapsed,
              **tensor.metrics, **storage,
              "linear_output_relative_error": float(np.linalg.norm(x@weight.T-x@restored.T)/np.linalg.norm(x@weight.T)),
              "disk_roundtrip_exact": True}
    write_json(output/"report.json", report)
    return report


def ablate(weight, base, output):
    variants = [("adaptive", base)]
    for k in [2, 4]:
        if k <= base.max_bases:
            variants.append((f"fixed_{k}", replace(base, fixed_bases=k)))
    for threshold in [0.8, 0.9, 0.95]:
        variants.append((f"threshold_{threshold}", replace(base, energy_threshold=threshold)))
    for b in [8, 16, 32]:
        variants.append((f"block_{b}", replace(base, block_size=b, max_bases=b)))
    for s in [6, 8]:
        variants.append((f"seed_{s}", replace(base, seed_bits=s, seed_budget=None)))
    results = [{"variant": name, "config": config.to_dict(), **compress(weight, config).metrics}
               for name, config in variants]
    write_json(output, {"experiment": "tensor reconstruction ablation; no perplexity measured", "results": results})
    return results


def tiny_llm(output):
    """A local random Llama model checks integration without downloading weights."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from .torch_integration import quantize_model
    from .evaluation import perplexity
    from .checkpoint import compress_checkpoint, restore_checkpoint
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)).eval()
    ids = torch.randint(0, 64, (1, 67))
    base_ppl = perplexity(model, ids, context_length=32)
    config = QuantConfig(block_size=8, seed_bits=5, max_bases=8, group_size=8)
    model.save_pretrained(output/"original_model", safe_serialization=True)
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                  **{f"token{i}": i+4 for i in range(60)}}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend,
        unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(output/"original_model")
    (output/"sample.txt").write_text(" ".join(f"token{i%60}" for i in range(67)))
    stats = compress_checkpoint(str(output/"original_model"), output/"compressed_model", config)
    restore_checkpoint(output/"compressed_model", output/"restored_model", shard_mb=1)
    dense = LlamaForCausalLM.from_pretrained(output/"restored_model").eval()
    dense_ppl = perplexity(dense, ids, context_length=32)
    layer_report = quantize_model(model, config)
    with torch.inference_mode():
        decoded_logits = model(ids).logits
        restored_logits = dense(ids).logits
        torch.testing.assert_close(decoded_logits, restored_logits, atol=2e-6, rtol=2e-5)
        generated = model.generate(ids[:, :4], max_new_tokens=4, do_sample=False, pad_token_id=0)
    report = {"experiment": "random tiny Llama, random tokens; integration test ONLY",
              "baseline": base_ppl, "compressed": dense_ppl, "checkpoint": stats,
              "compressed_layers": len(layer_report),
              "on_demand_vs_export_max_logit_difference": float((decoded_logits-restored_logits).abs().max()),
              "generated_token_ids": generated.tolist()}
    write_json(output/"report.json", report)
    return report
