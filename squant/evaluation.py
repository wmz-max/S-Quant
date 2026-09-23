import math
from pathlib import Path


def perplexity(model, input_ids, context_length=2048, stride=None, device=None):
    """Token-weighted causal NLL; overlap masks ensure each target is scored once.

    Default stride=context_length-1 retains one context token between chunks.
    No padding is scored and the final partial chunk is included.
    """
    import torch
    import torch.nn.functional as F
    stride = context_length-1 if stride is None else stride
    if context_length < 2 or not 1 <= stride < context_length:
        raise ValueError("Require context_length >= 2 and 1 <= stride < context_length")
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("Expected a single token sequence with at least two tokens")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    nll, count, previous_end = 0., 0, 1
    try:
        with torch.inference_mode():
            for begin in range(0, input_ids.shape[1]-1, stride):
                end = min(begin+context_length, input_ids.shape[1])
                if end <= previous_end:
                    break
                tokens = input_ids[:, begin:end].to(device)
                logits = model(input_ids=tokens, use_cache=False).logits[:, :-1].float()
                targets = tokens[:, 1:]
                first_target = max(previous_end, begin+1)
                offset = first_target-(begin+1)
                loss = F.cross_entropy(logits[:, offset:].reshape(-1, logits.shape[-1]),
                                       targets[:, offset:].reshape(-1), reduction="sum")
                nll += float(loss)
                count += end-first_target
                previous_end = end
                if end == input_ids.shape[1]:
                    break
    finally:
        model.train(was_training)
    mean_nll = nll/count
    return {"perplexity": math.exp(mean_nll) if mean_nll < 709 else float("inf"),
            "mean_nll": mean_nll, "scored_tokens": count,
            "context_length": context_length, "stride": stride,
            "protocol": "overlapping windows; each target token scored once; final window included"}


def evaluate_hf(model_path, text_file=None, context_length=2048, stride=None,
                device="cpu", max_tokens=None, revision=None, dtype="float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=False)
    kwargs = {"revision": revision, "trust_remote_code": False, "torch_dtype": getattr(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device != "auto":
        model.to(device)
    if text_file:
        text = Path(text_file).read_text()
        source = str(text_file)
    else:
        from datasets import load_dataset
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
        source = "Salesforce/wikitext:wikitext-2-raw-v1:test"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if max_tokens is not None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        tokens = tokens[:, :max_tokens]
    result = perplexity(model, tokens, context_length, stride)
    result.update(model=str(model_path), revision=revision, dataset=source,
                  max_tokens=max_tokens, dtype=dtype, device=device)
    return result


def zero_shot(model_path, tasks, device="cpu", limit=None, batch_size=1):
    import lm_eval
    # Standard dense export is accepted directly by lm-evaluation-harness.
    return lm_eval.simple_evaluate(model="hf", model_args={"pretrained": str(model_path),
                                   "trust_remote_code": False}, tasks=list(tasks),
                                   device=device, num_fewshot=0, batch_size=batch_size, limit=limit,
                                   random_seed=0, numpy_random_seed=0, torch_random_seed=0)
