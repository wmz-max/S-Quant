import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .config import QuantConfig
from .codec import compress, decompress, load, save
from .experiments import write_json


def _config(args):
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.command == "hf-compress":
        values.setdefault("device", "cuda")
    if getattr(args, "search_device", None):
        values["device"] = args.search_device
    return QuantConfig(**values)


def _weight(path):
    return np.load(path, allow_pickle=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="S-Quant paper-grounded reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print dependency and device availability")
    p = sub.add_parser("gpu-check", help="Check CUDA compression and reconstruction on the server GPU")
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("hardware-demo", help="Cycle trace of the reference integer reconstruction core")
    p.add_argument("--output", required=True)
    for name, description in [("demo", "Offline compression and packed-file demo"),
                               ("compress", "Compress a floating-point .npy tensor"),
                               ("search", "Matérn GP + expected hypervolume search"),
                               ("ablate", "Tensor reconstruction ablations"),
                               ("hf-compress", "Stream a safetensors model into S-Quant files")]:
        p = sub.add_parser(name, help=description)
        p.add_argument("--config")
        p.add_argument("--search-device",
                       help="numpy, cpu (PyTorch), or cuda[:index]; hf-compress defaults to cuda")
        p.add_argument("--output", required=True)
        if name in {"compress", "search", "ablate"}:
            p.add_argument("--input", required=True)
        if name == "search":
            p.add_argument("--trials", type=int, default=12)
            p.add_argument("--target-bpw", type=float)
            p.add_argument("--sample-weights", type=int, default=4096,
                           help="Use the first N weights; 0 means full input. Recorded in result.")
        if name == "hf-compress":
            from .checkpoint import DEFAULT_INCLUDE
            p.add_argument("--model", required=True)
            p.add_argument("--include", default=DEFAULT_INCLUDE)
            p.add_argument("--exclude", default="lm_head|embed")
            p.add_argument("--revision")
            p.add_argument("--cache-dir", default=".cache/huggingface")
    p = sub.add_parser("decompress", help="Restore .sqz to .npy")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("inspect", help="Show real bit accounting")
    p.add_argument("input")
    p = sub.add_parser("tiny-llm", help="Offline Llama checkpoint, forward and generation tests")
    p.add_argument("--output", required=True)
    p = sub.add_parser("hf-export", help="Export compressed checkpoint to dense HF safetensors")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-mb", type=int, default=512)
    p = sub.add_parser("evaluate", help="Causal perplexity on local text or WikiText-2 test")
    p.add_argument("--model", required=True)
    p.add_argument("--text-file")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--stride", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    p.add_argument("--revision")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--output", required=True)
    p = sub.add_parser("zero-shot", help="LM Evaluation Harness zero-shot tasks")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "gpu-check":
        from .server import gpu_check
        result = gpu_check(args.device)
    elif args.command == "doctor":
        import importlib.metadata
        import platform
        result = {"python": sys.version, "platform": platform.platform(), "dependencies": {}}
        for name in ["numpy", "torch", "transformers", "datasets", "scikit-learn", "lm-eval", "yowasp-yosys"]:
            try:
                result["dependencies"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["dependencies"][name] = None
        if result["dependencies"]["torch"]:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            result["mps_available"] = torch.backends.mps.is_available()
    elif args.command == "hardware-demo":
        from .hardware import generator_trace, systolic_multiply
        result = {"reconstruction": generator_trace(1, [-3, 7, 0], 8, 8, 0xB8, [True, False, True]),
                  "systolic": systolic_multiply([[1,-2,3],[4,5,-6]], [[7,8],[-9,10],[11,-12]]),
                  "scope": "functional reference cycles; no DRAM, energy, area, or paper speedup claim"}
        write_json(args.output, result)
    elif args.command == "demo":
        from .experiments import demo
        result = demo(args.output, _config(args) if args.config or args.search_device else None)
    elif args.command == "compress":
        tensor = compress(_weight(args.input), _config(args))
        result = {**tensor.metrics, **save(tensor, args.output)}
    elif args.command == "decompress":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, decompress(load(args.input)))
        result = {"output": str(path)}
    elif args.command == "inspect":
        tensor = load(args.input)
        result = {"shape": tensor.shape, "config": tensor.config.to_dict(), **tensor.metrics,
                  **tensor.storage(), "file_bits_per_weight": Path(args.input).stat().st_size*8/tensor.numel}
    elif args.command == "search":
        from .search import search, configuration_grid
        weight = _weight(args.input)
        if args.sample_weights < 0:
            parser.error("--sample-weights cannot be negative")
        sample = weight.ravel()[:args.sample_weights] if args.sample_weights else weight
        result = search(sample, configuration_grid(_config(args)), trials=args.trials,
                        target_bpw=args.target_bpw,
                        progress=lambda i, r: print(f"trial {i}: MSE={r['mse']:.6g}, BPW={r['logical_bits_per_weight']:.3f}", file=sys.stderr))
        result.update(sample_weights=int(sample.size), original_weights=int(weight.size))
        write_json(args.output, result)
    elif args.command == "ablate":
        from .experiments import ablate
        result = ablate(_weight(args.input), _config(args), args.output)
    elif args.command == "tiny-llm":
        from .experiments import tiny_llm
        result = tiny_llm(args.output)
    elif args.command == "hf-compress":
        from .checkpoint import compress_checkpoint
        result = compress_checkpoint(args.model, args.output, _config(args), args.include, args.exclude,
                    args.revision, args.cache_dir,
                    progress=lambda n, e: print(f"{n}: {e['kind']}", file=sys.stderr))
    elif args.command == "hf-export":
        from .checkpoint import restore_checkpoint
        result = restore_checkpoint(args.input, args.output, args.shard_mb)
    elif args.command == "evaluate":
        from .evaluation import evaluate_hf
        result = evaluate_hf(args.model, args.text_file, args.context_length, args.stride,
                             args.device, args.max_tokens, args.revision, args.dtype)
        write_json(args.output, result)
    elif args.command == "zero-shot":
        from .evaluation import zero_shot
        result = zero_shot(args.model, args.tasks, args.device, args.limit, args.batch_size)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str))
