"""Streaming Hugging Face safetensors checkpoints, one tensor at a time."""
import json
from pathlib import Path
import re
import shutil
from .codec import Compressor, decompress, load, save

DEFAULT_INCLUDE = r"\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|c_attn|c_proj|c_fc|fc1|fc2|out_proj)\.weight$"


def _json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def _fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_metadata(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".json", ".model", ".txt", ".tiktoken", ".jinja"}:
            if not path.name.endswith(".safetensors.index.json"):
                shutil.copy2(path, dest/path.name)


def resolve_checkpoint(model, revision=None, cache_dir=None):
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, revision=revision, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja"]))


def compress_checkpoint(model, output, config, include=DEFAULT_INCLUDE, exclude=r"lm_head|embed",
                        revision=None, cache_dir=None, progress=None):
    from safetensors import safe_open
    from safetensors.torch import save_file
    # Fail before downloading a large checkpoint if the requested backend is unavailable.
    compressor = Compressor(config)
    source = resolve_checkpoint(model, revision, cache_dir)
    index_path = source/"model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in shard_names):
            raise ValueError("Shard paths must be filenames")
        shards = [source/name for name in shard_names]
    elif (source/"model.safetensors").exists():
        shards = [source/"model.safetensors"]
        weight_map = None
    else:
        raise ValueError("Expected model.safetensors or model.safetensors.index.json; convert .bin checkpoints first")
    output = _fresh_directory(output)
    _copy_metadata(source, output/"hf_metadata")
    (output/"tensors").mkdir()
    manifest = {"format": "squant-hf", "version": 1, "source": str(model), "revision": revision,
                "resolved_source": str(source), "config": config.to_dict(),
                "include": include, "exclude": exclude, "tensors": {}, "complete": False}
    numel, compressed_numel, logical_bits, payload_bytes, original_bytes = 0, 0, 0., 0, 0
    seen = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if weight_map is not None and weight_map.get(name) != shard.name:
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate tensor: {name}")
                seen.add(name)
                weight = handle.get_tensor(name)
                dtype = str(weight.dtype).split(".")[-1]
                numel += weight.numel()
                original_bytes += weight.numel()*weight.element_size()
                number = len(manifest["tensors"])
                eligible = weight.ndim == 2 and weight.is_floating_point() and re.search(include, name) and not (exclude and re.search(exclude, name))
                if eligible:
                    packed = compressor.compress(weight.float().numpy())
                    filename = f"tensors/{number:06d}.sqz"
                    storage = save(packed, output/filename)
                    entry = {"kind": "squant", "path": filename, "dtype": dtype, "shape": list(weight.shape),
                             "metrics": packed.metrics, "storage": storage}
                    logical_bits += storage["logical_bits_per_weight"]*weight.numel()
                    payload_bytes += storage["payload_bytes"]
                    compressed_numel += weight.numel()
                else:
                    filename = f"tensors/{number:06d}.safetensors"
                    save_file({name: weight.contiguous()}, str(output/filename), metadata={"format": "pt"})
                    entry = {"kind": "passthrough", "path": filename, "dtype": dtype, "shape": list(weight.shape)}
                    logical_bits += weight.numel()*weight.element_size()*8
                    payload_bytes += weight.numel()*weight.element_size()
                manifest["tensors"][name] = entry
                _json(output/"manifest.json", manifest)
                if progress:
                    progress(name, entry)
    if weight_map is not None and seen != set(weight_map):
        raise ValueError("Some indexed tensors were not found")
    if not compressed_numel:
        raise ValueError("No tensors matched; inspect --include regex")
    manifest["complete"] = True
    manifest["summary"] = {"stored_weights": numel, "compressed_weights": compressed_numel,
                           "original_tensor_bytes": original_bytes,
                           "all_weights_logical_bits_per_weight": logical_bits/numel,
                           "all_weights_payload_bytes": payload_bytes}
    _json(output/"manifest.json", manifest)
    return manifest["summary"]


def restore_checkpoint(source, output, shard_mb=512):
    import torch
    from safetensors.torch import load_file, save_file
    source = Path(source)
    meta = json.loads((source/"manifest.json").read_text())
    if meta.get("format") != "squant-hf" or meta.get("version") != 1 or not meta.get("complete"):
        raise ValueError("Incomplete or unsupported checkpoint")
    if shard_mb <= 0:
        raise ValueError("shard_mb must be positive")
    output = _fresh_directory(output)
    _copy_metadata(source/"hf_metadata", output)
    tensors, chunk_bytes, total_bytes, shard_index, mapping = {}, 0, 0, 0, {}

    def flush():
        nonlocal tensors, chunk_bytes, shard_index
        if not tensors:
            return
        filename = f"model-{shard_index:05d}.safetensors"
        save_file(tensors, str(output/filename), metadata={"format": "pt"})
        mapping.update({name: filename for name in tensors})
        shard_index += 1
        tensors, chunk_bytes = {}, 0

    for name, entry in meta["tensors"].items():
        path = (source/entry["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Tensor path escapes checkpoint")
        if entry["kind"] == "squant":
            weight = torch.from_numpy(decompress(load(path))).to(getattr(torch, entry["dtype"]))
        elif entry["kind"] == "passthrough":
            weight = load_file(str(path))[name]
        else:
            raise ValueError("Unknown tensor kind")
        if list(weight.shape) != entry["shape"]:
            raise ValueError("Tensor shape mismatch")
        size = weight.numel()*weight.element_size()
        if tensors and chunk_bytes+size > shard_mb*1024**2:
            flush()
        tensors[name] = weight.contiguous()
        chunk_bytes += size
        total_bytes += size
    flush()
    _json(output/"model.safetensors.index.json", {"metadata": {"total_size": total_bytes}, "weight_map": mapping})
    return {"output": str(output), "tensors": len(mapping), "shards": shard_index, "dense_bytes": total_bytes}
