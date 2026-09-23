# S-Quant

## Installation

Requires Linux, an NVIDIA GPU, and Python 3.10+. Use an environment with CUDA-enabled PyTorch installed:

```bash
cd S-Quant
pip install -e '.[llm,search]'
CUDA_VISIBLE_DEVICES=0 python -m squant gpu-check
```

`gpu-check` verifies compression, reconstruction, and a forward pass on the GPU. If CUDA is unavailable, check `nvidia-smi` and install a CUDA-enabled PyTorch build compatible with your server's driver.

## Run the Full Pipeline

Use a Hugging Face model in safetensors format. Replace `/path/to/model` with a local model directory or a Hugging Face model ID you have access to. Use a new output directory.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_pipeline.sh \
  /path/to/model results/experiment1
```

The pipeline runs GPU checks, compression, model export, and WikiText-2 perplexity evaluation for both the original and compressed models. The first evaluation requires downloading the dataset.

Outputs are saved to `results/experiment1/`: `compressed/` contains the compressed checkpoint, `restored/` contains the exported model for inference, the two `*-ppl.json` files contain evaluation results, and `*.log` files contain logs.

Default parameters are in `configs/gpu.json`. To reduce compression GPU memory usage, lower `seed_chunk`, `block_chunk`, or `cache_mb`. The exported model requires memory for dense inference. For larger models, enable automatic device mapping across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 SQUANT_EVAL_DEVICE=auto \
  bash scripts/run_gpu_pipeline.sh /path/to/model results/experiment2
```

Compression search uses the first visible GPU. Automatic multi-GPU device mapping applies to evaluation.

## Run Individual Steps

```bash
export CUDA_VISIBLE_DEVICES=0

# Compress the model
python -m squant hf-compress --model /path/to/model \
  --config configs/gpu.json --output results/compressed

# Export the model for inference
python -m squant hf-export --input results/compressed --output results/restored

# Evaluate WikiText-2 perplexity
python -m squant evaluate --model results/restored --output results/ppl.json

# Run zero-shot evaluation (optional)
pip install -e '.[eval]'
python -m squant zero-shot --model results/restored \
  --tasks arc_easy arc_challenge hellaswag winogrande boolq \
  --output results/zero-shot.json
```

The default configuration is a starting point and does not guarantee the bit widths or accuracy reported in the paper. See [PAPER_MAPPING.md](docs/PAPER_MAPPING.md) for implementation assumptions.

## Citation

```bibtex
@inproceedings{wangs,
  title={S-Quant: Rethinking Weight Quantization with Seed-Based Generation},
  author={Wang, Mingzi and Zou, Lancheng and Yin, Shuo and He, Zhuolun and Yu, Bei},
  booktitle={Forty-third International Conference on Machine Learning}
}
```
