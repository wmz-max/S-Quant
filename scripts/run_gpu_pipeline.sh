#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/run_gpu_pipeline.sh MODEL OUTPUT_DIR [CONFIG]" >&2
    exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="$1"
result_dir="$2"
config="${3:-$project_dir/configs/gpu.json}"
interpreter="${SQUANT_PYTHON:-python}"
eval_device="${SQUANT_EVAL_DEVICE:-cuda:0}"

if [[ -e "$result_dir/compressed" || -e "$result_dir/restored" ]]; then
    echo "Use a new output directory; compressed/restored already exists in $result_dir" >&2
    exit 2
fi
"$interpreter" -m squant gpu-check --device cuda:0
mkdir -p "$result_dir"
"$interpreter" -m squant doctor > "$result_dir/environment.json"
"$interpreter" -m squant hf-compress --model "$model" --config "$config" --search-device cuda:0 \
    --output "$result_dir/compressed" 2>&1 | tee "$result_dir/compress.log"
"$interpreter" -m squant hf-export --input "$result_dir/compressed" \
    --output "$result_dir/restored" 2>&1 | tee "$result_dir/export.log"
"$interpreter" -m squant evaluate --model "$model" --device "$eval_device" --dtype float16 \
    --output "$result_dir/baseline-ppl.json" 2>&1 | tee "$result_dir/baseline-ppl.log"
"$interpreter" -m squant evaluate --model "$result_dir/restored" --device "$eval_device" --dtype float16 \
    --output "$result_dir/squant-ppl.json" 2>&1 | tee "$result_dir/squant-ppl.log"
