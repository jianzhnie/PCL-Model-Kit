#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_depth.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-depth32}"
TARGET_LAYERS="${TARGET_LAYERS:-32}"
COPY_SOURCE="${COPY_SOURCE:-7,14,21,27}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "=== LongCat-Flash-Chat Depth Expansion (M2) ==="
echo "  Input:  $MODEL_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Layers: 28 -> $TARGET_LAYERS ($INSERTION_MODE)"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --target_layers "$TARGET_LAYERS"
    --insertion_mode "$INSERTION_MODE"
)

[[ -n "$COPY_SOURCE" ]] && CMD+=(--copy_source "$COPY_SOURCE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh layers \"$MODEL_DIR\" \"$OUTPUT_DIR\" --orig_layers 28 --target_layers ${TARGET_LAYERS} --insertion_mode ${INSERTION_MODE}"
