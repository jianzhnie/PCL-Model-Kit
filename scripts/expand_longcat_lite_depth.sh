#!/bin/bash
# Expand LongCat-Flash-Lite depth (M2): 14 → 28 layers with identity-init.
#
# Usage:
#   bash scripts/expand_longcat_lite_depth.sh
#   TARGET_LAYERS=21 bash scripts/expand_longcat_lite_depth.sh       # 14→21
#   INSERTION_MODE=append bash scripts/expand_longcat_lite_depth.sh   # append mode
#
# Environment variables (override defaults):
#   MODEL_DIR        - source model directory
#   OUTPUT_DIR       - destination directory (auto-derived if not set)
#   TARGET_LAYERS    - target layer count (default: 28 = 2×)
#   COPY_SOURCE      - source mapping: seq, single int, or comma list (default: seq)
#   INSERTION_MODE   - interleave or append (default: interleave)
#   WORKERS          - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_depth.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
TARGET_LAYERS="${TARGET_LAYERS:-28}"
COPY_SOURCE="${COPY_SOURCE:-}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
WORKERS="${WORKERS:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-depth2}"

echo "============================================"
echo "  LongCat-Flash-Lite Depth Expansion (M2)"
echo "============================================"
echo "Model dir:         ${MODEL_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Target Layers:     14 → ${TARGET_LAYERS}"
echo "Insertion Mode:    ${INSERTION_MODE}"
echo "Copy Source:       ${COPY_SOURCE:-seq}"
echo "Workers:           ${WORKERS}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
CMD=(
    env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --target_layers "$TARGET_LAYERS"
    --insertion_mode "$INSERTION_MODE"
)

[ -n "$COPY_SOURCE" ] && CMD+=(--copy_source "$COPY_SOURCE")
[ -n "$WORKERS" ] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
echo ""
echo "To verify:"
echo "  bash scripts/verify_expanded_weights.sh layers ${MODEL_DIR} ${OUTPUT_DIR} --orig_layers 14 --target_layers ${TARGET_LAYERS} --insertion_mode ${INSERTION_MODE}"
