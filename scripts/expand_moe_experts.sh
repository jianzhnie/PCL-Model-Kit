#!/bin/bash
# Expand MoE experts in a sharded safetensors model.
#
# Usage:
#   bash scripts/expand_moe_experts.sh [target_experts] [target_topk]
#
# Examples:
#   bash scripts/expand_moe_experts.sh                  # double experts, keep topk
#   bash scripts/expand_moe_experts.sh 1024             # 1024 experts, keep topk
#   bash scripts/expand_moe_experts.sh 1024 24          # 1024 experts, topk=24
#   MODEL_DIR=/path/to/model bash scripts/expand_moe_experts.sh 1024 24

set -euo pipefail

EXPAND_SCRIPT="$(dirname "$0")/../utils/expand_moe_experts.py"

TARGET_EXPERTS="${1:-}"
TARGET_TOPK="${2:-}"
# Default paths - update these as needed
MODEL_DIR="${MODEL_DIR:-/mnt/xufan_400T/models/LongCat-Flash-Chat}"
SUFFIX="Experts"
[ -n "$TARGET_EXPERTS" ] && SUFFIX="${SUFFIX}-${TARGET_EXPERTS}"
[ -n "$TARGET_TOPK" ] && SUFFIX="${SUFFIX}-Topk${TARGET_TOPK}"
OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-${SUFFIX}}"

echo "============================================"
echo "  Expand MoE Experts"
echo "============================================"
echo "Model dir:      ${MODEL_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Target Experts: ${TARGET_EXPERTS:-auto (double the original)}"
echo "Target Topk:    ${TARGET_TOPK:-auto (unchanged)}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
CMD=(
    python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
)

if [ -n "$TARGET_EXPERTS" ]; then
    CMD+=(--target_experts "$TARGET_EXPERTS")
fi

if [ -n "$TARGET_TOPK" ]; then
    CMD+=(--target_topk "$TARGET_TOPK")
fi

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
