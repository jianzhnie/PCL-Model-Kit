#!/bin/bash
# Expand MoE experts in a sharded safetensors model.
#
# Usage:
#   bash scripts/expand_moe_experts.sh [target_experts]
#
# Examples:
#   bash scripts/expand_moe_experts.sh
#   bash scripts/expand_moe_experts.sh 1024
#   MODEL_DIR=/path/to/model bash scripts/expand_moe_experts.sh 1024

set -euo pipefail

EXPAND_SCRIPT="$(dirname "$0")/../utils/expand_moe_experts.py"

TARGET_EXPERTS="${1:-1024}"
# Default paths - update these as needed
MODEL_DIR="${MODEL_DIR:-/mnt/xufan_400T/models/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-ExpandedExperts-${TARGET_EXPERTS}}"

echo "============================================"
echo "  Expand MoE Experts"
echo "============================================"
echo "Model dir:      ${MODEL_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Target Experts: ${TARGET_EXPERTS:-auto (double the original)}"

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

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
