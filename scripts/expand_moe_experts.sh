#!/bin/bash
# Expand MoE experts in a sharded safetensors model.
#
# Usage:
#   bash scripts/expand_moe_experts.sh [target_experts]
#
# Examples:
#   bash scripts/expand_moe_experts.sh 1024
#   MODEL_DIR=/path/to/model bash scripts/expand_moe_experts.sh 1024

set -euo pipefail

EXPAND_SCRIPT="$(dirname "$0")/../utils/expand_moe_experts.py"

# Default paths - update these as needed
MODEL_DIR="${MODEL_DIR:-/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}-ExpandedExperts}"
TARGET_EXPERTS="${1:-1024}"

echo "============================================"
echo "  Expand MoE Experts"
echo "============================================"
echo "Model dir:      ${MODEL_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Target Experts: ${TARGET_EXPERTS}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
python3 "$EXPAND_SCRIPT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_experts "$TARGET_EXPERTS"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
