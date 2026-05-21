#!/bin/bash
# Expand MoE experts with Grouped Expert Routing (方案一).
#
# Unlike 方案二 (direct topk expansion), this keeps moe_topk unchanged
# and adds 'use_group_routing: true' to config.json so the inference
# forward pass can apply group-then-select routing.
#
# Usage:
#   bash scripts/expand_model_experts_group.sh [target_experts]
#
# Examples:
#   bash scripts/expand_model_experts_group.sh              # double experts
#   bash scripts/expand_model_experts_group.sh 1024         # 1024 experts
#   TARGET_EXPERTS=1024 bash scripts/expand_model_experts_group.sh
#
# Environment variables (override defaults):
#   MODEL_DIR          - source model directory
#   OUTPUT_DIR         - destination directory (auto-derived if not set)
#   TARGET_EXPERTS     - target number of routed experts (default: double original)
#   TARGET_ZERO_EXPERT - target number of zero experts (default: double original)
#   NOISE_SCALE        - Gaussian noise scale for duplicated classifier weights (default: 0.0)
#   WORKERS            - number of parallel workers for output shard writing (default: 1 = serial)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_experts.py"

# Positional args override env vars; both default to empty (auto-derive in Python)
TARGET_EXPERTS="${1:-${TARGET_EXPERTS:-}}"
TARGET_ZERO_EXPERT="${TARGET_ZERO_EXPERT:-}"
NOISE_SCALE="${NOISE_SCALE:-}"
WORKERS="${WORKERS:-}"

# Default paths - update these as needed
MODEL_DIR="${MODEL_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"

# Build output directory suffix from target values
SUFFIX=""
if [ -n "$TARGET_EXPERTS" ]; then
    SUFFIX="${TARGET_EXPERTS}E"
else
    SUFFIX="2xE"
fi
if [ -n "$TARGET_ZERO_EXPERT" ]; then
    SUFFIX="${SUFFIX}-${TARGET_ZERO_EXPERT}Zero-E"
fi
SUFFIX="${SUFFIX}-Group"

OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-${SUFFIX}}"

echo "============================================"
echo "  Expand MoE Experts (Grouped Routing)"
echo "============================================"
echo "Model dir:      ${MODEL_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Target Experts: ${TARGET_EXPERTS:-auto}"
echo "Target Zero Experts: ${TARGET_ZERO_EXPERT:-auto}"
echo "Routing:        group-then-select (topk unchanged)"
echo "Noise Scale:    ${NOISE_SCALE:-0.0}"
echo "Workers:        ${WORKERS:-1}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
CMD=(
    env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --use_group_routing
)

if [ -n "$TARGET_EXPERTS" ]; then
    CMD+=(--target_experts "$TARGET_EXPERTS")
fi

if [ -n "$NOISE_SCALE" ]; then
    CMD+=(--noise-scale "$NOISE_SCALE")
fi

if [ -n "$WORKERS" ]; then
    CMD+=(--workers "$WORKERS")
fi

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Verify: bash scripts/verify_expanded_weights.sh experts ${MODEL_DIR} ${OUTPUT_DIR}"
echo "  2. The model forward pass must implement group-then-select routing:"
echo "     - Group N*F experts into N groups of F copies each"
echo "     - Select best expert within each group"
echo "     - Then select topk from N group winners"
