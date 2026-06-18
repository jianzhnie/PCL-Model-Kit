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
#   TARGET_EXPERTS=1024 TARGET_TOPK=24 bash scripts/expand_moe_experts.sh
#
# Environment variables (override defaults):
#   MODEL_DIR            - source model directory
#   OUTPUT_DIR           - destination directory (auto-derived if not set)
#   TARGET_EXPERTS       - target number of routed experts (default: double original)
#   TARGET_TOPK          - target moe_topk (default: unchanged)
#   ROUTER_NOISE_SCALE   - Gaussian noise scale for duplicated router weights (default: 0.0)
#   EXPERT_NOISE_SCALE   - Gaussian noise scale for duplicated expert weights (default: 0.0)
#   WORKERS              - number of parallel workers for output shard writing (default: 1 = serial)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_experts.py"

# Positional args override env vars; both default to empty (auto-derive in Python)
TARGET_EXPERTS="${1:-${TARGET_EXPERTS:-}}"
TARGET_TOPK="${2:-${TARGET_TOPK:-}}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
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
if [ -n "$TARGET_TOPK" ]; then
    SUFFIX="${SUFFIX}-Topk${TARGET_TOPK}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-${SUFFIX}}"

echo "============================================"
echo "  Expand MoE Experts"
echo "============================================"
echo "Model dir:         ${MODEL_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Target Experts:    ${TARGET_EXPERTS:-auto (2x)}"
echo "Target Topk:       ${TARGET_TOPK:-auto (unchanged)}"
echo "Router Noise:      ${ROUTER_NOISE_SCALE:-0.0}"
echo "Expert Noise:      ${EXPERT_NOISE_SCALE:-0.0}"
echo "Workers:           ${WORKERS:-1}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo ""
CMD=(
    env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
)

if [ -n "$TARGET_EXPERTS" ]; then
    CMD+=(--target_experts "$TARGET_EXPERTS")
fi

if [ -n "$TARGET_TOPK" ]; then
    CMD+=(--target_topk "$TARGET_TOPK")
fi

if [ -n "$ROUTER_NOISE_SCALE" ]; then
    CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
fi

if [ -n "$EXPERT_NOISE_SCALE" ]; then
    CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
fi

if [ -n "$WORKERS" ]; then
    CMD+=(--workers "$WORKERS")
fi

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
