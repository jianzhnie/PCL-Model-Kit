#!/bin/bash
# Expand LongCat-Flash-Lite MoE experts (M1): 256 → 512 routed experts.
#
# Usage:
#   bash scripts/expand_longcat_lite_experts.sh
#   TARGET_EXPERTS=768 bash scripts/expand_longcat_lite_experts.sh   # 3× experts
#
# Environment variables (override defaults):
#   MODEL_DIR            - source model directory
#   OUTPUT_DIR           - destination directory (auto-derived if not set)
#   TARGET_EXPERTS       - target number of routed experts (default: 512 = 2×)
#   ROUTER_NOISE_SCALE   - Gaussian noise for router weights (default: 0.0)
#   EXPERT_NOISE_SCALE   - Gaussian noise for expert weights (default: 0.0)
#   WORKERS              - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_experts.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
WORKERS="${WORKERS:-4}"

SUFFIX=""
if [ -n "$TARGET_EXPERTS" ]; then
    SUFFIX="${TARGET_EXPERTS}E"
else
    SUFFIX="expertx2"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-${SUFFIX}}"

echo "============================================"
echo "  LongCat-Flash-Lite Expert Expansion (M1)"
echo "============================================"
echo "Model dir:         ${MODEL_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Target Experts:    ${TARGET_EXPERTS:-auto (2× = 512)}"
echo "Router Noise:      ${ROUTER_NOISE_SCALE:-0.0}"
echo "Expert Noise:      ${EXPERT_NOISE_SCALE:-0.0}"
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
)

[ -n "$TARGET_EXPERTS" ] && CMD+=(--target_experts "$TARGET_EXPERTS")
[ -n "$ROUTER_NOISE_SCALE" ] && CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[ -n "$EXPERT_NOISE_SCALE" ] && CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[ -n "$WORKERS" ] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
echo ""
echo "To verify:"
echo "  bash scripts/verify_expanded_weights.sh experts ${MODEL_DIR} ${OUTPUT_DIR}"
