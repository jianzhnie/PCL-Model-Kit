#!/bin/bash
# Expand LongCat-Flash-Lite combined (M1+M2): depth + experts in one pass.
# Default: 14→18 layers (interleave) + 256→512 experts.
#
# Usage:
#   bash scripts/expand_longcat_lite_combined.sh
#   TARGET_LAYERS=22 TARGET_EXPERTS=768 bash scripts/expand_longcat_lite_combined.sh
#
# Environment variables (override defaults):
#   MODEL_DIR            - source model directory
#   OUTPUT_DIR           - destination directory (auto-derived if not set)
#   TARGET_LAYERS        - target layer count (default: 18 = 14+4)
#   TARGET_EXPERTS       - target expert count (default: 512 = 2×)
#   INSERTION_MODE       - interleave or append (default: interleave)
#   ROUTER_NOISE_SCALE   - Gaussian noise for router weights (default: 0.0)
#   EXPERT_NOISE_SCALE   - Gaussian noise for expert weights (default: 0.0)
#   WORKERS              - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_combined.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
TARGET_LAYERS="${TARGET_LAYERS:-}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
COPY_SOURCE="${COPY_SOURCE:-}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
WORKERS="${WORKERS:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-combined}"

echo "============================================"
echo "  LongCat-Flash-Lite Combined Expansion"
echo "  (M1 experts + M2 depth in one pass)"
echo "============================================"
echo "Model dir:         ${MODEL_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Target Layers:     ${TARGET_LAYERS:-auto (14+4=18)}"
echo "Target Experts:    ${TARGET_EXPERTS:-auto (2× = 512)}"
echo "Insertion Mode:    ${INSERTION_MODE}"
echo "Router Noise:      ${ROUTER_NOISE_SCALE:-0.0}"
echo "Expert Noise:      ${EXPERT_NOISE_SCALE:-0.0}"
echo "Workers:           ${WORKERS}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

ORIG_EXPERTS=$(python3 -c "import json; print(json.load(open('${MODEL_DIR}/config.json')).get('n_routed_experts', 0))")
ORIG_LAYERS=$(python3 -c "import json; c=json.load(open('${MODEL_DIR}/config.json')); print(c.get('num_layers', c.get('num_hidden_layers', 0)))")
ACTUAL_TARGET_EXPERTS="${TARGET_EXPERTS:-$((ORIG_EXPERTS * 2))}"
ACTUAL_TARGET_LAYERS="${TARGET_LAYERS:-$((ORIG_LAYERS + 4))}"
EXPANSION_FACTOR=$(python3 -c "print(f'{${ACTUAL_TARGET_EXPERTS} / ${ORIG_EXPERTS}:.0f}' if ${ACTUAL_TARGET_EXPERTS} % ${ORIG_EXPERTS} == 0 else f'{${ACTUAL_TARGET_EXPERTS} / ${ORIG_EXPERTS}:.2f}')")
echo "Experts:           ${ORIG_EXPERTS} → ${ACTUAL_TARGET_EXPERTS} (${EXPANSION_FACTOR}×)"
echo "Layers:            ${ORIG_LAYERS} → ${ACTUAL_TARGET_LAYERS} (+$((ACTUAL_TARGET_LAYERS - ORIG_LAYERS)) identity layers)"

echo ""
CMD=(
    env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --insertion_mode "$INSERTION_MODE"
)

[ -n "$TARGET_LAYERS" ] && CMD+=(--target_layers "$TARGET_LAYERS")
[ -n "$TARGET_EXPERTS" ] && CMD+=(--target_experts "$TARGET_EXPERTS")
[ -n "$COPY_SOURCE" ] && CMD+=(--copy_source "$COPY_SOURCE")
[ -n "$ROUTER_NOISE_SCALE" ] && CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[ -n "$EXPERT_NOISE_SCALE" ] && CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[ -n "$WORKERS" ] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
echo ""
echo "To verify (combined mode):"
echo "  bash scripts/verify_expanded_weights.sh combined ${MODEL_DIR} ${OUTPUT_DIR} --orig_layers 14 --target_layers ${TARGET_LAYERS:-18} --insertion_mode ${INSERTION_MODE}"
