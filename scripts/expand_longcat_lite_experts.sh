#!/usr/bin/env bash
# Expand LongCat-Flash-Lite MoE experts (M1): N → K×N routed experts.
#
# Usage:
#   bash scripts/expand_longcat_lite_experts.sh
#   EXPERT_EXPANSION_FACTOR=3 bash scripts/expand_longcat_lite_experts.sh
#   TARGET_EXPERTS=768 bash scripts/expand_longcat_lite_experts.sh
#
# Environment variables:
#   MODEL_DIR               - source model directory
#   OUTPUT_DIR              - destination directory
#   EXPERT_EXPANSION_FACTOR - expansion multiplier (default: 2)
#   TARGET_EXPERTS          - target expert count (overrides factor)
#   ROUTER_NOISE_SCALE      - Gaussian noise for router (default: 0)
#   EXPERT_NOISE_SCALE      - Gaussian noise for experts (default: 0)
#   WORKERS                 - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_experts.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-expertx2}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
EXPERT_EXPANSION_FACTOR="${EXPERT_EXPANSION_FACTOR:-2}"
TARGET_TOPK="${TARGET_TOPK:-}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

ORIG_EXPERTS=$(python3 -c "
import json
print(json.load(open('${MODEL_DIR}/config.json')).get('n_routed_experts', 0))
")
ACTUAL_TARGET="${TARGET_EXPERTS:-$((ORIG_EXPERTS * EXPERT_EXPANSION_FACTOR))}"
EXPANSION_FACTOR=$(python3 -c "
n = ${ACTUAL_TARGET} / ${ORIG_EXPERTS}
print(f'{n:.0f}' if ${ACTUAL_TARGET} % ${ORIG_EXPERTS} == 0 else f'{n:.2f}')
")

echo "=== LongCat-Flash-Lite Expert Expansion (M1) ==="
echo "  Input:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Experts: ${ORIG_EXPERTS} → ${ACTUAL_TARGET} (${EXPANSION_FACTOR}×)"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --target_experts "$ACTUAL_TARGET"
)

[[ -n "$TARGET_TOPK" ]] && CMD+=(--target_topk "$TARGET_TOPK")
[[ -n "$ROUTER_NOISE_SCALE" ]] && \
    CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[[ -n "$EXPERT_NOISE_SCALE" ]] && \
    CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh experts \\"
echo "    \"$MODEL_DIR\" \\"
echo "    \"$OUTPUT_DIR\""
