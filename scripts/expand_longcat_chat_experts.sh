#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_experts.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-expertx2}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

ORIG_EXPERTS=$(python3 -c "import json; print(json.load(open('${MODEL_DIR}/config.json')).get('n_routed_experts', 0))")
ACTUAL_TARGET="${TARGET_EXPERTS:-$((ORIG_EXPERTS * 2))}"
EXPANSION_FACTOR=$(python3 -c "print(f'{${ACTUAL_TARGET} / ${ORIG_EXPERTS}:.0f}' if ${ACTUAL_TARGET} % ${ORIG_EXPERTS} == 0 else f'{${ACTUAL_TARGET} / ${ORIG_EXPERTS}:.2f}')")

echo "=== LongCat-Flash-Chat Expert Expansion (M1) ==="
echo "  Input:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Experts: ${ORIG_EXPERTS} → ${ACTUAL_TARGET} (${EXPANSION_FACTOR}×)"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
)

[[ -n "$TARGET_EXPERTS" ]] && CMD+=(--target_experts "$TARGET_EXPERTS")
[[ -n "$ROUTER_NOISE_SCALE" ]] && CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[[ -n "$EXPERT_NOISE_SCALE" ]] && CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh experts \"$MODEL_DIR\" \"$OUTPUT_DIR\""
