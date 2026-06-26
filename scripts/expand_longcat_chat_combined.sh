#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_combined.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-combined}"
TARGET_LAYERS="${TARGET_LAYERS:-32}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
COPY_SOURCE="${COPY_SOURCE:-7,14,21,27}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-}"
EXPERT_NOISE_SCALE="${EXPERT_NOISE_SCALE:-}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

ORIG_EXPERTS=$(python3 -c "import json; print(json.load(open('${MODEL_DIR}/config.json')).get('n_routed_experts', 0))")
ORIG_LAYERS=$(python3 -c "import json; c=json.load(open('${MODEL_DIR}/config.json')); print(c.get('num_layers', c.get('num_hidden_layers', 0)))")
ACTUAL_TARGET_EXPERTS="${TARGET_EXPERTS:-$((ORIG_EXPERTS * 2))}"
EXPANSION_FACTOR=$(python3 -c "print(f'{${ACTUAL_TARGET_EXPERTS} / ${ORIG_EXPERTS}:.0f}' if ${ACTUAL_TARGET_EXPERTS} % ${ORIG_EXPERTS} == 0 else f'{${ACTUAL_TARGET_EXPERTS} / ${ORIG_EXPERTS}:.2f}')")

echo "=== LongCat-Flash-Chat Combined Expansion (M1+M2) ==="
echo "  Input:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Layers:  ${ORIG_LAYERS} → ${TARGET_LAYERS} (+$((TARGET_LAYERS - ORIG_LAYERS)) identity layers, ${INSERTION_MODE})"
echo "  Experts: ${ORIG_EXPERTS} → ${ACTUAL_TARGET_EXPERTS} (${EXPANSION_FACTOR}×)"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --insertion_mode "$INSERTION_MODE"
)

[[ -n "$TARGET_LAYERS" ]] && CMD+=(--target_layers "$TARGET_LAYERS")
[[ -n "$TARGET_EXPERTS" ]] && CMD+=(--target_experts "$TARGET_EXPERTS")
[[ -n "$COPY_SOURCE" ]] && CMD+=(--copy_source "$COPY_SOURCE")
[[ -n "$ROUTER_NOISE_SCALE" ]] && CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[[ -n "$EXPERT_NOISE_SCALE" ]] && CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh combined \"$MODEL_DIR\" \"$OUTPUT_DIR\" --orig_layers 28 --target_layers ${TARGET_LAYERS:-32} --insertion_mode ${INSERTION_MODE}"
