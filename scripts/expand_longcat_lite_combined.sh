#!/usr/bin/env bash
# Expand LongCat-Flash-Lite combined (M1+M2): depth + experts in one pass.
#
# Usage:
#   bash scripts/expand_longcat_lite_combined.sh
#   EXPERT_EXPANSION_FACTOR=3 bash scripts/expand_longcat_lite_combined.sh
#   TARGET_LAYERS=22 TARGET_EXPERTS=768 \
#       bash scripts/expand_longcat_lite_combined.sh
#   COPY_SOURCE="3,6,9,12" bash scripts/expand_longcat_lite_combined.sh
#
# Environment variables:
#   MODEL_DIR               - source model directory
#   OUTPUT_DIR              - destination directory
#   TARGET_LAYERS           - target layer count (default: auto = orig+4)
#   TARGET_EXPERTS          - target expert count (overrides factor)
#   EXPERT_EXPANSION_FACTOR - expansion multiplier (default: 2)
#   COPY_SOURCE             - source mapping (default: seq)
#   INSERTION_MODE          - interleave or append (default: interleave)
#   ROUTER_NOISE_SCALE      - Gaussian noise for router (default: 0)
#   EXPERT_NOISE_SCALE      - Gaussian noise for experts (default: 0)
#   WORKERS                 - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_combined.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-combined}"
TARGET_LAYERS="${TARGET_LAYERS:-}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
EXPERT_EXPANSION_FACTOR="${EXPERT_EXPANSION_FACTOR:-2}"
COPY_SOURCE="${COPY_SOURCE:-}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
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
ORIG_LAYERS=$(python3 -c "
import json
c = json.load(open('${MODEL_DIR}/config.json'))
print(c.get('num_layers', c.get('num_hidden_layers', 0)))
")
ACTUAL_TARGET_EXPERTS="${TARGET_EXPERTS:-$((ORIG_EXPERTS * EXPERT_EXPANSION_FACTOR))}"
ACTUAL_TARGET_LAYERS="${TARGET_LAYERS:-$((ORIG_LAYERS + 4))}"
EXPANSION_FACTOR=$(python3 -c "
n = ${ACTUAL_TARGET_EXPERTS} / ${ORIG_EXPERTS}
print(f'{n:.0f}' if ${ACTUAL_TARGET_EXPERTS} % ${ORIG_EXPERTS} == 0 else f'{n:.2f}')
")

echo "=== LongCat-Flash-Lite Combined Expansion (M1+M2) ==="
echo "  Input:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Layers:  ${ORIG_LAYERS} → ${ACTUAL_TARGET_LAYERS}" \
     "(+$((ACTUAL_TARGET_LAYERS - ORIG_LAYERS)) identity, ${INSERTION_MODE})"
echo "  Experts: ${ORIG_EXPERTS} → ${ACTUAL_TARGET_EXPERTS}" \
     "(${EXPANSION_FACTOR}×)"
echo "  Source:  ${COPY_SOURCE:-seq}"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --insertion_mode "$INSERTION_MODE"
    --target_experts "$ACTUAL_TARGET_EXPERTS"
)

[[ -n "$TARGET_LAYERS" ]] && CMD+=(--target_layers "$TARGET_LAYERS")
[[ -n "$COPY_SOURCE" ]] && CMD+=(--copy_source "$COPY_SOURCE")
[[ -n "$ROUTER_NOISE_SCALE" ]] && \
    CMD+=(--router-noise-scale "$ROUTER_NOISE_SCALE")
[[ -n "$EXPERT_NOISE_SCALE" ]] && \
    CMD+=(--expert-noise-scale "$EXPERT_NOISE_SCALE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh combined \\"
echo "    \"$MODEL_DIR\" \\"
echo "    \"$OUTPUT_DIR\" \\"
echo "    --orig_layers ${ORIG_LAYERS}" \
     "--target_layers ${ACTUAL_TARGET_LAYERS} \\"
echo "    --copy_source \"${COPY_SOURCE:-seq}\"" \
     "--insertion_mode ${INSERTION_MODE}"
