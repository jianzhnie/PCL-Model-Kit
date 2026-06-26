#!/usr/bin/env bash
# Expand LongCat-Flash-Lite depth (M2): identity layer insertion.
#
# Usage:
#   bash scripts/expand_longcat_lite_depth.sh
#   TARGET_LAYERS=21 bash scripts/expand_longcat_lite_depth.sh
#   COPY_SOURCE="3,6,9,12" bash scripts/expand_longcat_lite_depth.sh
#   INSERTION_MODE=append bash scripts/expand_longcat_lite_depth.sh
#
# Environment variables:
#   MODEL_DIR        - source model directory
#   OUTPUT_DIR       - destination directory (auto-derived if not set)
#   TARGET_LAYERS    - target layer count (default: 28 = 2×)
#   COPY_SOURCE      - source mapping: seq, single int, or comma list (default: seq)
#   INSERTION_MODE   - interleave or append (default: interleave)
#   WORKERS          - parallel workers (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPAND_SCRIPT="$PROJECT_ROOT/utils/expand_moe_depth.py"

MODEL_DIR="${MODEL_DIR:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-depth2}"
TARGET_LAYERS="${TARGET_LAYERS:-28}"
COPY_SOURCE="${COPY_SOURCE:-}"
INSERTION_MODE="${INSERTION_MODE:-interleave}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

ORIG_LAYERS=$(python3 -c "import json; c=json.load(open('${MODEL_DIR}/config.json')); print(c.get('num_layers', c.get('num_hidden_layers', 0)))")

echo "=== LongCat-Flash-Lite Depth Expansion (M2) ==="
echo "  Input:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Layers:  ${ORIG_LAYERS} → ${TARGET_LAYERS} (+$((TARGET_LAYERS - ORIG_LAYERS)) identity layers, ${INSERTION_MODE})"
echo "  Source:  ${COPY_SOURCE:-seq}"

CMD=(env PYTHONPATH="$PROJECT_ROOT" python3 "$EXPAND_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --target_layers "$TARGET_LAYERS"
    --insertion_mode "$INSERTION_MODE"
)

[[ -n "$COPY_SOURCE" ]] && CMD+=(--copy_source "$COPY_SOURCE")
[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

"${CMD[@]}"

echo ""
echo "=== Done. Verify with: ==="
echo "bash scripts/verify_expanded_weights.sh layers \"$MODEL_DIR\" \"$OUTPUT_DIR\" --orig_layers ${ORIG_LAYERS} --target_layers ${TARGET_LAYERS} --insertion_mode ${INSERTION_MODE}"
