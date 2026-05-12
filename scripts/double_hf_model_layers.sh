#!/bin/bash
# Double model layers from 28 → 56 by copying original layer weights.
#
# Usage:
#   bash scripts/double_layers.sh                      # sequential copy (default)
#   bash scripts/double_layers.sh seq                  # same as above
#   bash scripts/double_layers.sh single 0             # all new layers copy layer 0
#   bash scripts/double_layers.sh single 5             # all new layers copy layer 5
#   bash scripts/double_layers.sh list "0,0,1,1,..."   # explicit 28-entry mapping

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOUBLE_SCRIPT="${PROJECT_DIR}/double_model_layers.py"

MODEL_DIR="${MODEL_DIR:-/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-56L}"
ORIGINAL_LAYERS="${ORIGINAL_LAYERS:-28}"

MODE="${1:-seq}"
ARG="${2:-}"

echo "============================================"
echo "  Double Model Layers"
echo "============================================"
echo "Model dir:     ${MODEL_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Layers:        ${ORIGINAL_LAYERS} → $((ORIGINAL_LAYERS * 2))"
echo "Copy mode:     ${MODE}"

case "$MODE" in
    seq)
        echo "  → Sequential: layer 28←0, 29←1, …, 55←27"
        COPY_ARG=""
        ;;
    single)
        if [ -z "$ARG" ]; then
            echo "ERROR: single mode requires a source layer index, e.g.:"
            echo "  bash scripts/double_layers.sh single 0"
            exit 1
        fi
        echo "  → All ${ORIGINAL_LAYERS} new layers copy from layer ${ARG}"
        COPY_ARG="--copy_source ${ARG}"
        ;;
    list)
        if [ -z "$ARG" ]; then
            echo "ERROR: list mode requires ${ORIGINAL_LAYERS} comma-separated source indices, e.g.:"
            echo "  bash scripts/double_layers.sh list \"0,0,1,1,2,2,...\""
            exit 1
        fi
        echo "  → Custom mapping: ${ARG}"
        COPY_ARG="--copy_source ${ARG}"
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: seq | single <N> | list <N,N,…>"
        exit 1
        ;;
esac

echo ""
python3 "$DOUBLE_SCRIPT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --original_layers "$ORIGINAL_LAYERS" \
    $COPY_ARG

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"