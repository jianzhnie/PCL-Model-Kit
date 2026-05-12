#!/bin/bash
# Double model layers from N → 2N by copying original layer weights.
#
# Usage:
#   bash scripts/expand_model_layers.sh [mode] [arg]
#
# Examples:
#   bash scripts/expand_model_layers.sh seq                  # sequential copy (default)
#   bash scripts/expand_model_layers.sh single 0             # all new layers copy layer 0
#   bash scripts/expand_model_layers.sh list "0,0,1,1,..."   # explicit mapping

set -euo pipefail

DOUBLE_SCRIPT="$(dirname "$0")/../utils/double_hf_model_layers.py"

MODEL_DIR="${MODEL_DIR:-/mnt/xufan_400T/models/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-56L}"
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

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

COPY_ARGS=()
case "$MODE" in
    seq)
        echo "  → Sequential: layer 28←0, 29←1, …, 55←27"
        ;;
    single)
        if [ -z "$ARG" ]; then
            echo "ERROR: single mode requires a source layer index, e.g.:"
            echo "  bash scripts/expand_model_layers.sh single 0"
            exit 1
        fi
        echo "  → All ${ORIGINAL_LAYERS} new layers copy from layer ${ARG}"
        COPY_ARGS=(--copy_source "$ARG")
        ;;
    list)
        if [ -z "$ARG" ]; then
            echo "ERROR: list mode requires ${ORIGINAL_LAYERS} comma-separated source indices, e.g.:"
            echo "  bash scripts/double_layers.sh list \"0,0,1,1,2,2,...\""
            exit 1
        fi
        echo "  → Custom mapping: ${ARG}"
        COPY_ARGS=(--copy_source "$ARG")
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
    "${COPY_ARGS[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
