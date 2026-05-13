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

DOUBLE_SCRIPT="$(dirname "$0")/../utils/expand_model_layers.py"

MODEL_DIR="${MODEL_DIR:-/mnt/xufan_400T/models/LongCat-Flash-Chat}"
ORIGINAL_LAYERS="${ORIGINAL_LAYERS:-28}"
TARGET_LAYERS="${TARGET_LAYERS:-}"

# Auto-derive output dir suffix from target layers
if [ -n "$TARGET_LAYERS" ]; then
    OUTPUT_DIR_DEFAULT="/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-${TARGET_LAYERS}L"
else
    OUTPUT_DIR_DEFAULT="/llm_workspace_1P/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat-$((ORIGINAL_LAYERS * 2))L"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

MODE="${1:-single}"
ARG="${2:-27}"

TARGET_DISPLAY="${TARGET_LAYERS:-$((ORIGINAL_LAYERS * 2))}"
NUM_NEW=$((TARGET_DISPLAY - ORIGINAL_LAYERS))

echo "============================================"
echo "  Expand Model Layers"
echo "============================================"
echo "Model dir:     ${MODEL_DIR}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Layers:        ${ORIGINAL_LAYERS} → ${TARGET_DISPLAY}"
echo "Copy mode:     ${MODE}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

COPY_ARGS=()
case "$MODE" in
    seq)
        echo "  → Sequential: layer ${ORIGINAL_LAYERS}←0, $((ORIGINAL_LAYERS+1))←1, …, $((TARGET_DISPLAY-1))←$((NUM_NEW-1))"
        ;;
    single)
        if [ -z "$ARG" ]; then
            echo "ERROR: single mode requires a source layer index, e.g.:"
            echo "  bash scripts/expand_model_layers.sh single 0"
            exit 1
        fi
        echo "  → All ${NUM_NEW} new layers copy from layer ${ARG}"
        COPY_ARGS=(--copy_source "$ARG")
        ;;
    list)
        if [ -z "$ARG" ]; then
            echo "ERROR: list mode requires ${NUM_NEW} comma-separated source indices, e.g.:"
            echo "  bash scripts/expand_model_layers.sh list \"0,0,1,1,2,2,...\""
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
CMD=(
    python3 "$DOUBLE_SCRIPT"
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --original_layers "$ORIGINAL_LAYERS"
)

if [ -n "$TARGET_LAYERS" ]; then
    CMD+=(--target_layers "$TARGET_LAYERS")
fi

if [ "${#COPY_ARGS[@]}" -gt 0 ]; then
    CMD+=("${COPY_ARGS[@]}")
fi

"${CMD[@]}"

echo ""
echo "Done. Output model at: ${OUTPUT_DIR}"
