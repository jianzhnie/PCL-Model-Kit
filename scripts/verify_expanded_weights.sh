#!/bin/bash
# Verify expanded model weights against the original model.
#
# Usage:
#   bash scripts/verify_expanded_weights.sh [type] [orig_dir] [exp_dir] [args...]
#
# Expansion Types:
#   layers  - Verify model depth expansion (requires --orig_layers, --target_layers)
#   experts - Verify MoE expert expansion
#
# Examples:
#   # Verify 28 -> 56 layer expansion
#   bash scripts/verify_expanded_weights.sh layers /path/to/orig /path/to/exp --orig_layers 28 --target_layers 56
#
#   # Verify MoE expert expansion
#   bash scripts/verify_expanded_weights.sh experts /path/to/orig /path/to/exp

set -euo pipefail

VERIFY_SCRIPT="$(dirname "$0")/../utils/verify_expanded_weights.py"

TYPE="${1:-}"
ORIG_DIR="${2:-}"
EXP_DIR="${3:-}"
shift 3 || true # Shift out the first 3 positional args, keep remaining in $@

if [[ -z "$TYPE" || -z "$ORIG_DIR" || -z "$EXP_DIR" ]]; then
    echo "Usage: bash $0 <layers|experts> <orig_dir> <exp_dir> [additional_args...]"
    exit 1
fi

echo "============================================"
echo "  Verify Expanded Weights"
echo "============================================"
echo "Type:      ${TYPE}"
echo "Original:  ${ORIG_DIR}"
echo "Expanded:  ${EXP_DIR}"
echo "============================================"

python3 "$VERIFY_SCRIPT" \
    --type "$TYPE" \
    --orig_dir "$ORIG_DIR" \
    --exp_dir "$EXP_DIR" \
    "$@"
