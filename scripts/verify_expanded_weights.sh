#!/bin/bash
# Verify expanded model weights against the original model.
#
# Usage:
#   bash scripts/verify_expanded_weights.sh [type] [orig_dir] [exp_dir] [args...]
#
# Expansion Types:
#   layers   - Verify model depth expansion (requires --orig_layers, --target_layers)
#   experts  - Verify MoE expert expansion
#   combined - Verify combined depth + expert expansion (requires --orig_layers, --target_layers)
#
# Examples:
#   # Verify 14 -> 28 layer expansion (interleave)
#   bash scripts/verify_expanded_weights.sh layers /path/to/orig /path/to/exp --orig_layers 14 --target_layers 28 --insertion_mode interleave
#
#   # Verify MoE expert expansion
#   bash scripts/verify_expanded_weights.sh experts /path/to/orig /path/to/exp
#
#   # Verify combined depth + expert expansion
#   bash scripts/verify_expanded_weights.sh combined /path/to/orig /path/to/exp --orig_layers 14 --target_layers 18 --insertion_mode interleave

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERIFY_SCRIPT="$PROJECT_ROOT/utils/verify_expanded_weights.py"

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

env PYTHONPATH="$PROJECT_ROOT" python3 "$VERIFY_SCRIPT" \
    --type "$TYPE" \
    --orig_dir "$ORIG_DIR" \
    --exp_dir "$EXP_DIR" \
    "$@"
