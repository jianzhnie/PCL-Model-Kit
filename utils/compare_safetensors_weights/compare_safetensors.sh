#!/usr/bin/env bash
set -euo pipefail

home_dir="/home/robin/hfhub/models/moonshotai"

python compare_safetensors_single.py \
    --source $home_dir/Kimi-K2-Base/model-1-of-61.safetensors \
    --target $home_dir/Kimi-K2-Base-mcore-2-hf/model-00001-of-000061.safetensors \
    --key model.layers.1.mlp.gate.e_score_correction_bias \
    --tolerance 1e-5 \
    --verbose

# 示例：比较两个目录（支持分片匹配）
# python compare_safetensors.py \
#   --source-dir $home_dir/Kimi-K2-Base/ \
#   --target-dir $home_dir/Kimi-K2-Base-mcore-2-hf/ \
#   --match-mode shard \
#   --jobs 8 \
#   --inner-jobs 16
