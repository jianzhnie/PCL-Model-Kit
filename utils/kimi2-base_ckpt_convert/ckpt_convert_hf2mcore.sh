#!/usr/bin/env bash

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重路径需根据实际情况修改
# load-dir: huggingface 权重路径
# save-dir: mcore 权重保存路径

tp=2
pp=16
ep=32

model_dir="/home/robin/hfhub/models/moonshotai"
hf_model_dir=$model_dir/Kimi-K2-Base
save_dir=$model_dir/Kimi-K2-Base-mcore_tp${tp}_pp${pp}_ep${ep}

python examples/mcore/kimi2/convert_ckpt_kimi2.py\
    --moe-grouped-gemm \
    --target-tensor-parallel-size $tp \
    --target-pipeline-parallel-size $pp \
    --target-expert-parallel-size $ep \
    --load-dir $hf_model_dir \
    --save-dir $save_dir \
    --num-layers 64 \
    --noop-layers 61,62,63 \
    --num-dense-layers 1
    # --mtp-num-layers 0 \
    # --num-layers-per-virtual-pipeline-stage 2 \
    # --noop-layers 47,62,63 \
    # --num-layer-list, --moe-tp-extend-ep 等参数根据任务需要进行配置
