# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 请根据实际并行策略修改参数
# 如果 PP > 1 且 layer 数不能整除 PP，需要指定 --num-layer-list
python examples/mcore/kimi2/convert_ckpt_kimi2_mcore2hf.py \
    --rotary-base 50000 \
    --moe-grouped-gemm \
    --source-tensor-parallel-size 2 \
    --source-pipeline-parallel-size 8 \
    --source-expert-parallel-size 32 \
    --load-dir /home/robin/hfhub/output/ckpt/k8s_pretrain_kimi2_1t_4k_A3_ptd_1024_dies \
    --save-dir /home/robin/hfhub/models/moonshotai/Kimi-K2-Base-mcore-2-hf_v2 \
    --num-layers 64 \
    --first-k-dense-replace 1 \
    --noop-layers 61,62,63
    # --num-layer-list, --moe-tp-extend-ep 等参数根据任务需要进行配置
