#!/usr/bin/env -S env -u BASH_ENV bash
# =============================================================================
# Megatron-Core (MCore) 到 Huggingface (HF) 模型权重转换脚本
#
# 基于 Kimi2-1T 模型架构配置
# 参考: scripts/pretrain_kimi2_1t_4k.sh
#
# 模型配置 (与 pretrain_kimi2_1t_4k.sh 保持一致):
#   - 32 层 Transformer
#   - Hidden size: 7168
#   - MoE: 128 experts, 前 2 层为 Dense
#   - Vocab size: 163840
#
# 默认并行配置 (与训练脚本一致):
#   - TP (Tensor Parallel): 2
#   - PP (Pipeline Parallel): 8
#   - EP (Expert Parallel): 64
# =============================================================================

set -euo pipefail

if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

# 可选的昇腾环境设置（如果存在）
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 默认路径配置
REPO_ROOT="${REPO_ROOT:-"/llm_workspace_1P/robin/Kimi2-PCL"}"
ckpt_dir="/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2_L32_exp_4096_dies/b834d725-34df-47ca-ab07-4b93a36b9e87/"
LOAD_DIR="${LOAD_DIR:-$ckpt_dir}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf}"

# 检查必要参数
if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source MCore checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target Hugging Face format output directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

# =============================================================================
# 并行配置 (与训练脚本 pretrain_kimi2_1t_4k.sh 保持一致)
# =============================================================================
TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-64}"

# =============================================================================
# 模型架构配置 (与 pretrain_kimi2_1t_4k.sh 保持一致)
# =============================================================================
NUM_LAYERS="${NUM_LAYERS:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"
VOCAB_SIZE="${VOCAB_SIZE:-163840}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"           # GQA 分组数 (对应 --num-query-groups)
KV_CHANNELS="${KV_CHANNELS:-128}"                   # kv-channels
QK_LAYERNORM="${QK_LAYERNORM:-true}"                # qk-layernorm 开关
ROTARY_BASE="${ROTARY_BASE:-50000}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"

# MoE 配置 (与 MOE_ARGS 保持一致)
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
N_SHARED_EXPERTS="${N_SHARED_EXPERTS:-1}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"
EXPERT_TP="${EXPERT_TP:-1}"                  # Expert tensor parallel size (训练使用 expert-tp=1)

# VPP / DualPipe 配置 (与训练脚本保持一致)
SCHEDULES_METHOD="${SCHEDULES_METHOD:-dualpipev}"
VPP_STAGE="${VPP_STAGE:-2}"

# =============================================================================
# 检查输入输出路径
# =============================================================================
if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 创建输出目录
mkdir -p "${SAVE_DIR}" || {
  echo "ERROR: Failed to create SAVE_DIR: ${SAVE_DIR}" >&2
  exit 3
}

# 检查转换脚本
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_ckpt_mcore2hf.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: Conversion script not found: ${CONVERT_SCRIPT}" >&2
  exit 4
fi

# =============================================================================
# 构建额外参数 (与训练脚本保持一致)
# =============================================================================
EXTRA_ARGS=()

# MoE Grouped GEMM (MOE_ARGS 中启用)
EXTRA_ARGS+=(--moe-grouped-gemm)

# QK LayerNorm (GQA_ARGS 中启用)
EXTRA_ARGS+=(--qk-layernorm)

# DualPipe 调度与 VPP (与训练脚本保持一致)
EXTRA_ARGS+=(--schedules-method "${SCHEDULES_METHOD}")

# =============================================================================
# 打印配置信息
# =============================================================================
echo "============================================================"
echo "MCore -> HF 模型转换"
echo "============================================================"
echo ""
echo "输入/输出:"
echo "  LOAD_DIR: ${LOAD_DIR}"
echo "  SAVE_DIR: ${SAVE_DIR}"
echo ""
echo "并行配置:"
echo "  TP: ${TP}, PP: ${PP}, EP: ${EP}"
echo ""
echo "模型架构:"
echo "  NUM_LAYERS: ${NUM_LAYERS}"
echo "  HIDDEN_SIZE: ${HIDDEN_SIZE}"
echo "  FFN_HIDDEN_SIZE: ${FFN_HIDDEN_SIZE}"
echo "  VOCAB_SIZE: ${VOCAB_SIZE}"
echo "  NUM_ATTENTION_HEADS: ${NUM_ATTENTION_HEADS}"
echo "  NUM_QUERY_GROUPS: ${NUM_QUERY_GROUPS} (GQA groups)"
echo "  KV_CHANNELS: ${KV_CHANNELS}"
echo "  QK_LAYERNORM: ${QK_LAYERNORM}"
echo "  ROTARY_BASE: ${ROTARY_BASE}"
echo "  MAX_POSITION_EMBEDDINGS: ${MAX_POSITION_EMBEDDINGS}"
echo ""
echo "MoE 配置:"
echo "  FIRST_K_DENSE_REPLACE: ${FIRST_K_DENSE_REPLACE}"
echo "  NUM_EXPERTS: ${NUM_EXPERTS}"
echo "  N_SHARED_EXPERTS: ${N_SHARED_EXPERTS}"
echo "  MOE_ROUTER_TOPK: ${MOE_ROUTER_TOPK}"
echo "  MOE_FFN_HIDDEN_SIZE: ${MOE_FFN_HIDDEN_SIZE}"
echo "  MOE_GROUPED_GEMM: enabled"
echo "  EXPERT_TP: ${EXPERT_TP}"
echo ""
echo "VPP / DualPipe 配置:"
echo "  SCHEDULES_METHOD: ${SCHEDULES_METHOD}"
echo "  VPP_STAGE: ${VPP_STAGE}"
echo "============================================================"
echo ""

# =============================================================================
# 执行转换 (参数与训练脚本严格一致)
# =============================================================================
python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --source-tensor-parallel-size "${TP}" \
  --source-pipeline-parallel-size "${PP}" \
  --source-expert-parallel-size "${EP}" \
  --vpp-stage "${VPP_STAGE}" \
  --num-layers "${NUM_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --num-query-groups "${NUM_QUERY_GROUPS}" \
  --qk-head-dim "${KV_CHANNELS}" \
  --rotary-base "${ROTARY_BASE}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --num-experts "${NUM_EXPERTS}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --n-shared-experts "${N_SHARED_EXPERTS}" \
  --moe-router-topk "${MOE_ROUTER_TOPK}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
  --expert-tensor-parallel-size "${EXPERT_TP}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "============================================================"
echo "转换完成!"
echo "HF 格式模型已保存到: ${SAVE_DIR}"
echo "============================================================"