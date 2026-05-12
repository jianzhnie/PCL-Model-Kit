#!/usr/bin/env -S env -u BASH_ENV bash
# =============================================================================
# Kimi2-1T-4k Megatron-Core (MCore) 到 HuggingFace (HF) 模型权重转换脚本
#
# 使用 convert_kimi2_mcore2hf.py 进行转换
#
# 模型配置:
#   - 32 层 Transformer
#   - Hidden size: 7168
#   - Attention heads: 64 (Q) / 2 (KV) - GQA
#   - MoE: 128 experts, 前 2 层为 Dense
#   - Vocab size: 163840
#
# 支持并行模式:
#   1. DualPipeV: SCHEDULES_METHOD=dualpipev (默认)
#   2. 标准 VPP: VPP_STAGE=N
#   3. 纯 PP: 不设 SCHEDULES_METHOD 和 VPP_STAGE
#
# 使用示例:
#   # 模式1: DualPipeV (与训练脚本一致)
#   bash scripts/ckpt_convert_kimi2_mcore2hf.sh
#
#   # 模式2: 标准 VPP (PP=4, 每 vpp stage 4 层)
#   VPP_STAGE=4 PP=4 bash scripts/ckpt_convert_kimi2_mcore2hf.sh
#
#   # 模式3: 纯 PP (PP=8, 无 VPP)
#   SCHEDULES_METHOD="" bash scripts/ckpt_convert_kimi2_mcore2hf.sh
#
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

REPO_ROOT="${REPO_ROOT:-"/llm_workspace_1P/robin/Kimi2-PCL"}"

# 路径配置
ckpt_dir="/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2_L32_exp_4096_dies/da943594-a2aa-4999-8729-4d935c0bfbfc"
LOAD_DIR="${LOAD_DIR:-${ckpt_dir}}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf_step10000}"

if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source MCore checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target HuggingFace format output directory)" >&2
  exit 1
fi

# 并行配置 (来自 pretrain_kimi2_1t_4k.sh)
# 注意: 训练脚本使用 --expert-model-parallel-size 64, 但 moe_tp_extend_ep=True 时
# EP group 包含 TP 维度, 纯 EP = 64/TP = 32, 转换脚本需要使用纯 EP 值
TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-32}"
VPP_STAGE="${VPP_STAGE:-}"
SCHEDULES_METHOD="${SCHEDULES_METHOD:-dualpipev}"

# 模型架构配置
NUM_LAYERS="${NUM_LAYERS:-32}"
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
KV_CHANNELS="${KV_CHANNELS:-128}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"
VOCAB_SIZE="${VOCAB_SIZE:-163840}"

# 可选配置
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
MOE_TP_EXTEND_EP="${MOE_TP_EXTEND_EP:-1}"
EXPERT_TP="${EXPERT_TP:-1}"
NOOP_LAYERS="${NOOP_LAYERS:-}"
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"
QK_LAYERNORM="${QK_LAYERNORM:-1}"

# 检查源目录
if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 创建输出目录
mkdir -p "${SAVE_DIR}"

# 检查转换脚本
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_kimi2_mcore2hf.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: Conversion script not found: ${CONVERT_SCRIPT}" >&2
  exit 3
fi

# 确定并行模式
MODE="pure_pp"
if [[ -n "${SCHEDULES_METHOD}" ]]; then
  MODE="dualpipe"
  if [[ -n "${VPP_STAGE}" ]]; then
    echo "ERROR: dualpipev 与 --vpp-stage 不兼容" >&2
    exit 5
  fi
elif [[ -n "${VPP_STAGE}" ]]; then
  MODE="standard_vpp"
fi

echo "==========================================="
echo " MCore -> HF 权重转换 (Kimi2-1T-4k)"
echo "==========================================="
echo "  模式: ${MODE}"
echo "  LOAD_DIR:  ${LOAD_DIR}"
echo "  SAVE_DIR:  ${SAVE_DIR}"
echo "  TP=${TP}, PP=${PP}, EP=${EP}"
if [[ "${MODE}" == "standard_vpp" ]]; then
  echo "  VPP_STAGE=${VPP_STAGE}"
fi
echo "  NUM_EXPERTS=${NUM_EXPERTS}"
echo "==========================================="
echo ""

# 构建转换参数
EXTRA_ARGS=()
if [[ "${MODE}" == "dualpipe" ]]; then
  EXTRA_ARGS+=(--schedules-method "${SCHEDULES_METHOD}")
fi
if [[ "${MODE}" == "standard_vpp" ]]; then
  EXTRA_ARGS+=(--num-layers-per-virtual-pipeline-stage "${VPP_STAGE}")
fi
if [[ -n "${NUM_LAYER_LIST}" ]]; then
  EXTRA_ARGS+=(--num-layer-list "${NUM_LAYER_LIST}")
fi
if [[ -n "${NOOP_LAYERS}" ]]; then
  EXTRA_ARGS+=(--noop-layers "${NOOP_LAYERS}")
fi
if [[ "${MOE_GROUPED_GEMM}" == "1" ]]; then
  EXTRA_ARGS+=(--moe-grouped-gemm)
fi
if [[ "${MOE_TP_EXTEND_EP}" == "1" ]]; then
  EXTRA_ARGS+=(--moe-tp-extend-ep)
fi
if [[ "${EXPERT_TP}" != "1" ]]; then
  EXTRA_ARGS+=(--expert-tensor-parallel-size "${EXPERT_TP}")
fi
if [[ "${QK_LAYERNORM}" == "1" ]]; then
  EXTRA_ARGS+=(--qk-layernorm)
fi

python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --source-tensor-parallel-size "${TP}" \
  --source-pipeline-parallel-size "${PP}" \
  --source-expert-parallel-size "${EP}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --num-experts "${NUM_EXPERTS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --num-query-groups "${NUM_QUERY_GROUPS}" \
  --kv-channels "${KV_CHANNELS}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "转换完成! 输出目录: ${SAVE_DIR}"
