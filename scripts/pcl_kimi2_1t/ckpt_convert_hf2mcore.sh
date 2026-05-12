#!/usr/bin/env -S env -u BASH_ENV bash
# =============================================================================
# Huggingface (HF) 到 Megatron-Core (MCore) 模型权重转换脚本
#
# 基于 Kimi2-1T 模型架构配置
# 参考: scripts/pretrain_kimi2_1t_4k.sh
#
# 模型配置:
#   - 32 层 Transformer
#   - Hidden size: 7168
#   - Attention heads: 64 (Q) / 2 (KV) - GQA
#   - MoE: 128 experts, 前 2 层为 Dense
#   - Vocab size: 163840
#
# 支持 3 种并行模式:
#   1. DualPipeV (默认): --schedules-method dualpipev
#      层分配为交替前/后取，VPP=2 自动设置
#   2. 标准 VPP: --vpp-stage N (不设 schedules-method)
#      每个虚拟 pipeline stage 包含 N 层
#   3. 纯 PP: 不设 schedules-method 和 vpp-stage
#      每个 PP rank 平分所有层
#
# 使用示例:
#   # 模式1: DualPipeV (与训练脚本一致, TP=2 PP=8 EP=64)
#   bash scripts/ckpt_convert_hf2mcore.sh
#
#   # 模式2: 标准 VPP (PP=4, 每 vpp stage 4 层, vpp_size=2)
#   SCHEDULES_METHOD="" VPP_STAGE=4 PP=4 EP=64 bash scripts/ckpt_convert_hf2mcore.sh
#
#   # 模式3: 纯 PP (PP=8, 无 VPP)
#   SCHEDULES_METHOD="" bash scripts/ckpt_convert_hf2mcore.sh
#
#   # 模式4: 单卡转换 (无并行, TP=1 PP=1 EP=1)
#   SCHEDULES_METHOD="" TP=1 PP=1 EP=1 bash scripts/ckpt_convert_hf2mcore.sh
#
#   # 自定义路径
#   LOAD_DIR=/path/to/hf SAVE_DIR=/path/to/output bash scripts/ckpt_convert_hf2mcore.sh
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

# 检查 REPO_ROOT 是否有效
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  exit 1
fi

LOAD_DIR="${LOAD_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-hf2mcore3}"

if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source HuggingFace checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/hf/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target Megatron-Core format output directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/hf/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi


# =============================================================================
# 并行配置
#
# 根据并行模式设置不同默认值:
#   DualPipeV: TP=2 PP=8 EP=64 SCHEDULES_METHOD=dualpipev
#   标准 VPP:  TP=2 PP=4 EP=64 VPP_STAGE=4
#   纯 PP:     TP=2 PP=8 EP=64
#
# 所有参数均可通过环境变量覆盖。
# =============================================================================
TP="${TP:-2}"                          # Tensor Parallel size
PP="${PP:-8}"                          # Pipeline Parallel size
EP="${EP:-8}"                         # Expert Parallel size
VPP_STAGE="${VPP_STAGE:-}"             # VPP stage 层数 (仅标准 VPP 模式使用)
PP_WORKERS="${PP_WORKERS:-2}"          # PP 并行工作进程数
IO_THREADS="${IO_THREADS:-2}"          # HF 权重加载线程数
SAVE_WORKERS="${SAVE_WORKERS:-1}"      # 保存权重线程数 (0=自动)
CAST_DTYPE="${CAST_DTYPE:-bf16}"       # 输出数据类型
EXPERT_TP="${EXPERT_TP:-1}"            # Expert tensor parallel size (训练使用 expert-tp=1)
SCHEDULES_METHOD="${SCHEDULES_METHOD:-}"  # 调度算法 (dualpipev | 留空)

# =============================================================================
# 模型架构配置 (与 models/config.json 保持一致)
# =============================================================================
NUM_LAYERS="${NUM_LAYERS:-32}"                    # 总层数
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"  # 前 N 层使用 Dense MLP
HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"                # 隐藏层维度
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"       # Dense FFN 中间维度
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"  # MoE FFN 中间维度
VOCAB_SIZE="${VOCAB_SIZE:-163840}"                # 词汇表大小
NUM_EXPERTS="${NUM_EXPERTS:-128}"                 # 专家数量
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"  # Q attention heads
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"          # KV query groups (GQA)
QK_HEAD_DIM="${QK_HEAD_DIM:-128}"                 # QK head 维度
V_HEAD_DIM="${V_HEAD_DIM:-128}"                   # V head 维度
ROTARY_BASE="${ROTARY_BASE:-50000}"               # RoPE 基数
# 可选配置
# 注意: 训练脚本使用 --untie-embeddings-and-output-weights，因此默认不绑定 embedding 和 lm_head。
# 如需绑定，请设置 TIE_WORD_EMBEDDINGS=1 并添加 --tie-word-embeddings。
TIE_WORD_EMBEDDINGS="${TIE_WORD_EMBEDDINGS:-0}"

NOOP_LAYERS="${NOOP_LAYERS:-}"           # 空层列表 (逗号分隔)
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"     # 自定义每层分配列表

if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 检查源目录是否包含必要的文件
if [[ ! -f "${LOAD_DIR}/config.json" ]]; then
  echo "WARNING: config.json not found in LOAD_DIR: ${LOAD_DIR}" >&2
  echo "  The conversion may fail if config.json is not present." >&2
fi

# 创建输出目录（如果不存在）
if [[ ! -d "${SAVE_DIR}" ]]; then
  echo "Creating SAVE_DIR: ${SAVE_DIR}"
  mkdir -p "${SAVE_DIR}" || {
    echo "ERROR: Failed to create SAVE_DIR: ${SAVE_DIR}" >&2
    exit 3
  }
fi

# 检查转换脚本是否存在
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_ckpt_hf2mcore.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: Conversion script not found: ${CONVERT_SCRIPT}" >&2
  echo "Please check REPO_ROOT setting (current: ${REPO_ROOT})" >&2
  exit 4
fi

# =============================================================================
# 确定并行模式并验证参数兼容性
# =============================================================================
MODE="pure_pp"  # 默认纯 PP
if [[ -n "${SCHEDULES_METHOD}" ]]; then
  MODE="dualpipe"
  if [[ -n "${VPP_STAGE}" ]]; then
    echo "ERROR: dualpipev 与 --vpp-stage 不兼容，请勿同时设置 SCHEDULES_METHOD 和 VPP_STAGE" >&2
    exit 5
  fi
  if [[ -n "${NUM_LAYER_LIST}" ]]; then
    echo "ERROR: dualpipev 与 --num-layer-list 不兼容" >&2
    exit 5
  fi
elif [[ -n "${VPP_STAGE}" ]]; then
  MODE="standard_vpp"
  if [[ -n "${NUM_LAYER_LIST}" ]]; then
    echo "ERROR: VPP 与 --num-layer-list 不兼容" >&2
    exit 5
  fi
  # 验证 VPP_STAGE 参数: (NUM_LAYERS / PP) 必须能被 VPP_STAGE 整除
  layers_per_pp=$((NUM_LAYERS / PP))
  if [[ $((layers_per_pp % VPP_STAGE)) -ne 0 ]]; then
    echo "ERROR: 每PP层数 (${layers_per_pp}) 必须能被 VPP_STAGE (${VPP_STAGE}) 整除" >&2
    exit 5
  fi
fi

echo "==========================================="
echo " HF -> MCore 权重转换"
echo "==========================================="
echo "  模式: ${MODE}"
echo "  LOAD_DIR:  ${LOAD_DIR}"
echo "  SAVE_DIR:  ${SAVE_DIR}"
echo "  TP=${TP}, PP=${PP}, EP=${EP}"
if [[ "${MODE}" == "standard_vpp" ]]; then
  echo "  VPP_STAGE=${VPP_STAGE} (vpp_size=$(( NUM_LAYERS / PP / VPP_STAGE )))"
fi
echo "  PP_WORKERS=${PP_WORKERS}, SAVE_WORKERS=${SAVE_WORKERS}"
echo "  EXPERT_TP=${EXPERT_TP}, CAST_DTYPE=${CAST_DTYPE}"
echo "==========================================="
echo ""

# =============================================================================
# 构建转换参数
# =============================================================================
EXTRA_ARGS=()
if [[ "${MODE}" == "dualpipe" ]]; then
  EXTRA_ARGS+=(--schedules-method "${SCHEDULES_METHOD}")
fi
if [[ "${MODE}" == "standard_vpp" ]]; then
  EXTRA_ARGS+=(--vpp-stage "${VPP_STAGE}")
fi
if [[ -n "${NUM_LAYER_LIST}" ]]; then
  EXTRA_ARGS+=(--num-layer-list "${NUM_LAYER_LIST}")
fi
QK_LAYERNORM="${QK_LAYERNORM:-1}"   # Kimi2-1T 模型启用 QK LayerNorm，默认开启

if [[ -n "${CAST_DTYPE}" ]]; then
  EXTRA_ARGS+=(--cast-dtype "${CAST_DTYPE}")
fi
if [[ "${QK_LAYERNORM:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--qk-layernorm)
fi
if [[ "${TIE_WORD_EMBEDDINGS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--tie-word-embeddings)
fi

python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --target-tensor-parallel-size "${TP}" \
  --target-pipeline-parallel-size "${PP}" \
  --target-expert-parallel-size "${EP}" \
  --pp-workers "${PP_WORKERS}" \
  --save-workers "${SAVE_WORKERS}" \
  --hf-io-threads "${IO_THREADS}" \
  --moe-grouped-gemm \
  --expert-tensor-parallel-size "${EXPERT_TP}" \
  --rotary-base "${ROTARY_BASE}" \
  --noop-layers "${NOOP_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --num-query-groups "${NUM_QUERY_GROUPS}" \
  --qk-head-dim "${QK_HEAD_DIM}" \
  --v-head-dim "${V_HEAD_DIM}" \
  --sha256-manifest "${SAVE_DIR}/sha256_manifest.json" \
  "${EXTRA_ARGS[@]}"
