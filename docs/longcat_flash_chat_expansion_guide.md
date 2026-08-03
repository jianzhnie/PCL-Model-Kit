# LongCat-Flash-Chat MoE 扩展指南

## 一、模型概览

| 参数 | 值 |
|------|------|
| `architectures` | `LongcatFlashForCausalLM` |
| `hidden_size` | 6144 |
| `expert_ffn_hidden_size` | 2048 |
| `num_layers` | 28 |
| `n_routed_experts` | 512 |
| `zero_expert_num` | 256 (identity 类型，无存储权重) |
| `moe_topk` | 12 |
| `num_attention_heads` | 64 |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536 |

> LongCat-Flash-Chat 的 256 个 zero expert 为 identity 类型，不在 safetensors 中存储权重参数。扩展时仅复制 routed expert 权重，zero expert 仅在 config 和 Router 维度中按比例同步扩展。

---

## 二、三种扩展方式

### 2.1 方案 M1：专家数扩展（Expert Upcycling）

#### 概述

将 512 个 routed expert 翻倍至 1024，推理激活参数不变（仍 top-12），总参数约 2×。

```bash
bash scripts/expand_longcat_chat_experts.sh
```

#### 三类张量的具体处理

**Expert 权重 (routed expert 0-511)**

映射关系由 `build_expert_target_map(512, 1024)` 构建：

```
new_idx = 512, 513, ..., 1023
src_idx = new_idx % 512 = 0, 1, ..., 511
```

每个专家的 `gate_proj.weight`、`up_proj.weight`、`down_proj.weight` 全部 `tensor.clone()`。原始 expert 0-511 保持不动，新增 expert 512-1023 是精确副本。

**Router 权重 (`mlp.router.classifier.weight`)**

```
原始 Router: [768, 6144]
             ├── real_part: tensor[:512]   (512 个 routed expert 的路由行)
             └── zero_part: tensor[512:]   (256 个 zero expert 的路由行)

扩展后:
  expanded_real = cat([real_part, real_part], dim=0)   → [1024, 6144]
  expanded_zero = cat([zero_part, zero_part], dim=0)   → [512, 6144]
  output = cat([expanded_real, expanded_zero], dim=0)  → [1536, 6144]
```

Router Bias (`e_score_correction_bias`) 同理: `[768] → [1536]`。

**非专家参数 (attention, norm, embed, lm_head)**

直接原样拷贝，不做任何修改。

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `n_routed_experts` | 512 | 1024 |
| `zero_expert_num` | 256 | 512 |
| `moe_topk` | 12 | 12 (不变) |
| `num_layers` | 28 | 28 (不变) |

#### 输出概要

```
原始: 43,756 个参数,  75 shards, 561.9 GB
扩展: 86,764 个参数, 约 148 shards (旧策略) 或 约 85 shards (--max_layers_per_shard=1), 2206.9 GB
新增: 43,008 个张量 (512 experts × 28 layers × 3 params)
      + 28 router weight 扩展 + 28 router bias 扩展
```

#### 关键特性

- **推理成本不变**: `moe_topk` 保持 12，每个 token 仍只激活 12 个专家
- **Function-preserving**: 副本与原始完全相同，Router 给同源副本相同分数，扩展后模型输出与原始模型数学等价
- **对称性未打破**: 默认不加噪声。如需后续训练分化，使用 `--router-noise-scale 1e-6 --expert-noise-scale 0.01`

---

### 2.2 方案 M2：深度扩展（Identity Layer Insertion）

#### 概述

在 28 层中均匀插入 4 个恒等初始化层，总层数 32。恒等层分布在网络的 1/4、1/2、3/4、尾部位置。

```bash
bash scripts/expand_longcat_chat_depth.sh
```

#### 恒等映射原理

新层通过将 `o_proj.weight` 和 `down_proj.weight` 置零实现恒等映射：

```
output = input + Attention(Norm(input)) + MLP(Norm(...))
       = input + 0 + 0   (因为 W_o = 0, W_down = 0)
       = input            ← 恒等映射
```

#### Interleave 模式布局

**默认 +4 层扩展 (28→32, `copy_source=7,14,21,27`)**：恒等层均匀分布

```
[L0] ... [L7] [ID←7] [L8] ... [L14] [ID←14] [L15] ... [L21] [ID←21] [L22] ... [L27] [ID←27]
```

| 扩展后索引 | 来源 | 类型 |
|:-:|:-:|:-:|
| 0-7 | orig 0-7 | 原始 |
| 8 | orig 7 | 恒等 (identity-initialized) |
| 9-15 | orig 8-14 | 原始 |
| 16 | orig 14 | 恒等 (identity-initialized) |
| 17-23 | orig 15-21 | 原始 |
| 24 | orig 21 | 恒等 (identity-initialized) |
| 25-30 | orig 22-27 | 原始 |
| 31 | orig 27 | 恒等 (identity-initialized) |

恒等层均匀分布在网络的 1/4、1/2、3/4、尾部位置，对后续训练通常比集中在前部更有利。

```bash
# 在层 7, 14, 21, 27 后面各插入一个恒等层
PYTHONPATH=/path/to/PCL-Model-Kit python3 utils/expand_moe_depth.py \
    --model_dir /path/to/LongCat-Flash-Chat \
    --output_dir /path/to/output \
    --target_layers 32 \
    --copy_source "7,14,21,27" \
    --insertion_mode interleave
```

**2× 扩展 (28→56)**：每个原始层后插入一个恒等层

```bash
TARGET_LAYERS=56 COPY_SOURCE="" bash scripts/expand_longcat_chat_depth.sh
```

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] ... [L27] [ID←27]
```

#### Append 模式布局

原始 28 层顺序不变，恒等层追加在末尾：

```bash
INSERTION_MODE=append bash scripts/expand_longcat_chat_depth.sh
```

```
[L0] [L1] ... [L27] [ID←7] [ID←14] [ID←21] [ID←27]
```

每个新层的处理：
- `self_attn.{0,1}.o_proj.weight` → 置零
- `mlp.experts.{0..511}.down_proj.weight` → 置零
- `mlps.{0,1}.down_proj.weight` → 置零
- 其余权重 → 从源层精确复制

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `num_layers` | 28 | 32 |
| 其余字段 | 不变 | 不变 |

#### 输出概要

```
扩展: 50,004 个参数, 约 86 shards (旧策略) 或 约 38 shards (--max_layers_per_shard=1), 1283.8 GB
新增恒等层: 4 层, 2,064 个张量置零
```

---

### 2.3 方案 M1+M2：联合扩展（Combined）

#### 概述

单次完成深度 + 专家扩展。默认 28→32 层（+4 层）+ 512→1024 专家。

```bash
bash scripts/expand_longcat_chat_combined.sh
```

#### 层映射

与 M2 的 28→32 interleave 完全一致，`copy_source=7,14,21,27`：

```
[L0] ... [L7] [ID←7] [L8] ... [L14] [ID←14] [L15] ... [L21] [ID←21] [L22] ... [L27] [ID←27]
```

| 扩展后索引 | 来源 | 类型 |
|:-:|:-:|:-:|
| 0-7 | orig 0-7 | KEPT (原始) |
| 8 | orig 7 | NEW (恒等) |
| 9-15 | orig 8-14 | KEPT (原始) |
| 16 | orig 14 | NEW (恒等) |
| 17-23 | orig 15-21 | KEPT (原始) |
| 24 | orig 21 | NEW (恒等) |
| 25-30 | orig 22-27 | KEPT (原始) |
| 31 | orig 27 | NEW (恒等) |

如需前置集中恒等层，可使用 `COPY_SOURCE="" bash scripts/expand_longcat_chat_combined.sh`（默认 seq 模式）。

#### 专家映射

每个原始 expert 复制出 1 个副本：`expert 0 → [0, 512]`，`expert 1 → [1, 513]`，...，`expert 511 → [511, 1023]`。共 512 对。

#### KEPT 层的张量结构（如扩展后 layer 0 ← orig 0）

| 组件 | 原始 shape | 扩展后 shape | 处理方式 |
|------|-----------|-------------|---------|
| `router.classifier.weight` | [768, 6144] | [1536, 6144] | 扩展 (见 Router 布局) |
| `router.e_score_correction_bias` | [768] | [1536] | 同上 |
| `experts.0-511.{gate,up,down}_proj` | 不变 | 不变 | 保留原值 |
| `experts.512-1023.{gate,up,down}_proj` | 新增 | 同原始 | clone(expert[i%512]) |
| `self_attn`, `layernorm`, `mlps` | 不变 | 不变 | 不变 |

#### NEW 恒等层的张量结构（如扩展后 layer 1 ← orig 0）

| 组件 | 处理方式 |
|------|---------|
| `router.classifier.weight` | 从 orig 0 的 router 扩展为 [1536, 6144] |
| `experts.0-1023.gate_proj.weight` | clone(orig 0 的 expert[i%512]) |
| `experts.0-1023.up_proj.weight` | clone(orig 0 的 expert[i%512]) |
| `experts.0-1023.down_proj.weight` | **全零** (identity init) |
| `self_attn.{0,1}.o_proj.weight` | **全零** (identity init) |
| `self_attn 其余 (q/kv proj 等)` | clone(orig 0) |
| `mlps.{0,1}.down_proj.weight` | **全零** (identity init) |
| `mlps.{0,1}.gate/up_proj` | clone(orig 0) |
| `input_layernorm`, `post_attention_layernorm` | clone(orig 0) |

#### Router 权重内部布局

```
原始 [768, 6144]:
  行 0-511:    real part (512 个 routed expert 路由权重)
  行 512-767:  zero part (256 个 zero expert 路由权重)

扩展后 [1536, 6144]:
  行 0-511:      real_block_0 (orig real part)
  行 512-1023:   real_block_1 (real part 的副本)
  行 1024-1279:  zero_block_0 (orig zero part)
  行 1280-1535:  zero_block_1 (zero part 的副本)
```

Bias 同理：`[768] → [1536]`，布局 `[real×2 | zero×2]`。

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `num_layers` | 28 | 32 |
| `n_routed_experts` | 512 | 1024 |
| `zero_expert_num` | 256 | 512 |
| `moe_topk` | 12 | 12 (不变) |

#### 输出概要

```
扩展: 99,156 个参数, 约 169 shards (旧策略) 或 约 73 shards (--max_layers_per_shard=1), 2521.3 GB
新增恒等层: 4 层, 置零参数: 4,112 个
```

---

## 三、验证方法

所有扩展输出均通过 `verify_expanded_weights.py` 验证，支持 `layers`、`experts`、`combined` 三种模式。

### 专家扩展验证

```bash
bash scripts/verify_expanded_weights.sh experts \
    /path/to/LongCat-Flash-Chat \
    /path/to/LongCat-Flash-Chat-expertx2
```

验证内容：Router shape `[1536, 6144]`、expert 0-1023 索引完整、expert 512 == expert 0 (bit-exact)、非专家参数不变。

### 深度扩展验证

```bash
bash scripts/verify_expanded_weights.sh layers \
    /path/to/LongCat-Flash-Chat \
    /path/to/LongCat-Flash-Chat-depth32 \
    --orig_layers 28 --target_layers 32 --copy_source "7,14,21,27" --insertion_mode interleave
```

验证内容：32 层结构完整、新层（8/16/24/31）`o_proj`/`down_proj` 全零、kept 层与原始层 bit-exact 匹配（含 interleave 重映射）。

### 联合扩展验证

```bash
bash scripts/verify_expanded_weights.sh combined \
    /path/to/LongCat-Flash-Chat \
    /path/to/LongCat-Flash-Chat-combined \
    --orig_layers 28 --target_layers 32 \
    --copy_source "7,14,21,27" --insertion_mode interleave
```

验证内容：同时检查层映射 + 专家复制 + 恒等初始化。

> **注意**: 如果使用了非默认的 `COPY_SOURCE`（如 `"6,13,20,26"`），验证时必须传入相同的 `--copy_source` 值，否则层映射将不匹配。

### 模型输出功能验证

`verify_model_output.py` 提供端到端的功能验证。Chat 模型较大（原始 562 GB），需要使用 `--sequential` 模式并确保 CPU 内存充足（>1 TB）。

```bash
python3 utils/verify_model_output.py \
    --orig_dir /path/to/LongCat-Flash-Chat \
    --exp_dir /path/to/LongCat-Flash-Chat-depth32 \
    --device npu --dtype float32 --atol 1e-5 \
    --mode all --sequential \
    --json_output /tmp/verification_results.json
```

> **注意**: NPU 仅 61 GB 内存，无法容纳 Chat 模型（562 GB float32）。如有大容量 CPU 内存环境（>1.5 TB）可尝试 CPU 推理验证。

| 验证方式 | 速度 | 覆盖范围 |
|---------|------|---------|
| `verify_expanded_weights.py` | 快（直接读取 safetensors） | 权重结构正确性 |
| `verify_model_output.py` | 慢（加载完整模型推理） | 端到端功能正确性 |

两者互补：权重验证通过但推理失败说明模型架构代码存在兼容性问题。

---

## 四、输出权重路径

| 扩展方式 | 输出路径 | 大小 |
|---------|---------|------|
| M1 专家扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-expertx2` | 2206.9 GB |
| M2 深度扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-depth32` | 1283.8 GB |
| M1+M2 联合 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-combined` | 2521.3 GB |

---

## 五、自定义扩展

### 指定目标专家数

```bash
TARGET_EXPERTS=768 bash scripts/expand_longcat_chat_experts.sh
```

### 指定扩展倍数

```bash
EXPERT_EXPANSION_FACTOR=3 bash scripts/expand_longcat_chat_experts.sh
EXPERT_EXPANSION_FACTOR=4 bash scripts/expand_longcat_chat_combined.sh
```

### 指定目标层数

```bash
# 深度 2× (28→56, 每层后插入恒等层)
TARGET_LAYERS=56 COPY_SOURCE="" bash scripts/expand_longcat_chat_depth.sh
```

### 均匀分布恒等层（默认配置）

默认 `copy_source=7,14,21,27` 让恒等层均匀分布在网络的 1/4、1/2、3/4、尾部位置：

```bash
# 28→32, 在层 7/14/21/27 后面各插入一个恒等层 (默认)
bash scripts/expand_longcat_chat_depth.sh

# 联合扩展同理 (默认)
bash scripts/expand_longcat_chat_combined.sh
```

自定义分布位置：

```bash
# 在层 6/13/20/26 后面各插入一个恒等层
COPY_SOURCE="6,13,20,26" bash scripts/expand_longcat_chat_depth.sh
```

### 带对称性破坏噪声（推荐用于后续训练）

```bash
ROUTER_NOISE_SCALE=1e-6 EXPERT_NOISE_SCALE=0.01 \
    bash scripts/expand_longcat_chat_experts.sh
```

### 同步扩展 moe_topk

```bash
# 专家数 2× 时 topk 也 2× (12→24)
TARGET_TOPK=24 bash scripts/expand_longcat_chat_experts.sh

# 联合扩展同理
TARGET_TOPK=24 bash scripts/expand_longcat_chat_combined.sh
```

### 使用 append 模式（非交错）

```bash
INSERTION_MODE=append bash scripts/expand_longcat_chat_depth.sh
```

### 控制分片的 layer 聚合度

默认 `--max_layers_per_shard=1` 确保每个 safetensors 文件只包含一个 layer 的权重（非 layer 参数单独存放），便于分布式加载和后续处理：

```bash
# 默认：每个 shard 最多 1 个 layer
bash scripts/expand_longcat_chat_combined.sh

# 允许每个 shard 包含 2 个 layer（减少 shard 数量，但 layer 分散度增加）
MAX_LAYERS_PER_SHARD=2 bash scripts/expand_longcat_chat_combined.sh
```

> 原始模型每分片固定 2 个 layer。扩展后使用 `--max_layers_per_shard=1` 可将分散度从 10-16 个文件/layer 优化到 1 个文件/layer。

---

## 六、方案对比

| 方案 | 参数增长 | 推理延迟 | Function Preserving | 适用场景 |
|------|---------|---------|:---:|---------|
| M1: 专家数 2× | ~2× | 不变 | ✅ 需对称性破坏 | 推理成本受限 |
| M2: 深度 +4 | ~1.14× | ~1.14× | ⚠️ 近似保持[[1]](#fn1) | 表达力优先 |
| M1+M2 联合 | ~2.3× | ~1.14× | ⚠️ 近似保持[[1]](#fn1) | 综合扩展 |

<a id="fn1">[1]</a>: LongCat-Flash 架构的双注意力 + 双 MLP + shortcut 连接导致 identity layer insertion **非严格函数保持**。权重验证通过，但端到端输出存在偏差（Lite 实测 cos_sim ≈ 0.97）。详见 [注意事项](#八注意事项) 第 6 条。

---

## 七、脚本与工具索引

### Shell 脚本

| 脚本 | 说明 |
|------|------|
| `scripts/expand_longcat_chat_experts.sh` | M1 专家数扩展 |
| `scripts/expand_longcat_chat_depth.sh` | M2 深度扩展 |
| `scripts/expand_longcat_chat_combined.sh` | M1+M2 联合扩展 |
| `scripts/verify_expanded_weights.sh` | 验证扩展权重（支持 experts/layers/combined） |

### Python 工具

| 文件 | 说明 |
|------|------|
| `utils/expand_moe_experts.py` | M1 专家扩展核心逻辑 |
| `utils/expand_moe_depth.py` | M2 深度扩展核心逻辑 |
| `utils/expand_moe_combined.py` | M1+M2 联合扩展核心逻辑 |
| `utils/verify_expanded_weights.py` | 权重验证（layers/experts/combined 三种模式）|
| `utils/verify_model_output.py` | 功能验证（前向 logit 比较 + 生成 token 比较）|
| `utils/analyze_shard_layout.py` | safetensors 分片布局分析（layer 分布、分散度等）|
| `utils/shared.py` | 共享工具：`build_layer_mapping`、`should_zero`、`expand_router_weight` 等 |

### 分片布局分析工具

`analyze_shard_layout.py` 用于分析模型的 safetensors 分片布局，提供**健康评分**、layer 分布矩阵、分散度图表、大小直方图等多维度诊断信息。

```bash
# 分析原始模型（完整报告）
python3 utils/analyze_shard_layout.py /path/to/LongCat-Flash-Chat

# 分析扩展后模型，显示更多分片示例
python3 utils/analyze_shard_layout.py /path/to/LongCat-Flash-Chat-combined --top 10

# 仅输出 JSON 数据（用于脚本解析）
python3 utils/analyze_shard_layout.py /path/to/model --json > report.json

# 不显示 Layer→Shard 映射矩阵（适合层数很多的模型）
python3 utils/analyze_shard_layout.py /path/to/model --no-matrix
```

#### 报告内容概览

运行工具后会输出一份完整的分片布局诊断报告，包含以下核心模块：

1. **健康评分 (0-100)**：综合评估分片布局质量
   - 🟢 良好 (≥80)：单层分片、低分散度、大小均匀
   - 🟡 一般 (50-79)：部分指标需优化
   - 🔴 需优化 (<50)：分散严重或大小差异大，建议重新扩展

2. **五维指标表**：
   | 指标 | 说明 | 理想值 |
   |------|------|--------|
   | 每分片层数 | 每个 safetensors 文件包含多少 layer | 1 层/分片 |
   | 层连续性 | layer 编号在分片内是否连续 | 连续 |
   | 分散度 | 每个 layer 的权重分布在多少个文件中 | 1-3 文件/layer |
   | 大小均匀 | 各分片文件大小差异 | max/min < 1.5× |
   | 非层参数 | `embed_tokens`、`norm` 等是否独立存放 | 独立 |

3. **Layer → Shard 映射矩阵**：ASCII 可视化网格，直观展示每个 layer 的权重分布在哪些分片中
   - 纵轴 = layer 编号，横轴 = safetensors 分片文件
   - `█` = 该 layer 的权重存在该分片中，`·` = 无权重

4. **分片大小直方图**：展示各分片文件大小的分布区间

5. **Layer 分散度图表**：每个 layer 跨越多少个分片文件的柱状图（层数 ≤40 时显示）

6. **综合评估**：自动列出 ✅ 优点 和 ❌ 问题，并给出优化建议

#### 示例输出解读

原始 LongCat-Flash-Chat 的典型报告：

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    safetensors 分片布局分析报告                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║  分片: 75 个文件              参数: 43,756 个           大小: 1.12 TB    ║
╠════════════════════════════════════════════════════════════════════════════╣
║  健康评分:  57/100  🟡 一般    █████████████████                         ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  每分片层数   │   层连续性    │    分散度     │   大小均匀    │   非层参数    │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 🔴 1 层: 0/70│ 🔴 连续: 5/70│🟡 avg 5.0   │ 🟢 max/min   │ 🟢 独立: 5   │
│              │              │  文件/层     │   1.0×       │   混合: 0    │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

说明：原始模型每分片固定 2 个 layer，所有 70 个分片都混合了不同 layer。每个 layer 固定跨 5 个分片，分片大小非常均匀（15.85–16.10 GB）。健康评分仅 57/100，主要扣分项是单层纯度（0%）和层连续性（仅 7% 连续）。

使用 `--max_layers_per_shard=1` 扩展后，健康评分可达 **100/100**：每个分片仅含 1 个 layer，分散度降至 1-3 文件/layer。

> 原始模型每分片固定 2 个 layer，每个 layer 分布在 5 个文件中。扩展后的模型因 size-based bin-packing 策略，layer 分散度可能达到 10-16 个文件/layer。如需优化为 layer-grouped 存储，可使用扩展脚本的 `--max_layers_per_shard` 参数（默认 1）。

#### 健康评分优化建议

若健康评分低于 80（🟡 或 🔴），报告会自动建议：

```
建议: 使用扩展脚本的 --max_layers_per_shard 1 重新扩展模型。
目标: 每个 safetensors 仅含 1 个 layer，单层过大时跨 2-3 个文件。
```

对应的环境变量设置：

```bash
# 默认：每个 shard 最多 1 个 layer（推荐，健康评分可达 90+）
bash scripts/expand_longcat_chat_combined.sh

# 允许每个 shard 包含 2 个 layer（减少 shard 数量，但分散度增加）
MAX_LAYERS_PER_SHARD=2 bash scripts/expand_longcat_chat_combined.sh
```

---

## 八、注意事项

1. **Identity zero expert**: LongCat-Flash-Chat 的 256 个 zero expert 为 identity 类型，不在 safetensors 中存储权重。扩展时仅在 config 和 Router 维度中按比例扩展 `zero_expert_num`（256→512），验证时自动跳过 zero expert 的权重索引检查。
2. **Interleave 模式**: 深度扩展默认使用 interleave 模式，新层交错插入原始层之间。验证时必须指定 `--insertion_mode interleave`，否则层映射不匹配。
3. **磁盘空间**: 扩展前确保目标目录有足够空间（联合扩展约需 2.5 TB，深度 +4 层约需 1.3 TB）。
4. **并行写入**: 默认使用 4 个 worker 并行写入，可通过 `WORKERS` 环境变量调整。设为 0 使用全部 CPU 核心。
5. **两遍处理**: 所有扩展脚本均使用两遍处理（Pass 1 扫描 header 计算布局，Pass 2 加载写入），确保输出 shard 文件名从一开始就是正确的。
6. **LongCat-Flash 架构的函数保持性限制 ⚠️**: LongCat-Flash-Chat 的 Decoder Layer 并非标准 Transformer 结构，其使用了 **双并行注意力头 + 双并行 MLP + 快捷连接（shortcut）** 的非标准残差路径（与 Lite 模型相同架构）。

   **标准 Transformer（理论假设）**：

   ```
   子层 1:  x = x + Attn(LN(x))
   子层 2:  x = x + FFN(LN(x))
   ```

   置零 `o_proj` + `down_proj` 后：`x = x + 0 + 0 = x` → **严格恒等**。

   **LongCat-Flash-Chat 实际结构**：

   ```
   子层 1:  x = x + Attn₀(LN₀(x))
   子层 2:  x = x + MLP₀(LN₁(x))      shortcut = MoE(LN₁(x))   ← 快捷输出暂存
   子层 3:  x = x + Attn₁(LN₂(x))
   子层 4:  x = x + MLP₁(LN₃(x)) + shortcut                    ← 快捷输出在此注入
   ```

   ![LongCat-Flash 架构图](longcat_flash_architecture.svg)

   置零 `o_proj` + `down_proj` 后：

   ```
   子层 1:  x = x + 0 = x
   子层 2:  x = x + 0 = x              shortcut = MoE(LN₁(x)) ≠ 0  ← 非零!
   子层 3:  x = x + 0 = x
   子层 4:  x = x + 0 + shortcut = x + shortcut ≠ x             ← 非恒等!
   ```

   快捷连接从子层 2 提取 MoE 输出，跨越子层 3 后注入子层 4。即使将新层的所有 `o_proj` 和 `down_proj` 置零，子层 2 的 MoE 输出仍为非零——**关键原因在于 zero expert 的 identity 特性**：

   1. **Routed experts (0-511)**: `down_proj = 0` → expert 输出 = 0 ✓
   2. **Zero experts (512-767, identity 类型)**: 不经过任何线性变换，输出直接等于输入（即 `expert_output = expert_input`），**完全绕过了 `down_proj`** → 输出 ≠ 0 ✗

   Router 从全部 768 个 expert 中选取 top-12，其中包含 zero expert。这些 zero expert 将其恒等输出（`LN₁(x)`）以 Router 权重加权后贡献给 MoE 输出：

   ```
   shortcut = MoE(LN₁(x))
            = Σ routed_experts(w_i × 0) + Σ zero_experts(w_j × LN₁(x))
            = α × LN₁(x)   (其中 α = 被选中 zero expert 的权重之和, α ≠ 0)
   ```

   子层 4 最终：`x = x + 0 + shortcut = x + α × LN₁(x) ≠ x` ← 非恒等!

   **影响**：
   - `verify_expanded_weights.py`（权重结构检查）**通过**——所有置零参数确实为零
   - `verify_model_output.py`（端到端功能检查）**不通过**——Chat 模型因体积过大（原始 562 GB，扩展后 1.3–2.5 TB）暂未运行端到端验证（NPU 仅 61 GB），但架构层面与 Lite 模型完全一致，故推断存在相同偏差
   - Lite 实测：max_abs_diff ≈ 15（+4 层）≈ 30（+14 层），误差随恒等层数量**线性累积**
   - `cos_sim` 保持在高位（0.96–0.99），输出方向高度相关，可用于训练初始化

   **适用场景**：尽管不是严格函数保持，扩展模型仍可用于后续训练——恒等层的输出与输入高度相关（cos_sim > 0.96），可作为良好的初始化起点。

   **若需要严格函数保持的深度扩展**，可选方案：
   1. 将恒等层中 Router 的 zero expert 路由权重置零，使 Router 仅选择 routed expert（其 `down_proj=0` 输出为零），从而 `shortcut = 0`
   2. 或重构残差连接，将 shortcut 注入点移到恒等层的 MLP 计算之前而非之后，使其不再跨越子层
