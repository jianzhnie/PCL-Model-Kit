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
扩展: 86,764 个参数, 148 shards, 2206.9 GB
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

在 28 层之间交错插入 28 个恒等初始化层，总层数 56。

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

**2× 扩展 (28→56)**：每个原始层后插入一个恒等层

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] ... [L27] [ID←27]
```

**+4 层扩展 (28→32, 默认 `copy_source=seq`)**：`source_list = [0,1,2,3]`，恒等层集中在前部

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] [L3] [ID←3] [L4] [L5] ... [L27]
```

**均匀分布恒等层**：通过 `--copy_source` 手动指定，可让恒等层分散到网络中部和尾部：

```bash
# 在层 7, 14, 21, 27 后面各插入一个恒等层
python -m utils.expand_moe_depth \
    --model_dir /path/to/LongCat-Flash-Chat \
    --output_dir /path/to/output \
    --target_layers 32 \
    --copy_source "7,14,21,27" \
    --insertion_mode interleave
```

#### Append 模式布局

原始 28 层顺序不变，恒等层追加在末尾：

```
[L0] [L1] ... [L27] [ID←0] [ID←1] ... [ID←27]
```

每个新层的处理：
- `self_attn.{0,1}.o_proj.weight` → 置零
- `mlp.experts.{0..511}.down_proj.weight` → 置零
- `mlps.{0,1}.down_proj.weight` → 置零
- 其余权重 → 从源层精确复制

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `num_layers` | 28 | 56 |
| 其余字段 | 不变 | 不变 |

#### 输出概要

```
扩展: 87,492 个参数, 150 shards, 2242.4 GB
新增恒等层: 28 层, 14,448 个张量置零
```

---

### 2.3 方案 M1+M2：联合扩展（Combined）

#### 概述

单次完成深度 + 专家扩展。默认 28→32 层（+4 层）+ 512→1024 专家。

```bash
bash scripts/expand_longcat_chat_combined.sh
```

#### 层映射

与 M2 的 28→32 interleave 完全一致，`source_list = [0,1,2,3]`：

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] [L3] [ID←3] [L4] [L5] ... [L27]
```

| 扩展后索引 | 来源 | 类型 |
|:-:|:-:|:-:|
| 0 | orig 0 | KEPT (原始) |
| 1 | orig 0 | NEW (恒等) |
| 2 | orig 1 | KEPT (原始) |
| 3 | orig 1 | NEW (恒等) |
| 4 | orig 2 | KEPT (原始) |
| 5 | orig 2 | NEW (恒等) |
| 6 | orig 3 | KEPT (原始) |
| 7 | orig 3 | NEW (恒等) |
| 8-31 | orig 4-27 | KEPT (无插入) |

如需均匀分布恒等层，可使用 `--copy_source "7,14,21,27"`。

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
扩展: 99,156 个参数, 169 shards, 2521.3 GB
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
    /path/to/LongCat-Flash-Chat-depth2 \
    --orig_layers 28 --target_layers 56 --insertion_mode interleave
```

验证内容：56 层结构完整、新层 `o_proj`/`down_proj` 全零、kept 层与原始层 bit-exact 匹配（含 interleave 重映射）。

### 联合扩展验证

```bash
bash scripts/verify_expanded_weights.sh combined \
    /path/to/LongCat-Flash-Chat \
    /path/to/LongCat-Flash-Chat-combined \
    --orig_layers 28 --target_layers 32 --insertion_mode interleave
```

验证内容：同时检查层映射 + 专家复制 + 恒等初始化。

---

## 四、输出权重路径

| 扩展方式 | 输出路径 | 大小 |
|---------|---------|------|
| M1 专家扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-expertx2` | 2206.9 GB |
| M2 深度扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-depth2` | 2242.4 GB |
| M1+M2 联合 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Chat-combined` | 2521.3 GB |

---

## 五、自定义扩展

### 指定目标专家数

```bash
TARGET_EXPERTS=768 bash scripts/expand_longcat_chat_experts.sh
```

### 指定目标层数

```bash
TARGET_LAYERS=42 bash scripts/expand_longcat_chat_depth.sh
```

### 均匀分布恒等层（推荐用于少量新层）

默认 `copy_source=seq` 会让新层集中在前部。少量扩展时建议手动指定 source 使恒等层均匀分布：

```bash
# 28→32, 在层 7/14/21/27 后面各插入一个恒等层
COPY_SOURCE="7,14,21,27" bash scripts/expand_longcat_chat_depth.sh

# 28→32, 联合扩展同理
COPY_SOURCE="7,14,21,27" bash scripts/expand_longcat_chat_combined.sh
```

### 带对称性破坏噪声（推荐用于后续训练）

```bash
ROUTER_NOISE_SCALE=1e-6 EXPERT_NOISE_SCALE=0.01 \
    bash scripts/expand_longcat_chat_experts.sh
```

### 使用 append 模式（非交错）

```bash
INSERTION_MODE=append bash scripts/expand_longcat_chat_depth.sh
```

---

## 六、方案对比

| 方案 | 参数增长 | 推理延迟 | Function Preserving | 适用场景 |
|------|---------|---------|:---:|---------|
| M1: 专家数 2× | ~2× | 不变 | 需对称性破坏 | 推理成本受限 |
| M2: 深度 2× | ~2× | ~2× | 完全保持 | 表达力优先 |
| M1+M2 联合 | ~2.3× | ~1.14× | 需对称性破坏 | 综合扩展 |

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
| `utils/shared.py` | 共享工具：`build_layer_mapping`、`should_zero`、`expand_router_weight` 等 |

---

## 八、注意事项

1. **Identity zero expert**: LongCat-Flash-Chat 的 256 个 zero expert 为 identity 类型，不在 safetensors 中存储权重。扩展时仅在 config 和 Router 维度中按比例扩展 `zero_expert_num`（256→512），验证时自动跳过 zero expert 的权重索引检查。
2. **Interleave 模式**: 深度扩展默认使用 interleave 模式，新层交错插入原始层之间。验证时必须指定 `--insertion_mode interleave`，否则层映射不匹配。
3. **磁盘空间**: 扩展前确保目标目录有足够空间（联合扩展约需 2.5 TB）。
4. **并行写入**: 默认使用 4 个 worker 并行写入，可通过 `WORKERS` 环境变量调整。推荐使用 16 个 worker 以加速大模型扩展。
5. **两遍处理**: 所有扩展脚本均使用两遍处理（Pass 1 扫描 header 计算布局，Pass 2 加载写入），确保输出 shard 文件名从一开始就是正确的。
