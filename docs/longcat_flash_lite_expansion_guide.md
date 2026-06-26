# LongCat-Flash-Lite MoE 扩展指南

## 一、模型概览

| 参数 | 值 |
|------|------|
| `architectures` | `LongcatFlashNgramForCausalLM` |
| `hidden_size` | 3072 |
| `expert_ffn_hidden_size` | 1024 |
| `num_layers` | 14 |
| `n_routed_experts` | 256 |
| `zero_expert_num` | 128 (identity 类型，无存储权重) |
| `moe_topk` | 12 |
| `num_attention_heads` | 32 |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536 |

> LongCat-Flash-Lite 的 128 个 zero expert 为 identity 类型，不在 safetensors 中存储权重参数。扩展时仅复制 routed expert 权重，zero expert 仅在 config 和 Router 维度中按比例同步扩展。

---

## 二、三种扩展方式

### 2.1 方案 M1：专家数扩展（Expert Upcycling）

#### 概述

将 256 个 routed expert 翻倍至 512，推理激活参数不变（仍 top-12），总参数约 2×。

```bash
bash scripts/expand_longcat_lite_experts.sh
```

#### 三类张量的具体处理

**Expert 权重 (routed expert 0-255)**

映射关系由 `build_expert_target_map(256, 512)` 构建：

```
new_idx = 256, 257, ..., 511
src_idx = new_idx % 256 = 0, 1, ..., 255
```

每个专家的 `gate_proj.weight`、`up_proj.weight`、`down_proj.weight` 全部 `tensor.clone()`。原始 expert 0-255 保持不动，新增 expert 256-511 是精确副本。

**Router 权重 (`mlp.router.classifier.weight`)**

```
原始 Router: [384, 3072]
             ├── real_part: tensor[:256]   (256 个 routed expert 的路由行)
             └── zero_part: tensor[256:]   (128 个 zero expert 的路由行)

扩展后:
  expanded_real = cat([real_part, real_part], dim=0)   → [512, 3072]
  expanded_zero = cat([zero_part, zero_part], dim=0)   → [256, 3072]
  output = cat([expanded_real, expanded_zero], dim=0)  → [768, 3072]
```

Router Bias (`e_score_correction_bias`) 同理: `[384] → [768]`。

**非专家参数 (attention, norm, embed, lm_head)**

直接原样拷贝，不做任何修改。

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `n_routed_experts` | 256 | 512 |
| `zero_expert_num` | 128 | 256 |
| `moe_topk` | 12 | 12 (不变) |
| `num_layers` | 14 | 14 (不变) |

#### 输出概要

```
原始: 11,160 个参数,  26 shards, 138.2 GB
扩展: 21,912 个参数,  46 shards, 205.9 GB
新增: 10,752 个张量 (256 experts × 14 layers × 3 params)
      + 14 router weight 扩展 + 14 router bias 扩展
```

#### 关键特性

- **推理成本不变**: `moe_topk` 保持 12，每个 token 仍只激活 12 个专家
- **Function-preserving**: 副本与原始完全相同，Router 给同源副本相同分数，扩展后模型输出与原始模型数学等价
- **对称性未打破**: 默认不加噪声。如需后续训练分化，使用 `--router-noise-scale 1e-6 --expert-noise-scale 0.01`

---

### 2.2 方案 M2：深度扩展（Identity Layer Insertion）

#### 概述

在 14 层之间交错插入 14 个恒等初始化层，总层数 28。

```bash
bash scripts/expand_longcat_lite_depth.sh
```

#### 恒等映射原理

新层通过将 `o_proj.weight` 和 `down_proj.weight` 置零实现恒等映射：

```
output = input + Attention(Norm(input)) + MLP(Norm(...))
       = input + 0 + 0   (因为 W_o = 0, W_down = 0)
       = input            ← 恒等映射
```

#### Interleave 模式布局

**2× 扩展 (14→28)**：每个原始层后插入一个恒等层

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] ... [L13] [ID←13]
```

**+4 层扩展 (14→18, 默认 `copy_source=seq`)**：`source_list = [0,1,2,3]`，恒等层集中在前部

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] [L3] [ID←3] [L4] [L5] ... [L13]
```

| 扩展后索引 | 来源 | 类型 |
|:-:|:-:|:-:|
| 0 | orig 0 | 原始 |
| 1 | orig 0 | 恒等 (o_proj=0, down_proj=0) |
| 2 | orig 1 | 原始 |
| 3 | orig 1 | 恒等 |
| 4 | orig 2 | 原始 |
| 5 | orig 2 | 恒等 |
| 6 | orig 3 | 原始 |
| 7 | orig 3 | 恒等 |
| 8-17 | orig 4-13 | 原始 (无插入) |

这是因为 `parse_copy_source(None, 14, 4)` 生成 `[0,1,2,3]`（`i % 14` 取前 4 个），`build_layer_mapping` 按原始层顺序遍历，在以层 0-3 为 source 的位置各插入一个恒等层。

**均匀分布恒等层**：通过 `--copy_source` 手动指定，可让恒等层分散到网络中部和尾部：

```bash
# 在层 3, 6, 9, 12 后面各插入一个恒等层
python -m utils.expand_moe_depth \
    --model_dir /path/to/LongCat-Flash-Lite \
    --output_dir /path/to/output \
    --target_layers 18 \
    --copy_source "3,6,9,12" \
    --insertion_mode interleave
```

得到的排布：

```
[L0] [L1] [L2] [L3] [ID←3] [L4] [L5] [L6] [ID←6] [L7] [L8] [L9] [ID←9] [L10] [L11] [L12] [ID←12] [L13]
```

恒等层均匀分布在网络的 1/4、1/2、3/4、尾部位置，对后续训练通常比集中在前部更有利。

#### Append 模式布局

原始 14 层顺序不变，4 个恒等层追加在末尾：

```
[L0] [L1] ... [L13] [ID←0] [ID←1] [ID←2] [ID←3]
```

每个新层的处理：
- `self_attn.{0,1}.o_proj.weight` → 置零
- `mlp.experts.{0..255}.down_proj.weight` → 置零
- `mlps.{0,1}.down_proj.weight` → 置零
- 其余权重 → 从源层精确复制

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `num_layers` | 14 | 28 |
| 其余字段 | 不变 | 不变 |

#### 输出概要

```
扩展: 22,276 个参数, 52 shards, 210.9 GB
新增恒等层: 14 层, 3,640 个张量置零
```

---

### 2.3 方案 M1+M2：联合扩展（Combined）

#### 概述

单次完成深度 + 专家扩展。默认 14→18 层（+4 层）+ 256→512 专家。

```bash
bash scripts/expand_longcat_lite_combined.sh
```

#### 层映射

与 M2 的 14→18 interleave 完全一致，`source_list = [0,1,2,3]`：

```
[L0] [ID←0] [L1] [ID←1] [L2] [ID←2] [L3] [ID←3] [L4] [L5] ... [L13]
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
| 8-17 | orig 4-13 | KEPT (无插入) |

如需均匀分布恒等层，可使用 `--copy_source "3,6,9,12"`。

#### 专家映射

每个原始 expert 复制出 1 个副本：`expert 0 → [0, 256]`，`expert 1 → [1, 257]`，...，`expert 255 → [255, 511]`。共 256 对。

#### KEPT 层的张量结构（如扩展后 layer 0 ← orig 0）

| 组件 | 原始 shape | 扩展后 shape | 处理方式 |
|------|-----------|-------------|---------|
| `router.classifier.weight` | [384, 3072] | [768, 3072] | 扩展 (见 Router 布局) |
| `router.e_score_correction_bias` | [384] | [768] | 同上 |
| `experts.0-255.{gate,up,down}_proj` | 不变 | 不变 | 保留原值 |
| `experts.256-511.{gate,up,down}_proj` | 新增 | 同原始 | clone(expert[i%256]) |
| `self_attn`, `layernorm`, `mlps` | 不变 | 不变 | 不变 |

#### NEW 恒等层的张量结构（如扩展后 layer 1 ← orig 0）

| 组件 | 处理方式 |
|------|---------|
| `router.classifier.weight` | 从 orig 0 的 router 扩展为 [768, 3072] |
| `experts.0-511.gate_proj.weight` | clone(orig 0 的 expert[i%256]) |
| `experts.0-511.up_proj.weight` | clone(orig 0 的 expert[i%256]) |
| `experts.0-511.down_proj.weight` | **全零** (identity init) |
| `self_attn.{0,1}.o_proj.weight` | **全零** (identity init) |
| `self_attn 其余 (q/kv proj 等)` | clone(orig 0) |
| `mlps.{0,1}.down_proj.weight` | **全零** (identity init) |
| `mlps.{0,1}.gate/up_proj` | clone(orig 0) |
| `input_layernorm`, `post_attention_layernorm` | clone(orig 0) |

#### Router 权重内部布局

```
原始 [384, 3072]:
  行 0-255:    real part (256 个 routed expert 路由权重)
  行 256-383:  zero part (128 个 zero expert 路由权重)

扩展后 [768, 3072]:
  行 0-255:    real_block_0 (orig real part)
  行 256-511:  real_block_1 (real part 的副本)
  行 512-639:  zero_block_0 (orig zero part)
  行 640-767:  zero_block_1 (zero part 的副本)
```

Bias 同理：`[384] → [768]`，布局 `[real×2 | zero×2]`。

#### Config 变更

| 字段 | 原始 | 扩展后 |
|---|---|---|
| `num_layers` | 14 | 18 |
| `n_routed_experts` | 256 | 512 |
| `zero_expert_num` | 128 | 256 |
| `moe_topk` | 12 | 12 (不变) |

#### 输出概要

```
扩展: 28,160 个参数, 55 shards, 246.0 GB
新增恒等层: 4 层, 置零参数: 2,064 个
```

---

## 三、验证方法

所有扩展输出均通过 `verify_expanded_weights.py` 验证，支持 `layers`、`experts`、`combined` 三种模式。

### 专家扩展验证

```bash
bash scripts/verify_expanded_weights.sh experts \
    /path/to/LongCat-Flash-Lite \
    /path/to/LongCat-Flash-Lite-expertx2
```

验证内容：Router shape `[768, 3072]`、expert 0-511 索引完整、expert 256 == expert 0 (bit-exact)、非专家参数不变。

### 深度扩展验证

```bash
bash scripts/verify_expanded_weights.sh layers \
    /path/to/LongCat-Flash-Lite \
    /path/to/LongCat-Flash-Lite-depth2 \
    --orig_layers 14 --target_layers 28 --insertion_mode interleave
```

验证内容：28 层结构完整、新层 `o_proj`/`down_proj` 全零、kept 层与原始层 bit-exact 匹配（含 interleave 重映射）。

### 联合扩展验证

```bash
bash scripts/verify_expanded_weights.sh combined \
    /path/to/LongCat-Flash-Lite \
    /path/to/LongCat-Flash-Lite-combined \
    --orig_layers 14 --target_layers 18 --insertion_mode interleave
```

验证内容：同时检查层映射 + 专家复制 + 恒等初始化。

---

## 四、输出权重路径

| 扩展方式 | 输出路径 | 大小 |
|---------|---------|------|
| M1 专家扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-expertx2` | 205.9 GB |
| M2 深度扩展 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-depth2` | 210.9 GB |
| M1+M2 联合 | `/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-combined` | 246.0 GB |

---

## 五、自定义扩展

### 指定目标专家数

```bash
TARGET_EXPERTS=768 bash scripts/expand_longcat_lite_experts.sh
```

### 指定扩展倍数

```bash
EXPERT_EXPANSION_FACTOR=3 bash scripts/expand_longcat_lite_experts.sh
EXPERT_EXPANSION_FACTOR=4 bash scripts/expand_longcat_lite_combined.sh
```

### 指定目标层数

```bash
TARGET_LAYERS=21 bash scripts/expand_longcat_lite_depth.sh
```

### 均匀分布恒等层（推荐用于少量新层）

默认 `copy_source=seq` 会让新层集中在前部。少量扩展时建议手动指定 source 使恒等层均匀分布：

```bash
# 14→18, 在层 3/6/9/12 后面各插入一个恒等层
COPY_SOURCE="3,6,9,12" bash scripts/expand_longcat_lite_depth.sh

# 14→18, 联合扩展同理
COPY_SOURCE="3,6,9,12" bash scripts/expand_longcat_lite_combined.sh
```

### 带对称性破坏噪声（推荐用于后续训练）

```bash
ROUTER_NOISE_SCALE=1e-6 EXPERT_NOISE_SCALE=0.01 \
    bash scripts/expand_longcat_lite_experts.sh
```

### 同步扩展 moe_topk

```bash
# 专家数 2× 时 topk 也 2× (12→24)
TARGET_TOPK=24 bash scripts/expand_longcat_lite_experts.sh

# 联合扩展同理
TARGET_TOPK=24 bash scripts/expand_longcat_lite_combined.sh
```

### 使用 append 模式（非交错）

```bash
INSERTION_MODE=append bash scripts/expand_longcat_lite_depth.sh
```

---

## 六、方案对比

| 方案 | 参数增长 | 推理延迟 | Function Preserving | 适用场景 |
|------|---------|---------|:---:|---------|
| M1: 专家数 2× | ~1.5× | 不变 | 需对称性破坏 | 推理成本受限 |
| M2: 深度 2× | ~1.5× | ~2× | 完全保持 | 表达力优先 |
| M1+M2 联合 | ~1.8× | ~1.3× | 需对称性破坏 | 综合扩展 |

---

## 七、脚本与工具索引

### Shell 脚本

| 脚本 | 说明 |
|------|------|
| `scripts/expand_longcat_lite_experts.sh` | M1 专家数扩展 |
| `scripts/expand_longcat_lite_depth.sh` | M2 深度扩展 |
| `scripts/expand_longcat_lite_combined.sh` | M1+M2 联合扩展 |
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

1. **Identity zero expert**: LongCat-Flash-Lite 的 128 个 zero expert 为 identity 类型，不在 safetensors 中存储权重。扩展时仅在 config 和 Router 维度中按比例扩展 `zero_expert_num`（128→256），验证时自动跳过 zero expert 的权重索引检查。
2. **Interleave 模式**: 深度扩展默认使用 interleave 模式，新层交错插入原始层之间。验证时必须指定 `--insertion_mode interleave`，否则层映射不匹配。
3. **磁盘空间**: 扩展前确保目标目录有足够空间（联合扩展约需 246 GB）。
4. **并行写入**: 默认使用 4 个 worker 并行写入，可通过 `WORKERS` 环境变量调整。设为 0 使用全部 CPU 核心。
5. **两遍处理**: 所有扩展脚本均使用两遍处理（Pass 1 扫描 header 计算布局，Pass 2 加载写入），确保输出 shard 文件名从一开始就是正确的。
