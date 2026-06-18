# MoE 专家数扩展 (M1) 使用指南

## 一、方案原理

### 核心思想

MoE 专家数扩展（方案 M1，Expert Upcycling）通过复制已有专家来增加总专家数，同时保持 Top-K 路由不变。由于每个 token 仍只激活 K 个专家，**推理时的计算量几乎不变**，但总参数量随专家数线性增长。

这是 MoE 架构独有的优势——在推理成本近似不变的前提下实现参数翻倍。

### 扩展机制

```
原始模型 (E=512, top-12):
  Token → Router → 选 12/512 个专家 → 加权求和 → 输出

扩展后 (E=1024, top-12):
  Token → Router' → 选 12/1024 个专家 → 加权求和 → 输出
                     ↑ 推理激活参数不变
```

### 三个扩展维度

| 组件 | 扩展方式 | 说明 |
|------|---------|------|
| **专家权重** | 每个专家复制为 k 份 | `gate_proj`, `up_proj`, `down_proj` 全部复制 |
| **Router 权重** | 输出维度 E → kE | 新增列从被复制专家的对应列初始化 |
| **Score Correction Bias** | 精确复制 | 保留原始路由偏好 |

### 对称性破坏（关键步骤）

复制后所有副本参数完全相同，Router 会给它们相同分数，导致训练时无法分化。必须打破对称性：

**Router 噪声**（`--router-noise-scale`）：

```python
# 复制的 Router 行加入小高斯噪声
W_router_new[new_idx] = W_router[src_idx] + randn() * noise_scale * std(W_router)
```

效果：Router 对不同副本产生微小偏好差异，训练时可逐渐分化。

**Expert 权重噪声**（`--expert-noise-scale`）：

```python
# 复制的专家权重加入噪声
expert_new.weight = expert_src.weight + randn() * expert_noise_scale * std(weight)
```

效果：更激进的对称性破坏，副本从初始化开始就有不同的函数行为，加速专业化。

### LongCat-Flash-Chat 模型参数

| 参数 | 值 |
|------|------|
| `n_routed_experts` | 512 |
| `zero_expert_num` | 256 |
| `moe_topk` | 12 |
| `expert_ffn_hidden_size` | 2048 |
| `hidden_size` | 6144 |
| `num_layers` | 28 |

扩展 2× 后：1024 个路由专家 + 512 个 zero expert，Router 输出维度从 768 → 1536。

### Zero Expert 处理

LongCat-Flash-Chat 使用 identity 类型的 zero expert（输出恒为 0 或恒等映射）。扩展时 zero expert 也按相同倍数复制，保持与 routed expert 的比例关系不变。

输出布局（以 2× 扩展为例）：

```
Router 维度: [real_experts × 2, zero_experts × 2]
            = [1024 routed, 512 zero] = 1536 total

Expert 索引: [0..1023] routed, [1024..1535] zero
```

---

## 二、使用方法

### 基本用法

```bash
python -m utils.expand_moe_experts \
    --model_dir /path/to/original_model \
    --output_dir /path/to/expanded_model
```

默认将专家数翻倍，不加噪声。

### 推荐用法（带对称性破坏）

```bash
python -m utils.expand_moe_experts \
    --model_dir /path/to/original_model \
    --output_dir /path/to/expanded_model \
    --router-noise-scale 1e-6 \
    --expert-noise-scale 0.01
```

### 完整参数

```bash
python -m utils.expand_moe_experts \
    --model_dir MODEL_DIR \
    --output_dir OUTPUT_DIR \
    [--target_experts N]          # 目标专家数，默认 2×原始
    [--target_topk K]             # 目标 top-k，默认不变
    [--use_group_routing]         # 启用分组路由（与 --target_topk 互斥）
    [--router-noise-scale FLOAT]   # Router 噪声 (推荐 1e-6)
    [--expert-noise-scale FLOAT]  # Expert 权重噪声 (推荐 0.01)
    [--workers N]                 # 并行 worker 数 (0=CPU核数)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | 必填 | 原始模型目录 |
| `--output_dir` | 必填 | 输出目录 |
| `--target_experts` | 2×原始 | 目标路由专家数，必须是原始的整数倍 |
| `--target_topk` | 不变 | 新的 top-k 值（增大可提升表达力但增加计算） |
| `--use_group_routing` | 否 | 启用分组路由，topk 不变，添加 `use_group_routing` 和 `expert_expansion_factor` 到 config |
| `--router-noise-scale` | 0.0 | Router 权重噪声，相对于 Router 权重标准差的比例 |
| `--expert-noise-scale` | 0.0 | Expert 权重噪声，相对于每个权重矩阵标准差的比例 |
| `--workers` | 1 | 输出分片并行度（>1 使用多进程加速写入） |

### LongCat-Flash-Chat 扩展示例

**2× 专家扩展（512→1024，推荐）**：

```bash
python -m utils.expand_moe_experts \
    --model_dir /Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat \
    --output_dir /path/to/LongCat-Flash-Chat-1024E \
    --target_experts 1024 \
    --router-noise-scale 1e-6 \
    --expert-noise-scale 0.01 \
    --workers 4
```

**2× 扩展 + 增大 top-k（更强表达力）**：

```bash
python -m utils.expand_moe_experts \
    --model_dir /Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat \
    --output_dir /path/to/LongCat-Flash-Chat-1024E-top24 \
    --target_experts 1024 \
    --target_topk 24 \
    --router-noise-scale 1e-6 \
    --expert-noise-scale 0.01
```

**2× 扩展 + 分组路由**：

```bash
python -m utils.expand_moe_experts \
    --model_dir /Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat \
    --output_dir /path/to/LongCat-Flash-Chat-1024E-grouped \
    --target_experts 1024 \
    --use_group_routing \
    --router-noise-scale 1e-6 \
    --expert-noise-scale 0.01
```

**4× 扩展（512→2048）**：

```bash
python -m utils.expand_moe_experts \
    --model_dir /Users/robin/hfhub/models/meituan-longcat/LongCat-Flash-Chat \
    --output_dir /path/to/LongCat-Flash-Chat-2048E \
    --target_experts 2048 \
    --router-noise-scale 1e-6 \
    --expert-noise-scale 0.01 \
    --workers 0
```

---

## 三、路由策略选择

### 方案对比

| 策略 | `--target_topk` | 推理成本 | 表达力 | 说明 |
|------|:-:|:-:|:-:|------|
| **保持 top-k 不变** | 不设 | 不变 | 不变 | 最保守，纯参数扩展 |
| **增大 top-k** | 设为 2×原值 | 线性增加 | 提升 | 激活更多专家，更强但更贵 |
| **分组路由** | N/A | 不变 | 微增 | 原 top-k 专家在各组内选取 |

### 推荐决策

```
推理成本不能增加？
  → 保持 top-k 不变 或 使用分组路由

可以接受推理成本增加？
  → 按比例增大 top-k（如 12→24）
```

---

## 四、验证方法

### 扩展后检查要点

```python
import json
from pathlib import Path

output_dir = Path("/path/to/expanded_model")

# 1. 检查 config
config = json.load(open(output_dir / "config.json"))
assert config["n_routed_experts"] == 1024  # 翻倍
assert config["zero_expert_num"] == 512     # 同步翻倍
assert config["moe_topk"] == 12            # 保持不变（除非指定 --target_topk）

# 2. 检查 index 中的专家数
index = json.load(open(output_dir / "model.safetensors.index.json"))
import re
experts_layer0 = set()
for key in index["weight_map"]:
    m = re.search(r"model\.layers\.0\.mlp\.experts\.(\d+)\.", key)
    if m:
        experts_layer0.add(int(m.group(1)))
print(f"Layer 0 experts: {min(experts_layer0)}-{max(experts_layer0)}")
# 应输出: Layer 0 experts: 0-1535 (1024 routed + 512 zero)
```

### 使用 verify_expanded_weights.py

```bash
python -m utils.verify_expanded_weights \
    --original_dir /path/to/original_model \
    --expanded_dir /path/to/expanded_model
```

---

## 五、后续训练建议

### 训练配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 5e-5 | 专家已有良好初始化 |
| Load balancing loss | 0.01 ~ 0.1 | 防止专家负载不均 |
| Z-loss | 1e-3 | 稳定 Router 输出 |
| 数据量 | 50-100B tokens | 足够的分化训练 |
| Warmup | 2000-5000 steps | Router 适应新专家 |

### 训练要点

1. **必须添加 load balancing loss**：防止 Router 将所有 token 路由到少数专家
2. **监控专家利用率**：健康的 MoE 应该所有专家被均匀激活
3. **冻结策略**：可先冻结非 MoE 参数（attention、norm），仅训练专家和 Router
4. **数据混合**：保持与原始预训练分布一致，避免遗忘

### 训练框架要求

- 需要支持 MoE 的训练框架：Megatron-LM、DeepSpeed-MoE、FSDP with expert parallelism
- All-to-all 通信：专家并行需要高带宽互联

---

## 六、技术细节

### 文件结构

```
utils/
├── expand_moe_experts.py        # M1 专家数扩展（本文档）
├── expand_moe_depth.py          # M2 深度扩展
├── expand_model_layers.py       # 简单层复制
├── verify_expanded_weights.py   # 权重校验
└── shared.py                    # 共享工具（Router 检测、Expert 解析等）
```

### 内部数据流

```
Pass 1: 扫描所有 shard header（不加载张量）
  → 计算输出大小和分片数

Pass 2: 逐 shard 加载张量
  → Router 权重: 按 expansion_factor 扩展维度，副本加噪声
  → Expert 权重: 原始保留，副本 clone（+可选噪声）
  → Zero Expert: 重新编号 + 复制
  → 其他张量: 直接拷贝
  → 按目标大小分片写出
```

### 并行模式

当 `--workers > 1` 时：
1. 单次 header 扫描完成全局 tensor→shard 分配
2. 每个 output shard 由独立 worker 处理
3. Worker 内使用 tensor cache 避免同一 input key 重复读取
4. 适合大模型（100+ 分片）场景，可显著加速 IO

### 输出格式

标准 HuggingFace safetensors 格式：
- `config.json`：更新 `n_routed_experts`、`zero_expert_num`、`moe_topk`
- `model.safetensors.index.json`：新的权重映射
- `model-XXXXX-of-YYYYY.safetensors`：分片权重
- tokenizer 等辅助文件原样复制

### 与 M2 深度扩展的组合

M1 和 M2 可以组合使用，实现更大的参数扩展：

```bash
# Step 1: 专家数 2× (M1)
python -m utils.expand_moe_experts \
    --model_dir /path/to/original \
    --output_dir /path/to/step1_experts2x \
    --target_experts 1024 \
    --router-noise-scale 1e-6 --expert-noise-scale 0.01

# Step 2: 深度 2× (M2)
python -m utils.expand_moe_depth \
    --model_dir /path/to/step1_experts2x \
    --output_dir /path/to/step2_depth2x \
    --target_layers 56
```

组合后：参数量约为原始的 4×，推理延迟约 2×（仅深度增加的贡献）。
