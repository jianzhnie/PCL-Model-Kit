# Kimi2 Huggingface -> MCore 转换代码 Review

> 分析文件: `utils/convert_ckpt_hf2mcore.py` + `scripts/ckpt_convert_hf2mcore.sh`
> 参考实现: `utils/convert_ckpt_deepseek3.py` + `utils/convert_kimi2_hf2mcore.py`
> 分析日期: 2026-04-17

---

## 1. 概述

`convert_ckpt_hf2mcore.py` 是一个较新版本的重写实现，相比 `convert_kimi2_hf2mcore.py` 增加了并行 I/O、SHA256 校验、dtype 转换等特性。但在核心的 MoE expert 权重转换逻辑中，存在与两个参考实现不一致的地方，可能导致输出格式错误。

---

## 2. Shell 脚本 Review (`ckpt_convert_hf2mcore.sh`)

### 2.1 参数与训练脚本一致性

| 参数 | 脚本默认值 | 训练脚本值 | 匹配 |
|------|-----------|-----------|------|
| TP | 2 | 2 | ✓ |
| PP | 8 | 8 | ✓ |
| EP | **8** | **64** | ⚠ 默认不同 |
| SCHEDULES_METHOD | **""** | **dualpipev** | ⚠ 默认不同 |
| NUM_LAYERS | 32 | 32 | ✓ |
| HIDDEN_SIZE | 7168 | 7168 | ✓ |
| EXPERT_TP | 1 | 1 | ✓ |
| FIRST_K_DENSE_REPLACE | 2 | 2 | ✓ |
| QK_LAYERNORM | 1 | 1 | ✓ |
| MOE_GROUPED_GEMM | 启用 | 启用 | ✓ |

Shell 脚本逻辑正确，三种并行模式（DualPipe/标准 VPP/纯 PP）切换合理。默认 EP 和 SCHEDULES_METHOD 与训练不一致，但可通过环境变量覆盖。

---

## 3. 正确的部分

以下部分经与 DeepSeek3 和 Kimi2-specific 对比验证，实现正确：

### 3.1 GQA Attention（`_set_layer_attn`）

```python
q_tp = torch.chunk(q_weight, self.tp_size, dim=0)   # [8192,7168] → 2 × [4096,7168]
k_tp = torch.chunk(k_weight, self.tp_size, dim=0)   # [256,7168]  → 2 × [128,7168]
v_tp = torch.chunk(v_weight, self.tp_size, dim=0)   # [256,7168]  → 2 × [128,7168]
o_proj_tp = torch.chunk(o_proj, self.tp_size, dim=1) # [7168,8192] → 2 × [7168,4096]

qkv_shards = [torch.cat([q_tp[i], k_tp[i], v_tp[i]], dim=0) for i in range(tp_size)]
# → [4352, 7168] per TP rank ✓
```

与 DeepSeek3/Kimi2-specific 完全一致。Shape 校验、QK LayerNorm 处理均正确。

### 3.2 Dense MLP（`_set_layer_mlp` dense path）

gate/up 交织后 TP 切分，down 按 dim=1 切分。与参考实现一致。✓

### 3.3 MoE Router、Shared Experts、LayerNorm、Preprocess/Postprocess

实现正确，Router bias 兼容 `e_score_correction_bias` 和 `gate.bias`，比 DeepSeek3 更灵活。✓

### 3.4 DualPipe/VPP 层映射（`_build_vpprank_layer_map`）

DualPipe 的前后半交替分配算法与 DeepSeek3/Kimi2-specific 一致。✓

### 3.5 Checkpoint Args（`_build_checkpoint_args`）

写入完整训练配置，所有字段与训练脚本匹配。✓

### 3.6 Checkpoint 格式（`_mp_prefix`、`_save_single_rank_file`）

目录命名和 VPP 保存格式（model0/model1/...）与 Megatron 标准一致。✓

---

## 4. 发现的 Bug

### Bug 1（严重）：Grouped GEMM weight1 多了 `.permute(1, 0, 2)` 导致内存布局错误

**位置**: `_set_layer_mlp` 第 1252 行

**当前代码**:
```python
w1 = fc1_tp[expert_tp_idx].permute(1, 0, 2).contiguous().reshape(self.hidden_size, -1).clone()
```

**DeepSeek3 参考实现** (第 849 行):
```python
mg_model[...][experts_weight1_key] = gemm_fc1_ep_tp[tp_rank].reshape(self.hidden_size, -1).clone()
```

**Kimi2-specific 参考实现** (第 768 行):
```python
w1 = fc1_shards[expert_tp_idx].reshape(self.hidden_size, -1).clone()
```

**分析**:

`fc1_tp[expert_tp_idx]` 的 shape 为 `[num_local_experts, hidden_size, intermediate]` = `[16, 7168, 24576]`。

**无 permute（正确格式）**:
```
[16, 7168, 24576].reshape(7168, -1) = [7168, 393216]
内存布局 (expert-sequential):
  Row 0: expert_0 rows 0-15   (连续 16 行来自同一个 expert)
  Row 1: expert_0 rows 16-31
  ...
  Row 447: expert_0 rows 7152-7167
  Row 448: expert_1 rows 0-15
  ...
```

**有 permute（当前代码）**:
```
[16, 7168, 24576].permute(1,0,2) = [7168, 16, 24576]
.reshape(7168, -1) = [7168, 393216]
内存布局 (expert-interleaved):
  Row 0: e0_row0, e1_row0, e2_row0, ..., e15_row0  (每个 expert 的第 0 行)
  Row 1: e0_row1, e1_row1, e2_row1, ..., e15_row1  (每个 expert 的第 1 行)
  ...
```

两种布局虽然最终 shape 相同 `[7168, 393216]`，但**内存中 expert 权重的排列方式完全不同**。

Megatron 加载时使用 `.view(num_local_experts, hidden, intermediate)` 恢复 3D 布局。这要求 expert-sequential 格式（无 permute）。使用 expert-interleaved 格式会导致：
- Expert 0 被错误地包含其他 expert 的行
- 模型推理结果错误

**影响范围**: 所有使用 `--moe-grouped-gemm` 的转换都受影响（包括默认配置）。

**修复方案**: 移除 `.permute(1, 0, 2).contiguous()`，直接 reshape：
```python
w1 = fc1_tp[expert_tp_idx].contiguous().reshape(self.hidden_size, -1).clone()
```

### Bug 2（中等）：`expert_tp_idx` 计算公式错误

**位置**: `_set_layer_mlp` 第 1245 行（grouped_gemm）和第 1291 行（non-grouped）

**当前代码**:
```python
expert_tp_idx = tp_rank * self.expert_tp_size // self.tp_size
```

**正确公式** (Kimi2-specific 第 767 行):
```python
expert_tp_idx = tp_rank % self.expert_tp_size
```

**对比** (tp_size=4, expert_tp_size=2):

| tp_rank | 当前公式 | 正确公式 |
|---------|---------|---------|
| 0 | 0×2//4 = 0 | 0%2 = 0 |
| 1 | 1×2//4 = **0** | 1%2 = **1** |
| 2 | 2×2//4 = **1** | 2%2 = **0** |
| 3 | 3×2//4 = 1 | 3%2 = 1 |

TP rank 0 和 1 被错误地分配到同一个 shard，而正确行为应该是 TP rank 0 和 2 共享 shard 0，TP rank 1 和 3 共享 shard 1。

**影响范围**: 仅 `expert_tp_size > 1` 时触发。默认 `expert_tp_size=1` 时两种公式均给出 0，不受影响。

### Bug 3（中等）：Grouped GEMM 中 `expert_tp_size > 1` 沿错误维度切分

**位置**: `_set_layer_mlp` 第 1232-1239 行

**当前代码**:
```python
if self.expert_tp_size > 1:
    num_tp_shards = self.tp_size // self.expert_tp_size
    fc1_tp = torch.chunk(gemm_fc1_ep[ep_rank], num_tp_shards, dim=0)  # 沿 expert 维度切
    fc2_tp = torch.chunk(gemm_fc2_ep[ep_rank], num_tp_shards, dim=0)
```

**DeepSeek3 参考实现** (第 815-820 行):
```python
gemm_fc1_ep_tp = torch.chunk(gemm_fc1_ep[ep_rank], self.tp_size, dim=2)  # 沿 intermediate 维度切
gemm_fc2_ep_tp = torch.chunk(gemm_fc2_ep[ep_rank], self.tp_size, dim=1)
```

**Kimi2-specific 参考实现** (第 756-761 行):
```python
fc1_shards = torch.chunk(fc1_ep, self.expert_tp_size, dim=2)  # 沿 intermediate 维度切
fc2_shards = torch.chunk(fc2_ep, self.expert_tp_size, dim=1)
```

**分析**:

Megatron 的 `expert_tensor_parallel_size` 语义是：每个 expert 的权重沿**中间维度**（gate+up 的输出维度）切分，而非沿 expert 数量维度切分。

- **DeepSeek3/Kimi2-specific**: 切分 dim=2（intermediate），每个 TP rank 获得所有 expert 但权重被切片
- **当前代码**: 切分 dim=0（expert），每个 TP rank 获得不同的 expert 子集

当前代码的切分方式实际上是 `moe_tp_extend_ep` 的语义（TP 用于扩展 EP），而非 `expert_tp_size` 的语义。

**影响范围**: 仅 `expert_tp_size > 1` 时触发。

---

## 5. 缺失功能

### 5.1 `moe_tp_extend_ep` 支持

DeepSeek3 和 Kimi2-specific 都支持 `--moe-tp-extend-ep` 模式，其中 TP 和 EP 组合成有效的 expert 并行度：

```python
bucket_num = self.ep_size * self.tp_size
gemm_fc1_ep = torch.chunk(gemm_fc1_3d, bucket_num, dim=0)
idx = ep_rank * self.tp_size + tp_rank
```

当前转换器没有此功能。在 DualPipe + TP>1 场景下可能需要。

---

## 6. 三个实现的 Grouped GEMM 权重处理对比

以 EP=64, TP=2, expert_tp_size=1, num_experts=128, hidden=7168, moe_ffn=12288 为例：

### 6.1 Expert 权重构建

三个实现完全一致：
```python
fc1 = torch.cat([gate, up], dim=0)  # [24576, 7168]
experts_linear_fc1_list.append(fc1.t())  # [7168, 24576]
experts_linear_fc2_list.append(down.t())  # [12288, 7168]
```

### 6.2 3D 张量构建

| 实现 | 方法 | 结果 |
|------|------|------|
| DeepSeek3 | `torch.cat(list).view(hidden,-1).view(experts,hidden,-1)` | [128,7168,24576] |
| Kimi2-specific | `torch.cat(list).view(hidden,-1).view(experts,hidden,-1)` | [128,7168,24576] |
| 当前 | `torch.stack(list,dim=0).reshape(experts,hidden,-1)` | [128,7168,24576] |

结果相同，`torch.stack` 更直观。✓

### 6.3 EP 切分后保存（关键差异）

EP 切分: `[128,7168,24576]` → 64 × `[2,7168,24576]`

**DeepSeek3** (正确):
```python
# TP 沿 intermediate 维度切
gemm_fc1_ep_tp = torch.chunk(gemm_fc1_ep[ep_rank], self.tp_size, dim=2)
# [2,7168,24576] → 2 × [2,7168,12288]

# 直接 reshape（无 permute）
w1 = gemm_fc1_ep_tp[tp_rank].reshape(self.hidden_size, -1).clone()
# → [7168, 2×12288] = [7168, 24576]
```

**Kimi2-specific** (正确):
```python
# expert_tp_size=1 → 不切分
fc1_shards = [fc1_ep]  # [2,7168,24576]

# 直接 reshape（无 permute）
w1 = fc1_shards[expert_tp_idx].reshape(self.hidden_size, -1).clone()
# → [7168, 2×24576] = [7168, 49152]
```

**当前转换器** (有 Bug):
```python
# expert_tp_size=1 → 不切分
fc1_tp = [gemm_fc1_ep[ep_rank]]  # [2,7168,24576]

# 多了 permute！
w1 = fc1_tp[0].permute(1,0,2).contiguous().reshape(self.hidden_size, -1).clone()
# permute: [7168,2,24576] → reshape: [7168, 49152]
# 但内存布局不同！
```

### 6.4 内存布局可视化

以 2 个 expert、hidden=4、intermediate=3 为例：

```
原始 3D: [2, 4, 3]
e0: [a,b,c], [d,e,f], [g,h,i], [j,k,l]
e1: [m,n,o], [p,q,r], [s,t,u], [v,w,x]

无 permute (正确 - DeepSeek3/Kimi2):
reshape(4, -1) = [4, 6]
Row 0: [a,b,c, m,n,o]  ← e0_row0 + e1_row0... 
Wait, let me recalculate.

Actually reshape [2,4,3] → [4,6]:
Flat: a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x
Row 0: a,b,c,d,e,f         ← e0 rows 0,1 (only 2 rows since 6/3=2)
Row 1: g,h,i,j,k,l         ← e0 rows 2,3
Row 2: m,n,o,p,q,r         ← e1 rows 0,1
Row 3: s,t,u,v,w,x         ← e1 rows 2,3

Wait, that doesn't look right either. Let me be more precise.

[2, 4, 3] flat memory (C-contiguous):
e0_r0: a b c
e0_r1: d e f
e0_r2: g h i
e0_r3: j k l
e1_r0: m n o
e1_r1: p q r
e1_r2: s t u
e1_r3: v w x

.reshape(4, 6):
Total = 24 elements, 4×6 = 24 ✓
Row 0: a b c d e f   (e0_r0 + e0_r1)
Row 1: g h i j k l   (e0_r2 + e0_r3)
Row 2: m n o p q r   (e1_r0 + e1_r1)
Row 3: s t u v w x   (e1_r2 + e1_r3)
```

```
有 permute (当前代码 - 错误):
[2,4,3].permute(1,0,2) = [4,2,3]
Memory after contiguous:
r0: a b c m n o   (e0_r0 + e1_r0)
r1: d e f p q r   (e0_r1 + e1_r1)
r2: g h i s t u   (e0_r2 + e1_r2)
r3: j k l v w x   (e0_r3 + e1_r3)

.reshape(4, 6):
Row 0: a b c m n o   ← 同一 hidden 行，来自不同 expert
Row 1: d e f p q r
Row 2: g h i s t u
Row 3: j k l v w x
```

**对比**: 无 permute 的 Row 0 = `e0_r0 + e0_r1`（连续来自同一 expert），有 permute 的 Row 0 = `e0_r0 + e1_r0`（来自不同 expert 的同一行）。当 Megatron 用 `.view(num_experts, hidden, intermediate)` 恢复时，无 permute 能正确还原，有 permute 会交叉错乱。

---

## 7. Non-Grouped GEMM 路径分析

Non-grouped 路径的实现正确（在 expert_tp_size=1 默认配置下）：

```python
local_fc1 = experts_linear_fc1_list[global_idx].t()  # [24576, 7168]
local_fc2 = experts_linear_fc2_list[global_idx].t()  # [12288, 7168]
# expert_tp_size=1: 不切分
local_fc1_shards = [local_fc1.contiguous().clone()]
local_fc2_shards = [local_fc2.contiguous().clone()]
```

- fc1 的双重 `.t()` 恢复 Megatron ColumnParallel 格式 [out, in] ✓
- fc2 的双重 `.t()` 恢复 Megatron RowParallel 格式 [out, in] ✓
- 与 DeepSeek3/Kimi2-specific 一致 ✓

但 `expert_tp_idx` 公式（第 1291 行）同样有 Bug 2 的问题。

---

## 8. 代码质量对比

### 8.1 相比 Kimi2-specific 的改进

| 特性 | 当前转换器 | Kimi2-specific |
|------|-----------|---------------|
| 并行 I/O | ThreadPoolExecutor + ProcessPoolExecutor | 串行 |
| SHA256 校验 | 支持 | 不支持 |
| dtype 转换 | `--cast-dtype` | 不支持 |
| tie_word_embeddings | 支持 | 不支持 |
| HF 权重选择性加载 | `safe_open` + 指定 keys | 全文件加载 |
| 单文件 safetensors 支持 | 自动检测 | 不支持 |
| 进度日志 | 详细计时 | 简单 |

### 8.2 需要对齐的地方

| 逻辑 | 当前转换器 | Kimi2-specific | 应该 |
|------|-----------|---------------|------|
| weight1 reshape | `.permute(1,0,2).reshape(...)` | `.reshape(...)` | `.reshape(...)` (无 permute) |
| expert_tp_idx | `tp*etp//tp` | `tp%etp` | `tp%etp` |
| expert_tp>1 切分维度 | dim=0 (expert) | dim=2 (intermediate) | dim=2 (intermediate) |
| moe_tp_extend_ep | 不支持 | 支持 | 需要支持 |

---

## 9. 总结

| 组件 | 状态 | 说明 |
|------|------|------|
| GQA Attention | ✓ 正确 | Q/K/V 融合与 TP 切分正确 |
| Dense MLP | ✓ 正确 | gate/up 交织与 TP 切分正确 |
| MoE Router | ✓ 正确 | 兼容两种 bias 格式 |
| MoE Shared Experts | ✓ 正确 | 按 TP 正确切分 |
| MoE Expert (grouped, tp=1) | ❌ **Bug 1** | `.permute(1,0,2)` 导致内存布局错误 |
| MoE Expert (grouped, tp>1) | ❌ **Bug 1+2+3** | permute + 公式 + 维度错误 |
| MoE Expert (non-grouped, tp=1) | ✓ 正确 | 默认配置正确 |
| MoE Expert (non-grouped, tp>1) | ❌ **Bug 2** | expert_tp_idx 公式错误 |
| DualPipe/VPP | ✓ 正确 | 层映射正确 |
| Checkpoint 格式 | ✓ 正确 | 命名和 VPP 格式正确 |
| Checkpoint Args | ✓ 改进 | 写入完整训练配置 |
| moe_tp_extend_ep | ⚠ 缺失 | 不支持 |

### 修复优先级

1. **Bug 1（严重）**: 移除 `.permute(1, 0, 2).contiguous()` → 影响所有 grouped_gemm 转换
2. **Bug 2（中等）**: `expert_tp_idx = tp_rank % self.expert_tp_size` → 仅影响 expert_tp>1
3. **Bug 3（中等）**: 改为沿 dim=2 切分 intermediate → 仅影响 expert_tp>1
4. **缺失功能**: 添加 `--moe-tp-extend-ep` 支持
