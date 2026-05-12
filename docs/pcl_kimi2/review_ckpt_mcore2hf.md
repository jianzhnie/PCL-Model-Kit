# Kimi2 MCore -> Huggingface 转换代码 Review

> 目标文件: `utils/convert_ckpt_mcore2hf.py` + `scripts/ckpt_convert_mcore2hf.sh`
> 参考实现: `utils/convert_ckpt_deepseek3_mcore2hf.py` (DeepSeek3 M2H)

---

## 1. 架构差异总览

| 配置项 | Kimi2-1T | DeepSeek3 |
|--------|----------|-----------|
| 注意力机制 | **GQA** (q/k/v_proj) | **MLA** (q_a/kv_a_proj, q_b/kv_b_proj) |
| 层数 | 32 | 61 |
| num_attention_heads | 64 | 128 |
| num_query_groups (KV heads) | 2 | 1 (MQA in MLA) |
| num_experts | 128 | 256 |
| first_k_dense_replace | 2 | 3 |
| expert_tp_size | 1 | 无 (隐含 = tp_size) |
| QK LayerNorm | q_layernorm, k_layernorm | q_a_layernorm, kv_a_layernorm |
| MTP 层 | 不支持 | 支持 |
| LoRA | 不支持 | 支持 |
| torch_npu | 不依赖 | 依赖 |

---

## 2. 注意力转换分析（正确）

### GQA QKV 拆分

```python
# MCore 融合 QKV → HF 分离 Q/K/V
q_per_tp = heads_per_tp * q_head_dim          # (64/2) * 128 = 4096
k_per_tp = kv_heads_per_tp * q_head_dim       # (2/2) * 128  = 128
v_per_tp = kv_heads_per_tp * self.v_head_dim  # (2/2) * 128  = 128
q_r, k_r, v_r = torch.split(qkv_shard, [q_per_tp, k_per_tp, v_per_tp], dim=0)
```

**TP=2 聚合验证**：
- Q: cat([4096, 7168] x 2) = [8192, 7168] = [64*128, 7168] ✓
- K: cat([128, 7168] x 2) = [256, 7168] = [2*128, 7168] ✓
- V: cat([128, 7168] x 2) = [256, 7168] = [2*128, 7168] ✓
- O: cat([7168, 4096] x 2, dim=1) = [7168, 8192] = [hidden, 64*128] ✓

**与 H2M 互为逆操作验证**：
- H2M: `chunk(q, tp, dim=0)` → M2H: `cat(q_parts, dim=0)` ✓
- H2M: `cat([q_tp, k_tp, v_tp], dim=0)` → M2H: `split(qkv, [q, k, v], dim=0)` ✓

### QKV 布局自动检测

代码实现了 `_infer_qkv_layout`，当 shard 形状与默认参数不匹配时自动推断。验证：
- TP=2 时 shard_rows = 4096+128+128 = 4352
- heads_per_tp = 32, 尝试 q_head_dim=128
- q_per_tp = 32*128 = 4096, rem = 256, denom = 256
- kv_heads_per_tp = 1 → detected_num_kv_heads = 2 = num_query_groups ✓

### QK LayerNorm 处理

```python
q_ln = models[(0, 0)].pop(q_norm_key, None)
k_ln = models[(0, 0)].pop(k_norm_key, None)
```

不参与 TP 切分，使用 `pop(..., None)` 安全处理 ✓

---

## 3. MLP 转换分析

### 3.1 Dense MLP（正确）

```python
fc1 = self._gather_tp_row(models, f'{prefix}.linear_fc1.weight')
fc2 = self._gather_tp_col(models, f'{prefix}.linear_fc2.weight')
gate, up = torch.chunk(fc1, 2, dim=0)
```

与 DeepSeek3 实现一致 ✓

### 3.2 MoE Router

**Router 重建** (`_reconstruct_router_lazy`)：
1. 先尝试从 base_models (ep_rank=0) 获取完整 router
2. 若 shape[0] != num_experts，按 EP 分片重建
3. 从各 EP rank 加载 local_experts 个 router 行

**Router Bias**：使用 try/except 安全处理可选的 `expert_bias` ✓

### 3.3 Shared Experts（正确）

```python
shared_fc1 = self._gather_tp_row(models, f'{prefix}.shared_experts.linear_fc1.weight')
shared_fc2 = self._gather_tp_col(models, f'{prefix}.shared_experts.linear_fc2.weight')
```

与 Dense MLP 处理方式一致 ✓

---

## 4. 发现的 Bug

### BUG 1（严重）：缺少 expert_tp_size 参数，expert 权重处理错误

**根因**：`MgCkptConvert` 没有 `expert_tp_size` 参数。当 `expert_tp_size=1` 时，所有 TP rank 持有**相同**的 expert 权重（不是 TP 分片），但转换器假设多个 TP owner 意味着权重被 TP 分片。

**影响**：Kimi2 训练配置 TP=2, EP=64, expert_tp_size=1。

**失败点 1 — Router 重建崩溃**（第 750-754 行）：

```python
owners = self._tp_ranks_for_ep(pp_rank, ep)  # 返回 [0, 1]（两个 TP rank）
if len(owners) != 1:
    raise ValueError('router 不支持跨 TP 分片重建')  # 必然触发！
```

`_tp_ranks_for_ep` 对每个 ep_rank 返回所有拥有该 EP 文件的 TP rank。TP=2 且 expert_tp_size=1 时，每个 ep_rank 都有两个 TP owner → ValueError。

**失败点 2 — Expert 权重维度翻倍**（grouped_gemm 路径，第 999-1002 行）：

```python
# owners = [0, 1], 两个 TP rank 持有完全相同的 expert 权重
shards_w1 = [w1_from_tp0, w1_from_tp1]  # 完全相同！
local_w1 = torch.cat(shards_w1, dim=1)   # dim=1 翻倍 → 错误！
```

**失败点 3 — Expert 权重维度翻倍**（non-grouped_gemm 路径，第 1044-1052 行）：

```python
# owners = [0, 1]
fc1_parts = [fc1_tp0, fc1_tp1]  # 完全相同！
fc1 = torch.cat(fc1_parts, dim=0)  # dim=0 翻倍 → 错误！
```

**数据流示例**（TP=2, EP=64, expert_tp_size=1）：

```
MCore 存储（每个 (tp, ep) 文件）:
  experts.weight1: [7168, 2 * (12288*2)] = [7168, 49152]  (2 local experts)
  experts.weight2: [2 * 12288, 7168] = [24576, 7168]

tp_rank=0, ep_rank=0: weight1 = [7168, 49152]  ← expert 权重
tp_rank=1, ep_rank=0: weight1 = [7168, 49152]  ← 完全相同（expert_tp=1）

当前代码:
  shards_w1 = [w1_tp0, w1_tp1]  # 两个 [7168, 49152]，完全相同
  local_w1 = cat(shards_w1, dim=1)  # [7168, 98304]  ← 翻倍！错误！

正确做法:
  local_w1 = shards_w1[0]  # [7168, 49152]  ← 只取一个
```

**修复方案**：添加 `expert_tp_size` 参数，当 `expert_tp_size < tp_size` 时每个 EP rank 只取一个 TP owner 的 expert 权重。

### BUG 2（次要）：标准 VPP Noop 层映射公式错误

**代码位置**：`_build_vpprank_layer_map` 非 dualpipe 分支（第 579-584 行）

```python
# 当前公式（VPP-first 布局，与 PP-first 分配不一致）:
vpp_idx = layer // self.vpp_stage // self.pp_size
pp_idx = (layer % (self.pp_size * self.vpp_stage)) // self.vpp_stage
```

层分配使用 PP-first（第 588 行注释也确认），但公式是 VPP-first。

**修复**：
```python
layers_per_pp = self.num_layers // self.pp_size
pp_idx = layer // layers_per_pp
vpp_idx = (layer % layers_per_pp) // self.vpp_stage
```

**验证**（num_layers=8, pp=2, vpp=2, stage=2, noop=3）：
- PP-first 分配: pp0_vpp0=[0,1], pp0_vpp1=[2,3], pp1_vpp0=[4,5], pp1_vpp1=[6,7]
- 层 3 应在 pp0_vpp1
- 当前: pp=1, vpp=0 → pp1_vpp0 ❌
- 修复: pp=0, vpp=1 → pp0_vpp1 ✓

**实际影响**：Kimi2 使用 DualPipeV 且无 noop 层，此 bug 不影响默认场景。

---

## 5. Shell 脚本验证

### ckpt_convert_mcore2hf.sh — 所有参数与训练配置匹配 ✓

| 参数 | 脚本值 | 训练值 | 匹配 |
|------|--------|--------|------|
| TP | 2 | 2 | ✓ |
| PP | 8 | 8 | ✓ |
| EP | 64 | 64 | ✓ |
| NUM_LAYERS | 32 | 32 | ✓ |
| HIDDEN_SIZE | 7168 | 7168 | ✓ |
| NUM_ATTENTION_HEADS | 64 | 64 | ✓ |
| NUM_QUERY_GROUPS | 2 | 2 | ✓ |
| QK_HEAD_DIM | 128 | 128 | ✓ |
| NUM_EXPERTS | 128 | 128 | ✓ |
| FIRST_K_DENSE_REPLACE | 2 | 2 | ✓ |
| SCHEDULES_METHOD | dualpipev | dualpipev | ✓ |
| VPP_STAGE | 2 | auto(=32/8/2) | ✓ |
| ROTARY_BASE | 50000 | 50000 | ✓ |

---

## 6. 总结

| 组件 | 状态 | 说明 |
|------|------|------|
| GQA 注意力 | ✓ 正确 | Q/K/V 拆分与聚合正确 |
| Dense MLP | ✓ 正确 | 与 DeepSeek3 一致 |
| Shared Experts | ✓ 正确 | 与 Dense MLP 一致 |
| MoE Router | ❌ 需修复 | BUG1: expert_tp_size=1 时崩溃 |
| MoE Expert (grouped_gemm) | ❌ 需修复 | BUG1: 权重维度翻倍 |
| MoE Expert (non-grouped) | ❌ 需修复 | BUG1: 权重维度翻倍 |
| QK LayerNorm | ✓ 正确 | 安全处理可选字段 |
| DualPipeV 层映射 | ✓ 正确 | 与 DeepSeek3 算法一致 |
| 标准 VPP noop | ⚠️ 次要 | BUG2: 公式与分配顺序不一致 |
| Shell 脚本 | ✓ 正确 | 参数与训练配置匹配 |
