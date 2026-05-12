# Kimi2 HF → MCore 转换代码 Review

> 对比文件: `utils/convert_kimi2_hf2mcore.py` vs `utils/convert_ckpt_deepseek3.py`
> 启动脚本: `scripts/ckpt_convert_kimi2_hf2mcore.sh`
> 训练配置: `scripts/pretrain_kimi2_1t_4k.sh`
> 分析日期: 2026-04-22 (更新)

---

## 1. 总览

| 项目 | 说明 |
|------|------|
| 转换脚本 | `utils/convert_kimi2_hf2mcore.py` |
| 启动脚本 | `scripts/ckpt_convert_kimi2_hf2mcore.sh` |
| 参考实现 | `utils/convert_ckpt_deepseek3.py` (DeepSeek-V3, MLA+MoE) |
| 架构差异 | **GQA+MoE** (非 MLA)，支持 QK LayerNorm，无 MTP 层 |

---

## 2. Shell 脚本 Review (`ckpt_convert_kimi2_hf2mcore.sh`)

### 2.1 默认参数与训练脚本一致性校验

| 参数 | 转换脚本默认值 | 训练脚本值 | 匹配 |
|------|---------------|-----------|------|
| NUM_LAYERS | 32 | 32 | ✓ |
| HIDDEN_SIZE | 7168 | 7168 | ✓ |
| NUM_ATTENTION_HEADS | 64 | 64 | ✓ |
| NUM_QUERY_GROUPS | 2 | 2 | ✓ |
| KV_CHANNELS | 128 | 128 | ✓ |
| FFN_HIDDEN_SIZE | 18432 | 18432 | ✓ |
| MOE_FFN_HIDDEN_SIZE | 12288 | 12288 | ✓ |
| NUM_EXPERTS | 128 | 128 | ✓ |
| FIRST_K_DENSE_REPLACE | 2 | 2 | ✓ |
| VOCAB_SIZE | 163840 | 163840 | ✓ |
| QK_LAYERNORM | 1 (启用) | 1 (启用) | ✓ |
| MOE_GROUPED_GEMM | 1 (启用) | 启用 | ✓ |
| TP | 2 | 2 | ✓ |
| PP | 8 | 8 | ✓ |
| EP | 32 | 64 | ⚠️ 减半 (见说明) |
| EXPERT_TP | 1 | 1 | ✓ |

> **EP 说明**: 训练使用 `--expert-model-parallel-size 64`，但 `moe_tp_extend_ep=True` 时 EP group 包含 TP 维度，纯 EP = 64/TP = 32。转换脚本的 EP 参数代表纯 EP 维度，因此默认值为 32。

### 2.2 注意事项

- **MOE_TP_EXTEND_EP 默认为 1 (启用)**: Shell 脚本中 `MOE_TP_EXTEND_EP="${MOE_TP_EXTEND_EP:-1}"` 默认启用，与训练配置一致。使用 `dualpipev + TP>1` 时此参数必须启用。
- **SAVE_DIR 命名**: 自动追加 `_tp{TP}_pp{PP}_ep{EP}` 后缀，避免覆盖。

### 2.3 三种并行模式支持

1. **DualPipeV 模式** (`SCHEDULES_METHOD=dualpipev`): 与训练脚本一致 ✓
2. **标准 VPP 模式** (`VPP_STAGE=N`): 通过环境变量控制 ✓
3. **纯 PP 模式**: 不设置 schedules-method 和 vpp-stage ✓

---

## 3. 转换代码详细 Review

### 3.1 类初始化与参数校验 (`__init__`, `_valid_parameter`)

**校验规则**:

| 校验项 | 规则 | 正确性 |
|--------|------|--------|
| first_k_dense_replace | ∈ [0, num_layers] | ✓ |
| num_experts % ep_size | == 0 | ✓ |
| num_attention_heads % tp_size | == 0 | ✓ |
| num_query_groups % tp_size | == 0 | ✓ |
| expert_tp_size ≤ tp_size | 数值约束 | ✓ |
| tp_size % expert_tp_size | == 0 | ✓ |
| moe_tp_extend_ep vs expert_tp_size>1 | 互斥 | ✓ |
| dualpipe + tp>1 | 警告 (非强制) | ✓ |
| num_layers % pp_size | == 0 | ✓ |
| num_layer_list vs vpp | 互斥 | ✓ |
| num_layer_list vs noop_layers | 互斥 | ✓ |

**与 DeepSeek3 差异**:
- DeepSeek3 强制 `dualpipe + tp>1` 时必须 `moe_tp_extend_ep` (raise ValueError)
- Kimi2 放宽为警告，更灵活
- DeepSeek3 硬编码 `first_k_dense_replace ≤ 3`，Kimi2 允许 [0, num_layers]

### 3.2 层映射逻辑

#### Pure PP 模式 (`get_pprank_hf_layeridxs`)

32 层、PP=8:
```
pp_rank=0: [0, 1, 2, 3]     ← 前 2 层为 Dense
pp_rank=1: [4, 5, 6, 7]
...
pp_rank=7: [28, 29, 30, 31]
```
与 DeepSeek3 相同的顺序映射逻辑。✓

#### DualPipe 模式 (`get_vpprank_hf_layeridxs`)

32 层、PP=8、vpp_size=2，每 PP stage 4 层:
```
dualpipe_layer_list 构建:
  layer_pop_num = 32/8/2 = 2
  迭代1: [0,1] + [30,31]
  迭代2: [2,3] + [28,29]
  ...
  迭代8: [14,15] + [16,17]

结果:
  pp_rank=0: vpp0=[0,1]   vpp1=[30,31]
  pp_rank=1: vpp0=[2,3]   vpp1=[28,29]
  ...
  pp_rank=7: vpp0=[14,15] vpp1=[16,17]
```
与 DeepSeek3 完全一致。✓

#### load_matched_hf_weights

| 条件 | 加载内容 |
|------|---------|
| pp_rank=0, 非 dualpipe | embed_tokens + 当前 PP 层 |
| pp_rank=0, dualpipe | embed_tokens + lm_head + model.norm + 当前 PP 层 |
| pp_rank=last, 非 dualpipe | lm_head + model.norm + 当前 PP 层 |

与 DeepSeek3 一致。DualPipe 模式下 embed 和 lm_head 都在 pp_rank=0。✓

**VPP 模式优化**: Kimi2 为每个 pp_rank 只调用一次 `load_matched_hf_weights` (line 1021)，获取所有 vpp_rank 需要的权重。DeepSeek3 为每个 vpp_rank 单独调用，可能重复加载同一 safetensors 文件。

### 3.3 Attention 层转换 (`set_model_layer_attn`, line 531-597)

#### GQA vs MLA 对比

**DeepSeek3 (MLA)**:
```
HF → MCore:
  q_a_proj + kv_a_proj_with_mqa → concat → linear_qkv (压缩)
  q_b_proj → linear_q_up_proj (上投影, TP 切分)
  kv_b_proj → linear_kv_up_proj (上投影, TP 切分)
  q_a_layernorm → q_layernorm
  kv_a_layernorm → kv_layernorm
```

**Kimi2 (GQA)**:
```
HF → MCore:
  q_proj → chunk(tp_size, dim=0) ──┐
  k_proj → chunk(tp_size, dim=0) ──┤ cat → linear_qkv
  v_proj → chunk(tp_size, dim=0) ──┘     (per TP shard)
  o_proj → chunk(tp_size, dim=1) → linear_proj
  q_layernorm → q_layernorm (可选)
  k_layernorm → k_layernorm (可选)
```

#### GQA QKV 组装 (TP=2)

```python
q_weight: [8192, 7168]  → chunk(2) → [4096, 7168] × 2
k_weight: [256, 7168]   → chunk(2) → [128, 7168] × 2
v_weight: [256, 7168]   → chunk(2) → [128, 7168] × 2
o_proj:   [7168, 8192]  → chunk(2, dim=1) → [7168, 4096] × 2

qkv_shards[i] = cat([q_tp[i], k_tp[i], v_tp[i]], dim=0)
  → [4096+128+128, 7168] = [4352, 7168] × 2
```

**Round-trip 验证**:
```
H2M: qkv_shards[i] = cat([q_tp[i], k_tp[i], v_tp[i]], dim=0)
M2H: split(qkv, [q_per_tp, k_per_tp, v_per_tp], dim=0) → cat(parts, dim=0)
→ 互为逆操作 ✓
```

**Shape 校验** (line 558-572):
```python
expected_q_rows = 64 * 128 = 8192
expected_k_rows = 2 * 128 = 256
expected_v_rows = 2 * 128 = 256
```
提供了维度校验防止配置错误。✓

**QK LayerNorm** (line 549-592):
```python
q_ln = weights_dict.pop(f"...q_layernorm.weight", None)
k_ln = weights_dict.pop(f"...k_layernorm.weight", None)
```
- 使用 `None` 默认值，兼容性 ✓
- LayerNorm 权重在 EP/TP 间复制 (不切分) ✓
- 丢弃 `rotary_emb.inv_freq` ✓

### 3.4 MLP 层转换 (`set_model_layer_mlp`, line 599-884)

#### Dense 层

```
gate_proj [18432, 7168] → chunk(tp, dim=0) → [9216, 7168] × 2
up_proj   [18432, 7168] → chunk(tp, dim=0) → [9216, 7168] × 2
fc1_shards = [cat(g, u)] = [18432, 7168] × 2  (interleaved gate+up)
down_proj [7168, 18432] → chunk(tp, dim=1) → [7168, 9216] × 2
```
SwiGLU 的 gate+up 交织，与 DeepSeek3 一致。✓

#### 自动检测 Dense/MoE 层 (line 604-621)

Kimi2 增加了基于 HF 实际键的自动检测:
```python
is_dense_layer = hf_layer_idx < self.first_k_dense_replace
has_moe_key = "mlp.gate.weight" in weights_dict
has_dense_key = "mlp.gate_proj.weight" in weights_dict
# 交叉校验: 配置与实际键冲突时自动纠正并警告
```
比 DeepSeek3 更鲁棒的防御性编程。✓

### 3.5 MoE Expert 权重处理

#### 3.5.1 单个 Expert 权重构建 (line 688-705)

```python
gate = [12288, 7168], up = [12288, 7168], down = [7168, 12288]

gate_chunks = chunk(gate, expert_tp_size, dim=0)
up_chunks = chunk(up, expert_tp_size, dim=0)
fc1 = interleave(zip(gate_chunks, up_chunks))  # [24576, 7168]
fc1.t() → [7168, 24576]  # 存入 list
down.t() → [12288, 7168]  # 存入 list
```

**Gate/Up 交织模式** (expert_tp_size=1):
```
fc1 = [gate, up] = [24576, 7168]  → .t() → [7168, 24576]
```

**Gate/Up 交织模式** (expert_tp_size=2):
```
gate_chunks = [g0=[6144,7168], g1=[6144,7168]]
up_chunks   = [u0=[6144,7168], u1=[6144,7168]]
zip + interleave: [g0, u0, g1, u1] = [24576, 7168]  → .t() → [7168, 24576]
```

#### 3.5.2 Grouped GEMM + moe_tp_extend_ep (line 743-765)

**Shape 推导** (EP=32, TP=2, expert_tp_size=1, num_experts=128):
```
Step 1: 每个专家 fc1 = [7168, 24576], fc2 = [12288, 7168]

Step 2: 拼接
  gemm_fc1 = cat(128 × [7168, 24576]).view(7168, 24576*128) = [7168, 3145728]
  gemm_fc1_3d = view(128, 7168, 24576)

Step 3: 分桶
  bucket_num = ep_size * tp_size = 64
  gemm_fc1_ep = chunk(128 × [7168, 24576], 64, dim=0) → 64 × [2, 7168, 24576]

Step 4: 分配到 (ep_rank, tp_rank)
  idx = ep_rank * 2 + tp_rank
  w1 = reshape(7168, -1) = [7168, 49152]  (2 experts)
  w2 = reshape(-1, 7168) = [24576, 7168]  (2 experts)
```

**Round-trip 验证**:
```
H2M: 128 experts → chunk(64) → bucket[ep*2+tp] → save to global_ep path (64 dirs)
M2H: load from global_ep path (64 dirs) → expert_idx = ep*2+tp → 恢复正确顺序 ✓
```

#### 3.5.3 Grouped GEMM + expert_tp_size > 1 (line 766-810)

```python
fc1_shards = chunk(fc1_ep, expert_tp_size, dim=2)
fc2_shards = chunk(fc2_ep, expert_tp_size, dim=1)
for tp_rank in range(tp_size):
    expert_tp_idx = tp_rank % expert_tp_size
    w1 = fc1_shards[expert_tp_idx].reshape(hidden_size, -1)
    w2 = fc2_shards[expert_tp_idx].reshape(-1, hidden_size)
```

当 `expert_tp_size=1` 时: 所有 TP rank 获得相同权重 (shards[0]) ✓
当 `expert_tp_size>1` 时: 每个 TP rank 获得其唯一分片 ✓

**Round-trip 验证** (expert_tp_size=2):
```
H2M: gate=[12288,7168] → chunk(2) → g0=[6144,7168], g1=[6144,7168]
     interleave → [g0,u0,g1,u1] → .t() → [7168,24576]
     grouped: [7168,24576] per expert → chunk(ep, dim=0) → [7168,24576]
     then chunk(expert_tp_size=2, dim=2) → shard[0]=[7168,12288], shard[1]=[7168,12288]
     tp_rank 0 → shard[0], tp_rank 1 → shard[1]

M2H: read shard[0]=[7168,12288], shard[1]=[7168,12288]
     cat → [7168,24576] → .t() → [24576,7168]
     chunk(expert_tp_size=2) → [12288,7168], [12288,7168]
     each chunk(2) → g_i, u_i
     gate = cat([g0,g1]) = [12288,7168] ✓
```

#### 3.5.4 Non-Grouped GEMM 路径

**moe_tp_extend_ep** (line 814-844):
```python
bucket_num = ep_size * tp_size
num_local_experts = num_experts // bucket_num
for ep_rank in range(ep_size):
    for tp_rank in range(tp_size):
        global_base = (ep_rank * tp_size + tp_rank) * num_local_experts
```
与 grouped_gemm 路径使用相同的索引映射。✓

注意这里 `.t()` 双重转置: list 中存储 `.t()` 后的值，取出时再 `.t()` 恢复，与 DeepSeek3 一致。

**expert_tp_size > 1** (line 855-884):
```python
fc1_shards = chunk(local_fc1, expert_tp_size, dim=0)
fc2_shards = chunk(local_fc2, expert_tp_size, dim=1)
for tp_rank in range(tp_size):
    expert_tp_idx = tp_rank % expert_tp_size
    mg_model[ep][tp][fc1_key] = fc1_shards[expert_tp_idx]
```
正确分片。✓

#### 3.5.5 Shared Expert 处理

```python
shared_gate_chunks = chunk(shared_gate, tp_size, dim=0)
shared_up_chunks = chunk(shared_up, tp_size, dim=0)
shared_fc1_shards = [cat([g, u]) for g, u in zip(gate_chunks, up_chunks)]
shared_fc2_shards = chunk(shared_down, tp_size, dim=1)
```
Shared expert 始终按 `tp_size` 切分 (不受 `expert_tp_size` 影响)。✓

> **效率改进**: DeepSeek3 将 shared expert 切分放在 expert 循环内 (重复 num_experts 次)，Kimi2 正确移到循环外。

### 3.6 Router 权重处理

```python
router_w = weights_dict.pop("mlp.gate.weight")
if router_w.shape[0] != self.num_experts:
    router_w = router_w[:self.num_experts, :].clone()  # 条件截断

router_b = weights_dict.pop("mlp.gate.e_score_correction_bias", None)  # 可选
weights_dict.pop("mlp.gate.bias", None)  # 兼容旧格式
```
- Router 权重条件截断 (比 DeepSeek3 的无条件截断更优) ✓
- Router bias 可选 ✓
- 兼容旧格式的 bias 键 ✓

### 3.7 Checkpoint 保存

#### 3.7.1 目录命名 (`generate_mg_weights_dir`)

**moe_tp_extend_ep + tp_size > 1**:
```python
global_ep = tp_rank + ep_rank * self.tp_size
prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{global_ep:03}"
```

与 M2H 的 `_mp_prefix` 使用相同公式。✓

#### 3.7.2 保存内容

```python
torch.save({
    'model': mg_model[ep_rank][tp_rank],
    'checkpoint_version': 3.0,
    'iteration': 1,
    'args': self._build_checkpoint_args(),  # Kimi2 新增
}, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)
```

**改进**: Kimi2 增加了 `'args'` 字段，DeepSeek3 没有此字段。

#### 3.7.3 `_build_checkpoint_args` 完整性校验

| Args 字段 | Kimi2 设置 | 训练脚本值 | 匹配 |
|-----------|-----------|-----------|------|
| hidden_size | 7168 | 7168 | ✓ |
| ffn_hidden_size | 18432 | 18432 | ✓ |
| num_attention_heads | 64 | 64 | ✓ |
| num_query_groups | 2 | 2 | ✓ |
| kv_channels / qk_head_dim | 128 | 128 | ✓ |
| v_head_dim | 128 | 128 | ✓ |
| num_experts | 128 | 128 | ✓ |
| moe_ffn_hidden_size | 12288 | 12288 | ✓ |
| first_k_dense_replace | 2 | 2 | ✓ |
| n_shared_experts | 1 | 1 | ✓ |
| moe_router_topk | 2 | 2 | ✓ |
| moe_router_num_groups | 8 | 8 | ✓ |
| moe_router_group_topk | 2 | 2 | ✓ |
| moe_router_topk_scaling_factor | 2.827 | 2.827 | ✓ |
| moe_router_enable_expert_bias | True | 启用 | ✓ |
| swiglu | True | 启用 | ✓ |
| untie_embeddings_and_output_weights | True | 启用 | ✓ |
| position_embedding_type | 'rope' | rope | ✓ |
| normalization | 'RMSNorm' | RMSNorm | ✓ |
| add_bias_linear | False | disable-bias-linear | ✓ |
| norm_epsilon | 1e-6 | 1e-6 | ✓ |
| bf16 | True | bf16 | ✓ |
| rotary_base | 50000.0 | 50000 | ✓ |
| vocab_size | 163840 | 163840 | ✓ |
| use_distributed_optimizer | True | 启用 | ✓ |
| qk_layernorm | 由参数控制 | 启用 | ✓ |
| use_mcore_models | True | 启用 | ✓ |

所有配置完全匹配。✓

**注意**: 以下训练配置未包含在 checkpoint args 中 (不影响加载):
- `moe-router-score-function sigmoid` — 运行时参数
- `moe-router-dtype fp32` — 运行时参数
- `swa-windows 128` — 推理参数
- RoPE scaling 参数 (`yarn`, `factor=32`, `mscale` 等) — 加载时需命令行指定

#### 3.7.4 VPP 保存格式

```python
model_dict = {
    'checkpoint_version': 3.0,
    'iteration': 1,
    'args': ...,
}
for vpp_rank in range(self.vpp_size):
    model_dict[f"model{vpp_rank}"] = mg_model[vpp_rank][ep_rank][tp_rank]
```
标准 Megatron VPP 保存格式。✓

**Postprocess 放置**:
- DualPipe: norm + lm_head 在 pp_rank=0, vpp_rank=last ✓
- 标准 PP: norm + lm_head 在 pp_rank=last, vpp_rank=last ✓

#### 3.7.5 未消费权重检查

```python
if pp_weights:
    unconsumed = list(pp_weights.keys())
    raise ValueError(f"存在未被消费的 HF 权重 ({len(unconsumed)}): {unconsumed[:20]}")
```
DeepSeek3 没有此检查。防止遗漏权重。✓

---

## 4. 与 DeepSeek3 参考实现的详细对比

### 4.1 设计性差异 (非 Bug)

| 特性 | Kimi2 | DeepSeek3 | 说明 |
|------|-------|-----------|------|
| 注意力 | GQA (q/k/v_proj + o_proj) | MLA (q_a/kv_a + q_b/kv_b) | 根本架构差异 |
| QK LayerNorm | q_layernorm + k_layernorm | q_a_layernorm + kv_a_layernorm | 各自正确 |
| MTP 层 | 无 | 支持 | Kimi2 不需要 |
| MLA mm_split | 无 | 支持 | Kimi2 不需要 |
| LoRA merge | 无 | 支持 | Kimi2 不需要 |
| expert_tp_size | 可配置 (默认 1) | 等价于 tp_size | Kimi2 更灵活 |
| NPU 硬编码 | 无 | .to('npu').cpu() | Kimi2 跨平台 |
| bitsandbytes | 条件导入 | 顶层导入 | Kimi2 更友好 |
| dualpipe 验证 | 警告 (非强制) | 强制 moe_tp_extend_ep | Kimi2 更灵活 |

### 4.2 代码质量改进

| 改进点 | Kimi2 | DeepSeek3 |
|--------|-------|-----------|
| Dense/MoE 检测 | 自动检测 + 警告 | 仅依赖索引 |
| Router bias | 可选 (None default) | 必选 |
| Router weight 截断 | 条件截断 | 无条件截断 |
| 未消费权重检查 | 有 | 无 |
| checkpoint args | 保存完整配置 | 未保存 |
| Shared expert 效率 | expert 循环外 | expert 循环内 (重复) |
| gc.collect() | 使用 | 未使用 |
| VPP 权重加载 | 一次加载所有 vpp 的权重 | 每个 vpp_rank 单独加载 |

### 4.3 关键逻辑一致性

| 逻辑路径 | Kimi2 vs DeepSeek3 | 一致性 |
|---------|-------------------|--------|
| Dense MLP gate/up 交织 | 相同 | ✓ |
| Expert fc1 交织 (zip interleave) | 等效 (相同内存布局) | ✓ |
| Expert .t() 双重转置 | 相同 | ✓ |
| Grouped GEMM 3D view + reshape | 相同 | ✓ |
| EP 切分 (dim=0) | 相同 | ✓ |
| TP 切分 (dim=2 fc1, dim=1 fc2) | 相同 | ✓ |
| moe_tp_extend_ep 桶映射 | 相同 (ep*tp 线性映射) | ✓ |
| DualPipe 层映射 | 相同 (前后半分配) | ✓ |
| Checkpoint 目录命名 | 相同格式 | ✓ |
| VPP 保存 (model0, model1) | 相同 | ✓ |

---

## 5. 发现的问题

### 5.1 无功能性 Bug

经过逐行对比和 Shape 推导验证，**未发现功能性 Bug**。所有权重转换逻辑、TP/EP/PP 切分、保存格式均正确。

### 5.2 代码正确性总结

| 转换路径 | 状态 | 说明 |
|---------|------|------|
| GQA QKV 拼接与 TP 切分 | ✅ | 正确的 split-cat 逻辑 |
| Dense MLP SwiGLU gate/up 交织 | ✅ | 正确 |
| Dense/MoE 自动检测 | ✅ | 交叉校验 + 警告 |
| MoE expert 权重构建与 TP 切分 | ✅ | 正确 |
| MoE shared expert 与独立 expert TP 差异 | ✅ | shared 按 tp_size, expert 按 expert_tp_size |
| MoE grouped_gemm + moe_tp_extend_ep | ✅ | 正确的分桶和索引映射 |
| MoE grouped_gemm + expert_tp_size | ✅ | 正确的分片 |
| MoE non-grouped + expert_tp_size | ✅ | 正确的 gather 逻辑 |
| DualPipe 层映射与权重放置 | ✅ | 正确 |
| Pure PP / VPP / DualPipe 三模式 | ✅ | 正确 |
| Noop 层处理 | ✅ | 正确 |
| Checkpoint args 完整性 | ✅ | 与训练配置匹配 |
| 未消费权重检测 | ✅ | 正确 |
| QK LayerNorm 可选支持 | ✅ | 正确 |
| gc.collect() 内存管理 | ✅ | 正确 |

### 5.3 建议改进 (非 Bug)

#### 5.3.1 Shell 脚本 MOE_TP_EXTEND_EP 默认值

Shell 脚本中 `MOE_TP_EXTEND_EP` 默认为 `1` (启用)，与训练配置一致。使用 `dualpipev + TP>1` 时此参数必须启用，默认已正确配置。

#### 5.3.2 `_build_checkpoint_args` 缺少部分训练参数

以下训练配置中的 RoPE scaling 参数未包含在 checkpoint args 中:
```
--beta-fast 1 --beta-slow 1
--rope-scaling-factor 32
--rope-scaling-mscale 1.0 --rope-scaling-mscale-all-dim 1.0
--rope-scaling-original-max-position-embeddings 4096
--rope-scaling-type yarn
```
这些是运行时参数，加载 checkpoint 时需通过命令行重新指定。不影响权重转换正确性，但建议后续补充。

#### 5.3.3 num_query_groups 与 tp_size 约束

GQA 下 `num_query_groups=2` 限制了 `tp_size` 最大为 2。代码已有校验，但建议在 Shell 脚本注释中说明。

#### 5.3.4 `moe_tp_extend_ep` 的必要性

当使用 `dualpipev + TP>1` 时，`moe_tp_extend_ep` 是必须的。DeepSeek3 参考代码中有强制校验:
```python
if self.dualpipe:
    if self.tp_size > 1 and not self.moe_tp_extend_ep:
        raise ValueError('When dualpipe is enabled, moe-tp-extend-ep should be used')
```
Kimi2 H2M converter 放宽为警告。Shell 脚本默认已启用 `MOE_TP_EXTEND_EP=1`，与训练配置一致，因此使用默认参数时不会触发此警告。

#### 5.3.5 `rotary_base` 参数

`convert_kimi2_hf2mcore.py` 保留了 `rotary_base` 参数（默认 50000.0），但 `convert_kimi2_mcore2hf.py` 已移除该参数。虽然不影响功能，但双向转换器的参数集不对称。考虑在 M2H 方向也移除（不影响权重转换）。

---

## 6. 结论

### 6.1 总体评价

`convert_kimi2_hf2mcore.py` 实现质量较高:

1. **架构适配正确**: GQA 注意力、SwiGLU MLP、MoE expert/shared expert 均正确实现
2. **并行支持完整**: TP/PP/EP/VPP/DualPipe 五种并行维度
3. **防御性编程**: Dense/MoE 自动检测、shape 校验、未消费权重检查
4. **Round-trip 一致**: 与 `convert_kimi2_mcore2hf.py` 构成正确的双向转换
5. **代码质量优于参考**: 多处改进 (条件导入、checkpoint args、shared expert 效率、未消费检查)

### 6.2 未发现 Bug

所有转换逻辑经过逐步 Shape 推导验证和 Round-trip 一致性检查，**未发现功能性 Bug**。

### 6.3 推荐使用

```bash
# 与训练配置一致 (注意手动启用 moe_tp_extend_ep)
TP=2 PP=8 EP=64 EXPERT_TP=1 \
MOE_TP_EXTEND_EP=1 \
SCHEDULES_METHOD=dualpipev \
bash scripts/ckpt_convert_kimi2_hf2mcore.sh
```
