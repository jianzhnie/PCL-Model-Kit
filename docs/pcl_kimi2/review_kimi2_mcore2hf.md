# Kimi2 MCore → HF 权重转换 Review 报告

> 对比文件: `utils/convert_kimi2_mcore2hf.py` vs `utils/convert_ckpt_deepseek3_mcore2hf.py`
> 启动脚本: `scripts/ckpt_convert_kimi2_mcore2hf.sh`
> 训练配置: `scripts/pretrain_kimi2_1t_4k.sh`
> 分析日期: 2026-04-22 (更新)

---

## 1. 整体架构差异 (设计差异，非 Bug)

| 特性 | DeepSeek3 | Kimi2 | 说明 |
|------|-----------|-------|------|
| 注意力机制 | **MLA** (Multi-head Latent Attention) | **GQA** (Grouped Query Attention) | 核心架构差异 |
| 专家数量 | 256 | 128 | 模型配置差异 |
| Dense 层数 | 3 | 2 | 模型配置差异 |
| 总层数 | 61 | 32 | 模型配置差异 |
| MTP 层 | 支持 (`mtp_flag`, `mtp.layers.*`) | 不需要 | Kimi2 无 MTP |
| LoRA/QLoRA | 支持 (`_merge_lora`, `save_lora_to_hf`) | 不需要 | 简化 |
| `mla_mm_split` | 支持 (MLA 专用) | 不需要 | GQA 不需要 |
| `expert_tp_size` | 无 (等价于 `expert_tp_size = tp_size`) | 支持 (默认 `expert_tp_size=1`) | **关键差异** |
| `torch_npu` | 导入 (昇腾 NPU) | 不导入 | CPU 环境兼容 |
| QK LayerNorm | MLA 内置 (q_a_layernorm, kv_a_layernorm) | GQA 可选 (q_layernorm, k_layernorm) | 不同命名 |
| Router bias | 必选 (`.pop()` 无默认值) | 可选 (`.pop(key, None)`) | 更鲁棒 |
| 全局变量 | 使用 `global TENSOR_SIZE`, `hf_weight_dict` | 实例属性 `self._tensor_size`, `self._hf_weight_dict` | 消除全局状态 |
| 路径生成 | 内联 if-else | 独立函数 `_mp_prefix()` | 更清晰 |
| 加载安全 | `weights_only=False` | 优先 `weights_only=True`，失败后 fallback | 更安全 |
| 文件检查 | 无 | `FileNotFoundError` + 配置提示 | 更友好 |
| VPP I/O | 每个 vpp_rank 重新读取文件 | 文件只读一次，`.pop()` 提取 vpp 模型 | 性能优化 |

---

## 2. Shell 脚本 Review (`ckpt_convert_kimi2_mcore2hf.sh`)

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
| MOE_TP_EXTEND_EP | 1 (启用) | 启用 | ✓ |
| TP | 2 | 2 | ✓ |
| PP | 8 | 8 | ✓ |
| EP | 32 | 64 | ⚠️ 减半 (见说明) |
| EXPERT_TP | 1 | 1 | ✓ |

> **EP 说明**: 训练使用 `--expert-model-parallel-size 64`，但 `moe_tp_extend_ep=True` 时 EP group 包含 TP 维度，纯 EP = 64/TP = 32。转换脚本的 EP 参数代表纯 EP 维度，因此默认值为 32。

### 2.2 三种并行模式支持

1. **DualPipeV 模式** (默认, `SCHEDULES_METHOD=dualpipev`): 训练脚本使用的模式 ✓
2. **标准 VPP 模式** (`VPP_STAGE=N`): 通过环境变量控制 ✓
3. **纯 PP 模式** (`SCHEDULES_METHOD=""`): 不设置 schedules-method ✓

---

## 3. 转换代码详细 Review

### 3.1 类初始化与参数校验

**GQA 维度计算** (line 130-136):
```python
self.q_head_dim = kv_channels         # 128
self.k_head_dim = kv_channels         # 128
self.v_head_dim = kv_channels         # 128
self.q_proj_rows = num_attention_heads * kv_channels   # 64 * 128 = 8192
self.k_proj_rows = num_query_groups * kv_channels      # 2 * 128 = 256
self.v_proj_rows = num_query_groups * kv_channels      # 2 * 128 = 256
self.qkv_proj_rows = 8192 + 256 + 256 = 8704
```

**实例属性** (替代全局变量):
```python
self._tensor_size = 0          # 替代 global TENSOR_SIZE
self._hf_weight_dict = {}      # 替代 global hf_weight_dict
```
消除了全局状态泄漏风险，支持多次调用 `run()` 和并发场景。✓

**EP 遍历范围** (line 170-175):
```python
self.ep_rank_list = list(range(self.ep_size))
```
始终遍历完整 ep_size。With moe_tp_extend_ep, 每个 (tp_rank, ep_rank) 对应唯一目录。✓

### 3.2 EP 目录路径构建 (`_mp_prefix`, line 53-75)

```python
def _mp_prefix(tp_rank, pp_rank, ep_rank, tp_size, pp_size, ep_size,
               moe_tp_extend_ep=False):
    if moe_tp_extend_ep and tp_size > 1:
        ep_suffix = tp_rank + ep_rank * tp_size
        if pp_size == 1:
            return f'mp_rank_{tp_rank:02}_{ep_suffix:03}'
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_suffix:03}'
    if ep_size == 1 and pp_size == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep_size == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp_size == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'
```

**关键设计**: `moe_tp_extend_ep` 检查优先于 `ep_size==1` 检查。这确保了即使 `ep_size=1`，当 `moe_tp_extend_ep=True` 时仍使用 `global_ep` 作为后缀，与 H2M 的 `generate_mg_weights_dir` 完全一致。

**验证** (EP=32, TP=2, PP=8):
```
(tp=0, ep=0)  → global_ep=0  → mp_rank_00_000_000
(tp=1, ep=0)  → global_ep=1  → mp_rank_01_000_001
(tp=0, ep=1)  → global_ep=2  → mp_rank_00_000_002
...
(tp=0, ep=31) → global_ep=62 → mp_rank_00_007_062
(tp=1, ep=31) → global_ep=63 → mp_rank_01_007_063
总计: 64 个唯一目录 (匹配训练 checkpoint) ✓
```

**所有 8 种组合验证**:

| moe_tp_extend_ep | ep_size | pp_size | H2M vs M2H |
|---|---|---|---|
| True | 1 | 1 | `mp_rank_{tp}_{global_ep}` ✓ |
| True | 1 | >1 | `mp_rank_{tp}_{pp}_{global_ep}` ✓ |
| True | >1 | 1 | `mp_rank_{tp}_{global_ep}` ✓ |
| True | >1 | >1 | `mp_rank_{tp}_{pp}_{global_ep}` ✓ |
| False | 1 | 1 | `mp_rank_{tp}` ✓ |
| False | 1 | >1 | `mp_rank_{tp}_{pp}` ✓ |
| False | >1 | 1 | `mp_rank_{tp}_{ep}` ✓ |
| False | >1 | >1 | `mp_rank_{tp}_{pp}_{ep}` ✓ |

### 3.3 Attention 转换 (`set_model_attn`, line 557-638)

#### GQA QKV 拆分逻辑 (TP=2):

```python
heads_per_tp = 64 / 2 = 32
kv_heads_per_tp = 2 / 2 = 1
q_per_tp = 32 * 128 = 4096
k_per_tp = 1 * 128 = 128
v_per_tp = 1 * 128 = 128

# 从 linear_qkv 拆分:
q_r, k_r, v_r = torch.split(cur_qkv, [4096, 128, 128], dim=0)

# 合并 TP shards:
q_proj = cat([q_r_tp0, q_r_tp1], dim=0)  # [8192, 7168]
k_proj = cat([k_r_tp0, k_r_tp1], dim=0)  # [256, 7168]
v_proj = cat([v_r_tp0, v_r_tp1], dim=0)  # [256, 7168]
o_proj = cat([proj_tp0, proj_tp1], dim=1) # [7168, 8192]
```

**Round-trip 验证**:
```
H2M: qkv_shards[i] = cat([q_tp[i], k_tp[i], v_tp[i]], dim=0)
M2H: split(qkv, [q_per_tp, k_per_tp, v_per_tp], dim=0) → cat(parts, dim=0)
→ split 和 chunk 互为逆操作 ✓
```

### 3.4 Dense MLP 转换 (line 677-695)

```python
cur_gate, cur_up = torch.chunk(cur_linear_fc1, 2, dim=0)  # gate/up deinterleave
gate_weights = cat(gate_list, dim=0)          # TP gather
up_weights = cat(up_list, dim=0)
down_weights = cat(down_list, dim=1)
```
与 DeepSeek3 完全一致。✓

### 3.5 MoE 转换逻辑

#### 3.5.1 local_expert_nums 计算 (line 738-745)

```python
if self.moe_tp_extend_ep and self.tp_size > 1:
    local_expert_nums = self.num_experts // (self.ep_size * self.tp_size)
else:
    local_expert_nums = self.num_experts // self.ep_size
```

**验证** (EP=32, TP=2, moe_tp_extend_ep=True):
```
local_expert_nums = 128 // (32 * 2) = 2   ← 每桶 2 个 expert ✓
```

#### 3.5.2 Grouped GEMM + moe_tp_extend_ep (line 747-792)

**权重读取与 reshape**:
```python
for ep_rank in range(32):
    for tp_rank in range(2):
        w1 = pop(experts.weight1)  # [7168, 49152]  (2 experts)
        w2 = pop(experts.weight2)  # [24576, 7168]  (2 experts)
        reshape(2, 7168, 24576)    # [local_expert_nums, hidden, inter*2]
```

**专家索引映射**:
```python
global_expert_idx = ep_rank * self.tp_size + tp_rank
expert_idx = global_expert_idx * local_expert_nums + idx
```

**验证** (EP=32, TP=2):
```
ep=0, tp=0 → global_expert_idx=0 → expert {0,1}
ep=0, tp=1 → global_expert_idx=1 → expert {2,3}
ep=1, tp=0 → global_expert_idx=2 → expert {4,5}
...
ep=31, tp=1 → global_expert_idx=63 → expert {126,127}
总计: 128 experts ✓
```

**Round-trip Shape 验证**:
```
H2M 写入: gate=[12288,7168], up=[12288,7168] → fc1.t()=[7168,24576]
           packed → chunk(64) → w1=[7168,49152]  (2 experts per bucket)
M2H 读回: w1=[7168,49152] → reshape(2,7168,24576) → 每个 expert .t() → gate=[12288,7168], up=[12288,7168] ✓
```

#### 3.5.3 Grouped GEMM + expert_tp_size > 1 (line 793-864)

**expert_tp_size=1** (默认):
```python
ep_weight1 = ep_weight1_list[0]  # 只取 tp_rank=0 (所有 TP rank 相同)
```
正确: experts 不参与 TP 切分。✓

**expert_tp_size > 1**:
```python
ep_weight1 = torch.cat([ep_weight1_list[i] for i in unique_tp_indices], dim=2)
ep_weight2 = torch.cat([ep_weight2_list[i] for i in unique_tp_indices], dim=1)
```
收集前 `expert_tp_size` 个 TP rank 的分片并合并。✓

**gate/up 反交织**:
```python
chunks = torch.chunk(ep_w1_expert, self.expert_tp_size, dim=0)
gate_list, up_list = [], []
for chunk in chunks:
    g, u = torch.chunk(chunk, 2, dim=0)
    gate_list.append(g.reshape(-1, self.hidden_size))
    up_list.append(u.reshape(-1, self.hidden_size))
local_gate = torch.cat(gate_list, dim=0)
local_up = torch.cat(up_list, dim=0)
```
先按 expert_tp_size 分块再分别拆 gate/up，Round-trip 一致。✓

#### 3.5.4 Non-Grouped GEMM + moe_tp_extend_ep (line 869-897)

```python
global_expert_base = (ep_rank * self.tp_size + tp_rank) * local_expert_nums
```
与 grouped_gemm 路径使用相同的索引映射。✓

**Round-trip 验证**:
```
H2M: global_base = (ep_rank * tp_size + tp_rank) * num_local_experts
M2H: global_expert_base = (ep_rank * tp_size + tp_rank) * local_expert_nums
→ 公式完全一致 ✓
```

#### 3.5.5 Non-Grouped + expert_tp_size > 1 (line 898-970)

```python
fc1_parts = [mg_models[(tp, ep_rank)].pop(local_fc1_key) for tp in range(expert_tp_size)]
cur_fc1 = torch.cat(fc1_parts, dim=0)
chunks = torch.chunk(cur_fc1, self.expert_tp_size, dim=0)
for chunk in chunks:
    g, u = torch.chunk(chunk, 2, dim=0)
    gate_list.append(g); up_list.append(u)
local_gate = torch.cat(gate_list, dim=0)
local_up = torch.cat(up_list, dim=0)
```

**Round-trip 验证** (expert_tp_size=2):
```
H2M: gate_chunks=[g0,g1], up_chunks=[u0,u1]
     fc1 = [g0,u0,g1,u1] → saved as [g0,u0] and [g1,u1] per TP shard
M2H: read [g0,u0] and [g1,u1] → cat → [g0,u0,g1,u1]
     chunk(2) → [g0,u0], [g1,u1]
     chunk(2) → g0,u0 and g1,u1
     gate = cat([g0,g1]) = original gate ✓
     up = cat([u0,u1]) = original up ✓
```

#### 3.5.6 Shared Expert 处理

```python
shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_tp(
    mg_models, shared_fc1_key)
shared_down_weights = self.linear_fc2_gather_from_tp(
    mg_models, shared_fc2_key)
```
Shared expert 始终按 `tp_size` 切分 (不受 `expert_tp_size` 影响)。✓

#### 3.5.7 Router bias 处理

```python
router_bias_weights = mg_models[...].pop(router_bias_key, None)
...
if router_bias_weights is not None:
    hf_dict[...] = router_bias_weights.clone()
```
可选读取，不丢失。Round-trip: mg 有 → hf 有 → mg 有; mg 无 → hf 无 → mg 无。✓

### 3.6 VPP I/O 优化 (line 1062-1081)

```python
raw_data = {}
for tp_rank, ep_rank in product(...):
    raw_data[(tp_rank, ep_rank)] = load_data(pt_path)  # 只读一次

for vpp_rank in range(self.vpp_size):
    mg_weights[(tp_rank, ep_rank)] = raw_data[(tp_rank, ep_rank)].pop(f'model{vpp_rank}')
    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)
    del mg_weights
del raw_data
```
文件读取次数从 `vpp_size × tp_size × ep_size` 降为 `tp_size × ep_size`。✓

---

## 4. MOE_TP_EXTEND_EP 完整 Round-trip 验证

> 配置: TP=2, PP=8, EP=32, moe_tp_extend_ep=True, moe_grouped_gemm=True, dualpipev
>
> **EP 说明**: 训练使用 `--expert-model-parallel-size 64`，但 `moe_tp_extend_ep=True` 时
> EP group 包含 TP 维度，纯 EP = 64/2 = 32。转换脚本使用纯 EP=32。

### 4.1 目录路径

```
global_ep = tp_rank + ep_rank * tp_size  (ep_rank ∈ [0,31])

H2M save: mp_rank_{tp:02}_{pp:03}_{global_ep:03}   (global_ep: 0..63, 共 64 目录)
M2H load: mp_rank_{tp:02}_{pp:03}_{ep_suffix:03}   (ep_suffix = tp + ep*tp, 0..63)
→ 完全一致 ✓
```

### 4.2 Expert 索引

```
H2M: idx = ep_rank * tp_size + tp_rank → bucket[idx] gets 2 experts
M2H: global_expert_idx = ep_rank * tp_size + tp_rank → expert_idx = global*2 + local_idx
→ 索引公式完全一致 ✓
```

### 4.3 Expert 权重 Tensor

```
mg 写入: w1=[7168, 49152] (2 experts: gate.t() cat up.t())
M2H 读:  w1=[7168, 49152] → reshape(2,7168,24576) → 每个 .t()=[24576,7168] → chunk(2) → gate=[12288,7168], up=[12288,7168]
H2M 重建: gate + up → cat → [24576,7168] → .t() → [7168,24576] → 2 experts packed
→ 完全可逆 ✓
```

### 4.4 Non-Grouped 路径

```
H2M: global_base = (ep_rank * tp_size + tp_rank) * num_local_experts
M2H: global_expert_base = (ep_rank * tp_size + tp_rank) * local_expert_nums
→ 公式一致 ✓
```

### 4.5 其他权重

| 权重类型 | Round-trip | 说明 |
|---|---|---|
| Router weight | ✓ | 从 (tp=0,ep=0) 读取，复制到所有 (ep,tp) |
| Router bias | ✓ | 可选保留，None default |
| Shared expert fc1/fc2 | ✓ | 按 tp_size gather/split |
| LayerNorm | ✓ | 不切分，直接复制 |
| Embedding | ✓ | cat(tp, dim=0) / chunk(tp, dim=0) |
| Final norm + lm_head | ✓ | cat(tp, dim=0) / chunk(tp, dim=0) |
| QK LayerNorm | ✓ | 可选读取 |
| DualPipe 层映射 | ✓ | 前后半分配一致 |

**结论: MOE_TP_EXTEND_EP 模式下 mg→hf→mg round-trip 完全正确，无参数丢失。**

---

## 5. 与 DeepSeek3 的详细差异分析

### 5.1 设计性差异 (非 Bug)

| 差异 | Kimi2 | DeepSeek3 | 正确性 |
|------|-------|-----------|--------|
| 注意力 | GQA: q_proj/k_proj/v_proj/o_proj | MLA: q_a_proj/kv_a_proj + q_b_proj/kv_b_proj | ✓ 各自正确 |
| QKV TP 切分 | 按 head 数切分后合并 | MLA 压缩部分不切 | ✓ |
| expert_tp_size | 独立参数 (默认 1) | 等于 tp_size | ✓ Kimi2 更灵活 |
| EP 路径命名 | `global_ep = tp + ep*tp` (独立函数) | 直接用 ep_rank (内联) | ✓ 正确处理交错命名 |
| Router bias | 可选 | 必选 | ✓ 更鲁棒 |
| 全局变量 | 无 (实例属性) | 有 (`TENSOR_SIZE`, `hf_weight_dict`) | ✓ 更安全 |
| `torch_npu` | 不导入 | 导入 | ✓ 跨平台 |
| gc.collect() | 使用 | 未使用 | ✓ 内存管理 |
| VPP I/O | 文件只读一次 | 每次重新读取 | ✓ 性能优化 |
| 安全加载 | weights_only=True 优先 | weights_only=False | ✓ 更安全 |
| 路径不存在 | FileNotFoundError + 配置提示 | 无 | ✓ 更友好 |

### 5.2 关键逻辑一致性

| 逻辑路径 | Kimi2 vs DeepSeek3 | 一致性 |
|---------|-------------------|--------|
| Dense MLP gate/up 拆分 | 相同 | ✓ |
| Expert fc1 转置 | 相同 (.t()) | ✓ |
| Expert fc2 转置 | 相同 (.t()) | ✓ |
| Grouped GEMM 3D view | 相同 reshape 链 | ✓ |
| EP 切分 (dim=0) | 相同 | ✓ |
| DualPipe 层映射 | 相同 (前后半分配) | ✓ |
| DualPipe postprocess (pp=0, vpp=last) | 相同 | ✓ |
| VPP 保存格式 (model0, model1) | 相同 | ✓ |

### 5.3 DeepSeek3 M2H 路径命名的特殊说明

DeepSeek3 的 `get_pt_path_by_tpppep_rank` (line 425-437) **没有** `moe_tp_extend_ep` 特殊处理 — 始终使用 raw `ep_rank`:

```python
# DeepSeek3: 直接用 ep_rank，无 global_ep 计算
mp_rank_path = f'mp_rank_{tp_rank:02d}'
if self.pp_size > 1:
    mp_rank_path += f'_{pp_rank:03d}'
if self.ep_size > 1:
    mp_rank_path += f'_{ep_rank:03d}'
```

这在 DeepSeek3 中可行是因为:
1. 每个 `(tp_rank, ep_rank)` 对已通过 `tp_rank` 部分产生唯一目录名
2. DeepSeek3 强制 `dualpipe + tp>1` 时必须 `moe_tp_extend_ep`

**Kimi2 的方案更优**: 使用 `global_ep = tp_rank + ep_rank * tp_size` 作为后缀，与 MCore 框架的交错命名约定完全匹配，确保 H2M 和 M2H 路径生成一致。

---

## 6. 发现的问题

### 6.1 已修复的问题

| 问题 | 类型 | 说明 |
|------|------|------|
| `_mp_prefix` 路径生成 | Bug 修复 | `moe_tp_extend_ep` 优先级高于 `ep_size==1`，修复后路径与 H2M 一致 |
| EP 遍历范围 | Bug 修复 | `ep_rank_list` 始终使用 `range(ep_size)` |
| `local_expert_nums` 计算 | Bug 修复 | moe_tp_extend_ep 时为 `num_experts / (ep_size * tp_size)` |
| gate/up 反交织 | Bug 修复 | expert_tp_size>1 时先分块再拆 gate/up |
| 全局变量 | 重构 | 消除 `TENSOR_SIZE`, `hf_weight_dict` 全局状态 |

### 6.2 当前代码正确性总结

| 转换路径 | 状态 | 说明 |
|---------|------|------|
| Embedding | ✅ | TP gather → cat(dim=0) |
| Final Norm + LM Head | ✅ | TP gather → cat(dim=0) |
| Layer Norm | ✅ | 不切分，直接复制 |
| Attention (GQA) | ✅ | QKV split → cat, proj cat(dim=1) |
| Dense MLP | ✅ | gate/up deinterleave + cat |
| MoE Router | ✅ | 可选 bias |
| MoE Shared Experts | ✅ | 同 Dense MLP |
| MoE Grouped GEMM + tp_extend_ep | ✅ | global_ep 交错命名 + 正确索引 |
| MoE Grouped GEMM + expert_tp=1 | ✅ | 只取 tp=0 |
| MoE Grouped GEMM + expert_tp>1 | ✅ | gather unique TP + deinterleave |
| MoE Non-Grouped + tp_extend_ep | ✅ | 正确索引映射 |
| MoE Non-Grouped + expert_tp>1 | ✅ | gather + deinterleave |
| QK LayerNorm | ✅ | 可选读取 |
| DualPipe 层映射 | ✅ | 前后半分配 |
| Checkpoint 路径构建 | ✅ | 交错命名一致 |
| VPP I/O | ✅ | 文件只读一次 + .pop() 提取 |
| MOE_TP_EXTEND_EP round-trip | ✅ | 完整验证通过 |

---

## 7. 结论

### 7.1 总体评价

`convert_kimi2_mcore2hf.py` 实现质量高，完全正确:

1. **架构适配正确**: GQA 注意力权重拆分/合并逻辑完全正确
2. **MoE 处理完整**: 支持所有 expert_tp_size × moe_tp_extend_ep × grouped_gemm 组合
3. **Round-trip 一致**: 与 `convert_kimi2_hf2mcore.py` 构成正确的双向转换
4. **MOE_TP_EXTEND_EP 完全正确**: 目录命名、专家索引、权重 shape 全部验证通过
5. **并行支持完整**: TP/PP/EP/VPP/DualPipe 五种并行维度
6. **代码质量优于参考**: 安全加载、文件检查、VPP I/O 优化、全局变量消除

### 7.2 推荐使用

```bash
# 默认配置 (与训练一致)
bash scripts/ckpt_convert_kimi2_mcore2hf.sh

# 自定义路径
LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/hf/output bash scripts/ckpt_convert_kimi2_mcore2hf.sh
```
