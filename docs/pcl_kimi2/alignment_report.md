# 训练框架 (MindSpeed-LLM) 与 HuggingFace 模型对齐报告

> 训练框架 dump 文件: `dump/model_iteration_0_module_sources.py`
> HF 模型文件: `models/modeling_deepseek.py`
> 训练脚本: `scripts/pretrain_kimi2_1t_4k.sh`
> 测试文件: `tests/test_modeling_comprehensive.py` (**122 tests passed**)

## 模块映射总览

训练框架 dump 包含 **19 个模块**，全部与 HF 模型对齐，无遗漏。

| # | 训练框架模块路径 | 训练框架类 | HF 对应 | 状态 |
|---|---|---|---|---|
| 1 | `module` | `GPTModel` | `DeepseekV3ForCausalLM` | 已对齐 |
| 2 | `module.embedding` | `LanguageModelEmbedding` | `DeepseekV3Model.embed_tokens` | 已对齐 |
| 3 | `module.embedding.word_embeddings` | `VocabParallelEmbedding` | `nn.Embedding` | 已对齐 |
| 4 | `module.rotary_pos_emb` | `RotaryEmbedding` | `DeepseekV3YarnRotaryEmbedding` | 已对齐 |
| 5 | `module.decoder` | `TransformerBlock` | `DeepseekV3Model` | 已对齐 |
| 6 | `module.decoder.layers.0` | `TransformerLayer` | `DeepseekV3DecoderLayer` | 已对齐 |
| 7 | `layers.0.input_layernorm` | `RMSNorm` | `DeepseekV3RMSNorm` | 已对齐 |
| 8 | `layers.0.self_attention` | `SelfAttention` | `DeepseekV3Attention` | 已对齐 |
| 9 | `layers.0.self_attention.core_attention` | `CustomDotProductAttention` | Attention forward 内联 | 已对齐 |
| 10 | `layers.0.core_attention.scale_mask_softmax` | `FusedScaleMaskSoftmax` | Attention forward 内联 | 已对齐 |
| 11 | `layers.0.self_attention.linear_proj` | `RowParallelLinear` | `o_proj` (`nn.Linear`) | 已对齐 |
| 12 | `layers.0.self_attention.linear_qkv` | `ColumnParallelLinear` | `q_proj`+`k_proj`+`v_proj` | 已对齐 |
| 13 | `layers.0.pre_cross_attn_layernorm` | `IdentityOp` | 无 (decoder-only 不需要) | 已对齐 |
| 14 | `layers.0.cross_attn_bda` | `IdentityFuncOp` | 无 (decoder-only 不需要) | 已对齐 |
| 15 | `layers.0.mlp` (dense) | `MLP` | `DeepseekV3MLP` | 已对齐 |
| 16 | `layers.1.mlp` (MoE) | `MoELayer` | `DeepseekV3MoE` | 已对齐 |
| 17 | `layers.1.mlp.router` | `TopKRouter` | `MoEGate` | 已对齐 |
| 18 | `layers.1.mlp.experts` | `MindSpeedGmmExperts` | `DeepseekV3MLP` x N | 已对齐 |
| 19 | `layers.1.mlp.shared_experts` | `SharedExpertMLP` | `DeepseekV3MLP` | 已对齐 ⚠️ |

> ⚠️ #19: 训练框架的 `SharedExpertMLP` 有可选的 `sigmoid gate` 机制（`gate_weight` 参数），HF 模型未实现。当前训练脚本中 **未启用** 该 gate（无 `--moe-shared-expert-gate` 参数），因此实际计算等价。

---

## 逐模块详细对齐分析

### 1. GPTModel ↔ DeepseekV3ForCausalLM

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| Embedding | `self.embedding(input_ids, position_ids)` | `self.embed_tokens(input_ids)` | ✓ |
| RoPE | `self.rotary_pos_emb(rotary_seq_len)` | `self.rotary_emb(x, seq_len)` | ✓ |
| Decoder | `self.decoder(hidden_states, ...)` | `self.model(...)` | ✓ |
| Output layer | `self.output_layer(hidden_states)` | `self.lm_head(hidden_states)` | ✓ |
| Loss | `CrossEntropyLoss` with shift | `CrossEntropyLoss` with shift | ✓ |
| Aux loss | `loss += aux_loss` | `loss += outputs.aux_loss` | ✓ |

### 2. LanguageModelEmbedding ↔ nn.Embedding

| 功能 | 训练框架 | HF | 说明 |
|---|---|---|---|
| Word embedding | `VocabParallelEmbedding` | `nn.Embedding` | 数学等价，训练框架做 vocab 并行切分 |
| Position embedding | 不使用 (RoPE 模型) | 不使用 | ✓ |
| Embedding dropout | `embedding_dropout` | 无 | 训练正则化，非架构差异 |
| fp32 residual | 可选 `fp32_residual_connection` | 无 | 训练技巧，非架构差异 |

### 3. RotaryEmbedding ↔ DeepseekV3YarnRotaryEmbedding

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| inv_freq 计算 | `1/(base^(arange(0,dim,2)/dim))` | 相同 | ✓ 数值差 0.0 |
| freqs 构建 | `outer(seq, inv_freq)` | 相同 | ✓ |
| emb 拼接 | `cat((freqs, freqs))` non-interleaved | 相同 | ✓ |
| YaRN scaling | 通过 config.rope_scaling 配置 | `DeepseekV3YarnRotaryEmbedding` | ✓ |
| mscale | `0.1 * mscale * log(scale) + 1` | 相同 | ✓ |
| cos/sin | `emb.cos() * _mscale`, `emb.sin() * _mscale` | 相同 | ✓ |

### 4. TransformerBlock ↔ DeepseekV3Model

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| Layer 循环 | `for layer in layers: hidden_states, ctx = layer(...)` | `for layer in layers: ...` | ✓ |
| Final LayerNorm | `final_layernorm(hidden_states)` (RMSNorm) | `self.norm(hidden_states)` (RMSNorm) | ✓ |
| Checkpoint | 可选 activation recompute | gradient_checkpointing | ✓ |

### 5. TransformerLayer ↔ DeepseekV3DecoderLayer

训练框架 TransformerLayer 的前向流程 (decoder-only):

```
residual = hidden_states
input_layernorm_output = input_layernorm(hidden_states)      # RMSNorm
attn_output = self_attention(input_layernorm_output, ...)     # GQA Attention
hidden_states = residual + attn_output                        # residual add

residual = hidden_states
pre_mlp_layernorm_output = pre_mlp_layernorm(hidden_states)   # RMSNorm
mlp_output = mlp(pre_mlp_layernorm_output)                    # Dense MLP 或 MoE
hidden_states = residual + mlp_output                         # residual add
```

HF DeepseekV3DecoderLayer 前向流程:

```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)                # RMSNorm
hidden_states = self_attn(hidden_states, ...)                 # GQA Attention
hidden_states = residual + hidden_states                      # residual add

residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)       # RMSNorm
hidden_states = mlp(hidden_states)                            # Dense MLP 或 MoE
hidden_states = residual + hidden_states                      # residual add
```

**完全对齐。** 训练框架中 `pre_cross_attn_layernorm` 对应 `IdentityOp`（decoder-only 模型跳过 cross attention），`pre_mlp_layernorm` 对应 HF 的 `post_attention_layernorm`。

### 6. RMSNorm ↔ DeepseekV3RMSNorm

**训练框架:**
```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    output = self._norm(x.float()).type_as(x)  # fp32 归一化 → 转回输入 dtype
    return output * self.weight                  # weight 保持参数 dtype (训练中为 bf16)
```

**HF:**
```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)             # 转 fp32
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (self.weight.to(input_dtype) * hidden_states).to(input_dtype)  # 显式 dtype 管理
```

**差异说明:** 数学公式完全相同: `weight * x * rsqrt(mean(x²) + eps)`。dtype 管理方式不同:
- Megatron: `type_as(x)` 后直接 `* weight`，依赖 Float16Module 保证 weight 和 input 同 dtype
- HF: 显式 `weight.to(input_dtype)` 确保输出类型一致（HF 参数默认 fp32，需额外转换）

数值验证 (同 weight, 同输入): **fp32 差 0.0, bf16 差 0.0** ✓

### 7. SelfAttention ↔ DeepseekV3Attention

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| QKV 投影 | 融合 `linear_qkv` 后 split | 分开 `q_proj`/`k_proj`/`v_proj` | ✓ 数学等价 |
| QK-norm | `RMSNorm(head_dim)` | `DeepseekV3RMSNorm(head_dim)` | ✓ |
| QK-norm 时机 | reshape 后、transpose 前 | reshape 后、transpose 前 | ✓ |
| GQA | `num_query_groups` 分组 KV | `repeat_kv` 扩展 KV 至 Q heads | ✓ |
| Scaling | `1/sqrt(head_dim)` | 相同 | ✓ |
| YaRN scaling | `scaling * mscale^2` | 相同 | ✓ |
| Softmax | fp32 softmax | fp32 softmax | ✓ |
| Attention dropout | 可选 | 可选 | ✓ |
| Output projection | `linear_proj` (RowParallelLinear) | `o_proj` (nn.Linear) | ✓ |

### 8. CustomDotProductAttention ↔ Attention forward 内联

训练框架使用 FlashAttention 优化内核，HF 使用标准 `torch.matmul`。数学公式完全一致:

```
attn_weights = Q @ K^T * scaling
attn_weights = attn_weights + attention_mask
attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(query_dtype)
attn_output = attn_weights @ V
```

### 9. FusedScaleMaskSoftmax ↔ Attention forward 内联

训练框架: `scale * input → mask → softmax(fp32) → cast`
HF: `QK^T * scaling → +mask → softmax(fp32) → cast`

**数学等价。** 训练框架可选 CUDA fused kernel，无 kernel 时 fallback 到相同的 torch softmax。

### 10. MLP ↔ DeepseekV3MLP

**训练框架 (非融合路径):**
```python
intermediate = linear_fc1(hidden_states)  # [gate, up] 拼接
intermediate = silu(intermediate[:,:h]) * intermediate[:,h:]  # SwiGLU
output = linear_fc2(intermediate)
```

**HF:**
```python
down_proj = down_proj(silu(gate_proj(x)) * up_proj(x))  # SwiGLU
```

数值验证: **差 0.0**

### 11. MoELayer ↔ DeepseekV3MoE

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| Router | `TopKRouter` | `MoEGate` | ✓ |
| Token dispatch | `token_dispatcher` (分布式) | `moe_forward`/`moe_infer` (单机) | ✓ |
| Expert 执行 | `experts(dispatched_input)` | 逐 expert 循环 | ✓ |
| Shared expert | `output + shared_experts(hidden_states)` | `y + shared_experts(identity)` | ✓ |

### 12. TopKRouter ↔ MoEGate

| 功能 | 训练框架 | HF | 对齐 |
|---|---|---|---|
| Gating | `linear(input, weight, bias)` fp32 | `F.linear(input, weight, bias)` fp32 | ✓ |
| Score function | `sigmoid` | `sigmoid` | ✓ |
| noaux_tc 路由 | group top-2 sum → topk_group → masked top-k | 相同 | ✓ |
| e_score_correction_bias | `scores + bias` 用于选择 | 相同 | ✓ |
| Weight normalization | `norm_topk_prob`: 除以 sum | 相同 | ✓ |
| Routed scaling | `* routed_scaling_factor` | 相同 | ✓ |
| Aux loss (seq) | `pi * fi` per-sequence mean | 相同 | ✓ |
| Z-loss | `logsumexp(logits)^2 * coeff` | 相同 | ✓ |

数值验证: **routed weight sum = 2.827 (= routed_scaling_factor)** ✓

### 13. SharedExpertMLP ↔ DeepseekV3MLP

**训练框架:**
```python
class SharedExpertMLP(MLP):
    def __init__(self, config, submodules, gate: bool):
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config=config, submodules=submodules)
        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.gate_weight = nn.Parameter(torch.empty((1, config.hidden_size)))

    def forward(self, hidden_states):
        output, _ = super().forward(hidden_states)       # SwiGLU MLP
        if self.use_shared_expert_gate:
            gate_score = F.sigmoid(F.linear(hidden_states, self.gate_weight))
            output = output * gate_score                 # 可选 sigmoid gate
        return output
```

**HF:**
```python
# DeepseekV3MoE.__init__:
if config.n_shared_experts is not None:
    intermediate_size = config.moe_intermediate_size * config.n_shared_experts  # 12288 * 1 = 12288
    self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)
    # 无 sigmoid gate
```

**对齐状态:** SwiGLU MLP 计算完全一致（中间维度 12288）。HF 未实现可选的 `sigmoid gate`。

| 方面 | 训练框架 | HF | 匹配 |
|------|----------|-----|------|
| SwiGLU | `silu(fc1_gate) * fc1_up` → `fc2` | `silu(gate_proj) * up_proj` → `down_proj` | ✓ 数学等价 |
| 中间维度 | `moe_shared_expert_intermediate_size` | `moe_intermediate_size * n_shared_experts` = 12288 | ✓ |
| Sigmoid gate | `output * sigmoid(x @ gate_weight)` | 无 | ⚠️ 当前未启用，不影响 |
| Overlap | `--moe-shared-expert-overlap` 异步流水线 | 无 | 训练优化，非架构 |

---

## 数值验证汇总

测试命令: `pytest tests/test_modeling_comprehensive.py -v`

| 验证项 | 结果 | 说明 |
|---|---|---|
| RMSNorm fp32 | max_diff = **0.0** | 同权重同输入，fp32 下 bit-for-bit 一致 |
| RMSNorm bf16 | max_diff = **0.0** | 同上，bf16 下 bit-for-bit 一致 |
| RoPE inv_freq | max_diff = **0.0** | `1/(base^(arange/dim))` 完全一致 |
| RoPE emb (cat freqs) | max_diff = **0.0** | `cat((freqs, freqs))` 布局一致 |
| rotate_half | **OK** | `cat(-x2, x1)` 一致 |
| MLP SwiGLU | max_diff = **0.0** | 拆分 gate/up vs 融合 fc1 等价 |
| MoEGate weight sum | **2.827** | = routed_scaling_factor ✓ |
| Full model forward | logits shape 正确, **无 NaN** | tiny config 端到端验证 |
| 测试 | **122 passed, 0 failed** | 覆盖 norm/rope/attn/moe/model/causalLM/classification |

---

## 关键训练参数与 HF Config 映射

来自 `scripts/pretrain_kimi2_1t_4k.sh`:

| 训练参数 | 值 | HF Config 字段 | 值 |
|---|---|---|---|
| `--normalization` | `RMSNorm` | (内置使用 `DeepseekV3RMSNorm`) | ✓ |
| `--qk-layernorm` | (flag) | `qk_layernorm=True` | ✓ |
| `--num-attention-heads` | 64 | `num_attention_heads=64` | ✓ |
| `--num-query-groups` | 2 | `num_query_groups=2` | ✓ |
| `--kv-channels` | 128 | `kv_channels=128` | ✓ |
| `--rope-scaling-type` | `yarn` | `rope_scaling.type=yarn` | ✓ |
| `--rope-scaling-factor` | 32 | `rope_scaling.factor=32.0` | ✓ |
| `--rope-scaling-mscale-all-dim` | 1.0 | `rope_scaling.mscale_all_dim=1.0` | ✓ |
| `--rotary-base` | 50000 | `rope_theta=50000.0` | ✓ |
| `--num-experts` | 128 | `n_routed_experts=128` | ✓ |
| `--n-shared-experts` | 1 | `n_shared_experts=1` | ✓ |
| `--moe-router-topk` | 2 | `num_experts_per_tok=2` | ✓ |
| `--moe-router-topk-scaling-factor` | 2.827 | `routed_scaling_factor=2.827` | ✓ |
| `--moe-router-num-groups` | 8 | `n_group=8` | ✓ |
| `--moe-router-group-topk` | 2 | `topk_group=2` | ✓ |
| `--moe-router-score-function` | `sigmoid` | `scoring_func=sigmoid` | ✓ |
| `--moe-router-enable-expert-bias` | (flag) | `moe_router_enable_expert_bias=True` | ✓ |
| `--moe-router-dtype` | `fp32` | `moe_router_dtype=fp32` | ✓ |
| `--moe-ffn-hidden-size` | 12288 | `moe_intermediate_size=12288` | ✓ |
| `--norm-epsilon` | 1e-6 | `rms_norm_eps=1e-6` | ✓ |
| `--disable-bias-linear` | (flag) | `attention_bias=False` | ✓ |
| `--attention-softmax-in-fp32` | (flag) | (内置 `softmax(dtype=float32)`) | ✓ |
| `--swiglu` | (flag) | `hidden_act=silu` | ✓ |

---

## 差异说明（非架构，不影响对齐）

| 差异 | 说明 |
|---|---|
| 分布式并行 | 训练框架使用 `RowParallelLinear`/`ColumnParallelLinear`/`VocabParallelEmbedding`，HF 使用标准 `nn.Linear`/`nn.Embedding`，数学等价 |
| 融合内核 | 训练框架使用 `FusedScaleMaskSoftmax`/`GroupedGEMM`/`FlashAttention`，HF 使用标准 PyTorch 实现，数学等价 |
| Embedding dropout | 训练框架在 embedding 后加了 dropout，HF 没有（训练正则化手段） |
| Identity 模块 | 训练框架中 `pre_cross_attn_layernorm`/`cross_attn_bda` 是 `IdentityOp`，HF 直接跳过 |
| Float16Module | 训练框架外层包装 fp16/bf16 转换，HF 通过 `.to(dtype)` 处理 |
