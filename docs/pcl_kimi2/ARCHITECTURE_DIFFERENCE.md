# Kimi2-PCL 与 Kimi-K2-Base 架构差异分析报告

Kimi2-PCL 项目在原始开源的 Kimi-K2-Base (`moonshotai/Kimi-K2-Base`) 架构上进行了大量深度改造。本次修改的核心目的是将原先主打推理、采用特殊注意力机制（MLA）的模型架构，调整为**更适合大规模分布式预训练的 GQA 架构**，同时增强了 MoE 的训练特性支持。

以下是两大模型架构的核心差异总结：

## 1. 核心注意力机制差异：MLA vs GQA

这是两个模型之间最根本的结构性差异。

### Kimi-K2-Base (MLA 架构)
原始的 Kimi-K2-Base 采用了类似于 DeepSeek-V2/V3 的 **MLA (Multi-Head Latent Attention)** 架构：
- 具有低秩投影维度配置：`q_lora_rank = 1536`，`kv_lora_rank = 512`。
- 在 `modeling_deepseek.py` 的 `DeepseekV3Attention` 实现中，Q 和 KV 投影被拆分为降维和升维两个阶段：
  - Q 被拆解为 `q_a_proj` (降维) -> `q_a_layernorm` -> `q_b_proj` (升维)。
  - KV 被压缩并合并为 `kv_a_proj_with_mqa` -> `kv_a_layernorm` -> `kv_b_proj`。

### Kimi2-PCL (GQA 架构)
Kimi2-PCL **完全移除了 MLA 架构**，回归到标准的 **GQA (Grouped Query Attention)** 架构：
- 移除了所有 `lora_rank` 相关的配置及代码（去除了 `q_a_proj` 等降维权重）。
- 重新采用了直接的 `q_proj`、`k_proj` 和 `v_proj` 线性层。
- 强制启用了 **QK LayerNorm**（`qk_layernorm=True`），即在 Q 和 K 与 RoPE 结合前，分别通过 `q_layernorm` 和 `k_layernorm` 进行归一化，以提升预训练稳定性。

## 2. MoE 路由与训练支持增强

Kimi-K2-Base 原本主要面向推理（Forward 甚至包含 `assert not self.training`），而 Kimi2-PCL 对 MoE 模块进行了完整的训练期增强。

### 路由分组与激活策略
- **Kimi-K2-Base**: 共有 384 个专家（`n_routed_experts=384`），每次激活 8 个专家（`num_experts_per_tok=8`）。未开启分组路由（`n_group=1`, `topk_group=1`）。
- **Kimi2-PCL**: 共有 128 个专家（`n_routed_experts=128`），每次激活 2 个专家（`num_experts_per_tok=2`）。开启了复杂的分组受限路由机制（`n_group=8`, `topk_group=2`），即在 8 个组中先选 2 个组，再在组内选出 2 个专家。

### 路由训练损失 (Aux Loss)
- **Kimi-K2-Base**: `MoEGate` 的前向传播仅返回选中的专家索引和权重 (`topk_idx`, `topk_weight`)。
- **Kimi2-PCL**: 为支持端到端预训练，`MoEGate` 增加了负载均衡损失和 Z-Loss 计算逻辑，前向传播会额外返回 `aux_loss`（由 `moe_aux_loss_coeff` 和 `moe_z_loss_coeff` 控制）。

### Router 偏置与精度
- **Kimi2-PCL** 新增了 `moe_router_enable_expert_bias` 支持（门控线性层附加偏置项），并允许配置 `moe_router_dtype` 以在低精度训练时保护 Router 精度。

## 3. 模型基础规模与维度调整

为了匹配 Kimi2-1T 的特定设计，宏观规模参数也发生了明显改变。以下是详细的参数对比表：

| 参数模块       | 参数名称                  | Kimi-K2-Base             | Kimi2-PCL (1T配置)     |
| :------------- | :------------------------ | :----------------------- | :--------------------- |
| **基础规模**   | `num_hidden_layers`       | 61                       | 32                     |
|                | `hidden_size`             | 7168                     | 7168                   |
|                | `vocab_size`              | 163840                   | 163840                 |
|                | `max_position_embeddings` | 131072                   | 131072                 |
| **注意力机制** | `num_attention_heads`     | 64                       | 64                     |
|                | `num_key_value_heads`     | 64                       | 32 (GQA: group_size=2) |
|                | `q_lora_rank`             | 1536                     | *无此参数* (移除MLA)   |
|                | `kv_lora_rank`            | 512                      | *无此参数* (移除MLA)   |
|                | `qk_nope_head_dim`        | 128                      | 128                    |
|                | `qk_rope_head_dim`        | 64                       | 64                     |
|                | `v_head_dim`              | 128                      | 128                    |
|                | `qk_layernorm`            | *未配置*                 | True                   |
| **MoE 架构**   | `n_routed_experts`        | 384                      | 128                    |
|                | `n_shared_experts`        | 1                        | 1                      |
|                | `num_experts_per_tok`     | 8                        | 2                      |
|                | `first_k_dense_replace`   | 1                        | 2                      |
|                | `moe_intermediate_size`   | 2048                     | 12288                  |
|                | `intermediate_size`       | 18432                    | 18432                  |
|                | `n_group`                 | 1                        | 8                      |
|                | `topk_group`              | 1                        | 2                      |
|                | `moe_router_topk`         | *未配置*                 | 2                      |
|                | `moe_aux_loss_coeff`      | 0.001 (`aux_loss_alpha`) | 0.01                   |
|                | `moe_z_loss_coeff`        | *未配置*                 | 0.001                  |
| **位置编码**   | `rope_theta`              | 50000.0                  | 50000.0                |
|                | `rope_scaling_type`       | yarn                     | yarn                   |

*说明：Kimi2-PCL 虽然层数减少，但大幅提高了 MoE 单个专家的维度（从 2048 提升到 12288），从而在较少的路由专家数（128）下支撑起 1T 的参数规模。*

## 4. 并行框架 (Megatron-Core) 的适配支持

除了纯算法层的修改，Kimi2-PCL 中的 `modeling_deepseek.py` 还融入了对 Megatron-Core 和 MindSpeed-LLM 分布式训练的适配：
- **专家并行 (Expert Parallel)**: 完善了 `ep_size` 的分布式通信逻辑，支持跨卡的 `all_to_all` Token 派发。
- **并行训练接口**: Kimi2-PCL 的模型对象在构建时支持接收更多关于训练时的并行拓扑环境参数（如流水线并行中的阶段切分等）。

## 总结

**Kimi2-PCL 并非简单的 Kimi-K2-Base 放大版**。它为了满足大规模 1T 参数集群预训练的稳定性与通信效率，大刀阔斧地**砍掉了 MLA 架构并退回 GQA**，同时**引入了 QK LayerNorm**。在 MoE 层，它减少了专家总数但做大了单专家参数量，并重新实现了负载均衡 Loss 与分组路由策略，使其成为一个完全“Pretrain-Ready”的重构版本。