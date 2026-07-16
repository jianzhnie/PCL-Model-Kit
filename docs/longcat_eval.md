# LongCat-Flash-Chat MoE 扩展与评估结果

## 一、模型概览

| 参数                             | 值                         |
| ------------------------------ | ------------------------- |
| `architectures`                | `LongcatFlashForCausalLM` |
| `hidden_size`                  | 6144                      |
| `expert_ffn_hidden_size`       | 2048                      |
| `num_layers`                   | 28                        |
| `n_routed_experts`             | 512                       |
| `zero_expert_num`              | 256 (identity 类型，无存储权重)   |
| `moe_topk`                     | 12                        |
| `num_attention_heads`          | 64                        |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536                |

## 二、模型扩展

### 2.1 方案 1：专家数扩展（Expert Upcycling）

将 512 个 routed expert 翻倍至 1024，推理激活参数不变（仍 top-12），总参数约 2× (1120B)。

### 2.2 方案 2：深度 + 专家 联合扩展（Combined）

深度 + 专家扩展，默认 28→32 层（+4 层）+ 512→1024 专家,  总参数约 2.3× (1280B) 。

## 三、评估结果

### MMLU

| Model                               | n-shot | Score |
| ----------------------------------- | ------ | ----- |
| LongCat-Flash-Chat(origin)          | 5      | 86.44 |
| LongCat-Flash-Chat-Expertx2         | 5      | 85.88 |
| LongCat-Flash-Chat-Expertx2-Depth32 | 5      | 85.79 |

### C-eval

| Groups                              | n-shot | Value |
| ----------------------------------- | ------ | ----- |
| LongCat-Flash-Chat(origin)          | 5      | __    |
| LongCat-Flash-Chat-Expertx2         | 5      | 86.33 |
| LongCat-Flash-Chat-Expertx2-Depth32 | 5      | 86.63 |
