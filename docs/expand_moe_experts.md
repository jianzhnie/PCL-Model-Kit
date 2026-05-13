# 方案 2：MoE 专家数扩增 — 实现文档

## 1. 目标

将模型的 MoE 专家数从 N 扩增到 kN（默认 k=2，即翻倍），同时尽可能保持模型性能。通过直接复制原有专家的权重来初始化新专家，并相应扩展路由器权重。

## 2. 核心思路

对于一个拥有 N 个专家的 MoE 层：

```
专家 0, 专家 1, ..., 专家 N-1          → 保持原样
专家 N, 专家 N+1, ..., 专家 2N-1      → 新增，从原专家复制
路由器 classifier.weight              → 在 dim=0 上扩展
路由器 e_score_correction_bias        → 在 dim=0 上扩展
```

**关键原则：** 不改变参数的维度（hidden_size、expert_ffn_hidden_size 不变），只增加专家数量。

## 3. 参数命名规范

### 3.1 专家权重

```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight    (expert_ffn_hidden_size, hidden_size)
model.layers.{L}.mlp.experts.{E}.up_proj.weight       (expert_ffn_hidden_size, hidden_size)
model.layers.{L}.mlp.experts.{E}.down_proj.weight     (hidden_size, expert_ffn_hidden_size)
```

正则匹配：`model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)`

注意区分 `mlp.experts`（MoE 专家）和 `mlps`（密集 FFN 层）。LongCat 模型中同时存在 `model.layers.{N}.mlps.{0,1}.*`（密集 MLP），它不会匹配到专家的正则表达式。

### 3.2 路由器权重（兼容多种命名）

| 命名风格 | classifier weight | score correction bias |
|---------|-------------------|----------------------|
| LongCat (router) | `mlp.router.classifier.weight` | `mlp.router.e_score_correction_bias` |
| 通用 (gate) | `mlp.gate.weight` | `mlp.gate.e_score_correction_bias` |

```python
ROUTER_WEIGHT_SUFFIXES = (
    "mlp.router.classifier.weight",
    "mlp.gate.weight",
)
ROUTER_BIAS_SUFFIXES = (
    "mlp.router.e_score_correction_bias",
    "mlp.gate.e_score_correction_bias",
)
```

判断方式是通过 `str.endswith()` 匹配后缀，确保不会误匹配专家内部的 `gate_proj.weight`。

### 3.3 配置键（兼容多种命名）

```python
EXPERT_COUNT_KEYS = ["n_routed_experts", "n_experts", "num_experts"]
TOPK_KEYS         = ["moe_topk", "num_experts_per_tok", "top_k"]
```

按顺序查找，使用第一个匹配的键。`n_routed_experts` 是 LongCat 模型使用的专家数键，`moe_topk` 是激活数键。

### 3.4 `--target_topk` 参数

通过 `--target_topk` 可调整每 token 激活的专家数。该参数**不影响任何模型权重**——它只修改 config 中的顶层值，控制运行时 `torch.topk` 的 `k` 参数：

```
moe_topk = 12  →  每 token 激活 12 个专家
moe_topk = 24  →  每 token 激活 24 个专家
```

扩增专家数后，每个专家被选中的概率自然减半。增大 topk 可以保持每个专家的负载均衡，但会增加每 token 的计算量。

## 4. 实现流程

```
Pass 1: 扫描 Headers 确定输出 Shard 布局
    │
    ├── 读取每个 safetensors 文件的 JSON header（只解析 header，不加载 tensor 数据，内存占用极低）
    ├── 模拟 shard 填充：按相同算法计算每个 tensor 的字节数，决定分片边界
    └── 输出：精确的 shard 数量、总输出大小
    
Pass 2: 加载 Tensor 并写入
    │
    ├── 遍历 shard 文件，用 safe_open 逐个加载 tensor（CPU 内存映射，低内存占用）
    ├── 非路由器、非专家参数 → 直接复制
    ├── 路由器参数 → 在 dim=0 上扩展（详见第 6 节）
    ├── 专家参数 → 保留原专家 + 添加新专家的副本
    └── 按 target_shard_size 分批写入输出 shard
```

### Pass 1 的必要性

safetensors 文件名格式为 `model_00005-of-00075.safetensors`（最后一段是总 shard 数）。需要在写入之前确定总数。如果先写临时名字（如 `model_00001-of-XXXXX.safetensors`）再改名：
- 中途崩溃会留下 `-of-XXXXX` 的文件，模型无法加载
- 改名操作对 500B 模型的大文件来说开销巨大

**解决方案：** 两次遍历。第一次只读 headers（解析 JSON 元信息，不读 tensor 数据）精确计算 shard 数，第二次写正确的文件名。

## 5. 专家复制策略

### 5.1 `build_expert_target_map` 算法

```python
def build_expert_target_map(original_experts, target_experts):
    targets = defaultdict(list)
    for new_idx in range(original_experts, target_experts):
        src_idx = new_idx % original_experts
        targets[src_idx].append(new_idx)
    return dict(targets)
```

**示例：4 → 8 扩展**
```
src expert 0 → new experts [4]      (8 % 4 = 0)
src expert 1 → new experts [5]      (9 % 4 = 1)
src expert 2 → new experts [6]      (10 % 4 = 2)
src expert 3 → new experts [7]      (11 % 4 = 3)
```

**示例：4 → 12 扩展（3 倍）**
```
src expert 0 → new experts [4, 8]
src expert 1 → new experts [5, 9]
src expert 2 → new experts [6, 10]
src expert 3 → new experts [7, 11]
```

### 5.2 验证：`validate_expert_layout`

在扩增前验证每个 MoE 层的专家索引是连续的 `[0, original_experts)`。如果索引不连续（可能是已经扩展过的模型或者损坏的模型）则直接拒绝，防止数据错误。

### 5.3 Pass 2 中的实现

```python
elif info := get_expert_info(key):
    layer_idx, expert_idx, rest = info
    # 保留原始专家
    current_tensors[key] = tensor
    # 添加新专家的副本
    for new_expert_idx in source_to_targets.get(expert_idx, []):
        new_key = f"model.layers.{layer_idx}.mlp.experts.{new_expert_idx}.{rest}"
        current_tensors[new_key] = tensor.clone()
```

使用 `tensor.clone()` 确保新旧专家的 tensor 在内存中独立，避免共享 storage。

## 6. 路由器扩展

### 6.1 标准扩展（无 zero_expert）

```python
new_tensor = torch.cat([tensor] * expansion_factor, dim=0)
```

- `classifier.weight`: `(N, hidden_size)` → `(2N, hidden_size)`
- `e_score_correction_bias`: `(N,)` → `(2N,)`

前 N 行和后 N 行的值完全相同，因此在扩增初期新旧专家获得的初始路由分数一致。

### 6.2 带 zero_expert_num 的扩展（重要边界情况）

**背景：** LongCat 模型使用"零专家"机制。`zero_expert_num=256` 意味着有 256 个虚拟专家，它们执行恒等映射（直接透传输入），不执行实际 MLP 计算。路由器实际维度 = `n_routed_experts + zero_expert_num`（例如 512 + 256 = 768）。

```python
# modeling_longcat_flash.py 中的实现
self.n_routed_experts = (
    config.n_routed_experts                       # 512 个真实专家
    if config.zero_expert_num is None
    else config.n_routed_experts + config.zero_expert_num  # 512 + 256 = 768
)
self.classifier = nn.Linear(config.hidden_size, self.n_routed_experts)
# → classifier.weight shape = (768, 6144)
```

**问题：** 直接 `torch.cat([tensor] * 2, dim=0)` 会导致：
```
[real(512) | zero(256) | real(512) | zero(256)]  ← 错误！zero 部分也被翻倍了
```

这将产生 `(1536, hidden_size)` 的路由器，其中 zero 专家权重出现在位置 512-767 和 1280-1535，结构混乱。

**正确做法：** 只复制真实专家部分的权重，zero 专家部分保持不变：

```python
if zero_expert_num > 0:
    real_part = tensor[:original_experts]    # 前 original_experts 行
    zero_part = tensor[original_experts:]    # 后 zero_expert_num 行
    expanded_real = torch.cat([real_part] * expansion_factor, dim=0)
    new_tensor = torch.cat([expanded_real, zero_part], dim=0)
else:
    new_tensor = torch.cat([tensor] * expansion_factor, dim=0)
```

**结果示例（512 → 1024，zero_expert_num=256）：**
```
输入:  [real(512) | zero(256)]  → shape (768, hidden_size)
输出:  [real(512) | real(512) | zero(256)]  → shape (1280, hidden_size)
                              ↑ zero 保持在末尾，数量不变
```

**配置输出：** `n_routed_experts` 更新为 `target_experts`，`zero_expert_num` 保持 256 不变。

### 6.3 Pass 1 中的路由器大小估算

在 Pass 1 中需要精确计算路由器扩增后的字节数：

```python
if is_router_param(key):
    validate_router_shape(key, shape, total_routed)
    # 新大小 = 原始字节数 × (target_experts + zero_expert_num) / total_routed
    rows_per_expert = nbytes / total_routed
    new_nbytes = int(rows_per_expert * (target_experts + zero_expert_num))
```

这里 key 对应的是 `total_routed = original_experts + zero_expert_num` 而非 `original_experts`。

## 7. Shard 大小自动检测

```python
def auto_detect_shard_size(model_dir, shard_files):
    # 从磁盘上已有的 shard 文件检测平均大小
    # 如果文件不存在（例如只下载了 index 和 config），回退到 8GB 默认值
    # 输出 shard 与原始 shard 大小保持一致
```

## 8. 安全网：Shard 数量不匹配处理

如果 Pass 2 写入的实际 shard 数量与 Pass 1 预测的不同（浮点精度累积等边缘情况），脚本会：

1. 将已写入的文件从 `model_N-of-{predicted}.safetensors` 重命名为 `model_N-of-{actual}.safetensors`
2. 更新 index 中的 shard 计数引用

## 9. dtype 类型大小表

```python
DTYPE_SIZES = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E5M2FN": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}
```

支持所有标准 safetensors 数据类型，包括 FP8 各变种。使用 `DTYPE_SIZES[dtype]` 直接索引（而非 `.get(dtype, default)`），确保未知 dtype 会立即报 KeyError 而非静默产生错误大小。

## 10. 辅助文件处理

**自动复制到输出目录：**
- `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`
- `modeling_*.py`、`configuration_*.py`
- `.gitattributes`、`README.md`、`LICENSE`

**跳过（已在脚本中重新生成）：**
- `.safetensors`、`.bin`、`.pt`、`.pth`、`.ckpt`、`.h5`（权重文件）
- `config.json`、`model.safetensors.index.json`

## 11. 配置更新示例

### 输入 config.json（LongCat 500B）
```json
{
  "n_routed_experts": 512,
  "zero_expert_num": 256,
  "zero_expert_type": "identity",
  "num_layers": 28,
  "hidden_size": 6144,
  "expert_ffn_hidden_size": 2048,
  "moe_topk": 12
}
```

### 输出 config.json（翻倍，topk 不变）
```json
{
  "n_routed_experts": 1024,
  "zero_expert_num": 256,
  "zero_expert_type": "identity",
  "num_layers": 28,
  "hidden_size": 6144,
  "expert_ffn_hidden_size": 2048,
  "moe_topk": 12
}
```

### 输出 config.json（翻倍 + topk=24）
```json
{
  "n_routed_experts": 1024,
  "zero_expert_num": 256,
  "zero_expert_type": "identity",
  "num_layers": 28,
  "hidden_size": 6144,
  "expert_ffn_hidden_size": 2048,
  "moe_topk": 24
}
```

## 12. 使用方法

### Shell 脚本

```bash
# 默认：自动翻倍专家数，topk 不变
bash scripts/expand_moe_experts.sh

# 指定目标专家数，topk 不变
bash scripts/expand_moe_experts.sh 1024

# 指定目标专家数 + 目标 topk
bash scripts/expand_moe_experts.sh 1024 24

# 只指定 topk（专家数自动翻倍）
bash scripts/expand_moe_experts.sh "" 24

# 自定义路径
MODEL_DIR=/path/to/source \
OUTPUT_DIR=/path/to/output \
bash scripts/expand_moe_experts.sh 1024 24
```

### Python 脚本

```bash
# 翻倍专家数
python3 utils/expand_moe_experts.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --target_experts 1024

# 翻倍专家数 + 调整 topk
python3 utils/expand_moe_experts.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --target_experts 1024 \
  --target_topk 24
```

## 13. 测试覆盖

| 测试 | 验证内容 |
|------|---------|
| `test_expansion` | 标准 4→8 专家，`mlp.router.*` 命名，路由器形状和值一致性 |
| `test_expansion_with_gate_router_and_n_experts_key` | `n_experts` 配置键 + `mlp.gate.*` 路由器命名风格 |
| `test_expansion_with_zero_experts` | `zero_expert_num=2`，验证路由器拆分为 real/zero 部分的正确性 |
| `test_target_topk_updates_config` | `--target_topk 24` 更新已有 `moe_topk` 键 |
| `test_target_topk_adds_key_when_not_present` | `--target_topk 16` 在缺少 topk 键时自动添加 |
| `test_expand_moe_experts_shell_script` | Shell 脚本端到端调用，默认自动翻倍 |

## 14. 使用 LongCat 500B 模型时的注意事项

1. **`zero_expert_num`**：扩增后保持 256 不变。路由器维度从 768 → 1280（1024 real + 256 zero），而非 1536。如果也要扩增 zero_expert_num，需要在 config.json 中手动修改。

2. **`zero_expert_type`**：保持 `"identity"` 不变。零专家在扩增后继续执行透传。

3. **`expert_ffn_hidden_size`**：保持 2048 不变。专家的内部结构（gate/up/down 维度）不发生变化。

4. **`moe_topk`**：默认保持 12 不变。每个 token 仍然选择 12 个专家。可通过 `--target_topk` 参数调整（如 `--target_topk 24`）。增大 topk 可让新增专家获得更多训练信号，但会增加每 token 计算量。

5. **`routed_scaling_factor`**：保持 6.0 不变。如果调整了 `moe_topk`，可能需要同时手动调整此值以保持输出分布稳定。

6. **非 MoE 参数不受影响：**
   - `self_attn.{0,1}.*` — MLA 注意力参数
   - `input_layernorm.{0,1}.*` / `post_attention_layernorm.{0,1}.*` — 双归一化层
   - `mlps.{0,1}.*` — 密集 FFN 层（非 MoE）
   - `embed_tokens`、`norm`、`lm_head`
   - `mtp.*` — Multi-Token Prediction 模块
   
   以上全部原样保留，不参与扩增。

7. **磁盘空间估算**：输出模型大小取决于专家部分占比。对于 500B 参数的 LongCat 模型，MoE FFN 占大多数参数，扩增后约 1.8x 原始体积。

8. **性能恢复**：扩增后必须继续训练（fine-tuning）来恢复性能。直接使用未训练的扩增模型效果会很差——路由器不知道如何利用新专家，新增专家的路由概率接近于随机。

## 15. 与方案 1（增加层数）的组合

两个方案可以独立使用，也可以组合实现更大的模型：

1. 先执行方案 1：28 层 → 56 层（参数翻倍，每层结构和专家数不变）
2. 再执行方案 2：512 专家 → 1024 专家（每层的专家数翻倍）

最终得到 **56 层、1024 专家** 的模型（约 1T 参数）。两个脚本调用顺序无关紧要——它们修改不同的参数维度和不同的 config 键。
