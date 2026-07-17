"""Shared utilities for model expansion scripts."""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

# ── Pre-compiled regex patterns (hot-path: called for every tensor key) ──
# Support multiple common layer naming conventions across HF model families:
#   - model.layers.N.* (Llama, Qwen, Mistral, LongCat, etc.)
#   - transformer.h.N.* (GPT-2, GPT-Neo, etc.)
#   - model.decoder.layers.N.* (BART, T5, etc.)
#   - encoder.layer.N.* (BERT, RoBERTa, etc.)
#   - transformer.blocks.N.* (MPT, etc.)
_LAYER_INDEX_RE = re.compile(
    r"(?:model\.layers\.|transformer\.h\.|model\.decoder\.layers\.|encoder\.layer\.|transformer\.blocks\.)"
    r"(\d+)\."
)
_EXPERT_INFO_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)")
_LAYER_PREFIX = "model.layers."

# Mapping from safetensors dtype string to element size in bytes
DTYPE_SIZES: dict[str, int] = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E5M2FN": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}

EXPERT_COUNT_KEYS = ["n_routed_experts", "n_experts", "num_experts"]
LAYER_COUNT_KEYS = ["num_layers", "num_hidden_layers", "n_layers", "n_layer"]

ROUTER_WEIGHT_SUFFIXES = (
    "mlp.router.classifier.weight",
    "mlp.gate.weight",
)
ROUTER_BIAS_SUFFIXES = (
    "mlp.router.e_score_correction_bias",
    "mlp.gate.e_score_correction_bias",
)
ALL_ROUTER_SUFFIXES = ROUTER_WEIGHT_SUFFIXES + ROUTER_BIAS_SUFFIXES


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def load_index(model_dir: Path) -> dict | None:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    # Support single-file models (no index, just model.safetensors)
    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        # Build a synthetic index
        return {
            "metadata": {"total_size": single_path.stat().st_size},
            "weight_map": {},  # will be populated from header scan
        }
    return None


def get_shard_files(model_dir: Path, index: dict | None) -> list[str]:
    """Get list of shard files from index, or single file for non-sharded models."""
    if index is None:
        return []
    weight_map = index.get("weight_map", {})
    if weight_map:
        return sorted(set(weight_map.values()))
    # Single-file model: synthetic index
    single = model_dir / "model.safetensors"
    if single.exists():
        return ["model.safetensors"]
    return []


def get_layer_index(param_name: str) -> int | None:
    """Extract layer index from parameter name. Returns None for non-layer params."""
    m = _LAYER_INDEX_RE.search(param_name)
    if m:
        return int(m.group(1))
    return None


# Supported layer prefixes for set_layer_index (order matters for matching)
_LAYER_PREFIXES = [
    "model.layers.",
    "transformer.h.",
    "model.decoder.layers.",
    "encoder.layer.",
    "transformer.blocks.",
]


def _find_layer_prefix(param_name: str) -> str | None:
    """Find which layer prefix a parameter name uses."""
    for prefix in _LAYER_PREFIXES:
        if param_name.startswith(prefix):
            return prefix
    return None


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name.

    Supports multiple common layer naming conventions.
    Uses string split/join (~4× faster than re.sub for this simple pattern).
    """
    prefix = _find_layer_prefix(param_name)
    if prefix is None:
        return param_name
    rest = param_name[len(prefix):]  # strip prefix
    dot_pos = rest.find(".")
    if dot_pos == -1:
        return param_name
    return f"{prefix}{new_index}.{rest[dot_pos + 1:]}"


def get_expert_info(param_name: str) -> tuple[int, int, str] | None:
    """Extract (layer_idx, expert_idx, rest) from parameter name."""
    m = _EXPERT_INFO_RE.search(param_name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def find_expert_count(config: dict) -> tuple[str | None, int, int]:
    """Read expert count and zero_expert_num from config.

    Returns (expert_count_key, original_experts, zero_expert_num).
    """
    for key in EXPERT_COUNT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            zero = config.get("zero_expert_num", 0) or 0
            return key, value, zero
    return None, 0, 0


def is_router_param(param_name: str) -> bool:
    """Check if the parameter is a router weight or bias tensor."""
    return param_name.endswith(ALL_ROUTER_SUFFIXES)


def is_router_weight(param_name: str) -> bool:
    """Check if the parameter is a router classifier/gate weight (not bias)."""
    return param_name.endswith(ROUTER_WEIGHT_SUFFIXES)


def is_router_bias(param_name: str) -> bool:
    """Check if the parameter is a router score correction bias (not weight)."""
    return param_name.endswith(ROUTER_BIAS_SUFFIXES)


def tensor_nbytes(tensor) -> int:
    """Return the size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def get_nbytes_from_meta(dtype: str, shape: list[int]) -> int:
    """Compute tensor byte size from safetensors metadata."""
    elem_size = DTYPE_SIZES[dtype]
    numel = 1
    for dim in shape:
        numel *= dim
    return elem_size * numel


def read_safetensors_header(path: Path) -> dict[str, tuple[str, list[int]]]:
    """Read only the JSON header of a safetensors file.

    Returns {tensor_name: (dtype, shape)}. No tensor data is loaded.
    """
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))

    result: dict[str, tuple[str, list[int]]] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        result[key] = (meta["dtype"], meta["shape"])
    return result


def make_expert_key(layer_idx: int, expert_idx: int, rest: str) -> str:
    """Construct an expert parameter key from its components."""
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{rest}"


def parse_copy_source(raw: str | None, num_original: int, num_new: int) -> list[int]:
    """Parse --copy_source into a mapping: offset → source_layer_index.

    offset runs from 0 to num_new - 1 (maps to target layers num_original + offset).

    Formats:
      None / "seq"  →  sequential: offset i → layer (i mod num_original)
      "5"           →  all new layers copy from layer 5
      "0,0,1,1,…"   →  explicit list, must have exactly num_new entries

    Raises ValueError on invalid input.
    """
    if raw is None or raw.strip().lower() == "seq":
        return [i % num_original for i in range(num_new)]

    raw = raw.strip()
    try:
        single = int(raw)
        if single < 0 or single >= num_original:
            raise ValueError(
                f"--copy_source {single} is out of range [0, {num_original - 1}]")
        return [single] * num_new
    except ValueError as e:
        if "out of range" in str(e):
            raise

    try:
        parts = [int(p.strip()) for p in raw.split(",")]
    except ValueError:
        raise ValueError(f"Invalid --copy_source format: {raw}")

    if len(parts) != num_new:
        raise ValueError(
            f"--copy_source list has {len(parts)} entries, expected {num_new} "
            f"(one per new layer)")

    for i, src in enumerate(parts):
        if src < 0 or src >= num_original:
            raise ValueError(
                f"--copy_source[{i}] = {src} is out of range [0, {num_original - 1}]")
    return parts


def auto_detect_shard_size(model_dir: Path, shard_files: list[str]) -> int:
    """Detect target shard size from existing shard files.

    Returns the average file size in bytes. Falls back to 8 GB if no files found.
    """
    file_sizes = []
    for fname in shard_files:
        fpath = model_dir / fname
        if fpath.exists():
            file_sizes.append(fpath.stat().st_size)

    if file_sizes:
        avg_size = int(sum(file_sizes) / len(file_sizes))
        print(f"Detected shard size from {len(file_sizes)} existing files: "
              f"{avg_size / 1e9:.2f} GB (average)")
        return avg_size

    print("WARNING: No shard files found on disk. Using default 8GB target. "
          "Output shards will match this size, not necessarily the originals.")
    return 8 * 1024 ** 3


def compute_optimal_shard_size(
    model_dir: Path,
    shard_files: list[str],
    config: dict,
    target_layers: int,
    target_experts: int,
    max_layers_per_shard: int = 1,
    ideal_shards_per_layer: int = 3,
) -> int:
    """Compute an optimal target shard size for expanded model layout.

    The goal is to minimize layer dispersion (number of shards a single layer
    spans) while keeping shard sizes uniform. This is done by:

    1. Scanning the original model headers to estimate per-layer data size
    2. Accounting for expansion factors (depth + experts)
    3. Computing a target size that yields ~ideal_shards_per_layer shards per layer

    Returns the computed target shard size in bytes.
    """
    from collections import defaultdict

    # ── Step 1: accumulate original tensor sizes by layer ──────────────────
    layer_bytes: dict[int, int] = defaultdict(int)
    non_layer_bytes: dict[str, int] = defaultdict(int)
    expert_bytes: dict[int, int] = defaultdict(int)
    other_layer_bytes: dict[int, int] = defaultdict(int)

    for fname in shard_files:
        fpath = model_dir / fname
        if not fpath.exists():
            continue
        header = read_safetensors_header(fpath)
        for key, (dtype, shape) in header.items():
            nbytes = get_nbytes_from_meta(dtype, shape)
            layer_idx = get_layer_index(key)
            if layer_idx is not None:
                layer_bytes[layer_idx] += nbytes
                if "mlp.experts" in key:
                    expert_bytes[layer_idx] += nbytes
                else:
                    other_layer_bytes[layer_idx] += nbytes
            else:
                # Group non-layer by module family
                module = _module_group(key)
                non_layer_bytes[module] += nbytes

    if not layer_bytes:
        print("WARNING: Could not estimate layer sizes. Using auto-detected shard size.")
        return auto_detect_shard_size(model_dir, shard_files)

    # ── Step 2: estimate expanded layer size ─────────────────────────────
    # Try multiple common layer count keys for different model families
    original_layers = 0
    for key in LAYER_COUNT_KEYS:
        if key in config and config[key] is not None:
            original_layers = config[key]
            break
    if original_layers == 0:
        original_layers = max(layer_bytes.keys()) + 1  # fallback from tensor keys

    # Try multiple common expert count keys for different model families
    original_experts = 0
    for key in EXPERT_COUNT_KEYS:
        if key in config and config[key] is not None:
            original_experts = config[key]
            break

    # Detect if this is actually a dense model (no expert weights found in tensors)
    total_expert = sum(expert_bytes.values())
    if original_experts == 0 and total_expert == 0:
        # Dense model: no experts, no expansion factor
        original_experts = 0
    elif original_experts == 0:
        # Model has expert weights but config missing key — fallback
        original_experts = 512

    # Handle single-expansion mode (depth-only or expert-only)
    # target_layers=0 means no depth expansion (expert-only)
    # target_experts=0 means no expert expansion (depth-only)
    if target_layers == 0:
        target_layers = original_layers
    if target_experts == 0:
        target_experts = original_experts

    # For dense models (no experts), expansion_factor is always 1.0 (no expert scaling)
    expansion_factor = target_experts / original_experts if original_experts > 0 else 1.0
    depth_factor = target_layers / original_layers

    # Calculate actual expert fraction from original model
    total_other = sum(other_layer_bytes.values())
    total_layer = total_expert + total_other

    if total_layer > 0:
        expert_fraction = total_expert / total_layer
    else:
        expert_fraction = 0.6  # fallback

    avg_layer_size = total_layer / len(layer_bytes)

    # After expansion:
    # - Expert weights scale by expansion_factor
    # - Other layer params (attention, norm, mlps) scale by depth_factor
    # - Router weights also scale by expansion_factor (dim0 expands)
    expanded_layer_size = avg_layer_size * (
        expert_fraction * expansion_factor + (1 - expert_fraction) * depth_factor
    )

    # Add a small safety margin for rounding and edge cases
    expanded_layer_size *= 1.02

    # ── Step 3: compute target shard size ────────────────────────────────
    # Compute original average shard size and total for reference
    original_avg_shard_size = 0
    original_total_size = 0
    if shard_files:
        sizes = [(model_dir / f).stat().st_size for f in shard_files if (model_dir / f).exists()]
        if sizes:
            original_avg_shard_size = int(sum(sizes) / len(sizes))
            original_total_size = sum(sizes)

    # Estimate original layers per shard from tensor data (not file sizes, which include overhead)
    if avg_layer_size > 0:
        original_layers_per_shard_data = original_avg_shard_size / avg_layer_size
    else:
        original_layers_per_shard_data = 1

    # Determine target number of output shards.
    # Goal: output shard count should be roughly proportional to model size increase,
    # while respecting practical limits on shard size.
    # Formula: target_shards ≈ original_shards × (expanded_size / original_size)
    #         = original_shards × (expanded_layer_size / avg_layer_size) × (target_layers / original_layers)
    # Simplified: target_shards ≈ original_shards × depth_factor × expansion_factor (for MoE)
    original_num_shards = len(shard_files)
    if original_num_shards > 0 and avg_layer_size > 0 and original_layers > 0:
        # Estimate how many shards the expanded model should have
        size_ratio = (expanded_layer_size / avg_layer_size) * (target_layers / original_layers)
        target_num_shards = max(1, int(original_num_shards * size_ratio))
    else:
        target_num_shards = max(1, target_layers)  # fallback: 1 shard per layer

    # Compute target size from desired shard count
    # Total expanded size = expanded_layer_size × target_layers + non_layer_bytes
    total_non_layer = sum(non_layer_bytes.values())
    # Non-layer params scale by depth_factor (embeddings, lm_head, etc.)
    expanded_non_layer = int(total_non_layer * depth_factor)
    total_expanded_size = int(expanded_layer_size * target_layers) + expanded_non_layer

    target_size = int(total_expanded_size / target_num_shards)

    # Round to a nice value for readability.
    if target_size < 5 * 1024 ** 3:  # < 5 GB
        rounding_unit = 100 * 1024 * 1024  # 0.1 GB
    else:
        rounding_unit = 512 * 1024 * 1024  # 0.5 GB
    target_size = ((target_size + rounding_unit // 2) // rounding_unit) * rounding_unit

    # Apply max_layers_per_shard constraint with anti-fragmentation guard.
    # If respecting max_layers_per_shard would produce too many tiny shards
    # (more than 2x original shard count), we relax the constraint to prevent
    # excessive fragmentation.
    max_size_for_layers = int(expanded_layer_size * max_layers_per_shard)
    if target_size > max_size_for_layers and max_layers_per_shard > 0:
        # Check if applying this constraint would cause shard explosion
        projected_shards = max(1, int(total_expanded_size / max_size_for_layers))
        if projected_shards > original_num_shards * 2 and original_num_shards > 1:
            # Too many shards would be created — relax constraint to keep shard count reasonable
            # But for no-expansion cases (size_ratio ≈ 1), prefer keeping original shard count
            if abs(size_ratio - 1.0) < 0.1:
                # No significant expansion: keep original number of shards
                relaxed_target = int(total_expanded_size / original_num_shards)
                print(f"  NOTE: No expansion detected, keeping original {original_num_shards} shards (~{relaxed_target / 1e9:.2f} GB each)")
            else:
                relaxed_target = int(total_expanded_size / (original_num_shards * 2))
                print(f"  NOTE: Relaxed max_layers_per_shard to avoid {projected_shards} shards (limit: {original_num_shards * 2}, ~{relaxed_target / 1e9:.2f} GB each)")
            relaxed_target = ((relaxed_target + rounding_unit // 2) // rounding_unit) * rounding_unit
            relaxed_target = max(max_size_for_layers, relaxed_target)
            target_size = min(target_size, relaxed_target)
        else:
            target_size = max_size_for_layers
            target_size = ((target_size + rounding_unit // 2) // rounding_unit) * rounding_unit
            print(f"  NOTE: Limited by max_layers_per_shard={max_layers_per_shard} to {target_size / 1e9:.2f} GB")

    # Minimum shard size: at least 0.1 GB, or if single-file, use file size
    single_file_size = 0
    if len(shard_files) == 1:
        sf_path = model_dir / shard_files[0]
        if sf_path.exists():
            single_file_size = sf_path.stat().st_size
    min_shard_size = max(
        100 * 1024 * 1024,  # 0.1 GB absolute minimum
        single_file_size if single_file_size > 0 else 0,
    )

    # Hard bounds: 0.1 GB ~ 50 GB per shard
    max_shard_size = 50 * 1024 ** 3

    if target_size > max_shard_size:
        target_size = max_shard_size
        target_size = ((target_size + rounding_unit // 2) // rounding_unit) * rounding_unit
        print(f"  NOTE: Capped at 50 GB max shard size")

    target_size = max(min_shard_size, min(max_shard_size, target_size))

    # Compute final shards per layer for reporting
    adjusted_shards = max(1, int(expanded_layer_size / target_size))

    print(f"\nOptimal shard size estimation:")
    print(f"  Original layers: {original_layers}, target layers: {target_layers}")
    print(f"  Original experts: {original_experts}, target experts: {target_experts}")
    print(f"  Expansion factor: {expansion_factor:.1f}x experts, {depth_factor:.1f}x depth")
    print(f"  Expert fraction: {expert_fraction*100:.1f}%")
    print(f"  Estimated layer size: {expanded_layer_size / 1e9:.2f} GB")
    print(f"  Target shards per layer: {adjusted_shards}")
    print(f"  Computed target shard size: {target_size / 1e9:.2f} GB")

    return target_size


# ═══════════════════════════════════════════════════════════════════════════════
# Shared expansion helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_expert_target_map(
    original_experts: int, target_experts: int,
) -> dict[int, list[int]]:
    """Build source expert -> list of new expert indices for duplication."""
    targets: dict[int, list[int]] = defaultdict(list)
    for new_idx in range(original_experts, target_experts):
        src_idx = new_idx % original_experts
        targets[src_idx].append(new_idx)
    return dict(targets)


def expand_router_weight(
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    router_noise_scale: float = 0.0,
) -> torch.Tensor:
    """Expand a router classifier/gate weight with optional noise on copies.

    When router_noise_scale > 0, duplicated blocks get small Gaussian noise to
    break symmetry so that fine-tuning can differentiate them.
    """
    if zero_expert_num > 0:
        real_part = tensor[:original_experts]
        zero_part = tensor[original_experts:]
    else:
        real_part = tensor
        zero_part = None

    real_blocks = [real_part]
    for _ in range(1, expansion_factor):
        if router_noise_scale > 0:
            noise = torch.randn_like(real_part) * router_noise_scale * real_part.std()
            real_blocks.append(real_part + noise)
        else:
            real_blocks.append(real_part)
    expanded_real = torch.cat(real_blocks, dim=0)

    if zero_part is not None:
        expanded_zero = torch.cat([zero_part] * expansion_factor, dim=0)
        return torch.cat([expanded_real, expanded_zero], dim=0)
    return expanded_real


def build_layer_mapping(
    original_layers: int,
    target_layers: int,
    source_list: list[int],
    insertion_mode: str,
) -> list[tuple[int, bool]]:
    """Build the final layer ordering as (source_layer, is_new).

    Returns a list of length target_layers where each entry indicates
    which original layer it comes from and whether it's a new identity layer.

    insertion_mode:
      - "interleave": insert new identity layers after their source layer
      - "append": original layers first, then new layers at the end
    """
    num_new = target_layers - original_layers

    if insertion_mode not in ("interleave", "append"):
        raise ValueError(
            f"Unknown insertion_mode '{insertion_mode}'. "
            f"Must be 'interleave' or 'append'.")

    if insertion_mode == "append":
        mapping = [(i, False) for i in range(original_layers)]
        for offset in range(num_new):
            mapping.append((source_list[offset], True))
        return mapping

    new_by_source: dict[int, int] = defaultdict(int)
    for src in source_list:
        new_by_source[src] += 1

    mapping = []
    for orig_idx in range(original_layers):
        mapping.append((orig_idx, False))
        count = new_by_source.get(orig_idx, 0)
        for _ in range(count):
            mapping.append((orig_idx, True))

    assert len(mapping) == target_layers, (
        f"Interleave mapping length {len(mapping)} != target {target_layers}. "
        f"This can happen if source_list references layers not in [0, original_layers). "
        f"source_list sources: {sorted(set(source_list))}"
    )
    return mapping


def expand_router_bias(
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
) -> torch.Tensor:
    """Expand a router score correction bias (exact copies, no noise)."""
    return expand_router_weight(
        tensor, original_experts, zero_expert_num, expansion_factor,
        router_noise_scale=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Identity-init zero patterns (used by depth expansion + verification)
# ═══════════════════════════════════════════════════════════════════════════════

ZERO_PATTERNS = [
    re.compile(r"self_attn\.(?:\d+\.)?o_proj\.weight$"),
    re.compile(r"mlp\.experts\.\d+\.down_proj\.weight$"),
    re.compile(r"mlps\.\d+\.down_proj\.weight$"),
    re.compile(r"mlp\.down_proj\.weight$"),
]


def should_zero(param_name: str) -> bool:
    """Check if a parameter in a new identity layer should be zeroed."""
    for pat in ZERO_PATTERNS:
        if pat.search(param_name):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Layer-aware shard assignment
# ═══════════════════════════════════════════════════════════════════════════════

def _layer_group(key: str) -> int:
    """Return the layer group for a parameter key.

    Non-layer params (embed, lm_head, norm) get group -1 so they are
    packed together in their own shard(s), separate from layer weights.
    """
    li = get_layer_index(key)
    return li if li is not None else -1


def _module_group(key: str) -> str:
    """Return the module family for a non-layer parameter key.

    For non-layer params (group == -1), group by the top-level module name
    so that embed_tokens, lm_head, norm etc. each stay in their own shard
    and are not mixed together.
    """
    # Layer params: use layer index as group (already handled by _layer_group)
    li = get_layer_index(key)
    if li is not None:
        return str(li)

    # Non-layer params: group by top-level module name
    if key.startswith("model."):
        # e.g. "model.embed_tokens.weight" -> "embed_tokens"
        # e.g. "model.norm.weight" -> "norm"
        rest = key[6:]  # strip "model."
        dot_pos = rest.find(".")
        if dot_pos != -1:
            return rest[:dot_pos]
    # Fallback: use the full key (should not happen for standard models)
    return key


def layer_sort_key(output_key: str) -> tuple[int, str, str]:
    """Sort key for layer-aware ordering: non-layer first, then by layer, then by module, then by name.

    For layer params, groups by layer index first, then by top-level module
    (self_attn, mlp, input_layernorm, etc.) to keep each layer's params
    tightly packed and avoid interleaving different modules across shards.
    """
    group = _layer_group(output_key)
    if group == -1:
        # Non-layer params: sort by module group, then by full key
        return (-1, _module_group(output_key), output_key)
    # Layer params: sort by layer index, then by module prefix, then by key
    # Extract module prefix (e.g. "self_attn", "mlp", "input_layernorm")
    module_prefix = ""
    if "model.layers." in output_key:
        # Strip "model.layers.N." to get the rest
        rest = output_key.split("model.layers.", 1)[1]
        # The first component after the layer number is the module prefix
        dot_pos = rest.find(".")
        if dot_pos != -1:
            module_prefix = rest[:dot_pos]
        else:
            module_prefix = rest
    return (group, module_prefix, output_key)


def assign_shards_layer_aware(
    items: list[tuple],
    target_shard_size: int,
    max_layers_per_shard: int = 1,
) -> tuple[dict[int, list[tuple[str, str, str, int, str]]], int, int]:
    """Assign output tensors to shards, limiting the number of layers per shard.

    Items should be pre-sorted by layer (via :func:`layer_sort_key`) for
    optimal grouping.  Each tuple is ``(input_shard, input_key, output_key,
    output_nbytes, action)``.

    **Sharding rules**:

    *Layer params* (``model.layers.N.*``):
      - Grouped by layer index — at most ``max_layers_per_shard`` distinct
        layers per output shard.
      - Size-based splitting applies normally.

    *Non-layer params* (embed, lm_head, norm, etc.):
      - Separated into dedicated shards, never mixed with layer params.
      - Grouped by *module family* so that e.g. ``embed_tokens`` and
        ``lm_head`` each get their own shard(s), mirroring the original
        model layout.  Tensors from the same module family stay together
        even if they exceed ``target_shard_size``.

    *Giant tensors* (single tensor > target size):
      - A warning is emitted once; the tensor is placed in its own shard.

    Returns ``(assignments, num_shards, total_bytes)`` where *assignments*
    maps shard index → list of ``(input_shard, input_key, output_key, output_nbytes, action)``.
    """
    import warnings

    current_shard = 0
    current_bytes = 0
    # Two separate sets: layer indices (int) and module names (str).
    # Never mixed — the sort order (non-layer first) guarantees a clean
    # transition boundary.
    current_layers: set[int] = set()
    current_modules: set[str] = set()
    total_bytes = 0
    assignments: dict[int, list[tuple[str, str, str, int, str]]] = defaultdict(list)
    warned_once = False

    for input_shard, input_key, output_key, output_nbytes, action in items:
        layer_idx = get_layer_index(output_key)
        module = _module_group(output_key) if layer_idx is None else None

        # ── Giant tensor warning ─────────────────────────────────────────
        if output_nbytes > target_shard_size and not warned_once:
            warnings.warn(
                f"Tensor {output_key!r} ({output_nbytes / 1e9:.2f} GB) exceeds "
                f"target shard size ({target_shard_size / 1e9:.2f} GB). "
                f"It will be placed in a shard by itself. "
                f"Consider increasing target shard size or using a smaller model.",
                UserWarning,
                stacklevel=3,
            )
            warned_once = True

        # ── Decide: start a new shard? ────────────────────────────────────
        new_shard = False

        if current_bytes > 0:
            if layer_idx is not None:
                # ── Layer param ──────────────────────────────────────────
                if current_modules:
                    # Non-layer → layer transition (sort order guarantees
                    # this is the first layer param after non-layer block)
                    new_shard = True
                elif layer_idx not in current_layers and len(current_layers) >= max_layers_per_shard:
                    # Adding a new layer would exceed the per-shard limit
                    new_shard = True
            else:
                # ── Non-layer param ───────────────────────────────────────
                assert module is not None
                if current_layers:
                    # Layer → non-layer transition (rare, but handle it)
                    new_shard = True
                elif module not in current_modules and current_modules:
                    # Different non-layer module family
                    new_shard = True

        # ── Size-based splitting ──────────────────────────────────────────
        if not new_shard and current_bytes + output_nbytes > target_shard_size and current_bytes > 0:
            if layer_idx is not None:
                # Layer params: prefer to keep the entire layer together.
                # Only split if adding this tensor would push us well beyond
                # the target (1.5×) — this prevents tiny leftover shards when
                # a layer barely exceeds the target.
                if current_bytes + output_nbytes > int(target_shard_size * 1.5):
                    new_shard = True
            else:
                # Non-layer: same module stays together, but cap at 1.5×
                # target to avoid single giant shards when the module has
                # many large tensors (e.g. ngram embeddings with 12×5 GB).
                if current_bytes + output_nbytes > int(target_shard_size * 1.5):
                    new_shard = True

        # ── Apply ─────────────────────────────────────────────────────────
        if new_shard:
            current_shard += 1
            current_bytes = 0
            current_layers = set()
            current_modules = set()

        assignments[current_shard].append(
            (input_shard, input_key, output_key, output_nbytes, action))
        current_bytes += output_nbytes
        if layer_idx is not None:
            current_layers.add(layer_idx)
        else:
            current_modules.add(module)
        total_bytes += output_nbytes

    num_shards = current_shard + 1 if assignments else 0
    return dict(assignments), num_shards, total_bytes
