"""Shared utilities for model expansion scripts."""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

# ── Pre-compiled regex patterns (hot-path: called for every tensor key) ──
_LAYER_INDEX_RE = re.compile(r"model\.layers\.(\d+)\.")
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
    return None


def get_layer_index(param_name: str) -> int | None:
    """Extract layer index from parameter name. Returns None for non-layer params."""
    m = _LAYER_INDEX_RE.search(param_name)
    if m:
        return int(m.group(1))
    return None


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name.

    Uses string split/join (~4× faster than re.sub for this simple pattern).
    """
    if not param_name.startswith(_LAYER_PREFIX):
        return param_name
    rest = param_name[len(_LAYER_PREFIX):]  # strip "model.layers."
    dot_pos = rest.find(".")
    if dot_pos == -1:
        return param_name
    return f"{_LAYER_PREFIX}{new_index}.{rest[dot_pos + 1:]}"


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


def layer_sort_key(output_key: str) -> tuple[int, str]:
    """Sort key for layer-aware ordering: non-layer first, then by layer, then by name."""
    return (_layer_group(output_key), output_key)


def assign_shards_layer_aware(
    items: list[tuple],
    target_shard_size: int,
    max_layers_per_shard: int = 1,
) -> tuple[dict[int, list[tuple[str, str, str, str]]], int, int]:
    """Assign output tensors to shards, limiting the number of layers per shard.

    Items should be pre-sorted by layer (via :func:`layer_sort_key`) for
    optimal grouping.  Each tuple is ``(input_shard, input_key, output_key,
    output_nbytes, action)``.

    A new shard is started when:
    * Adding a tensor from a new layer would exceed ``max_layers_per_shard``
    * The current shard would exceed ``target_shard_size``
    * A single tensor exceeds ``target_shard_size`` (warn and force into its
      own shard)
    * Adding a tensor from a different non-layer module (e.g. embed_tokens vs
      lm_head) would mix different modules in the same shard

    Non-layer tensors are grouped by module family (embed_tokens, lm_head,
    norm, etc.) so that each module's tensors stay together in the same
    shard, matching the original model's file organization.

    Returns ``(assignments, num_shards, total_bytes)`` where *assignments*
    maps shard index → list of ``(input_shard, input_key, output_key, action)``.
    """
    import warnings

    current_shard = 0
    current_bytes = 0
    current_groups: set[int | str] = set()
    total_bytes = 0
    assignments: dict[int, list[tuple[str, str, str, str]]] = defaultdict(list)
    warned_once = False

    for input_shard, input_key, output_key, output_nbytes, action in items:
        group = _layer_group(output_key)
        module = _module_group(output_key)

        # Warn when a single tensor is larger than the target shard size.
        # This can happen after router expansion (e.g. [768, 6144] -> [1536, 6144])
        # or with very large embedding tables.  We still place the tensor, but it
        # will occupy a shard by itself (or force the current shard to exceed
        # the target).
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

        # Decide whether we need a new shard
        new_shard = False
        if current_bytes > 0:
            # Check if current shard contains any layer groups (int >= 0)
            has_layer_group = any(isinstance(g, int) and g >= 0 for g in current_groups)
            # Check if current shard contains any non-layer groups (str, i.e. module names)
            has_nonlayer_group = any(isinstance(g, str) for g in current_groups)

            if group >= 0 and group not in current_groups:
                # Real layer group that is not yet in the current shard
                if len(current_groups) >= max_layers_per_shard:
                    new_shard = True
                elif has_nonlayer_group:
                    # Cannot mix real layer with non-layer tensors
                    new_shard = True
            elif group == -1 and len(current_groups) > 0:
                # Non-layer tensors: check if this is a different module family
                # We allow multiple non-layer tensors from the SAME module
                # (e.g. embed_tokens.weight + embed_tokens.bias) in one shard,
                # but do NOT mix different modules (e.g. embed_tokens + lm_head).
                if has_layer_group:
                    # Cannot mix non-layer with real layer tensors
                    new_shard = True
                elif module not in current_groups:
                    # Different non-layer module family (e.g. embed_tokens vs lm_head)
                    new_shard = True

        if not new_shard and current_bytes + output_nbytes > target_shard_size and current_bytes > 0:
            # Non-layer params: only split if this is a different module family.
            # Same module's tensors stay together even if they exceed target size.
            if group == -1:
                if module not in current_groups:
                    new_shard = True
            else:
                new_shard = True

        if new_shard:
            current_shard += 1
            current_bytes = 0
            current_groups = set()

        assignments[current_shard].append(
            (input_shard, input_key, output_key, action))
        current_bytes += output_nbytes
        # Use module name for non-layer groups so that _module_group is the
        # grouping key (embed_tokens, lm_head, norm, etc.)
        current_groups.add(module if group == -1 else group)
        total_bytes += output_nbytes

    num_shards = current_shard + 1 if assignments else 0
    return dict(assignments), num_shards, total_bytes
