"""Shared utilities for model expansion scripts."""

import json
import re
from pathlib import Path

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
    with open(model_dir / "config.json") as f:
        return json.load(f)


def load_index(model_dir: Path) -> dict | None:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return None


def get_layer_index(param_name: str) -> int | None:
    """Extract layer index from parameter name. Returns None for non-layer params."""
    m = re.search(r"model\.layers\.(\d+)\.", param_name)
    if m:
        return int(m.group(1))
    return None


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name."""
    return re.sub(
        r"model\.layers\.(\d+)\.",
        f"model.layers.{new_index}.",
        param_name,
    )


def get_expert_info(param_name: str) -> tuple[int, int, str] | None:
    """Extract (layer_idx, expert_idx, rest) from parameter name."""
    m = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", param_name)
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
