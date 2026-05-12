#!/usr/bin/env python3
"""
Expand the number of MoE experts in a sharded safetensors model.

For a model with N experts, creates a 2N-expert version by:
1. Duplicating each expert weight (expert E and expert E+N will be identical initially)
2. Expanding the router classifier weights by concatenating them with themselves.
3. Expanding the score correction bias similarly.

Processes shards one at a time to keep memory usage manageable.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Mapping from safetensors dtype string to element size in bytes
DTYPE_SIZES: dict[str, int] = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E5M2FN": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return None


def get_expert_info(param_name: str) -> tuple[int, int, str] | None:
    """Extract (layer_idx, expert_idx, rest) from parameter name.
    Example: model.layers.0.mlp.experts.5.down_proj.weight -> (0, 5, 'down_proj.weight')
    """
    m = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", param_name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def is_router_param(param_name: str) -> bool:
    """Check if the parameter is a router classifier or bias."""
    return "mlp.router.classifier.weight" in param_name or "mlp.router.e_score_correction_bias" in param_name


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the size of a torch tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def read_safetensors_header(path: Path) -> dict[str, tuple[str, list[int]]]:
    """Read only the JSON header of a safetensors file, return {tensor_name: (dtype, shape)}."""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))

    result: dict[str, tuple[str, list[int]]] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        result[key] = (meta["dtype"], meta["shape"])
    return result


def get_nbytes_from_meta(dtype: str, shape: list[int]) -> int:
    elem_size = DTYPE_SIZES.get(dtype, 4)  # default 4 if unknown
    numel = 1
    for dim in shape:
        numel *= dim
    return elem_size * numel


def main():
    parser = argparse.ArgumentParser(
        description="Expand MoE experts by duplicating weights and expanding routers"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the original model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output the expanded model",
    )
    parser.add_argument(
        "--target_experts",
        type=int,
        default=None,
        help="Target number of experts. Defaults to double the original.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load config & index ──────────────────────────────────────────────
    config = load_config(model_dir)
    index = load_index(model_dir)

    if not index:
        print("ERROR: Model is not sharded. This script handles sharded models.", file=sys.stderr)
        sys.exit(1)

    original_experts = config.get("n_routed_experts", 0)
    if original_experts == 0:
        print("ERROR: Could not find 'n_routed_experts' in config.json", file=sys.stderr)
        sys.exit(1)

    target_experts = args.target_experts if args.target_experts else original_experts * 2
    expansion_factor = target_experts // original_experts

    if target_experts % original_experts != 0:
        print(f"ERROR: Target experts ({target_experts}) must be a multiple of original ({original_experts})")
        sys.exit(1)

    print(f"Expanding experts: {original_experts} -> {target_experts} (factor: {expansion_factor})")

    shard_files = sorted(set(index["weight_map"].values()))

    # ── Update & write config ───────────────────────────────────────────
    updated_keys = []
    for key in ["n_routed_experts", "num_experts", "n_experts"]:
        if key in config:
            config[key] = target_experts
            updated_keys.append(key)
    
    if not updated_keys:
        print(f"WARNING: No expert count key found in config.json. Adding 'n_routed_experts'.")
        config["n_routed_experts"] = target_experts
        updated_keys.append("n_routed_experts")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Updated config.json written (keys updated: {', '.join(updated_keys)}).")

    # ── Pass 1: Scan headers to determine output layout ─────────────────
    print("\nPass 1/2: Scanning headers to determine output layout...")
    total_output_bytes = 0
    
    # We'll use a target shard size similar to the average original shard size
    total_original_size = index.get("metadata", {}).get("total_size", 0)
    if total_original_size == 0:
        # Fallback: sum up file sizes
        for sf in shard_files:
            total_original_size += (model_dir / sf).stat().st_size
    
    avg_shard_size = total_original_size // len(shard_files)
    target_shard_size = avg_shard_size
    print(f"Target shard size: {target_shard_size / 1e9:.2f} GB")

    for shard_file in tqdm(shard_files, desc="Scanning"):
        header = read_safetensors_header(model_dir / shard_file)
        for key, (dtype, shape) in header.items():
            nbytes = get_nbytes_from_meta(dtype, shape)
            
            if is_router_param(key):
                # Router weights are expanded by expansion_factor
                total_output_bytes += nbytes * expansion_factor
            elif get_expert_info(key):
                # Expert weights are duplicated
                total_output_bytes += nbytes * expansion_factor
            else:
                # Other weights stay the same
                total_output_bytes += nbytes

    num_output_shards = int(total_output_bytes // target_shard_size) + 1
    print(f"Planned output size: {total_output_bytes / 1e9:.2f} GB across ~{num_output_shards} shards")

    # ── Pass 2: Process and write ───────────────────────────────────────
    new_weight_map: dict[str, str] = {}
    output_shard_idx = 1
    current_tensors: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush_shard():
        nonlocal output_shard_idx, current_tensors, current_bytes
        if not current_tensors:
            return
        shard_name = f"model_{output_shard_idx:05d}-of-{num_output_shards:05d}.safetensors"
        output_path = output_dir / shard_name
        save_file(current_tensors, str(output_path))
        for t_name in current_tensors:
            new_weight_map[t_name] = shard_name
        output_shard_idx += 1
        current_tensors.clear()
        current_bytes = 0

    print("\nPass 2/2: Processing and writing tensors...")
    for shard_file in tqdm(shard_files, desc="Input shards"):
        with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                nbytes = tensor_nbytes(tensor)

                if is_router_param(key):
                    # Expand router: cat([W, W, ...], dim=0)
                    # For classifier.weight (out_features, in_features)
                    # For e_score_correction_bias (out_features)
                    new_tensor = torch.cat([tensor] * expansion_factor, dim=0)
                    new_nbytes = tensor_nbytes(new_tensor)
                    if current_bytes + new_nbytes > target_shard_size and current_tensors:
                        flush_shard()
                    current_tensors[key] = new_tensor
                    current_bytes += new_nbytes
                
                elif info := get_expert_info(key):
                    layer_idx, expert_idx, rest = info
                    # Duplicate experts
                    for i in range(expansion_factor):
                        new_expert_idx = expert_idx + i * original_experts
                        new_key = f"model.layers.{layer_idx}.mlp.experts.{new_expert_idx}.{rest}"
                        if current_bytes + nbytes > target_shard_size and current_tensors:
                            flush_shard()
                        current_tensors[new_key] = tensor.clone() if i > 0 else tensor
                        current_bytes += nbytes
                
                else:
                    # Regular param
                    if current_bytes + nbytes > target_shard_size and current_tensors:
                        flush_shard()
                    current_tensors[key] = tensor
                    current_bytes += nbytes

    flush_shard()

    # ── Write new index ──────────────────────────────────────────────────
    actual_shards = output_shard_idx - 1
    new_index = {
        "metadata": {"total_size": total_output_bytes},
        "weight_map": {k: re.sub(r"-of-\d+\.safetensors", f"-of-{actual_shards:05d}.safetensors", v) 
                       for k, v in new_weight_map.items()}
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy auxiliary files
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if fpath.is_file() and fpath.suffix not in skip_suffixes and fpath.name not in skip_names:
            shutil.copy2(fpath, output_dir / fpath.name)

    print(f"\nDone! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
