#!/usr/bin/env python3
"""
MoE Depth Expansion (M2): Insert identity-initialized MoE layers.

Implements the "MoE 层深度扩展" strategy from llm_param_expansion.md:
- Copies existing MoE layers to create new layers
- Zeros out o_proj weights (attention output → zero)
- Zeros out down_proj weights in ALL experts and shared MLPs
- Result: new layers satisfy Layer(x) ≈ x via residual connection

This is function-preserving: the expanded model produces identical outputs
to the original model at initialization.

Usage:
  python expand_moe_depth.py \
      --model_dir ./original_model \
      --output_dir ./expanded_model \
      [--original_layers 28] \
      [--target_layers 56] \
      [--copy_source seq] \
      [--insertion_mode interleave|append]
"""

import argparse
import copy
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

from utils.shared import (
    auto_detect_shard_size,
    get_layer_index,
    get_nbytes_from_meta,
    load_config,
    load_index,
    parse_copy_source,
    read_safetensors_header,
    set_layer_index,
    tensor_nbytes,
)

ZERO_PATTERNS = [
    re.compile(r"self_attn\.\d*\.?o_proj\.weight$"),
    re.compile(r"self_attn\.o_proj\.weight$"),
    re.compile(r"mlp\.experts\.\d+\.down_proj\.weight$"),
    re.compile(r"mlps\.\d+\.down_proj\.weight$"),
    re.compile(r"mlp\.down_proj\.weight$"),
]


def should_zero(param_name: str) -> bool:
    """Check if a parameter in a new layer should be zeroed for identity init."""
    for pat in ZERO_PATTERNS:
        if pat.search(param_name):
            return True
    return False


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
      - "interleave": insert new layers after each original layer
      - "append": original layers first, then new layers at the end
    """
    num_new = target_layers - original_layers

    if insertion_mode == "append":
        mapping = [(i, False) for i in range(original_layers)]
        for offset in range(num_new):
            mapping.append((source_list[offset], True))
        return mapping

    mapping = []
    new_per_original = [0] * original_layers
    for offset, src in enumerate(source_list):
        new_per_original[src] += 1

    new_offset = 0
    for orig_idx in range(original_layers):
        mapping.append((orig_idx, False))
        count = new_per_original[orig_idx]
        for _ in range(count):
            mapping.append((source_list[new_offset], True))
            new_offset += 1

    if len(mapping) != target_layers:
        leftover = num_new - new_offset
        for i in range(leftover):
            mapping.append((source_list[new_offset + i], True))

    return mapping[:target_layers]


def validate_layer_layout(index: dict, original_layers: int) -> list[int]:
    """Validate that the source model has contiguous layer indices."""
    actual_layers = sorted(
        {
            li
            for param_name in index["weight_map"]
            if (li := get_layer_index(param_name)) is not None
        }
    )
    if not actual_layers:
        print("ERROR: No layer parameters found.", file=sys.stderr)
        sys.exit(1)

    expected = list(range(original_layers))
    if actual_layers != expected:
        print(
            f"ERROR: Expected layers 0-{original_layers - 1}, got {actual_layers[:8]}...",
            file=sys.stderr,
        )
        sys.exit(1)
    return actual_layers


def main():
    parser = argparse.ArgumentParser(
        description="MoE Depth Expansion (M2): Insert identity-initialized MoE layers"
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to the original model directory")
    parser.add_argument("--output_dir", required=True,
                        help="Path to output the expanded model")
    parser.add_argument("--original_layers", type=int, default=None,
                        help="Number of layers in original model (auto-detected from config if omitted)")
    parser.add_argument("--target_layers", type=int, default=None,
                        help="Target total layers. Defaults to 2x original.")
    parser.add_argument("--copy_source", type=str, default=None,
                        help="Source mapping for new layers (see expand_model_layers.py)")
    parser.add_argument("--insertion_mode", choices=["interleave", "append"],
                        default="interleave",
                        help="How to arrange new layers: interleave after source or append at end")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    config = load_config(model_dir)
    index = load_index(model_dir)

    if not index:
        print("ERROR: Model is not sharded.", file=sys.stderr)
        sys.exit(1)

    original_layers = args.original_layers
    if original_layers is None:
        for key in ["num_layers", "num_hidden_layers", "n_layers"]:
            if key in config:
                original_layers = config[key]
                break
    if original_layers is None:
        print("ERROR: Cannot determine original layer count. Use --original_layers.", file=sys.stderr)
        sys.exit(1)

    target_layers = args.target_layers if args.target_layers else original_layers * 2
    num_new = target_layers - original_layers

    if num_new <= 0:
        print(f"ERROR: target_layers ({target_layers}) must exceed original ({original_layers}).", file=sys.stderr)
        sys.exit(1)

    validate_layer_layout(index, original_layers)
    shard_files = sorted(set(index["weight_map"].values()))

    try:
        source_list = parse_copy_source(args.copy_source, original_layers, num_new)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    layer_mapping = build_layer_mapping(original_layers, target_layers, source_list, args.insertion_mode)

    print(f"Model: {model_dir}")
    print(f"Layers: {original_layers} → {target_layers} (+{num_new} identity layers)")
    print(f"Insertion mode: {args.insertion_mode}")
    print(f"Identity init: zeroing o_proj + down_proj in new layers")

    new_layers_info = [(i, src) for i, (src, is_new) in enumerate(layer_mapping) if is_new]
    if len(new_layers_info) <= 10:
        for tgt, src in new_layers_info:
            print(f"  New layer {tgt} ← copy of layer {src} (identity-initialized)")
    else:
        for tgt, src in new_layers_info[:5]:
            print(f"  New layer {tgt} ← copy of layer {src} (identity-initialized)")
        print(f"  ... ({len(new_layers_info) - 5} more)")

    updated_config = copy.deepcopy(config)
    for key in ["num_layers", "num_hidden_layers", "n_layers"]:
        if key in updated_config:
            updated_config[key] = target_layers

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(updated_config, f, indent=2, ensure_ascii=False)
    print("Config written.")

    orig_to_new: dict[int, list[int]] = defaultdict(list)
    new_layer_set: set[int] = set()
    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if new_idx < original_layers and not is_new:
            continue
        orig_to_new[src].append(new_idx)
        if is_new:
            new_layer_set.add(new_idx)

    target_size_bytes = auto_detect_shard_size(model_dir, shard_files)
    print(f"Target shard size: {target_size_bytes / 1e9:.2f} GB")

    print("\nPass 1/2: Scanning headers...")
    num_output_shards = 1
    current_bytes = 0
    total_output_bytes = 0

    for shard_file in tqdm(shard_files, desc="Scanning"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            continue
        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            nbytes = get_nbytes_from_meta(dtype, shape)
            layer_idx = get_layer_index(key)

            if layer_idx is not None:
                for target_idx in [layer_idx] + orig_to_new.get(layer_idx, []):
                    if target_idx == layer_idx and layer_idx >= original_layers:
                        continue
                    if nbytes + current_bytes > target_size_bytes and current_bytes > 0:
                        num_output_shards += 1
                        current_bytes = 0
                    current_bytes += nbytes
                    total_output_bytes += nbytes
            else:
                if nbytes + current_bytes > target_size_bytes and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += nbytes
                total_output_bytes += nbytes

    print(f"Output: {total_output_bytes / 1e9:.2f} GB across {num_output_shards} shard(s)")

    print("\nPass 2/2: Writing expanded model...")
    new_weight_map: dict[str, str] = {}
    output_shard_idx = 1
    current_tensors: dict[str, torch.Tensor] = {}
    current_bytes = 0
    zeroed_count = 0

    def flush_shard():
        nonlocal output_shard_idx, current_tensors, current_bytes
        if not current_tensors:
            return
        shard_name = f"model-{output_shard_idx:05d}-of-{num_output_shards:05d}.safetensors"
        output_path = output_dir / shard_name
        save_file(current_tensors, str(output_path))
        for t_name in current_tensors:
            new_weight_map[t_name] = shard_name
        output_shard_idx += 1
        current_tensors.clear()
        current_bytes = 0

    remap = {}
    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if not is_new:
            remap[src] = new_idx

    for shard_file in tqdm(shard_files, desc="Processing"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                nbytes = tensor_nbytes(tensor)
                layer_idx = get_layer_index(key)

                if layer_idx is not None:
                    remapped_idx = remap.get(layer_idx, layer_idx)
                    orig_key = set_layer_index(key, remapped_idx)

                    if nbytes + current_bytes > target_size_bytes and current_tensors:
                        flush_shard()
                    current_tensors[orig_key] = tensor
                    current_bytes += nbytes

                    for target_idx in orig_to_new.get(layer_idx, []):
                        dup_key = set_layer_index(key, target_idx)
                        if nbytes + current_bytes > target_size_bytes and current_tensors:
                            flush_shard()

                        if target_idx in new_layer_set and should_zero(dup_key):
                            zeroed_tensor = torch.zeros_like(tensor)
                            current_tensors[dup_key] = zeroed_tensor
                            zeroed_count += 1
                        else:
                            current_tensors[dup_key] = tensor.clone()
                        current_bytes += nbytes
                else:
                    if nbytes + current_bytes > target_size_bytes and current_tensors:
                        flush_shard()
                    current_tensors[key] = tensor
                    current_bytes += nbytes

    flush_shard()

    actual_shards = output_shard_idx - 1
    if actual_shards != num_output_shards:
        print(f"Adjusting shard count: {num_output_shards} → {actual_shards}")
        for i in range(1, actual_shards + 1):
            old_name = output_dir / f"model-{i:05d}-of-{num_output_shards:05d}.safetensors"
            new_name = output_dir / f"model-{i:05d}-of-{actual_shards:05d}.safetensors"
            if old_name.exists() and old_name != new_name:
                old_name.rename(new_name)
        num_output_shards = actual_shards

    fixed_weight_map = {}
    for pname, sname in new_weight_map.items():
        fixed_weight_map[pname] = re.sub(
            r"-of-\d+\.safetensors",
            f"-of-{num_output_shards:05d}.safetensors",
            sname,
        )

    new_index = {
        "metadata": {"total_size": total_output_bytes},
        "weight_map": fixed_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if fpath.is_file() and fpath.suffix not in skip_suffixes and fpath.name not in skip_names:
            shutil.copy2(fpath, output_dir / fpath.name)

    print(f"\n{'='*60}")
    print("Expansion complete!")
    print(f"  Layers: {original_layers} → {target_layers}")
    print(f"  Identity layers (new): {len(new_layer_set)}")
    print(f"  Zeroed tensors (o_proj + down_proj): {zeroed_count}")
    print(f"  Output shards: {num_output_shards}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
