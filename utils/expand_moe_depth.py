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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from utils.shared import (
    assign_shards_layer_aware,
    auto_detect_shard_size,
    build_layer_mapping,
    get_layer_index,
    get_nbytes_from_meta,
    layer_sort_key,
    load_config,
    load_index,
    parse_copy_source,
    read_safetensors_header,
    set_layer_index,
    should_zero,
    tensor_nbytes,
)


def expand_tensor_meta(
    key: str,
    dtype: str,
    shape: list[int],
    remap: dict[int, int],
    orig_to_new: dict[int, list[int]],
    new_layer_set: set[int],
) -> list[tuple[str, int, str]]:
    """Return [(output_key, output_nbytes, action), ...] for a tensor.

    Mirrors the Pass 2 expansion logic but operates on metadata only.
    action: "keep" | "clone" | "zero"
    """
    nbytes = get_nbytes_from_meta(dtype, shape)
    layer_idx = get_layer_index(key)

    if layer_idx is None:
        return [(key, nbytes, "keep")]

    results: list[tuple[str, int, str]] = []
    remapped_idx = remap.get(layer_idx, layer_idx)
    results.append((set_layer_index(key, remapped_idx), nbytes, "keep"))

    for target_idx in orig_to_new.get(layer_idx, []):
        dup_key = set_layer_index(key, target_idx)
        if target_idx in new_layer_set and should_zero(dup_key):
            results.append((dup_key, nbytes, "zero"))
        else:
            results.append((dup_key, nbytes, "clone"))

    return results


def pre_scan_assignments(
    model_dir: Path,
    shard_files: list[str],
    target_shard_size: int,
    remap: dict[int, int],
    orig_to_new: dict[int, list[int]],
    new_layer_set: set[int],
    max_layers_per_shard: int = 1,
) -> tuple[dict[int, list[tuple[str, str, str, str]]], int, int, int, int]:
    """Scan all shard headers, sort by layer, assign to layer-grouped output shards.

    Returns (assignments, num_output_shards, total_output_bytes,
             total_original, total_duplicated).
    """
    # Step 1 — collect all expanded tensor metadata
    all_items: list[tuple[str, str, str, int, str]] = []

    for shard_file in tqdm(shard_files, desc="Scanning headers"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue
        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            for output_key, output_nbytes, action in expand_tensor_meta(
                key, dtype, shape, remap, orig_to_new, new_layer_set,
            ):
                all_items.append((shard_file, key, output_key, output_nbytes, action))

    # Step 2 — sort by layer (non-layer first, then layer 0, 1, …)
    all_items.sort(key=lambda x: layer_sort_key(x[2]))

    # Step 3 — assign to shards with layer-aware grouping
    assignments, num_output_shards, total_output_bytes = assign_shards_layer_aware(
        all_items, target_shard_size, max_layers_per_shard,
    )

    total_original = sum(
        1 for items in assignments.values() for *_, action in items if action == "keep")
    total_duplicated = sum(
        1 for items in assignments.values() for *_, action in items if action != "keep")

    return assignments, num_output_shards, total_output_bytes, total_original, total_duplicated


def _write_output_shard(args):
    """Worker for ProcessPoolExecutor. Writes a single output shard.

    args: (output_path, assignments, model_dir_str)
      assignments: list of (input_shard, input_key, output_key, action)
    """
    output_path, assignments, model_dir_str = args
    model_dir = Path(model_dir_str)

    by_input: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for sfile, in_key, out_key, action in assignments:
        by_input[sfile].append((in_key, out_key, action))

    tensors: dict[str, torch.Tensor] = {}
    try:
        for sfile, items in by_input.items():
            with safe_open(str(model_dir / sfile), framework="pt", device="cpu") as sf:
                tensor_cache: dict[str, torch.Tensor] = {}
                for in_key, out_key, action in items:
                    if in_key not in tensor_cache:
                        tensor_cache[in_key] = sf.get_tensor(in_key)
                    tensor = tensor_cache[in_key]

                    if action == "keep":
                        tensors[out_key] = tensor
                    elif action == "clone":
                        tensors[out_key] = tensor.clone()
                    elif action == "zero":
                        tensors[out_key] = torch.zeros_like(tensor)
        save_file(tensors, str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to write {output_path.name}: {e}") from e
    return [(name, output_path.name) for name in tensors]


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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for parallel output shard "
                             "writing (default 1 = serial; use 0 for CPU count)")
    parser.add_argument("--max_layers_per_shard", type=int, default=1,
                        help="Maximum number of layers to pack into a single "
                             "output safetensors file (default 1). "
                             "1 = one layer per file, 2-3 for smaller models.")
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

    target_layers = args.target_layers if args.target_layers is not None else original_layers * 2
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

    # ── Build target index mappings ───────────────────────────────────────
    orig_to_new: dict[int, list[int]] = defaultdict(list)
    new_layer_set: set[int] = set()
    remap: dict[int, int] = {}
    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if is_new:
            orig_to_new[src].append(new_idx)
            new_layer_set.add(new_idx)
        else:
            remap[src] = new_idx

    target_size_bytes = auto_detect_shard_size(model_dir, shard_files)
    print(f"Target shard size: {target_size_bytes / 1e9:.2f} GB")

    workers = args.workers if args.workers > 0 else (__import__("os").cpu_count() or 4)
    new_weight_map: dict[str, str] = {}

    # ── Pass 1: scan, sort by layer, assign to output shards ─────────────
    print(f"\nPass 1/2: Scanning headers, sorting by layer, assigning to shards...")
    (assignments_by_shard, num_output_shards, total_output_bytes,
     total_original, total_duplicated) = pre_scan_assignments(
        model_dir, shard_files, target_size_bytes,
        remap, orig_to_new, new_layer_set,
        max_layers_per_shard=args.max_layers_per_shard,
    )
    zeroed_count = sum(
        1 for items in assignments_by_shard.values()
        for _, _, out_key, action in items if action == "zero"
    )
    print(f"Output: {total_original:,} original + {total_duplicated:,} expanded "
          f"= {total_original + total_duplicated:,} tensors "
          f"({total_output_bytes / 1e9:.2f} GB in {num_output_shards} shards)")
    print(f"Layer grouping: ≤ {args.max_layers_per_shard} layer(s) per shard")

    # ── Pass 2: write output shards ──────────────────────────────────────
    print("Pass 2/2: Writing output shards...")
    tasks = []
    for shard_idx in sorted(assignments_by_shard):
        shard_name = f"model-{shard_idx + 1:05d}-of-{num_output_shards:05d}.safetensors"
        output_path = output_dir / shard_name
        tasks.append((
            output_path,
            assignments_by_shard[shard_idx],
            str(model_dir),
        ))

    if workers > 1:
        print(f"  Parallel: {workers} workers, {len(tasks)} shards")
        chunksize = max(1, len(tasks) // workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = list(tqdm(
                executor.map(_write_output_shard, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Writing shards",
            ))
            for weight_entries in futures:
                for name, shard_name in weight_entries:
                    new_weight_map[name] = shard_name
    else:
        print(f"  Serial: {len(tasks)} shards")
        for task in tqdm(tasks, desc="Writing shards"):
            for name, shard_name in _write_output_shard(task):
                new_weight_map[name] = shard_name

    # ── Fixup shard names in weight map ─────────────────────────────────
    fixed_weight_map = {}
    for pname, sname in new_weight_map.items():
        fixed_weight_map[pname] = re.sub(
            r"-of-\d+\.safetensors",
            f"-of-{num_output_shards:05d}.safetensors",
            sname,
        )

    metadata = {**index.get("metadata", {})}
    metadata["total_size"] = total_output_bytes
    new_index = {
        "metadata": metadata,
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
