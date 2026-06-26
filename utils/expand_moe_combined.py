#!/usr/bin/env python3
"""
Combined MoE Expansion: Depth (M2) + Expert Upcycling (M1).

Performs both expansions in a single pass:
  1. Depth expansion (M2): inserts identity-initialized MoE layers
  2. Expert expansion (M1): duplicates experts, expands routers

The two operations compose naturally:
  - Each original layer maps to 1+ target layers (remapped + identity copies)
  - Each expert weight maps to 2+ experts (original + copies) per target layer
  - Router weights expand from [E] to [2E] per target layer

Usage:
  python expand_moe_combined.py \
      --model_dir ./original_model \
      --output_dir ./expanded_model \
      [--target_layers 32] \
      [--target_experts 512] \
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
    EXPERT_COUNT_KEYS,
    auto_detect_shard_size,
    build_expert_target_map,
    build_layer_mapping,
    expand_router_bias,
    expand_router_weight,
    find_expert_count,
    get_expert_info,
    get_layer_index,
    get_nbytes_from_meta,
    is_router_bias,
    is_router_weight,
    load_config,
    load_index,
    make_expert_key,
    parse_copy_source,
    read_safetensors_header,
    set_layer_index,
    should_zero,
    tensor_nbytes,
)



# ═══════════════════════════════════════════════════════════════════════════════
# Combined tensor expansion
# ═══════════════════════════════════════════════════════════════════════════════

def expand_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    remap: dict[int, int],
    orig_to_new: dict[int, list[int]],
    new_layer_set: set[int],
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    expert_targets: dict[int, list[int]],
    target_experts: int,
    router_noise_scale: float = 0.0,
    expert_noise_scale: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Expand a single tensor for both depth and width.

    Returns {output_key: tensor}.  May return multiple keys when:
      - A source layer has identity-layer copies (depth expansion)
      - An expert weight is duplicated (expert expansion)
      - Both (Cartesian product of the two)
    """
    layer_idx = get_layer_index(key)

    # Non-layer params: copy as-is
    if layer_idx is None:
        return {key: tensor}

    remapped = remap[layer_idx]
    target_layers = [remapped] + orig_to_new.get(layer_idx, [])

    results: dict[str, torch.Tensor] = {}

    # Pre-expand router tensors once per source layer
    router_expanded = None
    if is_router_weight(key):
        router_expanded = expand_router_weight(
            tensor, original_experts, zero_expert_num,
            expansion_factor, router_noise_scale,
        )
    elif is_router_bias(key):
        router_expanded = expand_router_bias(
            tensor, original_experts, zero_expert_num, expansion_factor,
        )

    if router_expanded is not None:
        for i, tgt_layer in enumerate(target_layers):
            tgt_key = set_layer_index(key, tgt_layer)
            results[tgt_key] = router_expanded if i == 0 else router_expanded.clone()
        return results

    # Expert weights: need both layer and expert expansion
    expert_info = get_expert_info(key)
    if expert_info is not None:
        _, expert_idx, rest = expert_info
        if expert_idx < original_experts:
            # Routed expert: keep original + duplicate copies
            for tgt_layer in target_layers:
                is_new = tgt_layer in new_layer_set

                orig_key = make_expert_key(tgt_layer, expert_idx, rest)
                if is_new and should_zero(orig_key):
                    results[orig_key] = torch.zeros_like(tensor)
                else:
                    results[orig_key] = tensor.clone()

                for new_expert_idx in expert_targets.get(expert_idx, []):
                    dup_key = make_expert_key(tgt_layer, new_expert_idx, rest)
                    if is_new and should_zero(dup_key):
                        results[dup_key] = torch.zeros_like(tensor)
                    elif expert_noise_scale > 0:
                        noise = torch.randn_like(tensor) * expert_noise_scale * tensor.std()
                        results[dup_key] = tensor + noise
                    else:
                        results[dup_key] = tensor.clone()
        else:
            # Zero expert: remap index and duplicate copies
            zero_offset = expert_idx - original_experts
            base_new_idx = target_experts + zero_offset
            for tgt_layer in target_layers:
                is_new = tgt_layer in new_layer_set
                base_key = make_expert_key(tgt_layer, base_new_idx, rest)
                if is_new and should_zero(base_key):
                    results[base_key] = torch.zeros_like(tensor)
                else:
                    results[base_key] = tensor.clone()

                for f in range(1, expansion_factor):
                    copy_idx = target_experts + zero_offset + f * zero_expert_num
                    copy_key = make_expert_key(tgt_layer, copy_idx, rest)
                    if is_new and should_zero(copy_key):
                        results[copy_key] = torch.zeros_like(tensor)
                    else:
                        results[copy_key] = tensor.clone()
        return results

    # Regular layer params (attention, norms, etc.)
    for tgt_layer in target_layers:
        tgt_key = set_layer_index(key, tgt_layer)
        is_new = tgt_layer in new_layer_set
        if is_new and should_zero(tgt_key):
            results[tgt_key] = torch.zeros_like(tensor)
        else:
            results[tgt_key] = tensor.clone()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata-level expansion (for Pass 1 layout planning)
# ═══════════════════════════════════════════════════════════════════════════════

def expand_tensor_meta(
    key: str,
    dtype: str,
    shape: list[int],
    *,
    remap: dict[int, int],
    orig_to_new: dict[int, list[int]],
    new_layer_set: set[int],
    zero_expert_num: int,
    expansion_factor: int,
    expert_targets: dict[int, list[int]],
    target_experts: int,
    expert_noise_scale: float = 0.0,
) -> list[tuple[str, int, str]]:
    """Return [(output_key, output_nbytes, action), ...] for a tensor.

    Mirrors expand_tensor() but operates on metadata only.
    action: "keep" | "clone" | "clone_expert" | "zero" | "router_weight" | "router_bias"
    """
    results: list[tuple[str, int, str]] = []
    nbytes = get_nbytes_from_meta(dtype, shape)
    original_experts = target_experts // expansion_factor

    layer_idx = get_layer_index(key)
    if layer_idx is None:
        results.append((key, nbytes, "keep"))
        return results

    remapped = remap[layer_idx]
    target_layers = [remapped] + orig_to_new.get(layer_idx, [])

    if is_router_weight(key):
        new_dim0 = target_experts + zero_expert_num * expansion_factor
        new_shape = [new_dim0] + list(shape[1:])
        new_nbytes = get_nbytes_from_meta(dtype, new_shape)
        for tgt_layer in target_layers:
            results.append((set_layer_index(key, tgt_layer), new_nbytes, "router_weight"))
        return results

    if is_router_bias(key):
        new_dim0 = target_experts + zero_expert_num * expansion_factor
        new_shape = [new_dim0] + list(shape[1:])
        new_nbytes = get_nbytes_from_meta(dtype, new_shape)
        for tgt_layer in target_layers:
            results.append((set_layer_index(key, tgt_layer), new_nbytes, "router_bias"))
        return results

    expert_info = get_expert_info(key)
    if expert_info is not None:
        _, expert_idx, rest = expert_info
        if expert_idx < original_experts:
            # Routed expert
            for tgt_layer in target_layers:
                is_new = tgt_layer in new_layer_set
                orig_key = make_expert_key(tgt_layer, expert_idx, rest)
                if is_new and should_zero(orig_key):
                    results.append((orig_key, nbytes, "zero"))
                else:
                    tag = "clone" if tgt_layer != remapped else "keep"
                    results.append((orig_key, nbytes, tag))

                for _new_expert_idx in expert_targets.get(expert_idx, []):
                    dup_key = make_expert_key(tgt_layer, _new_expert_idx, rest)
                    if is_new and should_zero(dup_key):
                        results.append((dup_key, nbytes, "zero"))
                    else:
                        tag = "clone_expert" if expert_noise_scale > 0 else "clone"
                        results.append((dup_key, nbytes, tag))
        else:
            # Zero expert
            zero_offset = expert_idx - original_experts
            base_new_idx = target_experts + zero_offset
            for tgt_layer in target_layers:
                is_new = tgt_layer in new_layer_set
                base_key = make_expert_key(tgt_layer, base_new_idx, rest)
                if is_new and should_zero(base_key):
                    results.append((base_key, nbytes, "zero"))
                else:
                    tag = "clone" if tgt_layer != remapped else "keep"
                    results.append((base_key, nbytes, tag))
                for f in range(1, expansion_factor):
                    copy_idx = target_experts + zero_offset + f * zero_expert_num
                    copy_key = make_expert_key(tgt_layer, copy_idx, rest)
                    if is_new and should_zero(copy_key):
                        results.append((copy_key, nbytes, "zero"))
                    else:
                        results.append((copy_key, nbytes, "clone"))
        return results

    # Regular layer params
    for tgt_layer in target_layers:
        tgt_key = set_layer_index(key, tgt_layer)
        is_new = tgt_layer in new_layer_set
        if is_new and should_zero(tgt_key):
            results.append((tgt_key, nbytes, "zero"))
        else:
            tag = "clone" if tgt_layer != remapped else "keep"
            results.append((tgt_key, nbytes, tag))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-scan + assignment (for parallel path)
# ═══════════════════════════════════════════════════════════════════════════════

def pre_scan_assignments(
    model_dir: Path,
    shard_files: list[str],
    target_shard_size: int,
    *,
    remap: dict[int, int],
    orig_to_new: dict[int, list[int]],
    new_layer_set: set[int],
    zero_expert_num: int,
    expansion_factor: int,
    expert_targets: dict[int, list[int]],
    target_experts: int,
    expert_noise_scale: float = 0.0,
) -> tuple[dict[int, list[tuple[str, str, str, str]]], int, int, int, int]:
    """Scan all shard headers and assign each output tensor to an output shard.

    Returns (assignments, num_output_shards, total_output_bytes,
             total_original, total_duplicated).
    """
    current_shard = 0
    current_bytes = 0
    total_output_bytes = 0
    total_original = 0
    total_duplicated = 0
    assignments: dict[int, list[tuple[str, str, str, str]]] = defaultdict(list)

    for shard_file in tqdm(shard_files, desc="Pre-scanning"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue
        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            for output_key, output_nbytes, action in expand_tensor_meta(
                key, dtype, shape,
                remap=remap,
                orig_to_new=orig_to_new,
                new_layer_set=new_layer_set,
                zero_expert_num=zero_expert_num,
                expansion_factor=expansion_factor,
                expert_targets=expert_targets,
                target_experts=target_experts,
                expert_noise_scale=expert_noise_scale,
            ):
                if current_bytes + output_nbytes > target_shard_size and current_bytes > 0:
                    current_shard += 1
                    current_bytes = 0
                assignments[current_shard].append(
                    (shard_file, key, output_key, action))
                current_bytes += output_nbytes
                total_output_bytes += output_nbytes
                if action in ("clone", "clone_expert", "router_weight", "router_bias", "zero"):
                    total_duplicated += 1
                else:
                    total_original += 1

    num_output_shards = current_shard + 1 if assignments else 0
    return dict(assignments), num_output_shards, total_output_bytes, total_original, total_duplicated


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel output shard writer
# ═══════════════════════════════════════════════════════════════════════════════

def _write_output_shard(args):
    (output_path, assignments, model_dir_str,
     original_experts, zero_expert_num, expansion_factor,
     router_noise_scale, expert_noise_scale) = args

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
                    elif action == "clone_expert":
                        if expert_noise_scale > 0:
                            noise = torch.randn_like(tensor) * expert_noise_scale * tensor.std()
                            tensors[out_key] = tensor + noise
                        else:
                            tensors[out_key] = tensor.clone()
                    elif action == "zero":
                        tensors[out_key] = torch.zeros_like(tensor)
                    elif action == "router_weight":
                        tensors[out_key] = expand_router_weight(
                            tensor, original_experts, zero_expert_num,
                            expansion_factor, router_noise_scale,
                        )
                    elif action == "router_bias":
                        tensors[out_key] = expand_router_bias(
                            tensor, original_experts, zero_expert_num,
                            expansion_factor,
                        )
        save_file(tensors, str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to write {output_path.name}: {e}") from e
    return [(name, output_path.name) for name in tensors]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_model(index: dict, original_layers: int) -> list[int]:
    actual_layers = sorted({
        li for param_name in index["weight_map"]
        if (li := get_layer_index(param_name)) is not None
    })
    if not actual_layers:
        print("ERROR: No layer parameters found.", file=sys.stderr)
        sys.exit(1)
    expected = list(range(original_layers))
    if actual_layers != expected:
        print(f"ERROR: Expected layers 0-{original_layers - 1}, got {actual_layers[:8]}...",
              file=sys.stderr)
        sys.exit(1)
    return actual_layers


def validate_expert_layout(index: dict, original_experts: int, zero_expert_num: int) -> dict[int, list[int]]:
    """Validate that each MoE layer has contiguous expert indices.

    Expected: [0, original_experts) or [0, original_experts + zero_expert_num).
    """
    experts_by_layer: dict[int, set[int]] = defaultdict(set)
    for param_name in index["weight_map"]:
        info = get_expert_info(param_name)
        if info is None:
            continue
        layer_idx, expert_idx, _ = info
        experts_by_layer[layer_idx].add(expert_idx)

    if not experts_by_layer:
        print(
            "ERROR: No expert parameters matching 'model.layers.<idx>.mlp.experts.<idx>.' "
            "were found in the index.",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_routed = list(range(original_experts))
    expected_total = list(range(original_experts + zero_expert_num))

    validated: dict[int, list[int]] = {}
    for layer_idx, expert_indices in sorted(experts_by_layer.items()):
        actual = sorted(expert_indices)
        if actual != expected_routed and actual != expected_total:
            print(
                f"ERROR: Layer {layer_idx} has expert indices {actual[:8]}"
                f"{'...' if len(actual) > 8 else ''}, but expected contiguous "
                f"indices 0-{original_experts - 1} or "
                f"0-{original_experts + zero_expert_num - 1}.",
                file=sys.stderr,
            )
            sys.exit(1)
        validated[layer_idx] = actual
    return validated


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined MoE Expansion: Depth (M2) + Expert Upcycling (M1)"
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to the original model directory")
    parser.add_argument("--output_dir", required=True,
                        help="Path to output the expanded model")
    parser.add_argument("--target_layers", type=int, default=None,
                        help="Target number of layers (default: original + 4)")
    parser.add_argument("--target_experts", type=int, default=None,
                        help="Target number of experts (default: 2x original)")
    parser.add_argument("--copy_source", type=str, default=None,
                        help="Source mapping for new layers (seq = round-robin)")
    parser.add_argument("--insertion_mode", choices=["interleave", "append"],
                        default="interleave",
                        help="How to arrange new layers")
    parser.add_argument("--router-noise-scale", type=float, default=0.0,
                        help="Gaussian noise scale for duplicated router weights")
    parser.add_argument("--expert-noise-scale", type=float, default=0.0,
                        help="Gaussian noise scale for duplicated expert weights")
    parser.add_argument("--target_topk", type=int, default=None,
                        help="Target moe_topk. Defaults to unchanged.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (0 = CPU count)")
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

    # ── Determine dimensions ─────────────────────────────────────────────
    original_layers = None
    for key in ["num_layers", "num_hidden_layers", "n_layers"]:
        if key in config:
            original_layers = config[key]
            break
    if original_layers is None:
        print("ERROR: Cannot determine layer count from config.", file=sys.stderr)
        sys.exit(1)

    _, original_experts, zero_expert_num = find_expert_count(config)
    if original_experts == 0:
        print("ERROR: No expert count found in config.", file=sys.stderr)
        sys.exit(1)

    target_layers = args.target_layers or original_layers + 4
    target_experts = args.target_experts or original_experts * 2

    if target_layers <= original_layers:
        print(f"ERROR: target_layers ({target_layers}) must exceed original ({original_layers}).",
              file=sys.stderr)
        sys.exit(1)
    if target_experts <= original_experts:
        print(f"ERROR: target_experts ({target_experts}) must exceed original ({original_experts}).",
              file=sys.stderr)
        sys.exit(1)
    if target_experts % original_experts != 0:
        print(f"ERROR: target_experts ({target_experts}) must be a multiple of "
              f"original ({original_experts}).", file=sys.stderr)
        sys.exit(1)

    expansion_factor = target_experts // original_experts
    num_new_layers = target_layers - original_layers
    target_zero_expert_num = zero_expert_num * expansion_factor

    # ── Build mappings ───────────────────────────────────────────────────
    validate_model(index, original_layers)
    experts_by_layer = validate_expert_layout(index, original_experts, zero_expert_num)
    shard_files = sorted(set(index["weight_map"].values()))

    try:
        source_list = parse_copy_source(args.copy_source, original_layers, num_new_layers)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    layer_mapping = build_layer_mapping(
        original_layers, target_layers, source_list, args.insertion_mode,
    )
    expert_targets = build_expert_target_map(original_experts, target_experts)

    # remap: original_layer → new_layer_index (for non-new layers)
    remap: dict[int, int] = {}
    orig_to_new: dict[int, list[int]] = defaultdict(list)
    new_layer_set: set[int] = set()
    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if is_new:
            orig_to_new[src].append(new_idx)
            new_layer_set.add(new_idx)
        else:
            remap[src] = new_idx

    # ── Print plan ───────────────────────────────────────────────────────
    print(f"Model:      {model_dir}")
    print(f"Layers:     {original_layers}  →  {target_layers}  (+{num_new_layers} identity)")
    print(f"Experts:    {original_experts}  →  {target_experts}  (×{expansion_factor})")
    if zero_expert_num > 0:
        print(f"Zero exp:   {zero_expert_num}  →  {target_zero_expert_num}")
    print(f"Insertion:  {args.insertion_mode}")
    if args.router_noise_scale > 0:
        print(f"Router noise: {args.router_noise_scale}")
    if args.expert_noise_scale > 0:
        print(f"Expert noise: {args.expert_noise_scale}")

    new_layers_info = [(i, src) for i, (src, is_new) in enumerate(layer_mapping) if is_new]
    if len(new_layers_info) <= 12:
        for tgt, src in new_layers_info:
            print(f"  New layer {tgt}  ←  copy of layer {src}  (identity-initialized)")
    else:
        for tgt, src in new_layers_info[:6]:
            print(f"  New layer {tgt}  ←  copy of layer {src}  (identity-initialized)")
        print(f"  ... ({len(new_layers_info) - 6} more)")
    print(f"MoE layers: {len(experts_by_layer)} with {original_experts} experts each")

    # ── Update config ────────────────────────────────────────────────────
    new_config = copy.deepcopy(config)
    for key in ["num_layers", "num_hidden_layers", "n_layers"]:
        if key in new_config:
            new_config[key] = target_layers
    for key in EXPERT_COUNT_KEYS:
        if key in new_config:
            new_config[key] = target_experts
    if zero_expert_num > 0:
        new_config["zero_expert_num"] = target_zero_expert_num
    if args.target_topk is not None:
        topk_keys = ["moe_topk", "num_experts_per_tok", "top_k"]
        topk_set = False
        for key in topk_keys:
            if key in new_config:
                new_config[key] = args.target_topk
                topk_set = True
        if not topk_set:
            new_config["moe_topk"] = args.target_topk

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    print("Config written.")

    target_shard_size = auto_detect_shard_size(model_dir, shard_files)
    print(f"Target shard size: {target_shard_size / 1e9:.2f} GB")

    workers = args.workers if args.workers > 0 else (__import__("os").cpu_count() or 4)

    new_weight_map: dict[str, str] = {}
    zeroed_count = 0

    # ── Process ──────────────────────────────────────────────────────────
    if workers > 1:
        print(f"\nParallel mode: {workers} workers")
        print("Pass 1/2: Scanning headers and assigning to output shards...")
        (assignments_by_shard, num_output_shards, total_output_bytes,
         total_original, total_duplicated) = pre_scan_assignments(
            model_dir, shard_files, target_shard_size,
            remap=remap, orig_to_new=orig_to_new,
            new_layer_set=new_layer_set,
            zero_expert_num=zero_expert_num,
            expansion_factor=expansion_factor,
            expert_targets=expert_targets,
            target_experts=target_experts,
            expert_noise_scale=args.expert_noise_scale,
        )
        zeroed_count = sum(
            1 for items in assignments_by_shard.values()
            for _, _, _, action in items if action == "zero"
        )
        print(f"Output: {total_original:,} original + {total_duplicated:,} expanded "
              f"= {total_original + total_duplicated:,} tensors "
              f"({total_output_bytes / 1e9:.2f} GB in {num_output_shards} shards)")

        print("Pass 2/2: Writing output shards...")
        tasks = []
        for shard_idx in sorted(assignments_by_shard):
            shard_name = f"model-{shard_idx + 1:05d}-of-{num_output_shards:05d}.safetensors"
            output_path = output_dir / shard_name
            tasks.append((
                output_path,
                assignments_by_shard[shard_idx],
                str(model_dir),
                original_experts,
                zero_expert_num,
                expansion_factor,
                args.router_noise_scale,
                args.expert_noise_scale,
            ))

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
                for _, output_nbytes, _action in expand_tensor_meta(
                    key, dtype, shape,
                    remap=remap, orig_to_new=orig_to_new,
                    new_layer_set=new_layer_set,
                    zero_expert_num=zero_expert_num,
                    expansion_factor=expansion_factor,
                    expert_targets=expert_targets,
                    target_experts=target_experts,
                    expert_noise_scale=args.expert_noise_scale,
                ):
                    if output_nbytes + current_bytes > target_shard_size and current_bytes > 0:
                        num_output_shards += 1
                        current_bytes = 0
                    current_bytes += output_nbytes
                    total_output_bytes += output_nbytes

        print(f"Output: {total_output_bytes / 1e9:.2f} GB across {num_output_shards} shard(s)")

        print("\nPass 2/2: Writing expanded model...")
        output_shard_idx = 1
        current_tensors: dict[str, torch.Tensor] = {}
        current_bytes = 0

        def flush_shard():
            nonlocal output_shard_idx, current_tensors, current_bytes
            if not current_tensors:
                return
            shard_name = f"model-{output_shard_idx:05d}-of-{num_output_shards:05d}.safetensors"
            save_file(current_tensors, str(output_dir / shard_name))
            for t_name in current_tensors:
                new_weight_map[t_name] = shard_name
            output_shard_idx += 1
            current_tensors.clear()
            current_bytes = 0

        def maybe_flush(nbytes: int):
            if current_bytes + nbytes > target_shard_size and current_tensors:
                flush_shard()

        for shard_file in tqdm(shard_files, desc="Processing"):
            with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    tensor = sf.get_tensor(key)
                    for out_key, expanded in expand_tensor(
                        key, tensor,
                        remap=remap, orig_to_new=orig_to_new,
                        new_layer_set=new_layer_set,
                        original_experts=original_experts,
                        zero_expert_num=zero_expert_num,
                        expansion_factor=expansion_factor,
                        expert_targets=expert_targets,
                        target_experts=target_experts,
                        router_noise_scale=args.router_noise_scale,
                        expert_noise_scale=args.expert_noise_scale,
                    ).items():
                        nbytes = tensor_nbytes(expanded)
                        maybe_flush(nbytes)
                        current_tensors[out_key] = expanded
                        current_bytes += nbytes
                        out_layer = get_layer_index(out_key)
                        if out_layer in new_layer_set and should_zero(out_key):
                            zeroed_count += 1

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

    # ── Copy auxiliary files ────────────────────────────────────────────
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if fpath.is_file() and fpath.suffix not in skip_suffixes and fpath.name not in skip_names:
            shutil.copy2(fpath, output_dir / fpath.name)

    print(f"\n{'='*60}")
    print("Combined expansion complete!")
    print(f"  Layers:       {original_layers}  →  {target_layers}  (+{num_new_layers})")
    print(f"  Experts:      {original_experts}  →  {target_experts}  (×{expansion_factor})")
    print(f"  Zeroed params:{zeroed_count}")
    print(f"  Output size:  {total_output_bytes / 1e9:.2f} GB")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
