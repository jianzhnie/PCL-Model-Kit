#!/usr/bin/env python3
"""
Expand the number of MoE experts in a sharded safetensors model.

For a model with N experts, creates a kN-expert version by:
1. Duplicating each expert weight (with optional noise to break symmetry)
2. Expanding the router classifier weights (with optional noise on copies)
3. Expanding the score correction bias (exact copies)

Processes shards in two passes:
  Pass 1 — scan headers to plan output shard layout
  Pass 2 — load, expand, and write tensors

Usage:
  python expand_moe_experts.py \
      --model_dir ./original_model \
      --output_dir ./expanded_model \
      [--target_experts 1024] \
      [--target_topk 24] \
      [--router-noise-scale 1e-6]
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
    assign_shards_layer_aware,
    auto_detect_shard_size,
    build_expert_target_map,
    expand_router_bias,
    expand_router_weight,
    find_expert_count,
    get_expert_info,
    get_nbytes_from_meta,
    is_router_bias,
    is_router_weight,
    layer_sort_key,
    load_config,
    load_index,
    make_expert_key,
    read_safetensors_header,
    tensor_nbytes,
)

TOPK_KEYS = ["moe_topk", "num_experts_per_tok", "top_k"]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

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
                f"indices 0-{original_experts - 1} or 0-{original_experts + zero_expert_num - 1}.",
                file=sys.stderr,
            )
            sys.exit(1)
        validated[layer_idx] = actual
    return validated


def validate_router_shape(param_name: str, shape: list[int], total_routed: int) -> None:
    """Ensure router tensors have the expected first dimension."""
    if not shape:
        print(f"ERROR: Router tensor {param_name} has an empty shape.", file=sys.stderr)
        sys.exit(1)
    if shape[0] != total_routed:
        print(
            f"ERROR: Router tensor {param_name} has shape {shape}, expected first "
            f"dimension {total_routed} (n_routed_experts + zero_expert_num).",
            file=sys.stderr,
        )
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Config transformation
# ═══════════════════════════════════════════════════════════════════════════════

def expand_config(
    original_config: dict,
    target_experts: int,
    target_zero_expert_num: int,
    target_topk: int | None = None,
) -> dict:
    """Generate expanded config with deepcopy to avoid mutating the original."""
    new_config = copy.deepcopy(original_config)

    for key in EXPERT_COUNT_KEYS:
        if key in new_config:
            new_config[key] = target_experts

    if target_zero_expert_num > 0:
        new_config["zero_expert_num"] = target_zero_expert_num

    if target_topk is not None:
        topk_set = False
        for key in TOPK_KEYS:
            if key in new_config:
                new_config[key] = target_topk
                topk_set = True
        if not topk_set:
            new_config["moe_topk"] = target_topk

    return new_config


def describe_config_diff(old: dict, new: dict):
    """Print config key changes."""
    print("\n" + "=" * 60)
    print("Config 变更对照")
    print("=" * 60)
    keys = set(list(old.keys()) + list(new.keys()))
    for k in sorted(keys):
        ov = old.get(k)
        nv = new.get(k)
        if ov != nv:
            print(f"  {k}:  {ov}  →  {nv}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor expansion
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_tensor(
    key: str,
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    source_to_targets: dict[int, list[int]],
    target_experts: int,
    router_noise_scale: float = 0.0,
    expert_noise_scale: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Expand a single tensor, returning {output_key: expanded_tensor}."""
    total_routed = original_experts + zero_expert_num

    if is_router_weight(key):
        validate_router_shape(key, list(tensor.shape), total_routed)
        return {key: expand_router_weight(
            tensor, original_experts, zero_expert_num, expansion_factor, router_noise_scale,
        )}
    elif is_router_bias(key):
        validate_router_shape(key, list(tensor.shape), total_routed)
        return {key: expand_router_bias(
            tensor, original_experts, zero_expert_num, expansion_factor,
        )}
    elif info := get_expert_info(key):
        layer_idx, expert_idx, rest = info
        result: dict[str, torch.Tensor] = {}
        if expert_idx < original_experts:
            result[key] = tensor  # reference: copies below are independent tensors
            for new_expert_idx in source_to_targets.get(expert_idx, []):
                new_key = make_expert_key(layer_idx, new_expert_idx, rest)
                if expert_noise_scale > 0:
                    noise = torch.randn_like(tensor) * expert_noise_scale * tensor.std()
                    result[new_key] = tensor + noise
                else:
                    result[new_key] = tensor.clone()
        else:
            base_new_idx = expert_idx - original_experts + target_experts
            new_key = make_expert_key(layer_idx, base_new_idx, rest)
            result[new_key] = tensor.clone()
            zero_offset = expert_idx - original_experts
            for f in range(1, expansion_factor):
                copy_idx = target_experts + zero_offset + f * zero_expert_num
                copy_key = make_expert_key(layer_idx, copy_idx, rest)
                result[copy_key] = tensor.clone()
        return result
    else:
        return {key: tensor}


# ═══════════════════════════════════════════════════════════════════════════════
# Output layout planning (Pass 1)
# ═══════════════════════════════════════════════════════════════════════════════

def plan_output_layout(
    model_dir: Path,
    shard_files: list[str],
    target_shard_size: int,
    original_experts: int,
    zero_expert_num: int,
    target_experts: int,
    expansion_factor: int,
    source_to_targets: dict[int, list[int]],
    expert_noise_scale: float = 0.0,
) -> tuple[int, int, int, int]:
    """Scan all shard headers to determine the number of output shards needed.

    Returns (num_output_shards, total_output_bytes, total_original, total_duplicated).
    """
    total_output_bytes = 0
    num_output_shards = 1
    current_bytes = 0
    total_original = 0
    total_duplicated = 0
    total_routed = original_experts + zero_expert_num

    for shard_file in tqdm(shard_files, desc="Scanning"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue

        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            for _, output_nbytes, action in _expand_tensor_meta(
                key, dtype, shape, original_experts, zero_expert_num,
                total_routed, expansion_factor, source_to_targets, target_experts,
                expert_noise_scale,
            ):
                if output_nbytes + current_bytes > target_shard_size and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += output_nbytes
                total_output_bytes += output_nbytes
                if action in ("clone", "clone_expert", "clone_exact"):
                    total_duplicated += 1
                else:
                    total_original += 1

    return num_output_shards, total_output_bytes, total_original, total_duplicated


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Pass 2: pre-scan + multi-process output shard writing
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_tensor_meta(key: str, dtype: str, shape: list[int],
                        original_experts: int, zero_expert_num: int,
                        total_routed: int, expansion_factor: int,
                        source_to_targets: dict[int, list[int]],
                        target_experts: int,
                        expert_noise_scale: float = 0.0) -> list[tuple[str, int, str]]:
    """Return list of (output_key, output_nbytes, action) for an input tensor.

    action is one of: "keep", "clone", "clone_expert", "clone_exact",
                       "router_weight", "router_bias"
    - "clone": non-expert copy (always exact, no noise)
    - "clone_expert": routed expert copy (may receive expert noise)
    - "clone_exact": zero expert copy (always exact, no noise)
    Mirrors the actual expansion logic but operates on metadata only.
    """
    results: list[tuple[str, int, str]] = []
    nbytes = get_nbytes_from_meta(dtype, shape)

    if is_router_weight(key):
        validate_router_shape(key, shape, total_routed)
        new_dim0 = target_experts + zero_expert_num * expansion_factor
        new_shape = [new_dim0] + list(shape[1:])
        new_nbytes = get_nbytes_from_meta(dtype, new_shape)
        results.append((key, new_nbytes, "router_weight"))
    elif is_router_bias(key):
        validate_router_shape(key, shape, total_routed)
        new_dim0 = target_experts + zero_expert_num * expansion_factor
        new_shape = [new_dim0] + list(shape[1:])
        new_nbytes = get_nbytes_from_meta(dtype, new_shape)
        results.append((key, new_nbytes, "router_bias"))
    elif info := get_expert_info(key):
        layer_idx, expert_idx, rest = info
        if expert_idx < original_experts:
            results.append((key, nbytes, "keep"))
            for new_expert_idx in source_to_targets.get(expert_idx, []):
                new_key = make_expert_key(layer_idx, new_expert_idx, rest)
                tag = "clone_expert" if expert_noise_scale > 0 else "clone"
                results.append((new_key, nbytes, tag))
        else:
            base_new_idx = expert_idx - original_experts + target_experts
            new_key = make_expert_key(layer_idx, base_new_idx, rest)
            results.append((new_key, nbytes, "keep"))
            zero_offset = expert_idx - original_experts
            for f in range(1, expansion_factor):
                copy_idx = target_experts + zero_offset + f * zero_expert_num
                copy_key = make_expert_key(layer_idx, copy_idx, rest)
                results.append((copy_key, nbytes, "clone_exact"))
    else:
        results.append((key, nbytes, "keep"))

    return results


def _pre_scan_assignments(
    model_dir: Path,
    shard_files: list[str],
    target_shard_size: int,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    source_to_targets: dict[int, list[int]],
    target_experts: int,
    expert_noise_scale: float = 0.0,
    max_layers_per_shard: int = 1,
) -> tuple[dict[int, list[tuple[str, str, str, str]]], int, int, int, int]:
    """Scan all shard headers, sort by layer, assign to layer-grouped output shards.

    Returns (shard_assignments, num_output_shards, total_output_bytes,
             total_original, total_duplicated).
    """
    total_routed = original_experts + zero_expert_num

    # Step 1 — collect all expanded tensor metadata
    all_items: list[tuple[str, str, str, int, str]] = []

    for shard_file in tqdm(shard_files, desc="Scanning headers"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue
        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            for output_key, output_nbytes, action in _expand_tensor_meta(
                key, dtype, shape, original_experts, zero_expert_num,
                total_routed, expansion_factor, source_to_targets, target_experts,
                expert_noise_scale,
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

    return dict(assignments), num_output_shards, total_output_bytes, total_original, total_duplicated


def _write_output_shard(args):
    """Module-level worker for ProcessPoolExecutor. Writes a single output shard.

    Args is a tuple of:
      (output_path, assignments, model_dir_str, original_experts, zero_expert_num,
       expansion_factor, router_noise_scale, expert_noise_scale)

    assignments: list of (input_shard, input_key, output_key, action)
    """
    (output_path, assignments, model_dir_str, original_experts, zero_expert_num,
     expansion_factor, router_noise_scale, expert_noise_scale) = args

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
                    elif action == "clone_exact":
                        tensors[out_key] = tensor.clone()
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
        raise RuntimeError(
            f"Failed to write output shard {output_path.name}: {e}"
        ) from e
    return [(name, output_path.name) for name in tensors]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Expand MoE experts by duplicating weights and expanding routers"
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to the original model directory")
    parser.add_argument("--output_dir", required=True,
                        help="Path to output the expanded model")
    parser.add_argument("--target_experts", type=int, default=None,
                        help="Target number of experts. Defaults to double the original.")
    parser.add_argument("--target_topk", type=int, default=None,
                        help="Target moe_topk. Defaults to unchanged.")
    parser.add_argument("--use_group_routing", action="store_true", default=False,
                        help="Enable grouped expert routing (方案一). "
                             "Keeps moe_topk unchanged and adds use_group_routing + "
                             "expert_expansion_factor to config. "
                             "Mutually exclusive with --target_topk.")
    parser.add_argument("--router-noise-scale", type=float, default=0.0,
                        help="Gaussian noise scale for duplicated router weights "
                             "(default 0.0 = exact copies; recommend 1e-6 to break symmetry)")
    parser.add_argument("--expert-noise-scale", type=float, default=0.0,
                        help="Gaussian noise scale for duplicated expert weights "
                             "(default 0.0 = exact copies; recommend 0.01 for aggressive "
                             "symmetry breaking per arXiv:2604.19835)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for parallel output shard "
                             "writing (default 1 = serial; use 0 for CPU count)")
    parser.add_argument("--max_layers_per_shard", type=int, default=1,
                        help="Maximum number of layers per output safetensors "
                             "file (default 1). 1 = one layer per file.")
    args = parser.parse_args()

    if args.use_group_routing and args.target_topk is not None:
        print(
            "ERROR: --use_group_routing and --target_topk are mutually exclusive. "
            "Grouped routing keeps moe_topk unchanged by design.",
            file=sys.stderr,
        )
        sys.exit(1)

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

    expert_count_key, original_experts, zero_expert_num = find_expert_count(config)
    if original_experts == 0:
        print(
            "ERROR: Could not find any expert count key in config.json. "
            f"Tried: {', '.join(EXPERT_COUNT_KEYS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_routed = original_experts + zero_expert_num
    target_experts = args.target_experts if args.target_experts is not None else original_experts * 2

    if target_experts <= original_experts:
        print(
            f"ERROR: target_experts ({target_experts}) must be greater than "
            f"original_experts ({original_experts}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if target_experts % original_experts != 0:
        print(f"ERROR: Target experts ({target_experts}) must be a multiple of original ({original_experts})")
        sys.exit(1)

    expansion_factor = target_experts // original_experts
    target_zero_expert_num = zero_expert_num * expansion_factor

    print(f"\nExpert 槽位:  {original_experts}  →  {target_experts}  (expansion factor: {expansion_factor}x)")
    if zero_expert_num > 0:
        print(f"Zero expert:  {zero_expert_num}  →  {target_zero_expert_num}")
        print(f"Router dim:   {total_routed}  →  {target_experts + target_zero_expert_num}")
    if args.router_noise_scale > 0:
        print(f"Router noise: {args.router_noise_scale}")
    if args.expert_noise_scale > 0:
        print(f"Expert noise: {args.expert_noise_scale}")

    shard_files = sorted(set(index["weight_map"].values()))
    experts_by_layer = validate_expert_layout(index, original_experts, zero_expert_num)
    source_to_targets = build_expert_target_map(original_experts, target_experts)
    print(f"Detected {len(experts_by_layer)} MoE layer(s) with {original_experts} experts each")

    # ── Update & write config ───────────────────────────────────────────
    target_topk = None if args.use_group_routing else args.target_topk
    new_config = expand_config(config, target_experts, target_zero_expert_num, target_topk)
    if args.use_group_routing:
        new_config["use_group_routing"] = True
        new_config["expert_expansion_factor"] = expansion_factor
    describe_config_diff(config, new_config)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\nWARNING: Output directory already exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    print("Updated config.json written.")

    target_shard_size = auto_detect_shard_size(model_dir, shard_files)
    print(f"Target shard size: {target_shard_size / 1e9:.2f} GB")

    workers = args.workers if args.workers > 0 else (__import__("os").cpu_count() or 4)
    new_weight_map: dict[str, str] = {}

    # ── Pass 1: scan, sort by layer, assign to output shards ─────────────
    print("\nPass 1/2: Scanning headers, sorting by layer, assigning to shards...")
    (assignments_by_shard, num_output_shards, total_output_bytes,
     total_original, total_duplicated) = _pre_scan_assignments(
        model_dir, shard_files, target_shard_size,
        original_experts, zero_expert_num, expansion_factor,
        source_to_targets, target_experts, args.expert_noise_scale,
        max_layers_per_shard=args.max_layers_per_shard,
    )

    print(
        f"Output plan: {total_original:,} original + {total_duplicated:,} duplicated "
        f"= {total_original + total_duplicated:,} tensors"
    )
    print(
        f"Planned output size: {total_output_bytes / 1e9:.2f} GB across "
        f"{num_output_shards} shard(s) "
        f"(~{total_output_bytes / num_output_shards / 1e9:.2f} GB each)"
    )
    print(f"Layer grouping: ≤ {args.max_layers_per_shard} layer(s) per shard")

    # ── Pass 2: write output shards ──────────────────────────────────────
    print("\nPass 2/2: Writing output shards...")
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

    if workers > 1:
        print(f"  Parallel: {workers} workers, {len(tasks)} shards")
        chunksize = max(1, len(tasks) // workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = list(tqdm(
                executor.map(_write_output_shard, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Writing output shards",
            ))
            for weight_entries in futures:
                for name, shard_name in weight_entries:
                    new_weight_map[name] = shard_name
    else:
        print(f"  Serial: {len(tasks)} shards")
        for task in tqdm(tasks, desc="Writing shards"):
            for name, shard_name in _write_output_shard(task):
                new_weight_map[name] = shard_name

    # ── Write new index ──────────────────────────────────────────────────
    metadata = {**index.get("metadata", {})}
    metadata["total_size"] = total_output_bytes
    new_index = {
        "metadata": metadata,
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # ── Copy auxiliary files ────────────────────────────────────────────
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if fpath.is_file() and fpath.suffix not in skip_suffixes and fpath.name not in skip_names:
            shutil.copy2(fpath, output_dir / fpath.name)
            print(f"  Copied: {fpath.name}")

    print("\nVerification:")
    print(f"  Expert count key used: {expert_count_key or 'n_routed_experts'}")
    print(f"  Config experts: {target_experts} (real) + {target_zero_expert_num} (zero) "
          f"= {target_experts + target_zero_expert_num} total routed")
    if args.use_group_routing:
        print(f"  Routing: grouped (方案一), topk unchanged")
    elif args.target_topk is not None:
        print(f"  Topk: {args.target_topk}")
    print(f"  Output shards: {num_output_shards}")
    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
