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
      [--noise-scale 1e-6]
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
    find_expert_count,
    get_expert_info,
    get_nbytes_from_meta,
    is_router_bias,
    is_router_weight,
    load_config,
    load_index,
    make_expert_key,
    read_safetensors_header,
    tensor_nbytes,
)

TOPK_KEYS = ["moe_topk", "num_experts_per_tok", "top_k"]


# ═══════════════════════════════════════════════════════════════════════════════
# Expert index mapping
# ═══════════════════════════════════════════════════════════════════════════════

def build_expert_target_map(
    original_experts: int,
    target_experts: int,
) -> dict[int, list[int]]:
    """Build source expert -> list of new expert indices for duplication."""
    targets: dict[int, list[int]] = defaultdict(list)
    for new_idx in range(original_experts, target_experts):
        src_idx = new_idx % original_experts
        targets[src_idx].append(new_idx)
    return dict(targets)


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
                break
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

def expand_router_weight(
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    noise_scale: float = 0.0,
) -> torch.Tensor:
    """Expand a router classifier/gate weight with optional noise on copies.

    Layout: [real_experts * expansion_factor, zero_experts * expansion_factor]

    When noise_scale > 0, duplicated blocks get small Gaussian noise to break
    symmetry so that fine-tuning can differentiate them.
    """
    if zero_expert_num > 0:
        real_part = tensor[:original_experts]
        zero_part = tensor[original_experts:]
    else:
        real_part = tensor
        zero_part = None

    # Build real expert blocks — first copy exact, rest with noise
    real_blocks = [real_part]
    for _ in range(1, expansion_factor):
        if noise_scale > 0:
            noise = torch.randn_like(real_part) * noise_scale * real_part.std()
            real_blocks.append(real_part + noise)
        else:
            real_blocks.append(real_part)
    expanded_real = torch.cat(real_blocks, dim=0)

    if zero_part is not None:
        expanded_zero = torch.cat([zero_part] * expansion_factor, dim=0)
        return torch.cat([expanded_real, expanded_zero], dim=0)
    else:
        return expanded_real


def expand_router_bias(
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
) -> torch.Tensor:
    """Expand a router score correction bias (exact copies, no noise)."""
    return expand_router_weight(tensor, original_experts, zero_expert_num, expansion_factor, noise_scale=0.0)


def _expand_tensor(
    key: str,
    tensor: torch.Tensor,
    original_experts: int,
    zero_expert_num: int,
    expansion_factor: int,
    source_to_targets: dict[int, list[int]],
    target_experts: int,
    noise_scale: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Expand a single tensor, returning {output_key: expanded_tensor}."""
    total_routed = original_experts + zero_expert_num

    if is_router_weight(key):
        validate_router_shape(key, list(tensor.shape), total_routed)
        return {key: expand_router_weight(
            tensor, original_experts, zero_expert_num, expansion_factor, noise_scale,
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
            result[key] = tensor
            for new_expert_idx in source_to_targets.get(expert_idx, []):
                new_key = make_expert_key(layer_idx, new_expert_idx, rest)
                result[new_key] = tensor.clone()
        else:
            base_new_idx = expert_idx - original_experts + target_experts
            new_key = make_expert_key(layer_idx, base_new_idx, rest)
            result[new_key] = tensor
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
            ):
                if output_nbytes + current_bytes > target_shard_size and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += output_nbytes
                total_output_bytes += output_nbytes
                if action == "clone":
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
                        target_experts: int) -> list[tuple[str, int, str]]:
    """Return list of (output_key, output_nbytes, action) for an input tensor.

    action is one of: "keep", "clone", "router_weight", "router_bias"
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
                results.append((new_key, nbytes, "clone"))
        else:
            base_new_idx = expert_idx - original_experts + target_experts
            new_key = make_expert_key(layer_idx, base_new_idx, rest)
            results.append((new_key, nbytes, "keep"))
            zero_offset = expert_idx - original_experts
            for f in range(1, expansion_factor):
                copy_idx = target_experts + zero_offset + f * zero_expert_num
                copy_key = make_expert_key(layer_idx, copy_idx, rest)
                results.append((copy_key, nbytes, "clone"))
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
) -> tuple[dict[int, list[tuple[str, str, str, str]]], int, int, int, int]:
    """Pre-scan all shard headers and assign each output tensor to an output shard.

    Returns (shard_assignments, num_output_shards, total_output_bytes,
             total_original, total_duplicated) where shard_assignments maps
    output_shard_idx → [(input_shard, input_key, output_key, action)].
    """
    total_routed = original_experts + zero_expert_num
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
            for output_key, output_nbytes, action in _expand_tensor_meta(
                key, dtype, shape, original_experts, zero_expert_num,
                total_routed, expansion_factor, source_to_targets, target_experts,
            ):
                if current_bytes + output_nbytes > target_shard_size and current_bytes > 0:
                    current_shard += 1
                    current_bytes = 0
                assignments[current_shard].append(
                    (shard_file, key, output_key, action))
                current_bytes += output_nbytes
                total_output_bytes += output_nbytes
                if action == "clone":
                    total_duplicated += 1
                else:
                    total_original += 1

    num_output_shards = current_shard + 1 if assignments else 0
    return dict(assignments), num_output_shards, total_output_bytes, total_original, total_duplicated


def _write_output_shard(args):
    """Module-level worker for ProcessPoolExecutor. Writes a single output shard.

    Args is a tuple of:
      (output_path, assignments, model_dir_str, original_experts, zero_expert_num,
       expansion_factor, noise_scale)

    assignments: list of (input_shard, input_key, output_key, action)
    """
    (output_path, assignments, model_dir_str, original_experts, zero_expert_num,
     expansion_factor, noise_scale) = args

    model_dir = Path(model_dir_str)
    by_input: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for sfile, in_key, out_key, action in assignments:
        by_input[sfile].append((in_key, out_key, action))

    tensors: dict[str, torch.Tensor] = {}
    try:
        for sfile, items in by_input.items():
            with safe_open(str(model_dir / sfile), framework="pt", device="cpu") as sf:
                for in_key, out_key, action in items:
                    tensor = sf.get_tensor(in_key)
                    if action == "keep":
                        tensors[out_key] = tensor
                    elif action == "clone":
                        tensors[out_key] = tensor.clone()
                    elif action == "router_weight":
                        tensors[out_key] = expand_router_weight(
                            tensor, original_experts, zero_expert_num,
                            expansion_factor, noise_scale,
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
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="Gaussian noise scale for duplicated classifier weights "
                             "(default 0.0 = exact copies; recommend 1e-6 to break symmetry)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for parallel output shard "
                             "writing (default 1 = serial; use 0 for CPU count)")
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
    if args.noise_scale > 0:
        print(f"Noise scale:  {args.noise_scale}")

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
    if workers > 1:
        print(f"Parallel mode: {workers} workers for output shard writing")

    new_weight_map: dict[str, str] = {}

    if workers > 1:
        # ---- Parallel path: single scan for planning + assignment ----
        print("\nPass 1/2: Scanning headers and assigning tensors to output shards...")
        (assignments_by_shard, num_output_shards, total_output_bytes,
         total_original, total_duplicated) = _pre_scan_assignments(
            model_dir, shard_files, target_shard_size,
            original_experts, zero_expert_num, expansion_factor,
            source_to_targets, target_experts,
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
                args.noise_scale,
            ))

        print("\nPass 2/2: Writing output shards...")
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
        # ---- Serial path ----
        print("\nPass 1/2: Scanning headers to determine output layout...")
        num_output_shards, total_output_bytes, total_original, total_duplicated = plan_output_layout(
            model_dir, shard_files, target_shard_size,
            original_experts, zero_expert_num, target_experts,
            expansion_factor, source_to_targets,
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

        output_shard_idx = 1
        current_tensors: dict[str, torch.Tensor] = {}
        current_bytes = 0

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

        def maybe_flush(nbytes: int):
            nonlocal current_bytes, current_tensors
            if current_bytes + nbytes > target_shard_size and current_tensors:
                flush_shard()

        print("\nPass 2/2: Processing and writing tensors...")
        for shard_file in tqdm(shard_files, desc="Input shards"):
            with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    tensor = sf.get_tensor(key)
                    for out_key, expanded in _expand_tensor(
                        key, tensor, original_experts, zero_expert_num,
                        expansion_factor, source_to_targets, target_experts,
                        args.noise_scale,
                    ).items():
                        nbytes = tensor_nbytes(expanded)
                        maybe_flush(nbytes)
                        current_tensors[out_key] = expanded
                        current_bytes += nbytes

        flush_shard()

        # ── Fixup shard names if prediction was off ─────────────────────────
        actual_shards = output_shard_idx - 1
        if actual_shards != num_output_shards:
            print(
                f"\nWARNING: Predicted {num_output_shards} shards but wrote {actual_shards}. "
                "Adjusting shard names..."
            )
            # Rename files on disk
            for i in range(1, actual_shards + 1):
                old_name = output_dir / f"model-{i:05d}-of-{num_output_shards:05d}.safetensors"
                new_name = output_dir / f"model-{i:05d}-of-{actual_shards:05d}.safetensors"
                if old_name.exists() and old_name != new_name:
                    old_name.rename(new_name)
            # Fix weight_map entries
            for key in new_weight_map:
                new_weight_map[key] = re.sub(
                    r"-of-\d+\.safetensors",
                    f"-of-{actual_shards:05d}.safetensors",
                    new_weight_map[key],
                )
            num_output_shards = actual_shards

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
