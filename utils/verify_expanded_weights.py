#!/usr/bin/env python3
"""
Verify that an expanded model's weights match the original model's weights.

Supports:
1. Layer Expansion: Verifies original layers and duplicated layers.
2. MoE Expert Expansion: Verifies routers and expert weights.
"""

import argparse
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

from utils.shared import (
    ALL_ROUTER_SUFFIXES,
    build_layer_mapping,
    find_expert_count,
    get_expert_info,
    get_layer_index,
    load_config,
    load_index,
    make_expert_key,
    parse_copy_source,
    set_layer_index,
    should_zero,
)


# --- ModelWeightLoader ---


class ModelWeightLoader:
    """Thread-safe lazy loader for sharded safetensors models."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.config = load_config(model_dir)
        self.index = load_index(model_dir)
        self.weight_map = self.index["weight_map"] if self.index else None

        if not self.weight_map:
            files = sorted(list(model_dir.glob("*.safetensors")))
            if not files:
                raise FileNotFoundError(f"No safetensors found in {model_dir}")

            self.weight_map = {}
            for f in files:
                with safe_open(f, framework="pt") as sf:
                    for k in sf.keys():
                        if k in self.weight_map:
                            print(f"WARNING: Duplicate parameter {k} found in {f.name} and {self.weight_map[k]}")
                        self.weight_map[k] = f.name

        self._local = threading.local()
        self.params_by_shard = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.params_by_shard[shard].append(name)

    @property
    def shards(self):
        if not hasattr(self._local, "shards"):
            self._local.shards = {}
        return self._local.shards

    def get_tensor(self, name: str) -> torch.Tensor | None:
        if name not in self.weight_map:
            return None
        shard_name = self.weight_map[name]
        if shard_name not in self.shards:
            self.shards[shard_name] = safe_open(self.model_dir / shard_name,
                                                framework="pt")
        return self.shards[shard_name].get_tensor(name)

    def close(self):
        """Close all open safetensors handles in the current thread."""
        if hasattr(self._local, "shards"):
            self._local.shards.clear()


# --- Verification Functions ---


def verify_layers(orig_loader,
                  exp_loader,
                  original_layers,
                  target_layers,
                  copy_source,
                  insertion_mode="append",
                  workers=8):
    print(f"\n[Layers] Verifying {original_layers} -> {target_layers} layers "
          f"(mode={insertion_mode})")
    num_new = target_layers - original_layers
    source_list = parse_copy_source(copy_source, original_layers, num_new)
    layer_mapping = build_layer_mapping(
        original_layers, target_layers, source_list, insertion_mode)

    exp_to_orig: dict[int, int] = {}
    new_layer_set: set[int] = set()
    for exp_idx, (src, is_new) in enumerate(layer_mapping):
        exp_to_orig[exp_idx] = src
        if is_new:
            new_layer_set.add(exp_idx)

    # ── Structural pre-check ──────────────────────────────────────────────
    exp_layer_indices: set[int] = set()
    exp_layer_params: dict[int, set[str]] = defaultdict(set)
    exp_non_layer_params: set[str] = set()
    for name in exp_loader.weight_map:
        li = get_layer_index(name)
        if li is not None:
            exp_layer_indices.add(li)
            norm_name = name.split(f"model.layers.{li}.", 1)[-1]
            exp_layer_params[li].add(norm_name)
        else:
            exp_non_layer_params.add(name)

    expected_layers = set(range(target_layers))
    missing_layers = expected_layers - exp_layer_indices
    extra_layers = exp_layer_indices - expected_layers
    if missing_layers:
        return [f"Missing layers in expanded model: {sorted(missing_layers)}"]
    if extra_layers:
        return [f"Unexpected layers in expanded model: {sorted(extra_layers)}"]

    orig_layer_params: dict[int, set[str]] = defaultdict(set)
    orig_non_layer_params: set[str] = set()
    for name in orig_loader.weight_map:
        li = get_layer_index(name)
        if li is not None:
            norm_name = name.split(f"model.layers.{li}.", 1)[-1]
            orig_layer_params[li].add(norm_name)
        else:
            orig_non_layer_params.add(name)

    if orig_non_layer_params != exp_non_layer_params:
        print(f"WARNING: Non-layer parameter names differ.")
        diff = orig_non_layer_params ^ exp_non_layer_params
        print(f"  Difference: {diff}")

    for exp_li in range(target_layers):
        src_li = exp_to_orig[exp_li]
        sp = orig_layer_params.get(src_li, set())
        ep = exp_layer_params.get(exp_li, set())
        if sp != ep:
            kind = "new" if exp_li in new_layer_set else "kept"
            return [
                f"Param name mismatch in {kind} layer {exp_li} (←src {src_li}): "
                f"src-only={sp - ep}, exp-only={ep - sp}"
            ]

    print(f"  Structural check passed: "
          f"{len(exp_non_layer_params)} non-layer params, "
          f"{len(exp_layer_indices)} layers present, "
          f"{sum(len(v) for v in exp_layer_params.values())} layer params total")

    # ── Tensor value verification ─────────────────────────────────────────
    mismatches = []
    mismatches_lock = threading.Lock()
    exp_shards = sorted(exp_loader.params_by_shard.keys())

    def verify_shard(shard_name):
        local_mismatches = []
        with safe_open(exp_loader.model_dir / shard_name,
                       framework="pt") as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                l_idx = get_layer_index(exp_name)

                if l_idx is None:
                    src_name = exp_name
                else:
                    src_idx = exp_to_orig[l_idx]
                    src_name = set_layer_index(exp_name, src_idx)

                t_exp = sf_exp.get_tensor(exp_name)

                if l_idx is not None and l_idx in new_layer_set and should_zero(exp_name):
                    if not torch.all(t_exp == 0):
                        local_mismatches.append(
                            f"Identity layer {l_idx}: {exp_name} should be zero but is not")
                    continue

                t_orig = orig_loader.get_tensor(src_name)

                if t_orig is None:
                    local_mismatches.append(
                        f"Source missing: {exp_name} (expected source {src_name})"
                    )
                elif t_exp.shape != t_orig.shape:
                    local_mismatches.append(
                        f"Shape mismatch: {exp_name} shape={list(t_exp.shape)} "
                        f"vs {src_name} shape={list(t_orig.shape)}")
                elif not torch.equal(t_exp, t_orig):
                    local_mismatches.append(
                        f"Value mismatch: {exp_name} != {src_name}")
        if local_mismatches:
            with mismatches_lock:
                mismatches.extend(local_mismatches)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(verify_shard, shard): shard
            for shard in exp_shards
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Verifying Shards"):
            future.result()

    return mismatches


def verify_experts(orig_loader, exp_loader, router_suffixes=ALL_ROUTER_SUFFIXES,
                   workers=8, layer_mapping_args=None):
    """Verify expert expansion. Optionally handles combined depth+expert expansion.

    layer_mapping_args: optional dict with keys
        {original_layers, target_layers, copy_source, insertion_mode}
        When provided, maps expanded layer indices back to original layers.
    """
    _, orig_experts, orig_zero = find_expert_count(orig_loader.config)
    _, exp_experts, exp_zero = find_expert_count(exp_loader.config)

    if orig_experts == 0 or exp_experts == 0:
        return [
            f"Could not find expert count in config. "
            f"Original: {orig_experts}, Expanded: {exp_experts}"
        ]

    print(
        f"\n[Experts] Verifying {orig_experts} -> {exp_experts} routed experts"
    )
    print(f"           Zero experts: {orig_zero} (orig), {exp_zero} (exp)")

    expansion_factor = exp_experts // orig_experts
    if exp_experts % orig_experts != 0:
        return [
            f"Expanded expert count ({exp_experts}) is not a multiple "
            f"of original ({orig_experts})"
        ]

    if orig_zero > 0 and exp_zero != orig_zero * expansion_factor:
        return [
            f"Zero expert count mismatch: expected {orig_zero * expansion_factor} "
            f"(orig {orig_zero} × factor {expansion_factor}), got {exp_zero}"
        ]

    # ── Layer mapping for combined expansion ─────────────────────────────
    exp_to_orig_layer: dict[int, int] | None = None
    new_layer_set: set[int] = set()
    if layer_mapping_args:
        lm = layer_mapping_args
        num_new = lm["target_layers"] - lm["original_layers"]
        source_list = parse_copy_source(lm["copy_source"], lm["original_layers"], num_new)
        full_mapping = build_layer_mapping(
            lm["original_layers"], lm["target_layers"], source_list, lm["insertion_mode"])
        exp_to_orig_layer = {}
        for exp_idx, (src, is_new) in enumerate(full_mapping):
            exp_to_orig_layer[exp_idx] = src
            if is_new:
                new_layer_set.add(exp_idx)

    def _map_layer(exp_li: int) -> int:
        if exp_to_orig_layer is not None:
            if exp_li not in exp_to_orig_layer:
                raise ValueError(
                    f"Expanded layer {exp_li} not found in layer mapping "
                    f"(range 0-{len(exp_to_orig_layer) - 1})")
            return exp_to_orig_layer[exp_li]
        return exp_li

    # ── Structural pre-check (experts) ───────────────────────────────────
    exp_experts_by_layer: dict[int, set[int]] = defaultdict(set)
    exp_router_layers: set[int] = set()
    exp_expert_params: dict[int, set[str]] = defaultdict(set)

    for name in exp_loader.weight_map:
        info = get_expert_info(name)
        if info:
            l_idx, e_idx, rest = info
            exp_experts_by_layer[l_idx].add(e_idx)
            exp_expert_params[l_idx].add(rest)
        elif name.endswith(router_suffixes):
            li = get_layer_index(name)
            if li is not None:
                exp_router_layers.add(li)

    orig_experts_by_layer: dict[int, set[int]] = defaultdict(set)
    orig_router_layers: set[int] = set()
    orig_expert_params: dict[int, set[str]] = defaultdict(set)

    for name in orig_loader.weight_map:
        info = get_expert_info(name)
        if info:
            l_idx, e_idx, rest = info
            orig_experts_by_layer[l_idx].add(e_idx)
            orig_expert_params[l_idx].add(rest)
        elif name.endswith(router_suffixes):
            li = get_layer_index(name)
            if li is not None:
                orig_router_layers.add(li)

    if exp_to_orig_layer is not None:
        mapped_exp_router_layers = {_map_layer(li) for li in exp_router_layers}
        if not orig_router_layers.issubset(mapped_exp_router_layers):
            return [
                f"Router layer mismatch (combined). "
                f"Orig layers: {sorted(orig_router_layers)}, "
                f"Mapped exp layers: {sorted(mapped_exp_router_layers)}"
            ]
    elif orig_router_layers != exp_router_layers:
        return [
            f"Router layer mismatch. "
            f"Orig layers: {sorted(orig_router_layers)}, "
            f"Exp layers: {sorted(exp_router_layers)}"
        ]

    # Detect whether zero experts have stored weights in the original model.
    # Identity-type zero experts (e.g., LongCat-Flash-Lite) have no parameters
    # in safetensors; only routed expert weights are stored.
    orig_has_zero_expert_weights = False
    if orig_zero > 0:
        for name in orig_loader.weight_map:
            info = get_expert_info(name)
            if info:
                _, e_idx, _ = info
                if e_idx >= orig_experts:
                    orig_has_zero_expert_weights = True
                    break

    if orig_has_zero_expert_weights:
        target_total_experts = exp_experts + exp_zero
    else:
        target_total_experts = exp_experts

    for layer_idx in exp_experts_by_layer:
        actual_indices = sorted(exp_experts_by_layer[layer_idx])
        expected_indices = list(range(target_total_experts))
        if actual_indices != expected_indices:
            return [
                f"Layer {layer_idx}: expert indices mismatch. "
                f"Expected [0-{target_total_experts - 1}], "
                f"got {actual_indices[:8]}{'...' if len(actual_indices) > 8 else ''}"
            ]

        orig_li = _map_layer(layer_idx)
        op = orig_expert_params.get(orig_li, set())
        ep = exp_expert_params.get(layer_idx, set())
        if op != ep:
            return [
                f"Layer {layer_idx}: expert parameter name mismatch. "
                f"orig-only={op - ep}, exp-only={ep - op}"
            ]

    print(
        f"  Structural check passed: "
        f"{len(exp_experts_by_layer)} MoE layers, "
        f"{target_total_experts} experts/layer, "
        f"{len(exp_router_layers)} router layers"
    )

    mismatches = []
    mismatches_lock = threading.Lock()
    exp_shards = sorted(exp_loader.params_by_shard.keys())

    def verify_shard(shard_name):
        local_mismatches = []
        with safe_open(exp_loader.model_dir / shard_name,
                       framework="pt") as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                exp_li = get_layer_index(exp_name)
                orig_li = _map_layer(exp_li) if exp_li is not None else None
                is_new_layer = exp_li is not None and exp_li in new_layer_set

                def _remap_name(name):
                    if orig_li is not None and orig_li != exp_li:
                        return set_layer_index(name, orig_li)
                    return name

                # 1. Router parameters
                if exp_name.endswith(router_suffixes):
                    t_exp = sf_exp.get_tensor(exp_name)
                    orig_name = _remap_name(exp_name)
                    t_orig = orig_loader.get_tensor(orig_name)

                    if t_orig is None:
                        local_mismatches.append(
                            f"Source router missing: {exp_name}")
                        continue

                    expected_dim0 = orig_experts * expansion_factor + orig_zero * expansion_factor
                    if t_exp.shape[0] != expected_dim0:
                        local_mismatches.append(
                            f"Router shape mismatch: {exp_name} "
                            f"shape={list(t_exp.shape)} expected dim0={expected_dim0}"
                        )
                        continue

                    real_orig = t_orig[:orig_experts]
                    real_exp = t_exp[:exp_experts]

                    for f in range(expansion_factor):
                        part = real_exp[f * orig_experts:(f + 1) *
                                        orig_experts]
                        if not torch.equal(real_orig, part):
                            local_mismatches.append(
                                f"Router value mismatch "
                                f"(real part factor {f}): {exp_name}")

                    if orig_zero > 0:
                        zero_orig = t_orig[orig_experts:]
                        zero_exp = t_exp[exp_experts:]
                        for f in range(expansion_factor):
                            part = zero_exp[f * orig_zero:(f + 1) * orig_zero]
                            if not torch.equal(zero_orig, part):
                                local_mismatches.append(
                                    f"Router value mismatch "
                                    f"(zero part factor {f}): {exp_name}"
                                )
                    continue

                # 2. Expert parameters
                info = get_expert_info(exp_name)
                if info:
                    l_idx, e_idx, rest = info
                    src_l_idx = _map_layer(l_idx)

                    if is_new_layer and should_zero(exp_name):
                        t_exp = sf_exp.get_tensor(exp_name)
                        if not torch.all(t_exp == 0):
                            local_mismatches.append(
                                f"Identity layer {l_idx}: {exp_name} should be zero")
                        continue

                    if e_idx < orig_experts:
                        src_name = make_expert_key(src_l_idx, e_idx, rest)
                    elif e_idx < exp_experts:
                        src_e_idx = e_idx % orig_experts
                        src_name = make_expert_key(src_l_idx, src_e_idx, rest)
                    elif orig_has_zero_expert_weights:
                        src_e_idx = orig_experts + ((e_idx - exp_experts) % orig_zero)
                        src_name = make_expert_key(src_l_idx, src_e_idx, rest)
                    else:
                        continue

                    t_exp = sf_exp.get_tensor(exp_name)
                    t_orig = orig_loader.get_tensor(src_name)

                    if t_orig is None:
                        local_mismatches.append(
                            f"Source expert missing: {exp_name} "
                            f"(expected {src_name})")
                    elif t_exp.shape != t_orig.shape:
                        local_mismatches.append(
                            f"Expert shape mismatch: {exp_name} "
                            f"shape={list(t_exp.shape)} vs "
                            f"{src_name} shape={list(t_orig.shape)}")
                    elif not torch.equal(t_exp, t_orig):
                        local_mismatches.append(
                            f"Expert value mismatch: {exp_name} != {src_name}")
                    continue

                # 3. Regular parameters
                t_exp = sf_exp.get_tensor(exp_name)
                orig_name = _remap_name(exp_name)

                if is_new_layer and should_zero(exp_name):
                    if not torch.all(t_exp == 0):
                        local_mismatches.append(
                            f"Identity layer: {exp_name} should be zero")
                    continue

                t_orig = orig_loader.get_tensor(orig_name)
                if t_orig is None:
                    local_mismatches.append(f"Source missing: {exp_name}")
                elif t_exp.shape != t_orig.shape:
                    local_mismatches.append(
                        f"Shape mismatch: {exp_name} shape={list(t_exp.shape)} "
                        f"vs shape={list(t_orig.shape)}")
                elif not torch.equal(t_exp, t_orig):
                    local_mismatches.append(f"Value mismatch: {exp_name}")
        if local_mismatches:
            with mismatches_lock:
                mismatches.extend(local_mismatches)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(verify_shard, shard): shard
            for shard in exp_shards
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Verifying Shards"):
            future.result()

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Verify expanded model weights")
    parser.add_argument("--orig_dir",
                        type=str,
                        required=True,
                        help="Original model directory")
    parser.add_argument("--exp_dir",
                        type=str,
                        required=True,
                        help="Expanded model directory")
    parser.add_argument("--type",
                        type=str,
                        choices=["layers", "experts", "combined"],
                        required=True,
                        help="Expansion type")

    parser.add_argument("--orig_layers",
                        type=int,
                        default=28,
                        help="Original number of layers")
    parser.add_argument("--target_layers",
                        type=int,
                        default=56,
                        help="Target number of layers")
    parser.add_argument("--copy_source",
                        type=str,
                        default="seq",
                        help="Copy source mapping (seq, idx, or comma list)")
    parser.add_argument("--insertion_mode",
                        type=str,
                        choices=["interleave", "append"],
                        default="append",
                        help="Layer insertion mode used during expansion")
    parser.add_argument("--router_suffixes",
                        type=str,
                        default=None,
                        help="Comma-separated custom router suffixes")

    parser.add_argument("--workers",
                        type=int,
                        default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir)
    exp_dir = Path(args.exp_dir)

    if not orig_dir.exists() or not exp_dir.exists():
        print(f"ERROR: Directory not found. Orig: {orig_dir}, Exp: {exp_dir}")
        sys.exit(1)

    try:
        orig_loader = ModelWeightLoader(orig_dir)
        exp_loader = ModelWeightLoader(exp_dir)
    except Exception as e:
        print(f"ERROR initializing loaders: {e}")
        sys.exit(1)

    if args.type == "layers":
        try:
            mismatches = verify_layers(
                orig_loader,
                exp_loader,
                args.orig_layers,
                args.target_layers,
                args.copy_source,
                insertion_mode=args.insertion_mode,
                workers=args.workers,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.type == "combined":
        router_suffixes = ALL_ROUTER_SUFFIXES
        if args.router_suffixes:
            router_suffixes = tuple(s.strip()
                                    for s in args.router_suffixes.split(","))
        layer_mapping_args = {
            "original_layers": args.orig_layers,
            "target_layers": args.target_layers,
            "copy_source": args.copy_source,
            "insertion_mode": args.insertion_mode,
        }
        mismatches = verify_experts(
            orig_loader,
            exp_loader,
            router_suffixes=router_suffixes,
            workers=args.workers,
            layer_mapping_args=layer_mapping_args,
        )
    else:
        router_suffixes = ALL_ROUTER_SUFFIXES
        if args.router_suffixes:
            router_suffixes = tuple(s.strip()
                                    for s in args.router_suffixes.split(","))

        mismatches = verify_experts(
            orig_loader,
            exp_loader,
            router_suffixes=router_suffixes,
            workers=args.workers,
        )

    orig_loader.close()
    exp_loader.close()

    if mismatches:
        print(f"\n❌ Verification FAILED with {len(mismatches)} mismatches!")
        for m in mismatches[:50]:
            print(f"  - {m}")
        if len(mismatches) > 50:
            print(f"  ... and {len(mismatches) - 50} more")
        sys.exit(1)
    else:
        print("\n✅ Verification SUCCESSFUL! All weights match perfectly.")


if __name__ == "__main__":
    main()
