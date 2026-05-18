#!/usr/bin/env python3
"""
Verify that an expanded model's weights match the original model's weights.

Supports:
1. Layer Expansion: Verifies original layers and duplicated layers.
2. MoE Expert Expansion: Verifies routers and expert weights.
"""

import argparse
import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

# --- Constants & Shared Logic ---

EXPERT_COUNT_KEYS = ['n_routed_experts', 'n_experts', 'num_experts']
ROUTER_SUFFIXES = (
    'mlp.router.classifier.weight',
    'mlp.gate.weight',
    'mlp.router.e_score_correction_bias',
    'mlp.gate.e_score_correction_bias',
)


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        print(f"ERROR: config.json not found in {model_dir}", file=sys.stderr)
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR reading config.json in {model_dir}: {e}", file=sys.stderr)
        return {}


def load_index(model_dir: Path):
    index_path = model_dir / 'model.safetensors.index.json'
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return None


def get_layer_index(param_name: str) -> int | None:
    m = re.search(r'model\.layers\.(\d+)\.', param_name)
    if m:
        return int(m.group(1))
    return None


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name. e.g. model.layers.0.xxx → model.layers.5.xxx"""
    return re.sub(
        r'model\.layers\.(\d+)\.',
        f"model.layers.{new_index}.",
        param_name,
    )


def get_expert_info(param_name: str) -> tuple[int, int, str] | None:
    m = re.search(r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)',
                  param_name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def find_expert_count(config: dict) -> tuple[int, int]:
    """Returns (original_experts, zero_expert_num) from config."""
    for key in EXPERT_COUNT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            zero = config.get('zero_expert_num', 0) or 0
            return value, zero
    return 0, 0


def parse_copy_source(raw: str | None, num_original: int,
                      num_new: int) -> list[int]:
    """Parse --copy_source into a mapping: offset_from_original -> source_layer_idx.

    Validates that all source indices are within [0, num_original). Raises ValueError
    on invalid input.
    """
    if raw is None or raw.strip().lower() == 'seq':
        return [i % num_original for i in range(num_new)]

    raw = raw.strip()
    try:
        single = int(raw)
        if single < 0 or single >= num_original:
            raise ValueError(
                f"--copy_source {single} is out of range [0, {num_original - 1}].")
        return [single] * num_new
    except ValueError as e:
        if "out of range" in str(e):
            raise e

    try:
        parts = [int(p.strip()) for p in raw.split(',')]
    except ValueError:
        raise ValueError(f"Invalid --copy_source format: {raw}")

    if len(parts) != num_new:
        raise ValueError(
            f"--copy_source list has {len(parts)} entries, expected {num_new} (one per new layer).")

    for i, src in enumerate(parts):
        if src < 0 or src >= num_original:
            raise ValueError(
                f"--copy_source[{i}] = {src} is out of range [0, {num_original - 1}].")
    return parts


# --- ModelWeightLoader ---


class ModelWeightLoader:
    """Thread-safe lazy loader for sharded safetensors models."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.config = load_config(model_dir)
        self.index = load_index(model_dir)
        self.weight_map = self.index['weight_map'] if self.index else None

        if not self.weight_map:
            files = sorted(list(model_dir.glob('*.safetensors')))
            if not files:
                raise FileNotFoundError(f"No safetensors found in {model_dir}")
            
            self.weight_map = {}
            for f in files:
                with safe_open(f, framework='pt') as sf:
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
        if not hasattr(self._local, 'shards'):
            self._local.shards = {}
        return self._local.shards

    def get_tensor(self, name: str) -> torch.Tensor | None:
        if name not in self.weight_map:
            return None
        shard_name = self.weight_map[name]
        if shard_name not in self.shards:
            self.shards[shard_name] = safe_open(self.model_dir / shard_name,
                                                framework='pt')
        return self.shards[shard_name].get_tensor(name)

    def close(self):
        """Close all open safetensors handles in the current thread."""
        if hasattr(self._local, 'shards'):
            # safe_open handles don't have an explicit close(), 
            # but deleting the reference allows them to be closed by GC.
            self._local.shards.clear()


# --- Verification Functions ---


def verify_layers(orig_loader,
                  exp_loader,
                  original_layers,
                  target_layers,
                  copy_source,
                  workers=8):
    print(f"\n[Layers] Verifying {original_layers} -> {target_layers} layers")
    num_new = target_layers - original_layers
    mapping = parse_copy_source(copy_source, original_layers, num_new)

    # ── Structural pre-check ──────────────────────────────────────────────
    exp_layer_indices: set[int] = set()
    exp_layer_params: dict[int, set[str]] = defaultdict(set)
    exp_non_layer_params: set[str] = set()
    for name in exp_loader.weight_map:
        li = get_layer_index(name)
        if li is not None:
            exp_layer_indices.add(li)
            # Normalize name: model.layers.5.xxx -> xxx
            norm_name = re.sub(r'model\.layers\.\d+\.', '', name)
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
            norm_name = re.sub(r'model\.layers\.\d+\.', '', name)
            orig_layer_params[li].add(norm_name)
        else:
            orig_non_layer_params.add(name)

    if orig_non_layer_params != exp_non_layer_params:
        print(f"WARNING: Non-layer parameter names differ.")
        diff = orig_non_layer_params ^ exp_non_layer_params
        print(f"  Difference: {diff}")

    # Check original layers (0 to original_layers-1)
    for li in range(original_layers):
        op = orig_layer_params.get(li, set())
        ep = exp_layer_params.get(li, set())
        if op != ep:
            return [
                f"Param name mismatch in layer {li}: "
                f"orig-only={op-ep}, exp-only={ep-op}"
            ]

    # Check new layers (original_layers to target_layers-1)
    for offset, src in enumerate(mapping):
        new_li = original_layers + offset
        sp = orig_layer_params.get(src, set())
        ep = exp_layer_params.get(new_li, set())
        if sp != ep:
            return [
                f"Param name mismatch in new layer {new_li} (←src layer {src}): "
                f"src-only={sp-ep}, exp-only={ep-sp}"
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
                       framework='pt') as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                l_idx = get_layer_index(exp_name)

                if l_idx is None:
                    src_name = exp_name
                elif l_idx < original_layers:
                    src_name = exp_name
                else:
                    src_idx = mapping[l_idx - original_layers]
                    src_name = set_layer_index(exp_name, src_idx)

                t_exp = sf_exp.get_tensor(exp_name)
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
                           desc='Verifying Shards'):
            future.result()

    return mismatches


def verify_experts(orig_loader, exp_loader, router_suffixes=ROUTER_SUFFIXES, workers=8):
    orig_experts, orig_zero = find_expert_count(orig_loader.config)
    exp_experts, exp_zero = find_expert_count(exp_loader.config)

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

    # ── Structural pre-check ──────────────────────────────────────────────
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

    if orig_router_layers != exp_router_layers:
        return [
            f"Router layer mismatch. "
            f"Orig layers: {sorted(orig_router_layers)}, "
            f"Exp layers: {sorted(exp_router_layers)}"
        ]

    target_total_experts = exp_experts + exp_zero

    for layer_idx in exp_experts_by_layer:
        # Check expert indices
        actual_indices = sorted(exp_experts_by_layer[layer_idx])
        expected_indices = list(range(target_total_experts))
        if actual_indices != expected_indices:
            return [
                f"Layer {layer_idx}: expert indices mismatch. "
                f"Expected [0-{target_total_experts - 1}], "
                f"got {actual_indices[:8]}{'...' if len(actual_indices) > 8 else ''}"
            ]

        # Check expert parameter names (e.g., w1.weight, w2.weight)
        op = orig_expert_params.get(layer_idx, set())
        ep = exp_expert_params.get(layer_idx, set())
        if op != ep:
            return [
                f"Layer {layer_idx}: expert parameter name mismatch. "
                f"orig-only={op-ep}, exp-only={ep-op}"
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
                       framework='pt') as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                # 1. Router parameters
                if exp_name.endswith(router_suffixes):
                    t_exp = sf_exp.get_tensor(exp_name)
                    t_orig = orig_loader.get_tensor(exp_name)

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
                    if e_idx < orig_experts:
                        src_name = exp_name
                    elif e_idx < exp_experts:
                        src_e_idx = e_idx % orig_experts
                        src_name = (
                            f"model.layers.{l_idx}.mlp.experts.{src_e_idx}.{rest}"
                        )
                    else:
                        # Zero-expert: source is the corresponding original zero-expert
                        src_e_idx = orig_experts + ((e_idx - exp_experts) % orig_zero)
                        src_name = (
                            f"model.layers.{l_idx}.mlp.experts.{src_e_idx}.{rest}"
                        )

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
                t_orig = orig_loader.get_tensor(exp_name)
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
                           desc='Verifying Shards'):
            future.result()

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description='Verify expanded model weights')
    parser.add_argument('--orig_dir',
                        type=str,
                        required=True,
                        help='Original model directory')
    parser.add_argument('--exp_dir',
                        type=str,
                        required=True,
                        help='Expanded model directory')
    parser.add_argument('--type',
                        type=str,
                        choices=['layers', 'experts'],
                        required=True,
                        help='Expansion type')

    parser.add_argument('--orig_layers',
                        type=int,
                        default=28,
                        help='Original number of layers')
    parser.add_argument('--target_layers',
                        type=int,
                        default=56,
                        help='Target number of layers')
    parser.add_argument('--copy_source',
                        type=str,
                        default='seq',
                        help='Copy source mapping (seq, idx, or comma list)')
    parser.add_argument('--router_suffixes',
                        type=str,
                        default=None,
                        help='Comma-separated custom router suffixes')

    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='Number of parallel workers')
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

    if args.type == 'layers':
        try:
            mismatches = verify_layers(
                orig_loader,
                exp_loader,
                args.orig_layers,
                args.target_layers,
                args.copy_source,
                workers=args.workers,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        router_suffixes = ROUTER_SUFFIXES
        if args.router_suffixes:
            router_suffixes = tuple(s.strip()
                                    for s in args.router_suffixes.split(','))

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
        print('\n✅ Verification SUCCESSFUL! All weights match perfectly.')


if __name__ == '__main__':
    main()
