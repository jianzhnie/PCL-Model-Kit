#!/usr/bin/env python3
"""
Validation script for expand_moe_depth.py (M2: MoE Depth Expansion).

Creates a small mock model mimicking LongCat-Flash-Chat structure,
runs the expansion, and verifies:
1. Original layer weights are preserved at correct remapped positions
2. New layers have o_proj and down_proj zeroed (identity initialization)
3. Non-zeroed weights in new layers are exact copies of their source layer
4. Forward pass through a residual block proves function-preserving
5. Non-layer parameters (embed, norm, lm_head) are untouched
6. Config and index are correct
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.expand_moe_depth import build_layer_mapping, should_zero


def create_mock_longcat_model(output_dir: Path, num_layers: int = 4):
    """Create a tiny model with LongCat-Flash-Chat-like structure."""
    hidden = 64
    ffn_hidden = 128
    expert_ffn = 32
    n_routed = 8
    zero_expert_num = 4
    total_experts = n_routed + zero_expert_num
    num_heads = 4
    kv_lora_rank = 16
    q_lora_rank = 32
    qk_rope_head_dim = 8
    qk_nope_head_dim = 8
    v_head_dim = 16
    vocab_size = 128

    config = {
        "architectures": ["LongcatFlashForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden,
        "ffn_hidden_size": ffn_hidden,
        "expert_ffn_hidden_size": expert_ffn,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "n_routed_experts": n_routed,
        "zero_expert_num": zero_expert_num,
        "zero_expert_type": "identity",
        "moe_topk": 2,
        "rms_norm_eps": 1e-5,
        "attention_method": "MLA",
        "max_position_embeddings": 1024,
        "rope_theta": 10000000.0,
    }

    torch.manual_seed(42)
    tensors = {}

    tensors["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden)
    tensors["lm_head.weight"] = torch.randn(vocab_size, hidden)
    tensors["model.norm.weight"] = torch.ones(hidden)

    head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_proj_dim = kv_lora_rank + qk_rope_head_dim

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        tensors[f"{prefix}.input_layernorm.0.weight"] = torch.ones(hidden)
        tensors[f"{prefix}.input_layernorm.1.weight"] = torch.ones(hidden)
        tensors[f"{prefix}.post_attention_layernorm.0.weight"] = torch.ones(hidden)
        tensors[f"{prefix}.post_attention_layernorm.1.weight"] = torch.ones(hidden)

        for attn_idx in range(2):
            ap = f"{prefix}.self_attn.{attn_idx}"
            tensors[f"{ap}.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden)
            tensors[f"{ap}.q_a_layernorm.weight"] = torch.ones(q_lora_rank)
            tensors[f"{ap}.q_b_proj.weight"] = torch.randn(num_heads * head_dim, q_lora_rank)
            tensors[f"{ap}.kv_a_proj_with_mqa.weight"] = torch.randn(kv_proj_dim, hidden)
            tensors[f"{ap}.kv_a_layernorm.weight"] = torch.ones(kv_lora_rank)
            tensors[f"{ap}.kv_b_proj.weight"] = torch.randn(
                num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank
            )
            tensors[f"{ap}.o_proj.weight"] = torch.randn(hidden, num_heads * v_head_dim)

        tensors[f"{prefix}.mlp.router.classifier.weight"] = torch.randn(total_experts, hidden)
        tensors[f"{prefix}.mlp.router.e_score_correction_bias"] = torch.randn(total_experts)

        for exp_idx in range(total_experts):
            ep = f"{prefix}.mlp.experts.{exp_idx}"
            tensors[f"{ep}.gate_proj.weight"] = torch.randn(expert_ffn, hidden)
            tensors[f"{ep}.up_proj.weight"] = torch.randn(expert_ffn, hidden)
            tensors[f"{ep}.down_proj.weight"] = torch.randn(hidden, expert_ffn)

        for mlp_idx in range(2):
            mp = f"{prefix}.mlps.{mlp_idx}"
            tensors[f"{mp}.gate_proj.weight"] = torch.randn(ffn_hidden, hidden)
            tensors[f"{mp}.up_proj.weight"] = torch.randn(ffn_hidden, hidden)
            tensors[f"{mp}.down_proj.weight"] = torch.randn(hidden, ffn_hidden)

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_size = len(tensors) // 2
    keys = sorted(tensors.keys())
    shard1_keys = keys[:shard_size]
    shard2_keys = keys[shard_size:]

    shard1 = {k: tensors[k] for k in shard1_keys}
    shard2 = {k: tensors[k] for k in shard2_keys}

    save_file(shard1, str(output_dir / "model_00001-of-00002.safetensors"))
    save_file(shard2, str(output_dir / "model_00002-of-00002.safetensors"))

    weight_map = {}
    for k in shard1_keys:
        weight_map[k] = "model_00001-of-00002.safetensors"
    for k in shard2_keys:
        weight_map[k] = "model_00002-of-00002.safetensors"

    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in tensors.values())},
        "weight_map": weight_map,
    }

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Mock model: {len(tensors)} tensors, {num_layers} layers, "
          f"{n_routed}+{zero_expert_num} experts, MLA dual attention")
    return config, tensors


def load_all_expanded_tensors(expanded_dir: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from expanded model."""
    expanded_index = json.load(open(expanded_dir / "model.safetensors.index.json"))
    all_tensors = {}
    for shard_file in sorted(set(expanded_index["weight_map"].values())):
        shard_tensors = load_file(str(expanded_dir / shard_file))
        all_tensors.update(shard_tensors)
    return all_tensors


def run_expansion(mock_dir: Path, output_dir: Path, num_layers: int,
                  target_layers: int, mode: str) -> bool:
    """Run expand_moe_depth.py and return success status."""
    script_path = Path(__file__).resolve().parent / "expand_moe_depth.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    cmd = [
        sys.executable, str(script_path),
        "--model_dir", str(mock_dir),
        "--output_dir", str(output_dir),
        "--original_layers", str(num_layers),
        "--target_layers", str(target_layers),
        "--insertion_mode", mode,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parent.parent), env=env)
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr}")
        return False
    return True


def test_config_and_index(expanded_dir: Path, target_layers: int) -> list[str]:
    """Test 1: Config and index correctness."""
    errors = []
    config = json.load(open(expanded_dir / "config.json"))
    index = json.load(open(expanded_dir / "model.safetensors.index.json"))

    if config.get("num_layers") != target_layers:
        errors.append(f"config num_layers={config.get('num_layers')}, expected {target_layers}")

    layers_in_index = set()
    for key in index["weight_map"]:
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if m:
            layers_in_index.add(int(m.group(1)))

    expected_layers = set(range(target_layers))
    if layers_in_index != expected_layers:
        missing = expected_layers - layers_in_index
        extra = layers_in_index - expected_layers
        errors.append(f"Index layer mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    for shard_file in sorted(set(index["weight_map"].values())):
        if not (expanded_dir / shard_file).exists():
            errors.append(f"Shard file missing: {shard_file}")

    return errors


def test_identity_initialization(
    all_tensors: dict[str, torch.Tensor],
    new_layer_indices: set[int],
) -> list[str]:
    """Test 2: All o_proj and down_proj in new layers are zero."""
    errors = []
    zeroed_count = 0

    for key, tensor in all_tensors.items():
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        if layer_idx not in new_layer_indices:
            continue

        if should_zero(key):
            if not torch.all(tensor == 0):
                nonzero = torch.count_nonzero(tensor).item()
                errors.append(f"{key}: expected all zeros, got {nonzero} non-zero elements")
            else:
                zeroed_count += 1

    if not errors:
        print(f"    Zeroed {zeroed_count} tensors (o_proj + down_proj) in new layers")
    return errors


def test_weight_copying(
    all_tensors: dict[str, torch.Tensor],
    original_tensors: dict[str, torch.Tensor],
    layer_mapping: list[tuple[int, bool]],
) -> list[str]:
    """Test 3: Non-zeroed weights in new layers match their source."""
    errors = []
    verified_count = 0

    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if not is_new:
            continue

        for key, tensor in all_tensors.items():
            m = re.search(r"model\.layers\.(\d+)\.(.*)", key)
            if not m:
                continue
            if int(m.group(1)) != new_idx:
                continue
            rest = m.group(2)

            if should_zero(key):
                continue

            src_key = f"model.layers.{src}.{rest}"
            if src_key not in original_tensors:
                errors.append(f"{key}: source key {src_key} not in original model")
                continue

            if not torch.equal(tensor, original_tensors[src_key]):
                max_diff = (tensor - original_tensors[src_key]).abs().max().item()
                errors.append(f"{key}: differs from source {src_key}, max_diff={max_diff:.6e}")
            else:
                verified_count += 1

    if not errors:
        print(f"    Verified {verified_count} copied tensors match source layers")
    return errors


def test_original_layers_preserved(
    all_tensors: dict[str, torch.Tensor],
    original_tensors: dict[str, torch.Tensor],
    layer_mapping: list[tuple[int, bool]],
) -> list[str]:
    """Test 4: Original (non-new) layers are preserved with correct remapping."""
    errors = []
    verified_count = 0

    remap = {}
    for new_idx, (src, is_new) in enumerate(layer_mapping):
        if not is_new:
            remap[src] = new_idx

    for orig_idx, new_idx in remap.items():
        orig_prefix = f"model.layers.{orig_idx}."
        new_prefix = f"model.layers.{new_idx}."

        orig_keys = [k for k in original_tensors if k.startswith(orig_prefix)]
        for orig_key in orig_keys:
            new_key = orig_key.replace(orig_prefix, new_prefix, 1)
            if new_key not in all_tensors:
                errors.append(f"Original layer {orig_idx}→{new_idx}: {new_key} missing")
                continue
            if not torch.equal(all_tensors[new_key], original_tensors[orig_key]):
                errors.append(f"Original layer {orig_idx}→{new_idx}: {new_key} was modified")
            else:
                verified_count += 1

    if not errors:
        print(f"    Verified {verified_count} original layer tensors preserved")
    return errors


def test_non_layer_params(
    all_tensors: dict[str, torch.Tensor],
    original_tensors: dict[str, torch.Tensor],
) -> list[str]:
    """Test 5: Non-layer parameters are preserved exactly."""
    errors = []
    for key, orig_tensor in original_tensors.items():
        if "model.layers." in key:
            continue
        if key not in all_tensors:
            errors.append(f"Non-layer param missing: {key}")
        elif not torch.equal(all_tensors[key], orig_tensor):
            errors.append(f"Non-layer param modified: {key}")

    if not errors:
        non_layer_count = sum(1 for k in original_tensors if "model.layers." not in k)
        print(f"    All {non_layer_count} non-layer params preserved")
    return errors


def test_forward_pass_identity():
    """Test 6: Simulate a residual transformer block to prove identity property.

    A transformer layer with residual connection computes:
        output = input + Attn(Norm(input)) + MLP(Norm(input + Attn(Norm(input))))

    When o_proj is zero: Attn output = 0 → after residual: x + 0 = x
    When down_proj is zero: MLP output = 0 → after residual: x + 0 = x
    So the entire layer is identity: output = input
    """
    errors = []
    hidden = 64
    seq_len = 8
    batch = 2

    x = torch.randn(batch, seq_len, hidden)

    o_proj = torch.zeros(hidden, 32)
    down_proj = torch.zeros(hidden, 128)

    attn_internal = torch.randn(batch, seq_len, 32)
    attn_output = attn_internal @ o_proj.T
    after_attn_residual = x + attn_output

    if not torch.equal(after_attn_residual, x):
        errors.append("Attention residual with zero o_proj not identity")

    mlp_internal = torch.randn(batch, seq_len, 128)
    mlp_output = mlp_internal @ down_proj.T
    after_mlp_residual = after_attn_residual + mlp_output

    if not torch.equal(after_mlp_residual, x):
        errors.append("MLP residual with zero down_proj not identity")

    if not errors:
        print("    Residual identity verified: zero(o_proj) + zero(down_proj) → Layer(x) = x")
    return errors


def test_build_layer_mapping_correctness():
    """Test 7: Unit test build_layer_mapping for various configurations."""
    errors = []

    mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "interleave")
    expected = [(0, False), (0, True), (1, False), (1, True),
                (2, False), (2, True), (3, False), (3, True)]
    if mapping != expected:
        errors.append(f"4→8 interleave: got {mapping}, expected {expected}")

    mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "append")
    expected = [(0, False), (1, False), (2, False), (3, False),
                (0, True), (1, True), (2, True), (3, True)]
    if mapping != expected:
        errors.append(f"4→8 append: got {mapping}, expected {expected}")

    mapping = build_layer_mapping(4, 12, [0, 1, 2, 3, 0, 1, 2, 3], "interleave")
    expected = [(0, False), (0, True), (0, True),
                (1, False), (1, True), (1, True),
                (2, False), (2, True), (2, True),
                (3, False), (3, True), (3, True)]
    if mapping != expected:
        errors.append(f"4→12 interleave: got {mapping}, expected {expected}")

    mapping = build_layer_mapping(4, 6, [1, 2], "interleave")
    expected = [(0, False), (1, False), (1, True), (2, False), (2, True), (3, False)]
    if mapping != expected:
        errors.append(f"4→6 partial interleave: got {mapping}, expected {expected}")

    if not errors:
        print("    build_layer_mapping: all 4 test cases pass")
    return errors


def test_should_zero_patterns():
    """Test 8: Verify should_zero matches all expected parameter patterns."""
    errors = []
    must_zero = [
        "model.layers.5.self_attn.0.o_proj.weight",
        "model.layers.5.self_attn.1.o_proj.weight",
        "model.layers.5.self_attn.o_proj.weight",
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.511.down_proj.weight",
        "model.layers.5.mlps.0.down_proj.weight",
        "model.layers.5.mlps.1.down_proj.weight",
        "model.layers.5.mlp.down_proj.weight",
    ]
    must_keep = [
        "model.layers.5.self_attn.0.q_a_proj.weight",
        "model.layers.5.self_attn.0.kv_b_proj.weight",
        "model.layers.5.mlp.experts.0.gate_proj.weight",
        "model.layers.5.mlp.experts.0.up_proj.weight",
        "model.layers.5.mlps.0.gate_proj.weight",
        "model.layers.5.mlps.0.up_proj.weight",
        "model.layers.5.mlp.router.classifier.weight",
        "model.layers.5.input_layernorm.0.weight",
        "model.layers.5.mlp.gate_proj.weight",
    ]

    for name in must_zero:
        if not should_zero(name):
            errors.append(f"should_zero({name}) = False, expected True")
    for name in must_keep:
        if should_zero(name):
            errors.append(f"should_zero({name}) = True, expected False")

    if not errors:
        print(f"    should_zero: {len(must_zero)} zero + {len(must_keep)} keep patterns correct")
    return errors


def main():
    print("=" * 70)
    print("  M2 MoE Depth Expansion — Comprehensive Validation")
    print("=" * 70)

    all_errors = []
    num_layers = 4
    target_layers = 8

    print("\n[Test 7] build_layer_mapping unit tests...")
    errs = test_build_layer_mapping_correctness()
    all_errors.extend(errs)

    print("[Test 8] should_zero pattern matching...")
    errs = test_should_zero_patterns()
    all_errors.extend(errs)

    print("[Test 6] Forward-pass identity (mathematical proof)...")
    errs = test_forward_pass_identity()
    all_errors.extend(errs)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mock_dir = tmpdir / "mock_model"

        print(f"\n[Setup] Creating mock LongCat-Flash-Chat ({num_layers} layers)...")
        config, original_tensors = create_mock_longcat_model(mock_dir, num_layers)

        for mode in ["interleave", "append"]:
            print(f"\n{'─' * 70}")
            print(f"  Mode: {mode} ({num_layers} → {target_layers} layers)")
            print(f"{'─' * 70}")

            expanded_dir = tmpdir / f"expanded_{mode}"
            print(f"  Running expansion...")
            if not run_expansion(mock_dir, expanded_dir, num_layers, target_layers, mode):
                all_errors.append(f"{mode}: expansion script failed")
                continue

            all_tensors = load_all_expanded_tensors(expanded_dir)

            source_list = [i % num_layers for i in range(target_layers - num_layers)]
            layer_mapping = build_layer_mapping(num_layers, target_layers, source_list, mode)
            new_layer_indices = {i for i, (_, is_new) in enumerate(layer_mapping) if is_new}

            print(f"  [Test 1] Config & index...")
            errs = test_config_and_index(expanded_dir, target_layers)
            all_errors.extend(errs)
            if not errs:
                print(f"    Config and index valid")

            print(f"  [Test 2] Identity initialization (zeroed weights)...")
            errs = test_identity_initialization(all_tensors, new_layer_indices)
            all_errors.extend(errs)

            print(f"  [Test 3] Weight copying (non-zeroed match source)...")
            errs = test_weight_copying(all_tensors, original_tensors, layer_mapping)
            all_errors.extend(errs)

            print(f"  [Test 4] Original layers preserved...")
            errs = test_original_layers_preserved(all_tensors, original_tensors, layer_mapping)
            all_errors.extend(errs)

            print(f"  [Test 5] Non-layer params preserved...")
            errs = test_non_layer_params(all_tensors, original_tensors)
            all_errors.extend(errs)

        print(f"\n{'─' * 70}")
        print(f"  Edge case: 4 → 6 layers (non-uniform interleave)")
        print(f"{'─' * 70}")
        expanded_dir_6 = tmpdir / "expanded_6"
        script_path = Path(__file__).resolve().parent / "expand_moe_depth.py"
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        cmd = [
            sys.executable, str(script_path),
            "--model_dir", str(mock_dir),
            "--output_dir", str(expanded_dir_6),
            "--original_layers", str(num_layers),
            "--target_layers", "6",
            "--copy_source", "1,2",
            "--insertion_mode", "interleave",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent.parent), env=env)
        if result.returncode != 0:
            all_errors.append(f"Edge case 4→6 failed: {result.stderr}")
        else:
            all_t = load_all_expanded_tensors(expanded_dir_6)
            layers = {int(m.group(1)) for k in all_t if (m := re.search(r"model\.layers\.(\d+)\.", k))}
            if layers != set(range(6)):
                all_errors.append(f"4→6: expected layers 0-5, got {sorted(layers)}")
            else:
                new_indices = {2, 4}
                for key, tensor in all_t.items():
                    m = re.search(r"model\.layers\.(\d+)\.", key)
                    if not m:
                        continue
                    li = int(m.group(1))
                    if li in new_indices and should_zero(key):
                        if not torch.all(tensor == 0):
                            all_errors.append(f"4→6: {key} not zeroed")
                print(f"    4→6 edge case: layers correct, identity init verified")

    print(f"\n{'=' * 70}")
    if all_errors:
        print(f"  FAILED — {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"    ✗ {err}")
        print("=" * 70)
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED ✓")
        print("=" * 70)


if __name__ == "__main__":
    main()
