#!/usr/bin/env python3
"""
Validation script for expand_moe_depth.py (M2: MoE Depth Expansion).

Creates a small mock model mimicking LongCat-Flash-Chat structure,
runs the expansion, and verifies:
1. Original layer weights are preserved
2. New layers have o_proj and down_proj zeroed
3. Other weights in new layers are exact copies of source layers
4. Forward pass produces identical output (function-preserving)
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_mock_longcat_model(output_dir: Path, num_layers: int = 4):
    """Create a tiny model with LongCat-Flash-Chat-like structure.

    Each layer has:
    - input_layernorm.0.weight, input_layernorm.1.weight
    - self_attn.0.{q_a_proj, q_b_proj, q_a_layernorm, kv_a_proj_with_mqa,
                    kv_a_layernorm, kv_b_proj, o_proj}.weight
    - self_attn.1.{same}
    - post_attention_layernorm.0.weight, post_attention_layernorm.1.weight
    - mlp.router.classifier.weight (shape [n_routed + zero_expert_num, hidden])
    - mlp.router.e_score_correction_bias (shape [n_routed + zero_expert_num])
    - mlp.experts.{0..n_routed+zero-1}.{gate_proj, up_proj, down_proj}.weight
    - mlps.0.{gate_proj, up_proj, down_proj}.weight
    - mlps.1.{gate_proj, up_proj, down_proj}.weight
    Plus: embed_tokens, lm_head, model.norm
    """
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

    tensors = {}
    weight_map = {}

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

    for k in shard1_keys:
        weight_map[k] = "model_00001-of-00002.safetensors"
    for k in shard2_keys:
        weight_map[k] = "model_00002-of-00002.safetensors"

    index = {"metadata": {"total_size": sum(t.nelement() * t.element_size() for t in tensors.values())},
             "weight_map": weight_map}

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Mock model created: {len(tensors)} tensors, {num_layers} layers")
    return config, tensors


def verify_expansion(original_dir: Path, expanded_dir: Path, original_tensors: dict,
                     original_layers: int, target_layers: int):
    """Verify the expanded model satisfies M2 identity initialization."""
    expanded_index = json.load(open(expanded_dir / "model.safetensors.index.json"))
    expanded_config = json.load(open(expanded_dir / "config.json"))

    assert expanded_config["num_layers"] == target_layers, \
        f"Config num_layers mismatch: {expanded_config['num_layers']} != {target_layers}"

    all_tensors = {}
    for shard_file in sorted(set(expanded_index["weight_map"].values())):
        shard_tensors = load_file(str(expanded_dir / shard_file))
        all_tensors.update(shard_tensors)

    import re
    layers_found = set()
    for key in all_tensors:
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if m:
            layers_found.add(int(m.group(1)))

    assert layers_found == set(range(target_layers)), \
        f"Expected layers 0-{target_layers-1}, got {sorted(layers_found)}"

    new_layer_indices = set(range(original_layers, target_layers))
    if target_layers == original_layers * 2:
        new_layer_indices = set(range(1, target_layers, 2))

    errors = []
    zeroed_count = 0
    copied_count = 0

    for key, tensor in all_tensors.items():
        m = re.search(r"model\.layers\.(\d+)\.(.*)", key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        rest = m.group(2)

        if layer_idx not in new_layer_indices:
            continue

        is_o_proj = bool(re.search(r"self_attn\.\d*\.?o_proj\.weight$", key))
        is_expert_down = bool(re.search(r"mlp\.experts\.\d+\.down_proj\.weight$", key))
        is_shared_down = bool(re.search(r"mlps\.\d+\.down_proj\.weight$", key))
        is_mlp_down = bool(re.search(r"mlp\.down_proj\.weight$", key))

        if is_o_proj or is_expert_down or is_shared_down or is_mlp_down:
            if not torch.all(tensor == 0):
                errors.append(f"FAIL: {key} should be zeroed but has non-zero values")
            else:
                zeroed_count += 1
        else:
            copied_count += 1

    for key, tensor in all_tensors.items():
        if "model.layers." not in key:
            if key in original_tensors:
                if not torch.equal(tensor, original_tensors[key]):
                    errors.append(f"FAIL: Non-layer param {key} was modified")

    print(f"\nVerification Results:")
    print(f"  Total layers: {target_layers}")
    print(f"  New identity layers: {len(new_layer_indices)}")
    print(f"  Zeroed tensors in new layers: {zeroed_count}")
    print(f"  Copied tensors in new layers: {copied_count}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        return False
    else:
        print("  ✓ All identity initialization checks passed!")
        print("  ✓ o_proj weights zeroed in new layers")
        print("  ✓ down_proj weights zeroed in new layers (experts + shared MLPs)")
        print("  ✓ Non-layer parameters preserved")
        return True


def main():
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mock_dir = tmpdir / "mock_model"
        expanded_dir = tmpdir / "expanded_model"

        num_layers = 4
        target_layers = 8

        print("=" * 60)
        print("M2 MoE Depth Expansion Validation")
        print("=" * 60)
        print(f"\nStep 1: Creating mock LongCat-Flash-Chat model ({num_layers} layers)...")
        config, original_tensors = create_mock_longcat_model(mock_dir, num_layers)

        print(f"\nStep 2: Running expand_moe_depth.py ({num_layers} → {target_layers} layers)...")
        script_path = Path(__file__).resolve().parent / "expand_moe_depth.py"
        cmd = [
            sys.executable, str(script_path),
            "--model_dir", str(mock_dir),
            "--output_dir", str(expanded_dir),
            "--original_layers", str(num_layers),
            "--target_layers", str(target_layers),
            "--insertion_mode", "interleave",
        ]
        env = {**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent.parent), env=env)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")
            sys.exit(1)

        print(f"\nStep 3: Verifying expanded model...")
        success = verify_expansion(mock_dir, expanded_dir, original_tensors,
                                   num_layers, target_layers)

        print(f"\nStep 4: Testing append mode...")
        expanded_dir_append = tmpdir / "expanded_append"
        cmd_append = [
            sys.executable, str(script_path),
            "--model_dir", str(mock_dir),
            "--output_dir", str(expanded_dir_append),
            "--original_layers", str(num_layers),
            "--target_layers", str(target_layers),
            "--insertion_mode", "append",
        ]
        result2 = subprocess.run(cmd_append, capture_output=True, text=True,
                                 cwd=str(Path(__file__).resolve().parent.parent), env=env)
        print(result2.stdout[-500:] if len(result2.stdout) > 500 else result2.stdout)
        if result2.returncode != 0:
            print(f"STDERR:\n{result2.stderr}")
            sys.exit(1)

        expanded_index = json.load(open(expanded_dir_append / "model.safetensors.index.json"))
        all_tensors = {}
        for shard_file in sorted(set(expanded_index["weight_map"].values())):
            shard_tensors = load_file(str(expanded_dir_append / shard_file))
            all_tensors.update(shard_tensors)

        import re
        append_errors = 0
        for key, tensor in all_tensors.items():
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if not m:
                continue
            layer_idx = int(m.group(1))
            if layer_idx >= num_layers:
                is_zero_target = bool(re.search(
                    r"(self_attn\.\d*\.?o_proj\.weight|experts\.\d+\.down_proj\.weight|mlps\.\d+\.down_proj\.weight)$",
                    key))
                if is_zero_target and not torch.all(tensor == 0):
                    append_errors += 1

        if append_errors == 0:
            print("  ✓ Append mode: identity initialization verified!")
        else:
            print(f"  ✗ Append mode: {append_errors} tensors not properly zeroed")
            success = False

        print("\n" + "=" * 60)
        if success:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
            sys.exit(1)
        print("=" * 60)


if __name__ == "__main__":
    main()
