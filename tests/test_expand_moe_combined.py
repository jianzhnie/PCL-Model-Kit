"""Pytest tests for expand_moe_combined.py (M2 Depth + M1 Expert Upcycling)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
from utils.expand_moe_combined import (
    build_expert_target_map,
    build_layer_mapping,
    expand_router_weight,
    expand_tensor,
    expand_tensor_meta,
    get_expert_info,
    get_layer_index,
    should_zero,
)
from utils.shared import make_expert_key


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_fake_model(model_dir: Path, num_layers: int = 2, num_experts: int = 4,
                     hidden_size: int = 16, intermediate_size: int = 32,
                     vocab_size: int = 100, zero_expert_num: int = 0):
    """Create a synthetic safetensors MoE model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    total_experts = num_experts + zero_expert_num

    torch.manual_seed(42)
    tensors: dict[str, torch.Tensor] = {}
    tensors["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    tensors["model.norm.weight"] = torch.randn(hidden_size)
    tensors["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    for layer_idx in range(num_layers):
        p = f"model.layers.{layer_idx}"
        tensors[f"{p}.input_layernorm.weight"] = torch.randn(hidden_size)
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        tensors[f"{p}.mlp.gate.weight"] = torch.randn(total_experts, hidden_size)

        for ei in range(num_experts):
            ep = f"{p}.mlp.experts.{ei}"
            tensors[f"{ep}.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
            tensors[f"{ep}.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
            tensors[f"{ep}.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

        for ei in range(num_experts, total_experts):
            ep = f"{p}.mlp.experts.{ei}"
            tensors[f"{ep}.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
            tensors[f"{ep}.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
            tensors[f"{ep}.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

        tensors[f"{p}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    index = {
        "metadata": {"total_size": sum(t.element_size() * t.nelement() for t in tensors.values())},
        "weight_map": weight_map,
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    config = {
        "num_hidden_layers": num_layers,
        "n_routed_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "moe_topk": 2,
    }
    if zero_expert_num > 0:
        config["zero_expert_num"] = zero_expert_num
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config, index, tensors


def _run_combined(model_dir: Path, output_dir: Path, **extra) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "utils.expand_moe_combined",
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
    ]
    for k, v in extra.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_model_dir(tmp_path):
    """2 layers, 4 experts."""
    model_dir = tmp_path / "original"
    _make_fake_model(model_dir, num_layers=2, num_experts=4)
    return model_dir


@pytest.fixture
def fake_model_dir_ze(tmp_path):
    """2 layers, 4 experts + 2 zero experts."""
    model_dir = tmp_path / "original_ze"
    _make_fake_model(model_dir, num_layers=2, num_experts=4, zero_expert_num=2)
    return model_dir


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════

def test_build_layer_mapping():
    mapping = build_layer_mapping(2, 6, [0, 0, 1, 1], "interleave")
    assert mapping == [
        (0, False), (0, True), (0, True),
        (1, False), (1, True), (1, True),
    ]

    mapping = build_layer_mapping(2, 6, [0, 0, 1, 1], "append")
    assert mapping == [
        (0, False), (1, False),
        (0, True), (0, True), (1, True), (1, True),
    ]


def test_build_expert_target_map():
    assert build_expert_target_map(4, 8) == {0: [4], 1: [5], 2: [6], 3: [7]}
    assert build_expert_target_map(4, 12) == {0: [4, 8], 1: [5, 9], 2: [6, 10], 3: [7, 11]}


def test_expand_router_weight():
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    expanded = expand_router_weight(t, 4, 0, 2, 0.0)
    assert expanded.shape == (8, 4)
    assert torch.equal(expanded[:4], t)
    assert torch.equal(expanded[4:], t)

    expanded_n = expand_router_weight(t, 4, 0, 2, 1e-3)
    assert torch.equal(expanded_n[:4], t)
    assert not torch.equal(expanded_n[4:], t), "should have noise"

    t_ze = torch.cat([t, torch.zeros(2, 4)], dim=0)
    expanded_ze = expand_router_weight(t_ze, 4, 2, 2, 0.0)
    assert expanded_ze.shape == (12, 4)
    assert torch.equal(expanded_ze[8:], torch.zeros(4, 4))


class TestExpandTensor:
    """Unit tests for the core expand_tensor function."""

    @pytest.fixture
    def mappings(self):
        return dict(
            remap={0: 0, 1: 3},
            orig_to_new={0: [1, 2], 1: [4, 5]},
            new_layer_set={1, 2, 4, 5},
            expert_targets={0: [4], 1: [5], 2: [6], 3: [7]},
        )

    def common(self, mappings):
        return dict(**mappings, original_experts=4, zero_expert_num=0,
                    expansion_factor=2, target_experts=8,
                    router_noise_scale=0.0, expert_noise_scale=0.0)

    def test_non_layer(self, mappings):
        t = torch.randn(100, 16)
        result = expand_tensor("model.embed_tokens.weight", t, **self.common(mappings))
        assert len(result) == 1
        assert torch.equal(result["model.embed_tokens.weight"], t)

    def test_router_weight(self, mappings):
        t = torch.randn(4, 16)
        result = expand_tensor("model.layers.0.mlp.gate.weight", t, **self.common(mappings))
        assert len(result) == 3
        for key, val in result.items():
            assert val.shape == (8, 16)
            assert torch.equal(val[:4], t)

    def test_expert_gate_not_zeroed(self, mappings):
        t = torch.randn(32, 16)
        result = expand_tensor("model.layers.0.mlp.experts.0.gate_proj.weight",
                               t, **self.common(mappings))
        assert len(result) == 6
        for key, val in result.items():
            info = get_expert_info(key)
            layer_idx = info[0]
            if layer_idx in mappings["new_layer_set"]:
                assert val.sum() != 0, f"gate_proj zeroed: {key}"

    def test_expert_down_proj_zeroed(self, mappings):
        t = torch.randn(16, 32)
        result = expand_tensor("model.layers.0.mlp.experts.0.down_proj.weight",
                               t, **self.common(mappings))
        assert len(result) == 6
        for key, val in result.items():
            info = get_expert_info(key)
            layer_idx = info[0]
            if layer_idx in mappings["new_layer_set"]:
                assert val.sum() == 0, f"down_proj NOT zeroed: {key}"
            else:
                assert val.sum() != 0, f"down_proj zeroed in original layer: {key}"

    def test_o_proj_zeroed(self, mappings):
        t = torch.randn(16, 16)
        result = expand_tensor("model.layers.0.self_attn.o_proj.weight",
                               t, **self.common(mappings))
        assert len(result) == 3
        for key, val in result.items():
            layer_idx = get_layer_index(key)
            if layer_idx in mappings["new_layer_set"]:
                assert val.sum() == 0
            else:
                assert val.sum() != 0

    def test_layernorm_preserved(self, mappings):
        t = torch.randn(16)
        result = expand_tensor("model.layers.0.input_layernorm.weight",
                               t, **self.common(mappings))
        assert len(result) == 3
        for val in result.values():
            assert val.sum() != 0


class TestExpandTensorMeta:
    """Unit tests for metadata-level planning."""

    @pytest.fixture
    def mappings(self):
        return dict(
            remap={0: 0, 1: 3},
            orig_to_new={0: [1, 2], 1: [4, 5]},
            new_layer_set={1, 2, 4, 5},
            expert_targets={0: [4], 1: [5], 2: [6], 3: [7]},
        )

    def common(self, mappings, **extra):
        return dict(**mappings, zero_expert_num=0, expansion_factor=2,
                    target_experts=8, **extra)

    def test_router(self, mappings):
        results = expand_tensor_meta(
            "model.layers.0.mlp.gate.weight", "F32", [4, 16],
            **self.common(mappings))
        assert len(results) == 3
        assert all(a == "router_weight" for _, _, a in results)

    def test_expert_gate_not_zeroed(self, mappings):
        results = expand_tensor_meta(
            "model.layers.0.mlp.experts.0.gate_proj.weight", "F32", [32, 16],
            **self.common(mappings))
        assert len(results) == 6
        for out_key, _, action in results:
            layer_idx = get_layer_index(out_key)
            if layer_idx in mappings["new_layer_set"]:
                assert action == "clone", f"Expected clone, got {action}"

    def test_expert_down_zero_action(self, mappings):
        results = expand_tensor_meta(
            "model.layers.0.mlp.experts.0.down_proj.weight", "F32", [16, 32],
            **self.common(mappings))
        for out_key, _, action in results:
            layer_idx = get_layer_index(out_key)
            if layer_idx in mappings["new_layer_set"]:
                assert action == "zero", f"Expected zero, got {action}"

    def test_clone_expert_tag_with_noise(self, mappings):
        results = expand_tensor_meta(
            "model.layers.0.mlp.experts.0.gate_proj.weight", "F32", [32, 16],
            **self.common(mappings, expert_noise_scale=1e-3))
        clone_expert_count = sum(1 for _, _, a in results if a == "clone_expert")
        assert clone_expert_count > 0


# ═══════════════════════════════════════════════════════════════════════
# End-to-End Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """2→6 layers, 4→8 experts."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fake_model_dir):
        self.expanded_dir = tmp_path / "expanded"
        r = _run_combined(fake_model_dir, self.expanded_dir,
                          target_layers=6, target_experts=8)
        assert r.returncode == 0, f"Expansion failed:\n{r.stderr}\n{r.stdout}"

        self.out_config = json.load(open(self.expanded_dir / "config.json"))
        self.out_index = json.load(open(self.expanded_dir / "model.safetensors.index.json"))
        self.all_keys = set(self.out_index["weight_map"].keys())
        self.out_tensors = {}
        for sf in set(self.out_index["weight_map"].values()):
            self.out_tensors.update(load_file(str(self.expanded_dir / sf)))

        # Load originals for comparison
        self.orig_tensors = {}
        orig_index = json.load(open(fake_model_dir / "model.safetensors.index.json"))
        for sf in set(orig_index["weight_map"].values()):
            self.orig_tensors.update(load_file(str(fake_model_dir / sf)))

    def test_config(self):
        assert self.out_config["num_hidden_layers"] == 6
        assert self.out_config["n_routed_experts"] == 8

    def test_layers(self):
        layers = sorted({get_layer_index(k) for k in self.all_keys
                         if get_layer_index(k) is not None})
        assert layers == list(range(6))

    def test_expert_count_per_layer(self):
        for li in range(6):
            expert_indices = sorted({
                get_expert_info(k)[1] for k in self.all_keys
                if get_expert_info(k) is not None and get_expert_info(k)[0] == li
            })
            assert expert_indices == list(range(8)), f"Layer {li}: {expert_indices[:4]}..."

    def test_identity_layers_zeroed(self):
        identity_layers = {1, 2, 4, 5}
        for key, tensor in self.out_tensors.items():
            li = get_layer_index(key)
            if li is not None and li in identity_layers and should_zero(key):
                assert tensor.sum() == 0, f"{key} not zeroed"

    def test_router_expansion(self):
        for li in range(6):
            rk = f"model.layers.{li}.mlp.gate.weight"
            if rk in self.out_tensors:
                assert self.out_tensors[rk].shape[0] == 8

    def test_non_layer_params(self):
        assert torch.equal(self.out_tensors["model.embed_tokens.weight"],
                           self.orig_tensors["model.embed_tokens.weight"])
        assert torch.equal(self.out_tensors["model.norm.weight"],
                           self.orig_tensors["model.norm.weight"])

    def test_expert_duplication(self):
        for ei in range(4):
            orig_key = f"model.layers.0.mlp.experts.{ei}.gate_proj.weight"
            out_key = f"model.layers.0.mlp.experts.{ei}.gate_proj.weight"
            assert torch.equal(self.out_tensors[out_key], self.orig_tensors[orig_key])
            out_copy = f"model.layers.0.mlp.experts.{ei + 4}.gate_proj.weight"
            assert torch.equal(self.out_tensors[out_copy], self.orig_tensors[orig_key])

    def test_layer_remapping(self):
        for ei in range(4):
            orig_key = f"model.layers.1.mlp.experts.{ei}.gate_proj.weight"
            out_key = f"model.layers.3.mlp.experts.{ei}.gate_proj.weight"
            assert torch.equal(self.out_tensors[out_key], self.orig_tensors[orig_key])

    def test_identity_gate_weights_intact(self):
        identity_layers = {1, 2, 4, 5}
        for id_layer in identity_layers:
            gate_keys = [k for k in self.out_tensors
                         if f"model.layers.{id_layer}.mlp.experts." in k
                         and "gate_proj" in k]
            for gk in gate_keys[:2]:
                assert self.out_tensors[gk].sum() != 0


class TestEndToEndZeroExperts:
    """2→6 layers, 4→8 experts + 2→4 zero experts."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fake_model_dir_ze):
        self.expanded_dir = tmp_path / "expanded_ze"
        r = _run_combined(fake_model_dir_ze, self.expanded_dir,
                          target_layers=6, target_experts=8)
        assert r.returncode == 0, f"Expansion failed:\n{r.stderr}"

        self.out_config = json.load(open(self.expanded_dir / "config.json"))
        self.out_index = json.load(open(self.expanded_dir / "model.safetensors.index.json"))
        self.all_keys = set(self.out_index["weight_map"].keys())
        self.out_tensors = {}
        for sf in set(self.out_index["weight_map"].values()):
            self.out_tensors.update(load_file(str(self.expanded_dir / sf)))

    def test_config_zero_experts(self):
        assert self.out_config["num_hidden_layers"] == 6
        assert self.out_config["n_routed_experts"] == 8
        assert self.out_config["zero_expert_num"] == 4

    def test_expert_count_with_zeros(self):
        for li in range(6):
            expert_indices = sorted({
                get_expert_info(k)[1] for k in self.all_keys
                if get_expert_info(k) is not None and get_expert_info(k)[0] == li
            })
            assert expert_indices == list(range(12)), \
                f"Layer {li}: expected 0-11, got {min(expert_indices)}-{max(expert_indices)}"

    def test_router_with_zeros(self):
        for li in range(6):
            rk = f"model.layers.{li}.mlp.gate.weight"
            assert rk in self.out_tensors
            assert self.out_tensors[rk].shape[0] == 12  # 8 + 4

    def test_identity_zeroing(self):
        identity_layers = {1, 2, 4, 5}
        for key, tensor in self.out_tensors.items():
            li = get_layer_index(key)
            if li is not None and li in identity_layers and should_zero(key):
                assert tensor.sum() == 0