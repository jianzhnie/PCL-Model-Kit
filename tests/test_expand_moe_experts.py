"""Pytest tests for expand_moe_experts.py (M1: Expert Upcycling)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_default_model(model_dir: Path):
    """Create default model: 1 layer, 4 experts, router weight+bias."""
    model_dir.mkdir(parents=True)
    torch.manual_seed(42)

    config = {
        "model_type": "longcat",
        "n_routed_experts": 4,
        "hidden_size": 16,
        "expert_ffn_hidden_size": 32,
        "num_layers": 1,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(100, 16),
        "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 16),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
        "model.norm.weight": torch.randn(16),
    }
    for i in range(4):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
        weights[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.full((32, 16), float(i))
        weights[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.full((16, 32), float(i))

    shard1 = {k: v for i, (k, v) in enumerate(weights.items()) if i % 2 == 0}
    shard2 = {k: v for i, (k, v) in enumerate(weights.items()) if i % 2 != 0}
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for k in shard1:
        index["weight_map"][k] = "model-00001-of-00002.safetensors"
    for k in shard2:
        index["weight_map"][k] = "model-00002-of-00002.safetensors"
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return config, weights


def _run_experts(model_dir: Path, output_dir: Path, **extra) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "utils.expand_moe_experts",
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
    ]
    for k, v in extra.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))


def _load_all(expanded_dir: Path) -> dict[str, torch.Tensor]:
    index = json.load(open(expanded_dir / "model.safetensors.index.json"))
    all_w = {}
    for sf in set(index["weight_map"].values()):
        all_w.update(load_file(str(expanded_dir / sf)))
    return all_w


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_model_dir(tmp_path):
    model_dir = tmp_path / "original"
    _make_default_model(model_dir)
    return model_dir


# ═══════════════════════════════════════════════════════════════════════
# Basic expansion tests
# ═══════════════════════════════════════════════════════════════════════

class TestBasicExpansion:
    """4→8 expert expansion on default model."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, default_model_dir):
        self.model_dir = default_model_dir
        self.expanded_dir = tmp_path / "expanded"
        r = _run_experts(self.model_dir, self.expanded_dir, target_experts=8)
        assert r.returncode == 0, f"Expansion failed:\n{r.stderr}"
        self.config = json.load(open(self.expanded_dir / "config.json"))
        self.weights = _load_all(self.expanded_dir)

    def test_config(self):
        assert self.config["n_routed_experts"] == 8

    def test_router_expansion(self):
        rw = self.weights["model.layers.0.mlp.router.classifier.weight"]
        assert rw.shape == (8, 16)

    def test_expert_duplication(self):
        for i in range(4):
            orig = self.weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"]
            copy = self.weights[f"model.layers.0.mlp.experts.{i + 4}.gate_proj.weight"]
            assert torch.equal(copy, orig), f"Expert {i+4} != expert {i}"

    def test_non_expert_preserved(self):
        assert self.weights["model.embed_tokens.weight"].shape == (100, 16)
        assert self.weights["model.norm.weight"].shape == (16,)


def test_default_doubles_experts(tmp_path, default_model_dir):
    """Omitting --target_experts defaults to doubling."""
    expanded_dir = tmp_path / "expanded"
    r = _run_experts(default_model_dir, expanded_dir)
    assert r.returncode == 0
    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 8


def test_invalid_multiple_rejected(tmp_path, default_model_dir):
    """6 is not a multiple of 4."""
    r = _run_experts(default_model_dir, tmp_path / "expanded", target_experts=6)
    assert r.returncode == 1


def test_zero_target_rejected(tmp_path, default_model_dir):
    """--target_experts 0 should be rejected."""
    r = _run_experts(default_model_dir, tmp_path / "expanded", target_experts=0)
    assert r.returncode == 1


# ═══════════════════════════════════════════════════════════════════════
# Gate router + n_experts key
# ═══════════════════════════════════════════════════════════════════════

def test_gate_router_and_n_experts_key(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_experts": 4, "hidden_size": 16,
              "expert_ffn_hidden_size": 32, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(100, 16),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 16),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(4),
        "model.norm.weight": torch.randn(16),
    }
    for i in range(4):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

    shard1 = {k: v for i, (k, v) in enumerate(weights.items()) if i % 2 == 0}
    shard2 = {k: v for i, (k, v) in enumerate(weights.items()) if i % 2 != 0}
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for k in shard1:
        index["weight_map"][k] = "model-00001-of-00002.safetensors"
    for k in shard2:
        index["weight_map"][k] = "model-00002-of-00002.safetensors"
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=8)
    assert r.returncode == 0

    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_experts"] == 8

    all_w = _load_all(expanded_dir)
    assert all_w["model.layers.0.mlp.gate.weight"].shape == (8, 16)
    assert all_w["model.layers.0.mlp.gate.e_score_correction_bias"].shape == (8,)
    for i in range(4):
        assert torch.equal(
            all_w[f"model.layers.0.mlp.experts.{i + 4}.gate_proj.weight"],
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"])


# ═══════════════════════════════════════════════════════════════════════
# Top-K
# ═══════════════════════════════════════════════════════════════════════

def test_target_topk_updates_config(tmp_path, default_model_dir):
    expanded_dir = tmp_path / "expanded"
    r = _run_experts(default_model_dir, expanded_dir, target_experts=8, target_topk=24)
    assert r.returncode == 0
    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 8
    assert config["moe_topk"] == 24


def test_target_topk_adds_key_when_not_present(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 4, "hidden_size": 16, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(100, 16),
        "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 16),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
        "model.norm.weight": torch.randn(16),
    }
    for i in range(4):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=8, target_topk=16)
    assert r.returncode == 0
    config = json.load(open(expanded_dir / "config.json"))
    assert config["moe_topk"] == 16


# ═══════════════════════════════════════════════════════════════════════
# Zero experts
# ═══════════════════════════════════════════════════════════════════════

def test_expansion_with_zero_experts(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 4, "zero_expert_num": 2,
              "hidden_size": 16, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(100, 16),
        "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
        "model.norm.weight": torch.randn(16),
    }
    for i in range(4):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=8)
    assert r.returncode == 0

    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 8
    assert config["zero_expert_num"] == 4

    all_w = _load_all(expanded_dir)
    new_router = all_w["model.layers.0.mlp.router.classifier.weight"]
    assert new_router.shape == (12, 16)
    orig_router = weights["model.layers.0.mlp.router.classifier.weight"]
    torch.testing.assert_close(new_router[:4], orig_router[:4])
    torch.testing.assert_close(new_router[4:8], orig_router[:4])
    torch.testing.assert_close(new_router[8:10], orig_router[4:])
    torch.testing.assert_close(new_router[10:12], orig_router[4:])


def test_expansion_with_zero_expert_parameters(tmp_path):
    """Zero experts have their own weight slots in the model."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 4, "zero_expert_num": 2,
              "hidden_size": 16, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(100, 16),
        "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
        "model.norm.weight": torch.randn(16),
    }
    for i in range(6):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=8)
    assert r.returncode == 0

    config = json.load(open(expanded_dir / "config.json"))
    assert config["zero_expert_num"] == 4

    all_w = _load_all(expanded_dir)
    # Zero experts remapped: orig 4→8, orig 5→9, copies at 10, 11
    for old_idx, new_idx in [(4, 8), (5, 9)]:
        assert torch.equal(
            all_w[f"model.layers.0.mlp.experts.{new_idx}.gate_proj.weight"],
            weights[f"model.layers.0.mlp.experts.{old_idx}.gate_proj.weight"])
    for old_idx, copy_idx in [(4, 10), (5, 11)]:
        assert torch.equal(
            all_w[f"model.layers.0.mlp.experts.{copy_idx}.gate_proj.weight"],
            weights[f"model.layers.0.mlp.experts.{old_idx}.gate_proj.weight"])


# ═══════════════════════════════════════════════════════════════════════
# 3x expansion, multi-layer, shard sizes
# ═══════════════════════════════════════════════════════════════════════

def test_expansion_factor_greater_than_two(tmp_path):
    """4→12 experts (3x factor)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 4, "hidden_size": 8, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(50, 8),
        "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 8),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
        "model.norm.weight": torch.randn(8),
    }
    for i in range(4):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((16, 8), float(i))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=12)
    assert r.returncode == 0

    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 12

    all_w = _load_all(expanded_dir)
    assert all_w["model.layers.0.mlp.router.classifier.weight"].shape == (12, 8)
    orig_router = weights["model.layers.0.mlp.router.classifier.weight"]
    for factor in range(3):
        part = all_w["model.layers.0.mlp.router.classifier.weight"][factor * 4:(factor + 1) * 4]
        assert torch.equal(part, orig_router)

    for new_idx in range(4, 12):
        assert torch.equal(
            all_w[f"model.layers.0.mlp.experts.{new_idx}.gate_proj.weight"],
            weights[f"model.layers.0.mlp.experts.{new_idx % 4}.gate_proj.weight"])


def test_multi_layer_moe_expansion(tmp_path):
    """2 MoE layers, 2→4 experts each."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 2, "hidden_size": 8, "num_layers": 2}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.embed_tokens.weight": torch.randn(50, 8),
        "model.norm.weight": torch.randn(8),
    }
    for li in range(2):
        weights[f"model.layers.{li}.mlp.router.classifier.weight"] = torch.randn(2, 8)
        weights[f"model.layers.{li}.mlp.router.e_score_correction_bias"] = torch.randn(2)
        for ei in range(2):
            weights[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"] = \
                torch.full((16, 8), float(li * 10 + ei))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=4)
    assert r.returncode == 0

    all_w = _load_all(expanded_dir)
    for li in range(2):
        assert all_w[f"model.layers.{li}.mlp.router.classifier.weight"].shape == (4, 8)
        for ei in range(4):
            src_ei = ei % 2
            assert torch.equal(
                all_w[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"],
                weights[f"model.layers.{li}.mlp.experts.{src_ei}.gate_proj.weight"])

    assert torch.equal(all_w["model.embed_tokens.weight"], weights["model.embed_tokens.weight"])
    assert torch.equal(all_w["model.norm.weight"], weights["model.norm.weight"])


# ═══════════════════════════════════════════════════════════════════════
# Parallel vs Serial
# ═══════════════════════════════════════════════════════════════════════

def test_parallel_produces_same_output_as_serial(tmp_path, default_model_dir):
    out_serial = tmp_path / "serial"
    out_parallel = tmp_path / "parallel"

    r1 = _run_experts(default_model_dir, out_serial, target_experts=8, workers=1)
    assert r1.returncode == 0
    r2 = _run_experts(default_model_dir, out_parallel, target_experts=8, workers=2)
    assert r2.returncode == 0

    s_idx = json.load(open(out_serial / "model.safetensors.index.json"))
    p_idx = json.load(open(out_parallel / "model.safetensors.index.json"))
    assert sorted(s_idx["weight_map"].keys()) == sorted(p_idx["weight_map"].keys())

    s_cfg = json.load(open(out_serial / "config.json"))
    p_cfg = json.load(open(out_parallel / "config.json"))
    assert s_cfg == p_cfg

    s_w = {}
    for sf in set(s_idx["weight_map"].values()):
        s_w.update(load_file(str(out_serial / sf)))
    p_w = {}
    for sf in set(p_idx["weight_map"].values()):
        p_w.update(load_file(str(out_parallel / sf)))
    for key in s_w:
        assert torch.equal(s_w[key], p_w[key]), f"Mismatch: {key}"


def test_parallel_with_noise_and_zero_experts(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.manual_seed(42)

    config = {"model_type": "longcat", "n_routed_experts": 4, "zero_expert_num": 2,
              "hidden_size": 16, "num_layers": 1}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    weights = {
        "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
        "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
    }
    for i in range(6):
        weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
    save_file(weights, str(model_dir / "model.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: "model.safetensors" for k in weights}}
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    expanded_dir = tmp_path / "expanded"
    r = _run_experts(model_dir, expanded_dir, target_experts=8,
                     **{"router-noise-scale": "1e-6"}, workers=3)
    assert r.returncode == 0

    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 8
    assert config["zero_expert_num"] == 4

    all_w = _load_all(expanded_dir)
    assert all_w["model.layers.0.mlp.router.classifier.weight"].shape == (12, 16)
    assert torch.equal(
        all_w["model.layers.0.mlp.experts.0.gate_proj.weight"],
        all_w["model.layers.0.mlp.experts.4.gate_proj.weight"])


# ═══════════════════════════════════════════════════════════════════════
# Shell script tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (REPO_ROOT / "scripts" / "expand_moe_experts.sh").exists(),
    reason="Shell script not found")
def test_shell_script_default_doubles(tmp_path, default_model_dir):
    script = REPO_ROOT / "scripts" / "expand_moe_experts.sh"
    expanded_dir = tmp_path / "expanded"
    r = subprocess.run(
        ["bash", str(script)],
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ,
             "MODEL_DIR": str(default_model_dir),
             "OUTPUT_DIR": str(expanded_dir)},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 8


@pytest.mark.skipif(
    not (REPO_ROOT / "scripts" / "expand_moe_experts.sh").exists(),
    reason="Shell script not found")
def test_shell_script_with_explicit_target(tmp_path, default_model_dir):
    script = REPO_ROOT / "scripts" / "expand_moe_experts.sh"
    expanded_dir = tmp_path / "expanded"
    r = subprocess.run(
        ["bash", str(script), "12", "24"],
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ,
             "MODEL_DIR": str(default_model_dir),
             "OUTPUT_DIR": str(expanded_dir)},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    config = json.load(open(expanded_dir / "config.json"))
    assert config["n_routed_experts"] == 12
    assert config["moe_topk"] == 24
