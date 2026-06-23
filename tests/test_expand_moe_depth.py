"""Pytest tests for expand_moe_depth.py (M2: MoE Depth Expansion)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "utils" / "expand_moe_depth.py"

sys.path.insert(0, str(REPO_ROOT))
from utils.expand_moe_depth import build_layer_mapping, should_zero


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _create_mock_model(model_dir: Path, num_layers: int = 4):
    """Create a tiny LongCat-like model with MoE structure."""
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
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_proj_dim = kv_lora_rank + qk_rope_head_dim

    config = {
        "architectures": ["LongcatFlashForCausalLM"],
        "vocab_size": 128,
        "hidden_size": hidden,
        "ffn_hidden_size": ffn_hidden,
        "expert_ffn_hidden_size": expert_ffn,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "v_head_dim": v_head_dim,
        "n_routed_experts": n_routed,
        "zero_expert_num": zero_expert_num,
        "zero_expert_type": "identity",
        "moe_topk": 2,
    }

    torch.manual_seed(42)
    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.randn(128, hidden)
    tensors["lm_head.weight"] = torch.randn(128, hidden)
    tensors["model.norm.weight"] = torch.ones(hidden)

    for layer_idx in range(num_layers):
        p = f"model.layers.{layer_idx}"
        tensors[f"{p}.input_layernorm.0.weight"] = torch.ones(hidden)
        tensors[f"{p}.input_layernorm.1.weight"] = torch.ones(hidden)
        tensors[f"{p}.post_attention_layernorm.0.weight"] = torch.ones(hidden)
        tensors[f"{p}.post_attention_layernorm.1.weight"] = torch.ones(hidden)

        for attn_idx in range(2):
            ap = f"{p}.self_attn.{attn_idx}"
            tensors[f"{ap}.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden)
            tensors[f"{ap}.q_a_layernorm.weight"] = torch.ones(q_lora_rank)
            tensors[f"{ap}.q_b_proj.weight"] = torch.randn(num_heads * head_dim, q_lora_rank)
            tensors[f"{ap}.kv_a_proj_with_mqa.weight"] = torch.randn(kv_proj_dim, hidden)
            tensors[f"{ap}.kv_a_layernorm.weight"] = torch.ones(kv_lora_rank)
            tensors[f"{ap}.kv_b_proj.weight"] = torch.randn(
                num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
            tensors[f"{ap}.o_proj.weight"] = torch.randn(hidden, num_heads * v_head_dim)

        tensors[f"{p}.mlp.router.classifier.weight"] = torch.randn(total_experts, hidden)
        tensors[f"{p}.mlp.router.e_score_correction_bias"] = torch.randn(total_experts)

        for exp_idx in range(total_experts):
            ep = f"{p}.mlp.experts.{exp_idx}"
            tensors[f"{ep}.gate_proj.weight"] = torch.randn(expert_ffn, hidden)
            tensors[f"{ep}.up_proj.weight"] = torch.randn(expert_ffn, hidden)
            tensors[f"{ep}.down_proj.weight"] = torch.randn(hidden, expert_ffn)

        for mlp_idx in range(2):
            mp = f"{p}.mlps.{mlp_idx}"
            tensors[f"{mp}.gate_proj.weight"] = torch.randn(ffn_hidden, hidden)
            tensors[f"{mp}.up_proj.weight"] = torch.randn(ffn_hidden, hidden)
            tensors[f"{mp}.down_proj.weight"] = torch.randn(hidden, ffn_hidden)

    model_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(tensors.keys())
    mid = len(keys) // 2
    shard1 = {k: tensors[k] for k in keys[:mid]}
    shard2 = {k: tensors[k] for k in keys[mid:]}
    save_file(shard1, str(model_dir / "model_00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model_00002-of-00002.safetensors"))

    weight_map = {}
    for k in shard1:
        weight_map[k] = "model_00001-of-00002.safetensors"
    for k in shard2:
        weight_map[k] = "model_00002-of-00002.safetensors"

    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in tensors.values())},
        "weight_map": weight_map,
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config, tensors


def _run_expansion(model_dir: Path, output_dir: Path, num_layers: int,
                   target_layers: int, mode: str, **extra) -> subprocess.CompletedProcess:
    """Run expand_moe_depth.py via subprocess."""
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--original_layers", str(num_layers),
        "--target_layers", str(target_layers),
        "--insertion_mode", mode,
    ]
    for k, v in extra.items():
        cmd.extend([f"--{k}", str(v)])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))


def _load_all_tensors(expanded_dir: Path) -> dict[str, torch.Tensor]:
    index = json.load(open(expanded_dir / "model.safetensors.index.json"))
    all_tensors = {}
    for sf in sorted(set(index["weight_map"].values())):
        all_tensors.update(load_file(str(expanded_dir / sf)))
    return all_tensors


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock 4-layer LongCat model."""
    model_dir = tmp_path / "original"
    _create_mock_model(model_dir, num_layers=4)
    return model_dir


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════

def test_build_layer_mapping():
    mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "interleave")
    expected = [(0, False), (0, True), (1, False), (1, True),
                (2, False), (2, True), (3, False), (3, True)]
    assert mapping == expected

    mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "append")
    expected = [(0, False), (1, False), (2, False), (3, False),
                (0, True), (1, True), (2, True), (3, True)]
    assert mapping == expected

    mapping = build_layer_mapping(4, 12, [0, 1, 2, 3, 0, 1, 2, 3], "interleave")
    expected = [(0, False), (0, True), (0, True),
                (1, False), (1, True), (1, True),
                (2, False), (2, True), (2, True),
                (3, False), (3, True), (3, True)]
    assert mapping == expected

    mapping = build_layer_mapping(4, 6, [1, 2], "interleave")
    expected = [(0, False), (1, False), (1, True), (2, False), (2, True), (3, False)]
    assert mapping == expected


def test_should_zero_patterns():
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
        assert should_zero(name), f"should_zero({name}) should be True"
    for name in must_keep:
        assert not should_zero(name), f"should_zero({name}) should be False"


def test_forward_pass_identity():
    """o_proj=0 and down_proj=0 → Layer(x) = x via residual."""
    hidden, seq_len, batch = 64, 8, 2
    x = torch.randn(batch, seq_len, hidden)

    attn_output = torch.randn(batch, seq_len, 32) @ torch.zeros(hidden, 32).T
    assert torch.equal(x + attn_output, x), "Attn residual with zero o_proj not identity"

    mlp_output = torch.randn(batch, seq_len, 128) @ torch.zeros(hidden, 128).T
    assert torch.equal(x + mlp_output, x), "MLP residual with zero down_proj not identity"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — Interleave Mode
# ═══════════════════════════════════════════════════════════════════════

class TestDepthExpansionInterleave:
    """4→8 layers, interleave mode."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_model_dir):
        self.model_dir = mock_model_dir
        self.expanded_dir = tmp_path / "expanded"
        self.num_layers = 4
        self.target_layers = 8
        r = _run_expansion(self.model_dir, self.expanded_dir, self.num_layers,
                           self.target_layers, "interleave")
        assert r.returncode == 0, f"Expansion failed:\n{r.stderr}"
        self.tensors = _load_all_tensors(self.expanded_dir)

    def test_config_and_index(self):
        config = json.load(open(self.expanded_dir / "config.json"))
        assert config.get("num_layers") == self.target_layers

        index = json.load(open(self.expanded_dir / "model.safetensors.index.json"))
        layers = {int(re.search(r"model\.layers\.(\d+)\.", k).group(1))
                  for k in index["weight_map"] if "model.layers." in k}
        assert layers == set(range(self.target_layers))

        for sf in set(index["weight_map"].values()):
            assert (self.expanded_dir / sf).exists(), f"Missing shard: {sf}"

    def test_identity_initialization(self):
        source_list = [i % self.num_layers
                       for i in range(self.target_layers - self.num_layers)]
        layer_mapping = build_layer_mapping(
            self.num_layers, self.target_layers, source_list, "interleave")
        new_layer_set = {i for i, (_, is_new) in enumerate(layer_mapping) if is_new}

        zeroed = 0
        for key, tensor in self.tensors.items():
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if not m or int(m.group(1)) not in new_layer_set:
                continue
            if should_zero(key):
                assert torch.all(tensor == 0), f"{key}: expected all zeros"
                zeroed += 1
        assert zeroed > 0, "No tensors were zeroed"

    def test_weight_copying(self):
        original_tensors = {}
        for sf in ["model_00001-of-00002.safetensors", "model_00002-of-00002.safetensors"]:
            original_tensors.update(load_file(str(self.model_dir / sf)))

        source_list = [i % self.num_layers
                       for i in range(self.target_layers - self.num_layers)]
        layer_mapping = build_layer_mapping(
            self.num_layers, self.target_layers, source_list, "interleave")

        for new_idx, (src, is_new) in enumerate(layer_mapping):
            if not is_new:
                continue
            for key, tensor in self.tensors.items():
                m = re.search(r"model\.layers\.(\d+)\.(.*)", key)
                if not m or int(m.group(1)) != new_idx:
                    continue
                if should_zero(key):
                    continue
                src_key = f"model.layers.{src}.{m.group(2)}"
                assert src_key in original_tensors, f"Source key missing: {src_key}"
                assert torch.equal(tensor, original_tensors[src_key]), \
                    f"{key} differs from source {src_key}"

    def test_original_layers_preserved(self):
        original_tensors = {}
        for sf in ["model_00001-of-00002.safetensors", "model_00002-of-00002.safetensors"]:
            original_tensors.update(load_file(str(self.model_dir / sf)))

        source_list = [i % self.num_layers
                       for i in range(self.target_layers - self.num_layers)]
        layer_mapping = build_layer_mapping(
            self.num_layers, self.target_layers, source_list, "interleave")
        remap = {src: idx for idx, (src, is_new) in enumerate(layer_mapping)
                 if not is_new}

        for orig_idx, new_idx in remap.items():
            orig_prefix = f"model.layers.{orig_idx}."
            new_prefix = f"model.layers.{new_idx}."
            for orig_key, orig_val in original_tensors.items():
                if not orig_key.startswith(orig_prefix):
                    continue
                new_key = orig_key.replace(orig_prefix, new_prefix, 1)
                assert new_key in self.tensors, f"Missing: {new_key}"
                assert torch.equal(self.tensors[new_key], orig_val), \
                    f"Modified: {new_key}"

    def test_non_layer_params(self):
        original_tensors = {}
        for sf in ["model_00001-of-00002.safetensors", "model_00002-of-00002.safetensors"]:
            original_tensors.update(load_file(str(self.model_dir / sf)))

        for key, orig_val in original_tensors.items():
            if "model.layers." in key:
                continue
            assert key in self.tensors, f"Missing non-layer: {key}"
            assert torch.equal(self.tensors[key], orig_val), f"Modified: {key}"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — Append Mode
# ═══════════════════════════════════════════════════════════════════════

class TestDepthExpansionAppend:
    """4→8 layers, append mode."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_model_dir):
        self.model_dir = mock_model_dir
        self.expanded_dir = tmp_path / "expanded"
        r = _run_expansion(self.model_dir, self.expanded_dir, 4, 8, "append")
        assert r.returncode == 0, f"Expansion failed:\n{r.stderr}"
        self.tensors = _load_all_tensors(self.expanded_dir)

    def test_config(self):
        config = json.load(open(self.expanded_dir / "config.json"))
        assert config.get("num_layers") == 8

    def test_layer_count(self):
        index = json.load(open(self.expanded_dir / "model.safetensors.index.json"))
        layers = {int(re.search(r"model\.layers\.(\d+)\.", k).group(1))
                  for k in index["weight_map"] if "model.layers." in k}
        assert layers == set(range(8))

    def test_new_layers_zeroed(self):
        source_list = [i % 4 for i in range(4)]
        layer_mapping = build_layer_mapping(4, 8, source_list, "append")
        new_layer_set = {i for i, (_, is_new) in enumerate(layer_mapping) if is_new}

        for key, tensor in self.tensors.items():
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if not m or int(m.group(1)) not in new_layer_set:
                continue
            if should_zero(key):
                assert torch.all(tensor == 0), f"{key}: not zeroed"


# ═══════════════════════════════════════════════════════════════════════
# Edge Case
# ═══════════════════════════════════════════════════════════════════════

def test_non_uniform_interleave(tmp_path, mock_model_dir):
    """4→6 layers with custom copy_source."""
    expanded_dir = tmp_path / "expanded_6"
    r = _run_expansion(mock_model_dir, expanded_dir, 4, 6, "interleave",
                       copy_source="1,2")
    assert r.returncode == 0, f"Expansion failed:\n{r.stderr}"

    tensors = _load_all_tensors(expanded_dir)
    layers = {int(re.search(r"model\.layers\.(\d+)\.", k).group(1))
              for k in tensors if "model.layers." in k}
    assert layers == set(range(6))

    source_list = [1, 2]
    layer_mapping = build_layer_mapping(4, 6, source_list, "interleave")
    new_layer_set = {i for i, (_, is_new) in enumerate(layer_mapping) if is_new}

    for key, tensor in tensors.items():
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if not m or int(m.group(1)) not in new_layer_set:
            continue
        if should_zero(key):
            assert torch.all(tensor == 0), f"{key}: not zeroed"