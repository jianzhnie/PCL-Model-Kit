#!/usr/bin/env python3
"""
End-to-end alignment tests: verify expanded model can be correctly loaded
by HuggingFace with strict key matching, shape verification, and value
preservation checks.
"""

import json
import shutil
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Patched model directory (fixed relative imports)
sys.path.insert(0, "/tmp/longcat_model")

from configuration_longcat_flash import LongcatFlashConfig
from modeling_longcat_flash import LongcatFlashForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.expand_model_layers import main as expand_layers_main
from utils.expand_moe_experts import main as expand_experts_main


def make_tiny_config(**overrides):
    """Create a tiny config that mirrors the real LongCat architecture."""
    defaults = dict(
        vocab_size=320,
        hidden_size=32,
        ffn_hidden_size=64,
        expert_ffn_hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=8,
        n_routed_experts=4,
        zero_expert_num=2,
        zero_expert_type="identity",
        moe_topk=2,
        routed_scaling_factor=1.0,
        attention_method="MLA",
    )
    defaults.update(overrides)
    return LongcatFlashConfig(**defaults)


def save_model(model, output_dir: Path):
    """Save model in sharded safetensors format with config and index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    config = model.config

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    items = list(sd.items())
    mid = len(items) // 2
    shard1 = dict(items[:mid])
    shard2 = dict(items[mid:])

    save_file(shard1, str(output_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(output_dir / "model-00002-of-00002.safetensors"))

    weight_map = {}
    for k in shard1:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2:
        weight_map[k] = "model-00002-of-00002.safetensors"

    total_size = sum(v.element_size() * v.nelement() for v in sd.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    gen_config = {"bos_token_id": 1, "eos_token_id": 2}
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    return sd


def load_weights_from_dir(model_dir: Path):
    """Load all weights from safetensors files in a directory."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(model_dir.glob("*.safetensors"))

    all_weights = {}
    for shard in shard_files:
        spath = model_dir / shard if isinstance(shard, str) else shard
        all_weights.update(load_file(str(spath)))
    return all_weights


class TestHFAlignment(unittest.TestCase):
    """Verify expanded models load correctly in HuggingFace."""

    @classmethod
    def setUpClass(cls):
        cls.test_root = Path("/tmp/longcat_align_test")
        if cls.test_root.exists():
            shutil.rmtree(cls.test_root)

    def _create_and_save(self, test_name, **config_overrides):
        """Helper: create a tiny model and save to disk, return (dir, state_dict, config)."""
        model_dir = self.test_root / test_name / "original"
        config = make_tiny_config(**config_overrides)
        model = LongcatFlashForCausalLM(config)
        sd = save_model(model, model_dir)
        return model_dir, sd, config

    def _expand_layers(self, orig_dir, output_dir, original_layers, target_layers,
                       copy_source=None):
        """Run layer expansion script."""
        args = [
            "expand_model_layers.py",
            "--model_dir", str(orig_dir),
            "--output_dir", str(output_dir),
            "--original_layers", str(original_layers),
            "--target_layers", str(target_layers),
        ]
        if copy_source is not None:
            args.extend(["--copy_source", copy_source])
        sys.argv = args
        expand_layers_main()

    def _expand_experts(self, orig_dir, output_dir, target_experts, target_topk):
        """Run expert expansion script."""
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(orig_dir),
            "--output_dir", str(output_dir),
            "--target_experts", str(target_experts),
            "--target_topk", str(target_topk),
        ]
        expand_experts_main()

    def _load_model_strict(self, model_dir):
        """Load model with strict=True to verify no missing/unexpected keys."""
        config = LongcatFlashConfig.from_pretrained(str(model_dir))
        model = LongcatFlashForCausalLM(config)
        weights = load_weights_from_dir(model_dir)
        result = model.load_state_dict(weights, strict=True)
        return model, config, result

    def _load_model_lenient(self, model_dir):
        """Load model with strict=False, return missing/unexpected keys."""
        config = LongcatFlashConfig.from_pretrained(str(model_dir))
        model = LongcatFlashForCausalLM(config)
        weights = load_weights_from_dir(model_dir)
        result = model.load_state_dict(weights, strict=False)
        return model, config, result

    # ------------------------------------------------------------------
    # Layer expansion tests
    # ------------------------------------------------------------------

    def test_layer_expansion_strict_loading(self):
        """Expand 2→4 layers: strict load must succeed (no missing/unexpected keys)."""
        orig_dir, orig_sd, orig_config = self._create_and_save("layer_strict")
        exp_dir = self.test_root / "layer_strict" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4)
        model, config, result = self._load_model_strict(exp_dir)

        self.assertEqual(len(result.missing_keys), 0,
                         f"Missing keys: {result.missing_keys}")
        self.assertEqual(len(result.unexpected_keys), 0,
                         f"Unexpected keys: {result.unexpected_keys}")
        self.assertEqual(config.num_layers, 4)

    def test_layer_expansion_all_shapes_match(self):
        """Every weight file tensor shape must match the model's state_dict shape."""
        orig_dir, _, _ = self._create_and_save("layer_shapes")
        exp_dir = self.test_root / "layer_shapes" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4)
        model, config, _ = self._load_model_strict(exp_dir)

        model_sd = model.state_dict()
        file_weights = load_weights_from_dir(exp_dir)

        for key in model_sd:
            self.assertIn(key, file_weights, f"Key missing from weight files: {key}")
            self.assertEqual(list(model_sd[key].shape),
                             list(file_weights[key].shape),
                             f"Shape mismatch for {key}: "
                             f"model={list(model_sd[key].shape)} "
                             f"file={list(file_weights[key].shape)}")

    def test_layer_expansion_non_layer_params_preserved(self):
        """Non-layer params (embed, norm, lm_head) must be bit-identical."""
        orig_dir, orig_sd, _ = self._create_and_save("layer_nonlayer")
        exp_dir = self.test_root / "layer_nonlayer" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        non_layer_keys = [k for k in orig_sd if "model.layers." not in k]
        for key in non_layer_keys:
            self.assertIn(key, exp_weights, f"Non-layer key missing: {key}")
            self.assertTrue(
                torch.equal(orig_sd[key], exp_weights[key]),
                f"Non-layer param changed after expansion: {key}",
            )

    def test_layer_expansion_original_layers_preserved(self):
        """Original layer params must be bit-identical after expansion."""
        orig_dir, orig_sd, _ = self._create_and_save("layer_orig_preserve")
        exp_dir = self.test_root / "layer_orig_preserve" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        for key, orig_val in orig_sd.items():
            if "model.layers." not in key:
                continue
            # Only check layers 0 and 1 (original layers)
            import re
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) < 2:
                self.assertIn(key, exp_weights, f"Original layer key missing: {key}")
                self.assertTrue(
                    torch.equal(orig_val, exp_weights[key]),
                    f"Original layer param changed: {key}",
                )

    def test_layer_expansion_forward_pass(self):
        """Expanded model must produce valid output from a forward pass."""
        orig_dir, _, _ = self._create_and_save("layer_forward")
        exp_dir = self.test_root / "layer_forward" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4)
        model, config, _ = self._load_model_strict(exp_dir)
        model.eval()

        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            output = model(input_ids)

        expected_shape = (1, 8, config.vocab_size)
        self.assertEqual(list(output.logits.shape), list(expected_shape))

    def test_layer_expansion_single_copy_source(self):
        """Expand 2→4 layers using single copy_source (layer 0)."""
        orig_dir, orig_sd, _ = self._create_and_save("layer_single_copy")
        exp_dir = self.test_root / "layer_single_copy" / "expanded"

        self._expand_layers(orig_dir, exp_dir, 2, 4, copy_source="0")
        model, config, result = self._load_model_strict(exp_dir)

        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)
        self.assertEqual(config.num_layers, 4)

        # Layers 0,1 should be original; layers 2,3 should be copies of layer 0
        exp_weights = load_weights_from_dir(exp_dir)
        for suffix in ["input_layernorm.weight"]:
            key0 = f"model.layers.0.{suffix}"
            if key0 in orig_sd:
                for new_li in [2, 3]:
                    new_key = f"model.layers.{new_li}.{suffix}"
                    self.assertTrue(
                        torch.equal(orig_sd[key0], exp_weights[new_key]),
                        f"Copy of layer 0 → {new_li} failed for {suffix}",
                    )

    # ------------------------------------------------------------------
    # Expert expansion tests
    # ------------------------------------------------------------------

    def test_expert_expansion_strict_loading(self):
        """Expand 4→8 experts: strict load must succeed."""
        orig_dir, orig_sd, orig_config = self._create_and_save("expert_strict")
        exp_dir = self.test_root / "expert_strict" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        model, config, result = self._load_model_strict(exp_dir)

        self.assertEqual(len(result.missing_keys), 0,
                         f"Missing keys: {result.missing_keys}")
        self.assertEqual(len(result.unexpected_keys), 0,
                         f"Unexpected keys: {result.unexpected_keys}")
        self.assertEqual(config.n_routed_experts, 8)
        self.assertEqual(config.zero_expert_num, 4)
        self.assertEqual(config.moe_topk, 4)

    def test_expert_expansion_all_shapes_match(self):
        """Every weight file tensor shape must match the model's state_dict."""
        orig_dir, _, _ = self._create_and_save("expert_shapes")
        exp_dir = self.test_root / "expert_shapes" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        model, config, _ = self._load_model_strict(exp_dir)

        model_sd = model.state_dict()
        file_weights = load_weights_from_dir(exp_dir)

        for key in model_sd:
            self.assertIn(key, file_weights, f"Key missing from weight files: {key}")
            self.assertEqual(list(model_sd[key].shape),
                             list(file_weights[key].shape),
                             f"Shape mismatch for {key}: "
                             f"model={list(model_sd[key].shape)} "
                             f"file={list(file_weights[key].shape)}")

    def test_expert_expansion_router_shapes(self):
        """Router weights must expand to (n_routed_experts + zero_expert_num, ...)."""
        orig_dir, _, orig_config = self._create_and_save("expert_router")
        exp_dir = self.test_root / "expert_router" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        model, config, _ = self._load_model_strict(exp_dir)

        expected_router_dim = config.n_routed_experts + config.zero_expert_num  # 8+4=12
        model_sd = model.state_dict()

        for li in range(config.num_layers):
            router_key = f"model.layers.{li}.mlp.router.classifier.weight"
            bias_key = f"model.layers.{li}.mlp.router.e_score_correction_bias"

            self.assertIn(router_key, model_sd)
            self.assertEqual(model_sd[router_key].shape[0], expected_router_dim,
                             f"Router {router_key} wrong dim0")

            self.assertIn(bias_key, model_sd)
            self.assertEqual(model_sd[bias_key].shape[0], expected_router_dim,
                             f"Bias {bias_key} wrong dim0")

    def test_expert_expansion_non_expert_params_preserved(self):
        """All non-expert, non-router params must be bit-identical."""
        orig_dir, orig_sd, _ = self._create_and_save("expert_nonexpert")
        exp_dir = self.test_root / "expert_nonexpert" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        skip_patterns = ("router.classifier", "e_score_correction_bias", ".experts.")
        for key, orig_val in orig_sd.items():
            if any(p in key for p in skip_patterns):
                continue
            self.assertIn(key, exp_weights, f"Key missing: {key}")
            self.assertTrue(
                torch.equal(orig_val, exp_weights[key]),
                f"Param changed after expert expansion: {key}",
            )

    def test_expert_expansion_original_experts_preserved(self):
        """Original expert weights must be bit-identical after expansion."""
        orig_dir, orig_sd, _ = self._create_and_save("expert_orig")
        exp_dir = self.test_root / "expert_orig" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        for key, orig_val in orig_sd.items():
            if ".experts." not in key:
                continue
            # Only check experts 0-3 (original experts)
            import re
            m = re.search(r"experts\.(\d+)\.", key)
            if m and int(m.group(1)) < 4:
                self.assertIn(key, exp_weights, f"Original expert key missing: {key}")
                self.assertTrue(
                    torch.equal(orig_val, exp_weights[key]),
                    f"Original expert weight changed: {key}",
                )

    def test_expert_expansion_duplicated_experts_match(self):
        """Duplicated experts (4-7) must be copies of originals (0-3)."""
        orig_dir, orig_sd, _ = self._create_and_save("expert_dup")
        exp_dir = self.test_root / "expert_dup" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        for key, orig_val in orig_sd.items():
            if ".experts." not in key:
                continue
            m = __import__("re").search(r"experts\.(\d+)\.", key)
            if not m:
                continue
            orig_idx = int(m.group(1))
            # Expert 4 should match expert 0, 5→1, 6→2, 7→3
            dup_idx = orig_idx + 4
            dup_key = key.replace(f"experts.{orig_idx}.", f"experts.{dup_idx}.")

            self.assertIn(dup_key, exp_weights,
                          f"Duplicated expert key missing: {dup_key}")
            self.assertTrue(
                torch.equal(orig_val, exp_weights[dup_key]),
                f"Duplicated expert doesn't match original: {dup_key} vs {key}",
            )

    def test_expert_expansion_router_zero_part_duplicated(self):
        """Router zero-expert portion must be duplicated alongside zero_expert_num."""
        orig_dir, orig_sd, orig_config = self._create_and_save("expert_router_zero")
        exp_dir = self.test_root / "expert_router_zero" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        exp_weights = load_weights_from_dir(exp_dir)

        orig_experts = orig_config.n_routed_experts  # 4
        orig_zero = orig_config.zero_expert_num  # 2
        exp_experts = 8
        exp_zero = 4

        for li in range(orig_config.num_layers):
            router_key = f"model.layers.{li}.mlp.router.classifier.weight"
            bias_key = f"model.layers.{li}.mlp.router.e_score_correction_bias"

            # Check router weight zero-part duplication
            orig_router = orig_sd[router_key]
            orig_zero_part = orig_router[orig_experts:]  # rows 4:6
            exp_router = exp_weights[router_key]

            # First copy at rows 8:10, second copy at rows 10:12
            for f in range(2):
                start = exp_experts + f * orig_zero
                end = start + orig_zero
                part = exp_router[start:end]
                self.assertTrue(
                    torch.equal(orig_zero_part, part),
                    f"Router zero-part copy {f} mismatch at layer {li}",
                )

            # Check bias zero-part duplication
            orig_bias = orig_sd[bias_key]
            orig_zero_bias = orig_bias[orig_experts:]
            exp_bias = exp_weights[bias_key]

            for f in range(2):
                start = exp_experts + f * orig_zero
                end = start + orig_zero
                part = exp_bias[start:end]
                self.assertTrue(
                    torch.equal(orig_zero_bias, part),
                    f"Bias zero-part copy {f} mismatch at layer {li}",
                )

    def test_expert_expansion_forward_pass(self):
        """Expanded model must produce valid output from a forward pass."""
        orig_dir, _, _ = self._create_and_save("expert_forward")
        exp_dir = self.test_root / "expert_forward" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        model, config, _ = self._load_model_strict(exp_dir)
        model.eval()

        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            output = model(input_ids)

        expected_shape = (1, 8, config.vocab_size)
        self.assertEqual(list(output.logits.shape), list(expected_shape))

    def test_expert_expansion_config_file_matches(self):
        """config.json on disk must match what the model expects."""
        orig_dir, _, _ = self._create_and_save("expert_config")
        exp_dir = self.test_root / "expert_config" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)

        with open(exp_dir / "config.json") as f:
            disk_config = json.load(f)

        config = LongcatFlashConfig.from_pretrained(str(exp_dir))
        model = LongcatFlashForCausalLM(config)
        model_config = model.config.to_dict()

        # Key fields that must match exactly
        critical_fields = [
            "n_routed_experts", "zero_expert_num", "moe_topk",
            "hidden_size", "expert_ffn_hidden_size", "num_layers",
            "num_attention_heads", "vocab_size",
        ]
        for field in critical_fields:
            self.assertEqual(disk_config[field], model_config[field],
                             f"Config field '{field}' mismatch: "
                             f"disk={disk_config[field]} vs model={model_config[field]}")

    def test_expert_expansion_weight_index_complete(self):
        """model.safetensors.index.json must list every key in the model's state_dict."""
        orig_dir, _, _ = self._create_and_save("expert_index")
        exp_dir = self.test_root / "expert_index" / "expanded"

        self._expand_experts(orig_dir, exp_dir, 8, 4)
        model, config, _ = self._load_model_strict(exp_dir)

        idx_path = exp_dir / "model.safetensors.index.json"
        self.assertTrue(idx_path.exists(), "Index file missing")

        with open(idx_path) as f:
            index = json.load(f)

        file_keys = set(index["weight_map"].keys())
        model_keys = set(model.state_dict().keys())

        missing_in_index = model_keys - file_keys
        extra_in_index = file_keys - model_keys

        self.assertEqual(len(missing_in_index), 0,
                         f"Keys in model but missing from index: "
                         f"{sorted(missing_in_index)[:20]}")
        self.assertEqual(len(extra_in_index), 0,
                         f"Keys in index but missing from model: "
                         f"{sorted(extra_in_index)[:20]}")

    # ------------------------------------------------------------------
    # Combined expansion tests
    # ------------------------------------------------------------------

    def test_combined_expansion_strict_loading(self):
        """Expand 2→4 layers then 4→8 experts: strict load must succeed."""
        orig_dir, orig_sd, _ = self._create_and_save("combined_strict")
        layers_dir = self.test_root / "combined_strict" / "layers_expanded"
        final_dir = self.test_root / "combined_strict" / "final"

        self._expand_layers(orig_dir, layers_dir, 2, 4)
        self._expand_experts(layers_dir, final_dir, 8, 4)

        model, config, result = self._load_model_strict(final_dir)

        self.assertEqual(len(result.missing_keys), 0,
                         f"Missing keys: {result.missing_keys}")
        self.assertEqual(len(result.unexpected_keys), 0,
                         f"Unexpected keys: {result.unexpected_keys}")
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.n_routed_experts, 8)
        self.assertEqual(config.zero_expert_num, 4)

    def test_combined_expansion_all_shapes_match(self):
        """Every tensor in weight files must match model state_dict shape."""
        orig_dir, _, _ = self._create_and_save("combined_shapes")
        layers_dir = self.test_root / "combined_shapes" / "layers_expanded"
        final_dir = self.test_root / "combined_shapes" / "final"

        self._expand_layers(orig_dir, layers_dir, 2, 4)
        self._expand_experts(layers_dir, final_dir, 8, 4)

        model, config, _ = self._load_model_strict(final_dir)
        model_sd = model.state_dict()
        file_weights = load_weights_from_dir(final_dir)

        for key in model_sd:
            self.assertIn(key, file_weights, f"Key missing: {key}")
            self.assertEqual(list(model_sd[key].shape),
                             list(file_weights[key].shape),
                             f"Shape mismatch for {key}")

    def test_combined_expansion_forward_pass(self):
        """Combined expanded model must produce valid output."""
        orig_dir, _, _ = self._create_and_save("combined_forward")
        layers_dir = self.test_root / "combined_forward" / "layers_expanded"
        final_dir = self.test_root / "combined_forward" / "final"

        self._expand_layers(orig_dir, layers_dir, 2, 4)
        self._expand_experts(layers_dir, final_dir, 8, 4)

        model, config, _ = self._load_model_strict(final_dir)
        model.eval()

        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (2, 16))
            output = model(input_ids)

        expected_shape = (2, 16, config.vocab_size)
        self.assertEqual(list(output.logits.shape), list(expected_shape))

    def test_combined_expansion_weight_count(self):
        """Verify the total weight count after combined expansion."""
        orig_dir, _, orig_config = self._create_and_save("combined_count")
        layers_dir = self.test_root / "combined_count" / "layers_expanded"
        final_dir = self.test_root / "combined_count" / "final"

        self._expand_layers(orig_dir, layers_dir, 2, 4)
        self._expand_experts(layers_dir, final_dir, 8, 4)

        model, config, _ = self._load_model_strict(final_dir)
        sd = model.state_dict()

        # Every layer should have 8 routed experts + router with dim 12
        for li in range(4):
            for ei in range(8):
                self.assertIn(
                    f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight",
                    sd,
                    f"Missing expert {ei} in layer {li}",
                )

            router_key = f"model.layers.{li}.mlp.router.classifier.weight"
            self.assertEqual(sd[router_key].shape[0], 12)  # 8 routed + 4 zero


if __name__ == "__main__":
    unittest.main()
