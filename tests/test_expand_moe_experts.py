import json
import os
import subprocess
import shutil
import unittest
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

# Add the project root to sys.path to import the script
sys.path.append(str(Path(__file__).parent.parent))
from utils.expand_moe_experts import main as expand_main

class TestExpandMoeExperts(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.test_dir = self.project_root / "tests/tmp_moe_test"
        self.model_dir = self.test_dir / "original_model"
        self.output_dir = self.test_dir / "expanded_model"
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        self.model_dir.mkdir(parents=True)
        
        # 1. Create dummy config
        self.config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "hidden_size": 16,
            "expert_ffn_hidden_size": 32,
            "num_layers": 1
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(self.config, f)
            
        # 2. Create dummy weights
        # Layer 0: 4 experts, 1 router
        self.weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
            "model.norm.weight": torch.randn(16),
        }
        
        # Expert weights for 4 experts
        for i in range(4):
            self.weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
            self.weights[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.full((32, 16), float(i))
            self.weights[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.full((16, 32), float(i))
            
        # Shard the weights into 2 files
        shard1 = {k: v for i, (k, v) in enumerate(self.weights.items()) if i % 2 == 0}
        shard2 = {k: v for i, (k, v) in enumerate(self.weights.items()) if i % 2 != 0}
        
        save_file(shard1, str(self.model_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(self.model_dir / "model-00002-of-00002.safetensors"))
        
        # 3. Create dummy index
        self.index = {
            "metadata": {"total_size": 0},
            "weight_map": {}
        }
        for k in shard1: self.index["weight_map"][k] = "model-00001-of-00002.safetensors"
        for k in shard2: self.index["weight_map"][k] = "model-00002-of-00002.safetensors"
        
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(self.index, f)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_expansion(self):
        # Calculate original parameter sizes
        orig_expert_size = 0
        orig_router_size = 0
        orig_other_size = 0
        for k, v in self.weights.items():
            nbytes = v.element_size() * v.nelement()
            if "mlp.experts." in k:
                orig_expert_size += nbytes
            elif "mlp.router." in k:
                orig_router_size += nbytes
            else:
                orig_other_size += nbytes

        # Run expansion script (4 -> 8 experts)
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8"
        ]
        expand_main()
        
        # 1. Verify config
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        
        # 2. Verify index and Total Size
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        # New expert size = 2 * original
        # New router size = 2 * original (since 4 -> 8 experts, router dim doubles)
        expected_total_size = orig_other_size + (8 // 4) * orig_expert_size + (8 // 4) * orig_router_size
        self.assertEqual(new_index["metadata"]["total_size"], expected_total_size)
        
        # 3. Verify weights (Exact consistency)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
            
        # Check router expansion
        orig_router_w = self.weights["model.layers.0.mlp.router.classifier.weight"]
        new_router_w = all_weights["model.layers.0.mlp.router.classifier.weight"]
        self.assertTrue(torch.equal(new_router_w[:4], orig_router_w))
        self.assertTrue(torch.equal(new_router_w[4:], orig_router_w))
        
        # Check expert duplication
        for i in range(4):
            orig_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            new_key = f"model.layers.0.mlp.experts.{i+4}.gate_proj.weight"
            self.assertTrue(torch.equal(all_weights[new_key], self.weights[orig_key]))

    def test_expansion_ratio_and_invalid_multiple(self):
        """Test that script rejects target_experts that is not a multiple of original."""
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "6" # 6 is not a multiple of 4
        ]
        with self.assertRaises(SystemExit) as exc:
            expand_main()
        self.assertEqual(exc.exception.code, 1)

    def test_expansion_with_gate_router_and_n_experts_key(self):
        shutil.rmtree(self.test_dir)
        self.model_dir.mkdir(parents=True)

        config = {
            "model_type": "longcat",
            "n_experts": 4,
            "hidden_size": 16,
            "expert_ffn_hidden_size": 32,
            "num_layers": 1,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.gate.weight": torch.randn(4, 16),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(4),
            "model.norm.weight": torch.randn(16),
        }
        for i in range(4):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

        shard1 = {k: v for idx, (k, v) in enumerate(weights.items()) if idx % 2 == 0}
        shard2 = {k: v for idx, (k, v) in enumerate(weights.items()) if idx % 2 != 0}
        save_file(shard1, str(self.model_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(self.model_dir / "model-00002-of-00002.safetensors"))

        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        for key in shard1:
            index["weight_map"][key] = "model-00001-of-00002.safetensors"
        for key in shard2:
            index["weight_map"][key] = "model-00002-of-00002.safetensors"
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8"
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_experts"], 8)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        self.assertEqual(all_weights["model.layers.0.mlp.gate.weight"].shape, (8, 16))
        self.assertEqual(all_weights["model.layers.0.mlp.gate.e_score_correction_bias"].shape, (8,))
        # New experts 4-7 should be exact copies of 0-3
        for i in range(4):
            orig_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            new_key = f"model.layers.0.mlp.experts.{i+4}.gate_proj.weight"
            self.assertTrue(torch.equal(all_weights[new_key], weights[orig_key]),
                            f"Expert {i+4} should be an exact copy of expert {i}")

    def test_expand_moe_experts_shell_script_default_doubles_experts(self):
        script_path = self.project_root / "scripts/expand_moe_experts.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)

    def test_expand_moe_experts_shell_script_with_explicit_target(self):
        """Test expand_moe_experts.sh with explicit target_experts and target_topk args."""
        script_path = self.project_root / "scripts/expand_moe_experts.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)

        result = subprocess.run(
            ["bash", str(script_path), "12", "24"],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 12)
        self.assertEqual(new_config["moe_topk"], 24)

    def test_expansion_with_zero_experts(self):
        # 1. Create config with zero_expert_num
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "zero_expert_num": 2,
            "hidden_size": 16,
            "num_layers": 1
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # 2. Create weights (router dim = 4 + 2 = 6)
        weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
            "model.norm.weight": torch.randn(16),
        }
        # Experts (only 4 real experts)
        for i in range(4):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

        # Save weights and index
        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights}
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        # 3. Run expansion (4 -> 8 experts)
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8"
        ]
        expand_main()

        # 4. Verify
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        self.assertEqual(new_config["zero_expert_num"], 4)  # 2 * 2 = 4 (doubled)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # New router shape: 8 (real) + 4 (zero) = 12
        self.assertEqual(all_weights["model.layers.0.mlp.router.classifier.weight"].shape, (12, 16))
        self.assertEqual(all_weights["model.layers.0.mlp.router.e_score_correction_bias"].shape, (12,))

        # Verify router content: [real, real, zero, zero]
        orig_router = weights["model.layers.0.mlp.router.classifier.weight"]
        new_router = all_weights["model.layers.0.mlp.router.classifier.weight"]
        torch.testing.assert_close(new_router[:4], orig_router[:4])     # First real copy
        torch.testing.assert_close(new_router[4:8], orig_router[:4])    # Second real copy
        torch.testing.assert_close(new_router[8:10], orig_router[4:])   # First zero copy
        torch.testing.assert_close(new_router[10:12], orig_router[4:])  # Second zero copy

    def test_target_topk_updates_config(self):
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8",
            "--target_topk", "24",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        self.assertEqual(new_config["moe_topk"], 24)

    def test_target_topk_adds_key_when_not_present(self):
        # Recreate config without any topk key
        shutil.rmtree(self.test_dir)
        self.model_dir.mkdir(parents=True)
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "hidden_size": 16,
            "num_layers": 1
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
            "model.norm.weight": torch.randn(16),
        }
        for i in range(4):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights}
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_topk", "16",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["moe_topk"], 16)

    def test_expansion_with_zero_experts_parameters(self):
        # 1. Create config with zero_expert_num
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "zero_expert_num": 2,
            "hidden_size": 16,
            "num_layers": 1
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # 2. Create weights (router dim = 4 + 2 = 6)
        weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
            "model.norm.weight": torch.randn(16),
        }
        # Experts: 4 real + 2 zero-shot = 6 total
        for i in range(6):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

        # Save weights and index
        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights}
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        # 3. Run expansion (4 -> 8 experts)
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8"
        ]
        expand_main()

        # 4. Verify
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["zero_expert_num"], 4)  # 2 * 2 = 4 (doubled)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Original zero-shot experts (4, 5) should be moved to (8, 9)
        # because routed experts expanded from 4 to 8.
        # New zero-expert copies should be at (10, 11).
        for old_idx, new_idx in [(4, 8), (5, 9)]:
            old_key = f"model.layers.0.mlp.experts.{old_idx}.gate_proj.weight"
            new_key = f"model.layers.0.mlp.experts.{new_idx}.gate_proj.weight"
            self.assertIn(new_key, all_weights)
            self.assertTrue(torch.equal(all_weights[new_key], weights[old_key]),
                            f"Zero-shot expert weight mismatch: {new_key} vs {old_key}")

        # New zero-expert copies (10, 11) should be copies of original (4, 5)
        for old_idx, copy_idx in [(4, 10), (5, 11)]:
            old_key = f"model.layers.0.mlp.experts.{old_idx}.gate_proj.weight"
            copy_key = f"model.layers.0.mlp.experts.{copy_idx}.gate_proj.weight"
            self.assertIn(copy_key, all_weights)
            self.assertTrue(torch.equal(all_weights[copy_key], weights[old_key]),
                            f"Zero-expert copy mismatch: {copy_key} vs {old_key}")

        # Expert 4 is now a routed expert (copy of 0)
        self.assertTrue(torch.equal(all_weights["model.layers.0.mlp.experts.4.gate_proj.weight"],
                                   weights["model.layers.0.mlp.experts.0.gate_proj.weight"]))

    def test_expansion_factor_greater_than_two(self):
        """Expand 4→12 experts (3x expansion factor)."""
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "hidden_size": 8,
            "num_layers": 1,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.embed_tokens.weight": torch.randn(50, 8),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 8),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
            "model.norm.weight": torch.randn(8),
        }
        for i in range(4):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((16, 8), float(i))

        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights}
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "12"
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 12)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Router should have shape (12, 8) - 3 copies of original 4
        new_router = all_weights["model.layers.0.mlp.router.classifier.weight"]
        self.assertEqual(new_router.shape, (12, 8))
        orig_router = weights["model.layers.0.mlp.router.classifier.weight"]
        # Each of the 3 blocks should match the original
        for factor in range(3):
            part = new_router[factor * 4:(factor + 1) * 4]
            self.assertTrue(torch.equal(part, orig_router),
                            f"Router block {factor} should match original")

        # Verify experts: 0-3 original, 4-7 copies (mod 4), 8-11 copies (mod 4)
        for new_idx in range(4, 12):
            src_idx = new_idx % 4
            self.assertTrue(
                torch.equal(all_weights[f"model.layers.0.mlp.experts.{new_idx}.gate_proj.weight"],
                            weights[f"model.layers.0.mlp.experts.{src_idx}.gate_proj.weight"]),
                f"Expert {new_idx} should be copy of expert {src_idx}"
            )

        # Verify original experts 0-3 unchanged
        for i in range(4):
            self.assertTrue(torch.equal(
                all_weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"],
                weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"]
            ))

    def test_multi_layer_moe_expansion(self):
        """Test MoE expansion with multiple MoE layers."""
        config = {
            "model_type": "longcat",
            "n_routed_experts": 2,
            "hidden_size": 8,
            "num_layers": 2,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.embed_tokens.weight": torch.randn(50, 8),
            "model.norm.weight": torch.randn(8),
        }
        for li in range(2):
            weights[f"model.layers.{li}.mlp.router.classifier.weight"] = torch.randn(2, 8)
            weights[f"model.layers.{li}.mlp.router.e_score_correction_bias"] = torch.randn(2)
            for ei in range(2):
                weights[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"] = torch.full((16, 8), float(li * 10 + ei))

        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights}
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "4"
        ]
        expand_main()

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Verify both layers have 4 experts each
        for li in range(2):
            # Router shape doubled (2→4)
            router_key = f"model.layers.{li}.mlp.router.classifier.weight"
            self.assertEqual(all_weights[router_key].shape, (4, 8))

            # Expert copies
            for ei in range(4):
                expert_key = f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"
                self.assertIn(expert_key, all_weights)
                src_ei = ei % 2
                expected_val = weights[f"model.layers.{li}.mlp.experts.{src_ei}.gate_proj.weight"]
                self.assertTrue(torch.equal(all_weights[expert_key], expected_val),
                                f"Layer {li} expert {ei} mismatch")

        # Non-expert params should be unchanged
        self.assertTrue(torch.equal(all_weights["model.embed_tokens.weight"],
                                    weights["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(all_weights["model.norm.weight"],
                                    weights["model.norm.weight"]))

    def test_shard_file_sizes_match_target_moe(self):
        """Verify output shard sizes for MoE expansion match the original shard size pattern."""
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8"
        ]
        expand_main()

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        output_shards = set(new_index["weight_map"].values())
        output_sizes = []
        for sname in output_shards:
            spath = self.output_dir / sname
            self.assertTrue(spath.exists(), f"Shard {sname} should exist")
            output_sizes.append(spath.stat().st_size)

        # Get original shard sizes
        orig_sizes = []
        for sname in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
            orig_sizes.append((self.model_dir / sname).stat().st_size)

        avg_orig = sum(orig_sizes) / len(orig_sizes)
        for sz in output_sizes:
            self.assertLessEqual(sz, 2 * avg_orig + 1,
                                 f"Output shard size {sz} exceeds 2x avg {avg_orig}")

    def test_default_doubles_experts(self):
        """Test that omitting --target_experts defaults to doubling."""
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)  # 4 * 2

    def test_zero_target_experts_rejected(self):
        """Test that --target_experts 0 is rejected (division by zero)."""
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "0",
        ]
        with self.assertRaises(SystemExit) as exc:
            expand_main()
        self.assertEqual(exc.exception.code, 1)

    def test_parallel_produces_same_output_as_serial(self):
        """Parallel (workers=2) must produce identical output to serial (workers=1)."""
        # 1. Run serial expansion into output_serial
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir / "serial"),
            "--target_experts", "8",
        ]
        expand_main()

        # 2. Run parallel expansion into output_parallel
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir / "parallel"),
            "--target_experts", "8",
            "--workers", "2",
        ]
        expand_main()

        # 3. Compare results
        serial_idx = self.output_dir / "serial" / "model.safetensors.index.json"
        parallel_idx = self.output_dir / "parallel" / "model.safetensors.index.json"

        with open(serial_idx) as f:
            s_idx = json.load(f)
        with open(parallel_idx) as f:
            p_idx = json.load(f)

        # Same weight map (modulo shard filename differences)
        self.assertEqual(
            sorted(s_idx["weight_map"].keys()),
            sorted(p_idx["weight_map"].keys()),
        )

        # Same config
        with open(self.output_dir / "serial" / "config.json") as f:
            s_cfg = json.load(f)
        with open(self.output_dir / "parallel" / "config.json") as f:
            p_cfg = json.load(f)
        self.assertEqual(s_cfg, p_cfg)

        # Same tensors (value-wise)
        s_weights = {}
        for shard_name in set(s_idx["weight_map"].values()):
            s_weights.update(load_file(str(self.output_dir / "serial" / shard_name)))
        p_weights = {}
        for shard_name in set(p_idx["weight_map"].values()):
            p_weights.update(load_file(str(self.output_dir / "parallel" / shard_name)))

        for key in s_weights:
            self.assertTrue(torch.equal(s_weights[key], p_weights[key]),
                            f"Mismatch in tensor: {key}")

    def test_parallel_with_noise_and_zero_experts(self):
        """Parallel mode with noise injection and zero experts."""
        # Setup config with zero_expert_num
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "zero_expert_num": 2,
            "hidden_size": 16,
            "num_layers": 1,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
        }
        for i in range(6):  # 4 real + 2 zero
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        # Run with 3 workers + noise (ensures noise is threaded through workers)
        sys.argv = [
            "expand_moe_experts.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--target_experts", "8",
            "--router-noise-scale", "1e-6",
            "--workers", "3",
        ]
        expand_main()

        # Verify output exists and is loadable
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        self.assertEqual(new_config["zero_expert_num"], 4)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_idx = json.load(f)
        all_w = {}
        for shard_name in set(new_idx["weight_map"].values()):
            all_w.update(load_file(str(self.output_dir / shard_name)))

        # Router shape should be expanded: 8 real + 4 zero = 12
        self.assertEqual(
            all_w["model.layers.0.mlp.router.classifier.weight"].shape, (12, 16))

        # Expert 4 (first duplicate of expert 0) should exist and match
        self.assertTrue(torch.equal(
            all_w["model.layers.0.mlp.experts.0.gate_proj.weight"],
            all_w["model.layers.0.mlp.experts.4.gate_proj.weight"],
        ))


if __name__ == "__main__":
    unittest.main()
