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
        torch.testing.assert_close(
            all_weights["model.layers.0.mlp.experts.4.gate_proj.weight"],
            torch.full((32, 16), 0.0),
        )
        torch.testing.assert_close(
            all_weights["model.layers.0.mlp.experts.7.gate_proj.weight"],
            torch.full((32, 16), 3.0),
        )

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
        self.assertEqual(new_config["zero_expert_num"], 2)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
        
        # New router shape: 8 (real) + 2 (zero) = 10
        self.assertEqual(all_weights["model.layers.0.mlp.router.classifier.weight"].shape, (10, 16))
        self.assertEqual(all_weights["model.layers.0.mlp.router.e_score_correction_bias"].shape, (10,))

        # Verify router content: [real, real, zero]
        orig_router = weights["model.layers.0.mlp.router.classifier.weight"]
        new_router = all_weights["model.layers.0.mlp.router.classifier.weight"]
        torch.testing.assert_close(new_router[:4], orig_router[:4])   # First real copy
        torch.testing.assert_close(new_router[4:8], orig_router[:4])  # Second real copy
        torch.testing.assert_close(new_router[8:], orig_router[4:])   # Zero experts preserved at end

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
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
        
        # Original zero-shot experts (4, 5) should be moved to (8, 9)
        # because routed experts expanded from 4 to 8.
        # Check consistency: all_weights[new_key] must be EXACTLY weights[old_key]
        for old_idx, new_idx in [(4, 8), (5, 9)]:
            old_key = f"model.layers.0.mlp.experts.{old_idx}.gate_proj.weight"
            new_key = f"model.layers.0.mlp.experts.{new_idx}.gate_proj.weight"
            self.assertIn(new_key, all_weights)
            self.assertTrue(torch.equal(all_weights[new_key], weights[old_key]), 
                            f"Zero-shot expert weight mismatch: {new_key} vs {old_key}")
        
        # Expert 4 is now a routed expert (copy of 0)
        self.assertTrue(torch.equal(all_weights["model.layers.0.mlp.experts.4.gate_proj.weight"], 
                                   weights["model.layers.0.mlp.experts.0.gate_proj.weight"]))

if __name__ == "__main__":
    unittest.main()
