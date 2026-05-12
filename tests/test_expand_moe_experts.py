import json
import os
import shutil
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
import sys

# Add the project root to sys.path to import the script
sys.path.append(str(Path(__file__).parent.parent))
from utils.expand_moe_experts import main as expand_main

class TestExpandMoeExperts(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/tmp_moe_test")
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
        
        # 2. Verify index
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        # Should have original + 4*3 new expert weights
        # Original: embed(1) + router_w(1) + router_b(1) + norm(1) + experts(4*3=12) = 16
        # New: experts(4*3=12)
        # Total: 16 + 12 = 28
        self.assertEqual(len(new_index["weight_map"]), 28)
        
        # 3. Verify weights
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
            
        # Check router expansion
        self.assertEqual(all_weights["model.layers.0.mlp.router.classifier.weight"].shape, (8, 16))
        self.assertEqual(all_weights["model.layers.0.mlp.router.e_score_correction_bias"].shape, (8,))
        
        # Check router weight identity (first 4 should match last 4)
        orig_router_w = self.weights["model.layers.0.mlp.router.classifier.weight"]
        new_router_w = all_weights["model.layers.0.mlp.router.classifier.weight"]
        torch.testing.assert_close(new_router_w[:4], orig_router_w)
        torch.testing.assert_close(new_router_w[4:], orig_router_w)
        
        # Check expert duplication
        for i in range(4):
            orig_val = float(i)
            # Original expert
            torch.testing.assert_close(all_weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"], torch.full((32, 16), orig_val))
            # New expert (i+4)
            torch.testing.assert_close(all_weights[f"model.layers.0.mlp.experts.{i+4}.gate_proj.weight"], torch.full((32, 16), orig_val))

if __name__ == "__main__":
    unittest.main()
