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
from utils.expand_model_layers import main as double_main

class TestDoubleHfModelLayers(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/tmp_double_test")
        self.model_dir = self.test_dir / "original_model"
        self.output_dir = self.test_dir / "doubled_model"
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        self.model_dir.mkdir(parents=True)
        
        # 1. Create dummy config
        self.config = {
            "model_type": "longcat",
            "num_layers": 2,
            "hidden_size": 16
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(self.config, f)
            
        # 2. Create dummy weights
        # Layer 0 and Layer 1
        self.weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.input_layernorm.weight": torch.full((16,), 0.0),
            "model.layers.0.mlp.gate_proj.weight": torch.full((32, 16), 0.0),
            "model.layers.1.input_layernorm.weight": torch.full((16,), 1.0),
            "model.layers.1.mlp.gate_proj.weight": torch.full((32, 16), 1.0),
            "model.norm.weight": torch.randn(16),
        }
        
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

    def test_double_layers_sequential(self):
        # Run expansion script (2 -> 4 layers, sequential)
        sys.argv = [
            "double_hf_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2"
        ]
        double_main()
        
        # 1. Verify config
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 4)
        
        # 2. Verify index
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        # Original: embed(1) + layer0(2) + layer1(2) + norm(1) = 6 params
        # New layers: layer2(copy 0) + layer3(copy 1) = 4 params
        # Total: 10 params
        self.assertEqual(len(new_index["weight_map"]), 10)
        
        # 3. Verify weights
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
            
        # Check layers 0-1 (unchanged)
        torch.testing.assert_close(all_weights["model.layers.0.input_layernorm.weight"], torch.full((16,), 0.0))
        torch.testing.assert_close(all_weights["model.layers.1.input_layernorm.weight"], torch.full((16,), 1.0))
        
        # Check layers 2-3 (duplicated)
        # Layer 2 should copy Layer 0
        torch.testing.assert_close(all_weights["model.layers.2.input_layernorm.weight"], torch.full((16,), 0.0))
        # Layer 3 should copy Layer 1
        torch.testing.assert_close(all_weights["model.layers.3.input_layernorm.weight"], torch.full((16,), 1.0))

    def test_double_layers_custom_copy(self):
        # Run expansion script (2 -> 4 layers, all copy layer 1)
        sys.argv = [
            "double_hf_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--copy_source", "1"
        ]
        double_main()
        
        all_weights = {}
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
            
        # Both Layer 2 and Layer 3 should copy Layer 1
        torch.testing.assert_close(all_weights["model.layers.2.input_layernorm.weight"], torch.full((16,), 1.0))
        torch.testing.assert_close(all_weights["model.layers.3.input_layernorm.weight"], torch.full((16,), 1.0))

if __name__ == "__main__":
    unittest.main()
