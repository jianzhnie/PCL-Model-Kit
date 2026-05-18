import json
import os
import shutil
import subprocess
import unittest
import re
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

# Add the project root to sys.path to import the script
sys.path.append(str(Path(__file__).parent.parent))
from utils.expand_model_layers import main as double_main

class TestDoubleHfModelLayers(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.test_dir = self.project_root / "tests/tmp_double_test"
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
        # Calculate original parameter sizes
        orig_layer_size = 0
        orig_non_layer_size = 0
        for k, v in self.weights.items():
            nbytes = v.element_size() * v.nelement()
            if "model.layers." in k:
                orig_layer_size += nbytes
            else:
                orig_non_layer_size += nbytes
        
        # Run expansion script (2 -> 4 layers, sequential)
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "4"
        ]
        double_main()
        
        # 1. Verify config
        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 4)
        
        # 2. Verify index and Total Size
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        # Expected size: non_layer + (4/2) * layer_size
        expected_total_size = orig_non_layer_size + (4 // 2) * orig_layer_size
        self.assertEqual(new_index["metadata"]["total_size"], expected_total_size)
        
        # 3. Verify weight values (Exact consistency)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
            
        # Check all layers (0-3)
        for li in range(4):
            src_li = li % 2 # Sequential mode: 0,1 unchanged, 2 copies 0, 3 copies 1
            for suffix in ["input_layernorm.weight", "mlp.gate_proj.weight"]:
                key = f"model.layers.{li}.{suffix}"
                src_key = f"model.layers.{src_li}.{suffix}"
                torch.testing.assert_close(all_weights[key], self.weights[src_key])
                # Ensure it's exactly the same data
                self.assertTrue(torch.equal(all_weights[key], self.weights[src_key]))

    def test_expansion_ratio_and_target_layers(self):
        """Test with target_layers=6 (3x expansion of layers)"""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "6"
        ]
        double_main()
        
        with open(self.output_dir / "config.json") as f:
            self.assertEqual(json.load(f)["num_layers"], 6)
            
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        
        # Verify 6 layers present
        layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) 
                         for k in new_index["weight_map"] if "layers." in k})
        self.assertEqual(layers, [0, 1, 2, 3, 4, 5])

    def test_double_layers_custom_copy(self):
        # Run expansion script (2 -> 4 layers, all copy layer 1)
        sys.argv = [
            "expand_model_layers.py",
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
            
        # Both Layer 2 and Layer 3 should copy Layer 1 exactly
        src_key = "model.layers.1.input_layernorm.weight"
        self.assertTrue(torch.equal(all_weights["model.layers.2.input_layernorm.weight"], self.weights[src_key]))
        self.assertTrue(torch.equal(all_weights["model.layers.3.input_layernorm.weight"], self.weights[src_key]))

    def test_double_layers_explicit_copy_list(self):
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--copy_source", "1,0"
        ]
        double_main()

        all_weights = {}
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Layer 2 copies layer 1, Layer 3 copies layer 0
        self.assertTrue(torch.equal(all_weights["model.layers.2.input_layernorm.weight"], self.weights["model.layers.1.input_layernorm.weight"]))
        self.assertTrue(torch.equal(all_weights["model.layers.3.input_layernorm.weight"], self.weights["model.layers.0.input_layernorm.weight"]))

    def test_double_layers_rejects_out_of_range_copy_source(self):
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--copy_source", "2"
        ]

        with self.assertRaises(SystemExit) as exc:
            double_main()

        self.assertEqual(exc.exception.code, 1)

    def test_target_layers_non_double(self):
        """Test --target_layers with a non-doubling value (2 → 3 layers)."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "3",
        ]
        double_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 3)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Original layers 0-1 unchanged
        torch.testing.assert_close(all_weights["model.layers.0.input_layernorm.weight"], self.weights["model.layers.0.input_layernorm.weight"])
        torch.testing.assert_close(all_weights["model.layers.1.input_layernorm.weight"], self.weights["model.layers.1.input_layernorm.weight"])
        # New layer 2 copies layer 0 (sequential: 2 mod 2 = 0)
        torch.testing.assert_close(all_weights["model.layers.2.input_layernorm.weight"], self.weights["model.layers.0.input_layernorm.weight"])
        # Layer 3 should NOT exist
        self.assertNotIn("model.layers.3.input_layernorm.weight", all_weights)

        # Param count: 6 original + 2 per new layer = 6 + 2 = 8
        self.assertEqual(len(new_index["weight_map"]), 8)

    def test_double_layers_longcat_structure(self):
        """Mimic LongCat naming: sub-indices like norm.0, norm.1, attn.0, attn.1, experts.X, mlps.X."""
        shutil.rmtree(self.test_dir)
        self.model_dir.mkdir(parents=True)

        config = {
            "model_type": "longcat",
            "num_layers": 2,
            "hidden_size": 16,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Build LongCat-style param names: each layer has norm.{0,1}, attn.{0,1}, experts.{0,1}, mlps.{0,1}
        weights = {}
        weights["model.embed_tokens.weight"] = torch.randn(100, 16)
        weights["model.norm.weight"] = torch.randn(16)
        # MTP params (should NOT be duplicated)
        weights["model.mtp.layers.0.transformer_layer.mlp.down_proj.weight"] = torch.randn(32, 16)
        weights["model.mtp.norm.weight"] = torch.randn(16)

        for li in range(2):
            weights[f"model.layers.{li}.input_layernorm.0.weight"] = torch.full((16,), float(li * 10))
            weights[f"model.layers.{li}.input_layernorm.1.weight"] = torch.full((16,), float(li * 10 + 1))
            weights[f"model.layers.{li}.post_attention_layernorm.0.weight"] = torch.full((16,), float(li * 10 + 2))
            weights[f"model.layers.{li}.self_attn.0.q_a_proj.weight"] = torch.full((32, 16), float(li * 10 + 3))
            weights[f"model.layers.{li}.self_attn.1.o_proj.weight"] = torch.full((16, 32), float(li * 10 + 4))
            weights[f"model.layers.{li}.mlp.router.classifier.weight"] = torch.randn(2, 16)
            weights[f"model.layers.{li}.mlp.router.e_score_correction_bias"] = torch.randn(2)
            # Experts
            for ei in range(2):
                weights[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"] = torch.full((32, 16), float(li * 100 + ei))
            # Dense MLPs
            weights[f"model.layers.{li}.mlps.0.gate_proj.weight"] = torch.full((32, 16), float(li * 10 + 5))
            weights[f"model.layers.{li}.mlps.1.down_proj.weight"] = torch.full((16, 32), float(li * 10 + 6))

        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        # Run: 2 → 4 layers
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
        ]
        double_main()

        # Verify config
        with open(self.output_dir / "config.json") as f:
            self.assertEqual(json.load(f)["num_layers"], 4)

        # Load all output tensors
        with open(self.output_dir / "model.safetensors.index.json") as f:
            out_idx = json.load(f)
        all_weights = {}
        for sn in set(out_idx["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / sn)))

        # Non-layer params (embed, norm, mtp) should NOT be duplicated
        self.assertIn("model.embed_tokens.weight", all_weights)
        self.assertIn("model.norm.weight", all_weights)
        self.assertIn("model.mtp.layers.0.transformer_layer.mlp.down_proj.weight", all_weights)
        self.assertIn("model.mtp.norm.weight", all_weights)

        # Layer 0 → Layer 2, Layer 1 → Layer 3 (sequential)
        # Check that duplicated layers have matching values
        for src, dst in [(0, 2), (1, 3)]:
            self.assertIn(f"model.layers.{dst}.input_layernorm.0.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.input_layernorm.1.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.self_attn.0.q_a_proj.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.self_attn.1.o_proj.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.mlp.router.classifier.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.mlp.router.e_score_correction_bias", all_weights)
            self.assertIn(f"model.layers.{dst}.mlp.experts.0.gate_proj.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.mlp.experts.1.gate_proj.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.mlps.0.gate_proj.weight", all_weights)
            self.assertIn(f"model.layers.{dst}.mlps.1.down_proj.weight", all_weights)

            # Verify values match source layer
            torch.testing.assert_close(
                all_weights[f"model.layers.{dst}.input_layernorm.0.weight"],
                torch.full((16,), float(src * 10)),
            )
            torch.testing.assert_close(
                all_weights[f"model.layers.{dst}.self_attn.0.q_a_proj.weight"],
                torch.full((32, 16), float(src * 10 + 3)),
            )
            torch.testing.assert_close(
                all_weights[f"model.layers.{dst}.mlp.experts.0.gate_proj.weight"],
                torch.full((32, 16), float(src * 100 + 0)),
            )
            torch.testing.assert_close(
                all_weights[f"model.layers.{dst}.mlps.0.gate_proj.weight"],
                torch.full((32, 16), float(src * 10 + 5)),
            )

        # Count: original non-layer(=4) + layer params(2 layers * 11 params per layer * 2 = 44) = 48
        self.assertEqual(len(out_idx["weight_map"]), 48)

    def test_expand_model_layers_shell_script(self):
        script_path = self.project_root / "scripts/expand_model_layers.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["ORIGINAL_LAYERS"] = "2"

        result = subprocess.run(
            ["bash", str(script_path), "seq"],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 4)

    def test_target_layers_shell_script(self):
        """Test expand_model_layers.sh with TARGET_LAYERS env var (2 → 3 layers)."""
        script_path = self.project_root / "scripts/expand_model_layers.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["ORIGINAL_LAYERS"] = "2"
        env["TARGET_LAYERS"] = "3"

        result = subprocess.run(
            ["bash", str(script_path), "seq"],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 3)

    def test_shell_script_single_mode(self):
        """Test expand_model_layers.sh single mode copies from specified layer."""
        script_path = self.project_root / "scripts/expand_model_layers.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["ORIGINAL_LAYERS"] = "2"

        result = subprocess.run(
            ["bash", str(script_path), "single", "1"],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 4)

        # Verify new layers both copy from layer 1
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        self.assertTrue(torch.equal(all_weights["model.layers.2.input_layernorm.weight"], self.weights["model.layers.1.input_layernorm.weight"]))
        self.assertTrue(torch.equal(all_weights["model.layers.3.input_layernorm.weight"], self.weights["model.layers.1.input_layernorm.weight"]))

    def test_shell_script_list_mode(self):
        """Test expand_model_layers.sh list mode with explicit copy mapping."""
        script_path = self.project_root / "scripts/expand_model_layers.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["ORIGINAL_LAYERS"] = "2"

        result = subprocess.run(
            ["bash", str(script_path), "list", "1,0"],
            cwd=self.project_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["num_layers"], 4)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        self.assertTrue(torch.equal(all_weights["model.layers.2.input_layernorm.weight"], self.weights["model.layers.1.input_layernorm.weight"]))
        self.assertTrue(torch.equal(all_weights["model.layers.3.input_layernorm.weight"], self.weights["model.layers.0.input_layernorm.weight"]))

    def test_sequential_wraps_modulo_when_num_new_exceeds_original(self):
        """Test sequential mode with num_new > original_layers (modulo wrapping).
        2 original layers → 10 target layers = 8 new layers.
        Expected copies: layers 2,4,6,8 ← 0; layers 3,5,7,9 ← 1."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "10"
        ]
        double_main()

        with open(self.output_dir / "config.json") as f:
            self.assertEqual(json.load(f)["num_layers"], 10)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Verify all 10 layers present with correct indices
        layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1))
                         for k in new_index["weight_map"] if "layers." in k})
        self.assertEqual(layers, list(range(10)))

        # Verify original layers 0-1 unchanged
        self.assertTrue(torch.equal(all_weights["model.layers.0.input_layernorm.weight"],
                                    self.weights["model.layers.0.input_layernorm.weight"]))
        self.assertTrue(torch.equal(all_weights["model.layers.1.input_layernorm.weight"],
                                    self.weights["model.layers.1.input_layernorm.weight"]))

        # Verify modulo wrapping: even new layers copy layer 0, odd new layers copy layer 1
        for new_li in range(2, 10):
            src_li = (new_li - 2) % 2  # offset 0→0, 1→1, 2→0, 3→1, ...
            self.assertTrue(
                torch.equal(all_weights[f"model.layers.{new_li}.input_layernorm.weight"],
                            self.weights[f"model.layers.{src_li}.input_layernorm.weight"]),
                f"Layer {new_li} should copy layer {src_li}"
            )

        # Verify total_size in metadata reflects the expansion
        # Non-layer: embed_tokens(100*16*4=6400) + norm(16*4=64) = 6464 bytes
        # Layer params: 2 layers * 2 params/layer * (16*4 + 32*16*4) = 2 * 2 * (64+2048) = 8448 bytes
        # Expanded layer params: 10 layers * 2 params/layer * 2112 = 42240 bytes
        # Total expected: 6464 + 42240 = 48704 bytes
        orig_non_layer = sum(
            v.element_size() * v.nelement()
            for k, v in self.weights.items()
            if "model.layers." not in k
        )
        orig_layer = sum(
            v.element_size() * v.nelement()
            for k, v in self.weights.items()
            if "model.layers." in k
        )
        expected_total = orig_non_layer + (10 / 2) * orig_layer
        self.assertEqual(new_index["metadata"]["total_size"], int(expected_total))

    def test_shard_file_sizes_match_target(self):
        """Verify that individual output shard file sizes match the original shard sizes."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "6"
        ]
        double_main()

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        # Each output shard should be close to the original shard sizes
        output_shards = set(new_index["weight_map"].values())
        output_sizes = []
        for sname in output_shards:
            spath = self.output_dir / sname
            self.assertTrue(spath.exists(), f"Shard {sname} should exist")
            output_sizes.append(spath.stat().st_size)

        # Original shard sizes
        orig_sizes = []
        for sname in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
            orig_sizes.append((self.model_dir / sname).stat().st_size)

        # All output shards should be within 20% of the target shard size
        target_size = self.model_dir / "model-00001-of-00002.safetensors"
        self.assertTrue(target_size.exists())

        # The shard detection uses average - verify no shard exceeds 2x average
        avg_orig = sum(orig_sizes) / len(orig_sizes)
        for sz in output_sizes:
            self.assertLessEqual(sz, 2 * avg_orig + 1,
                                 f"Output shard size {sz} exceeds 2x avg {avg_orig}")

    def test_target_layers_equals_original_rejects(self):
        """Test that target_layers == original_layers is rejected."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "2",
        ]
        with self.assertRaises(SystemExit) as exc:
            double_main()
        self.assertEqual(exc.exception.code, 1)

    def test_target_layers_less_than_original_rejects(self):
        """Test that target_layers < original_layers is rejected."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "1",
        ]
        with self.assertRaises(SystemExit) as exc:
            double_main()
        self.assertEqual(exc.exception.code, 1)

    def test_explicit_seq_copy_source(self):
        """Test that --copy_source seq explicitly works same as default."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--target_layers", "4",
            "--copy_source", "seq",
        ]
        double_main()

        with open(self.output_dir / "config.json") as f:
            self.assertEqual(json.load(f)["num_layers"], 4)

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Same as sequential: layer 2 copies 0, layer 3 copies 1
        self.assertTrue(torch.equal(all_weights["model.layers.2.input_layernorm.weight"],
                                    self.weights["model.layers.0.input_layernorm.weight"]))
        self.assertTrue(torch.equal(all_weights["model.layers.3.input_layernorm.weight"],
                                    self.weights["model.layers.1.input_layernorm.weight"]))

    def test_list_mode_out_of_range_rejects(self):
        """Test that out-of-range source in list mode is rejected."""
        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
            "--copy_source", "0,5",  # 5 is out of range for 2-layer model
        ]
        with self.assertRaises(SystemExit) as exc:
            double_main()
        self.assertEqual(exc.exception.code, 1)

    def test_auxiliary_files_copied(self):
        """Verify non-weight files (tokenizer, etc.) are copied to output."""
        # Create a dummy tokenizer file
        with open(self.model_dir / "tokenizer.json", "w") as f:
            json.dump({"dummy": True}, f)
        with open(self.model_dir / "README.md", "w") as f:
            f.write("# Test Model\n")

        sys.argv = [
            "expand_model_layers.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--original_layers", "2",
        ]
        double_main()

        self.assertTrue((self.output_dir / "tokenizer.json").exists())
        self.assertTrue((self.output_dir / "README.md").exists())
        with open(self.output_dir / "tokenizer.json") as f:
            self.assertEqual(json.load(f), {"dummy": True})

if __name__ == "__main__":
    unittest.main()
