import json
import os
import subprocess
import shutil
import unittest
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.append(str(Path(__file__).parent.parent))
from utils.expand_moe_experts import main as expand_main


class TestExpandMoeExpertsGroup(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.test_dir = self.project_root / "tests/tmp_moe_group_test"
        self.model_dir = self.test_dir / "original_model"
        self.output_dir = self.test_dir / "expanded_model"

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        self.model_dir.mkdir(parents=True)

        # Create dummy config
        self.config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "hidden_size": 16,
            "expert_ffn_hidden_size": 32,
            "num_layers": 1,
            "moe_topk": 12,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(self.config, f)

        # Create dummy weights: 4 experts, 1 router
        self.weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(4),
            "model.norm.weight": torch.randn(16),
        }
        for i in range(4):
            self.weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))
            self.weights[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.full((32, 16), float(i))
            self.weights[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.full((16, 32), float(i))

        # Shard into 2 files
        shard1 = {k: v for i, (k, v) in enumerate(self.weights.items()) if i % 2 == 0}
        shard2 = {k: v for i, (k, v) in enumerate(self.weights.items()) if i % 2 != 0}

        save_file(shard1, str(self.model_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(self.model_dir / "model-00002-of-00002.safetensors"))

        self.index = {
            "metadata": {"total_size": 0},
            "weight_map": {},
        }
        for k in shard1:
            self.index["weight_map"][k] = "model-00001-of-00002.safetensors"
        for k in shard2:
            self.index["weight_map"][k] = "model-00002-of-00002.safetensors"

        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(self.index, f)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    # ── Core expansion tests ──────────────────────────────────────────────

    def test_group_expansion_keeps_topk_unchanged(self):
        """方案1 must keep moe_topk unchanged (12 stays 12)."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        self.assertEqual(new_config["moe_topk"], 12,
                         "方案1 must keep moe_topk unchanged")

    def test_group_expansion_adds_use_group_routing_flag(self):
        """Config must include use_group_routing: true and expert_expansion_factor."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertTrue(new_config.get("use_group_routing"),
                        "方案1 must set use_group_routing: true in config")
        self.assertEqual(new_config.get("expert_expansion_factor"), 2,
                         "2x expansion (4→8) should set expert_expansion_factor=2")

    def test_group_expansion_weight_correctness(self):
        """Weight expansion is identical to 方案2 — experts duplicated, router expanded."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)

        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        # Router: [real, real] = 2 copies of original 4
        orig_router = self.weights["model.layers.0.mlp.router.classifier.weight"]
        new_router = all_weights["model.layers.0.mlp.router.classifier.weight"]
        self.assertEqual(new_router.shape, (8, 16))
        self.assertTrue(torch.equal(new_router[:4], orig_router))
        self.assertTrue(torch.equal(new_router[4:], orig_router))

        # Expert copies: 0-3 original, 4-7 copies
        for i in range(4):
            orig_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            new_key = f"model.layers.0.mlp.experts.{i + 4}.gate_proj.weight"
            self.assertTrue(torch.equal(all_weights[new_key], self.weights[orig_key]),
                            f"Expert {i + 4} should copy expert {i}")

    def test_group_expansion_no_target_topk_arg(self):
        """方案1 rejects --target_topk (mutually exclusive with --use_group_routing)."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
            "--target_topk", "24",
        ]
        with self.assertRaises(SystemExit) as exc:
            expand_main()
        self.assertEqual(exc.exception.code, 1)  # conflict error

    # ── Default doubling ──────────────────────────────────────────────────

    def test_default_doubles_experts(self):
        """Omitting --target_experts defaults to doubling (4 → 8)."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)

    # ── Zero experts ──────────────────────────────────────────────────────

    def test_group_expansion_with_zero_experts(self):
        """Zero experts expanded correctly with group routing."""
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "zero_expert_num": 2,
            "hidden_size": 16,
            "num_layers": 1,
            "moe_topk": 12,
        }
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        weights = {
            "model.embed_tokens.weight": torch.randn(100, 16),
            "model.layers.0.mlp.router.classifier.weight": torch.randn(6, 16),
            "model.layers.0.mlp.router.e_score_correction_bias": torch.randn(6),
            "model.norm.weight": torch.randn(16),
        }
        for i in range(4):
            weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.full((32, 16), float(i))

        save_file(weights, str(self.model_dir / "model.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 8)
        self.assertEqual(new_config["zero_expert_num"], 4)
        self.assertEqual(new_config["moe_topk"], 12)  # unchanged
        self.assertTrue(new_config.get("use_group_routing"))

        # Router: 8 real + 4 zero = 12
        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))
        self.assertEqual(all_weights["model.layers.0.mlp.router.classifier.weight"].shape, (12, 16))

    # ── Multi-layer ───────────────────────────────────────────────────────

    def test_multi_layer_group_expansion(self):
        """Group routing with multiple MoE layers."""
        config = {
            "model_type": "longcat",
            "n_routed_experts": 2,
            "hidden_size": 8,
            "num_layers": 2,
            "moe_topk": 6,
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
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "4",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 4)
        self.assertEqual(new_config["moe_topk"], 6)  # unchanged
        self.assertTrue(new_config.get("use_group_routing"))

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        for li in range(2):
            router_key = f"model.layers.{li}.mlp.router.classifier.weight"
            self.assertEqual(all_weights[router_key].shape, (4, 8))
            for ei in range(4):
                expert_key = f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"
                self.assertIn(expert_key, all_weights)
                src_ei = ei % 2
                self.assertTrue(
                    torch.equal(all_weights[expert_key],
                                weights[f"model.layers.{li}.mlp.experts.{src_ei}.gate_proj.weight"]),
                    f"Layer {li} expert {ei} mismatch",
                )

    # ── Shell script tests ────────────────────────────────────────────────

    def test_shell_script_default_doubles_experts(self):
        """Shell script doubles experts by default."""
        script_path = self.project_root / "scripts/expand_model_experts_group.sh"
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
        self.assertTrue(new_config.get("use_group_routing"))

    def test_shell_script_with_explicit_target(self):
        """Shell script with explicit target_experts."""
        script_path = self.project_root / "scripts/expand_model_experts_group.sh"
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)

        result = subprocess.run(
            ["bash", str(script_path), "12"],
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
        self.assertEqual(new_config["moe_topk"], 12)  # topk unchanged
        self.assertTrue(new_config.get("use_group_routing"))

    # ── Invalid inputs ────────────────────────────────────────────────────

    def test_expansion_ratio_not_multiple_rejected(self):
        """target_experts must be a multiple of original."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "6",  # 6 is not a multiple of 4
        ]
        with self.assertRaises(SystemExit) as exc:
            expand_main()
        self.assertEqual(exc.exception.code, 1)

    def test_zero_target_experts_rejected(self):
        """target_experts=0 rejected."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "0",
        ]
        with self.assertRaises(SystemExit) as exc:
            expand_main()
        self.assertEqual(exc.exception.code, 1)

    # ── 3x expansion factor ───────────────────────────────────────────────

    def test_expansion_factor_greater_than_two(self):
        """Expand 4→12 experts (3x) with group routing."""
        config = {
            "model_type": "longcat",
            "n_routed_experts": 4,
            "hidden_size": 8,
            "num_layers": 1,
            "moe_topk": 6,
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
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(self.model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "12",
        ]
        expand_main()

        with open(self.output_dir / "config.json") as f:
            new_config = json.load(f)
        self.assertEqual(new_config["n_routed_experts"], 12)
        self.assertEqual(new_config["moe_topk"], 6)  # unchanged
        self.assertTrue(new_config.get("use_group_routing"))

    # ── Non-MoE params unchanged ──────────────────────────────────────────

    def test_non_moe_params_unchanged(self):
        """Non-expert parameters pass through unmodified."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        with open(self.output_dir / "model.safetensors.index.json") as f:
            new_index = json.load(f)
        all_weights = {}
        for shard_name in set(new_index["weight_map"].values()):
            all_weights.update(load_file(str(self.output_dir / shard_name)))

        self.assertTrue(torch.equal(all_weights["model.embed_tokens.weight"],
                                    self.weights["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(all_weights["model.norm.weight"],
                                    self.weights["model.norm.weight"]))

    # ── Parallel produces same output as serial ───────────────────────────

    def test_parallel_produces_same_output_as_serial(self):
        """Parallel (workers=2) must produce identical output to serial."""
        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir / "serial"),
            "--use_group_routing",
            "--target_experts", "8",
        ]
        expand_main()

        sys.argv = [
            "expand_moe_experts_group.py",
            "--model_dir", str(self.model_dir),
            "--output_dir", str(self.output_dir / "parallel"),
            "--use_group_routing",
            "--target_experts", "8",
            "--workers", "2",
        ]
        expand_main()

        with open(self.output_dir / "serial" / "config.json") as f:
            s_cfg = json.load(f)
        with open(self.output_dir / "parallel" / "config.json") as f:
            p_cfg = json.load(f)
        self.assertEqual(s_cfg, p_cfg)

        with open(self.output_dir / "serial" / "model.safetensors.index.json") as f:
            s_idx = json.load(f)
        with open(self.output_dir / "parallel" / "model.safetensors.index.json") as f:
            p_idx = json.load(f)

        self.assertEqual(sorted(s_idx["weight_map"].keys()), sorted(p_idx["weight_map"].keys()))

        s_weights = {}
        for shard_name in set(s_idx["weight_map"].values()):
            s_weights.update(load_file(str(self.output_dir / "serial" / shard_name)))
        p_weights = {}
        for shard_name in set(p_idx["weight_map"].values()):
            p_weights.update(load_file(str(self.output_dir / "parallel" / shard_name)))

        for key in s_weights:
            self.assertTrue(torch.equal(s_weights[key], p_weights[key]),
                            f"Mismatch in tensor: {key}")


    # ── Grouped routing logic test ────────────────────────────────────────

    def test_grouped_routing_topk_indices(self):
        """Verify the grouped routing tensor logic directly.

        For N=4, F=2, topk=2: 8 total real experts in 4 groups of 2.
        The router logits should be reshaped, best-per-group selected,
        then topk groups chosen.
        """
        import torch.nn.functional as F

        N = 4   # original experts
        F_val = 2   # expansion factor
        real_experts = N * F_val  # 8
        topk = 2

        # Simulate router scores: assign explicit values so we can predict winner
        # Expert layout: [E0, E1, E2, E3, E0', E1', E2', E3']
        # To group same-source copies: view(F, N).transpose
        # -> [[0.9, 0.2, 0.3, 0.95],  → first F=2 rows are N=4 elements each
        #     [0.1, 0.8, 0.3, 0.05]]
        # transpose -> groups: {E0,E0'}={0.9,0.1}, {E1,E1'}={0.2,0.8}, etc.
        scores = torch.tensor([
            [0.9, 0.2, 0.3, 0.95, 0.1, 0.8, 0.3, 0.05],
        ])  # (1, 8)

        grouped = scores.view(-1, F_val, N).transpose(-1, -2)  # (1, 4, 2)
        group_best, group_best_idx = grouped.max(dim=-1)  # (1, 4)

        # Verify per-group winners
        # Group 0: {0.9, 0.1} -> 0.9 (idx 0 = E0)
        # Group 1: {0.2, 0.8} -> 0.8 (idx 1 = E1')
        # Group 2: {0.3, 0.3} -> 0.3 (idx 0 = E2)
        # Group 3: {0.95,0.05} -> 0.95 (idx 0 = E3)
        expected_group_best = torch.tensor([[0.9, 0.8, 0.3, 0.95]])
        expected_group_idx = torch.tensor([[0, 1, 0, 0]])  # offset within each group
        self.assertTrue(torch.equal(group_best, expected_group_best),
                        f"Group best scores: {group_best}")
        self.assertTrue(torch.equal(group_best_idx, expected_group_idx),
                        f"Group best indices: {group_best_idx}")

        # topk=2 from 4 groups -> should pick group 3 (0.95) and group 0 (0.9)
        _, topk_group_ids = torch.topk(group_best, k=topk, dim=-1, sorted=False)
        # topk_group_ids may be [3, 0] or [0, 3] (unsorted); both valid

        # Final expert indices: group_id + offset * N
        topk_indices = topk_group_ids + group_best_idx.gather(1, topk_group_ids) * N

        # Expected: group 3 -> expert 3 + 0*4 = 3 (E3)
        # group 0 -> expert 0 + 0*4 = 0 (E0)
        selected = set(topk_indices[0].tolist())
        self.assertEqual(selected, {0, 3},
                         f"Expected experts {{0, 3}} (E0 from group 0, E3 from group 3), got {selected}")

    def test_grouped_routing_preserves_expert_diversity(self):
        """Grouped routing never selects two experts from the same group.

        Even when the same group has the two highest individual scores,
        only one can be selected (the best in that group).
        """
        N = 4
        F_val = 2
        real_experts = N * F_val
        topk = 3

        # Make experts 4 and 5 (both in group 2, indices 4 and 5 -> group 2)
        # have the highest scores. But group routing means at most one from group 2.
        scores = torch.tensor([[
            0.01, 0.02, 0.99, 0.98, 0.01, 0.01, 0.01, 0.01,
        ]])  # group 2 (idx 2,5): [0.99, 0.01], group 3 (idx 3,6): [0.98, 0.01]

        grouped = scores.view(-1, F_val, N).transpose(-1, -2)
        _, group_best_idx = grouped.max(dim=-1)  # group 2 wins idx 2 (val 0.99)
        group_best = grouped.max(dim=-1).values

        _, topk_group_ids = torch.topk(group_best, k=topk, dim=-1, sorted=True)
        topk_indices = topk_group_ids + group_best_idx.gather(1, topk_group_ids) * N
        selected = set(topk_indices[0].tolist())

        # Should select from 3 different groups, NOT both from group 2
        # Layout: [E0..E3, E0'..E3'], so expert i belongs to group i % N
        groups_selected = {idx % N for idx in selected}
        self.assertEqual(len(groups_selected), topk,
                         f"All {topk} selected experts must come from different groups, "
                         f"got groups {groups_selected}")


if __name__ == "__main__":
    unittest.main()
