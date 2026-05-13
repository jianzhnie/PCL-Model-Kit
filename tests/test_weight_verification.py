import json
import shutil
import unittest
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).parent.parent))
from utils.verify_expanded_weights import (
    ModelWeightLoader,
    parse_copy_source,
    verify_layers,
    verify_experts,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_index(weight_map: dict[str, str]) -> dict:
    return {"weight_map": weight_map}


def _save_model(dest: Path, weights: dict[str, torch.Tensor],
                config: dict | None = None,
                index: dict | None = None,
                shard_name: str = "model.safetensors"):
    """Save weights + optional config + optional index to a directory."""
    dest.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(dest / shard_name))
    if index:
        with open(dest / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)
    
    # Always save a config.json to suppress loader error messages,
    # unless it was already provided.
    final_config = config if config is not None else {"n_layers": 28}
    with open(dest / "config.json", "w") as f:
        json.dump(final_config, f)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestWeightVerification(unittest.TestCase):
    def setUp(self):
        # Use a unique directory for each test to avoid interference and I/O issues
        self.test_dir = Path(__file__).parent / f"tmp_verify_test_{self.id()}"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.orig_dir = self.test_dir / "orig"
        self.exp_dir = self.test_dir / "exp"
        self.orig_dir.mkdir(exist_ok=True)
        self.exp_dir.mkdir(exist_ok=True)
        self.loaders = []

    def tearDown(self):
        for loader in self.loaders:
            try:
                loader.close()
            except:
                pass
        self.loaders.clear()
        
        # Give a small delay to allow file handles to be released by the OS if needed
        # but shutil.rmtree with ignore_errors=True is usually enough.
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _get_loader(self, path):
        loader = ModelWeightLoader(path)
        self.loaders.append(loader)
        return loader

    # ── parse_copy_source ───────────────────────────────────────────────

    def test_parse_copy_source_seq(self):
        result = parse_copy_source("seq", 4, 4)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_parse_copy_source_seq_wrap(self):
        result = parse_copy_source("seq", 3, 7)
        self.assertEqual(result, [0, 1, 2, 0, 1, 2, 0])

    def test_parse_copy_source_single_valid(self):
        result = parse_copy_source("2", 5, 3)
        self.assertEqual(result, [2, 2, 2])

    def test_parse_copy_source_list_valid(self):
        result = parse_copy_source("0,1,0,1", 4, 4)
        self.assertEqual(result, [0, 1, 0, 1])

    def test_parse_copy_source_list_wrong_length(self):
        with self.assertRaises(ValueError):
            parse_copy_source("0,1", 4, 4)  # need 4 entries

    def test_parse_copy_source_none_defaults_to_seq(self):
        result = parse_copy_source(None, 3, 3)
        self.assertEqual(result, [0, 1, 2])

    # ── verify_layers — success paths ────────────────────────────────────

    def test_layers_success_seq(self):
        orig = {
            "model.layers.0.w": torch.randn(10, 10),
            "model.layers.1.w": torch.randn(10, 10),
            "model.embed.w": torch.randn(10, 10),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": orig["model.layers.0.w"],
            "model.layers.1.w": orig["model.layers.1.w"],
            "model.layers.2.w": orig["model.layers.0.w"].clone(),
            "model.layers.3.w": orig["model.layers.1.w"].clone(),
            "model.embed.w": orig["model.embed.w"],
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_layers(ol, el, 2, 4, "seq")), 0)

    def test_layers_success_single_source(self):
        """All new layers copy from layer 0."""
        orig = {"model.layers.0.w": torch.randn(2, 3)}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": orig["model.layers.0.w"],
            "model.layers.1.w": orig["model.layers.0.w"].clone(),
            "model.layers.2.w": orig["model.layers.0.w"].clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_layers(ol, el, 1, 3, "0")), 0)

    def test_layers_success_list_source(self):
        """New layers copy from explicitly listed sources."""
        orig = {
            "model.layers.0.w": torch.randn(2, 2).clone(),
            "model.layers.1.w": torch.randn(2, 2).clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": orig["model.layers.0.w"].clone(),
            "model.layers.1.w": orig["model.layers.1.w"].clone(),
            "model.layers.2.w": orig["model.layers.1.w"].clone(),  # ← 1
            "model.layers.3.w": orig["model.layers.0.w"].clone(),  # ← 0
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_layers(ol, el, 2, 4, "1,0")), 0)

    # ── verify_layers — mismatch detection ───────────────────────────────

    def test_layers_value_mismatch(self):
        orig = {"model.layers.0.w": torch.ones(2, 2).clone()}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": torch.ones(2, 2).clone(),
            "model.layers.1.w": torch.zeros(2, 2).clone(),  # wrong: should be ones
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_layers(ol, el, 1, 2, "0")
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Value mismatch" in x for x in m))

    def test_layers_shape_mismatch(self):
        orig = {"model.layers.0.w": torch.randn(2, 2).clone()}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": torch.randn(2, 2).clone(),
            "model.layers.1.w": torch.randn(3, 3).clone(),  # wrong shape
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_layers(ol, el, 1, 2, "0")
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Shape mismatch" in x for x in m))

    # ── verify_layers — structural checks ───────────────────────────────

    def test_layers_missing_param_in_exp(self):
        """Expanded model missing a param entirely."""
        orig = {
            "model.layers.0.w": torch.randn(2, 2).clone(),
            "model.layers.1.w": torch.randn(2, 2).clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        # exp is missing layer 1 which is expected
        exp = {
            "model.layers.0.w": orig["model.layers.0.w"].clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_layers(ol, el, 1, 2, "0")
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Missing layers" in x for x in m))

    def test_layers_source_missing(self):
        """Exp param points to a non-existent source."""
        orig = {"model.layers.0.w": torch.randn(2, 2).clone()}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}))

        exp = {
            "model.layers.0.w": orig["model.layers.0.w"].clone(),
            "model.layers.1.w": torch.randn(2, 2).clone(), # source model.layers.0.w found, but this is fine
            "model.layers.1.b": torch.randn(2).clone(), # source model.layers.0.b missing
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}))

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        # We need to make sure verify_layers actually gets to the point of checking values.
        # It needs the layer to be present in structural check.
        # Layer 1 is present in exp, and it has 2 params (w and b).
        # Layer 0 has 1 param (w).
        # Mapping for layer 1 is layer 0.
        # Layer 1 count (2) != Layer 0 count (1) -> structural failure.
        m = verify_layers(ol, el, 1, 2, "0")
        self.assertGreater(len(m), 0)
        self.assertTrue(any(
            "Param count mismatch" in x or 
            "Param name mismatch" in x or 
            "Source missing" in x 
            for x in m
        ))

    # ── verify_experts — success ─────────────────────────────────────────

    def test_experts_success(self):
        config = {"n_routed_experts": 2, "zero_expert_num": 1}
        _save_model(self.orig_dir, {}, config=config)
        _save_model(self.exp_dir, {}, config=config)

        router_orig = torch.randn(3, 10).clone()  # 2 real + 1 zero
        orig = {
            "model.layers.0.mlp.experts.0.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.experts.1.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.experts.2.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.router.classifier.weight": router_orig.clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config={"n_routed_experts": 2, "zero_expert_num": 1})

        exp_cfg = {"n_routed_experts": 4, "zero_expert_num": 1}
        router_exp = torch.cat([
            router_orig[:2], router_orig[:2], router_orig[2:]
        ], dim=0).clone()
        exp = {
            "model.layers.0.mlp.experts.0.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.1.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.experts.2.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.3.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.experts.4.w": orig["model.layers.0.mlp.experts.2.w"].clone(),
            "model.layers.0.mlp.router.classifier.weight": router_exp.clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config=exp_cfg)

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_experts(ol, el)), 0)

    def test_experts_success_no_zero(self):
        """MoE with no zero-shot experts."""
        config = {"n_routed_experts": 2}
        _save_model(self.orig_dir, {}, config=config)
        _save_model(self.exp_dir, {}, config=config)

        r_orig = torch.randn(2, 10).clone()
        orig = {
            "model.layers.0.mlp.experts.0.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.experts.1.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.router.classifier.weight": r_orig.clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config={"n_routed_experts": 2})

        r_exp = torch.cat([r_orig, r_orig], dim=0).clone()
        exp = {
            "model.layers.0.mlp.experts.0.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.1.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.experts.2.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.3.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.router.classifier.weight": r_exp.clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config={"n_routed_experts": 4})

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_experts(ol, el)), 0)

    # ── verify_experts — mismatch detection ──────────────────────────────

    def test_experts_value_mismatch(self):
        config = {"n_routed_experts": 2}
        orig = {"model.layers.0.mlp.experts.0.w": torch.ones(2, 2).clone(),
                "model.layers.0.mlp.experts.1.w": torch.ones(2, 2).clone(),
                "model.layers.0.mlp.router.classifier.weight": torch.randn(2, 10).clone()}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config=config)

        exp = {"model.layers.0.mlp.experts.0.w": torch.ones(2, 2).clone(),
               "model.layers.0.mlp.experts.1.w": torch.ones(2, 2).clone(),
               "model.layers.0.mlp.experts.2.w": torch.zeros(2, 2).clone(),  # wrong
               "model.layers.0.mlp.experts.3.w": torch.ones(2, 2).clone(),
               "model.layers.0.mlp.router.classifier.weight": torch.cat([
                   orig["model.layers.0.mlp.router.classifier.weight"],
                   orig["model.layers.0.mlp.router.classifier.weight"],
               ], dim=0).clone()}
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config={"n_routed_experts": 4})

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_experts(ol, el)
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Expert value mismatch" in x for x in m))

    def test_experts_router_mismatch(self):
        config = {"n_routed_experts": 2}
        r_orig = torch.randn(2, 10).clone()
        orig = {"model.layers.0.mlp.experts.0.w": torch.randn(2, 2).clone(),
                "model.layers.0.mlp.experts.1.w": torch.randn(2, 2).clone(),
                "model.layers.0.mlp.router.classifier.weight": r_orig.clone()}
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config=config)

        r_bad = torch.cat([r_orig, torch.randn(2, 10).clone()], dim=0).clone()  # wrong second half
        exp = {"model.layers.0.mlp.experts.0.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
               "model.layers.0.mlp.experts.1.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
               "model.layers.0.mlp.experts.2.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
               "model.layers.0.mlp.experts.3.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
               "model.layers.0.mlp.router.classifier.weight": r_bad.clone()}
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config={"n_routed_experts": 4})

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_experts(ol, el)
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Router value mismatch" in x for x in m))

    # ── verify_experts — structural checks ───────────────────────────────

    def test_experts_zero_count_mismatch(self):
        """Zero expert count changed across expansion."""
        config_o = {"n_routed_experts": 2, "zero_expert_num": 1}
        config_e = {"n_routed_experts": 4, "zero_expert_num": 2}
        _save_model(self.orig_dir, {}, config=config_o)
        _save_model(self.exp_dir, {}, config=config_e)
        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_experts(ol, el)
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Zero expert count mismatch" in x for x in m))

    def test_experts_source_expert_missing(self):
        """Expanded model expert points to missing source expert."""
        config_o = {"n_routed_experts": 2}
        config_e = {"n_routed_experts": 4}
        
        orig = {
            "model.layers.0.mlp.experts.0.w": torch.randn(2, 2).clone(),
            # expert 1 missing in source
            "model.layers.0.mlp.router.classifier.weight": torch.randn(2, 10).clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config=config_o)

        exp = {
            "model.layers.0.mlp.experts.0.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.1.w": torch.randn(2, 2).clone(), # Source expert 1 missing
            "model.layers.0.mlp.experts.2.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.3.w": torch.randn(2, 2).clone(), # Source expert 1 missing
            "model.layers.0.mlp.router.classifier.weight": torch.randn(4, 10).clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config=config_e)

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_experts(ol, el)
        self.assertGreater(len(m), 0)
        self.assertTrue(any("Source expert missing" in x for x in m))

    # ── Parallel execution ───────────────────────────────────────────────

    def test_parallel_execution(self):
        """Multi-shard model verified with multiple workers."""
        num_layers = 4
        orig_w = {f"model.layers.{i}.w": torch.randn(10, 10).clone() for i in range(num_layers)}

        shard1 = {f"model.layers.{i}.w": orig_w[f"model.layers.{i}.w"].clone() for i in range(2)}
        shard2 = {f"model.layers.{i}.w": orig_w[f"model.layers.{i}.w"].clone() for i in range(2, 4)}
        save_file(shard1, str(self.orig_dir / "model-00001.safetensors"))
        save_file(shard2, str(self.orig_dir / "model-00002.safetensors"))
        with open(self.orig_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                f"model.layers.{i}.w": f"model-{1 if i < 2 else 2:05d}.safetensors"
                for i in range(4)
            }}, f)
        _save_model(self.orig_dir, {}, config={"n_layers": 4})

        # 4 → 8 layers, split across 2 shards
        exp_s1 = {f"model.layers.{i}.w": orig_w[f"model.layers.{i}.w"].clone() for i in range(4)}
        exp_s2 = {f"model.layers.{i+4}.w": orig_w[f"model.layers.{i}.w"].clone() for i in range(4)}
        save_file(exp_s1, str(self.exp_dir / "model-00001.safetensors"))
        save_file(exp_s2, str(self.exp_dir / "model-00002.safetensors"))
        with open(self.exp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                **{f"model.layers.{i}.w": "model-00001.safetensors" for i in range(4)},
                **{f"model.layers.{i+4}.w": "model-00002.safetensors" for i in range(4)},
            }}, f)
        _save_model(self.exp_dir, {}, config={"n_layers": 8})

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        # Run with multiple workers to stress thread safety
        m = verify_layers(ol, el, 4, 8, "seq", workers=4)
        self.assertEqual(len(m), 0)

    def test_parallel_experts(self):
        """Multi-shard expert model verified with multiple workers."""
        config_o = {"n_routed_experts": 2}
        config_e = {"n_routed_experts": 4}

        r_o = torch.randn(2, 10).clone()
        r_e = torch.cat([r_o, r_o], dim=0).clone()
        e0, e1 = torch.randn(2, 2).clone(), torch.randn(2, 2).clone()

        orig_s1 = {"model.layers.0.mlp.experts.0.w": e0.clone(),
                    "model.layers.0.mlp.experts.1.w": e1.clone()}
        orig_s2 = {"model.layers.0.mlp.router.classifier.weight": r_o.clone()}
        save_file(orig_s1, str(self.orig_dir / "model-00001.safetensors"))
        save_file(orig_s2, str(self.orig_dir / "model-00002.safetensors"))
        with open(self.orig_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                "model.layers.0.mlp.experts.0.w": "model-00001.safetensors",
                "model.layers.0.mlp.experts.1.w": "model-00001.safetensors",
                "model.layers.0.mlp.router.classifier.weight": "model-00002.safetensors",
            }}, f)
        _save_model(self.orig_dir, {}, config=config_o)

        exp_s1 = {"model.layers.0.mlp.experts.0.w": e0.clone(),
                   "model.layers.0.mlp.experts.1.w": e1.clone(),
                   "model.layers.0.mlp.experts.2.w": e0.clone(),
                   "model.layers.0.mlp.experts.3.w": e1.clone()}
        exp_s2 = {"model.layers.0.mlp.router.classifier.weight": r_e.clone()}
        save_file(exp_s1, str(self.exp_dir / "model-00001.safetensors"))
        save_file(exp_s2, str(self.exp_dir / "model-00002.safetensors"))
        with open(self.exp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                "model.layers.0.mlp.experts.0.w": "model-00001.safetensors",
                "model.layers.0.mlp.experts.1.w": "model-00001.safetensors",
                "model.layers.0.mlp.experts.2.w": "model-00001.safetensors",
                "model.layers.0.mlp.experts.3.w": "model-00001.safetensors",
                "model.layers.0.mlp.router.classifier.weight": "model-00002.safetensors",
            }}, f)
        _save_model(self.exp_dir, {}, config=config_e)

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        m = verify_experts(ol, el, workers=4)
        self.assertEqual(len(m), 0)

    def test_experts_success_different_router_name(self):
        """MoE with gate.weight instead of router.classifier.weight."""
        config = {"n_routed_experts": 2}
        _save_model(self.orig_dir, {}, config=config)
        _save_model(self.exp_dir, {}, config=config)

        r_orig = torch.randn(2, 10).clone()
        orig = {
            "model.layers.0.mlp.experts.0.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.experts.1.w": torch.randn(2, 2).clone(),
            "model.layers.0.mlp.gate.weight": r_orig.clone(),
        }
        _save_model(self.orig_dir, orig,
                    index=_make_index({k: "model.safetensors" for k in orig}),
                    config={"n_routed_experts": 2})

        r_exp = torch.cat([r_orig, r_orig], dim=0).clone()
        exp = {
            "model.layers.0.mlp.experts.0.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.1.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.experts.2.w": orig["model.layers.0.mlp.experts.0.w"].clone(),
            "model.layers.0.mlp.experts.3.w": orig["model.layers.0.mlp.experts.1.w"].clone(),
            "model.layers.0.mlp.gate.weight": r_exp.clone(),
        }
        _save_model(self.exp_dir, exp,
                    index=_make_index({k: "model.safetensors" for k in exp}),
                    config={"n_routed_experts": 4})

        ol = self._get_loader(self.orig_dir)
        el = self._get_loader(self.exp_dir)
        self.assertEqual(len(verify_experts(ol, el)), 0)


    # ── ModelWeightLoader ────────────────────────────────────────────────

    def test_loader_single_file_no_index(self):
        """Loader handles a single safetensors file with no index."""
        w = {"a": torch.randn(2, 2).clone(), "b": torch.randn(3, 3).clone()}
        _save_model(self.orig_dir, w) # Use helper to get config.json
        loader = ModelWeightLoader(self.orig_dir)
        self.assertEqual(len(loader.weight_map), 2)
        t = loader.get_tensor("a")
        self.assertTrue(torch.equal(t, w["a"]))

    def test_loader_missing_tensor(self):
        w = {"a": torch.randn(2, 2).clone()}
        _save_model(self.orig_dir, w,
                    index=_make_index({k: "model.safetensors" for k in w}))
        loader = ModelWeightLoader(self.orig_dir)
        self.assertIsNone(loader.get_tensor("nonexistent"))


if __name__ == "__main__":
    unittest.main()
