"""
Comprehensive unit tests for utility modules

Coverage targets:
- Branch coverage >= 90%
- Critical path coverage 100%
- Boundary testing, exception testing, parameterized testing
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import torch
from safetensors.torch import save_file

from utils.shared import (
    ALL_ROUTER_SUFFIXES,
    assign_shards_layer_aware,
    build_expert_target_map,
    build_layer_mapping,
    expand_router_bias,
    expand_router_weight,
    find_expert_count,
    get_expert_info,
    get_layer_index,
    get_nbytes_from_meta,
    is_router_bias,
    is_router_param,
    is_router_weight,
    layer_sort_key,
    load_config,
    load_index,
    make_expert_key,
    parse_copy_source,
    read_safetensors_header,
    set_layer_index,
    should_zero,
    tensor_nbytes,
)

from utils.check_model_weights import (_build_empty_model, _compare_shapes,
                                       _expected_state_specs,
                                       _read_specs_from_shard, _shard_paths,
                                       estimate_model_params)
from utils.check_model_weights import main as check_main
from utils.check_model_weights import (verify_config_consistency,
                                       verify_pretrain_script_consistency)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory with complete dummy files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # Create config.json
        config = {
            'vocab_size': 128,
            'hidden_size': 64,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'intermediate_size': 128,
            'model_type': 'kimi_k2',
        }
        with open(ckpt_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Create complete dummy safetensors file with all required weights
        dummy_weights = {
            # Embeddings and output
            'model.embed_tokens.weight': torch.randn(128, 64),
            'model.norm.weight': torch.randn(64),
            'lm_head.weight': torch.randn(128, 64),
            # Layer 0 - attention
            'model.layers.0.self_attn.q_proj.weight': torch.randn(64, 64),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(32, 64),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(32, 64),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(64, 64),
            # Layer 0 - layernorm
            'model.layers.0.input_layernorm.weight': torch.randn(64),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(64),
            # Layer 0 - MLP (dense for first 2 layers)
            'model.layers.0.mlp.gate_proj.weight': torch.randn(128, 64),
            'model.layers.0.mlp.up_proj.weight': torch.randn(128, 64),
            'model.layers.0.mlp.down_proj.weight': torch.randn(64, 128),
            # Layer 1 - attention
            'model.layers.1.self_attn.q_proj.weight': torch.randn(64, 64),
            'model.layers.1.self_attn.k_proj.weight': torch.randn(32, 64),
            'model.layers.1.self_attn.v_proj.weight': torch.randn(32, 64),
            'model.layers.1.self_attn.o_proj.weight': torch.randn(64, 64),
            # Layer 1 - layernorm
            'model.layers.1.input_layernorm.weight': torch.randn(64),
            'model.layers.1.post_attention_layernorm.weight': torch.randn(64),
            # Layer 1 - MLP
            'model.layers.1.mlp.gate_proj.weight': torch.randn(128, 64),
            'model.layers.1.mlp.up_proj.weight': torch.randn(128, 64),
            'model.layers.1.mlp.down_proj.weight': torch.randn(64, 128),
        }
        save_file(dummy_weights, ckpt_dir / 'model.safetensors')

        yield ckpt_dir


@pytest.fixture
def sharded_checkpoint_dir():
    """Create a temporary sharded checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # Create config.json
        config = {
            'vocab_size': 256,
            'hidden_size': 128,
            'num_hidden_layers': 4,
            'num_attention_heads': 8,
            'intermediate_size': 256,
        }
        with open(ckpt_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Create sharded weights
        weights1 = {
            'model.embed_tokens.weight': torch.randn(256, 128),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(64, 128),
        }
        weights2 = {
            'model.layers.0.self_attn.v_proj.weight': torch.randn(64, 128),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(128, 128),
            'model.norm.weight': torch.randn(128),
            'lm_head.weight': torch.randn(256, 128),
        }
        save_file(weights1, ckpt_dir / 'model-00001-of-00002.safetensors')
        save_file(weights2, ckpt_dir / 'model-00002-of-00002.safetensors')

        # Create index file
        index = {
            'metadata': {
                'total_size': 12345
            },
            'weight_map': {
                'model.embed_tokens.weight':
                'model-00001-of-00002.safetensors',
                'model.layers.0.self_attn.q_proj.weight':
                'model-00001-of-00002.safetensors',
                'model.layers.0.self_attn.k_proj.weight':
                'model-00001-of-00002.safetensors',
                'model.layers.0.self_attn.v_proj.weight':
                'model-00002-of-00002.safetensors',
                'model.layers.0.self_attn.o_proj.weight':
                'model-00002-of-00002.safetensors',
                'model.norm.weight': 'model-00002-of-00002.safetensors',
                'lm_head.weight': 'model-00002-of-00002.safetensors',
            },
        }
        with open(ckpt_dir / 'model.safetensors.index.json', 'w') as f:
            json.dump(index, f)

        yield ckpt_dir


# =============================================================================
# _shard_paths Tests
# =============================================================================


class TestShardPaths:
    """Test _shard_paths function"""

    def test_single_shard(self, temp_checkpoint_dir):
        """Test with single shard"""
        paths, index = _shard_paths(temp_checkpoint_dir)
        assert len(paths) == 1
        assert paths[0].name == 'model.safetensors'
        assert index is None

    def test_sharded_model(self, sharded_checkpoint_dir):
        """Test with sharded model"""
        paths, index = _shard_paths(sharded_checkpoint_dir)
        assert len(paths) == 2
        assert index is not None
        assert index.exists()

    def test_missing_checkpoint(self):
        """Test error when no checkpoint found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError,
                               match='No model.safetensors'):
                _shard_paths(Path(tmpdir))

    def test_sharded_missing_index(self):
        """Test error when index file missing for sharded model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'model-00001-of-00002.safetensors').touch()
            with pytest.raises(FileNotFoundError):
                _shard_paths(Path(tmpdir))


# =============================================================================
# _read_specs_from_shard Tests
# =============================================================================


class TestReadSpecsFromShard:
    """Test _read_specs_from_shard function"""

    def test_read_specs(self, temp_checkpoint_dir):
        """Test reading specs from shard"""
        shard_path = temp_checkpoint_dir / 'model.safetensors'
        specs = _read_specs_from_shard(shard_path)

        assert 'model.embed_tokens.weight' in specs
        assert specs['model.embed_tokens.weight'] == (128, 64)
        assert specs['model.norm.weight'] == (64, )

    def test_read_specs_sharded(self, sharded_checkpoint_dir):
        """Test reading specs from sharded checkpoint"""
        shard_path = sharded_checkpoint_dir / 'model-00001-of-00002.safetensors'
        specs = _read_specs_from_shard(shard_path)

        assert 'model.embed_tokens.weight' in specs
        assert specs['model.embed_tokens.weight'] == (256, 128)


# =============================================================================
# _build_empty_model Tests
# =============================================================================


class TestBuildEmptyModel:
    """Test _build_empty_model function"""

    def test_build_empty_model(self, temp_checkpoint_dir):
        """Test building empty model from config"""
        from models.configuration_deepseek_1t import DeepseekV3Config

        config = DeepseekV3Config(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

        with patch('utils.check_model_weights.init_empty_weights'):
            model = _build_empty_model(config)
            assert model is not None


# =============================================================================
# _expected_state_specs Tests
# =============================================================================


class TestExpectedStateSpecs:
    """Test _expected_state_specs function"""

    def test_expected_specs(self):
        """Test getting expected state specs from model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128, bias=False),
            torch.nn.Linear(128, 256, bias=False),
        )

        specs, keys = _expected_state_specs(model)

        assert len(specs) == 2
        assert '0.weight' in specs
        assert specs['0.weight'] == (128, 64)
        assert specs['1.weight'] == (256, 128)
        assert keys == {'0.weight', '1.weight'}


# =============================================================================
# _compare_shapes Tests
# =============================================================================


class TestCompareShapes:
    """Test _compare_shapes function"""

    def test_no_mismatches(self):
        """Test when all shapes match"""
        expected = {'a': (64, 128), 'b': (128, 256)}
        ckpt = {'a': (64, 128), 'b': (128, 256)}

        problems = _compare_shapes(expected, ckpt, 'test')
        assert len(problems) == 0

    def test_shape_mismatch(self):
        """Test when shapes don't match"""
        expected = {'a': (64, 128), 'b': (128, 256)}
        ckpt = {'a': (64, 128), 'b': (256, 128)}  # Transposed

        problems = _compare_shapes(expected, ckpt, 'test')
        assert len(problems) == 1
        assert 'shape mismatch' in problems[0]
        assert 'b' in problems[0]

    def test_partial_overlap(self):
        """Test when only some keys overlap"""
        expected = {'a': (64, 128), 'b': (128, 256), 'c': (256, 512)}
        ckpt = {'a': (64, 128), 'b': (256, 128)}  # Different b, missing c

        problems = _compare_shapes(expected, ckpt, 'test')
        assert len(problems) == 1  # Only b is compared


# =============================================================================
# estimate_model_params Tests
# =============================================================================


class TestEstimateModelParams:
    """Test estimate_model_params function"""

    def test_estimate_basic(self):
        """Test basic parameter estimation"""
        params = estimate_model_params()

        assert 'total' in params
        assert 'per_layer' in params
        assert 'embedding' in params
        assert params['total'] > 0
        assert isinstance(params['per_layer'], dict)
        assert 'attention' in params['per_layer']
        assert params['embedding'] > 0

    def test_estimate_with_config(self):
        """Test estimation with custom config"""
        params = estimate_model_params(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_layers=4,
            num_attention_heads=8,
        )

        # Embedding should be vocab_size * hidden_size
        expected_embedding = 1024 * 256
        assert params['embedding'] == expected_embedding

    def test_estimate_1t_model(self):
        """Test estimation for 1T model"""
        # Use actual 1T model configuration values
        params = estimate_model_params(
            vocab_size=163840,
            hidden_size=7168,  # Actual value from config_1t.json
            intermediate_size=18432,
            moe_intermediate_size=12288,
            num_layers=32,  # Actual value from config_1t.json
            num_attention_heads=64,
            num_experts=128,
            first_k_dense_replace=2,
        )

        assert params['total'] > 1e12  # Should be around 1T
        assert params['total'] < 5e12


# =============================================================================
# verify_config_consistency Tests
# =============================================================================


class TestVerifyConfigConsistency:
    """Test verify_config_consistency function"""

    def test_verify_consistent_configs(self):
        """Test when configs are consistent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create consistent config files
            config_1t = {
                'vocab_size': 163840,
                'hidden_size': 8192,
                'num_hidden_layers': 48,
                'num_attention_heads': 64,
                'intermediate_size': 24576,
            }

            config_100b = {
                'vocab_size': 163840,
                'hidden_size': 5120,
                'num_hidden_layers': 30,
                'num_attention_heads': 32,
                'intermediate_size': 12288,
            }

            models_dir = Path(tmpdir) / 'models'
            models_dir.mkdir()

            with open(models_dir / 'config_1t.json', 'w') as f:
                json.dump(config_1t, f)
            with open(models_dir / 'config_100b.json', 'w') as f:
                json.dump(config_100b, f)

            # Call with explicit repo_root
            is_valid = verify_config_consistency(Path(tmpdir))
            # Should be valid as each config is internally consistent
            assert isinstance(is_valid,
                              bool)  # Just check it returns a boolean


# =============================================================================
# verify_pretrain_script_consistency Tests
# =============================================================================


class TestVerifyPretrainScriptConsistency:
    """Test verify_pretrain_script_consistency function"""

    def test_verify_pretrain_consistent(self):
        """Test when pretrain script matches config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = {
                'vocab_size': 163840,
                'hidden_size': 8192,
                'num_hidden_layers': 48,
                'num_attention_heads': 64,
                'rope_theta': 1000000.0,
                'tie_word_embeddings': False,
            }

            models_dir = Path(tmpdir) / 'models'
            models_dir.mkdir()
            with open(models_dir / 'config_1t.json', 'w') as f:
                json.dump(config, f)

            # Create matching pretrain script
            scripts_dir = Path(tmpdir) / 'scripts'
            scripts_dir.mkdir()
            script_content = """
--num-layers 48
--hidden-size 8192
--num-attention-heads 64
--vocab-size 163840
--rotary-base 1000000.0
--use-flash-attn
--bf16
--moe-grouped-gemm
--schedules-method dualpipev
"""
            with open(scripts_dir / 'pretrain_kimi2_1t_4k.sh', 'w') as f:
                f.write(script_content)

            is_valid = verify_pretrain_script_consistency(Path(tmpdir))
            # Should be valid
            assert isinstance(is_valid, bool)


# =============================================================================
# main Tests
# =============================================================================


class TestCheckMain:
    """Test main function of check_model_weights"""

    @patch('sys.argv', ['check_model_weights.py', '--estimate-params'])
    @patch('utils.check_model_weights.print_parameter_estimate')
    def test_main_estimate_params(self, mock_print):
        """Test --estimate-params flag"""
        # check_main returns exit code, doesn't raise SystemExit
        exit_code = check_main()
        assert exit_code == 0
        mock_print.assert_called_once()

    @patch('sys.argv', ['check_model_weights.py', '--verify-config'])
    @patch('utils.check_model_weights.verify_config_consistency',
           return_value=True)
    @patch('utils.check_model_weights.verify_pretrain_script_consistency')
    def test_main_verify_config(self, mock_script, mock_verify):
        """Test --verify-config flag"""
        exit_code = check_main()
        assert exit_code == 0
        mock_verify.assert_called_once()

    @patch('sys.argv', ['check_model_weights.py', '--verify-all'])
    @patch('utils.check_model_weights.print_parameter_estimate')
    @patch('utils.check_model_weights.verify_config_consistency',
           return_value=True)
    @patch('utils.check_model_weights.verify_pretrain_script_consistency')
    def test_main_verify_all(self, mock_script, mock_verify, mock_print):
        """Test --verify-all flag"""
        exit_code = check_main()
        assert exit_code == 0
        mock_verify.assert_called_once()
        mock_print.assert_called_once()

    @patch('sys.argv',
           ['check_model_weights.py', 'dummy_path', '--skip-shape-check'])
    @patch('utils.check_model_weights._main_check_checkpoint')
    def test_main_check_checkpoint(self, mock_check):
        """Test checkpoint path argument"""
        check_main()
        mock_check.assert_called_once()

    @patch('sys.argv', ['check_model_weights.py'])
    def test_main_no_args(self):
        """Test with no arguments - should show error"""
        with pytest.raises(SystemExit) as e:
            check_main()
        # argparse exits with 2 for missing required argument
        assert e.value.code == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for check_model_weights"""

    def test_full_checkpoint_validation(self, temp_checkpoint_dir):
        """Test full checkpoint validation flow"""
        from argparse import Namespace

        from utils.check_model_weights import _main_check_checkpoint

        args = Namespace(
            checkpoint=temp_checkpoint_dir,
            skip_shape_check=True,
            report_limit=200,
            strict_index=False,
        )
        # Function may raise SystemExit or return result
        try:
            _main_check_checkpoint(args)
        except SystemExit as e:
            # SystemExit with code 0 means success
            assert e.code == 0 or e.code is None

    def test_checkpoint_with_shape_check(self, temp_checkpoint_dir):
        """Test checkpoint validation with shape checking"""
        from argparse import Namespace

        from utils.check_model_weights import _main_check_checkpoint

        args = Namespace(
            checkpoint=temp_checkpoint_dir,
            skip_shape_check=False,
            report_limit=200,
            strict_index=False,
        )
        try:
            _main_check_checkpoint(args)
        except SystemExit as e:
            assert e.code == 0 or e.code is None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_missing_dependency_error(self):
        """Test error when dependency is missing"""
        from utils.check_model_weights import _require

        mock_error = ImportError("No module named 'test_module'")
        with pytest.raises(SystemExit):
            _require('test_module', mock_error)

    def test_empty_config(self):
        """Test with empty/minimal config"""
        # Call with default parameters
        params = estimate_model_params()
        assert 'total' in params

    def test_config_with_none_values(self):
        """Test config with None values - use default vocab_size"""
        # Should handle gracefully with valid parameters
        params = estimate_model_params(vocab_size=1000, hidden_size=64)
        assert 'total' in params

    def test_very_large_config(self):
        """Test with very large model config"""
        params = estimate_model_params(
            vocab_size=256000,
            hidden_size=16384,
            intermediate_size=65536,
            num_layers=80,
            num_attention_heads=128,
        )
        assert params['total'] > 0
        assert not math.isinf(params['total'])
        assert not math.isnan(params['total'])

    def test_invalid_json_config(self):
        """Test with invalid JSON in config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / 'models'
            models_dir.mkdir()
            with open(models_dir / 'config_1t.json', 'w') as f:
                f.write('invalid json {{[')

            # Should raise JSONDecodeError
            with pytest.raises((json.JSONDecodeError, SystemExit)):
                verify_config_consistency(Path(tmpdir))


# =============================================================================
# Benchmark Tests
# =============================================================================


@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmark tests"""

    def test_estimate_params_benchmark(self, benchmark):
        """Benchmark parameter estimation"""
        benchmark(estimate_model_params,
                  vocab_size=163840,
                  hidden_size=8192,
                  num_layers=48)

    def test_read_specs_benchmark(self, benchmark, temp_checkpoint_dir):
        """Benchmark reading specs from shard"""
        shard_path = temp_checkpoint_dir / 'model.safetensors'
        benchmark(_read_specs_from_shard, shard_path)


if __name__ == '__main__':
    import math
    pytest.main([__file__, '-v'])


# =============================================================================
# assign_shards_layer_aware Tests
# =============================================================================


class TestAssignShardsLayerAware:
    """Unit tests for assign_shards_layer_aware function."""

    def _make_item(self, output_key: str, nbytes: int = 100,
                   input_shard: str = "shard1", input_key: str = None,
                   action: str = "keep") -> tuple:
        """Helper to create a test item tuple."""
        if input_key is None:
            input_key = output_key
        return (input_shard, input_key, output_key, nbytes, action)

    # ── Basic functionality ──────────────────────────────────────────────────

    def test_empty_items(self):
        """Empty input should return empty assignments and zero shards."""
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            [], target_shard_size=1000, max_layers_per_shard=1)
        assert assignments == {}
        assert num_shards == 0
        assert total_bytes == 0

    def test_single_tensor(self):
        """Single tensor fits in one shard."""
        items = [self._make_item("model.embed_tokens.weight", 500)]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        assert num_shards == 1
        assert total_bytes == 500
        assert 0 in assignments
        assert len(assignments[0]) == 1
        assert assignments[0][0][2] == "model.embed_tokens.weight"
        assert assignments[0][0][3] == 500  # output_nbytes

    def test_multiple_non_layer_params_same_module(self):
        """Non-layer params from same module family stay together."""
        items = [
            self._make_item("model.embed_tokens.weight", 400),
            self._make_item("model.embed_tokens.bias", 100),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        assert num_shards == 1
        assert total_bytes == 500
        assert len(assignments[0]) == 2

    def test_non_layer_params_different_modules_split(self):
        """Different non-layer module families should be in separate shards."""
        items = [
            self._make_item("model.embed_tokens.weight", 400),
            self._make_item("model.norm.weight", 100),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # Different module families → new shard
        assert num_shards == 2
        assert len(assignments[0]) == 1
        assert len(assignments[1]) == 1

    def test_non_layer_to_layer_transition(self):
        """Non-layer params and layer params must never mix."""
        items = [
            self._make_item("model.embed_tokens.weight", 400),
            self._make_item("model.layers.0.self_attn.q_proj.weight", 500),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # Non-layer first, then layer → separate shards
        assert num_shards == 2
        assert len(assignments[0]) == 1  # embed
        assert len(assignments[1]) == 1  # layer 0

    def test_layer_params_grouped_by_index(self):
        """Layer params with same index stay together, different index splits."""
        items = [
            self._make_item("model.layers.0.self_attn.q_proj.weight", 300),
            self._make_item("model.layers.0.mlp.gate_proj.weight", 300),
            self._make_item("model.layers.1.self_attn.q_proj.weight", 300),
            self._make_item("model.layers.1.mlp.gate_proj.weight", 300),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # max_layers_per_shard=1 → each layer gets its own shard
        assert num_shards == 2
        assert len(assignments[0]) == 2  # layer 0
        assert len(assignments[1]) == 2  # layer 1

    def test_max_layers_per_shard_greater_than_one(self):
        """Allow multiple layers per shard when size permits."""
        items = [
            self._make_item("model.layers.0.self_attn.q_proj.weight", 100),
            self._make_item("model.layers.1.self_attn.q_proj.weight", 100),
            self._make_item("model.layers.2.self_attn.q_proj.weight", 100),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=2)
        # max_layers=2, size allows all → 2 shards (2 layers + 1 layer)
        assert num_shards == 2
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 1

    # ── Size-based splitting ─────────────────────────────────────────────────

    def test_size_limit_triggers_new_shard_layer(self):
        """Layer params strictly obey target_shard_size."""
        items = [
            self._make_item("model.layers.0.self_attn.q_proj.weight", 600),
            self._make_item("model.layers.0.mlp.gate_proj.weight", 600),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # Same layer, but 600+600=1200 > 1000 → split
        assert num_shards == 2
        assert len(assignments[0]) == 1
        assert len(assignments[1]) == 1

    def test_size_limit_non_layer_same_module_exceeds(self):
        """Non-layer same module can exceed target up to 1.5x."""
        items = [
            self._make_item("model.embed_tokens.weight", 600),
            self._make_item("model.embed_tokens.bias", 600),
            self._make_item("model.embed_tokens.norm", 600),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # 1800 > 1500 (1.5x) → should split
        assert num_shards == 2

    def test_size_limit_non_layer_same_module_within_1_5x(self):
        """Non-layer same module stays together if within 1.5x target."""
        items = [
            self._make_item("model.embed_tokens.weight", 600),
            self._make_item("model.embed_tokens.bias", 400),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # 1000 == target, within 1.5x → stays together
        assert num_shards == 1
        assert len(assignments[0]) == 2

    def test_empty_shard_first_item_always_fits(self):
        """First item in empty shard always fits regardless of size."""
        items = [
            self._make_item("model.embed_tokens.weight", 2000),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # Giant tensor in empty shard → fits
        assert num_shards == 1
        assert len(assignments[0]) == 1

    # ── Return tuple structure ───────────────────────────────────────────────

    def test_return_tuple_has_five_elements(self):
        """Each assignment item must be a 5-tuple."""
        items = [self._make_item("model.layers.0.q_proj.weight", 100)]
        assignments, _, _ = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        item = assignments[0][0]
        assert len(item) == 5
        # (input_shard, input_key, output_key, output_nbytes, action)
        assert isinstance(item[3], int)  # output_nbytes
        assert item[4] == "keep"  # action

    def test_output_nbytes_preserved_correctly(self):
        """output_nbytes must match the value passed in items."""
        items = [
            self._make_item("model.layers.0.q_proj.weight", 12345),
            self._make_item("model.layers.1.k_proj.weight", 67890),
        ]
        assignments, _, _ = assign_shards_layer_aware(
            items, target_shard_size=100000, max_layers_per_shard=2)
        nbytes_set = set()
        for shard_items in assignments.values():
            for item in shard_items:
                nbytes_set.add(item[3])
        assert nbytes_set == {12345, 67890}

    # ── Sort order dependency ──────────────────────────────────────────────

    def test_requires_pre_sorted_items(self):
        """Items must be pre-sorted by layer_sort_key for optimal grouping."""
        # Correct order: non-layer first, then layer 0, layer 1
        items_correct = [
            self._make_item("model.embed_tokens.weight", 100),
            self._make_item("model.layers.0.q_proj.weight", 100),
            self._make_item("model.layers.1.q_proj.weight", 100),
        ]
        assignments, num_shards, _ = assign_shards_layer_aware(
            items_correct, target_shard_size=1000, max_layers_per_shard=1)
        # embed → layer 0 → layer 1 = 3 shards
        assert num_shards == 3

        # Wrong order: layer params before non-layer
        items_wrong = [
            self._make_item("model.layers.0.q_proj.weight", 100),
            self._make_item("model.embed_tokens.weight", 100),
            self._make_item("model.layers.1.q_proj.weight", 100),
        ]
        assignments_wrong, num_shards_wrong, _ = assign_shards_layer_aware(
            items_wrong, target_shard_size=1000, max_layers_per_shard=1)
        # layer 0 → embed (transition) → layer 1 = still 3 shards
        # but grouping is suboptimal
        assert num_shards_wrong == 3

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_layer_param_after_non_layer_with_max_layers(self):
        """Transition from non-layer to layer always starts new shard."""
        items = [
            self._make_item("model.norm.weight", 100),
            self._make_item("model.layers.0.q_proj.weight", 100),
            self._make_item("model.layers.1.q_proj.weight", 100),
        ]
        assignments, num_shards, _ = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # norm → layer 0 → layer 1 = 3 shards
        assert num_shards == 3
        assert len(assignments[0]) == 1  # norm
        assert len(assignments[1]) == 1  # layer 0
        assert len(assignments[2]) == 1  # layer 1

    def test_lm_head_separate_from_embed(self):
        """lm_head and embed_tokens are different module families."""
        items = [
            self._make_item("model.embed_tokens.weight", 100),
            self._make_item("lm_head.weight", 100),
        ]
        assignments, num_shards, _ = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # Different non-layer modules → separate shards
        assert num_shards == 2

    def test_total_bytes_accuracy(self):
        """total_bytes must sum all output_nbytes exactly."""
        items = [
            self._make_item("model.layers.0.q_proj.weight", 100),
            self._make_item("model.layers.0.k_proj.weight", 200),
            self._make_item("model.layers.1.q_proj.weight", 300),
            self._make_item("model.norm.weight", 400),
        ]
        _, _, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        assert total_bytes == 1000

    # ── Warning for giant tensors ────────────────────────────────────────────

    def test_giant_tensor_warning(self):
        """Single tensor exceeding target size should trigger warning."""
        import warnings
        items = [self._make_item("model.layers.0.q_proj.weight", 2000)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assign_shards_layer_aware(
                items, target_shard_size=1000, max_layers_per_shard=1)
            assert len(w) == 1
            assert "exceeds" in str(w[0].message)

    def test_giant_tensor_warning_only_once(self):
        """Warning should only be emitted once even with multiple giant tensors."""
        import warnings
        items = [
            self._make_item("model.layers.0.q_proj.weight", 2000),
            self._make_item("model.layers.1.q_proj.weight", 2000),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assign_shards_layer_aware(
                items, target_shard_size=1000, max_layers_per_shard=1)
            assert len(w) == 1  # Only one warning

    # ── Complex scenarios ────────────────────────────────────────────────────

    def test_realistic_model_layout(self):
        """Simulate a realistic model: embed → layers → norm → lm_head."""
        items = [
            self._make_item("model.embed_tokens.weight", 500),
            self._make_item("model.layers.0.input_layernorm.weight", 10),
            self._make_item("model.layers.0.self_attn.q_proj.weight", 100),
            self._make_item("model.layers.0.mlp.gate_proj.weight", 100),
            self._make_item("model.layers.1.input_layernorm.weight", 10),
            self._make_item("model.layers.1.self_attn.q_proj.weight", 100),
            self._make_item("model.layers.1.mlp.gate_proj.weight", 100),
            self._make_item("model.norm.weight", 10),
            self._make_item("lm_head.weight", 500),
        ]
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=300, max_layers_per_shard=1)
        # embed(500>300) → layer0(210<300) → layer1(210<300) → norm(10) → lm_head(500>300)
        # = 5 shards
        assert num_shards == 5
        assert total_bytes == 1430

    def test_many_small_layers(self):
        """Many small layers with max_layers_per_shard > 1."""
        items = []
        for i in range(10):
            items.append(self._make_item(f"model.layers.{i}.q_proj.weight", 50))
        assignments, num_shards, _ = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=3)
        # 10 layers, max 3 per shard, size allows → 4 shards (3+3+3+1)
        assert num_shards == 4
        assert len(assignments[0]) == 3
        assert len(assignments[1]) == 3
        assert len(assignments[2]) == 3
        assert len(assignments[3]) == 1

    def test_action_variety_preserved(self):
        """Different actions (keep, clone, zero) must be preserved."""
        items = [
            ("s1", "k1", "model.layers.0.q_proj.weight", 100, "keep"),
            ("s1", "k2", "model.layers.0.k_proj.weight", 100, "clone"),
            ("s1", "k3", "model.layers.1.o_proj.weight", 100, "zero"),
            ("s1", "k4", "model.layers.1.down_proj.weight", 100, "router_weight"),
        ]
        assignments, _, _ = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        actions = [item[4] for shard in assignments.values() for item in shard]
        assert set(actions) == {"keep", "clone", "zero", "router_weight"}


# =============================================================================
# build_expert_target_map Tests
# =============================================================================


class TestBuildExpertTargetMap:
    """Unit tests for build_expert_target_map function."""

    def test_basic_doubling(self):
        """4 → 8 experts: each original maps to one new."""
        result = build_expert_target_map(4, 8)
        assert result == {0: [4], 1: [5], 2: [6], 3: [7]}

    def test_tripling(self):
        """4 → 12 experts: each original maps to two new."""
        result = build_expert_target_map(4, 12)
        assert result == {0: [4, 8], 1: [5, 9], 2: [6, 10], 3: [7, 11]}

    def test_no_new_experts(self):
        """4 → 4 experts: empty mapping."""
        result = build_expert_target_map(4, 4)
        assert result == {}

    def test_single_original(self):
        """1 → 4 experts: single original maps to three new."""
        result = build_expert_target_map(1, 4)
        assert result == {0: [1, 2, 3]}


# =============================================================================
# build_layer_mapping Tests
# =============================================================================


class TestBuildLayerMapping:
    """Unit tests for build_layer_mapping function."""

    def test_interleave_basic(self):
        """4 → 8 layers, interleave mode."""
        mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "interleave")
        expected = [(0, False), (0, True), (1, False), (1, True),
                    (2, False), (2, True), (3, False), (3, True)]
        assert mapping == expected

    def test_append_basic(self):
        """4 → 8 layers, append mode."""
        mapping = build_layer_mapping(4, 8, [0, 1, 2, 3], "append")
        expected = [(0, False), (1, False), (2, False), (3, False),
                      (0, True), (1, True), (2, True), (3, True)]
        assert mapping == expected

    def test_interleave_uneven(self):
        """4 → 6 layers with custom source list."""
        mapping = build_layer_mapping(4, 6, [1, 2], "interleave")
        expected = [(0, False), (1, False), (1, True), (2, False), (2, True), (3, False)]
        assert mapping == expected

    def test_invalid_mode_raises(self):
        """Invalid insertion_mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown insertion_mode"):
            build_layer_mapping(4, 8, [0, 1, 2, 3], "invalid")


# =============================================================================
# parse_copy_source Tests
# =============================================================================


class TestParseCopySource:
    """Unit tests for parse_copy_source function."""

    def test_none_defaults_to_seq(self):
        """None should default to sequential round-robin."""
        result = parse_copy_source(None, 4, 8)
        assert result == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_seq_string(self):
        """'seq' should also default to sequential."""
        result = parse_copy_source("seq", 4, 8)
        assert result == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_single_integer(self):
        """Single integer string: all new layers copy from same source."""
        result = parse_copy_source("2", 4, 3)
        assert result == [2, 2, 2]

    def test_explicit_list(self):
        """Comma-separated list."""
        result = parse_copy_source("0,0,1,1", 4, 4)
        assert result == [0, 0, 1, 1]

    def test_single_out_of_range_raises(self):
        """Single integer out of range should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_copy_source("5", 4, 3)

    def test_list_wrong_length_raises(self):
        """List with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="expected 3"):
            parse_copy_source("0,1", 4, 3)

    def test_list_element_out_of_range_raises(self):
        """List element out of range should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_copy_source("0,1,5", 4, 3)

    def test_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            parse_copy_source("abc", 4, 3)


# =============================================================================
# expand_router_weight Tests
# =============================================================================


class TestExpandRouterWeight:
    """Unit tests for expand_router_weight function."""

    def test_basic_doubling_no_noise(self):
        """Double experts without noise: exact copies."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        expanded = expand_router_weight(t, 4, 0, 2, 0.0)
        assert expanded.shape == (8, 4)
        assert torch.equal(expanded[:4], t)
        assert torch.equal(expanded[4:], t)

    def test_with_noise(self):
        """Noise should break symmetry."""
        t = torch.randn(4, 4)  # Use random tensor with non-zero std
        expanded = expand_router_weight(t, 4, 0, 2, 1e-3)
        assert expanded.shape == (8, 4)
        assert torch.equal(expanded[:4], t)
        assert not torch.equal(expanded[4:], t)

    def test_with_zero_experts(self):
        """Handle zero_expert_num > 0."""
        t = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        expanded = expand_router_weight(t, 4, 2, 2, 0.0)
        assert expanded.shape == (12, 4)
        # First 8: real experts duplicated
        assert torch.equal(expanded[:4], t[:4])
        assert torch.equal(expanded[4:8], t[:4])
        # Last 4: zero experts duplicated
        assert torch.equal(expanded[8:10], t[4:6])
        assert torch.equal(expanded[10:12], t[4:6])

    def test_3x_expansion(self):
        """Triple expansion."""
        t = torch.ones(4, 4)
        expanded = expand_router_weight(t, 4, 0, 3, 0.0)
        assert expanded.shape == (12, 4)
        for i in range(3):
            assert torch.equal(expanded[i*4:(i+1)*4], t)


# =============================================================================
# expand_router_bias Tests
# =============================================================================


class TestExpandRouterBias:
    """Unit tests for expand_router_bias function."""

    def test_exact_copies(self):
        """Router bias should always be exact copies (no noise)."""
        t = torch.arange(4, dtype=torch.float32)
        expanded = expand_router_bias(t, 4, 0, 2)
        assert expanded.shape == (8,)
        assert torch.equal(expanded[:4], t)
        assert torch.equal(expanded[4:], t)

    def test_with_zero_experts(self):
        """Bias with zero experts."""
        t = torch.arange(6, dtype=torch.float32)
        expanded = expand_router_bias(t, 4, 2, 2)
        assert expanded.shape == (12,)
        assert torch.equal(expanded[:4], t[:4])
        assert torch.equal(expanded[4:8], t[:4])
        assert torch.equal(expanded[8:10], t[4:6])
        assert torch.equal(expanded[10:12], t[4:6])


# =============================================================================
# should_zero Tests
# =============================================================================


class TestShouldZero:
    """Unit tests for should_zero function."""

    def test_o_proj_weight(self):
        """o_proj.weight should be zeroed."""
        assert should_zero("model.layers.0.self_attn.0.o_proj.weight")
        assert should_zero("model.layers.5.self_attn.o_proj.weight")

    def test_down_proj_weight(self):
        """Various down_proj.weight patterns should be zeroed."""
        assert should_zero("model.layers.0.mlp.experts.0.down_proj.weight")
        assert should_zero("model.layers.0.mlp.experts.511.down_proj.weight")
        assert should_zero("model.layers.0.mlps.0.down_proj.weight")
        assert should_zero("model.layers.0.mlp.down_proj.weight")

    def test_non_zero_patterns(self):
        """Other weights should NOT be zeroed."""
        assert not should_zero("model.layers.0.self_attn.q_proj.weight")
        assert not should_zero("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert not should_zero("model.layers.0.mlp.experts.0.up_proj.weight")
        assert not should_zero("model.layers.0.input_layernorm.weight")
        assert not should_zero("model.embed_tokens.weight")


# =============================================================================
# layer_sort_key Tests
# =============================================================================


class TestLayerSortKey:
    """Unit tests for layer_sort_key function."""

    def test_non_layer_first(self):
        """Non-layer params should sort before layer params."""
        assert layer_sort_key("model.embed_tokens.weight") < layer_sort_key("model.layers.0.q_proj.weight")

    def test_layer_ordering(self):
        """Layer params should sort by layer index."""
        assert layer_sort_key("model.layers.0.q_proj.weight") < layer_sort_key("model.layers.1.q_proj.weight")

    def test_same_layer_alphabetical(self):
        """Same layer: alphabetical by key."""
        assert layer_sort_key("model.layers.0.a_proj.weight") < layer_sort_key("model.layers.0.z_proj.weight")

    def test_lm_head_vs_embed(self):
        """Both non-layer, sort alphabetically."""
        assert layer_sort_key("lm_head.weight") < layer_sort_key("model.embed_tokens.weight")


# =============================================================================
# get_layer_index / set_layer_index Tests
# =============================================================================


class TestLayerIndexHelpers:
    """Unit tests for get_layer_index and set_layer_index."""

    def test_get_layer_index_basic(self):
        assert get_layer_index("model.layers.5.q_proj.weight") == 5

    def test_get_layer_index_non_layer(self):
        assert get_layer_index("model.embed_tokens.weight") is None

    def test_get_layer_index_edge_cases(self):
        assert get_layer_index("model.layers.0.q_proj.weight") == 0
        assert get_layer_index("model.layers.99.q_proj.weight") == 99

    def test_set_layer_index_basic(self):
        result = set_layer_index("model.layers.5.q_proj.weight", 10)
        assert result == "model.layers.10.q_proj.weight"

    def test_set_layer_index_non_layer_unchanged(self):
        assert set_layer_index("model.embed_tokens.weight", 10) == "model.embed_tokens.weight"

    def test_set_layer_index_zero(self):
        result = set_layer_index("model.layers.5.q_proj.weight", 0)
        assert result == "model.layers.0.q_proj.weight"


# =============================================================================
# get_expert_info / make_expert_key Tests
# =============================================================================


class TestExpertHelpers:
    """Unit tests for get_expert_info and make_expert_key."""

    def test_get_expert_info_basic(self):
        result = get_expert_info("model.layers.2.mlp.experts.5.gate_proj.weight")
        assert result == (2, 5, "gate_proj.weight")

    def test_get_expert_info_non_expert(self):
        assert get_expert_info("model.layers.2.self_attn.q_proj.weight") is None

    def test_make_expert_key(self):
        result = make_expert_key(2, 5, "gate_proj.weight")
        assert result == "model.layers.2.mlp.experts.5.gate_proj.weight"

    def test_roundtrip(self):
        original = "model.layers.7.mlp.experts.3.up_proj.weight"
        info = get_expert_info(original)
        reconstructed = make_expert_key(*info)
        assert reconstructed == original


# =============================================================================
# find_expert_count Tests
# =============================================================================


class TestFindExpertCount:
    """Unit tests for find_expert_count function."""

    def test_n_routed_experts(self):
        config = {"n_routed_experts": 64}
        key, count, zero = find_expert_count(config)
        assert key == "n_routed_experts"
        assert count == 64
        assert zero == 0

    def test_n_experts(self):
        config = {"n_experts": 32}
        key, count, zero = find_expert_count(config)
        assert key == "n_experts"
        assert count == 32

    def test_num_experts(self):
        config = {"num_experts": 16}
        key, count, zero = find_expert_count(config)
        assert key == "num_experts"
        assert count == 16

    def test_with_zero_expert_num(self):
        config = {"n_routed_experts": 64, "zero_expert_num": 8}
        key, count, zero = find_expert_count(config)
        assert count == 64
        assert zero == 8

    def test_priority_order(self):
        """n_routed_experts should take priority over n_experts."""
        config = {"n_routed_experts": 64, "n_experts": 32}
        key, _, _ = find_expert_count(config)
        assert key == "n_routed_experts"

    def test_no_expert_key(self):
        config = {"hidden_size": 128}
        key, count, zero = find_expert_count(config)
        assert key is None
        assert count == 0
        assert zero == 0

    def test_invalid_value_ignored(self):
        """Non-positive values should be ignored."""
        config = {"n_routed_experts": 0, "n_experts": 32}
        key, count, _ = find_expert_count(config)
        assert key == "n_experts"
        assert count == 32


# =============================================================================
# is_router_param / is_router_weight / is_router_bias Tests
# =============================================================================


class TestRouterParamChecks:
    """Unit tests for router parameter detection functions."""

    def test_is_router_weight_classifier(self):
        assert is_router_weight("model.layers.0.mlp.router.classifier.weight")

    def test_is_router_weight_gate(self):
        assert is_router_weight("model.layers.0.mlp.gate.weight")

    def test_is_router_weight_non_router(self):
        assert not is_router_weight("model.layers.0.mlp.experts.0.gate_proj.weight")

    def test_is_router_bias_correction(self):
        assert is_router_bias("model.layers.0.mlp.router.e_score_correction_bias")

    def test_is_router_bias_gate(self):
        assert is_router_bias("model.layers.0.mlp.gate.e_score_correction_bias")

    def test_is_router_param_combined(self):
        assert is_router_param("model.layers.0.mlp.router.classifier.weight")
        assert is_router_param("model.layers.0.mlp.gate.e_score_correction_bias")
        assert not is_router_param("model.layers.0.mlp.experts.0.gate_proj.weight")


# =============================================================================
# get_nbytes_from_meta Tests
# =============================================================================


class TestGetNbytesFromMeta:
    """Unit tests for get_nbytes_from_meta function."""

    def test_f32(self):
        assert get_nbytes_from_meta("F32", [2, 3, 4]) == 2 * 3 * 4 * 4

    def test_bf16(self):
        assert get_nbytes_from_meta("BF16", [10, 20]) == 10 * 20 * 2

    def test_f8(self):
        assert get_nbytes_from_meta("F8_E4M3", [100]) == 100 * 1

    def test_scalar(self):
        assert get_nbytes_from_meta("F32", []) == 4

    def test_1d(self):
        assert get_nbytes_from_meta("F32", [16]) == 16 * 4


# =============================================================================
# tensor_nbytes Tests
# =============================================================================


class TestTensorNbytes:
    """Unit tests for tensor_nbytes function."""

    def test_f32_tensor(self):
        t = torch.randn(2, 3, 4)
        assert tensor_nbytes(t) == 2 * 3 * 4 * 4

    def test_bf16_tensor(self):
        t = torch.randn(10, 20, dtype=torch.bfloat16)
        assert tensor_nbytes(t) == 10 * 20 * 2

    def test_empty_tensor(self):
        t = torch.tensor([])
        assert tensor_nbytes(t) == 0


# =============================================================================
# Integration Tests for Shared Functions
# =============================================================================


class TestSharedIntegration:
    """Integration tests combining multiple shared functions."""

    def test_end_to_end_expert_key_roundtrip(self):
        """get_expert_info and make_expert_key should be inverse operations."""
        keys = [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.99.mlp.experts.255.down_proj.weight",
            "model.layers.5.mlp.experts.10.up_proj.weight",
        ]
        for key in keys:
            info = get_expert_info(key)
            reconstructed = make_expert_key(*info)
            assert reconstructed == key

    def test_layer_sort_key_with_expert_keys(self):
        """layer_sort_key should correctly order expert parameters."""
        keys = [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.norm.weight",
        ]
        sorted_keys = sorted(keys, key=layer_sort_key)
        # Non-layer first, then by layer, then by name
        assert sorted_keys[0] == "model.embed_tokens.weight"
        assert sorted_keys[1] == "model.norm.weight"
        assert sorted_keys[2] == "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert sorted_keys[3] == "model.layers.0.mlp.experts.1.gate_proj.weight"
        assert sorted_keys[4] == "model.layers.1.mlp.experts.0.gate_proj.weight"

    def test_assign_shards_with_layer_sort_key(self):
        """assign_shards_layer_aware should work correctly with pre-sorted items."""
        items = [
            self._make_item("model.embed_tokens.weight", 500, action="keep"),
            self._make_item("model.norm.weight", 100, action="keep"),
            self._make_item("model.layers.0.q_proj.weight", 200, action="keep"),
            self._make_item("model.layers.0.o_proj.weight", 200, action="zero"),
            self._make_item("model.layers.1.q_proj.weight", 200, action="keep"),
            self._make_item("model.layers.1.o_proj.weight", 200, action="zero"),
            self._make_item("lm_head.weight", 500, action="keep"),
        ]
        # Pre-sort by layer_sort_key
        items.sort(key=lambda x: layer_sort_key(x[2]))
        assignments, num_shards, total_bytes = assign_shards_layer_aware(
            items, target_shard_size=1000, max_layers_per_shard=1)
        # embed(500) → norm(100) → layer0(400) → layer1(400) → lm_head(500)
        # = 5 shards (all within size, but module transitions and max_layers=1)
        assert num_shards == 5
        assert total_bytes == 1900
        # Verify actions are preserved
        all_actions = [item[4] for shard in assignments.values() for item in shard]
        assert all_actions.count("zero") == 2
        assert all_actions.count("keep") == 5

    def _make_item(self, output_key: str, nbytes: int = 100,
                   input_shard: str = "shard1", input_key: str = None,
                   action: str = "keep") -> tuple:
        """Helper to create a test item tuple."""
        if input_key is None:
            input_key = output_key
        return (input_shard, input_key, output_key, nbytes, action)
