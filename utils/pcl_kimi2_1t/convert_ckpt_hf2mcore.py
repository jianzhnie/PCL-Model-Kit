"""
Huggingface (HF) 到 Megatron-Core (MCore) 模型权重转换

基于 Kimi2-1T 模型架构，支持：
- GQA (Grouped Query Attention)
- MoE (Mixture of Experts)  with 128 experts
- TP (Tensor Parallel) / PP (Pipeline Parallel) / EP (Expert Parallel)
- DualPipeV 调度算法
- QK LayerNorm

参考：
- scripts/pretrain_kimi2_1t_4k.sh (训练配置)
- megatron_model.py (MCore 模型结构)
- model_param_mapping.json (参数映射)
- models/config.json (HF 模型配置)
"""

import argparse
import hashlib
import json
import logging as logger
import os
import time
import types
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from multiprocessing import get_context
from typing import Any, Optional, Union

import torch
from safetensors import safe_open

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value == '':
        return None
    return list(map(int, value.split(',')))


def _ensure_iter_path(save_dir: str) -> str:
    iter_dir = os.path.join(save_dir, 'iter_0000001')
    os.makedirs(iter_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'latest_checkpointed_iteration.txt')
    if not os.path.isfile(latest_path):
        with open(latest_path, 'w') as f:
            f.write('1')
    return iter_dir


def _dtype_from_str(s: str) -> torch.dtype:
    v = (s or '').lower()
    if v in ('fp16', 'float16'):
        return torch.float16
    if v in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if v in ('fp32', 'float32'):
        return torch.float32
    raise ValueError(f'不支持的 dtype: {s}')


def _sha256_file(path: str, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_sha256_manifest(save_dir: str, out_path: str | None) -> str | None:
    if not out_path:
        return None
    files: list[str] = []
    for root, _, fns in os.walk(save_dir):
        for fn in fns:
            if fn.endswith('.pt') or fn.endswith('.txt'):
                files.append(os.path.join(root, fn))
    files.sort()
    payload: dict[str, str] = {}
    for p in files:
        rel = os.path.relpath(p, save_dir)
        payload[rel] = _sha256_file(p)
    with open(out_path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int,
               ep: int) -> str:
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


class CkptConvert:

    def __init__(
        self,
        hf_model_path: str,
        mg_save_path: str,
        num_layers: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        first_k_dense_replace: int,
        hidden_size: int,
        ffn_hidden_size: int | None,
        moe_ffn_hidden_size: int | None,
        vocab_size: int | None,
        num_experts: int,
        num_attention_heads: int,
        num_query_groups: int,
        qk_head_dim: int,
        v_head_dim: int,
        moe_grouped_gemm: bool,
        expert_tp_size: int = 1,
        schedules_method: str | None = None,
        vpp_stage: int | None = None,
        num_layer_list: str | None = None,
        noop_layers: str | None = None,
        qlora_nf4: bool = False,
        rotary_base: float = 50000.0,
        print_init_summary: bool = True,
        pp_workers: int = 1,
        save_workers: int = 0,
        cast_dtype: str | None = None,
        tie_word_embeddings: bool | None = None,
        hf_io_threads: int = 1,
        qk_layernorm: bool = False,
    ):
        self.verbose = os.environ.get('CKPT_CONVERT_VERBOSE', '1') != '0'
        self.log_file_load = os.environ.get('CKPT_CONVERT_LOG_FILE',
                                            '0') != '0'
        self.log_layer_progress = os.environ.get('CKPT_CONVERT_LOG_LAYER',
                                                 '1') != '0'
        self.log_save_progress = os.environ.get('CKPT_CONVERT_LOG_SAVE',
                                                '1') != '0'
        self.print_init_summary = print_init_summary
        self.pp_workers = int(pp_workers)
        self.save_workers = int(save_workers)
        self.hf_io_threads = max(1, int(hf_io_threads))
        self.cast_dtype = cast_dtype
        self._target_dtype = _dtype_from_str(
            cast_dtype) if cast_dtype else None
        self.tie_word_embeddings = tie_word_embeddings
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layers = num_layers
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.moe_grouped_gemm = moe_grouped_gemm
        self.expert_tp_size = max(1, int(expert_tp_size))
        if self.expert_tp_size > self.tp_size:
            raise ValueError(f'expert_tp_size ({self.expert_tp_size}) '
                             f'cannot exceed tp_size ({self.tp_size})')
        if self.tp_size % self.expert_tp_size != 0:
            raise ValueError(f'tp_size ({self.tp_size}) must be divisible by '
                             f'expert_tp_size ({self.expert_tp_size})')
        self.schedules_method = schedules_method
        self.dualpipe = schedules_method == 'dualpipev'
        self.vpp_stage = vpp_stage
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.qlora_nf4 = qlora_nf4
        self.rotary_base = rotary_base
        self.num_query_groups = num_query_groups
        self.qk_layernorm = qk_layernorm

        self.noop_layers_list = sorted(_parse_int_list(noop_layers) or [])
        if self.dualpipe:
            if vpp_stage is not None:
                raise ValueError('dualpipev 与 vpp-stage 不兼容')
            self.vpp_size = 2
            layers_each_pp = self.num_layers // self.pp_size
            if layers_each_pp % 2 != 0:
                raise ValueError('dualpipev 需要每个 PP 的层数为偶数')
            self.vpp_stage = layers_each_pp // 2
        elif vpp_stage is not None:
            self.vpp_size = (self.num_layers // self.pp_size) // vpp_stage
        else:
            self.vpp_size = None

        self._validate()
        self.weight_map = self._read_weight_map()
        self.layer_keys_map: dict[int, list[str]] = defaultdict(list)
        for k in self.weight_map.keys():
            if k.startswith('model.layers.'):
                layer_id = int(k.split('model.layers.')[1].split('.')[0])
                self.layer_keys_map[layer_id].append(k)

        # Detect MoE vs Dense layers from actual HF weight structure
        self._detect_moe_structure()

        self.iter_path = _ensure_iter_path(self.mg_save_path)

        if self.vpp_stage is None:
            self.pprank_layer_idxs: dict[int, list[int]] = defaultdict(list)
            self._build_pprank_layer_map()
        else:
            self.vpprank_layer_idxs: dict[int,
                                          dict[int,
                                               list[int]]] = defaultdict(dict)
            self._build_vpprank_layer_map()

        if self.verbose and self.print_init_summary:
            shard_files = sorted(set(self.weight_map.values()))
            num_noop = len(self.noop_layers_list)
            num_real_layers = self.num_layers - num_noop
            logger.info(
                'HF->Megatron: layers=%d (real=%d noop=%d) tp=%d pp=%d ep=%d vpp=%s dualpipe=%s',
                int(self.num_layers),
                int(num_real_layers),
                int(num_noop),
                int(self.tp_size),
                int(self.pp_size),
                int(self.ep_size),
                str(self.vpp_size),
                str(self.dualpipe),
            )
            logger.info(
                'HF->Megatron: hidden=%d heads=%d qk=%d v=%d rotary_base=%s first_k_dense=%d',
                int(self.hidden_size),
                int(self.num_attention_heads),
                int(self.qk_head_dim),
                int(self.v_head_dim),
                str(self.rotary_base),
                int(self.first_k_dense_replace),
            )
            logger.info(
                'HF weights: tensors=%d shards=%d (e.g. %s)',
                len(self.weight_map),
                len(shard_files),
                shard_files[:min(5, len(shard_files))],
            )
            has_gqa = 'model.layers.0.self_attn.q_proj.weight' in self.weight_map and 'model.layers.0.self_attn.k_proj.weight' in self.weight_map
            logger.info('HF attention format: gqa=%s', str(has_gqa))
            if self.vpp_stage is None:
                for pp_rank in range(self.pp_size):
                    ls = self.pprank_layer_idxs[pp_rank]
                    head = ls[:min(6, len(ls))]
                    tail = ls[max(0, len(ls) - 6):]
                    logger.info(
                        'PP layer map: pp=%d layers=%d head=%s tail=%s',
                        pp_rank,
                        len(ls),
                        head,
                        tail,
                    )
            else:
                for pp_rank in range(self.pp_size):
                    for vpp_rank in range(self.vpp_size):
                        ls = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                        head = ls[:min(6, len(ls))]
                        tail = ls[max(0, len(ls) - 6):]
                        logger.info(
                            'VPP layer map: pp=%d vpp=%d layers=%d head=%s tail=%s',
                            pp_rank,
                            vpp_rank,
                            len(ls),
                            head,
                            tail,
                        )

    def run_one_pp_rank(self, pp_rank: int) -> None:
        if self.vpp_stage is None:
            logger.info('pp_rank=%s/%s', pp_rank, self.pp_size)
            mg_model: dict[int, dict[int, dict[
                str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
            t0 = time.time()
            weights = self._load_matched_hf_weights(pp_rank, None)
            if pp_rank == 0:
                self._set_preprocess(weights, mg_model)
            for local_layer_idx, hf_layer in enumerate(
                    self.pprank_layer_idxs[pp_rank]):
                lt0 = time.time()
                self._set_layer_norm(hf_layer, local_layer_idx, weights,
                                     mg_model)
                self._set_layer_attn(hf_layer, local_layer_idx, weights,
                                     mg_model)
                self._set_layer_mlp(hf_layer, local_layer_idx, weights,
                                    mg_model)
                if self.log_layer_progress:
                    logger.info('Converted layer hf=%d local=%d pp=%d (%.2fs)',
                                int(hf_layer), int(local_layer_idx),
                                int(pp_rank),
                                time.time() - lt0)
            if pp_rank == self.pp_size - 1 and not self.dualpipe:
                self._set_postprocess(weights, mg_model)
            self._assert_consumed(weights, f'pp_rank={pp_rank}')
            self._save_pp_rank(pp_rank, mg_model, vpp=False)
            if self.verbose:
                logger.info('pp_rank=%d done (%.2fs)', int(pp_rank),
                            time.time() - t0)
            return

        logger.info(
            'pp_rank=%s/%s (vpp=%s stage=%s)',
            pp_rank,
            self.pp_size,
            self.vpp_size,
            self.vpp_stage,
        )
        mg_model: dict[int,
                       dict[int,
                            dict[int, dict[str, torch.Tensor]]]] = defaultdict(
                                lambda: defaultdict(lambda: defaultdict(dict)))
        for vpp_rank in range(self.vpp_size):
            t0 = time.time()
            weights = self._load_matched_hf_weights(pp_rank, vpp_rank)
            if pp_rank == 0 and vpp_rank == 0:
                self._set_preprocess(weights, mg_model[vpp_rank])
            if self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                self._set_postprocess(weights, mg_model[vpp_rank])

            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
            for local_layer_idx, hf_layer in enumerate(layer_list):
                lt0 = time.time()
                self._set_layer_norm(hf_layer, local_layer_idx, weights,
                                     mg_model[vpp_rank])
                self._set_layer_attn(hf_layer, local_layer_idx, weights,
                                     mg_model[vpp_rank])
                self._set_layer_mlp(hf_layer, local_layer_idx, weights,
                                    mg_model[vpp_rank])
                if self.log_layer_progress:
                    logger.info(
                        'Converted layer hf=%d local=%d pp=%d vpp=%d (%.2fs)',
                        int(hf_layer),
                        int(local_layer_idx),
                        int(pp_rank),
                        int(vpp_rank),
                        time.time() - lt0,
                    )

            if (not self.dualpipe) and (pp_rank == self.pp_size -
                                        1) and (vpp_rank == self.vpp_size - 1):
                self._set_postprocess(weights, mg_model[vpp_rank])
            self._assert_consumed(weights,
                                  f'pp_rank={pp_rank} vpp_rank={vpp_rank}')
            if self.verbose:
                logger.info('pp=%d vpp=%d done (%.2fs)', int(pp_rank),
                            int(vpp_rank),
                            time.time() - t0)
        self._save_pp_rank(pp_rank, mg_model, vpp=True)

    def _validate(self) -> None:
        if not os.path.isdir(self.hf_model_path):
            raise FileNotFoundError(self.hf_model_path)
        if self.num_layers <= 0:
            raise ValueError('num_layers 必须 > 0')
        if self.pp_size <= 0 or self.tp_size <= 0 or self.ep_size <= 0:
            raise ValueError('并行度必须 > 0')
        if self.num_layers % self.pp_size != 0 and self.num_layer_list is None:
            raise ValueError('num_layers 必须能整除 pp_size，或显式给定 num-layer-list')
        if self.first_k_dense_replace < 0 or self.first_k_dense_replace > self.num_layers:
            raise ValueError('first-k-dense-replace 非法')
        if self.num_experts is None:
            raise ValueError('num_experts 不能为空')
        if self.num_experts % self.ep_size != 0:
            raise ValueError('num_experts 必须能整除 ep_size')
        if self.num_layer_list is not None and self.vpp_stage is not None:
            raise ValueError('num-layer-list 与 vpp/dualpipev 不可同时配置')
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError('vocab-size 必须 > 0')
        if self.ffn_hidden_size is not None and self.ffn_hidden_size <= 0:
            raise ValueError('ffn-hidden-size 必须 > 0')
        if self.moe_ffn_hidden_size is not None and self.moe_ffn_hidden_size <= 0:
            raise ValueError('moe-ffn-hidden-size 必须 > 0')
        self._validate_attention_config()

    def _validate_attention_config(self) -> None:
        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f'num_attention_heads ({self.num_attention_heads}) '
                f'must be divisible by tp_size ({self.tp_size})')
        if self.num_query_groups % self.tp_size != 0:
            raise ValueError(f'num_query_groups ({self.num_query_groups}) '
                             f'must be divisible by tp_size ({self.tp_size})')
        if self.num_attention_heads % self.num_query_groups != 0:
            raise ValueError(
                f'num_attention_heads ({self.num_attention_heads}) '
                f'must be divisible by num_query_groups ({self.num_query_groups})'
            )

    def _detect_moe_structure(self) -> None:
        """Detect MoE vs Dense layers from actual HF weight structure.

        Inspects the HF weight_map to determine which layers have MoE
        weights (mlp.gate.weight) vs Dense weights (mlp.gate_proj.weight),
        and validates against first_k_dense_replace.

        Raises:
            ValueError: If the detected structure is inconsistent or
                conflicts with first_k_dense_replace.
        """
        self.hf_dense_layers: list[int] = []
        self.hf_moe_layers: list[int] = []
        self.hf_unknown_layers: list[int] = []

        if not self.weight_map:
            return

        for layer_id in sorted(
                set(
                    int(k.split('model.layers.')[1].split('.')[0])
                    for k in self.weight_map
                    if k.startswith('model.layers.'))):
            has_moe_router = (f'model.layers.{layer_id}.mlp.gate.weight'
                              in self.weight_map)
            has_dense_gate = (f'model.layers.{layer_id}.mlp.gate_proj.weight'
                              in self.weight_map)
            if has_moe_router:
                self.hf_moe_layers.append(layer_id)
            elif has_dense_gate:
                self.hf_dense_layers.append(layer_id)
            else:
                self.hf_unknown_layers.append(layer_id)

        if self.verbose and self.print_init_summary:
            logger.info(
                'HF structure: dense_layers=%s moe_layers=%s unknown=%s',
                self.hf_dense_layers[:20],
                self.hf_moe_layers[:20],
                self.hf_unknown_layers[:20],
            )

        # Validate against first_k_dense_replace
        expected_dense = set(range(self.first_k_dense_replace))
        actual_dense = set(self.hf_dense_layers)
        actual_moe = set(self.hf_moe_layers)

        if expected_dense != actual_dense and actual_moe:
            # Check if the detected MoE layers match what we expect
            expected_moe = set(
                range(self.first_k_dense_replace, self.num_layers)) - set(
                    self.noop_layers_list)
            wrong_dense = actual_moe & expected_dense
            wrong_moe = actual_dense & expected_moe
            if wrong_dense:
                logger.warning(
                    'HF layers %s are MoE but first_k_dense_replace=%d '
                    'says they should be dense. Check --first-k-dense-replace.',
                    sorted(wrong_dense), self.first_k_dense_replace)
            if wrong_moe:
                logger.warning(
                    'HF layers %s are Dense but first_k_dense_replace=%d '
                    'says they should be MoE. Check --first-k-dense-replace.',
                    sorted(wrong_moe), self.first_k_dense_replace)

        # Validate critical global weights
        if 'model.embed_tokens.weight' not in self.weight_map:
            raise KeyError('HF model missing model.embed_tokens.weight')
        if not self.tie_word_embeddings:
            if 'lm_head.weight' not in self.weight_map:
                raise KeyError('HF model missing lm_head.weight and '
                               'tie_word_embeddings is not enabled')
        if 'model.norm.weight' not in self.weight_map:
            raise KeyError(
                'HF model missing model.norm.weight (final layernorm)')

    def _build_checkpoint_args(self) -> argparse.Namespace:
        """Build checkpoint args namespace for Megatron's check_checkpoint_args.

        Megatron's checkpoint loading validates that the checkpoint args match
        the current training configuration via getattr(checkpoint_args, name).
        A plain dict fails because dicts don't support attribute access.
        """
        ns = argparse.Namespace()

        # Core model architecture
        ns.num_layers = self.num_layers
        ns.hidden_size = self.hidden_size
        ns.ffn_hidden_size = self.ffn_hidden_size or self.hidden_size * 4
        ns.num_attention_heads = self.num_attention_heads
        ns.num_query_groups = self.num_query_groups
        ns.kv_channels = self.qk_head_dim
        ns.qk_head_dim = self.qk_head_dim
        ns.v_head_dim = self.v_head_dim
        ns.seq_length = 4096
        ns.max_position_embeddings = 131072
        ns.vocab_size = self.vocab_size
        ns.padded_vocab_size = self.vocab_size
        ns.make_vocab_size_divisible_by = 1

        # Parallelism
        ns.tensor_model_parallel_size = self.tp_size
        ns.pipeline_model_parallel_size = self.pp_size
        ns.expert_model_parallel_size = self.ep_size
        ns.expert_tensor_parallel_size = self.expert_tp_size
        ns.context_parallel_size = 1

        # MoE
        ns.num_experts = self.num_experts
        ns.moe_grouped_gemm = self.moe_grouped_gemm
        ns.moe_ffn_hidden_size = self.moe_ffn_hidden_size
        ns.first_k_dense_replace = self.first_k_dense_replace
        ns.n_shared_experts = 1
        ns.moe_router_topk = 2
        ns.moe_router_num_groups = 8
        ns.moe_router_group_topk = 2
        ns.moe_router_topk_scaling_factor = 2.827
        ns.moe_router_enable_expert_bias = True
        ns.moe_token_dispatcher_type = 'alltoall'
        ns.seq_aux = True
        ns.norm_topk_prob = True

        # Optimizer
        ns.use_distributed_optimizer = True

        # MTP (Multi-Token Prediction)
        ns.mtp_num_layers = 0

        # Model format
        ns.use_mcore_models = True
        ns.use_legacy_models = False
        ns.untie_embeddings_and_output_weights = not self.tie_word_embeddings
        ns.swiglu = True
        ns.position_embedding_type = 'rope'
        ns.normalization = 'RMSNorm'
        ns.add_bias_linear = False
        ns.norm_epsilon = 1e-6

        # Data type
        target_dtype = self._target_dtype or torch.bfloat16
        ns.bf16 = (target_dtype == torch.bfloat16)
        ns.fp16 = (target_dtype == torch.float16)
        ns.params_dtype = target_dtype

        # Rotary
        ns.rotary_base = self.rotary_base
        ns.use_rotary_position_embeddings = True

        # QK LayerNorm
        ns.qk_layernorm = self.qk_layernorm

        # VPP / DualPipe
        if self.dualpipe:
            ns.schedules_method = 'dualpipev'
        if self.vpp_stage is not None:
            ns.num_layers_per_virtual_pipeline_stage = self.vpp_stage

        return ns

    def _log_expected_param_summary(self) -> None:
        """Log expected Megatron parameter names per (ep, tp) rank pair."""
        num_dense = min(self.first_k_dense_replace, self.num_layers)
        num_moe = self.num_layers - num_dense - len(self.noop_layers_list)

        # Per-layer param counts
        dense_params = 8  # input_layernorm, pre_mlp_layernorm, qkv, proj, q_ln, k_ln, fc1, fc2
        # MoE layer: dense_params (8) + router.weight + router.expert_bias +
        #            shared_fc1 + shared_fc2 + experts.weight1 + experts.weight2 = 12
        moe_params = 12
        if not self.moe_grouped_gemm:
            # non-grouped gemm uses local_experts instead of weight1/weight2
            moe_params = 8 + 2 * (self.num_experts // self.ep_size) + 4

        expected = 1 + num_dense * dense_params + num_moe * moe_params + 2
        # +1 for embedding, +2 for final_layernorm + output_layer

        if self.verbose:
            logger.info(
                'Expected params per (ep,tp) rank: ~%d '
                '(embedding=1, %d dense_layers*%d, %d moe_layers*%d, post=2)',
                expected, num_dense, dense_params, num_moe, moe_params)

    def _read_weight_map(self) -> dict[str, str]:
        """Read weight map from HF model directory.

        Tries to load from index.json first, then falls back to single safetensors file.

        Returns:
            Dictionary mapping tensor names to file names

        Raises:
            FileNotFoundError: If neither index.json nor single safetensors file exists
            json.JSONDecodeError: If index.json is malformed
            RuntimeError: If there's an error reading the safetensors file
        """
        index_path = os.path.join(self.hf_model_path,
                                  'model.safetensors.index.json')
        if os.path.isfile(index_path):
            if self.verbose:
                logger.info('Using HF index: %s', index_path)
            try:
                with open(index_path) as f:
                    data = json.load(f)
                    if 'weight_map' not in data:
                        raise ValueError(
                            f'Invalid index file: {index_path} missing "weight_map" key'
                        )
                    return data['weight_map']
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f'Malformed index file {index_path}: {e}',
                    e.doc,
                    e.pos,
                ) from e

        single = os.path.join(self.hf_model_path, 'model.safetensors')
        if os.path.isfile(single):
            if self.verbose:
                logger.info('Using single HF safetensors: %s', single)
            try:
                with safe_open(single, framework='pt', device='cpu') as f:
                    return {k: 'model.safetensors' for k in f.keys()}
            except Exception as e:
                raise RuntimeError(
                    f'Error reading safetensors file {single}: {e}') from e

        raise FileNotFoundError(
            f'找不到 HF safetensors index 或单文件: {self.hf_model_path}')

    def _assert_consumed(self, weights: dict[str, torch.Tensor],
                         context: str) -> None:
        if not weights:
            return
        keys = sorted(weights.keys())
        head = keys[:20]
        raise ValueError(f'{context} 未被消费的 HF 权重={len(keys)}: {head}')

    def _get_layer_files_map(self) -> dict[object, set[str]]:
        weight_map = self.weight_map
        layer_files_map: dict[object, set[str]] = defaultdict(set)
        for k, v in weight_map.items():
            if k.startswith('model.layers.'):
                layer_id = int(k.split('model.layers.')[1].split('.')[0])
                layer_files_map[layer_id].add(v)
            else:
                layer_files_map[k].add(v)
        return layer_files_map

    def _build_pprank_layer_map(self) -> None:
        layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
        if self.num_layer_list is not None:
            layer_list = list(map(int, self.num_layer_list.split(',')))
            if len(layer_list) != self.pp_size or sum(
                    layer_list) != self.num_layers:
                raise ValueError('num-layer-list 非法')
            layers_each_pp = layer_list

        if self.noop_layers_list and self.num_layer_list is None:
            base = self.num_layers // self.pp_size
            for layer in self.noop_layers_list:
                pp_rank = layer // base
                layers_each_pp[pp_rank] -= 1

        num_noop = len(self.noop_layers_list)
        num_real_layers = self.num_layers - num_noop
        real_layers = list(range(num_real_layers))
        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                real_layers.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]

    def _build_vpprank_layer_map(self) -> None:
        if not self.dualpipe:
            layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                               for _ in range(self.pp_size)]
            if self.noop_layers_list:
                layers_per_pp = self.num_layers // self.pp_size
                for layer in self.noop_layers_list:
                    pp_idx = layer // layers_per_pp
                    vpp_idx = (layer % layers_per_pp) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            num_noop = len(self.noop_layers_list)
            num_real_layers = self.num_layers - num_noop
            real_layers = list(range(num_real_layers))
            # Megatron 标准 VPP 采用 PP-first 分配：每个物理 PP rank 的 VPP stages 连续
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                        real_layers.pop(0)
                        for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]
            return

        noop_list = self.noop_layers_list
        min_noop = noop_list[0] if noop_list else None

        layers_each_pp = self.num_layers // self.pp_size
        layer_pop_num = layers_each_pp // 2
        all_layers = list(range(self.num_layers))
        dualpipe_layers: list[int] = []
        while all_layers:
            dualpipe_layers.extend(all_layers[:layer_pop_num])
            dualpipe_layers.extend(all_layers[-layer_pop_num:])
            all_layers = all_layers[layer_pop_num:-layer_pop_num]

        pp_rank = 0
        vpp_rank = 0
        each_pp_layer = self.num_layers // self.pp_size
        for idx, layer in enumerate(dualpipe_layers):
            if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

            if not noop_list:
                self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
            else:
                if layer in noop_list:
                    if (idx + 1) % self.vpp_stage == 0:
                        vpp_rank += 1
                    if (idx + 1) % each_pp_layer == 0:
                        pp_rank += 1
                        vpp_rank = 0
                    continue
                if layer < min_noop:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    before = 0
                    for n in noop_list:
                        if n < layer:
                            before += 1
                        else:
                            break
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer -
                                                                      before)

            if (idx + 1) % self.vpp_stage == 0:
                vpp_rank += 1
            if (idx + 1) % each_pp_layer == 0:
                pp_rank += 1
                vpp_rank = 0

    def _load_safetensors_keys(self, filename: str,
                               keys: list[str]) -> dict[str, torch.Tensor]:
        """Load specified keys from a safetensors file with error handling.

        Args:
            filename: Name of the safetensors file
            keys: List of tensor keys to load

        Returns:
            Dictionary mapping keys to tensors

        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If a requested key is not found in the file
            RuntimeError: If there's an error reading the file
        """
        path = os.path.join(self.hf_model_path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Safetensors file not found: {path}')

        out: dict[str, torch.Tensor] = {}
        t0 = time.time()
        try:
            with safe_open(path, framework='pt', device='cpu') as f:
                available_keys = set(f.keys())
                for k in keys:
                    if k not in available_keys:
                        raise KeyError(
                            f'Tensor "{k}" not found in {filename}. '
                            f'Available keys: {list(available_keys)[:10]}...')
                    out[k] = f.get_tensor(k)
        except Exception as e:
            if isinstance(e, (FileNotFoundError, KeyError)):
                raise
            raise RuntimeError(
                f'Error reading safetensors file {filename}: {e}') from e

        if self.log_file_load:
            dt = time.time() - t0
            logger.info('Loaded %d tensors from %s (%.2fs)', len(keys),
                        filename, dt)
        return out

    def _load_matched_hf_weights(
            self, pp_rank: int,
            vpp_rank: int | None) -> dict[str, torch.Tensor]:
        if vpp_rank is None:
            layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

        need_pre = pp_rank == 0 and (vpp_rank is None or vpp_rank == 0)
        need_post = False
        if self.dualpipe:
            need_post = pp_rank == 0 and (vpp_rank is not None) and (
                vpp_rank == self.vpp_size - 1)
        else:
            need_post = pp_rank == self.pp_size - 1 and (
                vpp_rank is None or vpp_rank == self.vpp_size - 1)

        required: set[str] = set()
        for layer in layer_list:
            required.update(self.layer_keys_map.get(layer, []))
        if need_pre:
            required.add('model.embed_tokens.weight')
        if need_post:
            required.add('model.norm.weight')
            if not self.tie_word_embeddings:
                required.add('lm_head.weight')
            else:
                required.add('model.embed_tokens.weight')

        files_to_keys: dict[str, list[str]] = defaultdict(list)
        for k in required:
            fn = self.weight_map.get(k)
            if fn is None:
                raise KeyError(f'HF weight_map 缺少 key: {k}')
            files_to_keys[fn].append(k)

        if self.verbose:
            layer_count = len(layer_list)
            logger.info(
                'Load HF weights: pp=%s vpp=%s layers=%d need_pre=%s need_post=%s keys=%d files=%d',
                str(pp_rank),
                str(vpp_rank),
                int(layer_count),
                str(need_pre),
                str(need_post),
                int(len(required)),
                int(len(files_to_keys)),
            )

        all_weights: dict[str, torch.Tensor] = {}
        t0 = time.time()
        if self.hf_io_threads > 1 and len(files_to_keys) > 1:
            with ThreadPoolExecutor(max_workers=self.hf_io_threads) as ex:
                futures = [
                    ex.submit(self._load_safetensors_keys, fn, ks)
                    for fn, ks in files_to_keys.items()
                ]
                for fut in as_completed(futures):
                    all_weights.update(fut.result())
        else:
            for fn, ks in files_to_keys.items():
                all_weights.update(self._load_safetensors_keys(fn, ks))
        if self.verbose:
            dt = time.time() - t0
            logger.info('Loaded HF batch: pp=%s vpp=%s tensors=%d (%.2fs)',
                        str(pp_rank), str(vpp_rank), len(all_weights), dt)
        return all_weights

    def _maybe_quant_nf4(self, state: dict[str, torch.Tensor], key: str,
                         weight: torch.Tensor) -> None:
        if not self.qlora_nf4:
            return
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError('启用 --qlora-nf4 需要 bitsandbytes') from e
        quantweight = bnb.nn.Params4bit(weight,
                                        requires_grad=False,
                                        quant_type='nf4').to('cpu')
        state[key] = quantweight.data
        for k, v in quantweight.quant_state.as_dict(packed=True).items():
            state[f'{key}.{k}'] = v.detach().cpu()

    def _set_preprocess(
            self, weights: dict[str, torch.Tensor],
            mg_model: dict[int, dict[int, dict[str, torch.Tensor]]]) -> None:
        """
        转换输入嵌入层权重。

        权重映射:
            HF: model.embed_tokens.weight -> MCore: embedding.word_embeddings.weight

        TP 切分策略:
            - dim=0 (在 vocab_size 维度切分)
        """
        if self.tie_word_embeddings and self.pp_size == 1 and self.vpp_stage is None:
            emb = weights['model.embed_tokens.weight']
        else:
            emb = weights.pop('model.embed_tokens.weight')
        emb_tp = torch.chunk(emb, self.tp_size, dim=0)
        # NOTE: torch.chunk on a contiguous tensor produces contiguous views.
        # torch.save serializes the full underlying Storage, not just the view.
        # Must use .clone() to create independent tensors with own Storage.
        emb_shards = [t.contiguous().clone() for t in emb_tp]
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'embedding.word_embeddings.weight'] = emb_shards[tp_rank]

    def _set_postprocess(
            self, weights: dict[str, torch.Tensor],
            mg_model: dict[int, dict[int, dict[str, torch.Tensor]]]) -> None:
        """
        转换输出层权重 (Final LayerNorm 和 LM Head)。

        权重映射:
            HF: model.norm.weight -> MCore: decoder.final_layernorm.weight
            HF: lm_head.weight -> MCore: output_layer.weight

        注意:
            - 如果 tie_word_embeddings=True，lm_head 可能与 embed_tokens 共享权重
            - LM Head TP 切分策略: dim=0 (在 vocab_size 维度切分)
        """
        final_norm = weights.pop('model.norm.weight')
        lm_head = weights.pop('lm_head.weight', None)
        if lm_head is None:
            if self.tie_word_embeddings:
                lm_head = weights.pop('model.embed_tokens.weight')
            else:
                raise KeyError('缺少 lm_head.weight 且未启用 tie_word_embeddings')
        lm_head_tp = torch.chunk(lm_head, self.tp_size, dim=0)
        lm_head_shards = [t.contiguous().clone() for t in lm_head_tp]
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'decoder.final_layernorm.weight'] = final_norm
                mg_model[ep_rank][tp_rank][
                    'output_layer.weight'] = lm_head_shards[tp_rank]
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      'output_layer.weight',
                                      lm_head_shards[tp_rank])

    def _set_layer_norm(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        """
        转换层的 LayerNorm 权重。

        权重映射:
            HF: input_layernorm.weight -> MCore: input_layernorm.weight
            HF: post_attention_layernorm.weight -> MCore: pre_mlp_layernorm.weight

        注意:
            - LayerNorm 权重不参与 TP 切分 (所有 rank 相同)
        """
        in_norm = weights.pop(
            f'model.layers.{hf_layer}.input_layernorm.weight')
        post_norm = weights.pop(
            f'model.layers.{hf_layer}.post_attention_layernorm.weight')
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    f'decoder.layers.{local_layer_idx}.input_layernorm.weight'] = in_norm
                mg_model[ep_rank][tp_rank][
                    f'decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight'] = post_norm

    def _set_layer_attn(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        """
        转换 Attention 层权重从 HF 格式到 MCore 格式。

        权重映射:
            HF: q_proj, k_proj, v_proj (分离) -> MCore: linear_qkv (融合)
            HF: o_proj -> MCore: linear_proj
            HF: q_layernorm, k_layernorm -> MCore: q_layernorm, k_layernorm (可选)

        维度计算 (基于 GQA 配置):
            - q_head_dim = qk_head_dim = 128
            - Q projection rows = num_attention_heads * q_head_dim = 64 * 128 = 8192
            - K projection rows =  num_query_groups * q_head_dim = 2 * 128 = 256
            - V projection rows = num_query_groups * v_head_dim = 2 * 128 = 256
            - 融合 QKV rows = 8192 + 256 + 256 = 8704

        TP 切分策略:
            - QKV: dim=0 (在 heads 维度切分)
            - O_proj: dim=1 (在 hidden_size 维度切分)
        """
        prefix = f'decoder.layers.{local_layer_idx}.self_attention'
        qkv_key = f'{prefix}.linear_qkv.weight'
        proj_key = f'{prefix}.linear_proj.weight'

        q_weight = weights.pop(
            f'model.layers.{hf_layer}.self_attn.q_proj.weight')
        k_weight = weights.pop(
            f'model.layers.{hf_layer}.self_attn.k_proj.weight')
        v_weight = weights.pop(
            f'model.layers.{hf_layer}.self_attn.v_proj.weight')
        o_proj = weights.pop(
            f'model.layers.{hf_layer}.self_attn.o_proj.weight')
        q_ln = weights.pop(
            f'model.layers.{hf_layer}.self_attn.q_layernorm.weight', None)
        k_ln = weights.pop(
            f'model.layers.{hf_layer}.self_attn.k_layernorm.weight', None)
        weights.pop(f'model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq',
                    None)

        q_norm_key = f'{prefix}.q_layernorm.weight'
        k_norm_key = f'{prefix}.k_layernorm.weight'

        expected_q_head_dim = self.qk_head_dim
        expected_q_rows = self.num_attention_heads * expected_q_head_dim
        expected_k_rows = self.num_query_groups * self.qk_head_dim
        expected_v_rows = self.num_query_groups * self.v_head_dim
        if q_weight.shape[0] != expected_q_rows:
            raise ValueError(
                f'Q projection row mismatch: expected {expected_q_rows}, got {q_weight.shape[0]}'
            )
        if k_weight.shape[0] != expected_k_rows:
            raise ValueError(
                f'K projection row mismatch: expected {expected_k_rows}, got {k_weight.shape[0]}'
            )
        if v_weight.shape[0] != expected_v_rows:
            raise ValueError(
                f'V projection row mismatch: expected {expected_v_rows}, got {v_weight.shape[0]}'
            )

        q_tp = torch.chunk(q_weight, self.tp_size, dim=0)
        k_tp = torch.chunk(k_weight, self.tp_size, dim=0)
        v_tp = torch.chunk(v_weight, self.tp_size, dim=0)
        o_proj_tp = torch.chunk(o_proj, self.tp_size, dim=1)
        qkv_shards = [
            torch.cat([q_tp[i], k_tp[i], v_tp[i]], dim=0).clone()
            for i in range(self.tp_size)
        ]
        o_proj_shards = [t.contiguous().clone() for t in o_proj_tp]

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][qkv_key] = qkv_shards[tp_rank]
                mg_model[ep_rank][tp_rank][proj_key] = o_proj_shards[tp_rank]
                if q_ln is not None:
                    mg_model[ep_rank][tp_rank][q_norm_key] = q_ln
                if k_ln is not None:
                    mg_model[ep_rank][tp_rank][k_norm_key] = k_ln
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank], proj_key,
                                      o_proj_shards[tp_rank])

    def _set_layer_mlp(
        self,
        hf_layer: int,
        local_layer_idx: int,
        weights: dict[str, torch.Tensor],
        mg_model: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        """
        转换 MLP 层权重从 HF 格式到 MCore 格式。

        支持两种模式:
        1. Dense MLP (前 first_k_dense_replace 层):
            - HF: gate_proj, up_proj (分离) -> MCore: linear_fc1 (融合)
            - HF: down_proj -> MCore: linear_fc2

        2. MoE (剩余层):
            - HF: gate.weight -> MCore: router.weight
            - HF: gate.e_score_correction_bias -> MCore: router.expert_bias (可选)
            - HF: shared_experts.gate_proj/up_proj -> MCore: shared_experts.linear_fc1
            - HF: shared_experts.down_proj -> MCore: shared_experts.linear_fc2
            - HF: experts.{i}.{gate,up,down}_proj -> MCore: experts.local_experts.{i}.linear_fc{1,2}

        TP 切分策略:
            - linear_fc1: dim=0 (在 intermediate_size 维度切分)
            - linear_fc2: dim=1 (在 hidden_size 维度切分)
        """
        prefix = f'decoder.layers.{local_layer_idx}.mlp'

        # Determine if this layer is Dense MLP or MoE based on actual
        # weight structure.  first_k_dense_replace is the primary hint,
        # but we also check the loaded weights to catch mismatches.
        is_dense_layer = hf_layer < self.first_k_dense_replace
        has_moe_key = f'model.layers.{hf_layer}.mlp.gate.weight' in weights
        has_dense_key = f'model.layers.{hf_layer}.mlp.gate_proj.weight' in weights

        if is_dense_layer and has_moe_key and not has_dense_key:
            logger.warning(
                'layer %d: first_k_dense_replace=%d says dense, '
                'but HF model has MoE structure. Using MoE path.', hf_layer,
                self.first_k_dense_replace)
            is_dense_layer = False
        elif not is_dense_layer and has_dense_key and not has_moe_key:
            logger.warning(
                'layer %d: first_k_dense_replace=%d says MoE, '
                'but HF model has dense MLP structure. Using dense path.',
                hf_layer, self.first_k_dense_replace)
            is_dense_layer = True
        elif not is_dense_layer and not has_moe_key and not has_dense_key:
            raise KeyError(
                f'layer {hf_layer}: neither MoE (mlp.gate.weight) nor '
                f'dense (mlp.gate_proj.weight) keys found in HF weights')

        if is_dense_layer:
            gate = weights.pop(f'model.layers.{hf_layer}.mlp.gate_proj.weight')
            up = weights.pop(f'model.layers.{hf_layer}.mlp.up_proj.weight')
            down = weights.pop(f'model.layers.{hf_layer}.mlp.down_proj.weight')

            gate_tp = torch.chunk(gate, self.tp_size, dim=0)
            up_tp = torch.chunk(up, self.tp_size, dim=0)
            fc1_shards = [
                torch.cat([g, u], dim=0).contiguous().clone()
                for g, u in zip(gate_tp, up_tp)
            ]
            fc2_tp = torch.chunk(down, self.tp_size, dim=1)
            fc2_shards = [t.contiguous().clone() for t in fc2_tp]
            for ep_rank in range(self.ep_size):
                for tp_rank in range(self.tp_size):
                    mg_model[ep_rank][tp_rank][
                        f'{prefix}.linear_fc1.weight'] = fc1_shards[tp_rank]
                    mg_model[ep_rank][tp_rank][
                        f'{prefix}.linear_fc2.weight'] = fc2_shards[tp_rank]
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{prefix}.linear_fc1.weight',
                                          fc1_shards[tp_rank])
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{prefix}.linear_fc2.weight',
                                          fc2_shards[tp_rank])
            return

        router_w_raw = weights.pop(f'model.layers.{hf_layer}.mlp.gate.weight')
        if router_w_raw.shape[0] != self.num_experts:
            router_w = router_w_raw[:self.num_experts, :].clone()
        else:
            router_w = router_w_raw
        router_b_raw = weights.pop(
            f'model.layers.{hf_layer}.mlp.gate.e_score_correction_bias', None)
        # 始终 pop gate.bias（无论 e_score_correction_bias 是否存在），
        # 防止旧 checkpoint 中同时含有两者时触发 _assert_consumed 报错。
        fallback_b = weights.pop(f'model.layers.{hf_layer}.mlp.gate.bias',
                                 None)
        if router_b_raw is None:
            router_b_raw = fallback_b
        # expert bias 为可选字段；部分旧版 checkpoint 不含此项，跳过即可。
        has_router_bias = router_b_raw is not None
        if not has_router_bias:
            logger.warning(
                'layer %d: mlp.gate.e_score_correction_bias not found in HF checkpoint, '
                'skipping expert bias (moe_router_enable_expert_bias=False path).',
                hf_layer,
            )
        if has_router_bias:
            if router_b_raw.shape[0] != self.num_experts:
                router_b = router_b_raw[:self.num_experts].clone()
            else:
                router_b = router_b_raw
        else:
            router_b = None

        shared_gate = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.gate_proj.weight')
        shared_up = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.up_proj.weight')
        shared_down = weights.pop(
            f'model.layers.{hf_layer}.mlp.shared_experts.down_proj.weight')

        shared_fc1 = torch.cat([shared_gate, shared_up], dim=0)
        shared_fc1_tp = torch.chunk(shared_fc1, self.tp_size, dim=0)
        shared_fc2_tp = torch.chunk(shared_down, self.tp_size, dim=1)
        shared_fc1_shards = [t.contiguous().clone() for t in shared_fc1_tp]
        shared_fc2_shards = [t.contiguous().clone() for t in shared_fc2_tp]

        experts_linear_fc1_list: list[torch.Tensor] = []
        experts_linear_fc2_list: list[torch.Tensor] = []
        expert_tp_size = self.expert_tp_size
        for expert in range(self.num_experts):
            gate = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'
            )
            up = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight')
            down = weights.pop(
                f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'
            )

            gate_chunks = torch.chunk(gate, expert_tp_size, dim=0)
            up_chunks = torch.chunk(up, expert_tp_size, dim=0)
            fc1 = torch.cat(
                [x for pair in zip(gate_chunks, up_chunks) for x in pair],
                dim=0)
            experts_linear_fc1_list.append(fc1.t())
            experts_linear_fc2_list.append(down.t())

        router_key = f'{prefix}.router.weight'
        router_bias_key = f'{prefix}.router.expert_bias'
        shared_fc1_key = f'{prefix}.shared_experts.linear_fc1.weight'
        shared_fc2_key = f'{prefix}.shared_experts.linear_fc2.weight'
        experts_weight1_key = f'{prefix}.experts.weight1'
        experts_weight2_key = f'{prefix}.experts.weight2'

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][router_key] = router_w
                if router_b is not None:
                    mg_model[ep_rank][tp_rank][router_bias_key] = router_b
                mg_model[ep_rank][tp_rank][shared_fc1_key] = shared_fc1_shards[
                    tp_rank]
                mg_model[ep_rank][tp_rank][shared_fc2_key] = shared_fc2_shards[
                    tp_rank]
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      shared_fc1_key,
                                      shared_fc1_shards[tp_rank])
                self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                      shared_fc2_key,
                                      shared_fc2_shards[tp_rank])

        if self.moe_grouped_gemm:
            # experts_linear_fc1_list 每个元素形状: [hidden_size, intermediate_size*2]
            # 需要 view 成 [num_experts, hidden_size, intermediate_size*2] 然后做 EP/TP 切分
            gemm_fc1 = torch.stack(experts_linear_fc1_list,
                                   dim=0).reshape(self.num_experts,
                                                  self.hidden_size, -1)
            gemm_fc2 = torch.stack(experts_linear_fc2_list,
                                   dim=0).reshape(self.num_experts, -1,
                                                  self.hidden_size)
            if gemm_fc1.shape[1] != self.hidden_size or gemm_fc2.shape[
                    2] != self.hidden_size:
                raise ValueError(
                    f'moe grouped gemm hidden_size 不匹配: hidden_size={self.hidden_size} gemm_fc1={tuple(gemm_fc1.shape)} gemm_fc2={tuple(gemm_fc2.shape)}'
                )
            gemm_fc1_ep = torch.chunk(gemm_fc1, self.ep_size, dim=0)
            gemm_fc2_ep = torch.chunk(gemm_fc2, self.ep_size, dim=0)
            for ep_rank in range(self.ep_size):
                if self.expert_tp_size > 1:
                    # Weight-sharding: split each expert's intermediate
                    # dimension across expert_tp_size TP ranks (not
                    # expert-subsampling).  Each TP rank gets ALL experts
                    # but with a shard of the intermediate dim.
                    fc1_shards = torch.chunk(gemm_fc1_ep[ep_rank],
                                             self.expert_tp_size,
                                             dim=2)
                    fc2_shards = torch.chunk(gemm_fc2_ep[ep_rank],
                                             self.expert_tp_size,
                                             dim=1)
                else:
                    # expert_tp=1: 专家权重不做 TP 切分，所有 TP rank 获得相同权重
                    fc1_shards = [gemm_fc1_ep[ep_rank]]
                    fc2_shards = [gemm_fc2_ep[ep_rank]]
                for tp_rank in range(self.tp_size):
                    expert_tp_idx = tp_rank % self.expert_tp_size
                    # Use expert-first layout: [num_local, hidden_size,
                    # intermediate*2] -> [hidden_size, -1].  This matches
                    # MCore's grouped_gemm convention where weight1 is
                    # viewed as [num_local_experts, hidden_size,
                    # intermediate*2].
                    # NOTE: reshape may return a VIEW of the full
                    # (all-experts) storage.  torch.save serializes the
                    # entire underlying Storage, not just the viewed
                    # portion, causing ~EP× disk bloat.  .clone()
                    # creates an independent tensor with its own Storage
                    # containing only the needed expert subset.
                    shard = fc1_shards[expert_tp_idx]
                    w1 = shard.reshape(self.hidden_size, -1).clone()
                    w2 = fc2_shards[expert_tp_idx].reshape(
                        -1, self.hidden_size).clone()
                    mg_model[ep_rank][tp_rank][experts_weight1_key] = w1
                    mg_model[ep_rank][tp_rank][experts_weight2_key] = w2
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          experts_weight1_key, w1)
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          experts_weight2_key, w2)
            return

        num_local_experts = self.num_experts // self.ep_size
        for ep_rank in range(self.ep_size):
            for local_idx in range(num_local_experts):
                global_idx = local_idx + ep_rank * num_local_experts
                local_fc1 = experts_linear_fc1_list[global_idx].t()
                local_fc2 = experts_linear_fc2_list[global_idx].t()
                local_prefix = f'{prefix}.experts.local_experts.{local_idx}'
                if self.expert_tp_size > 1:
                    local_fc1_tp = torch.chunk(local_fc1,
                                               self.expert_tp_size,
                                               dim=0)
                    local_fc2_tp = torch.chunk(local_fc2,
                                               self.expert_tp_size,
                                               dim=1)
                    local_fc1_shards = [
                        t.contiguous().clone() for t in local_fc1_tp
                    ]
                    local_fc2_shards = [
                        t.contiguous().clone() for t in local_fc2_tp
                    ]
                else:
                    # expert_tp=1: 不切分，所有 TP rank 获得相同权重
                    local_fc1_shards = [local_fc1.contiguous().clone()]
                    local_fc2_shards = [local_fc2.contiguous().clone()]
                for tp_rank in range(self.tp_size):
                    expert_tp_idx = tp_rank % self.expert_tp_size
                    mg_model[ep_rank][tp_rank][
                        f'{local_prefix}.linear_fc1.weight'] = local_fc1_shards[
                            expert_tp_idx]
                    mg_model[ep_rank][tp_rank][
                        f'{local_prefix}.linear_fc2.weight'] = local_fc2_shards[
                            expert_tp_idx]
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{local_prefix}.linear_fc1.weight',
                                          local_fc1_shards[expert_tp_idx])
                    self._maybe_quant_nf4(mg_model[ep_rank][tp_rank],
                                          f'{local_prefix}.linear_fc2.weight',
                                          local_fc2_shards[expert_tp_idx])

    def _maybe_cast(self, t: torch.Tensor) -> torch.Tensor:
        if self._target_dtype is None:
            return t
        if not torch.is_floating_point(t):
            return t
        if t.dtype == self._target_dtype:
            return t
        return t.to(dtype=self._target_dtype)

    def _cast_model_dict(
            self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._target_dtype is None:
            return d
        return {
            k: (v if ('router.expert_bias' in k or 'router.score_bias' in k)
                else self._maybe_cast(v))
            for k, v in d.items()
        }

    def _save_single_rank_file(
        self,
        pp_rank: int,
        tp_rank: int,
        ep_rank: int,
        mg_model: Union[dict[int, dict[int, dict[str, torch.Tensor]]],
                        dict[int, dict[int, dict[int, dict[str,
                                                           torch.Tensor]]]]],
        vpp: bool,
    ) -> None:
        """保存单个 rank 的文件（线程安全版本）"""
        prefix = _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                            self.pp_size, self.ep_size)
        outdir = os.path.join(self.iter_path, prefix)
        outpath = os.path.join(outdir, 'model_optim_rng.pt')

        # Lazy-build checkpoint args (cached after first call)
        if not hasattr(self, '_cached_ckpt_args'):
            self._cached_ckpt_args = self._build_checkpoint_args()
        ckpt_args = self._cached_ckpt_args

        if vpp:
            # 动态遍历所有 VPP stages，避免硬编码 model0/model1 导致 stage 丢失
            vpp_size = getattr(self, 'vpp_size', 2)
            payload: dict[str, Any] = {
                f'model{i}':
                self._cast_model_dict(mg_model[i][ep_rank][tp_rank])
                for i in range(vpp_size)
            }
            payload.update({
                'checkpoint_version': 3.0,
                'iteration': 1,
                'args': ckpt_args,
            })
        else:
            payload = {
                'model': self._cast_model_dict(mg_model[ep_rank][tp_rank]),
                'checkpoint_version': 3.0,
                'iteration': 1,
                'args': ckpt_args,
            }

        torch.save(payload,
                   outpath,
                   pickle_protocol=4,
                   _use_new_zipfile_serialization=True)

    def _save_pp_rank(
        self,
        pp_rank: int,
        mg_model: Union[dict[int, dict[int, dict[str, torch.Tensor]]],
                        dict[int, dict[int, dict[int, dict[str,
                                                           torch.Tensor]]]]],
        vpp: bool,
    ) -> None:
        """保存一个 PP rank 的所有权重文件，使用线程池并行化 EP×TP 保存"""
        t0 = time.time()
        if self.log_save_progress:
            logger.info(
                'Saving Megatron ckpt: pp=%d vpp=%s -> %s',
                int(pp_rank),
                str(vpp),
                self.iter_path,
            )

        # 生成所有要保存的 rank 组合
        rank_tasks = []
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                rank_tasks.append((tp_rank, ep_rank))

        # 预先创建所有输出目录（避免多线程同时创建导致的竞态条件）
        for tp_rank, ep_rank in rank_tasks:
            prefix = _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                                self.pp_size, self.ep_size)
            outdir = os.path.join(self.iter_path, prefix)
            os.makedirs(outdir, exist_ok=True)

        total_tasks = len(rank_tasks)
        completed = 0

        # 使用线程池并行保存
        max_save_workers = min(self.ep_size * self.tp_size, 32)  # 限制最大线程数
        if self.log_save_progress:
            logger.info('Parallel saving: %d tasks with max %d workers',
                        total_tasks, max_save_workers)

        # 优先使用用户显式指定的 save_workers；0 表示自动
        if self.save_workers > 0:
            save_workers = min(self.save_workers, max_save_workers)
        else:
            save_workers = max_save_workers
        # 环境变量可覆盖上述决定（向后兼容）
        save_workers = int(
            os.environ.get('CKPT_CONVERT_SAVE_WORKERS', str(save_workers)))
        save_workers = min(save_workers, max_save_workers)

        if save_workers <= 1:
            # 串行保存模式（向后兼容）
            for tp_rank, ep_rank in rank_tasks:
                self._save_single_rank_file(pp_rank, tp_rank, ep_rank,
                                            mg_model, vpp)
                completed += 1
                if self.log_save_progress and completed % 10 == 0:
                    logger.info('Saved pp=%d progress: %d/%d', int(pp_rank),
                                int(completed), int(total_tasks))
        else:
            # 并行保存模式
            with ThreadPoolExecutor(max_workers=save_workers) as executor:
                # 提交所有任务
                future_to_rank = {}
                for tp_rank, ep_rank in rank_tasks:
                    future = executor.submit(self._save_single_rank_file,
                                             pp_rank, tp_rank, ep_rank,
                                             mg_model, vpp)
                    future_to_rank[future] = (tp_rank, ep_rank)

                # 等待任务完成并更新进度
                for future in as_completed(future_to_rank):
                    tp_rank, ep_rank = future_to_rank[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error('Error saving pp=%d tp=%d ep=%d: %s',
                                     int(pp_rank), int(tp_rank), int(ep_rank),
                                     str(e))
                        raise

                    completed += 1
                    if self.log_save_progress and completed % 10 == 0:
                        logger.info('Saved pp=%d progress: %d/%d',
                                    int(pp_rank), int(completed),
                                    int(total_tasks))

        if self.log_save_progress:
            dt = time.time() - t0
            logger.info('Saved pp=%d done: %d files in %.2fs (%.2f files/s)',
                        int(pp_rank), int(total_tasks), dt,
                        total_tasks / max(dt, 0.01))

    def run(self) -> None:
        self._log_expected_param_summary()
        workers = int(self.pp_workers)
        if workers <= 1:
            for pp_rank in range(self.pp_size):
                self.run_one_pp_rank(pp_rank)
            return

        workers = min(workers, self.pp_size)
        ctx = get_context('spawn')
        cfg = dict(
            hf_model_path=self.hf_model_path,
            mg_save_path=self.mg_save_path,
            num_layers=self.num_layers,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            ep_size=self.ep_size,
            first_k_dense_replace=self.first_k_dense_replace,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            moe_ffn_hidden_size=self.moe_ffn_hidden_size,
            vocab_size=self.vocab_size,
            num_experts=self.num_experts,
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            moe_grouped_gemm=self.moe_grouped_gemm,
            expert_tp_size=self.expert_tp_size,
            schedules_method=self.schedules_method,
            vpp_stage=None if self.dualpipe else self.vpp_stage,
            num_layer_list=self.num_layer_list,
            noop_layers=self.noop_layers,
            qlora_nf4=self.qlora_nf4,
            rotary_base=self.rotary_base,
            print_init_summary=False,
            pp_workers=1,
            save_workers=self.save_workers,
            cast_dtype=self.cast_dtype,
            tie_word_embeddings=self.tie_word_embeddings,
            hf_io_threads=self.hf_io_threads,
            qk_layernorm=self.qk_layernorm,
        )

        t0 = time.time()
        logger.info('Parallel convert: pp_workers=%d pp=%d', int(workers),
                    int(self.pp_size))
        futures = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for pp_rank in range(self.pp_size):
                futures.append(ex.submit(_worker_run_one_pp_rank, cfg,
                                         pp_rank))
            for fut in as_completed(futures):
                fut.result()
        logger.info('Parallel convert done (%.2fs)', time.time() - t0)


def _worker_run_one_pp_rank(cfg: dict, pp_rank: int) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    converter = CkptConvert(**cfg)
    converter.run_one_pp_rank(int(pp_rank))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir',
                        type=str,
                        required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument(
        '--target-tensor-parallel-size',
        type=int,
        default=1,
        help='Target tensor model parallel size, defaults to 1.')
    parser.add_argument(
        '--target-pipeline-parallel-size',
        type=int,
        default=1,
        help='Target pipeline model parallel size, defaults to 1.')
    parser.add_argument(
        '--target-expert-parallel-size',
        type=int,
        default=1,
        help='Target expert model parallel size, defaults to 1.')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage',
                        '--vpp-stage',
                        dest='num_layers_per_virtual_pipeline_stage',
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument('--expert-tensor-parallel-size',
                        type=int,
                        default=1,
                        help='Expert tensor parallel size (default: 1, '
                        'experts not split by TP).')
    parser.add_argument('--noop-layers',
                        type=str,
                        default='',
                        help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers',
                        type=int,
                        default=0,
                        help='Multi-Token prediction layer num '
                        '(reserved, not yet implemented in converter)')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument('--num-layers',
                        type=int,
                        default=None,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=2,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=None,
                        help='Override hidden size (default: from HF config).')
    parser.add_argument('--ffn-hidden-size',
                        type=int,
                        default=None,
                        help='Override ffn hidden size.')
    parser.add_argument('--moe-ffn-hidden-size',
                        type=int,
                        default=None,
                        help='Override moe ffn hidden size.')
    parser.add_argument('--vocab-size',
                        type=int,
                        default=None,
                        help='Override vocab size.')
    parser.add_argument('--num-experts',
                        type=int,
                        default=None,
                        help='Override num experts (default: from HF config).')
    parser.add_argument(
        '--num-attention-heads',
        type=int,
        default=None,
        help='Override attention heads (default: from HF config).',
    )
    parser.add_argument(
        '--num-query-groups',
        type=int,
        default=None,
        help='Number of query groups for GQA (default: from HF config).',
    )
    parser.add_argument('--qk-head-dim',
                        type=int,
                        default=None,
                        help='Override qk head dim.')
    parser.add_argument('--v-head-dim',
                        type=int,
                        default=None,
                        help='Override v head dim.')
    parser.add_argument(
        '--qk-layernorm',
        action='store_true',
        help='Enable QK LayerNorm (must match pretrain config)')
    parser.add_argument('--max-position-embeddings',
                        type=int,
                        default=None,
                        help='Override max position embeddings.')
    parser.add_argument('--tie-word-embeddings',
                        action='store_true',
                        help='Tie word embeddings and output layer.')
    parser.add_argument(
        '--schedules-method',
        type=str,
        default=None,
        choices=['dualpipev'],
        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--qlora-nf4',
                        action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')
    parser.add_argument('--rotary-base',
                        type=float,
                        default=None,
                        help='Rotary base for RoPE')
    parser.add_argument('--pp-workers',
                        type=int,
                        default=1,
                        help='Parallelize by pp_rank with processes.')
    parser.add_argument(
        '--save-workers',
        type=int,
        default=0,
        help='Parallelize saving within each pp_rank with threads (0=auto).')
    parser.add_argument('--cast-dtype',
                        type=str,
                        default=None,
                        choices=['fp32', 'bf16', 'fp16'],
                        help='Cast floating tensors before saving.')
    parser.add_argument('--hf-io-threads',
                        type=int,
                        default=1,
                        help='Thread workers for reading HF safetensors.')
    parser.add_argument('--sha256-manifest',
                        type=str,
                        default=None,
                        help='Write sha256 manifest json to this path.')

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=int(args.num_layers),
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        vocab_size=args.vocab_size,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        qk_head_dim=args.qk_head_dim,
        v_head_dim=args.v_head_dim,
        moe_grouped_gemm=args.moe_grouped_gemm,
        expert_tp_size=args.expert_tensor_parallel_size,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        qlora_nf4=args.qlora_nf4,
        rotary_base=args.rotary_base,
        pp_workers=args.pp_workers,
        save_workers=args.save_workers,
        cast_dtype=args.cast_dtype,
        tie_word_embeddings=args.tie_word_embeddings,
        hf_io_threads=args.hf_io_threads,
        qk_layernorm=args.qk_layernorm,
    )
    converter.run()
    _write_sha256_manifest(args.save_dir, args.sha256_manifest)


if __name__ == '__main__':
    main()
