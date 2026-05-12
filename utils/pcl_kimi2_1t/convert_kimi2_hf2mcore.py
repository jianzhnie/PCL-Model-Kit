#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""
将 Kimi2-1T-4k HuggingFace 权重转换为 Megatron-Core (MCore) 格式。

模型架构: GQA + MoE (非 MLA)
参考: convert_ckpt_deepseek3.py (框架风格) + convert_ckpt_hf2mcore.py (GQA/MoE 逻辑)
"""
import argparse
import gc
import json
import logging as logger
import os
from collections import defaultdict

import numpy as np
import safetensors
import safetensors.torch
import torch

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)

# Kimi2-1T-4k 默认维度
HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
FIRST_K_DENSE_REPLACE = 2
NUM_LAYERS = 32
NUM_ATTENTION_HEADS = 64
NUM_QUERY_GROUPS = 2
QK_HEAD_DIM = 128
V_HEAD_DIM = 128
FFN_HIDDEN_SIZE = 18432
MOE_FFN_HIDDEN_SIZE = 12288
VOCAB_SIZE = 163840


class CkptConvert(object):
    """
    Converts a HuggingFace checkpoint to Megatron format for Kimi2 GQA+MoE.

    Args:
        hf_model_path (str): HuggingFace model path.
        mg_save_path (str): Megatron model save path.
        num_layers (int): Number of transformer layers.
        tp_size (int, optional): Degree of tensor model parallelism. Defaults to 1.
        pp_size (int, optional): Degree of pipeline model parallelism. Defaults to 1.
        ep_size (int, optional): Degree of expert model parallelism. Defaults to 1.
        vpp_stage (int, optional): The stage number in the virtual pipeline parallelism. Defaults to None.
        num_dense_layers (int, optional): The number of first k dense layers. Defaults to 2.
        num_layer_list (str, optional): Specifies the number of parallel pipeline layers. Defaults to None.
        noop_layers (str, optional): should be skipped during conversion. Defaults to None.
        moe_grouped_gemm (bool, optional): Whether to use grouped GEMM for MoE layers.
        expert_tp_size (int, optional): Expert tensor parallel size. Defaults to 1.
        dualpipe (bool, optional): Whether to use dualpipe.
        qlora_nf4 (bool, optional): Whether to use QLORA NF4. Defaults to False.
    """

    def __init__(
        self,
        hf_model_path: str,
        mg_save_path: str,
        num_layers: int,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_dense_layers: int = FIRST_K_DENSE_REPLACE,
        num_layer_list: str = None,
        noop_layers: str = None,
        vpp_stage: int = None,
        moe_grouped_gemm: bool = False,
        moe_tp_extend_ep: bool = False,
        expert_tp_size: int = 1,
        dualpipe: bool = False,
        qlora_nf4: bool = False,
        qk_layernorm: bool = False,
        num_experts: int = NUM_EXPERTS,
        hidden_size: int = HIDDEN_SIZE,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_query_groups: int = NUM_QUERY_GROUPS,
        qk_head_dim: int = QK_HEAD_DIM,
        v_head_dim: int = V_HEAD_DIM,
        ffn_hidden_size: int = FFN_HIDDEN_SIZE,
        moe_ffn_hidden_size: int = MOE_FFN_HIDDEN_SIZE,
        vocab_size: int = VOCAB_SIZE,
        rotary_base: float = 50000.0,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.num_layers = num_layers
        self.vpp_stage = vpp_stage
        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.expert_tp_size = expert_tp_size
        self.dualpipe = True if dualpipe == 'dualpipev' else False
        self.first_k_dense_replace = num_dense_layers
        self.qk_layernorm = qk_layernorm
        self.qlora_nf4 = qlora_nf4

        if not os.path.exists(self.hf_model_path):
            raise FileNotFoundError(
                f"Model path does not exist: {self.hf_model_path}")
        if dualpipe:
            if vpp_stage:
                raise ValueError(
                    'dualpipe is not compatible with virtual pipeline parallel.'
                )
            self.vpp_size = 2
            self.vpp_stage = self.num_layers // self.pp_size // self.vpp_size

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.vocab_size = vocab_size
        self.rotary_base = rotary_base

        self._valid_parameter()

        if self.vpp_stage is None:
            self.pprank_layer_idxs = defaultdict()
            self.get_pprank_hf_layeridxs()
        else:
            self.vpprank_layer_idxs = defaultdict(dict)
            self.get_vpprank_hf_layeridxs()

    @staticmethod
    def qlora_nf4_weight(weight):
        """Quantize weights"""
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError('启用 --qlora-nf4 需要安装 bitsandbytes') from e
        quantweight = bnb.nn.Params4bit(weight,
                                        requires_grad=weight.requires_grad,
                                        quant_type='nf4').to('cpu')
        return quantweight.data, quantweight.quant_state

    def qlora_nf4_quant(self, mg_model, ep_rank, tp_rank, key, weight):
        """Save quant state"""
        quant_data, quant_state = self.qlora_nf4_weight(weight)
        mg_model[ep_rank][tp_rank][key] = quant_data
        for k, v in quant_state.as_dict(packed=True).items():
            mg_model[ep_rank][tp_rank]['{}.{}'.format(key, k)] = v.detach()

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        logger.info(f"Loading the checkpoint from {file_path}.")
        return safetensors.torch.load_file(file_path)

    @staticmethod
    def mg_path_process(mg_path):
        """megatron model path"""
        iter_mg_path = os.path.join(mg_path, 'iter_0000001')
        if not os.path.exists(mg_path):
            os.makedirs(mg_path, exist_ok=True)

        with open(os.path.join(mg_path, 'latest_checkpointed_iteration.txt'),
                  'w') as f:
            f.write('1')
        return iter_mg_path

    def generate_mg_weights_dir(self, tp_rank, pp_rank, ep_rank):
        """Generate the megatron weight directory."""
        if self.moe_tp_extend_ep and self.tp_size > 1:
            # Interleaved EP naming: global_ep = tp_rank + ep_rank * tp_size
            global_ep = tp_rank + ep_rank * self.tp_size
            if self.pp_size == 1:
                prefix = f"mp_rank_{tp_rank:02}_{global_ep:03}"
            else:
                prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{global_ep:03}"
        elif self.ep_size == 1 and self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}"
        elif self.ep_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
        elif self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{ep_rank:03}"
        else:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}"
        return prefix

    def _valid_parameter(self):
        if self.first_k_dense_replace < 0 or self.first_k_dense_replace > self.num_layers:
            raise ValueError(
                'first_k_dense_replace should be in [0, num_layers]')

        if self.num_experts % self.ep_size != 0:
            raise ValueError('num_experts should be divisible by ep_size')

        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                'num_attention_heads should be divisible by tp_size')

        if self.num_query_groups % self.tp_size != 0:
            raise ValueError('num_query_groups should be divisible by tp_size')

        if self.expert_tp_size > self.tp_size:
            raise ValueError('expert_tp_size cannot exceed tp_size')
        if self.tp_size % self.expert_tp_size != 0:
            raise ValueError('tp_size must be divisible by expert_tp_size')
        if self.moe_tp_extend_ep and self.expert_tp_size > 1:
            raise ValueError(
                'moe_tp_extend_ep and expert_tp_size>1 are mutually exclusive')

        if self.dualpipe:
            if self.tp_size > 1 and self.expert_tp_size != 1:
                # 当 dualpipe 启用且 TP>1 时，通常需要 expert_tp_size=1，
                # 否则专家权重在 TP 内切分可能与 dualpipe 的 all-gather 逻辑冲突。
                # 这里仅做警告，不强制报错，因为某些实现可能支持。
                logger.warning(
                    'dualpipev with tp_size>1 usually requires expert_tp_size=1'
                )

        if self.num_layer_list is None:
            if self.num_layers % self.pp_size != 0:
                raise ValueError(
                    'number of layers should be divisible by the pipeline parallel size'
                )
            if self.vpp_stage is not None:
                if (self.num_layers % self.pp_size) % self.vpp_stage != 0:
                    raise ValueError(
                        'number of pp_stage should be divisible by the vpp_stage'
                    )
        else:
            layer_list = list(map(int, self.num_layer_list.split(',')))
            if self.vpp_stage is not None:
                raise ValueError(
                    'num_layer_list and vpp cannot be configured at the same time'
                )
            if len(layer_list) != self.pp_size:
                raise ValueError(
                    'number of layer_list should be equal to pipeline parallel size'
                )
            if sum(layer_list) != self.num_layers:
                raise ValueError(
                    'sum of layer_list should be equal to num_layers')
            if self.noop_layers is not None:
                raise ValueError(
                    'num_layer_list and noop_layers cannot be configured at the same time'
                )
            if self.num_layers != NUM_LAYERS:
                raise ValueError(
                    'num_layer_list supports only full parameters')

    def _build_checkpoint_args(self):
        """Build checkpoint args namespace for Megatron loading compatibility."""
        import argparse
        ns = argparse.Namespace()
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
        ns.tensor_model_parallel_size = self.tp_size
        ns.pipeline_model_parallel_size = self.pp_size
        ns.expert_model_parallel_size = self.ep_size
        ns.expert_tensor_parallel_size = self.expert_tp_size
        ns.moe_tp_extend_ep = self.moe_tp_extend_ep
        ns.context_parallel_size = 1
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
        ns.use_distributed_optimizer = True
        ns.mtp_num_layers = 0
        ns.use_mcore_models = True
        ns.use_legacy_models = False
        ns.untie_embeddings_and_output_weights = True
        ns.swiglu = True
        ns.position_embedding_type = 'rope'
        ns.normalization = 'RMSNorm'
        ns.add_bias_linear = False
        ns.norm_epsilon = 1e-6
        ns.bf16 = True
        ns.fp16 = False
        ns.params_dtype = torch.bfloat16
        ns.rotary_base = self.rotary_base
        if hasattr(self, '_target_dtype') and self._target_dtype is not None:
            ns.bf16 = (self._target_dtype == torch.bfloat16)
            ns.fp16 = (self._target_dtype == torch.float16)
            ns.params_dtype = self._target_dtype
        ns.use_rotary_position_embeddings = True
        ns.qk_layernorm = self.qk_layernorm
        if self.dualpipe:
            ns.schedules_method = 'dualpipev'
        if self.vpp_stage is not None:
            ns.num_layers_per_virtual_pipeline_stage = self.vpp_stage
        return ns

    def get_layer_files_map(self):
        """layer -> safetensors file map"""
        layer_map_dict = defaultdict(set)

        # Try index.json first, then fall back to single safetensors file
        index_path = os.path.join(self.hf_model_path,
                                  'model.safetensors.index.json')
        single_path = os.path.join(self.hf_model_path, 'model.safetensors')

        if os.path.isfile(index_path):
            with open(index_path) as f:
                weights_map = json.load(f)
            weights_map = weights_map['weight_map']
        elif os.path.isfile(single_path):
            # Single safetensors file: all keys map to the same file
            import safetensors
            with safetensors.safe_open(single_path,
                                       framework='pt',
                                       device='cpu') as f:
                weights_map = {k: 'model.safetensors' for k in f.keys()}
        else:
            raise FileNotFoundError(
                f'找不到 HF safetensors index 或单文件: {self.hf_model_path}')

        for key, value in weights_map.items():
            if key.startswith('model.layers.'):
                layer_name = int(key.split('model.layers.')[1].split('.')[0])
                layer_map_dict[layer_name].add(value)
            else:
                layer_map_dict[key].add(value)
        return layer_map_dict

    def get_pprank_hf_layeridxs(self) -> None:
        """pp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(
            list(map(int, self.noop_layers.split(','))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        if self.num_layer_list is None:
            layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(','))):
                    cur_pp_rank = layer // (self.num_layers // self.pp_size)
                    layers_each_pp[cur_pp_rank] -= 1
        else:
            layers_each_pp = list(map(int, self.num_layer_list.split(',')))

        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]

    def get_vpprank_hf_layeridxs(self) -> None:
        """vpp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(
            list(map(int, self.noop_layers.split(','))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        if not self.dualpipe:
            if self.vpp_stage is not None:
                layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                                   for _ in range(self.pp_size)]
                if self.noop_layers is not None:
                    for layer in list(map(int, self.noop_layers.split(','))):
                        vpp_idx = layer // self.vpp_stage // self.pp_size
                        pp_idx = layer % (self.pp_size *
                                          self.vpp_stage) // self.vpp_stage
                        layers_each_vpp[pp_idx][vpp_idx] -= 1

                for vpp_rank in range(self.vpp_size):
                    for pp_rank in range(self.pp_size):
                        self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                            num_layer_list_.pop(0)
                            for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                        ]
        else:
            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(',')))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[
                0]

            dualpipe_layer_list = []
            layers_each_pp = self.num_layers // self.pp_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.num_layers)]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            pp_rank, vpp_rank = 0, 0
            each_pp_layer = self.num_layers // self.pp_size
            for idx, layer in enumerate(dualpipe_layer_list):
                if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

                if not self.noop_layers:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    if layer in noop_layers_list:
                        if (idx + 1) % self.vpp_stage == 0:
                            vpp_rank += 1
                        if (idx + 1) % each_pp_layer == 0:
                            pp_rank += 1
                            vpp_rank = 0
                        continue
                    if layer < min_noop_layer:
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer)
                    if layer > min_noop_layer:
                        before_nums = sum(noop_layers_list < layer)
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer - before_nums)

                if (idx + 1) % self.vpp_stage == 0:
                    vpp_rank += 1
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0

    def load_matched_hf_weights(self, pp_rank, vpp_rank=None):
        """Read the safetensors file corresponding to the layer of pp_rank."""
        if vpp_rank is None:
            if self.vpp_stage is not None:
                # Collect all layers across all vpp_ranks for this pp_rank
                layer_list = []
                for vr in range(self.vpp_size):
                    layer_list.extend(self.vpprank_layer_idxs[pp_rank][vr])
            else:
                layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank].copy()

        layer_files_map_dict = self.get_layer_files_map()

        st_filename_list = []
        for layer in layer_list:
            st_filename_list.extend(list(layer_files_map_dict[layer]))

        if pp_rank == 0:
            st_filename_list.extend(
                list(layer_files_map_dict['model.embed_tokens.weight']))
            if self.dualpipe:
                st_filename_list.extend(
                    list(layer_files_map_dict['lm_head.weight']))
                st_filename_list.extend(
                    list(layer_files_map_dict['model.norm.weight']))

        if pp_rank == self.pp_size - 1 and not self.dualpipe:
            st_filename_list.extend(
                list(layer_files_map_dict['model.norm.weight']))
            st_filename_list.extend(
                list(layer_files_map_dict['lm_head.weight']))

        st_filename_list = list(set(st_filename_list))
        st_filename_list.sort()

        all_pp_weights = {}
        for filename in st_filename_list:
            cur_weights = self.load_hf_model(
                os.path.join(self.hf_model_path, filename))
            all_pp_weights.update(cur_weights)

        return all_pp_weights

    def set_model_preprocess(self, weights_dict, mg_model):
        """Embedding layer process"""
        emb_weight = weights_dict.pop('model.embed_tokens.weight')

        for ep_rank in range(self.ep_size):
            emb_weight_lst = torch.chunk(emb_weight, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'embedding.word_embeddings.weight'] = emb_weight_lst[
                        tp_rank].clone()

    def set_model_postprocess(self, weights_dict, mg_model):
        """Final norm & LM Head process"""
        final_norm = weights_dict.pop('model.norm.weight')
        lm_head = weights_dict.pop('lm_head.weight')

        for ep_rank in range(self.ep_size):
            lm_head_lst = torch.chunk(lm_head, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    'decoder.final_layernorm.weight'] = final_norm.clone()
                mg_model[ep_rank][tp_rank][
                    'output_layer.weight'] = lm_head_lst[tp_rank].clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                         'output_layer.weight',
                                         lm_head_lst[tp_rank].clone())

    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, weights_dict,
                             mg_model):
        """Layernorm process"""
        input_norm = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.input_layernorm.weight")
        post_attn_norm = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight")

        input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
        post_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][input_norm_key] = input_norm.clone()
                mg_model[ep_rank][tp_rank][
                    post_norm_key] = post_attn_norm.clone()

    def set_model_layer_attn(self, hf_layer, local_layer_idx, weights_dict,
                             mg_model):
        """Attention layer process for GQA."""
        prefix = f"decoder.layers.{local_layer_idx}.self_attention"
        qkv_key = f"{prefix}.linear_qkv.weight"
        dense_key = f"{prefix}.linear_proj.weight"
        q_norm_key = f"{prefix}.q_layernorm.weight"
        k_norm_key = f"{prefix}.k_layernorm.weight"

        q_weight = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.q_proj.weight")
        k_weight = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.k_proj.weight")
        v_weight = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.v_proj.weight")
        dense_weight = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.o_proj.weight")

        q_ln = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.q_layernorm.weight", None)
        k_ln = weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.k_layernorm.weight", None)

        # 丢弃 rotary_emb inv_freq（转换时不需要）
        weights_dict.pop(
            f"model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq", None)

        expected_q_rows = self.num_attention_heads * self.qk_head_dim
        expected_k_rows = self.num_query_groups * self.qk_head_dim
        expected_v_rows = self.num_query_groups * self.v_head_dim
        if q_weight.shape[0] != expected_q_rows:
            raise ValueError(
                f"Q projection row mismatch: expected {expected_q_rows}, got {q_weight.shape[0]}"
            )
        if k_weight.shape[0] != expected_k_rows:
            raise ValueError(
                f"K projection row mismatch: expected {expected_k_rows}, got {k_weight.shape[0]}"
            )
        if v_weight.shape[0] != expected_v_rows:
            raise ValueError(
                f"V projection row mismatch: expected {expected_v_rows}, got {v_weight.shape[0]}"
            )

        q_tp = torch.chunk(q_weight, self.tp_size, dim=0)
        k_tp = torch.chunk(k_weight, self.tp_size, dim=0)
        v_tp = torch.chunk(v_weight, self.tp_size, dim=0)
        dense_tp = torch.chunk(dense_weight, self.tp_size, dim=1)

        qkv_shards = [
            torch.cat([q_tp[i], k_tp[i], v_tp[i]], dim=0).clone()
            for i in range(self.tp_size)
        ]
        dense_shards = [t.clone() for t in dense_tp]

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][qkv_key] = qkv_shards[tp_rank]
                mg_model[ep_rank][tp_rank][dense_key] = dense_shards[tp_rank]
                if q_ln is not None:
                    mg_model[ep_rank][tp_rank][q_norm_key] = q_ln.clone()
                if k_ln is not None:
                    mg_model[ep_rank][tp_rank][k_norm_key] = k_ln.clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, qkv_key,
                                         qkv_shards[tp_rank].clone())
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, dense_key,
                                         dense_shards[tp_rank].clone())

    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, weights_dict,
                            mg_model):
        """MLP layer process for Dense / MoE."""
        prefix = f"decoder.layers.{local_layer_idx}.mlp"

        is_dense_layer = hf_layer_idx < self.first_k_dense_replace
        has_moe_key = f"model.layers.{hf_layer_idx}.mlp.gate.weight" in weights_dict
        has_dense_key = f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight" in weights_dict

        if is_dense_layer and has_moe_key and not has_dense_key:
            logger.warning(
                'layer %d: first_k_dense_replace=%d says dense, but HF model has MoE structure. Using MoE path.',
                hf_layer_idx, self.first_k_dense_replace)
            is_dense_layer = False
        elif not is_dense_layer and has_dense_key and not has_moe_key:
            logger.warning(
                'layer %d: first_k_dense_replace=%d says MoE, but HF model has dense MLP structure. Using dense path.',
                hf_layer_idx, self.first_k_dense_replace)
            is_dense_layer = True
        elif not is_dense_layer and not has_moe_key and not has_dense_key:
            raise KeyError(
                f"layer {hf_layer_idx}: neither MoE (mlp.gate.weight) nor dense (mlp.gate_proj.weight) keys found"
            )

        if is_dense_layer:
            gate_proj = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight")
            up_proj = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.up_proj.weight")
            down_proj = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.down_proj.weight")

            gate_chunks = torch.chunk(gate_proj, self.tp_size, dim=0)
            up_chunks = torch.chunk(up_proj, self.tp_size, dim=0)
            fc1_shards = [
                torch.cat([g, u], dim=0).clone()
                for g, u in zip(gate_chunks, up_chunks)
            ]
            fc2_shards = [
                t.clone() for t in torch.chunk(down_proj, self.tp_size, dim=1)
            ]

            for ep_rank in range(self.ep_size):
                for tp_rank in range(self.tp_size):
                    mg_model[ep_rank][tp_rank][
                        f"{prefix}.linear_fc1.weight"] = fc1_shards[tp_rank]
                    mg_model[ep_rank][tp_rank][
                        f"{prefix}.linear_fc2.weight"] = fc2_shards[tp_rank]
                    if self.qlora_nf4:
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                             f"{prefix}.linear_fc1.weight",
                                             fc1_shards[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                             f"{prefix}.linear_fc2.weight",
                                             fc2_shards[tp_rank].clone())
            return

        # MoE layer
        router_w = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.mlp.gate.weight")
        if router_w.shape[0] != self.num_experts:
            router_w = router_w[:self.num_experts, :].clone()

        router_b = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias",
            None)
        weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.bias",
                         None)  # 兼容旧格式

        shared_gate = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight")
        shared_up = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight")
        shared_down = weights_dict.pop(
            f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight")

        shared_gate_chunks = torch.chunk(shared_gate, self.tp_size, dim=0)
        shared_up_chunks = torch.chunk(shared_up, self.tp_size, dim=0)
        shared_fc1_shards = [
            torch.cat([g, u], dim=0).clone()
            for g, u in zip(shared_gate_chunks, shared_up_chunks)
        ]
        shared_fc2_shards = [
            t.clone() for t in torch.chunk(shared_down, self.tp_size, dim=1)
        ]

        experts_linear_fc1_list = []
        experts_linear_fc2_list = []

        for expert_idx in range(self.num_experts):
            gate = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            )
            up = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
            )
            down = weights_dict.pop(
                f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
            )

            gate_chunks = torch.chunk(gate, self.expert_tp_size, dim=0)
            up_chunks = torch.chunk(up, self.expert_tp_size, dim=0)
            fc1 = torch.cat(
                [x for pair in zip(gate_chunks, up_chunks) for x in pair],
                dim=0)
            experts_linear_fc1_list.append(fc1.t())
            experts_linear_fc2_list.append(down.t())

        router_key = f"{prefix}.router.weight"
        router_bias_key = f"{prefix}.router.expert_bias"
        shared_fc1_key = f"{prefix}.shared_experts.linear_fc1.weight"
        shared_fc2_key = f"{prefix}.shared_experts.linear_fc2.weight"
        experts_weight1_key = f"{prefix}.experts.weight1"
        experts_weight2_key = f"{prefix}.experts.weight2"

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][router_key] = router_w.clone()
                if router_b is not None:
                    if router_b.shape[0] != self.num_experts:
                        router_b = router_b[:self.num_experts].clone()
                    mg_model[ep_rank][tp_rank][
                        router_bias_key] = router_b.clone()
                mg_model[ep_rank][tp_rank][shared_fc1_key] = shared_fc1_shards[
                    tp_rank].clone()
                mg_model[ep_rank][tp_rank][shared_fc2_key] = shared_fc2_shards[
                    tp_rank].clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                         shared_fc1_key,
                                         shared_fc1_shards[tp_rank].clone())
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                         shared_fc2_key,
                                         shared_fc2_shards[tp_rank].clone())

        if self.moe_grouped_gemm:
            gemm_fc1 = torch.cat(experts_linear_fc1_list).view(
                self.hidden_size, -1)
            gemm_fc2 = torch.cat(experts_linear_fc2_list).view(
                -1, self.hidden_size)

            gemm_fc1_3d = gemm_fc1.view(self.num_experts, self.hidden_size, -1)
            gemm_fc2_3d = gemm_fc2.view(self.num_experts, -1, self.hidden_size)

            if self.moe_tp_extend_ep:
                # Split experts across EP*TP combined: each (tp_rank, ep_rank)
                # gets a unique subset of experts
                bucket_num = self.ep_size * self.tp_size
                gemm_fc1_ep = torch.chunk(gemm_fc1_3d, bucket_num, dim=0)
                gemm_fc2_ep = torch.chunk(gemm_fc2_3d, bucket_num, dim=0)

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        idx = ep_rank * self.tp_size + tp_rank
                        w1 = gemm_fc1_ep[idx].reshape(self.hidden_size,
                                                      -1).clone()
                        w2 = gemm_fc2_ep[idx].reshape(
                            -1, self.hidden_size).clone()
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2
                        if self.qlora_nf4:
                            self.qlora_nf4_quant(mg_model, ep_rank,
                                                 tp_rank, experts_weight1_key,
                                                 w1.clone())
                            self.qlora_nf4_quant(mg_model, ep_rank,
                                                 tp_rank, experts_weight2_key,
                                                 w2.clone())
            else:
                # Standard EP: split by EP only, then optionally by expert_tp_size
                gemm_fc1_ep = torch.chunk(gemm_fc1_3d, self.ep_size, dim=0)
                gemm_fc2_ep = torch.chunk(gemm_fc2_3d, self.ep_size, dim=0)

                for ep_rank in range(self.ep_size):
                    fc1_ep = gemm_fc1_ep[ep_rank]
                    fc2_ep = gemm_fc2_ep[ep_rank]

                    if self.expert_tp_size > 1:
                        if fc1_ep.shape[2] % self.expert_tp_size != 0:
                            raise ValueError(
                                f"grouped_gemm weight1 intermediate_dim ({fc1_ep.shape[2]}) "
                                f"must be divisible by expert_tp_size ({self.expert_tp_size})"
                            )
                        if fc2_ep.shape[1] % self.expert_tp_size != 0:
                            raise ValueError(
                                f"grouped_gemm weight2 intermediate_dim ({fc2_ep.shape[1]}) "
                                f"must be divisible by expert_tp_size ({self.expert_tp_size})"
                            )
                        fc1_shards = torch.chunk(fc1_ep,
                                                 self.expert_tp_size,
                                                 dim=2)
                        fc2_shards = torch.chunk(fc2_ep,
                                                 self.expert_tp_size,
                                                 dim=1)
                    else:
                        fc1_shards = [fc1_ep]
                        fc2_shards = [fc2_ep]

                    for tp_rank in range(self.tp_size):
                        expert_tp_idx = tp_rank % self.expert_tp_size
                        w1 = fc1_shards[expert_tp_idx].reshape(
                            self.hidden_size, -1).clone()
                        w2 = fc2_shards[expert_tp_idx].reshape(
                            -1, self.hidden_size).clone()
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2
                        if self.qlora_nf4:
                            self.qlora_nf4_quant(mg_model, ep_rank,
                                                 tp_rank, experts_weight1_key,
                                                 w1.clone())
                            self.qlora_nf4_quant(mg_model, ep_rank,
                                                 tp_rank, experts_weight2_key,
                                                 w2.clone())
            return

        # non-grouped gemm
        if self.moe_tp_extend_ep:
            bucket_num = self.ep_size * self.tp_size
            num_local_experts = self.num_experts // bucket_num
            for ep_rank in range(self.ep_size):
                for tp_rank in range(self.tp_size):
                    global_base = (ep_rank * self.tp_size +
                                   tp_rank) * num_local_experts
                    for local_experts_idx in range(num_local_experts):
                        global_experts_idx = global_base + local_experts_idx
                        local_fc1 = experts_linear_fc1_list[
                            global_experts_idx].t()
                        local_fc2 = experts_linear_fc2_list[
                            global_experts_idx].t()

                        local_prefix = f"{prefix}.experts.local_experts.{local_experts_idx}"
                        mg_model[ep_rank][tp_rank][
                            f"{local_prefix}.linear_fc1.weight"] = local_fc1.clone(
                            )
                        mg_model[ep_rank][tp_rank][
                            f"{local_prefix}.linear_fc2.weight"] = local_fc2.clone(
                            )
                        if self.qlora_nf4:
                            self.qlora_nf4_quant(
                                mg_model, ep_rank, tp_rank,
                                f"{local_prefix}.linear_fc1.weight",
                                local_fc1.clone())
                            self.qlora_nf4_quant(
                                mg_model, ep_rank, tp_rank,
                                f"{local_prefix}.linear_fc2.weight",
                                local_fc2.clone())
            return

        num_local_experts = self.num_experts // self.ep_size
        for ep_rank in range(self.ep_size):
            for local_experts_idx in range(num_local_experts):
                global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                local_fc1 = experts_linear_fc1_list[global_experts_idx].t()
                local_fc2 = experts_linear_fc2_list[global_experts_idx].t()

                local_prefix = f"{prefix}.experts.local_experts.{local_experts_idx}"

                if self.expert_tp_size > 1:
                    local_fc1_tp = torch.chunk(local_fc1,
                                               self.expert_tp_size,
                                               dim=0)
                    local_fc2_tp = torch.chunk(local_fc2,
                                               self.expert_tp_size,
                                               dim=1)
                    fc1_shards = [t.clone() for t in local_fc1_tp]
                    fc2_shards = [t.clone() for t in local_fc2_tp]
                else:
                    fc1_shards = [local_fc1.clone()]
                    fc2_shards = [local_fc2.clone()]

                for tp_rank in range(self.tp_size):
                    expert_tp_idx = tp_rank % self.expert_tp_size
                    mg_model[ep_rank][tp_rank][
                        f"{local_prefix}.linear_fc1.weight"] = fc1_shards[
                            expert_tp_idx]
                    mg_model[ep_rank][tp_rank][
                        f"{local_prefix}.linear_fc2.weight"] = fc2_shards[
                            expert_tp_idx]
                    if self.qlora_nf4:
                        self.qlora_nf4_quant(
                            mg_model, ep_rank, tp_rank,
                            f"{local_prefix}.linear_fc1.weight",
                            fc1_shards[expert_tp_idx].clone())
                        self.qlora_nf4_quant(
                            mg_model, ep_rank, tp_rank,
                            f"{local_prefix}.linear_fc2.weight",
                            fc2_shards[expert_tp_idx].clone())

    def generate_pp_local_layer_idx(self):
        """generate each pp local layer index"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pp_size):
            if self.num_layer_list is not None:
                layer_list = list(map(int, self.num_layer_list.split(',')))
                pp_local_layer_idx[pp_rank] = [
                    i for i in range(layer_list[pp_rank])
                ]
            else:
                pp_local_layer_idx[pp_rank] = [
                    i for i in range(self.num_layers // self.pp_size)
                ]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(',')))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        return pp_local_layer_idx

    def generate_vpp_local_layer_idx(self):
        vpp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pp_size):
            vpp_local_layer_idx[pp_rank] = defaultdict()

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                vpp_local_layer_idx[pp_rank][vpp_rank] = [
                    i for i in range(self.vpp_stage)
                ]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(',')))
            num_layers_each_pp = self.num_layers // self.pp_size
            if not self.dualpipe:
                for num_noop_layer in noop_list:
                    pp_idx = num_noop_layer % (
                        self.pp_size * self.vpp_stage) // self.vpp_stage
                    vpp_idx = num_noop_layer // self.vpp_stage // self.pp_size
                    local_noop_idx = num_noop_layer % num_layers_each_pp % self.vpp_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)
            else:
                for noop_layer in noop_list:
                    if noop_layer >= self.num_layers // 2:
                        mapping_layer = -(noop_layer - self.num_layers + 1)
                        vpp_idx = 1
                        pp_idx = mapping_layer // (
                            (self.num_layers // 2) // self.pp_size)
                        local_noop_idx = self.vpp_stage - 1 - (
                            mapping_layer - pp_idx * self.vpp_stage)
                    else:
                        vpp_idx = 0
                        pp_idx = noop_layer // (
                            (self.num_layers // 2) // self.pp_size)
                        local_noop_idx = noop_layer - pp_idx * self.vpp_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)

        return vpp_local_layer_idx

    def run(self):
        """save megatron format checkpoint"""
        pp_local_layer_idx = self.generate_pp_local_layer_idx()
        save_model_path = self.mg_path_process(self.mg_save_path)

        if self.vpp_stage is None:
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict(
                    lambda: defaultdict(lambda: defaultdict(dict)))

                pp_weights = self.load_matched_hf_weights(pp_rank)
                if pp_rank == 0:
                    self.set_model_preprocess(pp_weights, mg_model)

                layer_list = self.pprank_layer_idxs[pp_rank]

                local_idx = 0
                cur_pp_local_idx = pp_local_layer_idx[pp_rank]

                for hf_layer in layer_list:
                    logger.info(f"Converting the weights of layer {hf_layer}.")
                    local_layer_idx = cur_pp_local_idx[local_idx]
                    self.set_model_layer_norm(hf_layer, local_layer_idx,
                                              pp_weights, mg_model)
                    self.set_model_layer_attn(hf_layer, local_layer_idx,
                                              pp_weights, mg_model)
                    self.set_model_layer_mlp(hf_layer, local_layer_idx,
                                             pp_weights, mg_model)
                    local_idx += 1

                if pp_rank == self.pp_size - 1:
                    self.set_model_postprocess(pp_weights, mg_model)

                # 检查是否所有权重都被消费
                if pp_weights:
                    unconsumed = list(pp_weights.keys())
                    raise ValueError(
                        f"pp_rank={pp_rank} 存在未被消费的 HF 权重 ({len(unconsumed)}): "
                        f"{unconsumed[:20]}")

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(
                            tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(
                            save_model_path, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path,
                                                      'model_optim_rng.pt')
                        logger.info(f"Saving to {save_file_name}")

                        torch.save(
                            {
                                'model': mg_model[ep_rank][tp_rank],
                                'checkpoint_version': 3.0,
                                'iteration': 1,
                                'args': self._build_checkpoint_args(),
                            },
                            save_file_name,
                            pickle_protocol=4,
                            _use_new_zipfile_serialization=True)

                del mg_model
                gc.collect()
        else:
            vpp_local_layer_idx = self.generate_vpp_local_layer_idx()
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict()

                # Load all weights for the entire pp_rank (across all vpp_ranks)
                # to handle safetensors files that contain multiple layers' weights
                pp_weights = self.load_matched_hf_weights(pp_rank)

                for vpp_rank in range(self.vpp_size):
                    mg_model[vpp_rank] = defaultdict(
                        lambda: defaultdict(lambda: defaultdict(dict)))
                    vpp_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

                    if pp_rank == 0 and vpp_rank == 0:
                        self.set_model_preprocess(pp_weights,
                                                  mg_model[vpp_rank])

                    if self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(pp_weights,
                                                   mg_model[vpp_rank])

                    local_idx = 0
                    cur_vpp_local_idx = vpp_local_layer_idx[pp_rank][vpp_rank]

                    for hf_layer in vpp_list:
                        logger.info(
                            f"Converting the weights of layer {hf_layer}.")
                        local_layer_idx = cur_vpp_local_idx[local_idx]
                        self.set_model_layer_norm(hf_layer, local_layer_idx,
                                                  pp_weights,
                                                  mg_model[vpp_rank])
                        self.set_model_layer_attn(hf_layer, local_layer_idx,
                                                  pp_weights,
                                                  mg_model[vpp_rank])
                        self.set_model_layer_mlp(hf_layer, local_layer_idx,
                                                 pp_weights,
                                                 mg_model[vpp_rank])
                        local_idx += 1

                    if not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(pp_weights,
                                                   mg_model[vpp_rank])

                # Check unconsumed after all vpp_ranks are processed
                if pp_weights:
                    unconsumed = list(pp_weights.keys())
                    raise ValueError(
                        f"pp_rank={pp_rank} 存在未被消费的 HF 权重 ({len(unconsumed)}): "
                        f"{unconsumed[:20]}")

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(
                            tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(
                            save_model_path, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path,
                                                      'model_optim_rng.pt')
                        logger.info(f"Saving to {save_file_name}")
                        model_dict = {
                            'checkpoint_version': 3.0,
                            'iteration': 1,
                            'args': self._build_checkpoint_args(),
                        }

                        for vpp_rank in range(self.vpp_size):
                            model_key = f"model{vpp_rank}"
                            model_dict[model_key] = mg_model[vpp_rank][
                                ep_rank][tp_rank]

                        torch.save(model_dict,
                                   save_file_name,
                                   pickle_protocol=4,
                                   _use_new_zipfile_serialization=True)

                del mg_model
                gc.collect()

        logger.info('Done!')


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
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument('--noop-layers',
                        type=str,
                        default=None,
                        help='Specity the noop layers.')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument('--num-layers',
                        type=int,
                        default=NUM_LAYERS,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=FIRST_K_DENSE_REPLACE,
                        help='Customizing the number of dense layers.')
    parser.add_argument(
        '--expert-tensor-parallel-size',
        type=int,
        default=1,
        help=
        'Expert tensor parallel size (default: 1, experts not split by TP).')
    parser.add_argument(
        '--schedules-method',
        type=str,
        default=None,
        choices=['dualpipev'],
        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--qlora-nf4',
                        action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')
    parser.add_argument(
        '--qk-layernorm',
        action='store_true',
        help='Enable QK LayerNorm (must match pretrain config)')
    parser.add_argument(
        '--moe-tp-extend-ep',
        action='store_true',
        help=
        'use tp group to extend experts parallelism instead of sharding weight tensor of experts in tp group'
    )
    parser.add_argument('--num-experts',
                        type=int,
                        default=NUM_EXPERTS,
                        help='Number of experts.')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=HIDDEN_SIZE,
                        help='Hidden size.')
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=NUM_ATTENTION_HEADS,
                        help='Number of attention heads.')
    parser.add_argument('--num-query-groups',
                        type=int,
                        default=NUM_QUERY_GROUPS,
                        help='Number of query groups for GQA.')
    parser.add_argument('--kv-channels',
                        type=int,
                        default=QK_HEAD_DIM,
                        help='KV channels (head dim).')
    parser.add_argument('--ffn-hidden-size',
                        type=int,
                        default=FFN_HIDDEN_SIZE,
                        help='FFN hidden size for dense layers.')
    parser.add_argument('--moe-ffn-hidden-size',
                        type=int,
                        default=MOE_FFN_HIDDEN_SIZE,
                        help='FFN hidden size for MoE experts.')
    parser.add_argument('--vocab-size',
                        type=int,
                        default=VOCAB_SIZE,
                        help='Vocabulary size.')
    parser.add_argument('--rotary-base',
                        type=float,
                        default=50000.0,
                        help='Rotary base for RoPE')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        num_dense_layers=args.first_k_dense_replace,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        expert_tp_size=args.expert_tensor_parallel_size,
        dualpipe=args.schedules_method,
        qlora_nf4=args.qlora_nf4,
        qk_layernorm=args.qk_layernorm,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        qk_head_dim=args.kv_channels,
        v_head_dim=args.kv_channels,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        vocab_size=args.vocab_size,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        rotary_base=args.rotary_base)
    converter.run()


if __name__ == '__main__':
    main()
