#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Kimi2 MCore to HuggingFace 权重转换脚本

支持:
- GQA (Grouped Query Attention)
- MoE (Mixture of Experts) with 128 experts
- TP (Tensor Parallel) / PP (Pipeline Parallel) / EP (Expert Parallel)
- DualPipeV 调度算法
- QK LayerNorm

模型配置 (来自 scripts/pretrain_kimi2_1t_4k.sh):
- num_layers: 32
- hidden_size: 7168
- num_attention_heads: 64
- num_query_groups: 2 (GQA)
- kv_channels: 128
- num_experts: 128
- first_k_dense_replace: 2
- moe_ffn_hidden_size: 12288
- ffn_hidden_size: 18432
- vocab_size: 163840
"""

import argparse
import gc
import json
import logging as logger
import os
from collections import defaultdict
from itertools import product

import numpy as np
import safetensors.torch
import torch

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)

# Kimi2-1T 模型配置
HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
NUM_ATTENTION_HEADS = 64
NUM_QUERY_GROUPS = 2
KV_CHANNELS = 128
FFN_HIDDEN_SIZE = 18432
MOE_FFN_HIDDEN_SIZE = 12288
VOCAB_SIZE = 163840
NUM_LAYERS = 32


def _mp_prefix(tp_rank: int,
               pp_rank: int,
               ep_rank: int,
               tp_size: int,
               pp_size: int,
               ep_size: int,
               moe_tp_extend_ep: bool = False) -> str:
    """Generate Megatron checkpoint directory prefix.

    When moe_tp_extend_ep is True and tp_size > 1, the expert suffix is
    global_ep = tp_rank + ep_rank * tp_size instead of raw ep_rank.
    """
    if moe_tp_extend_ep and tp_size > 1:
        # When moe_tp_extend_ep is active, effective EP = ep_size * tp_size.
        # The ep_suffix must always be included regardless of raw ep_size.
        ep_suffix = tp_rank + ep_rank * tp_size
        if pp_size == 1:
            return f'mp_rank_{tp_rank:02}_{ep_suffix:03}'
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_suffix:03}'

    if ep_size == 1 and pp_size == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep_size == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp_size == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


def load_data(file_path):
    logger.info(f"Loading the checkpoint from {file_path}.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {file_path}\n"
            f"Please verify your parallel configuration arguments match the checkpoint layout "
            f"(e.g., --moe-tp-extend-ep, --source-tensor-parallel-size, "
            f"--source-pipeline-parallel-size, --source-expert-parallel-size)."
        )
    # Try weights_only=True first for security, fall back to False
    try:
        return torch.load(file_path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(file_path, map_location='cpu', weights_only=False)


def tensor_memory_size(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()


class MgCkptConvert(object):
    """Kimi2 mg -> hf"""

    def __init__(
        self,
        mg_model_path: str,
        hf_save_path: str,
        num_layers: int,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        vpp_stage: int = None,
        num_dense_layers: int = 2,
        num_layer_list: str = None,
        noop_layers: str = None,
        moe_grouped_gemm: bool = False,
        moe_tp_extend_ep: bool = False,
        expert_tp_size: int = 1,
        dualpipe: bool = False,
        hidden_size: int = HIDDEN_SIZE,
        num_experts: int = NUM_EXPERTS,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_query_groups: int = NUM_QUERY_GROUPS,
        kv_channels: int = KV_CHANNELS,
        ffn_hidden_size: int = FFN_HIDDEN_SIZE,
        moe_ffn_hidden_size: int = MOE_FFN_HIDDEN_SIZE,
        vocab_size: int = VOCAB_SIZE,
        qk_layernorm: bool = False,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.vpp_stage = vpp_stage

        self.mg_model_path = mg_model_path
        self.hf_save_path = hf_save_path
        self.iter_path = self.get_iter_path(self.mg_model_path)

        if not os.path.exists(self.hf_save_path):
            os.makedirs(self.hf_save_path)

        self.num_layers = num_layers
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.expert_tp_size = expert_tp_size
        self.dualpipe = True if dualpipe == 'dualpipev' else False
        self.first_k_dense_replace = num_dense_layers
        self.num_layer_list_cmd = num_layer_list

        # 模型架构参数
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.kv_channels = kv_channels
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.vocab_size = vocab_size
        self.qk_layernorm = qk_layernorm

        # GQA 维度计算
        self.q_head_dim = kv_channels
        self.k_head_dim = kv_channels
        self.v_head_dim = kv_channels
        self.q_proj_rows = num_attention_heads * kv_channels
        self.k_proj_rows = num_query_groups * kv_channels
        self.v_proj_rows = num_query_groups * kv_channels
        self.qkv_proj_rows = self.q_proj_rows + self.k_proj_rows + self.v_proj_rows

        self.tp_rank_list = list(range(self.tp_size))
        # Always iterate over full ep_size range.
        # With moe_tp_extend_ep, global_ep = tp_rank + ep_rank * tp_size maps each
        # (tp_rank, ep_rank) pair to a unique directory, giving ep_size * tp_size
        # total files — one per expert bucket.
        self.ep_rank_list = list(range(self.ep_size))
        self.pp_rank_list = list(range(self.pp_size))

        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage

        if dualpipe:
            self.vpp_size = 2
            self.vpp_stage = self.num_layers // self.pp_size // self.vpp_size

        if num_layer_list is None:
            self.num_layer_list = [self.num_layers // self.pp_size
                                   ] * self.pp_size
        else:
            self.num_layer_list = list(map(int, num_layer_list.split(',')))

        num_noop_layers = 0 if self.noop_layers is None else len(
            list(map(int, self.noop_layers.split(','))))
        self.num_real_layers = self.num_layers - num_noop_layers

        self.model_index = {}
        self.pprank_layer_idxs = defaultdict()
        self.vpprank_layer_idxs = defaultdict(dict)
        self.layeridx_vpprank = defaultdict()
        self.layeridx_pprank = defaultdict()

        if self.vpp_stage is not None:
            self.calc_vpprank_layeridxs()
            self.calc_layeridx_vpprank()
        else:
            self.calc_pprank_layeridxs()
            self.calc_layeridx_pprank()
        self.last_save_hf_layer = self.get_last_hf_layer()

        self._valid_parameter()

        self._tensor_size = 0
        self._hf_weight_dict = {}

    def _valid_parameter(self):
        if self.num_layer_list_cmd is None:
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
            layer_list = list(map(int, self.num_layer_list_cmd.split(',')))

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

        if self.last_save_hf_layer == -1:
            raise ValueError(
                'Does not contain a valid model layer. Please check the parameters!'
            )

        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                'num_attention_heads should be divisible by tp_size')

        if self.num_query_groups % self.tp_size != 0:
            raise ValueError('num_query_groups should be divisible by tp_size')

        if self.expert_tp_size > self.tp_size:
            raise ValueError('expert_tp_size cannot exceed tp_size')
        if self.tp_size % self.expert_tp_size != 0:
            raise ValueError('tp_size must be divisible by expert_tp_size')

    @staticmethod
    def get_iter_path(ckpt_path, iteration=None):
        """If the iteration is empty, read from ckpt_path/latest_checkpointed_iteration.txt"""
        if iteration is None:
            latest_iter_file = os.path.join(
                ckpt_path, 'latest_checkpointed_iteration.txt')
            if os.path.exists(latest_iter_file):
                with open(latest_iter_file, 'r') as f:
                    try:
                        iteration = int(f.read().strip())
                    except ValueError:
                        raise ValueError(f"{latest_iter_file} not find")
            else:
                raise FileNotFoundError(f"can not find {latest_iter_file}")

        directory = os.path.join(ckpt_path, f'iter_{iteration:07d}')

        return directory

    def get_last_hf_layer(self):
        """Obtains the last saved hf layer index, combine the postprocess weight"""
        if self.dualpipe:
            if not self.vpprank_layer_idxs[0][1]:
                return self.vpprank_layer_idxs[0][0][-1]
            else:
                return self.vpprank_layer_idxs[0][1][-1]

        # {pp0:{[0,1],[4,5]}, pp1:{[2,3],[]}}  --> last hf: 3
        for pp_rank in range(self.pp_size - 1, -1, -1):
            if self.vpp_stage is not None:
                for vpp_rank in range(self.vpp_size - 1, -1, -1):
                    layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                    if layer_list:
                        return layer_list[-1]
            else:
                layer_list = self.pprank_layer_idxs[pp_rank]
                if layer_list:
                    return layer_list[-1]
        return -1

    def calc_pprank_layeridxs(self) -> None:
        """pp->hf layers, {pp1: [0,1,2,3]}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]
        layers_each_pp = self.num_layer_list.copy()

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(','))):
                cur_pp_rank = layer // (self.num_layers // self.pp_size)
                layers_each_pp[cur_pp_rank] -= 1

        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]
        logger.info(f"###### pprank->hf layer: {self.pprank_layer_idxs}")

    def calc_vpprank_layeridxs(self) -> None:
        """vpp rank -> hf layers, {pp1: {vpp1: [0, 2], vpp2: [1, 3]}}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]
        layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                           for _ in range(self.pp_size)]

        if not self.dualpipe:
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
            # dualpipe_layer_list example
            # pp2: [0 1 2 3 4 5 6 7] -> [0 1 6 7 | 2 3 4 5]
            # pp4: [0 1 2 3 4 5 6 7] -> [0 7 | 1 6 | 2 5 | 3 4]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # calc pp idx and vpp idx of each hf layer
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

    def calc_layeridx_pprank(self):
        """hf layer -> pp_rank & local layer index, {layer5: (pp2, local_layer2)}"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pp_size):
            pp_local_layer_idx[pp_rank] = [
                i for i in range(self.num_layer_list[pp_rank])
            ]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(',')))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        for pp_rank, layeridxs in self.pprank_layer_idxs.items():
            for idx, layer in enumerate(layeridxs):
                self.layeridx_pprank[layer] = (
                    pp_rank, pp_local_layer_idx[pp_rank][idx])
        logger.info(
            f"###### hf layer->pprank&local idx: {self.layeridx_pprank}")

    def calc_layeridx_vpprank(self):
        """hf -> pp_rank & vpp_rank & vpp local layer index, {hf layer: (pp_rank, vpp_rank, vpp_local_idx)}"""
        vpprank_layer_idxs_all = defaultdict(dict)
        layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                           for _ in range(self.pp_size)]

        if not self.dualpipe:
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    vpprank_layer_idxs_all[pp_rank][vpp_rank] = [
                        i for i in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]

            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(','))):
                    pp_idx = layer % (self.pp_size *
                                      self.vpp_stage) // self.vpp_stage
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    local_vpp_idx = layer - (vpp_idx * self.pp_size +
                                             pp_idx) * self.vpp_stage
                    vpprank_layer_idxs_all[pp_idx][vpp_idx].remove(
                        local_vpp_idx)

            for pp_rank in self.vpprank_layer_idxs:
                for vpp_rank, layer_list in self.vpprank_layer_idxs[
                        pp_rank].items():
                    for local_idx, hf_layer in enumerate(layer_list):
                        self.layeridx_vpprank[hf_layer] = (
                            pp_rank, vpp_rank, vpprank_layer_idxs_all[pp_rank]
                            [vpp_rank][local_idx])
        else:
            vpprank_hflayer_idxs = defaultdict(dict)
            dualpipe_layer_list = []
            layers_each_pp = self.num_layers // self.pp_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.num_layers)]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # vpprank_hflayer_idxs {pp_rank: {vpp_rank: [hf_layer1, hf_layer2, ...]}}
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    pp_list = dualpipe_layer_list[pp_rank *
                                                  layers_each_pp:(pp_rank +
                                                                  1) *
                                                  layers_each_pp]
                    vpprank_hflayer_idxs[pp_rank][vpp_rank] = pp_list[
                        vpp_rank * self.vpp_stage:(vpp_rank + 1) *
                        self.vpp_stage]

            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(',')))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[
                0]

            for pp_rank in vpprank_hflayer_idxs:
                for vpp_rank, layer_list in vpprank_hflayer_idxs[
                        pp_rank].items():
                    for local_idx, hf_layer in enumerate(layer_list):
                        if not self.noop_layers:
                            self.layeridx_vpprank[hf_layer] = (pp_rank,
                                                               vpp_rank,
                                                               local_idx)
                        else:
                            if hf_layer in noop_layers_list:
                                continue
                            if hf_layer < min_noop_layer:
                                self.layeridx_vpprank[hf_layer] = (pp_rank,
                                                                   vpp_rank,
                                                                   local_idx)
                            if hf_layer > min_noop_layer:
                                before_nums = sum(noop_layers_list < hf_layer)
                                self.layeridx_vpprank[hf_layer -
                                                      before_nums] = (
                                                          pp_rank, vpp_rank,
                                                          local_idx)

    def get_pt_path_by_tpppep_rank(self,
                                   iter_path,
                                   tp_rank,
                                   pp_rank=None,
                                   ep_rank=None):
        """get megatron weight path"""
        prefix = _mp_prefix(
            tp_rank,
            pp_rank if pp_rank is not None else 0,
            ep_rank if ep_rank is not None else 0,
            self.tp_size,
            self.pp_size,
            self.ep_size,
            self.moe_tp_extend_ep,
        )
        return os.path.join(iter_path, prefix, 'model_optim_rng.pt')

    def set_model_preprocess(self, hf_dict, mg_models):
        """embedding"""
        emb_list = []
        for tp_rank in self.tp_rank_list:
            cur_tp_emb = mg_models[(
                tp_rank,
                self.ep_rank_list[0])].pop('embedding.word_embeddings.weight')
            emb_list.append(cur_tp_emb.clone())
        emb_weights = torch.cat(emb_list, dim=0)
        hf_dict['model.embed_tokens.weight'] = emb_weights

    def set_model_postprocess(self, hf_dict, mg_models):
        """final_norm & output_layer"""
        final_norm_key = 'decoder.final_layernorm.weight'
        final_norm = mg_models[(self.tp_rank_list[0],
                                self.ep_rank_list[0])].pop(final_norm_key)
        hf_dict['model.norm.weight'] = final_norm.clone()

        lm_head_list = []
        for tp_rank in self.tp_rank_list:
            cur_tp_head = mg_models[(
                tp_rank, self.ep_rank_list[0])].pop('output_layer.weight')
            lm_head_list.append(cur_tp_head.clone())
        lm_head_weights = torch.cat(lm_head_list, dim=0)
        hf_dict['lm_head.weight'] = lm_head_weights.clone()

    def set_model_layer_norm(self, hf_dict, mg_models, hf_layer_idx,
                             local_layer_idx):
        """input norm & post attn norm"""
        input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
        pre_mlp_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"

        input_norm = mg_models[(self.tp_rank_list[0],
                                self.ep_rank_list[0])].pop(input_norm_key)
        pre_mlp_norm = mg_models[(self.tp_rank_list[0],
                                  self.ep_rank_list[0])].pop(pre_mlp_norm_key)

        hf_dict[
            f"model.layers.{hf_layer_idx}.input_layernorm.weight"] = input_norm.clone(
            )
        hf_dict[
            f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight"] = pre_mlp_norm.clone(
            )

    def set_model_attn(self, hf_dict, mg_models, hf_layer_idx,
                       local_layer_idx):
        """
        GQA Attention 转换

        MCore 格式 (per TP rank):
        - linear_qkv: [q_per_tp + k_per_tp + v_per_tp, hidden_size]
          布局: [Q_tp; K_tp; V_tp] along dim=0
        - linear_proj: [hidden_size, proj_dim_per_tp]

        HF 格式:
        - q_proj: [num_attention_heads * head_dim, hidden_size]
        - k_proj: [num_query_groups * head_dim, hidden_size]
        - v_proj: [num_query_groups * head_dim, hidden_size]
        - o_proj: [hidden_size, num_attention_heads * head_dim]
        """
        prefix = f"decoder.layers.{local_layer_idx}"
        qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
        proj_key = f"{prefix}.self_attention.linear_proj.weight"

        linear_proj_list = []
        q_parts = []
        k_parts = []
        v_parts = []

        # GQA: 每个 TP shard 内部是 [Q_tp; K_tp; V_tp] 布局
        # 必须先按 shard 拆分 QKV，再合并各 shard 的 Q/K/V
        heads_per_tp = self.num_attention_heads // self.tp_size
        kv_heads_per_tp = self.num_query_groups // self.tp_size
        q_per_tp = heads_per_tp * self.q_head_dim
        k_per_tp = kv_heads_per_tp * self.k_head_dim
        v_per_tp = kv_heads_per_tp * self.v_head_dim

        for tp_rank in self.tp_rank_list:
            cur_linear_proj = mg_models[(tp_rank,
                                         self.ep_rank_list[0])].pop(proj_key)
            linear_proj_list.append(cur_linear_proj.clone())

            cur_qkv = mg_models[(tp_rank, self.ep_rank_list[0])].pop(qkv_key)
            # Per-shard QKV split
            q_r, k_r, v_r = torch.split(cur_qkv,
                                        [q_per_tp, k_per_tp, v_per_tp],
                                        dim=0)
            q_parts.append(q_r)
            k_parts.append(k_r)
            v_parts.append(v_r)

        # 合并 TP shards
        o_proj = torch.cat(linear_proj_list, dim=1)
        q_proj = torch.cat(q_parts, dim=0)
        k_proj = torch.cat(k_parts, dim=0)
        v_proj = torch.cat(v_parts, dim=0)

        # QK LayerNorm (可选)
        q_norm_key = f"{prefix}.self_attention.q_layernorm.weight"
        k_norm_key = f"{prefix}.self_attention.k_layernorm.weight"
        q_ln = mg_models[(self.tp_rank_list[0],
                          self.ep_rank_list[0])].pop(q_norm_key, None)
        k_ln = mg_models[(self.tp_rank_list[0],
                          self.ep_rank_list[0])].pop(k_norm_key, None)

        hf_dict[
            f"model.layers.{hf_layer_idx}.self_attn.q_proj.weight"] = q_proj.clone(
            )
        hf_dict[
            f"model.layers.{hf_layer_idx}.self_attn.k_proj.weight"] = k_proj.clone(
            )
        hf_dict[
            f"model.layers.{hf_layer_idx}.self_attn.v_proj.weight"] = v_proj.clone(
            )
        hf_dict[
            f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight"] = o_proj.clone(
            )

        if q_ln is not None:
            hf_dict[
                f"model.layers.{hf_layer_idx}.self_attn.q_layernorm.weight"] = q_ln.clone(
                )
        if k_ln is not None:
            hf_dict[
                f"model.layers.{hf_layer_idx}.self_attn.k_layernorm.weight"] = k_ln.clone(
                )

    def linear_fc1_gather_from_tp(self, mg_models, fc1_key, ep_rank=0):
        """cat linear fc1 (gate and up) — split gate/up per shard, then concat"""
        gate_list, up_list = [], []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc1 = mg_models[(tp_rank, ep_rank)].pop(fc1_key)
            cur_gate, cur_up = torch.chunk(cur_linear_fc1, 2, dim=0)
            gate_list.append(cur_gate.clone())
            up_list.append(cur_up.clone())

        gate_weights = torch.cat(gate_list, dim=0)
        up_weights = torch.cat(up_list, dim=0)
        return gate_weights, up_weights

    def linear_fc2_gather_from_tp(self, mg_models, fc2_key, ep_rank=0):
        """cat linear fc2 (down)"""
        down_list = []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc2 = mg_models[(tp_rank, ep_rank)].pop(fc2_key)
            down_list.append(cur_linear_fc2.clone())

        down_weights = torch.cat(down_list, dim=1)
        return down_weights

    def set_model_mlp(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx):
        """
        Dense MLP + MoE 转换

        Dense 层 (hf_layer_idx < first_k_dense_replace):
        - HF: gate_proj, up_proj, down_proj

        MoE 层:
        - HF: gate.weight, gate.e_score_correction_bias
        - HF: shared_experts.gate_proj, up_proj, down_proj
        - HF: experts.{i}.gate_proj, up_proj, down_proj
        """
        prefix = f"decoder.layers.{local_layer_idx}"

        if hf_layer_idx < self.first_k_dense_replace:
            # Dense MLP
            linear_fc1_key = f"{prefix}.mlp.linear_fc1.weight"
            linear_fc2_key = f"{prefix}.mlp.linear_fc2.weight"

            gate_weights, up_weights = self.linear_fc1_gather_from_tp(
                mg_models, linear_fc1_key)
            down_weights = self.linear_fc2_gather_from_tp(
                mg_models, linear_fc2_key)

            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight"] = gate_weights.clone(
                )
            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.up_proj.weight"] = up_weights.clone(
                )
            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.down_proj.weight"] = down_weights.clone(
                )
        else:
            # MoE
            router_key = f"{prefix}.mlp.router.weight"
            router_bias_key = f"{prefix}.mlp.router.expert_bias"
            shared_fc1_key = f"{prefix}.mlp.shared_experts.linear_fc1.weight"
            shared_fc2_key = f"{prefix}.mlp.shared_experts.linear_fc2.weight"
            expert_weight1_key = f"{prefix}.mlp.experts.weight1"
            expert_weight2_key = f"{prefix}.mlp.experts.weight2"

            router_weights = mg_models[(self.tp_rank_list[0],
                                        self.ep_rank_list[0])].pop(router_key)
            router_bias_weights = mg_models[(self.tp_rank_list[0],
                                             self.ep_rank_list[0])].pop(
                                                 router_bias_key, None)

            shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_tp(
                mg_models, shared_fc1_key)
            shared_down_weights = self.linear_fc2_gather_from_tp(
                mg_models, shared_fc2_key)

            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.gate.weight"] = router_weights.clone(
                )
            if router_bias_weights is not None:
                hf_dict[
                    f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias"] = router_bias_weights.clone(
                    )
            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight"] = shared_gate_weights.clone(
                )
            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight"] = shared_up_weights.clone(
                )
            hf_dict[
                f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight"] = shared_down_weights.clone(
                )

            # Experts
            hf_local_gate_key = 'model.layers.{}.mlp.experts.{}.gate_proj.weight'
            hf_local_up_key = 'model.layers.{}.mlp.experts.{}.up_proj.weight'
            hf_local_down_key = 'model.layers.{}.mlp.experts.{}.down_proj.weight'

            if self.moe_tp_extend_ep and self.tp_size > 1:
                # With moe_tp_extend_ep, experts are split across ep_size * tp_size
                # buckets.  Each (tp_rank, ep_rank) file holds
                # num_experts // (ep_size * tp_size) experts.
                local_expert_nums = self.num_experts // (self.ep_size *
                                                         self.tp_size)
            else:
                local_expert_nums = self.num_experts // self.ep_size

            if self.moe_grouped_gemm:
                for ep_rank in self.ep_rank_list:
                    ep_weight1_list, ep_weight2_list = [], []
                    for tp_rank in self.tp_rank_list:
                        cur_weight1 = mg_models[(
                            tp_rank, ep_rank)].pop(expert_weight1_key)
                        cur_weight2 = mg_models[(
                            tp_rank, ep_rank)].pop(expert_weight2_key)
                        # grouped_gemm 格式: weight1: [hidden_size, num_local_experts * intermediate_size * 2]
                        ep_weight1_list.append(
                            cur_weight1.reshape(local_expert_nums,
                                                self.hidden_size, -1))
                        ep_weight2_list.append(
                            cur_weight2.reshape(local_expert_nums, -1,
                                                self.hidden_size))

                    if self.moe_tp_extend_ep:
                        # all experts cut into tp_size*ep_size
                        for tp_rank in self.tp_rank_list:
                            cur_weight1_bucket = ep_weight1_list[tp_rank]
                            cur_weight2_bucket = ep_weight2_list[tp_rank]
                            cur_w1_list = torch.chunk(cur_weight1_bucket,
                                                      local_expert_nums,
                                                      dim=0)
                            cur_w2_list = torch.chunk(cur_weight2_bucket,
                                                      local_expert_nums,
                                                      dim=0)

                            global_expert_idx = ep_rank * self.tp_size + tp_rank
                            for idx in range(local_expert_nums):
                                local_w1 = cur_w1_list[idx].reshape(
                                    self.hidden_size, -1)
                                local_w2 = cur_w2_list[idx].reshape(
                                    -1, self.hidden_size)
                                expert_idx = global_expert_idx * local_expert_nums + idx
                                gate, up = torch.chunk(local_w1.t(), 2, dim=0)
                                down = local_w2.t()
                                hf_dict[hf_local_gate_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = gate.contiguous().clone()
                                hf_dict[hf_local_up_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = up.contiguous().clone()
                                hf_dict[hf_local_down_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = down.contiguous().clone()
                    else:
                        # moe_tp_extend_ep=False: experts split by expert_tp_size across TP ranks
                        if self.expert_tp_size > 1:
                            # TP ranks 0..expert_tp_size-1 each hold a unique
                            # shard; higher ranks are replicas.
                            unique_tp_indices = list(range(
                                self.expert_tp_size))
                            ep_weight1 = torch.cat([
                                ep_weight1_list[i] for i in unique_tp_indices
                            ],
                                                   dim=2)
                            ep_weight2 = torch.cat([
                                ep_weight2_list[i] for i in unique_tp_indices
                            ],
                                                   dim=1)

                            for local_idx in range(local_expert_nums):
                                expert_idx = ep_rank * local_expert_nums + local_idx
                                ep_w1_expert = ep_weight1[local_idx].reshape(
                                    self.hidden_size, -1).t()
                                # gate/up are interleaved per expert_tp shard
                                chunks = torch.chunk(ep_w1_expert,
                                                     self.expert_tp_size,
                                                     dim=0)
                                gate_list, up_list = [], []
                                for chunk in chunks:
                                    g, u = torch.chunk(chunk, 2, dim=0)
                                    gate_list.append(
                                        g.reshape(-1, self.hidden_size))
                                    up_list.append(
                                        u.reshape(-1, self.hidden_size))
                                local_gate = torch.cat(gate_list, dim=0)
                                local_up = torch.cat(up_list, dim=0)
                                local_down = ep_weight2[local_idx].t()

                                hf_dict[hf_local_gate_key.format(
                                    hf_layer_idx, expert_idx
                                )] = local_gate.contiguous().clone()
                                hf_dict[hf_local_up_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = local_up.contiguous().clone(
                                    )
                                hf_dict[hf_local_down_key.format(
                                    hf_layer_idx, expert_idx
                                )] = local_down.contiguous().clone()
                        else:
                            # expert_tp_size=1: all TP ranks hold identical expert weights
                            ep_weight1 = ep_weight1_list[0]
                            ep_weight2 = ep_weight2_list[0]

                            for local_idx in range(local_expert_nums):
                                expert_idx = ep_rank * local_expert_nums + local_idx
                                ep_weight1_expert = ep_weight1[
                                    local_idx].reshape(self.hidden_size,
                                                       -1).t()
                                gate, up = torch.chunk(ep_weight1_expert,
                                                       2,
                                                       dim=0)
                                local_gate = gate.reshape(-1, self.hidden_size)
                                local_up = up.reshape(-1, self.hidden_size)
                                local_down = ep_weight2[local_idx].t()

                                hf_dict[hf_local_gate_key.format(
                                    hf_layer_idx, expert_idx
                                )] = local_gate.contiguous().clone()
                                hf_dict[hf_local_up_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = local_up.contiguous().clone(
                                    )
                                hf_dict[hf_local_down_key.format(
                                    hf_layer_idx, expert_idx
                                )] = local_down.contiguous().clone()
            else:
                # local_experts 格式
                local_prefix = f"{prefix}.mlp.experts.local_experts"

                if self.moe_tp_extend_ep:
                    for ep_rank in self.ep_rank_list:
                        for tp_rank in self.tp_rank_list:
                            global_expert_base = (ep_rank * self.tp_size +
                                                  tp_rank) * local_expert_nums
                            for local_idx in range(local_expert_nums):
                                expert_idx = global_expert_base + local_idx
                                local_fc1_key = f"{local_prefix}.{local_idx}.linear_fc1.weight"
                                local_fc2_key = f"{local_prefix}.{local_idx}.linear_fc2.weight"

                                cur_fc1 = mg_models[(
                                    tp_rank, ep_rank)].pop(local_fc1_key)
                                cur_gate, cur_up = torch.chunk(cur_fc1,
                                                               2,
                                                               dim=0)
                                cur_down = mg_models[(
                                    tp_rank, ep_rank)].pop(local_fc2_key)

                                hf_dict[hf_local_gate_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = cur_gate.contiguous().clone(
                                    )
                                hf_dict[hf_local_up_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = cur_up.contiguous().clone()
                                hf_dict[hf_local_down_key.format(
                                    hf_layer_idx,
                                    expert_idx)] = cur_down.contiguous().clone(
                                    )
                else:
                    for ep_rank in self.ep_rank_list:
                        for local_idx in range(local_expert_nums):
                            expert_idx = ep_rank * local_expert_nums + local_idx
                            local_fc1_key = f"{local_prefix}.{local_idx}.linear_fc1.weight"
                            local_fc2_key = f"{local_prefix}.{local_idx}.linear_fc2.weight"

                            if self.expert_tp_size > 1:
                                # expert_tp_size>1: each TP rank holds a
                                # unique shard; gather from the first
                                # expert_tp_size TP ranks.
                                fc1_parts = [
                                    mg_models[(tp, ep_rank)].pop(local_fc1_key)
                                    for tp in range(self.expert_tp_size)
                                ]
                                fc2_parts = [
                                    mg_models[(tp, ep_rank)].pop(local_fc2_key)
                                    for tp in range(self.expert_tp_size)
                                ]
                                cur_fc1 = torch.cat(fc1_parts, dim=0)
                                cur_fc2 = torch.cat(fc2_parts, dim=1)

                                # Release replica TP ranks
                                for tp in range(self.expert_tp_size,
                                                self.tp_size):
                                    mg_models[(tp, ep_rank)].pop(
                                        local_fc1_key, None)
                                    mg_models[(tp, ep_rank)].pop(
                                        local_fc2_key, None)
                            else:
                                # expert_tp_size=1: all TP ranks hold
                                # identical expert weights
                                cur_fc1 = mg_models[(
                                    self.tp_rank_list[0],
                                    ep_rank)].pop(local_fc1_key)
                                cur_fc2 = mg_models[(
                                    self.tp_rank_list[0],
                                    ep_rank)].pop(local_fc2_key)

                                for tp_rank in self.tp_rank_list[1:]:
                                    mg_models[(tp_rank, ep_rank)].pop(
                                        local_fc1_key, None)
                                    mg_models[(tp_rank, ep_rank)].pop(
                                        local_fc2_key, None)

                            if self.expert_tp_size > 1:
                                # gate/up are interleaved per expert_tp
                                # shard; de-interleave them.
                                chunks = torch.chunk(cur_fc1,
                                                     self.expert_tp_size,
                                                     dim=0)
                                gate_list, up_list = [], []
                                for chunk in chunks:
                                    g, u = torch.chunk(chunk, 2, dim=0)
                                    gate_list.append(g)
                                    up_list.append(u)
                                local_gate = torch.cat(gate_list, dim=0)
                                local_up = torch.cat(up_list, dim=0)
                            else:
                                local_gate, local_up = torch.chunk(cur_fc1,
                                                                   2,
                                                                   dim=0)
                            local_down = cur_fc2

                            hf_dict[hf_local_gate_key.format(
                                hf_layer_idx,
                                expert_idx)] = local_gate.contiguous().clone()
                            hf_dict[hf_local_up_key.format(
                                hf_layer_idx,
                                expert_idx)] = local_up.contiguous().clone()
                            hf_dict[hf_local_down_key.format(
                                hf_layer_idx,
                                expert_idx)] = local_down.contiguous().clone()

    def save_safetensors(self, hf_dict, cur_file_idx):
        """save safetensors file"""
        num_files = self.num_real_layers

        safetensors_file_name = f"model-{cur_file_idx:05d}-of-{num_files:06d}.safetensors"
        for key in hf_dict.keys():
            self.model_index[key] = safetensors_file_name
            self._tensor_size += tensor_memory_size(hf_dict[key])

        logger.info(f"Saving to {safetensors_file_name}")
        safetensors.torch.save_file(hf_dict,
                                    os.path.join(self.hf_save_path,
                                                 safetensors_file_name),
                                    metadata={'format': 'pt'})

    def read_pp_rank_weights(self, pp_rank, mg_models):
        """get pp_rank weights"""
        layer_list = self.pprank_layer_idxs[pp_rank]

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and layer == 0:
                self.set_model_preprocess(self._hf_weight_dict, mg_models)
            local_idx = self.layeridx_pprank[layer][1]

            self.set_model_layer_norm(self._hf_weight_dict, mg_models, layer,
                                      local_idx)
            self.set_model_attn(self._hf_weight_dict, mg_models, layer,
                                local_idx)
            self.set_model_mlp(self._hf_weight_dict, mg_models, layer,
                               local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(self._hf_weight_dict, layer + 1)
                self._hf_weight_dict = {}

        if pp_rank == self.pp_size - 1:
            self.set_model_postprocess(self._hf_weight_dict, mg_models)
            self.save_safetensors(self._hf_weight_dict,
                                  self.last_save_hf_layer + 1)
            self._hf_weight_dict = {}

    def read_vpp_rank_weights(self, pp_rank, vpp_rank, mg_models):
        """get vpp_rank weights"""
        layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and vpp_rank == 0 and layer == 0:
                self.set_model_preprocess(self._hf_weight_dict, mg_models)
            local_idx = self.layeridx_vpprank[layer][2]

            self.set_model_layer_norm(self._hf_weight_dict, mg_models, layer,
                                      local_idx)
            self.set_model_attn(self._hf_weight_dict, mg_models, layer,
                                local_idx)
            self.set_model_mlp(self._hf_weight_dict, mg_models, layer,
                               local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(self._hf_weight_dict, layer + 1)
                self._hf_weight_dict = {}

        # dualpipe: post weight(norm+lm_head) in pp0vpp-1
        dualpipe_flag = self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1
        # no dualpipe: post weight in pp-1vpp-1
        norm_flag = not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1

        if dualpipe_flag or norm_flag:
            self.set_model_postprocess(self._hf_weight_dict, mg_models)
            self.save_safetensors(self._hf_weight_dict,
                                  self.last_save_hf_layer + 1)
            self._hf_weight_dict = {}

    def run(self):
        for pp_rank in self.pp_rank_list:
            if self.vpp_stage is None:
                mg_weights = {}
                for tp_rank, ep_rank in product(self.tp_rank_list,
                                                self.ep_rank_list):
                    model_path = self.get_pt_path_by_tpppep_rank(
                        self.iter_path, tp_rank, pp_rank, ep_rank)
                    tmp_model = load_data(model_path)['model']
                    mg_weights[(tp_rank, ep_rank)] = tmp_model

                self.read_pp_rank_weights(pp_rank, mg_weights)
                del mg_weights
            else:
                # Load each checkpoint file once, then extract vpp models
                # sequentially.  Avoids re-reading the same file for each
                # vpp_rank, cutting disk I/O by (vpp_size-1)/vpp_size.
                raw_data = {}
                for tp_rank, ep_rank in product(self.tp_rank_list,
                                                self.ep_rank_list):
                    pt_path = self.get_pt_path_by_tpppep_rank(
                        self.iter_path, tp_rank, pp_rank, ep_rank)
                    raw_data[(tp_rank, ep_rank)] = load_data(pt_path)

                for vpp_rank in range(self.vpp_size):
                    mg_weights = {}
                    for tp_rank, ep_rank in product(self.tp_rank_list,
                                                    self.ep_rank_list):
                        mg_weights[(tp_rank, ep_rank)] = raw_data[(
                            tp_rank, ep_rank)].pop(f'model{vpp_rank}')
                    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)
                    del mg_weights

                del raw_data
            gc.collect()

        model_index_file_path = os.path.join(self.hf_save_path,
                                             'model.safetensors.index.json')
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(
                {
                    'metadata': {
                        'total_size': self._tensor_size
                    },
                    'weight_map': self.model_index
                },
                json_file,
                indent=4)
        logger.info('Done!')


def get_args():
    parser = argparse.ArgumentParser(
        description='Kimi2 MCore to HuggingFace checkpoint converter')
    parser.add_argument('--load-dir',
                        type=str,
                        required=True,
                        help='Directory to load MCore model checkpoint from')
    parser.add_argument(
        '--save-dir',
        type=str,
        required=True,
        help='Directory to save HuggingFace model checkpoint to')
    parser.add_argument(
        '--source-tensor-parallel-size',
        type=int,
        default=1,
        help='Source tensor model parallel size, defaults to 1')
    parser.add_argument(
        '--source-pipeline-parallel-size',
        type=int,
        default=1,
        help='Source pipeline model parallel size, default to 1')
    parser.add_argument('--source-expert-parallel-size',
                        type=int,
                        default=1,
                        help='Source expert model parallel size, default to 1')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage',
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        default=False,
                        help='Use moe grouped gemm.')
    parser.add_argument('--noop-layers',
                        type=str,
                        default=None,
                        help='Specify the noop layers.')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument(
        '--moe-tp-extend-ep',
        action='store_true',
        help=
        'use tp group to extend experts parallelism instead of sharding weight tensor of experts in tp group'
    )
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
    parser.add_argument('--num-layers',
                        type=int,
                        default=NUM_LAYERS,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=2,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=HIDDEN_SIZE,
                        help='Hidden size.')
    parser.add_argument('--num-experts',
                        type=int,
                        default=NUM_EXPERTS,
                        help='Number of experts.')
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
                        default=KV_CHANNELS,
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
    parser.add_argument(
        '--qk-layernorm',
        action='store_true',
        help='Enable QK LayerNorm (must match training config)')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    converter = MgCkptConvert(
        mg_model_path=args.load_dir,
        hf_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.source_tensor_parallel_size,
        pp_size=args.source_pipeline_parallel_size,
        ep_size=args.source_expert_parallel_size,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_dense_layers=args.first_k_dense_replace,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        expert_tp_size=args.expert_tensor_parallel_size,
        dualpipe=args.schedules_method,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        kv_channels=args.kv_channels,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        vocab_size=args.vocab_size,
        qk_layernorm=args.qk_layernorm,
    )
    converter.run()


if __name__ == '__main__':
    main()
