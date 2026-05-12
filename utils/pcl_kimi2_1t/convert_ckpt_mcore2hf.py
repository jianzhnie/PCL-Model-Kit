import argparse
import gc
import hashlib
import inspect
import json
import logging as logger
import os
import shutil
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import safetensors.torch
import torch

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value == '':
        return None
    return list(map(int, value.split(',')))


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int,
               ep: int) -> str:
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


def _resolve_iter_dir(load_dir: str) -> str:
    latest = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    if os.path.isfile(latest):
        with open(latest) as f:
            it = f.read().strip()
        latest_dir = os.path.join(load_dir, f'iter_{int(it):07d}')
        if os.path.isdir(latest_dir):
            return latest_dir
    if os.path.isdir(os.path.join(load_dir, 'iter_0000001')):
        return os.path.join(load_dir, 'iter_0000001')
    if os.path.basename(load_dir).startswith('iter_'):
        return load_dir
    raise FileNotFoundError(f'无法定位迭代目录: {load_dir}')


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
    for fn in sorted(os.listdir(save_dir)):
        if fn.endswith('.safetensors') or fn.endswith('.json'):
            files.append(fn)
    payload: dict[str, str] = {}
    for fn in files:
        p = os.path.join(save_dir, fn)
        if os.path.isfile(p):
            payload[fn] = _sha256_file(p)
    with open(out_path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _read_safetensors_header_keys(path: str) -> list[str]:
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = f.read(int(header_len))
    meta = json.loads(header.decode('utf-8'))
    keys = [k for k in meta.keys() if k != '__metadata__']
    keys.sort()
    return keys


def _write_hf_index_by_scan(save_dir: str) -> tuple[str, int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_files = sorted(fn for fn in os.listdir(save_dir)
                         if fn.endswith('.safetensors'))
    for fn in shard_files:
        p = os.path.join(save_dir, fn)
        if not os.path.isfile(p):
            continue
        total_size += int(os.path.getsize(p))
        for k in _read_safetensors_header_keys(p):
            weight_map[k] = fn
    index_path = os.path.join(save_dir, 'model.safetensors.index.json')
    with open(index_path, 'w') as f:
        json.dump(
            {
                'metadata': {
                    'total_size': total_size
                },
                'weight_map': weight_map
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return index_path, total_size


def _torch_load_compat(path: str, disable_mmap: bool) -> dict:
    base = {'map_location': 'cpu'}
    sig = None
    try:
        sig = inspect.signature(torch.load)
    except Exception:
        sig = None

    support_weights_only = bool(sig and 'weights_only' in sig.parameters)
    support_mmap = bool(sig and 'mmap' in sig.parameters and not disable_mmap)

    candidates: list[dict] = []
    if support_weights_only:
        kw = dict(base)
        kw['weights_only'] = True
        if support_mmap:
            kw['mmap'] = True
        candidates.append(kw)
    kw = dict(base)
    if support_weights_only:
        kw['weights_only'] = False
    if support_mmap:
        kw['mmap'] = True
    candidates.append(kw)
    kw = dict(base)
    if support_weights_only:
        kw['weights_only'] = False
    candidates.append(kw)

    last = None
    for kw in candidates:
        try:
            return torch.load(path, **kw)
        except Exception as e:
            last = e
            continue
    raise last  # type: ignore[misc]


class MgCkptConvert:

    def __init__(
        self,
        mg_load_dir: str,
        hf_save_dir: str,
        num_layers: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        first_k_dense_replace: int,
        hidden_size: int,
        num_experts: int,
        num_attention_heads: int,
        num_query_groups: int,
        qk_head_dim: int,
        moe_grouped_gemm: bool,
        schedules_method: str | None,
        vpp_stage: int | None,
        num_layer_list: str | None,
        noop_layers: str | None,
        qk_layernorm: bool,
        rotary_base: float,
        vocab_size: int | None = None,
        max_position_embeddings: int | None = None,
        v_head_dim: int | None = None,
        tie_word_embeddings: bool = False,
        ffn_hidden_size: int | None = None,
        moe_ffn_hidden_size: int | None = None,
        n_shared_experts: int = 1,
        moe_router_topk: int = 2,
        hf_config_template: str | None = None,
        cast_dtype: str | None = None,
        io_threads: int = 4,
        disable_mmap: bool = False,
        expert_tp_size: int = 1,
    ):
        self.verbose = os.environ.get('CKPT_CONVERT_VERBOSE', '1') != '0'
        self.log_rank_load = os.environ.get('CKPT_CONVERT_LOG_RANK',
                                            '1') != '0'
        self.mg_load_dir = mg_load_dir
        self.hf_save_dir = hf_save_dir
        self.iter_dir = _resolve_iter_dir(mg_load_dir)
        if self.verbose:
            logger.info('Resolved iter_dir: %s', self.iter_dir)

        self.num_layers = num_layers
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.expert_tp_size = max(1, int(expert_tp_size))
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.qk_head_dim = qk_head_dim
        # GQA: qk_head_dim = kv_channels, v_head_dim defaults to qk_head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else qk_head_dim
        self.moe_grouped_gemm = moe_grouped_gemm
        self.qk_layernorm = qk_layernorm
        self.schedules_method = schedules_method
        self.dualpipe = schedules_method == 'dualpipev'
        self.vpp_stage = vpp_stage
        self.num_layer_list = num_layer_list
        self.noop_layers_list = sorted(_parse_int_list(noop_layers) or [])
        self.rotary_base = rotary_base
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.n_shared_experts = n_shared_experts
        self.moe_router_topk = moe_router_topk
        self.detected_num_kv_heads = None
        self._resolved_q_head_dim = None
        self._resolved_kv_heads_per_tp = None
        self.cast_dtype = cast_dtype
        self.io_threads = max(1, int(io_threads))
        self.disable_mmap = bool(disable_mmap)
        self._target_dtype = _dtype_from_str(
            cast_dtype) if cast_dtype else None
        self._sparse_cache: dict[tuple[int, int, int, int | None],
                                 dict[str, torch.Tensor]] = {}

        os.makedirs(self.hf_save_dir, exist_ok=True)

        self.vpp_size, self._vpp_model_keys = self._detect_vpp()
        if self.verbose:
            logger.info('Detected vpp_size=%s dualpipe=%s', self.vpp_size,
                        self.dualpipe)
        if self.vpp_size is not None and self.vpp_stage is None:
            if self.dualpipe:
                layers_each_pp = self.num_layers // self.pp_size
                self.vpp_stage = layers_each_pp // 2
            else:
                raise ValueError('检测到 vpp，但未提供 --vpp-stage 且非 dualpipev')

        self._validate()

        self.num_real_layers = self.num_layers - len(self.noop_layers_list)

        if self.vpp_size is None:
            self.pprank_layer_idxs: dict[int, list[int]] = defaultdict(list)
            self.layer2loc: dict[int, tuple[int, int]] = {}
            self._build_pprank_layer_map()
        else:
            self.vpprank_layer_idxs: dict[int,
                                          dict[int,
                                               list[int]]] = defaultdict(dict)
            self.layer2loc_vpp: dict[int, tuple[int, int, int]] = {}
            self._build_vpprank_layer_map()

        # GQA: RoPE uses full qk_head_dim
        self._head_dim_for_rope = qk_head_dim
        self._reset_inv_freq()

        self.weight_map: dict[str, str] = {}
        self.total_size_bytes = 0
        self._rank_dir_map: dict[tuple[int, int], list[int]] = {}
        self._build_rank_dir_map()
        if self.verbose:
            logger.info('Rank dir map size=%d', len(self._rank_dir_map))
            sample_keys = sorted(
                self._rank_dir_map.keys())[:min(12, len(self._rank_dir_map))]
            for k in sample_keys:
                eps = self._rank_dir_map[k]
                logger.info('Rank dir map sample tp=%d pp=%d: ep_count=%d',
                            k[0], k[1], len(eps))
            missing = []
            for tp in range(self.tp_size):
                for pp in range(self.pp_size):
                    if (tp, pp) not in self._rank_dir_map:
                        missing.append((tp, pp))
            if missing:
                logger.info('Missing (tp,pp) in rank dirs: %s', missing[:20])

    def _reset_inv_freq(self) -> None:
        # GQA: 使用完整的 head_dim 计算 RoPE 频率
        inv_dim = int(self._head_dim_for_rope)
        if inv_dim <= 0:
            self.inv_freq = torch.empty((0, ), dtype=torch.float32)
            return
        inv_freq = 1.0 / (self.rotary_base**(
            torch.arange(0, inv_dim, 2, dtype=torch.float32) / inv_dim))
        self.inv_freq = inv_freq

    def _infer_qkv_layout(self, shard_rows: int) -> tuple[int, int] | None:
        if self.num_attention_heads % self.tp_size != 0:
            return None
        heads_per_tp = self.num_attention_heads // self.tp_size
        if heads_per_tp <= 0:
            return None
        max_q_head_dim = shard_rows // heads_per_tp
        if max_q_head_dim <= 0:
            return None

        # GQA: 只考虑标准 head_dim
        preferred: list[int] = []
        preferred.append(self.qk_head_dim)
        preferred = [d for d in preferred if isinstance(d, int) and d > 0]

        seen: set[int] = set()
        candidates: list[int] = []
        for d in preferred:
            if d not in seen:
                candidates.append(d)
                seen.add(d)

        for d in range(8, max_q_head_dim + 1, 8):
            if d not in seen:
                candidates.append(d)
                seen.add(d)

        for q_head_dim in candidates:
            q_per_tp = heads_per_tp * q_head_dim
            rem = shard_rows - q_per_tp
            if rem < 0:
                continue
            denom = q_head_dim + self.v_head_dim
            if denom <= 0 or rem % denom != 0:
                continue
            kv_heads_per_tp = rem // denom
            if kv_heads_per_tp <= 0:
                continue
            return q_head_dim, kv_heads_per_tp

        return None

    def _repo_models_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'models'))

    def _select_config_template(self) -> str:
        models_dir = self._repo_models_dir()
        cfg_file = os.path.join(models_dir, 'config.json')
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file) as f:
                    d = json.load(f)
                if int(d.get('hidden_size', -1)) == self.hidden_size and int(
                        d.get('num_hidden_layers',
                              -1)) == self.num_real_layers:
                    return cfg_file
            except Exception:
                pass
            # Fallback: use the template even if dimensions don't match
            # because _write_hf_artifacts will override key fields anyway
            return cfg_file
        raise FileNotFoundError(f'找不到可用的 HF config 模板: {cfg_file}')

    def _write_hf_artifacts(self) -> None:
        models_dir = self._repo_models_dir()
        os.makedirs(self.hf_save_dir, exist_ok=True)

        cfg_path = self._select_config_template()
        with open(cfg_path) as f:
            cfg = json.load(f)

        cfg['hidden_size'] = self.hidden_size
        cfg['num_hidden_layers'] = self.num_real_layers
        cfg['num_attention_heads'] = self.num_attention_heads
        cfg['num_key_value_heads'] = self.num_query_groups
        cfg['kv_channels'] = self.qk_head_dim
        cfg['qk_layernorm'] = self.qk_layernorm
        cfg['rope_theta'] = float(self.rotary_base)
        cfg['ep_size'] = self.ep_size
        cfg['first_k_dense_replace'] = self.first_k_dense_replace
        cfg['tie_word_embeddings'] = self.tie_word_embeddings
        if self.max_position_embeddings is not None:
            cfg['max_position_embeddings'] = self.max_position_embeddings
        if 'n_routed_experts' in cfg:
            cfg['n_routed_experts'] = self.num_experts
        if 'num_experts' in cfg:
            cfg['num_experts'] = self.num_experts

        if self.vocab_size is not None:
            cfg['vocab_size'] = self.vocab_size
        if self.ffn_hidden_size is not None:
            cfg['intermediate_size'] = self.ffn_hidden_size
        if self.moe_ffn_hidden_size is not None:
            cfg['moe_intermediate_size'] = self.moe_ffn_hidden_size
        cfg['n_shared_experts'] = self.n_shared_experts
        cfg['num_experts_per_tok'] = self.moe_router_topk

        with open(os.path.join(self.hf_save_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        gen_cfg = os.path.join(models_dir, 'generation_config.json')
        if os.path.isfile(gen_cfg):
            shutil.copyfile(
                gen_cfg,
                os.path.join(self.hf_save_dir, 'generation_config.json'))

        for fn in [
                'configuration_deepseek.py', 'modeling_deepseek.py',
                '__init__.py'
        ]:
            src = os.path.join(models_dir, fn)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(self.hf_save_dir, fn))

    def _validate(self) -> None:
        if not os.path.isdir(self.mg_load_dir):
            raise FileNotFoundError(f'加载目录不存在: {self.mg_load_dir}')
        if self.num_layers <= 0:
            raise ValueError('num_layers 必须 > 0')
        if self.tp_size <= 0 or self.pp_size <= 0 or self.ep_size <= 0:
            raise ValueError('并行度必须 > 0')
        if self.num_experts % self.ep_size != 0:
            raise ValueError('num_experts 必须能整除 ep_size')
        if self.dualpipe and self.vpp_size is None:
            raise ValueError('dualpipev 需要 vpp checkpoint (model0/model1)')
        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f'num_attention_heads ({self.num_attention_heads}) 必须能整除 tp_size ({self.tp_size})'
            )

    def _detect_vpp(self) -> tuple[int | None, list[str] | None]:
        ckpt_path = self._resolve_rank_ckpt_path(0, 0, None)
        if self.verbose:
            logger.info('Detecting vpp from: %s', ckpt_path)
        state = _torch_load_compat(ckpt_path, disable_mmap=self.disable_mmap)
        model_keys = sorted([
            k for k in state.keys() if k.startswith('model') and k != 'model'
        ])
        if 'model0' in state and 'model1' in state:
            return len(model_keys), model_keys
        return None, None

    def _build_rank_dir_map(self) -> None:
        mp_dirs: list[str] = []
        try:
            for d in os.listdir(self.iter_dir):
                if d.startswith('mp_rank_'):
                    mp_dirs.append(d)
        except FileNotFoundError:
            mp_dirs = []
        rank_map: dict[tuple[int, int], set[int]] = defaultdict(set)
        for d in mp_dirs:
            parts = d.split('_')
            if len(parts) < 3:
                continue
            try:
                tp = int(parts[2])
            except ValueError:
                continue
            idxs = []
            for p in parts[3:]:
                try:
                    idxs.append(int(p))
                except ValueError:
                    idxs.append(None)
            # Positional parsing: _mp_prefix always formats as
            # mp_rank_{tp:02}_{pp:03}_{ep:03} (pp before ep)
            pp = None
            ep = None
            if self.pp_size > 1 and self.ep_size > 1 and len(idxs) >= 2:
                pp = idxs[0]
                ep = idxs[1]
            elif self.pp_size > 1 and self.ep_size == 1:
                pp = idxs[0] if idxs else None
                ep = 0
            elif self.pp_size == 1 and self.ep_size > 1:
                pp = 0
                ep = idxs[0] if idxs else None
            else:
                pp = 0
                ep = 0
            if pp is None or ep is None:
                continue
            rank_map[(tp, pp)].add(ep)
        self._rank_dir_map = {k: sorted(list(v)) for k, v in rank_map.items()}

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

        real_layers = list(range(self.num_real_layers))
        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                real_layers.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]
            for local_idx, hf_layer in enumerate(
                    self.pprank_layer_idxs[pp_rank]):
                self.layer2loc[hf_layer] = (pp_rank, local_idx)

    def _build_vpprank_layer_map(self) -> None:
        if self.dualpipe:
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
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer)
                    else:
                        before = 0
                        for n in noop_list:
                            if n < layer:
                                before += 1
                            else:
                                break
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer - before)

                if (idx + 1) % self.vpp_stage == 0:
                    vpp_rank += 1
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0
        else:
            layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                               for _ in range(self.pp_size)]
            if self.noop_layers_list:
                layers_per_pp = self.num_layers // self.pp_size
                for layer in self.noop_layers_list:
                    pp_idx = layer // layers_per_pp
                    vpp_idx = (layer % layers_per_pp) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            real_layers = list(range(self.num_real_layers))
            # Megatron 标准 VPP 采用 PP-first 分配：每个物理 PP rank 的 VPP stages 连续
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                        real_layers.pop(0)
                        for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                for local_idx, hf_layer in enumerate(
                        self.vpprank_layer_idxs[pp_rank][vpp_rank]):
                    self.layer2loc_vpp[hf_layer] = (pp_rank, vpp_rank,
                                                    local_idx)

    def _load_rank_state(self, tp_rank: int, pp_rank: int, ep_rank: int | None,
                         vpp_rank: int | None) -> dict[str, torch.Tensor]:
        ckpt_path = self._resolve_rank_ckpt_path(tp_rank, pp_rank, ep_rank)
        t0 = time.perf_counter()
        if self.verbose and self.log_rank_load:
            logger.info('Loading rank ckpt: tp=%d pp=%d ep=%s vpp=%s path=%s',
                        tp_rank, pp_rank, ep_rank, vpp_rank, ckpt_path)
        state = _torch_load_compat(ckpt_path, disable_mmap=self.disable_mmap)
        if self.verbose and self.log_rank_load:
            dt = time.perf_counter() - t0
            logger.info('Loaded rank ckpt: tp=%d pp=%d ep=%s vpp=%s in %.2fs',
                        tp_rank, pp_rank, ep_rank, vpp_rank, dt)
        if vpp_rank is None:
            return state['model']
        return state[f'model{vpp_rank}']

    def _resolve_rank_ckpt_path(self, tp_rank: int, pp_rank: int,
                                ep_rank: int | None) -> str:
        candidates: list[str] = []
        if ep_rank is not None:
            candidates.append(
                _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                           self.pp_size, self.ep_size))
            candidates.append(
                f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}')
            candidates.append(
                f'mp_rank_{tp_rank:02}_{ep_rank:03}_{pp_rank:03}')
            candidates.append(f'mp_rank_{tp_rank:02}_{ep_rank:03}')
        else:
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}_000')
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}_001')
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}')
            candidates.append(f'mp_rank_{tp_rank:02}')
        for p in candidates:
            path = os.path.join(self.iter_dir, p, 'model_optim_rng.pt')
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f'无法定位 rank 文件: tp={tp_rank}, pp={pp_rank}, ep={ep_rank}, iter_dir={self.iter_dir}'
        )

    def _load_models_for_stage(
        self, pp_rank: int, vpp_rank: int | None
    ) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
        models: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        if self.verbose:
            logger.info('Loading models for stage: pp_rank=%d vpp_rank=%s',
                        pp_rank, vpp_rank)

        def load_one(
            tp_rank: int
        ) -> tuple[int, int | None, dict[str, torch.Tensor]
                   | None]:
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if not eps:
                return tp_rank, None, None
            base_ep = eps[0]
            st = self._load_rank_state(tp_rank, pp_rank, base_ep, vpp_rank)
            return tp_rank, base_ep, st

        max_workers = min(self.io_threads, self.tp_size)
        if max_workers <= 1:
            for tp_rank in range(self.tp_size):
                tp, base_ep, st = load_one(tp_rank)
                if st is None or base_ep is None:
                    continue
                models[(tp, base_ep)] = st
                models[(tp, 0)] = st
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(load_one, tp) for tp in range(self.tp_size)
                ]
                for fut in as_completed(futures):
                    tp, base_ep, st = fut.result()
                    if st is None or base_ep is None:
                        continue
                    models[(tp, base_ep)] = st
                    models[(tp, 0)] = st

        if self.verbose:
            logger.info(
                'Loaded models for stage: pp_rank=%d vpp_rank=%s keys=%d',
                pp_rank, vpp_rank, len(models))
        return models

    def _tp_ranks_for_ep(self, pp_rank: int, ep_rank: int) -> list[int]:
        tps: list[int] = []
        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if ep_rank in eps:
                tps.append(tp_rank)
        return tps

    def _load_sparse_ep_state(self, tp_rank: int, pp_rank: int, ep_rank: int,
                              vpp_rank: int | None) -> dict[str, torch.Tensor]:
        key = (int(tp_rank), int(pp_rank), int(ep_rank),
               None if vpp_rank is None else int(vpp_rank))
        cached = self._sparse_cache.get(key)
        if cached is not None:
            return cached
        state = self._load_rank_state(tp_rank, pp_rank, ep_rank, vpp_rank)
        keep: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if '.mlp.experts.' in k or '.mlp.router.' in k:
                keep[k] = v
        self._sparse_cache[key] = keep
        return keep

    def _reconstruct_router_lazy(self, base_models: dict[tuple[int, int],
                                                         dict[str,
                                                              torch.Tensor]],
                                 pp_rank: int, vpp_rank: int | None,
                                 key: str) -> torch.Tensor:
        for tp_rank in range(self.tp_size):
            d = base_models.get((tp_rank, 0))
            if d is None:
                continue
            t = d.get(key)
            if t is not None and t.shape[0] == self.num_experts:
                return t.clone()

        sample = None
        for tp_rank in range(self.tp_size):
            d = base_models.get((tp_rank, 0))
            if d is None:
                continue
            if key in d:
                sample = d[key]
                break
        if sample is None:
            tps = self._tp_ranks_for_ep(pp_rank, 0)
            if not tps:
                raise ValueError(f'找不到 ep=0 的权重: pp={pp_rank} key={key}')
            d0 = self._load_sparse_ep_state(tps[0], pp_rank, 0, vpp_rank)
            if key not in d0:
                raise ValueError(f'找不到 router 权重: pp={pp_rank} key={key}')
            sample = d0[key]
            del d0

        out = torch.empty((self.num_experts, ) + sample.shape[1:],
                          dtype=sample.dtype)
        num_local = self.num_experts // self.ep_size

        for ep in range(self.ep_size):
            owners = self._tp_ranks_for_ep(pp_rank, ep)
            if not owners:
                raise ValueError(f'找不到 ep={ep} 的权重目录: pp={pp_rank} key={key}')
            # With expert_tp_size < tp_size, multiple TP ranks hold identical
            # router weights for the same EP rank.  Just pick the first owner.
            tp = owners[0]
            d = self._load_sparse_ep_state(tp, pp_rank, ep, vpp_rank)
            if key in d:
                part = d[key]
            else:
                d_full = self._load_rank_state(tp, pp_rank, ep, vpp_rank)
                if key not in d_full:
                    raise ValueError(
                        f'找不到 router 权重: pp={pp_rank} ep={ep} key={key}')
                part = d_full[key]
                del d_full
            if part.shape[0] == self.num_experts:
                out = part.clone()
                del d
                gc.collect()
                break
            if part.shape[0] != num_local:
                raise ValueError(
                    f'router 分片形状异常: pp={pp_rank} ep={ep} key={key} shape={part.shape}'
                )
            out[ep * num_local:(ep + 1) * num_local] = part
            del d
            gc.collect()
        return out

    def _gather_tp_row(self,
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       key: str,
                       ep_rank: int = 0) -> torch.Tensor:
        parts = [
            models[(tp_rank, ep_rank)].pop(key)
            for tp_rank in range(self.tp_size)
        ]
        return torch.cat(parts, dim=0)

    def _gather_tp_col(self,
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       key: str,
                       ep_rank: int = 0) -> torch.Tensor:
        parts = [
            models[(tp_rank, ep_rank)].pop(key)
            for tp_rank in range(self.tp_size)
        ]
        return torch.cat(parts, dim=1)

    def _set_preprocess(
            self, hf: dict[str, torch.Tensor],
            models: dict[tuple[int, int], dict[str, torch.Tensor]]) -> None:
        parts = [
            models[(tp_rank, 0)].pop('embedding.word_embeddings.weight')
            for tp_rank in range(self.tp_size)
        ]
        hf['model.embed_tokens.weight'] = torch.cat(parts, dim=0)

    def _set_postprocess(
            self, hf: dict[str, torch.Tensor],
            models: dict[tuple[int, int], dict[str, torch.Tensor]]) -> None:
        hf['model.norm.weight'] = models[(
            0, 0)].pop('decoder.final_layernorm.weight')
        parts = [
            models[(tp_rank, 0)].pop('output_layer.weight')
            for tp_rank in range(self.tp_size)
        ]
        if not self.tie_word_embeddings:
            hf['lm_head.weight'] = torch.cat(parts, dim=0)

    def _set_layer_norm(self, hf: dict[str, torch.Tensor],
                        models: dict[tuple[int, int], dict[str, torch.Tensor]],
                        hf_layer: int, local_idx: int) -> None:
        in_key = f'decoder.layers.{local_idx}.input_layernorm.weight'
        mlp_key = f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'
        hf[f'model.layers.{hf_layer}.input_layernorm.weight'] = models[(
            0, 0)].pop(in_key)
        hf[f'model.layers.{hf_layer}.post_attention_layernorm.weight'] = models[
            (0, 0)].pop(mlp_key)

    def _set_layer_attn(self, hf: dict[str, torch.Tensor],
                        models: dict[tuple[int, int], dict[str, torch.Tensor]],
                        hf_layer: int, local_idx: int) -> None:
        prefix = f'decoder.layers.{local_idx}.self_attention'
        qkv_key = f'{prefix}.linear_qkv.weight'
        proj_key = f'{prefix}.linear_proj.weight'
        q_norm_key = f'{prefix}.q_layernorm.weight'
        k_norm_key = f'{prefix}.k_layernorm.weight'

        linear_proj_list: list[torch.Tensor] = []
        q_parts: list[torch.Tensor] = []
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []

        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f'num_attention_heads={self.num_attention_heads} 不能整除 tp_size={self.tp_size}'
            )
        # GQA: Q/K share head_dim for RoPE compatibility
        q_head_dim = self._resolved_q_head_dim or self.qk_head_dim
        kv_heads_per_tp_fixed = self._resolved_kv_heads_per_tp

        for tp_rank in range(self.tp_size):
            linear_proj_list.append(models[(tp_rank, 0)].pop(proj_key))
            qkv_shard = models[(tp_rank, 0)].pop(qkv_key)
            heads_per_tp = self.num_attention_heads // self.tp_size

            if self._resolved_q_head_dim is None:
                q_per_tp_try = heads_per_tp * q_head_dim
                rem_try = qkv_shard.shape[0] - q_per_tp_try
                denom_try = q_head_dim + self.v_head_dim
                bad = rem_try < 0 or denom_try <= 0 or rem_try % denom_try != 0
                if bad:
                    inferred = self._infer_qkv_layout(int(qkv_shard.shape[0]))
                    if inferred is None:
                        raise ValueError(
                            f'{qkv_key} 分片形状异常: {qkv_shard.shape}, q_per_tp={q_per_tp_try}'
                        )
                    inferred_q_head_dim, inferred_kv_heads_per_tp = inferred
                    # GQA: 如果推断的 head_dim 与配置不同，更新 rope 计算
                    if inferred_q_head_dim != self.qk_head_dim:
                        self._head_dim_for_rope = int(inferred_q_head_dim)
                        self._reset_inv_freq()
                    self._resolved_q_head_dim = int(inferred_q_head_dim)
                    self._resolved_kv_heads_per_tp = int(
                        inferred_kv_heads_per_tp)
                    q_head_dim = self._resolved_q_head_dim
                    kv_heads_per_tp_fixed = self._resolved_kv_heads_per_tp
                    if self.verbose:
                        logger.info(
                            'Auto-detected qkv layout: q_head_dim=%d kv_heads_per_tp=%d (from shard_rows=%d)',
                            q_head_dim,
                            kv_heads_per_tp_fixed,
                            int(qkv_shard.shape[0]),
                        )

            # GQA: 标准 Q/K/V 分割
            q_per_tp = heads_per_tp * q_head_dim
            rem = qkv_shard.shape[0] - q_per_tp
            if rem < 0:
                raise ValueError(
                    f'{qkv_key} 分片形状异常: {qkv_shard.shape}, q_per_tp={q_per_tp}'
                )
            # GQA: K 和 V 使用相同的 head_dim
            denom = q_head_dim + self.v_head_dim
            if denom <= 0 or rem % denom != 0:
                raise ValueError(
                    f'{qkv_key} 分片剩余维度异常: rem={rem}, q_head_dim={q_head_dim}, v_head_dim={self.v_head_dim}'
                )
            kv_heads_per_tp = kv_heads_per_tp_fixed or (rem // denom)
            if self.detected_num_kv_heads is None:
                self.detected_num_kv_heads = kv_heads_per_tp * self.tp_size
                if self.detected_num_kv_heads != self.num_query_groups:
                    raise ValueError(
                        f'Detected num_kv_heads={self.detected_num_kv_heads} from checkpoint '
                        f'does not match --num-query-groups={self.num_query_groups}. '
                        f'Check your training config.')
            # GQA: K uses q_head_dim for RoPE compatibility
            k_per_tp = kv_heads_per_tp * q_head_dim
            v_per_tp = kv_heads_per_tp * self.v_head_dim
            q_r, k_r, v_r = torch.split(qkv_shard,
                                        [q_per_tp, k_per_tp, v_per_tp],
                                        dim=0)
            q_parts.append(q_r)
            k_parts.append(k_r)
            v_parts.append(v_r)

        o_proj = torch.cat(linear_proj_list, dim=1)
        hf[f'model.layers.{hf_layer}.self_attn.q_proj.weight'] = torch.cat(
            q_parts, dim=0)
        hf[f'model.layers.{hf_layer}.self_attn.k_proj.weight'] = torch.cat(
            k_parts, dim=0)
        hf[f'model.layers.{hf_layer}.self_attn.v_proj.weight'] = torch.cat(
            v_parts, dim=0)
        hf[f'model.layers.{hf_layer}.self_attn.o_proj.weight'] = o_proj
        q_ln = models[(0, 0)].pop(q_norm_key, None)
        if q_ln is not None:
            hf[f'model.layers.{hf_layer}.self_attn.q_layernorm.weight'] = q_ln
        k_ln = models[(0, 0)].pop(k_norm_key, None)
        if k_ln is not None:
            hf[f'model.layers.{hf_layer}.self_attn.k_layernorm.weight'] = k_ln
        # Note: inv_freq is registered with persistent=False in the HF model,
        # so it is recomputed at load time. Written here for verification
        # tooling (verify_mcore2hf.py) compatibility.
        hf[f'model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq'] = self.inv_freq.clone(
        )
        return

    def _set_layer_mlp(self, hf: dict[str, torch.Tensor],
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       hf_layer: int, local_idx: int) -> None:
        prefix = f'decoder.layers.{local_idx}.mlp'
        if hf_layer < self.first_k_dense_replace:
            fc1 = self._gather_tp_row(models, f'{prefix}.linear_fc1.weight')
            fc2 = self._gather_tp_col(models, f'{prefix}.linear_fc2.weight')
            gate, up = torch.chunk(fc1, 2, dim=0)
            hf[f'model.layers.{hf_layer}.mlp.gate_proj.weight'] = gate
            hf[f'model.layers.{hf_layer}.mlp.up_proj.weight'] = up
            hf[f'model.layers.{hf_layer}.mlp.down_proj.weight'] = fc2
            return

        pp_rank = self.layer2loc[hf_layer][
            0] if self.vpp_size is None else self.layer2loc_vpp[hf_layer][0]
        vpp_rank = None if self.vpp_size is None else self.layer2loc_vpp[
            hf_layer][1]
        router = self._reconstruct_router_lazy(models, pp_rank, vpp_rank,
                                               f'{prefix}.router.weight')
        hf[f'model.layers.{hf_layer}.mlp.gate.weight'] = router

        try:
            router_bias = self._reconstruct_router_lazy(
                models, pp_rank, vpp_rank, f'{prefix}.router.expert_bias')
            # 只写规范键名 e_score_correction_bias，避免与 gate.bias 同时存在
            # 导致 hf2mcore 中 _assert_consumed 报"未消费权重"错误。
            hf[f'model.layers.{hf_layer}.mlp.gate.e_score_correction_bias'] = router_bias
        except ValueError:
            pass

        shared_fc1 = self._gather_tp_row(
            models, f'{prefix}.shared_experts.linear_fc1.weight')
        shared_fc2 = self._gather_tp_col(
            models, f'{prefix}.shared_experts.linear_fc2.weight')
        shared_gate, shared_up = torch.chunk(shared_fc1, 2, dim=0)
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.gate_proj.weight'] = shared_gate
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.up_proj.weight'] = shared_up
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.down_proj.weight'] = shared_fc2

        if self.moe_grouped_gemm:
            w1_key = f'{prefix}.experts.weight1'
            w2_key = f'{prefix}.experts.weight2'
            num_local = self.num_experts // self.ep_size
            # expert_tp_size controls how many TP ranks hold unique shards
            # of the expert weight.  expert_tp_size=1 → replicated (need 1
            # owner); expert_tp_size=tp_size → fully sharded (need all).
            num_expert_tp_owners = max(1, self.expert_tp_size)
            for ep_rank in range(self.ep_size):
                owners = self._tp_ranks_for_ep(pp_rank, ep_rank)
                if not owners:
                    raise ValueError(f'找不到 ep={ep_rank} 的权重目录: pp={pp_rank}')
                # Only load from unique TP owners that hold different shards.
                # When expert_tp_size=1, all owners hold identical data, so
                # we only need the first one.
                unique_owners = owners[:num_expert_tp_owners]
                shards_w1: list[torch.Tensor] = []
                shards_w2: list[torch.Tensor] = []
                for tp_rank in unique_owners:
                    d = self._load_sparse_ep_state(tp_rank, pp_rank, ep_rank,
                                                   vpp_rank)
                    if w1_key not in d or w2_key not in d:
                        raise ValueError(
                            f'找不到 moe 权重: pp={pp_rank} ep={ep_rank} tp={tp_rank}'
                        )
                    shards_w1.append(d[w1_key])
                    shards_w2.append(d[w2_key])
                    del d
                    gc.collect()

                local_w1 = shards_w1[0] if len(shards_w1) == 1 else torch.cat(
                    shards_w1, dim=1)
                local_w2 = shards_w2[0] if len(shards_w2) == 1 else torch.cat(
                    shards_w2, dim=0)
                # Expert-first layout: weight1 is stored as
                # [hidden_size, num_local * intermediate*2], where each
                # expert's rows are contiguous.  Reshape directly to
                # [num_local, hidden_size, intermediate*2].
                w1_3d = local_w1.reshape(num_local, self.hidden_size,
                                         -1).contiguous()
                w2_3d = local_w2.view(num_local, -1, self.hidden_size)
                for li in range(num_local):
                    expert = ep_rank * num_local + li
                    fc1 = w1_3d[li].t()
                    if self.expert_tp_size > 1 and len(shards_w1) > 1:
                        # gate/up are interleaved per expert_tp shard;
                        # de-interleave them.
                        chunks = torch.chunk(fc1, self.expert_tp_size, dim=0)
                        gate_list, up_list = [], []
                        for chunk in chunks:
                            g, u = torch.chunk(chunk, 2, dim=0)
                            gate_list.append(g)
                            up_list.append(u)
                        gate = torch.cat(gate_list, dim=0)
                        up = torch.cat(up_list, dim=0)
                    else:
                        gate, up = torch.chunk(fc1, 2, dim=0)
                    down = w2_3d[li].t()
                    hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate
                    hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up
                    hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = down
                # Evict processed EP rank from sparse cache
                for tp_rank in owners:
                    cache_key = (int(tp_rank), int(pp_rank), int(ep_rank),
                                 None if vpp_rank is None else int(vpp_rank))
                    self._sparse_cache.pop(cache_key, None)
        else:
            num_local = self.num_experts // self.ep_size
            # expert_tp_size controls how many TP ranks hold unique shards
            # of the expert weight.  expert_tp_size=1 → replicated (need 1
            # owner); expert_tp_size=tp_size → fully sharded (need all).
            num_expert_tp_owners = max(1, self.expert_tp_size)
            for ep_rank in range(self.ep_size):
                owners = self._tp_ranks_for_ep(pp_rank, ep_rank)
                if not owners:
                    raise ValueError(f'找不到 ep={ep_rank} 的权重目录: pp={pp_rank}')
                # Only load from unique TP owners that hold different shards.
                unique_owners = owners[:num_expert_tp_owners]
                local_states: dict[int, dict[str, torch.Tensor]] = {}
                for tp_rank in unique_owners:
                    local_states[tp_rank] = self._load_sparse_ep_state(
                        tp_rank, pp_rank, ep_rank, vpp_rank)

                for li in range(num_local):
                    expert = ep_rank * num_local + li
                    local_prefix = f'{prefix}.experts.local_experts.{li}'
                    fc1_key = f'{local_prefix}.linear_fc1.weight'
                    fc2_key = f'{local_prefix}.linear_fc2.weight'

                    if len(unique_owners) == 1:
                        tp_rank = unique_owners[0]
                        fc1 = local_states[tp_rank].pop(fc1_key)
                        fc2 = local_states[tp_rank].pop(fc2_key)
                        gate, up = torch.chunk(fc1, 2, dim=0)
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = fc2
                    else:
                        fc1_parts = [
                            local_states[tp].pop(fc1_key)
                            for tp in unique_owners
                        ]
                        fc2_parts = [
                            local_states[tp].pop(fc2_key)
                            for tp in unique_owners
                        ]
                        fc1 = torch.cat(fc1_parts, dim=0)
                        fc2 = torch.cat(fc2_parts, dim=1)
                        if self.expert_tp_size > 1:
                            # gate/up are interleaved per expert_tp
                            # shard; de-interleave them.
                            chunks = torch.chunk(fc1,
                                                 self.expert_tp_size,
                                                 dim=0)
                            gate_list, up_list = [], []
                            for chunk in chunks:
                                g, u = torch.chunk(chunk, 2, dim=0)
                                gate_list.append(g)
                                up_list.append(u)
                            gate = torch.cat(gate_list, dim=0)
                            up = torch.cat(up_list, dim=0)
                        else:
                            gate, up = torch.chunk(fc1, 2, dim=0)
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = fc2

                for _, st in local_states.items():
                    del st
                # Evict processed EP rank from sparse cache
                for tp_rank in owners:
                    cache_key = (int(tp_rank), int(pp_rank), int(ep_rank),
                                 None if vpp_rank is None else int(vpp_rank))
                    self._sparse_cache.pop(cache_key, None)
                gc.collect()

    def _maybe_cast(self, t: torch.Tensor) -> torch.Tensor:
        if self._target_dtype is None:
            return t
        if not torch.is_floating_point(t):
            return t
        if t.dtype == self._target_dtype:
            return t
        return t.to(dtype=self._target_dtype)

    def _save_shard(self, tensors: dict[str, torch.Tensor], shard_idx: int,
                    total_shards: int) -> None:
        name = f'model-{shard_idx:05d}-of-{total_shards:06d}.safetensors'
        path = os.path.join(self.hf_save_dir, name)
        packed: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            v = self._maybe_cast(v)
            if not v.is_contiguous():
                packed[k] = v.contiguous()
            else:
                packed[k] = v
        safetensors.torch.save_file(packed, path, metadata={'format': 'pt'})
        if self.verbose:
            logger.info('Saved shard %d/%d: %s (tensors=%d)', shard_idx,
                        total_shards, path, len(tensors))

    def _convert_one_stage(self, pp_rank: int, vpp_rank: int | None,
                           total_shards: int) -> None:
        self._sparse_cache = {}

        models = self._load_models_for_stage(pp_rank=pp_rank,
                                             vpp_rank=vpp_rank)

        if pp_rank == 0 and vpp_rank in (None, 0):
            base_tensors: dict[str, torch.Tensor] = {}
            self._set_preprocess(base_tensors, models)
            self._save_shard(base_tensors, 1, total_shards)

        if self.vpp_size is None:
            layers = self.pprank_layer_idxs[pp_rank]
            for hf_layer in layers:
                local_idx = self.layer2loc[hf_layer][1]
                layer_tensors: dict[str, torch.Tensor] = {}
                self._set_layer_norm(layer_tensors, models, hf_layer,
                                     local_idx)
                self._set_layer_attn(layer_tensors, models, hf_layer,
                                     local_idx)
                self._set_layer_mlp(layer_tensors, models, hf_layer, local_idx)
                self._save_shard(layer_tensors, hf_layer + 2, total_shards)

            if pp_rank == self.pp_size - 1:
                tail_tensors: dict[str, torch.Tensor] = {}
                self._set_postprocess(tail_tensors, models)
                self._save_shard(tail_tensors, total_shards, total_shards)
        else:
            layers = self.vpprank_layer_idxs.get(pp_rank, {}).get(vpp_rank, [])
            for hf_layer in layers:
                local_idx = self.layer2loc_vpp[hf_layer][2]
                layer_tensors: dict[str, torch.Tensor] = {}
                self._set_layer_norm(layer_tensors, models, hf_layer,
                                     local_idx)
                self._set_layer_attn(layer_tensors, models, hf_layer,
                                     local_idx)
                self._set_layer_mlp(layer_tensors, models, hf_layer, local_idx)
                self._save_shard(layer_tensors, hf_layer + 2, total_shards)

            is_post = False
            if self.dualpipe:
                if pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                    is_post = True
            else:
                if pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
                    is_post = True
            if is_post:
                tail_tensors: dict[str, torch.Tensor] = {}
                self._set_postprocess(tail_tensors, models)
                self._save_shard(tail_tensors, total_shards, total_shards)

        del models
        gc.collect()

    def run(self) -> None:
        total_shards = self.num_real_layers + 2

        if self.verbose:
            logger.info(
                'Start converting: layers=%d real_layers=%d tp=%d pp=%d ep=%d vpp=%s dualpipe=%s save_dir=%s',
                self.num_layers, self.num_real_layers, self.tp_size,
                self.pp_size, self.ep_size, self.vpp_size, self.dualpipe,
                self.hf_save_dir)

        stages: list[tuple[int, int | None]] = []
        if self.vpp_size is None:
            stages = [(pp, None) for pp in range(self.pp_size)]
        else:
            for pp in range(self.pp_size):
                for vpp in range(self.vpp_size):
                    stages.append((pp, vpp))

        workers = int(os.environ.get('CKPT_CONVERT_PP_WORKERS', '1'))
        workers = max(1, min(workers, len(stages)))
        if workers <= 1:
            for pp_rank, vpp_rank in stages:
                self._convert_one_stage(pp_rank, vpp_rank, total_shards)
        else:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context

            ctx = get_context('spawn')
            cfg = dict(
                mg_load_dir=self.mg_load_dir,
                hf_save_dir=self.hf_save_dir,
                num_layers=self.num_layers,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                ep_size=self.ep_size,
                first_k_dense_replace=self.first_k_dense_replace,
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                num_attention_heads=self.num_attention_heads,
                num_query_groups=self.num_query_groups,
                qk_head_dim=self.qk_head_dim,
                moe_grouped_gemm=self.moe_grouped_gemm,
                schedules_method=self.schedules_method,
                vpp_stage=self.vpp_stage,
                num_layer_list=self.num_layer_list,
                noop_layers=','.join(map(str, self.noop_layers_list)),
                qk_layernorm=self.qk_layernorm,
                rotary_base=float(self.rotary_base),
                vocab_size=self.vocab_size,
                max_position_embeddings=self.max_position_embeddings,
                tie_word_embeddings=self.tie_word_embeddings,
                ffn_hidden_size=self.ffn_hidden_size,
                moe_ffn_hidden_size=self.moe_ffn_hidden_size,
                n_shared_experts=self.n_shared_experts,
                moe_router_topk=self.moe_router_topk,
                hf_config_template=self.hf_config_template,
                cast_dtype=self.cast_dtype,
                io_threads=1,
                disable_mmap=self.disable_mmap,
            )
            t0 = time.time()
            logger.info('Parallel convert: pp_workers=%d stages=%d',
                        int(workers), int(len(stages)))
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=ctx) as ex:
                futures = [
                    ex.submit(_worker_convert_stage, cfg, pp, vpp,
                              total_shards) for pp, vpp in stages
                ]
                for fut in as_completed(futures):
                    fut.result()
            logger.info('Parallel convert done (%.2fs)', time.time() - t0)

        self._write_hf_artifacts()
        _write_hf_index_by_scan(self.hf_save_dir)


def _worker_convert_stage(cfg: dict, pp_rank: int, vpp_rank: int | None,
                          total_shards: int) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    conv = MgCkptConvert(**cfg)
    conv._convert_one_stage(int(pp_rank), vpp_rank, int(total_shards))


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
                        '--vpp-stage',
                        dest='num_layers_per_virtual_pipeline_stage',
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument('--noop-layers',
                        type=str,
                        default='',
                        help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers',
                        type=int,
                        default=0,
                        help='Multi-Token prediction layer num')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument(
        '--qk-layernorm',
        action='store_true',
        help='Enable QK LayerNorm (must match training config)')
    parser.add_argument(
        '--schedules-method',
        type=str,
        default=None,
        choices=['dualpipev'],
        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--pp-workers',
                        type=int,
                        default=None,
                        help='Override CKPT_CONVERT_PP_WORKERS in-process.')
    parser.add_argument('--io-threads',
                        type=int,
                        default=4,
                        help='Thread workers for loading rank ckpt files.')
    parser.add_argument('--disable-mmap',
                        action='store_true',
                        help='Disable torch.load mmap when supported.')
    parser.add_argument('--cast-dtype',
                        type=str,
                        default=None,
                        choices=['fp32', 'bf16', 'fp16'],
                        help='Cast floating tensors before saving.')
    parser.add_argument('--sha256-manifest',
                        type=str,
                        default=None,
                        help='Write sha256 manifest json to this path.')
    parser.add_argument('--num-layers',
                        type=int,
                        default=None,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=2,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--rotary-base',
                        type=float,
                        default=50000.0,
                        help='Rotary base for RoPE')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=None,
                        help='Override hidden size.')
    parser.add_argument('--num-experts',
                        type=int,
                        default=None,
                        help='Override num experts.')
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=None,
                        help='Override attention heads.')
    parser.add_argument('--num-query-groups',
                        type=int,
                        default=None,
                        help='Number of query groups for GQA')
    parser.add_argument('--qk-head-dim',
                        type=int,
                        default=128,
                        help='Q/K head dimension (kv-channels).')
    parser.add_argument(
        '--v-head-dim',
        type=int,
        default=None,
        help='V head dimension (default: same as qk-head-dim).')
    parser.add_argument('--max-position-embeddings',
                        type=int,
                        default=None,
                        help='Override max position embeddings.')
    parser.add_argument('--tie-word-embeddings',
                        action='store_true',
                        help='Tie word embeddings and output layer.')
    parser.add_argument('--vocab-size',
                        type=int,
                        default=None,
                        help='Override vocab size.')
    parser.add_argument('--ffn-hidden-size',
                        type=int,
                        default=None,
                        help='Override dense FFN intermediate size.')
    parser.add_argument('--moe-ffn-hidden-size',
                        type=int,
                        default=None,
                        help='Override MoE expert intermediate size.')
    parser.add_argument('--n-shared-experts',
                        type=int,
                        default=1,
                        help='Number of shared experts (default: 1)')
    parser.add_argument('--moe-router-topk',
                        type=int,
                        default=2,
                        help='MoE router top-k (default: 2)')
    parser.add_argument('--expert-tensor-parallel-size',
                        type=int,
                        default=1,
                        help='Expert tensor parallel size (default: 1, '
                        'experts not split by TP).')

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    logger.info('Arguments: %s', args)
    if args.pp_workers is not None:
        os.environ['CKPT_CONVERT_PP_WORKERS'] = str(int(args.pp_workers))

    if args.num_layers is None:
        raise ValueError('必须提供 --num-layers')
    if args.hidden_size is None:
        raise ValueError('必须提供 --hidden-size')
    if args.num_attention_heads is None:
        raise ValueError('必须提供 --num-attention-heads')
    if args.num_query_groups is None:
        raise ValueError('必须提供 --num-query-groups')
    if args.num_experts is None:
        raise ValueError('必须提供 --num-experts')

    converter = MgCkptConvert(
        mg_load_dir=args.load_dir,
        hf_save_dir=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.source_tensor_parallel_size,
        pp_size=args.source_pipeline_parallel_size,
        ep_size=args.source_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        qk_head_dim=args.qk_head_dim,
        v_head_dim=args.v_head_dim,
        moe_grouped_gemm=args.moe_grouped_gemm,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        rotary_base=args.rotary_base,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        tie_word_embeddings=args.tie_word_embeddings,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        n_shared_experts=args.n_shared_experts,
        moe_router_topk=args.moe_router_topk,
        cast_dtype=args.cast_dtype,
        io_threads=args.io_threads,
        disable_mmap=args.disable_mmap,
        qk_layernorm=args.qk_layernorm,
        expert_tp_size=args.expert_tensor_parallel_size,
    )
    converter.run()
    _write_sha256_manifest(args.save_dir, args.sha256_manifest)


if __name__ == '__main__':
    main()
