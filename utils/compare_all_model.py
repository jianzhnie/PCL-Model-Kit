#!/usr/bin/env python3
"""
MCore 权重完全对齐对比工具

将两个 MCore checkpoint 的分布式权重（TP/PP/EP/VPP）分别合并为完整权重后，
逐 tensor 对比，输出详细的差异统计。

支持两个 checkpoint 使用不同的并行配置。

用法:
  # 相同并行配置
  python compare_all_model_v2.py \\
    --dir-a /path/to/ckpt_a \\
    --dir-b /path/to/ckpt_b \\
    --tp 2 --pp 8 --ep 32 \\
    --schedules-method dualpipev \\
    --num-layers 32

  # 不同并行配置
  python compare_all_model_v2.py \\
    --dir-a /path/to/ckpt_a --tp-a 2 --pp-a 8 --ep-a 32 \\
    --dir-b /path/to/ckpt_b --tp-b 2 --pp-b 8 --ep-b 1 \\
    --schedules-method dualpipev --num-layers 32
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_mcore_weights_from_ckpt import (CheckpointCache, CheckpointLoader,
                                         LayerMapperFactory,
                                         MoeParallelStrategy, ParallelConfig,
                                         _peek_checkpoint_keys,
                                         _resolve_iter_dir, _torch_load_compat)

# ---------------------------------------------------------------------------
# Tensor Merging
# ---------------------------------------------------------------------------


def _merge_tensor(
    name: str,
    tp_ep_tensors: Dict[Tuple[int, int], torch.Tensor],
    strategy: MoeParallelStrategy,
) -> torch.Tensor:
    """将 TP/EP 分布式 shard 合并为一个完整 tensor。"""

    is_ep = strategy.is_ep_sharded(name)
    tp_dim = strategy.get_tp_parallel_dim(name)

    def _cat_along(tensors: List[torch.Tensor],
                   dim: Optional[int]) -> torch.Tensor:
        if len(tensors) <= 1:
            return tensors[0]
        if dim is not None:
            return torch.cat(tensors, dim=dim)
        return tensors[0]

    if is_ep:
        # 先按 EP 分组，组内合并 TP，再跨 EP 合并
        ep_groups: Dict[int, List[Tuple[int,
                                        torch.Tensor]]] = defaultdict(list)
        for (tp, ep), t in tp_ep_tensors.items():
            ep_groups[ep].append((tp, t))

        ep_merged: List[torch.Tensor] = []
        for ep in sorted(ep_groups):
            sorted_by_tp = [t for _, t in sorted(ep_groups[ep])]
            ep_merged.append(_cat_along(sorted_by_tp, tp_dim))

        if len(ep_merged) <= 1:
            return ep_merged[0]
        if '.experts.weight1' in name:
            return torch.cat(ep_merged, dim=1)
        if '.experts.weight2' in name:
            return torch.cat(ep_merged, dim=0)
        return ep_merged[0]
    else:
        # 非 EP sharded：按 TP 分组，EP 维度取任意副本（它们相同）
        tp_first: Dict[int, torch.Tensor] = {}
        for (tp, _ep), t in tp_ep_tensors.items():
            tp_first.setdefault(tp, t)
        return _cat_along([tp_first[tp] for tp in sorted(tp_first)], tp_dim)


# ---------------------------------------------------------------------------
# Layer Index Conversion
# ---------------------------------------------------------------------------


def _global_layer_name(name: str, pp_rank: int, vpp_rank, loc2layer,
                       num_layers: int, pp_size: int) -> str:
    """将 local layer index 转换为 global index。"""
    m = re.match(r'((?:decoder\.|model\.)?layers\.)(\d+)(.*)', name)
    if not m:
        return name

    prefix, local_idx, suffix = m.group(1), int(m.group(2)), m.group(3)
    layers_per_pp = num_layers // pp_size

    if vpp_rank is not None:
        loc = (pp_rank, vpp_rank, local_idx)
        gid = loc2layer.get(loc, pp_rank * layers_per_pp + local_idx)
    else:
        gid = pp_rank * layers_per_pp + local_idx

    return f'{prefix}{gid}{suffix}'


# ---------------------------------------------------------------------------
# Load & Merge
# ---------------------------------------------------------------------------


def load_merged_state(
    ckpt_dir: str,
    tp: int,
    pp: int,
    ep: int,
    num_layers: int,
    schedules_method: Optional[str] = None,
    vpp_stage: Optional[int] = None,
    io_threads: int = 4,
) -> Dict[str, torch.Tensor]:
    """加载 MCore checkpoint 并合并为完整的 state dict。"""

    parallel = ParallelConfig(
        tp_size=tp,
        pp_size=pp,
        ep_size=ep,
        schedules_method=schedules_method,
        vpp_stage=vpp_stage,
    )
    strategy = MoeParallelStrategy()

    iter_dir = _resolve_iter_dir(ckpt_dir)
    cache = CheckpointCache(max_size=max(8, tp * ep))
    loader = CheckpointLoader(iter_dir, parallel, cache)
    loader.build_rank_map()

    # ---- Detect VPP ----
    if parallel.dualpipe:
        parallel.vpp_size = 2
    else:
        _detect_vpp(loader, parallel, tp, pp)

    if not parallel.vpp_size:
        parallel.vpp_size = 1
    if not parallel.vpp_stage:
        if parallel.dualpipe:
            parallel.vpp_stage = max(1, num_layers // (pp * 2))
        elif parallel.vpp_size > 1:
            parallel.vpp_stage = max(1,
                                     (num_layers // pp) // parallel.vpp_size)
        else:
            parallel.vpp_stage = max(1, num_layers // pp)

    # ---- Build layer mapping ----
    mapper = LayerMapperFactory.create(parallel.dualpipe)
    l2l = mapper.build_mapping(num_layers, pp, parallel.vpp_stage)
    loc2layer = {loc: gid for gid, loc in l2l.items()}

    # ---- Enumerate stages ----
    if parallel.vpp_size > 1:
        stages = [(p, v) for p in range(pp) for v in range(parallel.vpp_size)]
    else:
        stages = [(p, None) for p in range(pp)]

    merged: Dict[str, torch.Tensor] = {}

    for pp_rank, vpp_rank in stages:
        models = loader.load_stage(pp_rank, vpp_rank, io_threads)
        if not models:
            print(f'  警告: PP={pp_rank}, VPP={vpp_rank} 加载为空，跳过')
            continue

        # Collect weight tensors
        wt: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = defaultdict(dict)
        for (tp_r, ep_r), state in models.items():
            for name, tensor in state.items():
                if isinstance(tensor,
                              torch.Tensor) and '_extra_state' not in name:
                    wt[name][(tp_r, ep_r)] = tensor

        # Merge each weight
        for name, tp_ep_t in wt.items():
            gname = _global_layer_name(name, pp_rank, vpp_rank, loc2layer,
                                       num_layers, pp)
            if not (gname.startswith('module.') or gname.startswith('model.')):
                gname = f'module.{gname}'
            if gname in merged:
                print(f'  警告: {gname} 重复，跳过 (PP={pp_rank}, VPP={vpp_rank})')
                continue
            merged[gname] = _merge_tensor(name, tp_ep_t, strategy)

    return merged


def _detect_vpp(loader, parallel, tp, pp):
    """尝试检测 VPP 大小。"""
    try:
        for t in range(tp):
            for p in range(pp):
                ep_ranks = loader._rank_dir_map.get((t, p), [])
                ep = ep_ranks[0] if ep_ranks else None
                try:
                    path = loader.get_ckpt_path(t, p, ep)
                except FileNotFoundError:
                    continue

                keys = _peek_checkpoint_keys(path)
                if keys is None:
                    state = _torch_load_compat(path)
                    keys = list(state.keys())

                mk = [k for k in keys if re.match(r'^model\d+$', k)]
                if len(mk) > 1:
                    parallel.vpp_size = len(mk)
                    return
                return  # 找到一个非 VPP checkpoint 即可确定
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_states(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    label_a: str = 'A',
    label_b: str = 'B',
    atol: float = 1e-6,
) -> bool:
    """对比两个合并后的 state dict，输出详细报告，返回是否完全一致。"""

    keys_a, keys_b = set(state_a), set(state_b)
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    match_n = 0
    exact_n = 0
    val_mismatch: List[dict] = []
    shape_mismatch: List[tuple] = []

    for k in common:
        ta, tb = state_a[k], state_b[k]

        if ta.shape != tb.shape:
            shape_mismatch.append((k, list(ta.shape), list(tb.shape)))
            continue

        if tb.dtype != ta.dtype:
            tb = tb.to(ta.dtype)

        if ta.is_floating_point():
            close = torch.allclose(ta, tb, atol=atol)
            exact = torch.equal(ta, tb)
            if close:
                match_n += 1
                if exact:
                    exact_n += 1
            else:
                diff = (ta.float() - tb.float()).abs()
                val_mismatch.append({
                    'key': k,
                    'max_diff': diff.max().item(),
                    'mean_diff': diff.mean().item(),
                    'shape': list(ta.shape),
                })
        else:
            if torch.equal(ta, tb):
                match_n += 1
                exact_n += 1
            else:
                n = (~torch.eq(ta, tb)).sum().item()
                val_mismatch.append({
                    'key': k,
                    'max_diff': float(n),
                    'mean_diff': 0.0,
                    'shape': list(ta.shape),
                })

    # ---- Report ----
    _print_report(
        label_a,
        label_b,
        state_a,
        state_b,
        keys_a,
        keys_b,
        common,
        only_a,
        only_b,
        shape_mismatch,
        val_mismatch,
        match_n,
        exact_n,
        atol,
    )

    total_issues = len(only_a) + len(only_b) + len(shape_mismatch) + len(
        val_mismatch)
    return total_issues == 0


def _print_report(
    label_a,
    label_b,
    state_a,
    state_b,
    keys_a,
    keys_b,
    common,
    only_a,
    only_b,
    shape_mismatch,
    val_mismatch,
    match_n,
    exact_n,
    atol,
):
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'{label_a} 权重数: {len(keys_a)}')
    print(f'{label_b} 权重数: {len(keys_b)}')
    print(f'共同 key:     {len(common)}')
    print(f'容差 (atol):  {atol}')
    print(sep)

    if only_a:
        print(f'\n仅在 {label_a} 中 ({len(only_a)}):')
        for k in only_a[:30]:
            print(f'  + {k}  {list(state_a[k].shape)}')
        if len(only_a) > 30:
            print(f'  ... 还有 {len(only_a) - 30} 个')

    if only_b:
        print(f'\n仅在 {label_b} 中 ({len(only_b)}):')
        for k in only_b[:30]:
            print(f'  - {k}  {list(state_b[k].shape)}')
        if len(only_b) > 30:
            print(f'  ... 还有 {len(only_b) - 30} 个')

    if shape_mismatch:
        print(f'\nShape 不一致 ({len(shape_mismatch)}):')
        for k, sa, sb in shape_mismatch[:30]:
            print(f'  {k}: {label_a}={sa}, {label_b}={sb}')

    if val_mismatch:
        # 按 max_diff 降序排列（用副本，不修改原始数据）
        sorted_mismatch = sorted(val_mismatch,
                                 key=lambda x: x['max_diff'],
                                 reverse=True)
        print(f'\n数值不一致 ({len(val_mismatch)}):')
        for item in sorted_mismatch[:50]:
            print(f'  {item['key']}: max_diff={item['max_diff']:.2e}, '
                  f'mean_diff={item['mean_diff']:.2e}, shape={item['shape']}')
        if len(val_mismatch) > 50:
            print(f'  ... 还有 {len(val_mismatch) - 50} 个')

    print(f'\n{sep}')
    print(f'匹配: {match_n} (精确 bit-wise: {exact_n}), '
          f'数值不一致: {len(val_mismatch)}, '
          f'Shape 不匹配: {len(shape_mismatch)}, '
          f'仅{label_a}: {len(only_a)}, 仅{label_b}: {len(only_b)}')

    ok = (len(only_a) == 0 and len(only_b) == 0 and len(shape_mismatch) == 0
          and len(val_mismatch) == 0)
    print(f'\n{'完全一致!' if ok else '存在差异'}')
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description='MCore 权重完全对齐对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('--dir-a', required=True, help='Checkpoint A 目录')
    p.add_argument('--dir-b', required=True, help='Checkpoint B 目录')

    g = p.add_argument_group('并行配置（通用，可被单独配置覆盖）')
    g.add_argument('--tp', type=int, default=1, help='Tensor 并行')
    g.add_argument('--pp', type=int, default=1, help='Pipeline 并行')
    g.add_argument('--ep', type=int, default=1, help='Expert 并行')

    g2 = p.add_argument_group('并行配置（单独指定，覆盖通用配置）')
    g2.add_argument('--tp-a', type=int, default=None)
    g2.add_argument('--pp-a', type=int, default=None)
    g2.add_argument('--ep-a', type=int, default=None)
    g2.add_argument('--tp-b', type=int, default=None)
    g2.add_argument('--pp-b', type=int, default=None)
    g2.add_argument('--ep-b', type=int, default=None)

    g3 = p.add_argument_group('模型配置')
    g3.add_argument('--num-layers', type=int, default=32)
    g3.add_argument('--schedules-method',
                    type=str,
                    default=None,
                    help='调度方法 (如 dualpipev)')
    g3.add_argument('--vpp-stage', type=int, default=None)
    g3.add_argument('--io-threads', type=int, default=4)
    p.add_argument('--atol', type=float, default=1e-6, help='浮点容差')

    args = p.parse_args()

    cfg_a = dict(
        tp=args.tp_a if args.tp_a is not None else args.tp,
        pp=args.pp_a if args.pp_a is not None else args.pp,
        ep=args.ep_a if args.ep_a is not None else args.ep,
    )
    cfg_b = dict(
        tp=args.tp_b if args.tp_b is not None else args.tp,
        pp=args.pp_b if args.pp_b is not None else args.pp,
        ep=args.ep_b if args.ep_b is not None else args.ep,
    )

    print(f'加载 A: {args.dir_a}')
    print(f'  并行: TP={cfg_a['tp']}, PP={cfg_a['pp']}, EP={cfg_a['ep']}')
    state_a = load_merged_state(
        args.dir_a,
        num_layers=args.num_layers,
        schedules_method=args.schedules_method,
        vpp_stage=args.vpp_stage,
        io_threads=args.io_threads,
        **cfg_a,
    )
    print(f'  合并后权重数: {len(state_a)}')

    print(f'\n加载 B: {args.dir_b}')
    print(f'  并行: TP={cfg_b['tp']}, PP={cfg_b['pp']}, EP={cfg_b['ep']}')
    state_b = load_merged_state(
        args.dir_b,
        num_layers=args.num_layers,
        schedules_method=args.schedules_method,
        vpp_stage=args.vpp_stage,
        io_threads=args.io_threads,
        **cfg_b,
    )
    print(f'  合并后权重数: {len(state_b)}')

    ok = compare_states(state_a, state_b, 'A', 'B', args.atol)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
