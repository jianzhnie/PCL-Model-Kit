#!/usr/bin/env python3
"""
Analyze how model layers are distributed across safetensors shard files.

Reports per-shard layer counts, layer dispersion, non-layer parameter
placement, shard size distribution, and a visual layer→shard mapping matrix.

Usage:
  python analyze_shard_layout.py /path/to/model_dir
  python analyze_shard_layout.py /path/to/model_dir --no-matrix
  python analyze_shard_layout.py /path/to/model_dir --top 10
  python analyze_shard_layout.py /path/to/model_dir --json > report.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_size(num_bytes: int) -> str:
    if num_bytes >= 1e12:
        return f"{num_bytes / 1e12:.2f} TB"
    elif num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    return f"{num_bytes:,} B"


def _bar(value: float, max_val: float, width: int = 20, full: str = "█") -> str:
    """Draw a proportional bar.  *value* and *max_val* should be > 0."""
    if max_val <= 0:
        return ""
    n = max(0, min(width, round(value / max_val * width)))
    return full * n


def _level_icon(ratio: float, thresholds: tuple[float, float] = (0.3, 0.7)) -> str:
    """Return 🟢 / 🟡 / 🔴 for ratio (0 = best, 1 = worst)."""
    if ratio <= thresholds[0]:
        return "🟢"
    elif ratio <= thresholds[1]:
        return "🟡"
    return "🔴"


# ═══════════════════════════════════════════════════════════════════════════════
# analysis engine
# ═══════════════════════════════════════════════════════════════════════════════

def load_index(model_dir: Path) -> dict | None:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            return json.load(f)
    return None


def analyze(model_dir: Path) -> dict:
    index = load_index(model_dir)
    if not index:
        raise FileNotFoundError(f"No model.safetensors.index.json in {model_dir}")

    weight_map = index["weight_map"]
    metadata = index.get("metadata", {})

    shard_layers: dict[str, set[int]] = defaultdict(set)
    shard_params: dict[str, int] = defaultdict(int)
    shard_bytes: dict[str, int] = defaultdict(int)
    layer_params: dict[int, int] = defaultdict(int)

    for param_name, shard_file in weight_map.items():
        shard_params[shard_file] += 1
        m = LAYER_RE.search(param_name)
        if m:
            li = int(m.group(1))
            shard_layers[shard_file].add(li)
            layer_params[li] += 1

    for shard_file in set(list(shard_layers) + list(shard_params)):
        fpath = model_dir / shard_file
        if fpath.exists():
            shard_bytes[shard_file] = fpath.stat().st_size

    # layer → set of shards
    layer_to_shards: dict[int, set[str]] = defaultdict(set)
    for shard, layers in shard_layers.items():
        for li in layers:
            layer_to_shards[li].add(shard)

    layer_shard_count = {li: len(ss) for li, ss in layer_to_shards.items()}

    # contiguous check: are layers in each shard consecutive?
    contiguous_count = 0
    for ls in shard_layers.values():
        sl = sorted(ls)
        if sl[-1] - sl[0] + 1 == len(sl):
            contiguous_count += 1

    # non-layer params
    non_layer_shards: dict[str, list[str]] = defaultdict(list)
    for param_name, shard_file in weight_map.items():
        if not LAYER_RE.search(param_name):
            non_layer_shards[shard_file].append(param_name)

    # shard ordering: sort by ordinal then by first layer
    def _shard_sort_key(s: str) -> tuple[int, int]:
        layers = sorted(shard_layers.get(s, set()))
        first_layer = layers[0] if layers else 9999
        # extract numeric part from filename
        nums = re.findall(r"\d+", s)
        ordinal = int(nums[0]) if nums else 0
        return (ordinal, first_layer)

    all_shards_sorted = sorted(
        set(list(shard_layers) + list(non_layer_shards)), key=_shard_sort_key
    )

    total_shards = len(shard_layers)
    all_layers = sorted({li for ls in shard_layers.values() for li in ls})
    layers_per_shard = {s: len(ls) for s, ls in shard_layers.items()}

    return {
        "model_dir": str(model_dir),
        "total_shards": total_shards,
        "total_params": len(weight_map),
        "total_size_bytes": metadata.get("total_size", 0),
        "layers_per_shard": layers_per_shard,
        "shard_layers": {s: sorted(ls) for s, ls in shard_layers.items()},
        "shard_params": dict(shard_params),
        "shard_bytes": dict(shard_bytes),
        "layer_shard_count": layer_shard_count,
        "layer_to_shards": {li: sorted(ss) for li, ss in layer_to_shards.items()},
        "layer_params": dict(layer_params),
        "non_layer_shards": dict(non_layer_shards),
        "contiguous_count": contiguous_count,
        "all_shards_sorted": all_shards_sorted,
        "all_layers": all_layers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# health score
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_health(result: dict) -> tuple[int, str, list[str]]:
    """Return (score 0-100, label, reasons)."""
    total_shards = result["total_shards"]
    layers_per_shard = result["layers_per_shard"]
    layer_shard_count = result["layer_shard_count"]
    non_layer_shards = result["non_layer_shards"]
    shard_layers = result["shard_layers"]
    shard_bytes = result["shard_bytes"]

    score = 100
    reasons = []

    # 1. single-layer purity (40 pts)
    single = sum(1 for n in layers_per_shard.values() if n == 1)
    purity = single / max(total_shards, 1)
    purity_score = int(40 * purity)
    score -= (40 - purity_score)
    if purity < 0.5:
        reasons.append(f"仅 {purity*100:.0f}% 分片为单层 (扣 {40 - purity_score} 分)")

    # 2. dispersion (25 pts)
    if layer_shard_count:
        avg_disp = sum(layer_shard_count.values()) / len(layer_shard_count)
        if avg_disp <= 3:
            pass  # perfect
        elif avg_disp <= 6:
            deduct = int(25 * (avg_disp - 3) / 13)
            score -= deduct
            if deduct > 0:
                reasons.append(f"平均分散度 {avg_disp:.1f} 文件/layer (扣 {deduct} 分)")
        else:
            deduct = min(25, int(25 * (avg_disp - 3) / 13))
            score -= deduct
            reasons.append(f"平均分散度 {avg_disp:.1f} 文件/layer (扣 {deduct} 分)")

    # 3. non-layer mixing (15 pts)
    mixed = [s for s in non_layer_shards if s in shard_layers]
    if mixed:
        score -= 15
        reasons.append(f"非 layer 参数混入 layer 分片 (扣 15 分)")

    # 4. shard size uniformity (20 pts)
    layer_sizes = [b for s, b in shard_bytes.items() if s in shard_layers]
    if layer_sizes and min(layer_sizes) > 0:
        ratio = max(layer_sizes) / min(layer_sizes)
        if ratio > 3:
            score -= 20
            reasons.append(f"分片大小差异 {ratio:.1f}× (扣 20 分)")
        elif ratio > 1.5:
            deduct = int(20 * (ratio - 1.5) / 1.5)
            score -= deduct
            if deduct > 0:
                reasons.append(f"分片大小差异 {ratio:.1f}× (扣 {deduct} 分)")

    score = max(0, min(100, score))
    if score >= 80:
        label = "🟢 良好"
    elif score >= 50:
        label = "🟡 一般"
    else:
        label = "🔴 需优化"

    return score, label, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def _matrix(result: dict) -> str:
    """Build a compact Layer → Shard mapping matrix (ASCII grid)."""
    all_layers = result["all_layers"]
    shard_layers = result["shard_layers"]
    all_shards = result["all_shards_sorted"]

    if not all_layers:
        return ""

    layer_shards = {s for s in shard_layers}
    shards_with_layers = [s for s in all_shards if s in layer_shards]

    if not shards_with_layers:
        return ""

    n_layers = len(all_layers)
    n_shards = len(shards_with_layers)
    layer_width = max(3, len(str(max(all_layers))))

    # compress shards axis so the matrix fits in ~70 cols (report width 78)
    max_cols = max(20, 70 - layer_width - 4)
    if n_shards <= max_cols:
        groups = [(s,) for s in shards_with_layers]
    else:
        group_size = math.ceil(n_shards / max_cols)
        groups = []
        for i in range(0, n_shards, group_size):
            groups.append(tuple(shards_with_layers[i : i + group_size]))

    n_cols = len(groups)

    lines = []
    # header: show first shard index of each group
    header = " " * (layer_width + 2)
    # determine labels: show shard index at regular intervals
    label_interval = max(1, n_shards // 10)
    for ci in range(n_cols):
        real_idx = sum(len(groups[i]) for i in range(ci))  # first shard in this group
        if real_idx % label_interval == 0 or ci == 0 or ci == n_cols - 1:
            label = f"{real_idx}"
        else:
            label = ""
        header += f"{label:>2s}" if len(label) <= 2 else label[:2]
    lines.append(header)

    # separator
    sep = " " * (layer_width + 1) + "┌" + "─" * n_cols
    lines.append(sep)

    # data rows
    for li in all_layers:
        row = f"{li:>{layer_width}d} │"
        for group in groups:
            present = any(li in shard_layers.get(s, set()) for s in group)
            row += "█" if present else "·"
        lines.append(row)

    # column guide
    guide = " " * (layer_width + 1) + "└" + "─" * n_cols
    lines.append(guide)
    guide2 = " " * (layer_width + 2)
    step = max(1, n_shards // 10)
    for ci in range(n_cols):
        real_shard_idx = sum(len(groups[i]) for i in range(ci))
        if real_shard_idx % step == 0:
            guide2 += "│"
        else:
            guide2 += " "
    lines.append(guide2)
    legend = f"· = 无权重  █ = 有权重  (共 {n_shards} 个分片, 压缩为 {n_cols} 列)"
    lines.append(f"  {legend}")

    return "\n".join(lines)


def _size_histogram(result: dict) -> str:
    """Shard size histogram with 10 buckets."""
    shard_bytes = result["shard_bytes"]
    shard_layers = result["shard_layers"]

    sizes = [b / 1e9 for s, b in shard_bytes.items() if s in shard_layers]
    if not sizes:
        return ""

    hist_min, hist_max = min(sizes), max(sizes)
    if hist_min == hist_max:
        return ""

    n_buckets = 8
    bucket_w = (hist_max - hist_min) / max(n_buckets, 1)
    # auto-detect decimal precision from bucket width
    if bucket_w >= 1:
        fmt = "{:.0f}"
    elif bucket_w >= 0.1:
        fmt = "{:.1f}"
    elif bucket_w >= 0.01:
        fmt = "{:.2f}"
    else:
        fmt = "{:.3f}"

    # avoid all-zero buckets when range is tiny
    if bucket_w < 1e-9:
        return ""

    buckets = [0] * n_buckets
    for sz in sizes:
        idx = min(n_buckets - 1, int((sz - hist_min) / bucket_w))
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    lines = []
    bar_w = 36
    for i in range(n_buckets):
        lo = hist_min + i * bucket_w
        hi = lo + bucket_w
        count = buckets[i]
        bar = _bar(count, max_count, bar_w)
        lines.append(f"  {fmt.format(lo)}-{fmt.format(hi)} GB │{bar}│ {count} 个")
    return "\n".join(lines)


def _dispersion_chart(result: dict) -> str:
    """Per-layer dispersion bar chart."""
    layer_shard_count = result["layer_shard_count"]
    if not layer_shard_count:
        return ""

    all_layers = result["all_layers"]
    max_c = max(layer_shard_count.values()) if layer_shard_count else 1
    bar_w = 50

    lines = []
    for li in all_layers:
        c = layer_shard_count.get(li, 0)
        bar = _bar(c, max_c, bar_w)
        lines.append(f"  layer {li:3d} │{bar}│ {c}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# print report
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(result: dict, top_n: int = 5, show_matrix: bool = True):
    layers_per_shard = result["layers_per_shard"]
    shard_layers = result["shard_layers"]
    shard_bytes = result["shard_bytes"]
    layer_shard_count = result["layer_shard_count"]
    non_layer_shards = result["non_layer_shards"]
    total_shards = result["total_shards"]
    all_layers = result["all_layers"]

    # ── distribution stats ─────────────────────────────────────────────────
    dist: dict[int, int] = defaultdict(int)
    for n in layers_per_shard.values():
        dist[n] += 1

    single_layer = dist.get(1, 0)
    contiguous = result["contiguous_count"]
    scattered = total_shards - contiguous

    if layer_shard_count:
        avg_disp = sum(layer_shard_count.values()) / len(layer_shard_count)
        max_disp = max(layer_shard_count.values())
        min_disp = min(layer_shard_count.values())
    else:
        avg_disp = max_disp = min_disp = 0

    layer_sizes = [b / 1e9 for s, b in shard_bytes.items() if s in shard_layers]
    mixed_non_layer = [s for s in non_layer_shards if s in shard_layers]
    clean_non_layer = [s for s in non_layer_shards if s not in shard_layers]

    total_on_disk = sum(shard_bytes.values())

    # ── health score ───────────────────────────────────────────────────────
    score, score_label, score_reasons = _compute_health(result)

    W = 78

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = "safetensors 分片布局分析报告"
    print(f"║{title:^{W - 2}}║")
    print("╠" + "═" * (W - 2) + "╣")
    model_path = result["model_dir"]
    # Truncate long paths
    if len(model_path) > W - 8:
        model_path = "…" + model_path[-(W - 10) :]
    print(f"║  模型: {model_path:<{W - 9}}║")
    print("╠" + "═" * (W - 2) + "╣")

    info1 = f"分片: {total_shards + len(clean_non_layer)} 个文件"
    info2 = f"参数: {result['total_params']:,} 个"
    info3 = f"大小: {_fmt_size(total_on_disk)}"
    print(f"║  {info1:<30s}{info2:<25s}{info3:<20s}║")

    total_size = result["total_size_bytes"]
    if total_size > 0 and total_on_disk > 0:
        info_extra = f"index 记录: {_fmt_size(total_size)}"
        print(f"║  {info_extra:<{W - 4}}║")

    # health bar
    bar = _bar(score, 100, 30)
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  健康评分: {score:3d}/100  {score_label:<8s}  {bar}║")
    print("╚" + "═" * (W - 2) + "╝")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # CORE METRICS TABLE
    # ──────────────────────────────────────────────────────────────────────────
    # compute icons for each dimension
    purity = single_layer / max(total_shards, 1)
    purity_icon = _level_icon(1 - purity, (0.8, 0.95))

    contig_ratio = scattered / max(total_shards, 1)
    contig_icon = _level_icon(contig_ratio, (0.1, 0.3))

    disp_ratio = max(0, min(1, (avg_disp - 1) / 15)) if avg_disp > 0 else 0
    disp_icon = _level_icon(disp_ratio, (0.2, 0.4))

    size_ratio = (
        max(layer_sizes) / max(min(layer_sizes), 0.001) if layer_sizes and min(layer_sizes) > 0 else 1
    )
    size_icon = _level_icon((size_ratio - 1) / 3, (0.15, 0.4)) if size_ratio > 1 else "🟢"

    non_layer_icon = "🟢" if not mixed_non_layer else "🔴"

    # table
    col_w = (W - 7) // 5
    sep = "┌" + "┬".join("─" * col_w for _ in range(5)) + "┐"
    sep2 = "├" + "┼".join("─" * col_w for _ in range(5)) + "┤"
    sep3 = "└" + "┴".join("─" * col_w for _ in range(5)) + "┘"

    def _cell(text: str, w: int = col_w) -> str:
        return f"{text:^{w}}"

    print(sep)
    print(
        "│"
        + _cell("每分片层数")
        + "│"
        + _cell("层连续性")
        + "│"
        + _cell("分散度")
        + "│"
        + _cell("大小均匀")
        + "│"
        + _cell("非层参数")
        + "│"
    )
    print(sep2)

    purity_text = f"1 层: {single_layer}/{total_shards}" if total_shards > 0 else "N/A"
    contig_text = f"连续: {contiguous}/{total_shards}" if total_shards > 0 else "N/A"
    disp_text = f"avg {avg_disp:.1f} 文件/层" if avg_disp > 0 else "N/A"
    size_text = f"max/min {size_ratio:.1f}×" if size_ratio > 1 else "完全均匀"
    non_layer_text = f"独立: {len(clean_non_layer)}  混合: {len(mixed_non_layer)}" if non_layer_shards else "无"

    print(
        "│"
        + _cell(f"{purity_icon} {purity_text}")
        + "│"
        + _cell(f"{contig_icon} {contig_text}")
        + "│"
        + _cell(f"{disp_icon} {disp_text}")
        + "│"
        + _cell(f"{size_icon} {size_text}")
        + "│"
        + _cell(f"{non_layer_icon} {non_layer_text}")
        + "│"
    )
    print(sep3)
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Layer → Shard 映射矩阵
    # ──────────────────────────────────────────────────────────────────────────
    if show_matrix and all_layers:
        print("┌─ Layer → Shard 映射矩阵 " + "─" * (W - 25) + "┐")
        print("│  横轴 = safetensors 分片文件, 纵轴 = layer 编号                      │")
        print("│  █ = 该 layer 的权重存在该分片中    · = 无权重                          │")
        print("├" + "─" * (W - 2) + "┤")
        m = _matrix(result)
        if m:
            for line in m.split("\n"):
                # pad or truncate each line to W-4
                if len(line) > W - 4:
                    line = line[: W - 4]
                print(f"│ {line:<{W - 4}} │")
        print("└" + "─" * (W - 2) + "┘")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 分片大小直方图
    # ──────────────────────────────────────────────────────────────────────────
    if layer_sizes:
        print("┌─ 分片大小分布 (直方图) " + "─" * (W - 23) + "┐")
        hist = _size_histogram(result)
        if hist:
            for line in hist.split("\n"):
                print(f"│ {line:<{W - 4}} │")
        # summary line
        print(f"│  范围: {min(layer_sizes):.2f} ~ {max(layer_sizes):.2f} GB"
              f"  |  平均: {sum(layer_sizes)/len(layer_sizes):.2f} GB"
              f"  |  合计: {sum(layer_sizes):.2f} GB{' ' * (W - 58)}│")
        print("└" + "─" * (W - 2) + "┘")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Layer 分散度详情
    # ──────────────────────────────────────────────────────────────────────────
    if layer_shard_count and len(all_layers) <= 40:
        print("┌─ 每个 Layer 的分散度 " + "─" * (W - 21) + "┐")
        chart = _dispersion_chart(result)
        if chart:
            for line in chart.split("\n"):
                print(f"│ {line:<{W - 4}} │")
        print("└" + "─" * (W - 2) + "┘")
        print()
    elif layer_shard_count:
        # compact: show distribution + extremes
        disp_dist: dict[int, int] = defaultdict(int)
        for c in layer_shard_count.values():
            disp_dist[c] += 1
        print("┌─ 每个 Layer 的分散度 (紧凑模式: 层数过多) " + "─" * (W - 39) + "┐")
        for c in sorted(disp_dist):
            examples = sorted(li for li, ci in layer_shard_count.items() if ci == c)[:6]
            ex_str = ", ".join(str(x) for x in examples)
            print(f"│  {c:2d} 文件/layer: {disp_dist[c]:3d} 层  例: {ex_str:<30s}{' ' * (W - 52)}│")
        print(f"│  范围: {min_disp} ~ {max_disp} 文件/layer, 平均 {avg_disp:.1f} 文件/layer{' ' * (W - 42)}│")
        print("└" + "─" * (W - 2) + "┘")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # 4. 非 layer 参数
    # ──────────────────────────────────────────────────────────────────────────
    if non_layer_shards:
        print("┌─ 非 layer 参数详情 " + "─" * (W - 18) + "┐")
        if mixed_non_layer:
            print(f"│  🔴 {len(mixed_non_layer)} 个分片中非 layer 参数与 layer 权重混合:{' ' * (W - 40)}│")
            for s in sorted(mixed_non_layer)[:3]:
                ls = sorted(shard_layers.get(s, set()))
                ls_str = f"[{ls[0]}-{ls[-1]}]" if len(ls) > 4 else str(ls)
                print(f"│     {s}: 包含 layer {ls_str}{' ' * (W - 30)}│")
        if clean_non_layer:
            print(f"│  🟢 {len(clean_non_layer)} 个分片独立存放非 layer 参数{' ' * (W - 30)}│")
        # list param families
        families = set()
        for plist in non_layer_shards.values():
            for p in plist:
                short = re.sub(r"layers\.\d+", "layers.N", p)
                families.add(short)
        if families:
            fam_list = ", ".join(sorted(families))
            if len(fam_list) > W - 8:
                fam_list = fam_list[: W - 12] + "…"
            print(f"│  种类: {fam_list:<{W - 8}}│")
        print("└" + "─" * (W - 2) + "┘")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 每分片 layer 数量分布
    # ──────────────────────────────────────────────────────────────────────────
    print("┌─ 每分片包含的 layer 数量 " + "─" * (W - 23) + "┐")
    max_dist = max(dist.values()) if dist else 1
    for n in sorted(dist):
        bar = _bar(dist[n], max_dist, 50)
        pct = dist[n] / max(total_shards, 1) * 100
        print(f"│  {n:2d} 层/分片  {dist[n]:4d} 个 ({pct:5.1f}%)  {bar:<50s} │")
    print("└" + "─" * (W - 2) + "┘")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # 6. 综合评估
    # ──────────────────────────────────────────────────────────────────────────
    issues = []
    ok = []
    info_lines = []

    if total_shards == 0:
        ok.append("模型无非 layer 参数，无需 layer 分片分析。")
    elif single_layer == total_shards:
        ok.append("每个分片仅含 1 个 layer，分片策略理想。")
    elif purity > 0.8:
        issues.append(f"大部分分片仅含 1 个 layer，但仍有 {total_shards - single_layer} 个分片混合了多层。")
    else:
        issues.append(f"{total_shards - single_layer}/{total_shards} 个分片混合了多个 layer，不利于 layer-wise 加载。")

    if avg_disp > 10:
        issues.append(f"Layer 极度分散：平均每个 layer 跨越 {avg_disp:.1f} 个文件 (范围 {min_disp}~{max_disp})。")
    elif avg_disp > 5:
        issues.append(f"Layer 较为分散：平均每个 layer 跨越 {avg_disp:.1f} 个文件。建议控制在 1-3 个。")
    else:
        ok.append(f"Layer 分散度较低：平均 {avg_disp:.1f} 文件/layer。")

    if scattered > total_shards * 0.5:
        issues.append(f"{scattered}/{total_shards} ({scattered / max(total_shards, 1) * 100:.0f}%) 分片内 layer 编号不连续（跳跃存放）。")

    if mixed_non_layer:
        issues.append(f"{len(mixed_non_layer)} 个分片中非 layer 参数与 layer 权重混合，应分离。")

    if layer_sizes and min(layer_sizes) > 0:
        ratio = max(layer_sizes) / min(layer_sizes)
        if ratio > 3:
            issues.append(f"分片大小差异 {ratio:.1f}× ({min(layer_sizes):.1f}~{max(layer_sizes):.1f} GB)，碎片化严重。")
        elif ratio > 1.5:
            issues.append(f"分片大小差异 {ratio:.1f}× ({min(layer_sizes):.1f}~{max(layer_sizes):.1f} GB)。")

    # detailed reasons from health score
    if score_reasons:
        info_lines.append(f"📊 扣分明细: {'; '.join(score_reasons)}")

    print("┌─ 综合评估 " + "─" * (W - 11) + "┐")
    for line in ok:
        truncated = line[: W - 4] if len(line) > W - 4 else line
        print(f"│  ✅ {truncated:<{W - 7}}│")
    for line in issues:
        truncated = line[: W - 4] if len(line) > W - 4 else line
        print(f"│  ❌ {truncated:<{W - 7}}│")
    for line in info_lines:
        truncated = line[: W - 4] if len(line) > W - 4 else line
        print(f"│  ℹ️  {truncated:<{W - 7}}│")

    if not issues:
        print(f"│{' ' * (W - 2)}│")
        print(f"│  结论: 分片布局良好，无需优化。{' ' * (W - 22)}│")
    else:
        print(f"│{' ' * (W - 2)}│")
        print(f"│  建议: 使用扩展脚本的 --max_layers_per_shard 1 重新扩展模型。{' ' * (W - 42)}│")
        print(f"│  目标: 每个 safetensors 仅含 1 个 layer，单层过大时跨 2-3 个文件。{' ' * (W - 45)}│")

    print("└" + "─" * (W - 2) + "┘")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Appendix: example shards
    # ──────────────────────────────────────────────────────────────────────────
    if top_n > 0:
        print(f"附: 分片示例")
        print("─" * 50)

        few = sorted(layers_per_shard.items(), key=lambda x: x[1])[:top_n]
        print(f"  Layer 最少的分片:")
        for shard, n in few:
            ls = shard_layers[shard]
            sz = shard_bytes.get(shard, 0) / 1e9
            print(f"    {shard}: {n} 层 {ls}, {sz:.2f} GB")

        many = sorted(layers_per_shard.items(), key=lambda x: -x[1])[:top_n]
        print(f"  Layer 最多的分片:")
        for shard, n in many:
            ls = shard_layers[shard]
            sz = shard_bytes.get(shard, 0) / 1e9
            label = f"[{ls[0]}-{ls[-1]}]" if len(ls) > 5 else str(ls)
            print(f"    {shard}: {n} 层 {label}, {sz:.2f} GB")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="分析 safetensors 分片布局：layer 分布、分散度、映射矩阵"
    )
    parser.add_argument(
        "model_dir",
        help="模型目录 (需包含 model.safetensors.index.json)",
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="示例分片数量 (默认 3)",
    )
    parser.add_argument(
        "--no-matrix", action="store_true",
        help="不显示 Layer→Shard 映射矩阵",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="输出原始 JSON 数据到 stdout (用于脚本解析)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"错误: 目录不存在 {model_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        result = analyze(model_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Convert sets to lists for JSON serialization
        json_result = {
            "model_dir": result["model_dir"],
            "total_shards": result["total_shards"],
            "total_params": result["total_params"],
            "total_size_bytes": result["total_size_bytes"],
            "layers_per_shard": result["layers_per_shard"],
            "shard_layers": result["shard_layers"],
            "shard_params": result["shard_params"],
            "shard_bytes": result["shard_bytes"],
            "layer_shard_count": result["layer_shard_count"],
            "layer_to_shards": result["layer_to_shards"],
            "layer_params": result["layer_params"],
            "non_layer_shards": result["non_layer_shards"],
            "contiguous_count": result["contiguous_count"],
            "all_shards_sorted": result["all_shards_sorted"],
            "all_layers": result["all_layers"],
        }
        print(json.dumps(json_result, indent=2, ensure_ascii=False))
    else:
        print_report(result, top_n=args.top, show_matrix=not args.no_matrix)


if __name__ == "__main__":
    main()
