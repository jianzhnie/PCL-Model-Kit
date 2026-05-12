#!/usr/bin/env python3
"""
Double the number of transformer layers in a sharded safetensors model.

For a model with 28 layers, creates a 56-layer version by:
1. Keeping original layers 0-27 unchanged
2. Duplicating them as layers 28-55, with configurable source mapping

Copy modes (--copy_source):
  (default)  new layer N copies from layer N-28  (sequential: 28←0, 29←1, …)
  5          all new layers copy from layer 5
  0,0,1,1,…  comma-separated list of 28 source indices, one per new layer

Processes shards one at a time to keep memory usage manageable.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return None


def get_layer_index(param_name: str) -> int | None:
    """Extract layer index from parameter name. Returns None for non-layer params."""
    m = re.match(r"model\.layers\.(\d+)\.", param_name)
    if m:
        return int(m.group(1))
    return None


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name. e.g. model.layers.0.xxx → model.layers.5.xxx"""
    return re.sub(
        r"model\.layers\.(\d+)\.",
        f"model.layers.{new_index}.",
        param_name,
    )


def parse_copy_source(raw: str | None, num_original: int) -> list[int]:
    """Parse --copy_source into a list mapping: new_layer_idx → source_layer_idx.

    new_layer_idx runs from num_original to 2*num_original - 1.

    Formats:
      None / "seq"  →  sequential: [0, 1, 2, ..., num_original-1]
      "5"           →  all new layers copy from layer 5
      "0,0,1,1,…"   →  explicit list, must have exactly num_original entries
    """
    if raw is None or raw.strip().lower() == "seq":
        return list(range(num_original))

    raw = raw.strip()
    # Single integer → all new layers copy from that source
    try:
        single = int(raw)
        return [single] * num_original
    except ValueError:
        pass

    # Comma-separated list
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != num_original:
        print(
            f"ERROR: --copy_source list has {len(parts)} entries, "
            f"expected {num_original} (one per new layer).",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        result = [int(p) for p in parts]
    except ValueError as e:
        print(f"ERROR: Invalid integer in --copy_source: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate all source indices are in range
    for i, src in enumerate(result):
        if src < 0 or src >= num_original:
            print(
                f"ERROR: --copy_source[{i}] = {src} is out of range "
                f"[0, {num_original - 1}].",
                file=sys.stderr,
            )
            sys.exit(1)
    return result


def build_reverse_map(source_list: list[int], num_original: int) -> dict[int, list[int]]:
    """Build reverse mapping: source_layer → [new_layer_indices].

    new_layer_indices range from num_original to 2*num_original - 1.
    """
    rev: dict[int, list[int]] = defaultdict(list)
    for offset, src in enumerate(source_list):
        new_idx = num_original + offset
        rev[src].append(new_idx)
    return dict(rev)


def auto_detect_shard_size(model_dir: Path, shard_files: list[str]) -> int:
    """Detect the target shard size from existing shard files on disk.

    Returns the average file size in bytes. Falls back to estimating from the
    safetensors index if files aren't available yet.
    """
    # Try actual file sizes first
    file_sizes = []
    for fname in shard_files:
        fpath = model_dir / fname
        if fpath.exists():
            file_sizes.append(fpath.stat().st_size)

    if file_sizes:
        avg_size = int(sum(file_sizes) / len(file_sizes))
        print(f"Detected shard size from {len(file_sizes)} existing files: "
              f"{avg_size / 1e9:.2f} GB (average)")
        return avg_size

    # Fallback: typical size for large models (will print a warning)
    print("WARNING: No shard files found on disk. Using default 8GB target. "
          "Output shards will match this size, not necessarily the originals.")
    return 8 * 1024 ** 3


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the size of a torch tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def main():
    parser = argparse.ArgumentParser(
        description="Double model layers by duplicating existing layer weights"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the original model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output the doubled model",
    )
    parser.add_argument(
        "--original_layers",
        type=int,
        default=28,
        help="Number of layers in the original model (default: 28)",
    )
    parser.add_argument(
        "--copy_source",
        type=str,
        default=None,
        help="Which original layer(s) to copy for the new layers. "
             "Default: sequential (28←0, 29←1, …). "
             "Single int: all new layers copy from that layer. "
             "Comma list: explicit N entries, one per new layer.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    original_layers = args.original_layers
    num_new = original_layers  # doubling: same number of new layers as original

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load config & index ──────────────────────────────────────────────
    config = load_config(model_dir)
    index = load_index(model_dir)

    if not index:
        print("ERROR: Model is not sharded (no model.safetensors.index.json). "
              "This script handles sharded models.", file=sys.stderr)
        sys.exit(1)

    shard_files = sorted(set(index["weight_map"].values()))

    # ── Validate layer count against the actual model ────────────────────
    actual_layers = set()
    for param_name in index["weight_map"]:
        li = get_layer_index(param_name)
        if li is not None:
            actual_layers.add(li)
    actual_max = max(actual_layers) if actual_layers else -1
    actual_count = len(actual_layers)

    if actual_max >= original_layers:
        print(f"ERROR: Model already has layers up to index {actual_max}. "
              f"Either the model was already doubled or --original_layers ({original_layers}) "
              f"is too low. Refusing to run — this would create overlapping layer indices.",
              file=sys.stderr)
        sys.exit(1)

    if actual_count != original_layers:
        if actual_max + 1 != original_layers:
            print(f"WARNING: Detected {actual_count} unique layer indices "
                  f"(max={actual_max}), but --original_layers={original_layers}. "
                  f"Expected {original_layers} contiguous layers (0-{original_layers - 1}).")

    # ── Parse copy source mapping ────────────────────────────────────────
    source_list = parse_copy_source(args.copy_source, original_layers)
    # Reverse map: source_layer → [target new layer indices]
    source_to_targets = build_reverse_map(source_list, original_layers)

    if args.copy_source and args.copy_source.strip().lower() != "seq":
        # Print mapping summary
        print("Copy mapping (new_layer → source_layer):")
        for offset, src in enumerate(source_list):
            print(f"  layer {original_layers + offset} ← layer {src}")
    else:
        print(f"Copy mode: sequential (layer N ← layer N-{original_layers})")

    # Detect shard size from existing files to keep output shards same size
    target_size_bytes = auto_detect_shard_size(model_dir, shard_files)

    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Original layers: {original_layers} (detected: 0-{actual_max})")
    print(f"New total layers: {original_layers * 2}")
    print(f"Target shard size: {target_size_bytes / 1e9:.2f} GB")
    print(f"Found {len(shard_files)} shard files, {len(index['weight_map'])} parameters")

    # ── Update & write config ───────────────────────────────────────────
    config["num_layers"] = original_layers * 2
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("Updated config.json written.")

    # ── Classify parameters ─────────────────────────────────────────────
    layer_params = set()
    non_layer_params = set()
    for param_name in index["weight_map"]:
        if get_layer_index(param_name) is not None:
            layer_params.add(param_name)
        else:
            non_layer_params.add(param_name)
    print(f"Layer params (×2 after duplication): {len(layer_params)}")
    print(f"Non-layer params (unchanged):        {len(non_layer_params)}")

    # ── Process shards ──────────────────────────────────────────────────
    new_weight_map: dict[str, str] = {}
    output_shard_idx = 1
    current_tensors: dict[str, torch.Tensor] = {}
    current_bytes = 0
    total_original = 0
    total_duplicated = 0

    def flush_shard():
        """Write accumulated tensors to an output shard, then clear the buffer."""
        nonlocal output_shard_idx, current_tensors, current_bytes
        if not current_tensors:
            return

        shard_name = f"model_{output_shard_idx:05d}-of-XXXXX.safetensors"
        output_path = output_dir / shard_name
        n_tensors = len(current_tensors)
        size_gb = current_bytes / 1e9
        print(f"  Writing shard #{output_shard_idx}: {n_tensors} tensors ({size_gb:.2f} GB)")
        save_file(current_tensors, str(output_path))

        for t_name in current_tensors:
            new_weight_map[t_name] = shard_name

        output_shard_idx += 1
        current_tensors.clear()
        current_bytes = 0

    print("\nProcessing shards...")
    for shard_file in tqdm(shard_files, desc="Input shards", unit="shard"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                nbytes = tensor_nbytes(tensor)

                # ── Original parameter ──────────────────────────────────
                if nbytes + current_bytes > target_size_bytes and current_tensors:
                    flush_shard()

                current_tensors[key] = tensor
                current_bytes += nbytes
                total_original += 1

                # ── Duplicated layer parameter ──────────────────────────
                layer_idx = get_layer_index(key)
                if layer_idx is not None and layer_idx < original_layers:
                    for target_idx in source_to_targets.get(layer_idx, []):
                        dup_key = set_layer_index(key, target_idx)

                        if nbytes + current_bytes > target_size_bytes and current_tensors:
                            flush_shard()

                        current_tensors[dup_key] = tensor.clone()
                        current_bytes += nbytes
                        total_duplicated += 1

    # Flush remaining tensors
    flush_shard()

    num_output_shards = output_shard_idx - 1
    print(f"\nOriginal parameters:  {total_original}")
    print(f"Duplicated parameters: {total_duplicated}")
    print(f"Output shards:         {num_output_shards}")

    # ── Rename shard files with correct count ───────────────────────────
    print("\nFinalizing shard file names...")
    for i in range(1, num_output_shards + 1):
        old = output_dir / f"model_{i:05d}-of-XXXXX.safetensors"
        new = output_dir / f"model_{i:05d}-of-{num_output_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)

    # Fix weight map with correct total count
    fixed_weight_map: dict[str, str] = {}
    for pname, sname in new_weight_map.items():
        fixed_weight_map[pname] = sname.replace("-of-XXXXX", f"-of-{num_output_shards:05d}")

    # ── Write new index ──────────────────────────────────────────────────
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": fixed_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"Index written: {len(fixed_weight_map)} entries")

    # ── Copy auxiliary files ────────────────────────────────────────────
    # Copy everything except weight files, index, and config (already handled)
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if not fpath.is_file():
            continue
        if fpath.suffix in skip_suffixes or fpath.name in skip_names:
            continue
        shutil.copy2(fpath, output_dir / fpath.name)
        print(f"  Copied: {fpath.name}")

    # ── Quick verification ──────────────────────────────────────────────
    print("\nVerification:")
    out_cfg = load_config(output_dir)
    print(f"  num_layers in config: {out_cfg['num_layers']}")

    out_idx = load_index(output_dir)
    if out_idx:
        layers_found = defaultdict(list)
        for pname in out_idx["weight_map"]:
            li = get_layer_index(pname)
            if li is not None:
                layers_found[li].append(pname)

        sorted_layers = sorted(layers_found.keys())
        print(f"  Layers present: {sorted_layers[0]} - {sorted_layers[-1]} "
              f"({len(sorted_layers)} total)")

        # Spot-check: verify each new layer has the same param structure as its source
        print(f"  Verifying copy mapping (new_layer ← source_layer):")
        all_ok = True
        for offset, src in enumerate(source_list):
            new_li = original_layers + offset
            if src in layers_found and new_li in layers_found:
                src_set = {re.sub(r"layers\.\d+", "layers.N", p) for p in layers_found[src]}
                new_set = {re.sub(r"layers\.\d+", "layers.N", p) for p in layers_found[new_li]}
                if src_set != new_set:
                    print(f"    ✗ layer {new_li} ← layer {src}: structure mismatch!")
                    all_ok = False
        if all_ok:
            print(f"    ✓ All {num_new} new layers match their source structure")

        # Verify all 56 layers present
        expected = set(range(original_layers * 2))
        if set(sorted_layers) == expected:
            print(f"  ✓ All {original_layers * 2} layers present")
        else:
            missing = expected - set(sorted_layers)
            extra = set(sorted_layers) - expected
            if missing:
                print(f"  ✗ Missing layers: {sorted(missing)}")
            if extra:
                print(f"  ✗ Unexpected layers: {sorted(extra)}")

        # Parameter count per layer
        for li in sorted_layers[:3]:
            print(f"    Layer {li}: {len(layers_found[li])} params")
        print(f"    ...")
        for li in sorted_layers[-3:]:
            print(f"    Layer {li}: {len(layers_found[li])} params")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
