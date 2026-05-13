#!/usr/bin/env python3
"""
Expand the number of transformer layers in a sharded safetensors model.

Keeps original layers 0 through N-1 unchanged, then creates new layers by
duplicating selected original layers with configurable source mapping.

Copy modes (--copy_source):
  (default)  sequential: new layer i copies from layer (i mod N)
  5          all new layers copy from layer 5
  0,0,1,1,…  comma-separated list of source indices, one per new layer

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

# Mapping from safetensors dtype string to element size in bytes
DTYPE_SIZES: dict[str, int] = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E5M2FN": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
}


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
    # Use search instead of match to handle potential prefixes like 'transformer.'
    m = re.search(r"model\.layers\.(\d+)\.", param_name)
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


def parse_copy_source(raw: str | None, num_original: int, num_new: int) -> list[int]:
    """Parse --copy_source into a list mapping: new_layer_idx → source_layer_idx.

    new_layer_idx runs from num_original to num_original + num_new - 1.

    Formats:
      None / "seq"  →  sequential: [0, 1, 2, …, wrapping if num_new > num_original]
      "5"           →  all new layers copy from layer 5
      "0,0,1,1,…"   →  explicit list, must have exactly num_new entries
    """
    if raw is None or raw.strip().lower() == "seq":
        return [i % num_original for i in range(num_new)]

    raw = raw.strip()
    # Single integer → all new layers copy from that source
    try:
        single = int(raw)
        if single < 0 or single >= num_original:
            print(
                f"ERROR: --copy_source {single} is out of range "
                f"[0, {num_original - 1}].",
                file=sys.stderr,
            )
            sys.exit(1)
        return [single] * num_new
    except ValueError:
        pass

    # Comma-separated list
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != num_new:
        print(
            f"ERROR: --copy_source list has {len(parts)} entries, "
            f"expected {num_new} (one per new layer).",
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


def validate_layer_layout(index: dict, original_layers: int) -> list[int]:
    """Validate that the source model has contiguous layer indices [0, original_layers)."""
    actual_layers = sorted(
        {
            li
            for param_name in index["weight_map"]
            if (li := get_layer_index(param_name)) is not None
        }
    )
    if not actual_layers:
        print(
            "ERROR: No transformer layer parameters matching 'model.layers.<idx>.' "
            "were found in the index.",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_layers = list(range(original_layers))
    if actual_layers != expected_layers:
        print(
            f"ERROR: Detected layer indices {actual_layers[:8]}"
            f"{'...' if len(actual_layers) > 8 else ''}, but expected contiguous "
            f"indices 0-{original_layers - 1}. Refusing to expand a model with "
            "missing, extra, or already-expanded layers.",
            file=sys.stderr,
        )
        sys.exit(1)
    return actual_layers


def build_reverse_map(source_list: list[int], num_original: int) -> dict[int, list[int]]:
    """Build reverse mapping: source_layer → [new_layer_indices].

    new_layer_indices range from num_original to num_original + len(source_list) - 1.
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


def read_safetensors_header(path: Path) -> dict[str, int]:
    """Read only the JSON header of a safetensors file, return {tensor_name: nbytes}.

    The safetensors format is:
        8 bytes: header_size (little-endian u64)
        header_size bytes: JSON dict with keys → {dtype, shape, data_offsets}
        remaining bytes: raw tensor data

    This function reads only the header, not the tensor data, so it's fast
    even for multi-GB files.
    """
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))

    result: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        elem_size = DTYPE_SIZES[meta["dtype"]]
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        result[key] = elem_size * numel
    return result


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the size of a torch tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def main():
    parser = argparse.ArgumentParser(
        description="Expand model layers by duplicating existing layer weights"
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
        "--target_layers",
        type=int,
        default=None,
        help="Target total number of layers. Defaults to double the original (original_layers * 2).",
    )
    parser.add_argument(
        "--copy_source",
        type=str,
        default=None,
        help="Which original layer(s) to copy for the new layers. "
             "Default: sequential (new layer i copies from layer i mod N). "
             "Single int: all new layers copy from that layer. "
             "Comma list: explicit N entries, one per new layer.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    original_layers = args.original_layers

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

    # ── Determine target layers ──────────────────────────────────────────
    target_layers = args.target_layers if args.target_layers else original_layers * 2
    num_new = target_layers - original_layers

    if num_new <= 0:
        print(
            f"ERROR: target_layers ({target_layers}) must be greater than "
            f"original_layers ({original_layers}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Validate layer count against the actual model ────────────────────
    actual_layers = validate_layer_layout(index, original_layers)
    actual_max = actual_layers[-1]

    # ── Parse copy source mapping ────────────────────────────────────────
    source_list = parse_copy_source(args.copy_source, original_layers, num_new)
    # Reverse map: source_layer → [target new layer indices]
    source_to_targets = build_reverse_map(source_list, original_layers)

    if args.copy_source and args.copy_source.strip().lower() != "seq":
        # Print mapping summary
        print("Copy mapping (new_layer → source_layer):")
        for offset, src in enumerate(source_list):
            print(f"  layer {original_layers + offset} ← layer {src}")
    else:
        print(f"Copy mode: sequential (layer N ← layer N mod {original_layers})")

    # Detect shard size from existing files to keep output shards same size
    target_size_bytes = auto_detect_shard_size(model_dir, shard_files)

    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Original layers: {original_layers} (detected: 0-{actual_max})")
    print(f"Target layers: {target_layers} (+{num_new} new)")
    print(f"Target shard size: {target_size_bytes / 1e9:.2f} GB")
    print(f"Found {len(shard_files)} shard files, {len(index['weight_map'])} parameters")

    # ── Update & write config ───────────────────────────────────────────
    updated_keys = []
    for key in ["num_layers", "num_hidden_layers", "n_layers"]:
        if key in config:
            config[key] = target_layers
            updated_keys.append(key)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Updated config.json written (keys updated: {', '.join(updated_keys) if updated_keys else 'none'}).")

    # ── Classify parameters ─────────────────────────────────────────────
    layer_params = set()
    non_layer_params = set()
    for param_name in index["weight_map"]:
        if get_layer_index(param_name) is not None:
            layer_params.add(param_name)
        else:
            non_layer_params.add(param_name)
    print(f"Layer params (will be expanded):  {len(layer_params)}")
    print(f"Non-layer params (unchanged):      {len(non_layer_params)}")

    # ── Pass 1: scan headers to determine exact shard count ─────────────
    # Read only safetensors headers (no tensor data) — fast and low-memory.
    # We simulate shard filling with the same algorithm as Pass 2 so the
    # predicted count matches exactly.
    print("\nPass 1/2: Scanning headers to determine output layout...")
    num_output_shards = 1
    current_bytes = 0
    total_original = 0
    total_duplicated = 0
    total_output_bytes = 0

    for shard_file in tqdm(shard_files, desc="Scanning", unit="shard"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue

        header = read_safetensors_header(shard_path)
        for key, nbytes in header.items():
            # Original
            if nbytes + current_bytes > target_size_bytes and current_bytes > 0:
                num_output_shards += 1
                current_bytes = 0
            current_bytes += nbytes
            total_output_bytes += nbytes
            total_original += 1

            # Duplicates
            layer_idx = get_layer_index(key)
            if layer_idx is not None and layer_idx < original_layers:
                for _ in source_to_targets.get(layer_idx, []):
                    if nbytes + current_bytes > target_size_bytes and current_bytes > 0:
                        num_output_shards += 1
                        current_bytes = 0
                    current_bytes += nbytes
                    total_output_bytes += nbytes
                    total_duplicated += 1
    print(f"Output plan: {total_original:,} original + {total_duplicated:,} duplicated "
          f"= {total_original + total_duplicated:,} tensors")
    print(f"Total output size: {total_output_bytes / 1e9:.2f} GB "
          f"across {num_output_shards} shard(s) "
          f"(~{total_output_bytes / num_output_shards / 1e9:.2f} GB each)")

    # ── Pass 2: actual tensor processing ────────────────────────────────
    new_weight_map: dict[str, str] = {}
    output_shard_idx = 1
    current_tensors: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush_shard():
        """Write accumulated tensors to an output shard, then clear the buffer."""
        nonlocal output_shard_idx, current_tensors, current_bytes
        if not current_tensors:
            return

        shard_name = f"model_{output_shard_idx:05d}-of-{num_output_shards:05d}.safetensors"
        output_path = output_dir / shard_name
        n_tensors = len(current_tensors)
        size_gb = current_bytes / 1e9
        print(f"  Writing shard #{output_shard_idx}/{num_output_shards}: "
              f"{n_tensors} tensors ({size_gb:.2f} GB)")
        save_file(current_tensors, str(output_path))

        for t_name in current_tensors:
            new_weight_map[t_name] = shard_name

        output_shard_idx += 1
        current_tensors.clear()
        current_bytes = 0

    print("\nPass 2/2: Loading tensors and writing shards...")
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

                # ── Duplicated layer parameter ──────────────────────────
                layer_idx = get_layer_index(key)
                if layer_idx is not None and layer_idx < original_layers:
                    for target_idx in source_to_targets.get(layer_idx, []):
                        dup_key = set_layer_index(key, target_idx)

                        if nbytes + current_bytes > target_size_bytes and current_tensors:
                            flush_shard()

                        current_tensors[dup_key] = tensor.clone()
                        current_bytes += nbytes

    # Flush remaining tensors
    flush_shard()

    actual_shards = output_shard_idx - 1
    if actual_shards != num_output_shards:
        print(f"WARNING: Predicted {num_output_shards} shards but wrote {actual_shards}. "
              f"Adjusting index...")
        # Fix up shard file names and the count for the index
        for i in range(1, actual_shards + 1):
            old_name = output_dir / f"model_{i:05d}-of-{num_output_shards:05d}.safetensors"
            new_name = output_dir / f"model_{i:05d}-of-{actual_shards:05d}.safetensors"
            if old_name.exists() and old_name != new_name:
                old_name.rename(new_name)
        num_output_shards = actual_shards

    print(f"\nOutput shards: {actual_shards}")

    # Build final weight map with correct count
    fixed_weight_map: dict[str, str] = {}
    for pname, sname in new_weight_map.items():
        fixed_weight_map[pname] = re.sub(
            r"-of-\d+\.safetensors",
            f"-of-{num_output_shards:05d}.safetensors",
            sname,
        )

    # ── Write new index ──────────────────────────────────────────────────
    new_metadata = {**index.get("metadata", {})}
    if "total_size" in new_metadata:
        new_metadata["total_size"] = total_output_bytes
    new_index = {
        "metadata": new_metadata,
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
    layer_key = next((k for k in ["num_layers", "num_hidden_layers", "n_layers"] if k in out_cfg), None)
    if layer_key:
        print(f"  {layer_key} in config: {out_cfg[layer_key]}")
    else:
        print(f"  WARNING: no recognized layer-count key found in config")

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

        # Verify all target layers present
        expected = set(range(target_layers))
        if set(sorted_layers) == expected:
            print(f"  ✓ All {target_layers} layers present")
        else:
            missing = expected - set(sorted_layers)
            extra = set(sorted_layers) - expected
            if missing:
                print(f"  ✗ Missing layers: {sorted(missing)}")
            if extra:
                print(f"  ✗ Unexpected layers: {sorted(extra)}")

        # Parameter count summary
        input_params = len(index["weight_map"])
        output_params = len(out_idx["weight_map"])
        non_layer_out = len([p for p in out_idx["weight_map"] if get_layer_index(p) is None])
        layer_params_out = output_params - non_layer_out
        print(f"  Input parameters:  {input_params:,}")
        print(f"  Output parameters: {output_params:,} ({non_layer_out:,} non-layer + {layer_params_out:,} layer)")
        if input_params > 0:
            print(f"  Expansion ratio:   {output_params / input_params:.2f}x")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
