#!/usr/bin/env python3
"""
Extract weight names, shapes and dtypes from safetensors files by reading headers.
No safetensors dependency required — parses the binary header directly.

Usage:
  python utils/get_hf_weights_from_safetensors.py /path/to/model_dir -o weight_map.json --pretty
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# safetensors dtype → (name, bytes_per_element)
DTYPE_MAP = {
    'F32': ('float32', 4),
    'F16': ('float16', 2),
    'BF16': ('bfloat16', 2),
    'F64': ('float64', 8),
    'I64': ('int64', 8),
    'I32': ('int32', 4),
    'I16': ('int16', 2),
    'I8': ('int8', 1),
    'U8': ('uint8', 1),
    'BOOL': ('bool', 1),
}


def read_safetensor_header(path: Path) -> dict:
    """Read the JSON header from a safetensors file (first 8 bytes = header length)."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(header_len))


def find_shard_files(model_path: Path) -> list[Path]:
    """Resolve safetensors file(s) from a path (file or directory)."""
    if model_path.is_file():
        if model_path.suffix != '.safetensors':
            sys.exit(f'Not a safetensors file: {model_path}')
        return [model_path]

    # Try index file first
    index_file = model_path / 'model.safetensors.index.json'
    if index_file.is_file():
        with open(index_file) as f:
            index = json.load(f)
        shard_names = sorted(set(index.get('weight_map', {}).values()))
        return [model_path / n for n in shard_names]

    # Single file in directory
    single = model_path / 'model.safetensors'
    if single.is_file():
        return [single]

    sys.exit(f'No safetensors found in: {model_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract weight info from safetensors')
    parser.add_argument('model_path', help='Directory or .safetensors file')
    parser.add_argument('-o',
                        '--output',
                        default='weight_map.json',
                        help='Output JSON')
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        sys.exit(f'Path not found: {model_path}')

    shards = find_shard_files(model_path)
    print(f'Found {len(shards)} shard(s)')

    weight_map = {}
    total_size = 0

    for shard in shards:
        header = read_safetensor_header(shard)
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            shape = meta.get('shape', [])
            dtype_raw = meta.get('dtype', 'F32').upper()
            dtype_name, elem_bytes = DTYPE_MAP.get(dtype_raw,
                                                   (dtype_raw.lower(), 4))
            numel = 1
            for d in shape:
                numel *= d
            total_size += numel * elem_bytes
            weight_map[name] = {'shape': shape, 'dtype': dtype_name}

    result = {
        'metadata': {
            'total_size': total_size
        },
        'weight_map': weight_map,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f, indent=2 if args.pretty else None)

    print(f'Weights: {len(weight_map)}, Size: {total_size / (1024**3):.2f} GB')
    print(f'Saved to: {output}')

    # Show first 10 weights
    for name, info in list(weight_map.items())[:10]:
        print(f'  {name}: {info['shape']} ({info['dtype']})')
    if len(weight_map) > 10:
        print(f'  ... and {len(weight_map) - 10} more')


if __name__ == '__main__':
    main()
