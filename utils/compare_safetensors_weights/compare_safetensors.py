#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
import sys
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import torch
from safetensors import safe_open

try:
    import tqdm as _tqdm
except Exception:
    _tqdm = None


def _tqdm_iter(iterable, total, desc):
    if _tqdm is None:
        return iterable
    return _tqdm.tqdm(iterable, total=total, desc=desc)


def _compare_keys_chunk(file1_path, file2_path, keys, tolerance, verbose,
                        only_diff, header):
    mismatch_count = 0
    checked_count = 0
    output_lines = []

    try:
        with safe_open(file1_path, framework='pt', device='cpu') as f1, \
             safe_open(file2_path, framework='pt', device='cpu') as f2:
            for key in keys:
                checked_count += 1

                s1_shape = None
                s2_shape = None
                s1_dtype = None
                s2_dtype = None

                try:
                    slice1 = f1.get_slice(key)
                    slice2 = f2.get_slice(key)
                    s1_shape = slice1.get_shape()
                    s2_shape = slice2.get_shape()
                    s1_dtype = slice1.get_dtype()
                    s2_dtype = slice2.get_dtype()
                except Exception:
                    pass

                if s1_shape is not None and s2_shape is not None and s1_shape != s2_shape:
                    mismatch_count += 1
                    output_lines.append(
                        f"\n[{header}][MISMATCH] Shape mismatch for '{key}':\n"
                    )
                    output_lines.append(f'  File 1: {s1_shape}\n')
                    output_lines.append(f'  File 2: {s2_shape}\n')
                    continue

                if s1_dtype is not None and s2_dtype is not None and s1_dtype != s2_dtype:
                    mismatch_count += 1
                    output_lines.append(
                        f"\n[{header}][MISMATCH] Dtype mismatch for '{key}':\n"
                    )
                    output_lines.append(f'  File 1: {s1_dtype}\n')
                    output_lines.append(f'  File 2: {s2_dtype}\n')
                    continue

                with torch.no_grad():
                    tensor1 = f1.get_tensor(key)
                    tensor2 = f2.get_tensor(key)

                if s1_shape is None or s2_shape is None:
                    if tensor1.shape != tensor2.shape:
                        mismatch_count += 1
                        output_lines.append(
                            f"\n[{header}][MISMATCH] Shape mismatch for '{key}':\n"
                        )
                        output_lines.append(f'  File 1: {tensor1.shape}\n')
                        output_lines.append(f'  File 2: {tensor2.shape}\n')
                        continue

                if s1_dtype is None or s2_dtype is None:
                    if tensor1.dtype != tensor2.dtype:
                        mismatch_count += 1
                        output_lines.append(
                            f"\n[{header}][MISMATCH] Dtype mismatch for '{key}':\n"
                        )
                        output_lines.append(f'  File 1: {tensor1.dtype}\n')
                        output_lines.append(f'  File 2: {tensor2.dtype}\n')
                        continue

                if not torch.allclose(
                        tensor1, tensor2, atol=tolerance, rtol=tolerance):
                    mismatch_count += 1
                    diff = (tensor1 - tensor2).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    output_lines.append(
                        f"\n[{header}][MISMATCH] Value mismatch for '{key}':\n"
                    )
                    output_lines.append(f'  Max diff: {max_diff:.6e}\n')
                    output_lines.append(f'  Mean diff: {mean_diff:.6e}\n')
                elif verbose and not only_diff:
                    output_lines.append(f'[OK] {key}\n')

    except Exception as e:
        return 1, checked_count, [
            f'\n[{header}] Error during comparison: {e}\n'
        ]

    return mismatch_count, checked_count, output_lines


def _compare_safetensors_pair(file1_path,
                              file2_path,
                              tolerance=1e-5,
                              verbose=False,
                              only_diff=False,
                              label=None,
                              inner_jobs=1):

    if not os.path.exists(file1_path):
        return 1, f'Error: File not found: {file1_path}\n'
    if not os.path.exists(file2_path):
        return 1, f'Error: File not found: {file2_path}\n'

    output_lines = []
    header = label or os.path.basename(file1_path)
    if not only_diff:
        output_lines.append(
            f'Comparing:\n  File 1: {file1_path}\n  File 2: {file2_path}\n')

    try:
        with safe_open(file1_path, framework='pt', device='cpu') as f1:
            keys1 = set(f1.keys())
        with safe_open(file2_path, framework='pt', device='cpu') as f2:
            keys2 = set(f2.keys())
    except Exception as e:
        return 1, f'Error opening safetensors files: {e}\n'

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    mismatch_count = 0

    if missing_in_2:
        mismatch_count += 1
        output_lines.append(
            f'\n[{header}] Keys present in File 1 but missing in File 2 ({len(missing_in_2)}):\n'
        )
        for k in sorted(list(missing_in_2))[:10]:
            output_lines.append(f'  - {k}\n')
        if len(missing_in_2) > 10:
            output_lines.append(f'  ... and {len(missing_in_2) - 10} more.\n')

    if missing_in_1:
        mismatch_count += 1
        output_lines.append(
            f'\n[{header}] Keys present in File 2 but missing in File 1 ({len(missing_in_1)}):\n'
        )
        for k in sorted(list(missing_in_1))[:10]:
            output_lines.append(f'  - {k}\n')
        if len(missing_in_1) > 10:
            output_lines.append(f'  ... and {len(missing_in_1) - 10} more.\n')

    common_keys = sorted(list(keys1.intersection(keys2)))
    if not only_diff:
        output_lines.append(
            f'\nComparing {len(common_keys)} common tensors...\n')

    checked_count = 0
    if common_keys:
        inner_jobs = int(inner_jobs or 1)
        if inner_jobs < 1:
            inner_jobs = 1

        if inner_jobs == 1 or len(common_keys) == 1:
            mc, cc, out_lines = _compare_keys_chunk(
                file1_path,
                file2_path,
                common_keys,
                tolerance,
                verbose,
                only_diff,
                header,
            )
            mismatch_count += mc
            checked_count += cc
            output_lines.extend(out_lines)
        else:
            chunk_size = max(1, math.ceil(len(common_keys) / inner_jobs))
            chunks = [
                common_keys[i:i + chunk_size]
                for i in range(0, len(common_keys), chunk_size)
            ]
            results = [None] * len(chunks)

            with ThreadPoolExecutor(max_workers=inner_jobs) as ex:
                fut_to_idx = {
                    ex.submit(_compare_keys_chunk, file1_path, file2_path,
                              chunk, tolerance, verbose, only_diff, header):
                    idx
                    for idx, chunk in enumerate(chunks)
                }

                for fut in _tqdm_iter(as_completed(fut_to_idx),
                                      total=len(fut_to_idx),
                                      desc=f'{header} keys'):
                    idx = fut_to_idx[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        results[idx] = (1, 0, [
                            f'\n[{header}] Error during comparison: {e}\n'
                        ])

            for mc, cc, out_lines in results:
                mismatch_count += mc
                checked_count += cc
                output_lines.extend(out_lines)

    if not only_diff:
        output_lines.append('\n' + '=' * 50 + '\n')
        output_lines.append('Comparison Summary:\n')
        output_lines.append(f'  Total keys checked: {checked_count}\n')
        output_lines.append(f'  Mismatched tensors: {mismatch_count}\n')
        if missing_in_1 or missing_in_2:
            output_lines.append(
                f'  Missing keys: {len(missing_in_1) + len(missing_in_2)}\n')

    if only_diff and mismatch_count == 0:
        return 0, ''

    return (0 if mismatch_count == 0 else 1), ''.join(output_lines)


def _iter_safetensors_files(directory, pattern, recursive=False):
    directory = os.path.abspath(directory)
    if recursive and '**' not in pattern:
        pattern = f'**/{pattern}'
    return sorted(
        glob.glob(os.path.join(directory, pattern), recursive=recursive))


def _extract_shard_key(path_or_name):
    name = os.path.basename(path_or_name)
    m = re.search(r'(\d+)-of-(\d+)\.safetensors$', name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _build_file_map(files, base_dir, match_mode):
    if match_mode == 'path':
        return {os.path.relpath(p, base_dir): p for p in files}, []

    if match_mode == 'basename':
        collisions = []
        m = {}
        for p in files:
            key = os.path.basename(p)
            if key in m:
                collisions.append(key)
            else:
                m[key] = p
        return m, collisions

    if match_mode == 'shard':
        collisions = []
        m = {}
        for p in files:
            shard_key = _extract_shard_key(p)
            if shard_key is None:
                continue
            shard_idx, _ = shard_key
            if shard_idx in m:
                collisions.append(str(shard_idx))
            else:
                m[shard_idx] = p
        return m, collisions

    raise ValueError(f'Unknown match mode: {match_mode}')


def _compare_dir_pair(source_dir,
                      target_dir,
                      pattern='*.safetensors',
                      recursive=False,
                      tolerance=1e-5,
                      verbose=False,
                      only_diff=False,
                      match_mode='path',
                      jobs=1,
                      inner_jobs=1):
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)

    print(
        f"Scanning files in {source_dir} and {target_dir} with pattern='{pattern}' recursive={recursive}..."
    )
    source_files = _iter_safetensors_files(source_dir, pattern, recursive)
    target_files = _iter_safetensors_files(target_dir, pattern, recursive)
    source_map, source_collisions = _build_file_map(source_files, source_dir,
                                                    match_mode)
    target_map, target_collisions = _build_file_map(target_files, target_dir,
                                                    match_mode)

    if source_collisions or target_collisions:
        print(
            'Warning: duplicate match keys detected; results may be incomplete.'
        )
        if source_collisions:
            print(
                f'  Source duplicate keys (showing up to 10): {source_collisions[:10]}'
            )
        if target_collisions:
            print(
                f'  Target duplicate keys (showing up to 10): {target_collisions[:10]}'
            )

    all_names = sorted(set(source_map.keys()) | set(target_map.keys()))

    if not all_names:
        return 0, ''

    print(f'Found {len(all_names)} files to compare.')

    exit_code = 0
    results = {}

    # Identify pairs to compare and missing files
    pairs = []
    for name in all_names:
        src = source_map.get(name)
        tgt = target_map.get(name)

        if src is None:
            exit_code = 1
            if match_mode == 'shard':
                results[
                    name] = f'\n[shard {name}] Missing in source dir: {source_dir}\n  target: {tgt}\n'
            else:
                results[
                    name] = f'\n[{name}] Missing in source dir: {os.path.join(source_dir, str(name))}\n'
        elif tgt is None:
            exit_code = 1
            if match_mode == 'shard':
                results[
                    name] = f'\n[shard {name}] Missing in target dir: {target_dir}\n  source: {src}\n'
            else:
                results[
                    name] = f'\n[{name}] Missing in target dir: {os.path.join(target_dir, str(name))}\n'
        else:
            label = f'shard {name}' if match_mode == 'shard' else name
            pairs.append((name, src, tgt, label))

    if pairs:
        print(f'Starting comparison with {jobs} processes...')
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut_to_name = {
                ex.submit(_compare_safetensors_pair, src, tgt, tolerance,
                          verbose, only_diff, label, inner_jobs): name
                for name, src, tgt, label in pairs
            }

            for fut in _tqdm_iter(as_completed(fut_to_name),
                                  total=len(pairs),
                                  desc='Comparing'):
                name = fut_to_name[fut]
                try:
                    code, out = fut.result()
                    if code != 0:
                        exit_code = 1
                    results[name] = out
                except Exception as e:
                    exit_code = 1
                    results[
                        name] = f'\n[{name}] Error during comparison: {e}\n'

    # Assemble final output in deterministic order
    final_output = []
    for name in sorted(all_names):
        if name in results and results[name]:
            final_output.append(results[name])

    return exit_code, ''.join(final_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two safetensors files.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', help='Path to the source safetensors file')
    group.add_argument('--source-dir',
                       help='Directory containing source safetensors files')
    parser.add_argument('--target', help='Path to the target safetensors file')
    parser.add_argument('--target-dir',
                        help='Directory containing target safetensors files')
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-5,
        help='Tolerance for floating point comparison (default: 1e-5)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print details for matching tensors too')
    parser.add_argument(
        '--only-diff',
        action='store_true',
        help='Only print mismatches (suppress summary and OK logs)')
    parser.add_argument(
        '--pattern',
        default='*.safetensors',
        help='Glob pattern for directory mode (default: *.safetensors)')
    parser.add_argument('--recursive',
                        action='store_true',
                        help='Recursively search for files in directories')
    parser.add_argument(
        '--match-mode',
        choices=['path', 'basename', 'shard'],
        default='path',
        help=
        'How to match files in directory mode: path|basename|shard (default: path)'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Number of worker processes parallelism (file mode).')
    parser.add_argument(
        '--inner-jobs',
        type=int,
        default=1,
        help='Per-file key parallelism in directory mode (default: 1)')

    args = parser.parse_args()

    if args.source_dir:
        if not args.target_dir:
            print('Error: --target-dir is required when using --source-dir')
            sys.exit(2)
        if args.jobs < 1:
            print('Error: --jobs must be >= 1')
            sys.exit(2)
        if args.inner_jobs < 1:
            print('Error: --inner-jobs must be >= 1')
            sys.exit(2)
        code, out = _compare_dir_pair(args.source_dir, args.target_dir,
                                      args.pattern, args.recursive,
                                      args.tolerance, args.verbose,
                                      args.only_diff, args.match_mode,
                                      args.jobs, args.inner_jobs)
        if out:
            print(out, end='')
        sys.exit(code)

    if not args.target:
        print('Error: --target is required when using --source')
        sys.exit(2)

    code, out = _compare_safetensors_pair(
        args.source, args.target, args.tolerance, args.verbose, args.only_diff,
        None, args.inner_jobs if args.inner_jobs != 1 else args.jobs)
    if out:
        print(out, end='')
    sys.exit(code)
