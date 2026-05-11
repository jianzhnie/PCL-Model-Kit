#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import torch
from safetensors import safe_open


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with color support and simplified INFO logging.
    """
    grey = '\x1b[38;20m'
    green = '\x1b[32;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    # Detailed format for Debug/Error
    detailed_format = '%(asctime)s - %(levelname)s - %(message)s'
    # Simplified format for Info (just the message, like print)
    info_format = '%(message)s'

    FORMATS = {
        logging.DEBUG: grey + detailed_format + reset,
        logging.INFO: grey + info_format + reset,
        logging.WARNING: yellow + detailed_format + reset,
        logging.ERROR: red + detailed_format + reset,
        logging.CRITICAL: bold_red + detailed_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logger(verbose=False):
    """
    Setup a logger with a custom formatter and sys.stdout output.
    """
    # Get the logger for this module
    logger = logging.getLogger(__name__)

    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO if not verbose else logging.DEBUG)

    # Use sys.stdout for standard output to match 'print' behavior
    # This ensures piped output works as expected
    handler = logging.StreamHandler(sys.stdout)

    # Set the custom formatter
    handler.setFormatter(CustomFormatter())

    logger.addHandler(handler)

    # Propagate set to False to prevent double logging if root logger is configured
    logger.propagate = False

    return logger


# Initialize logger at module level
logger = logging.getLogger(__name__)


def compare_safetensors(file1_path,
                        file2_path,
                        tolerance=1e-5,
                        verbose=False,
                        key=None):
    """
    Compare two safetensors files and report differences.
    Uses safe_open for memory-efficient loading.
    """
    if not os.path.exists(file1_path):
        logger.error(f'Error: File not found: {file1_path}')
        sys.exit(1)
    if not os.path.exists(file2_path):
        logger.error(f'Error: File not found: {file2_path}')
        sys.exit(1)

    logger.info(f'Comparing:\n  File 1: {file1_path}\n  File 2: {file2_path}')

    try:
        # Open files context managers
        f1 = safe_open(file1_path, framework='pt', device='cpu')
        f2 = safe_open(file2_path, framework='pt', device='cpu')
    except Exception as e:
        logger.error(f'Error opening safetensors files: {e}')
        sys.exit(1)

    keys1 = set(f1.keys())
    keys2 = set(f2.keys())

    if key is not None:
        if key not in keys1 and key not in keys2:
            logger.error(f'\n[MISMATCH] Key missing in both files:\n  - {key}')
            sys.exit(1)
        if key not in keys1:
            val = f2.get_tensor(key)
            logger.error(
                f'\n[MISMATCH] Key present in File 2 but missing in File 1:\n  - {key}'
            )
            logger.error(f'    Shape: {val.shape}')
            logger.error(f'    Value: {val}')
            sys.exit(1)
        if key not in keys2:
            val = f1.get_tensor(key)
            logger.error(
                f'\n[MISMATCH] Key present in File 1 but missing in File 2:\n  - {key}'
            )
            logger.error(f'    Shape: {val.shape}')
            logger.error(f'    Value: {val}')
            sys.exit(1)

        missing_in_2 = set()
        missing_in_1 = set()
        common_keys = {key}
    else:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        common_keys = keys1.intersection(keys2)

    if key is None and missing_in_2:
        logger.warning(
            f'\nKeys present in File 1 but missing in File 2 ({len(missing_in_2)}):'
        )
        for k in sorted(list(missing_in_2))[:10]:
            val = f1.get_tensor(k)
            logger.warning(f'  - {k}')
            logger.warning(f'    Shape: {val.shape}')
            logger.warning(f'    Value: {val}')
        if len(missing_in_2) > 10:
            logger.warning(f'  ... and {len(missing_in_2) - 10} more.')

    if key is None and missing_in_1:
        logger.warning(
            f'\nKeys present in File 2 but missing in File 1 ({len(missing_in_1)}):'
        )
        for k in sorted(list(missing_in_1))[:10]:
            val = f2.get_tensor(k)
            logger.warning(f'  - {k}')
            logger.warning(f'    Shape: {val.shape}')
            logger.warning(f'    Value: {val}')
        if len(missing_in_1) > 10:
            logger.warning(f'  ... and {len(missing_in_1) - 10} more.')

    logger.info(f'\nComparing {len(common_keys)} common tensors...')

    diff_count = 0
    checked_count = 0

    for tensor_key in sorted(list(common_keys)):
        # Load tensors only when needed (memory efficient)
        tensor1 = f1.get_tensor(tensor_key)
        tensor2 = f2.get_tensor(tensor_key)
        checked_count += 1

        # Check shape
        if tensor1.shape != tensor2.shape:
            logger.error(f"\n[MISMATCH] Shape mismatch for '{tensor_key}':")
            logger.error(f'  File 1: {tensor1.shape}')
            logger.error(f'  File 2: {tensor2.shape}')
            logger.error(f'  File 1 value: {tensor1}')
            logger.error(f'  File 2 value: {tensor2}')
            diff_count += 1
            continue

        # Check dtype
        if tensor1.dtype != tensor2.dtype:
            logger.error(f"\n[MISMATCH] Dtype mismatch for '{tensor_key}':")
            logger.error(f'  File 1: {tensor1.dtype}')
            logger.error(f'  File 2: {tensor2.dtype}')
            logger.error(f'  File 1 value: {tensor1}')
            logger.error(f'  File 2 value: {tensor2}')
            diff_count += 1
            continue

        # Check values
        if not torch.allclose(tensor1, tensor2, atol=tolerance,
                              rtol=tolerance):
            diff = (tensor1 - tensor2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            max_diff_idx = torch.argmax(diff).item()
            v1 = tensor1.view(-1)[max_diff_idx].item()
            v2 = tensor2.view(-1)[max_diff_idx].item()

            logger.error(f"\n[MISMATCH] Value mismatch for '{tensor_key}':")
            logger.error(f'  Max diff: {max_diff:.6e}')
            logger.error(f'  Mean diff: {mean_diff:.6e}')
            logger.error(f'  At flat index {max_diff_idx}: {v1} vs {v2}')
            logger.error(f'  File 1 value: {tensor1}')
            logger.error(f'  File 2 value: {tensor2}')
            diff_count += 1
        elif verbose:
            logger.info(f'[OK] {tensor_key}')

    logger.info('\n' + '=' * 50)
    logger.info('Comparison Summary:')
    logger.info(f'  Total keys checked: {checked_count}')
    logger.info(f'  Mismatched tensors: {diff_count}')
    if missing_in_1 or missing_in_2:
        logger.warning(
            f'  Missing keys: {len(missing_in_1) + len(missing_in_2)}')

    if diff_count == 0 and not missing_in_1 and not missing_in_2:
        logger.info('\nSUCCESS: Files are identical!')
        sys.exit(0)
    else:
        logger.error('\nFAILURE: Files differ.')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two safetensors files.')
    parser.add_argument('--source',
                        required=True,
                        help='Path to the source safetensors file')
    parser.add_argument('--target',
                        required=True,
                        help='Path to the target safetensors file')
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-5,
        help='Tolerance for floating point comparison (default: 1e-5)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print details for matching tensors too')
    parser.add_argument('--key',
                        default=None,
                        help='Only compare a single tensor key (exact match)')

    args = parser.parse_args()

    # Setup logging configuration
    setup_logger(args.verbose)

    compare_safetensors(args.source,
                        args.target,
                        args.tolerance,
                        args.verbose,
                        key=args.key)
