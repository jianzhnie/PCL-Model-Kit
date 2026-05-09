#!/usr/bin/env python3
"""Compare two weight map JSON files (generated vs actual checkpoint)."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Compare two weight map JSON files')
    parser.add_argument('generated', help='Generated weight map JSON')
    parser.add_argument('actual', help='Actual checkpoint weight map JSON')
    args = parser.parse_args()

    with open(args.generated) as f:
        gen = json.load(f)['weight_map']
    with open(args.actual) as f:
        act = json.load(f)['weight_map']

    g, a = set(gen), set(act)
    extra, missing = g - a, a - g
    common = g & a
    mismatches = [(k, gen[k]['shape'], act[k]['shape']) for k in sorted(common)
                  if gen[k]['shape'] != act[k]['shape']]

    print(f'Generated: {len(g)} weights')
    print(f'Actual:    {len(a)} weights')
    print(f'Common:    {len(common)} weights')
    print()

    if extra:
        print(f'Extra in generated ({len(extra)}):')
        for k in sorted(extra):
            print(f'  {k}  {gen[k]['shape']}')
        print()

    if missing:
        print(f'Missing from generated ({len(missing)}):')
        for k in sorted(missing):
            print(f'  {k}  {act[k]['shape']}')
        print()

    if mismatches:
        print(f'Shape mismatches ({len(mismatches)}):')
        for k, gs, ais in mismatches:
            print(f'  {k}  generated={gs}  actual={ais}')
        print()

    if not extra and not missing and not mismatches:
        print('RESULT: Perfect match!')
    else:
        print(
            f'RESULT: {len(extra)} extra, {len(missing)} missing, {len(mismatches)} shape mismatches'
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
