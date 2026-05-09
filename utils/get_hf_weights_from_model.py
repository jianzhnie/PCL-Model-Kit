#!/usr/bin/env python3
"""
Extract weight names and shapes from model definition using meta device.
No GPU/memory required — instantiates the model on a zero-memory meta device
and reads parameter shapes directly from state_dict.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DTYPE_BYTES = {'float32': 4, 'float16': 2, 'bfloat16': 2}


def main():
    parser = argparse.ArgumentParser(
        description='Extract weight info from model definition (meta device)')
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--output', '-o', type=str, default='weight_map.json')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=list(DTYPE_BYTES))
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()

    from models.configuration_deepseek import DeepseekV3Config
    from models.modeling_deepseek import DeepseekV3ForCausalLM

    # Load config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = DeepseekV3Config(**config_dict)
    else:
        config = DeepseekV3Config()

    # Force ep_size=1 so all experts are instantiated (not sharded)
    config.ep_size = 1

    print('Config:')
    for k in [
            'vocab_size', 'hidden_size', 'num_hidden_layers',
            'num_attention_heads', 'num_query_groups', 'intermediate_size',
            'moe_intermediate_size', 'n_routed_experts', 'n_shared_experts',
            'first_k_dense_replace', 'moe_layer_freq', 'kv_channels'
    ]:
        print(f'  {k}: {getattr(config, k, 'N/A')}')

    # Instantiate on meta device (zero memory allocation)
    with torch.device('meta'):
        model = DeepseekV3ForCausalLM(config)

    # Extract weight info from state_dict
    element_size = DTYPE_BYTES[args.dtype]
    weight_map = {}
    total_size = 0

    for name, param in model.state_dict().items():
        shape = list(param.shape)
        numel = 1
        for d in shape:
            numel *= d
        total_size += numel * element_size
        weight_map[name] = {'shape': shape, 'dtype': args.dtype}

    result = {
        'metadata': {
            'total_size': total_size,
            'num_weights': len(weight_map),
        },
        'weight_map': weight_map,
    }

    # Write output
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f, indent=2 if args.pretty else None)

    print(
        f'\nWeights: {len(weight_map)}, Size: {total_size / (1024**3):.2f} GB')
    print(f'Saved to: {output}')

    # Show example weights
    examples = [
        'model.embed_tokens.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.self_attn.q_layernorm.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.layers.2.mlp.gate.weight',
        'model.layers.2.mlp.experts.0.gate_proj.weight',
        'model.layers.2.mlp.shared_experts.gate_proj.weight',
        'model.norm.weight',
        'lm_head.weight',
    ]
    print('\nExample weights:')
    for name in examples:
        if name in weight_map:
            print(f'  {name}: {weight_map[name]['shape']}')


if __name__ == '__main__':
    main()
