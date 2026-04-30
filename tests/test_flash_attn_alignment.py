"""
Numerical alignment test: DeepseekV3Attention (eager) vs DeepseekV3FlashAttention.

Uses CPU with mocked NPU flash attention operators to verify data flow correctness.
On Ascend NPU hardware, replace mocks with real torch_npu ops.

Run:
    python -m tests.test_flash_attn_alignment
"""
import math
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# CPU mock for NPU flash attention operators (BSND / TND layouts with GQA)
# ---------------------------------------------------------------------------


def _mock_npu_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                              causal=False):
    """Mock npu_flash_attn_func: BSND layout [B,S,H,D], GQA supported."""
    B, Sq, H_q, D = q.shape
    Skv = k.shape[1]
    H_kv = k.shape[2]
    n_rep = H_q // H_kv

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    # GQA: expand KV heads
    if n_rep > 1:
        k = (k[:, :, :, None, :]
             .expand(B, Skv, H_kv, n_rep, D)
             .reshape(B, Skv, H_q, D))
        v = (v[:, :, :, None, :]
             .expand(B, Skv, H_kv, n_rep, D)
             .reshape(B, Skv, H_q, D))

    # Transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if causal:
        # Bottom-right aligned causal mask (handles Sq != Skv)
        mask = torch.ones(Sq, Skv, device=q.device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=Skv - Sq)
        attn = attn.masked_fill(~mask, float('-inf'))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)  # [B, Sq, H, D]


def _mock_npu_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None, causal=False):
    """Mock npu_flash_attn_varlen_func: TND layout [T,H,D], GQA supported."""
    H_q = q.shape[1]
    H_kv = k.shape[1]
    D = q.shape[2]
    n_rep = H_q // H_kv

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    output_parts = []
    for i in range(len(cu_seqlens_q) - 1):
        q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        kv_s, kv_e = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

        qi = q[q_s:q_e]    # [Sq, H_q, D]
        ki = k[kv_s:kv_e]  # [Skv, H_kv, D]
        vi = v[kv_s:kv_e]

        Sq = qi.shape[0]
        Skv = ki.shape[0]

        if n_rep > 1:
            ki = (ki[:, :, None, :]
                  .expand(Skv, H_kv, n_rep, D)
                  .reshape(Skv, H_q, D))
            vi = (vi[:, :, None, :]
                  .expand(Skv, H_kv, n_rep, D)
                  .reshape(Skv, H_q, D))

        qi = qi.transpose(0, 1)  # [H, Sq, D]
        ki = ki.transpose(0, 1)
        vi = vi.transpose(0, 1)

        attn = torch.matmul(qi, ki.transpose(-2, -1)) * softmax_scale
        if causal:
            mask = torch.ones(Sq, Skv, device=q.device, dtype=torch.bool)
            mask = torch.tril(mask, diagonal=Skv - Sq)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, vi)          # [H, Sq, D]
        output_parts.append(out.transpose(0, 1))  # [Sq, H_q, D]

    return torch.cat(output_parts, dim=0)


# ---------------------------------------------------------------------------
# Patch NPU operators with CPU mocks, then import model
# ---------------------------------------------------------------------------
import models.modeling_deepseek as _mdl

_mdl.npu_flash_attn_func = _mock_npu_flash_attn_func
_mdl.npu_flash_attn_varlen_func = _mock_npu_flash_attn_varlen_func

from models.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3FlashAttention,
    _get_unpad_data,
)
from models.configuration_deepseek import DeepseekV3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    defaults = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=32,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        max_position_embeddings=512,
        rope_theta=10000.0,
        n_routed_experts=None,
    )
    defaults.update(overrides)
    return DeepseekV3Config(**defaults)


def copy_weights(eager, flash):
    flash.load_state_dict(eager.state_dict(), strict=True)


def build_rope(seq_len, dim, device="cpu", dtype=torch.float32):
    inv_freq = 1.0 / (10000.0
                       ** (torch.arange(0, dim, 2, device=device) / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_no_padding():
    """Standard path: no padding — npu_flash_attn_func path."""
    config = make_config()
    eager = DeepseekV3Attention(config, 0).eval()
    flash = DeepseekV3FlashAttention(config, 0).eval()
    copy_weights(eager, flash)

    bsz, seq_len = 2, 64
    hidden = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
    cos, sin = build_rope(seq_len, config.kv_channels)

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (bsz, seq_len), hidden, 0)

    eager_out, _, _ = eager(hidden, attention_mask=causal_mask,
                            position_ids=position_ids,
                            position_embeddings=(cos, sin))
    flash_out, _, _ = flash(hidden, attention_mask=None,
                            position_ids=position_ids,
                            position_embeddings=(cos, sin))

    max_diff = (eager_out - flash_out).abs().max().item()
    passed = torch.allclose(eager_out, flash_out, atol=1e-5, rtol=1e-5)
    print(f"[Prefill no-padding]  max_diff={max_diff:.2e}  passed={passed}")
    return passed


@torch.no_grad()
def test_prefill_with_padding():
    """Varlen path: right-padded sequences — npu_flash_attn_varlen_func."""
    config = make_config()
    eager = DeepseekV3Attention(config, 0).eval()
    flash = DeepseekV3FlashAttention(config, 0).eval()
    copy_weights(eager, flash)

    bsz, seq_len = 4, 32
    hidden = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[0, 20:] = 0
    attention_mask[1, 28:] = 0
    attention_mask[2, 15:] = 0

    cos, sin = build_rope(seq_len, config.kv_channels)

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (bsz, seq_len), hidden, 0)

    eager_out, _, _ = eager(hidden, attention_mask=causal_mask,
                            position_ids=position_ids,
                            position_embeddings=(cos, sin))
    flash_out, _, _ = flash(hidden, attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=(cos, sin))

    # Compare only valid (non-padded) positions
    valid = attention_mask.bool().unsqueeze(-1).expand_as(eager_out)
    eager_valid = eager_out[valid]
    flash_valid = flash_out[valid]

    max_diff = (eager_valid - flash_valid).abs().max().item()
    passed = torch.allclose(eager_valid, flash_valid, atol=1e-5, rtol=1e-5)
    print(f"[Prefill w/ padding]  max_diff={max_diff:.2e}  passed={passed}")
    return passed


@torch.no_grad()
def test_decode_with_cache():
    """Decode step: single-token query with KV cache."""
    config = make_config()
    eager = DeepseekV3Attention(config, 0).eval()
    flash = DeepseekV3FlashAttention(config, 0).eval()
    copy_weights(eager, flash)

    bsz, prefill_len, decode_len = 2, 16, 1
    dim = config.kv_channels

    from transformers.cache_utils import DynamicCache

    # --- Prefill both ---
    hidden_pf = torch.randn(bsz, prefill_len, config.hidden_size)
    pos_pf = torch.arange(prefill_len).unsqueeze(0).expand(bsz, -1)
    cos_pf, sin_pf = build_rope(prefill_len, dim)

    eager_cache = DynamicCache()
    flash_cache = DynamicCache()

    _, _, eager_cache = eager(
        hidden_pf, attention_mask=None, position_ids=pos_pf,
        past_key_value=eager_cache, use_cache=True,
        position_embeddings=(cos_pf, sin_pf))
    _, _, flash_cache = flash(
        hidden_pf, attention_mask=None, position_ids=pos_pf,
        past_key_value=flash_cache, use_cache=True,
        position_embeddings=(cos_pf, sin_pf))

    # --- Decode step ---
    hidden_dec = torch.randn(bsz, decode_len, config.hidden_size)
    pos_dec = torch.tensor([[prefill_len]]).expand(bsz, -1)
    total_len = prefill_len + decode_len
    cos_dec, sin_dec = build_rope(total_len, dim)

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (bsz, decode_len), hidden_dec, prefill_len)

    eager_out, _, _ = eager(
        hidden_dec, attention_mask=causal_mask, position_ids=pos_dec,
        past_key_value=eager_cache,
        position_embeddings=(cos_dec, sin_dec))
    flash_out, _, _ = flash(
        hidden_dec, attention_mask=None, position_ids=pos_dec,
        past_key_value=flash_cache,
        position_embeddings=(cos_dec, sin_dec))

    max_diff = (eager_out - flash_out).abs().max().item()
    passed = torch.allclose(eager_out, flash_out, atol=1e-5, rtol=1e-5)
    print(f"[Decode w/ cache]     max_diff={max_diff:.2e}  passed={passed}")
    return passed


@torch.no_grad()
def test_different_gqa_ratios():
    """Various GQA head ratios: (num_heads, num_kv_groups)."""
    ratios = [(8, 2), (8, 4), (16, 2), (16, 4), (16, 8)]
    results = []

    for num_heads, kv_groups in ratios:
        config = make_config(
            num_attention_heads=num_heads,
            num_query_groups=kv_groups,
            kv_channels=64,
            hidden_size=num_heads * 64,
        )
        eager = DeepseekV3Attention(config, 0).eval()
        flash = DeepseekV3FlashAttention(config, 0).eval()
        copy_weights(eager, flash)

        bsz, seq_len = 2, 32
        hidden = torch.randn(bsz, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
        cos, sin = build_rope(seq_len, config.kv_channels)

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        causal_mask = _prepare_4d_causal_attention_mask(
            None, (bsz, seq_len), hidden, 0)

        eager_out, _, _ = eager(hidden, attention_mask=causal_mask,
                                position_ids=position_ids,
                                position_embeddings=(cos, sin))
        flash_out, _, _ = flash(hidden, attention_mask=None,
                                position_ids=position_ids,
                                position_embeddings=(cos, sin))

        max_diff = (eager_out - flash_out).abs().max().item()
        passed = torch.allclose(eager_out, flash_out, atol=1e-5, rtol=1e-5)
        results.append(passed)
        print(f"  GQA {num_heads:>2}/{kv_groups}: "
              f"max_diff={max_diff:.2e}  passed={passed}")

    all_passed = all(results)
    print(f"[GQA ratios]          all_passed={all_passed}")
    return all_passed


def main():
    print("=" * 60)
    print("DeepseekV3 Flash Attention Alignment Test (CPU mock)")
    print("=" * 60)
    results = [
        test_prefill_no_padding(),
        test_prefill_with_padding(),
        test_decode_with_cache(),
        test_different_gqa_ratios(),
    ]
    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print(f"SOME TESTS FAILED: {sum(results)}/{len(results)} passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
