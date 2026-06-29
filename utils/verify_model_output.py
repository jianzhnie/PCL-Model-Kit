#!/usr/bin/env python3
"""Verify expanded model output matches original model (function-preserving).

Loads both models with transformers, runs forward passes and/or generation
on the same inputs, and compares logits and/or generated tokens.

Identity-initialized layers should produce bit-exact identical output.

Features:
  - Multiple test prompts (short, long, code, special tokens)
  - Dual mode: forward (logit comparison) + generate (token-by-token)
  - Per-token metrics in addition to aggregate statistics
  - Auto device detection (cuda > npu > cpu)
  - Structured JSON output for CI pipelines
  - Robust error handling and memory management
  - Configurable tolerance thresholds
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Monkey-patch: fix input_embeds → inputs_embeds typo in cached model code
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_longcat_ngram():
    """Fix incompatibilities in the cached LongCat N-gram model code.

    The cached model code calls create_causal_mask with all keyword args,
    but the installed version may have config and input_embeds as
    positional-only parameters. This patch converts them as needed,
    and filters out any kwargs not accepted by the installed version.
    """
    try:
        import functools
        import inspect
        from transformers.masking_utils import create_causal_mask as _orig_ccm
        import transformers.masking_utils

        _ccm_params = inspect.signature(_orig_ccm).parameters
        _ccm_names = set(_ccm_params)
        # Determine which params are positional-only
        _pos_only = [n for n, p in _ccm_params.items()
                     if p.kind == inspect.Parameter.POSITIONAL_ONLY]

        @functools.wraps(_orig_ccm)
        def _patched_ccm(*args, **kwargs):
            # Move positional-only params from kwargs to positional args
            pos_args = list(args)
            for pname in _pos_only:
                if len(pos_args) <= _pos_only.index(pname) and pname in kwargs:
                    pos_args.append(kwargs.pop(pname))
            # Filter to only kwargs the function actually accepts
            filtered = {k: v for k, v in kwargs.items() if k in _ccm_names}
            return _orig_ccm(*pos_args, **filtered)

        transformers.masking_utils.create_causal_mask = _patched_ccm
    except Exception:
        import warnings
        warnings.warn(
            "LongCat ngram model patch failed; "
            "model inference may error if using the cached LongCat code."
        )


_patch_longcat_ngram()


# ═══════════════════════════════════════════════════════════════════════════════
# Device detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_device(preferred: str | None = None) -> str:
    """Auto-detect the best available device.

    Priority: preferred > cuda > npu > cpu
    """
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# Test prompts
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    # Simple English
    "Hello, this is a test of model output consistency.",
    # Chinese (multi-byte chars)
    "你好，这是一段用于测试模型一致性的文本。",
    # Code
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return",
    # Mixed content
    "The model has 14 layers and 8 experts per layer. Configuration:",
    # Empty/short input
    "Hi",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ForwardResult:
    """Result of a forward-pass comparison."""
    prompt: str
    num_tokens: int
    exact_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    cos_sim: float
    per_token_max_diff: list[float] = field(default_factory=list)
    shape_match: bool = True
    orig_shape: tuple = ()
    exp_shape: tuple = ()


@dataclass
class GenerateResult:
    """Result of a generation comparison."""
    prompt: str
    match: bool
    orig_tokens: int
    exp_tokens: int
    orig_text: str = ""
    exp_text: str = ""
    first_divergence: int | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Result builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_generate_result(
    prompt: str, orig_new: torch.Tensor, exp_new: torch.Tensor,
    tokenizer,
) -> GenerateResult:
    """Build a GenerateResult by comparing two generated token ID sequences."""
    match = torch.equal(orig_new, exp_new)
    first_div = None
    if not match:
        min_len = min(len(orig_new), len(exp_new))
        for pos in range(min_len):
            if orig_new[pos] != exp_new[pos]:
                first_div = pos
                break
        if first_div is None:
            first_div = min_len  # length mismatch

    return GenerateResult(
        prompt=prompt,
        match=match,
        orig_tokens=len(orig_new),
        exp_tokens=len(exp_new),
        orig_text=tokenizer.decode(orig_new, skip_special_tokens=True),
        exp_text=tokenizer.decode(exp_new, skip_special_tokens=True),
        first_divergence=first_div,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════════════════════

def empty_cache(device: str):
    """Clear device cache."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def _get_memory_usage() -> str:
    """Return peak RSS memory usage as a human-readable string (from getrusage)."""
    try:
        import resource
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        return f"{peak_rss_gb:.1f} GB"
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_stale_shards(exp_dir: str, dry_run: bool = False) -> tuple[int, int]:
    """Remove safetensors shard files not referenced by model.safetensors.index.json.

    Args:
        exp_dir: Path to expanded model directory.
        dry_run: If True, only report without deleting.

    Returns:
        (num_deleted, bytes_freed)
    """
    index_path = os.path.join(exp_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"  WARNING: no index file found at {index_path}, "
              f"cannot determine active shards")
        return (0, 0)

    with open(index_path) as f:
        index = json.load(f)
    active_shards = set(index.get("weight_map", {}).values())

    all_safetensors = [f for f in os.listdir(exp_dir)
                       if f.startswith("model-") and f.endswith(".safetensors")]
    stale = sorted(set(all_safetensors) - active_shards)

    if not stale:
        print("  No stale shard files found (directory is clean)")
        return (0, 0)

    total_bytes = sum(os.path.getsize(os.path.join(exp_dir, f)) for f in stale)

    # Report
    print(f"  Found {len(stale)} stale shard files "
          f"({total_bytes / (1024 ** 3):.1f} GB)")
    if dry_run:
        print("  (dry-run, no files deleted)")
        for fname in stale[:5]:  # Show first 5 as sample
            sz = os.path.getsize(os.path.join(exp_dir, fname))
            print(f"    Would delete: {fname} ({sz / (1024 ** 3):.1f} GB)")
        if len(stale) > 5:
            print(f"    ... and {len(stale) - 5} more")
        return (len(stale), total_bytes)

    # Actually delete
    deleted = 0
    freed = 0
    for fname in stale:
        fpath = os.path.join(exp_dir, fname)
        file_sz = os.path.getsize(fpath)
        os.remove(fpath)
        deleted += 1
        freed += file_sz

    print(f"  Deleted {deleted} files, freed {freed / (1024 ** 3):.1f} GB")
    return (deleted, freed)


# ═══════════════════════════════════════════════════════════════════════════════
# Core verification
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_dir: str, device: str, dtype: torch.dtype,
                             seed: int | None = None,
                             load_tokenizer: bool = True):
    """Load a tokenizer and model from a directory.

    Returns (model, tokenizer). Caller must call model.eval() if needed.

    If seed is provided, torch.manual_seed(seed) is called before loading
    so that any randomly-initialized MISSING parameters are deterministic.
    Set load_tokenizer=False to skip tokenizer loading when caller
    already has one.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = None
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )

    # For NPU/CUDA, use "auto" to shard across all devices via accelerate.
    # For CPU, "auto" works but we pass "cpu" directly.
    _device_map = "auto" if device in ("npu", "cuda") else device

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        trust_remote_code=True,
        device_map=_device_map,
    )
    model.eval()

    return model, tokenizer


def _tokenize_prompts(tokenizer, prompts: list[str], max_tokens: int):
    """Tokenize all prompts at once. Returns list of (prompt, input_ids)."""
    tokenized = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][:, :max_tokens]
        tokenized.append((prompt, input_ids))
    return tokenized


def run_forward_sequential(
    orig_dir: str, exp_dir: str, tokenizer,
    prompts: list[str], device: str, dtype: torch.dtype,
    max_tokens: int, atol: float = 1e-5,
    seed: int | None = None,
):
    """Forward comparison loading one model at a time (memory-efficient).

    Loads orig model → runs all prompts → saves logits → unloads →
    loads exp model → runs all prompts → compares.

    Best for CPU inference with large models where loading two
    models simultaneously is infeasible.
    """
    tokenized = _tokenize_prompts(tokenizer, prompts, max_tokens)

    # ── Orig pass ───────────────────────────────────────────────────────
    print(f"  [Sequential] Loading original model: {orig_dir}")
    t0 = time.time()
    orig_model, _ = load_model_and_tokenizer(
        orig_dir, device, dtype, seed=seed, load_tokenizer=False,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

    orig_logits_list: list[torch.Tensor] = []
    for i, (prompt, input_ids) in enumerate(tokenized):
        num_tokens = input_ids.shape[1]
        print(f"    [{i + 1}/{len(tokenized)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\" "
              f"({num_tokens} tokens)")
        with torch.no_grad():
            out = orig_model(input_ids.to(device))
        orig_logits_list.append(out.logits.cpu())

    del orig_model
    gc.collect()
    empty_cache(device)

    # ── Exp pass ────────────────────────────────────────────────────────
    print(f"  [Sequential] Loading expanded model: {exp_dir}")
    t0 = time.time()
    exp_model, _ = load_model_and_tokenizer(
        exp_dir, device, dtype, seed=seed, load_tokenizer=False,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

    results: list[ForwardResult] = []
    for i, (prompt, input_ids) in enumerate(tokenized):
        num_tokens = input_ids.shape[1]
        print(f"    [{i + 1}/{len(tokenized)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\" "
              f"({num_tokens} tokens)")
        with torch.no_grad():
            out = exp_model(input_ids.to(device))
        exp_logits = out.logits.cpu()
        orig_logits = orig_logits_list[i]

        if orig_logits.shape != exp_logits.shape:
            print(f"      ❌ Shape mismatch: {orig_logits.shape} vs {exp_logits.shape}")
            results.append(ForwardResult(
                prompt=prompt, num_tokens=num_tokens,
                exact_match=False, max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"), cos_sim=0.0,
                shape_match=False,
                orig_shape=tuple(orig_logits.shape),
                exp_shape=tuple(exp_logits.shape),
            ))
            continue

        metrics = _compare_logits(orig_logits, exp_logits)
        result = ForwardResult(
            prompt=prompt, num_tokens=num_tokens,
            shape_match=True,
            orig_shape=tuple(orig_logits.shape),
            exp_shape=tuple(exp_logits.shape),
            **metrics,
        )
        results.append(result)

        if result.exact_match:
            print(f"      ✅ Bit-exact match")
        elif result.max_abs_diff < atol:
            print(f"      ✅ Numerical match (max_diff={result.max_abs_diff:.2e})")
        else:
            print(f"      ❌ Mismatch! max_diff={result.max_abs_diff:.2e} "
                  f"mean_diff={result.mean_abs_diff:.2e}")

    del exp_model
    gc.collect()
    orig_logits_list.clear()
    empty_cache(device)

    return results


def _compare_logits(orig_logits: torch.Tensor, exp_logits: torch.Tensor) -> dict:
    """Compare two logit tensors and return a dict of metrics."""
    exact_match = torch.equal(orig_logits, exp_logits)
    abs_diff = (orig_logits.float() - exp_logits.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Cosine similarity along vocab dim (per-position, then averaged)
    # Avoids OOM from flattening the entire [batch, seq, vocab] tensor
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_logits.float(), exp_logits.float(), dim=-1,
    ).mean().item()

    # Per-token max diff (along the sequence dimension)
    # Shape: [batch, seq_len, vocab] → per_seq_position: [seq_len]
    if orig_logits.dim() >= 2:
        seq_dim = 1 if orig_logits.dim() == 3 else 0
        per_token = abs_diff.amax(dim=tuple(
            i for i in range(orig_logits.dim()) if i != seq_dim
        )).tolist()
    else:
        per_token = [abs_diff.item()]

    return {
        "exact_match": exact_match,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "cos_sim": cos_sim,
        "per_token_max_diff": per_token,
    }


def run_forward_compare(
    orig_model, exp_model, tokenizer, prompts: list[str],
    device: str, max_tokens: int, atol: float = 1e-5,
) -> list[ForwardResult]:
    """Run forward-pass comparison on multiple prompts.

    Returns a list of ForwardResult, one per prompt.
    """
    results: list[ForwardResult] = []

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][:, :max_tokens]
        num_tokens = input_ids.shape[1]
        print(f"  [{i + 1}/{len(prompts)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\" "
              f"({num_tokens} tokens)")

        with torch.no_grad():
            orig_out = orig_model(input_ids.to(device))
            exp_out = exp_model(input_ids.to(device))

        orig_logits = orig_out.logits.cpu()
        exp_logits = exp_out.logits.cpu()

        shape_match = orig_logits.shape == exp_logits.shape
        if not shape_match:
            results.append(ForwardResult(
                prompt=prompt, num_tokens=num_tokens,
                exact_match=False, max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"), cos_sim=0.0,
                shape_match=False,
                orig_shape=tuple(orig_logits.shape),
                exp_shape=tuple(exp_logits.shape),
            ))
            print(f"    ❌ Shape mismatch: {orig_logits.shape} vs {exp_logits.shape}")
            continue

        metrics = _compare_logits(orig_logits, exp_logits)
        result = ForwardResult(
            prompt=prompt, num_tokens=num_tokens,
            shape_match=True,
            orig_shape=tuple(orig_logits.shape),
            exp_shape=tuple(exp_logits.shape),
            **metrics,
        )
        results.append(result)

        if result.exact_match:
            print(f"    ✅ Bit-exact match")
        elif result.max_abs_diff < atol:
            print(f"    ✅ Numerical match (max_diff={result.max_abs_diff:.2e})")
        else:
            print(f"    ❌ Mismatch! max_diff={result.max_abs_diff:.2e} "
                  f"mean_diff={result.mean_abs_diff:.2e}")

    return results


def run_generate_compare(
    orig_model, exp_model, tokenizer, prompts: list[str],
    device: str, max_new_tokens: int, temperature: float = 0.0,
    do_sample: bool = False,
) -> list[GenerateResult]:
    """Run token-by-token generation comparison.

    Uses greedy decoding (temperature=0, do_sample=False) by default
    for deterministic comparison. Non-zero temperature with a fixed seed
    can also be used.

    Returns a list of GenerateResult, one per prompt.
    """
    results: list[GenerateResult] = []

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        print(f"  [{i + 1}/{len(prompts)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")

        with torch.no_grad():
            orig_ids = orig_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            exp_ids = exp_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        orig_new = orig_ids[0, input_ids.shape[1]:]  # strip prompt
        exp_new = exp_ids[0, input_ids.shape[1]:]

        result = _build_generate_result(prompt, orig_new.cpu(), exp_new.cpu(), tokenizer)
        results.append(result)

        if result.match:
            print(f"    ✅ Identical generation ({result.orig_tokens} tokens)")
        else:
            print(f"    ❌ Divergence at token {result.first_divergence}: "
                  f"orig={result.orig_tokens}t, exp={result.exp_tokens}t")
            print(f"      Orig: \"{result.orig_text[:80]}\"")
            print(f"      Exp:  \"{result.exp_text[:80]}\"")

    return results


def run_generate_sequential(
    orig_dir: str, exp_dir: str, tokenizer,
    prompts: list[str], device: str, dtype: torch.dtype,
    max_new_tokens: int, temperature: float = 0.0,
    do_sample: bool = False,
    seed: int | None = None,
) -> list[GenerateResult]:
    """Generation comparison loading one model at a time (memory-efficient).

    Loads orig model → generates all prompts → saves IDs → unloads →
    loads exp model → generates all prompts → compares.
    """
    # Tokenize all prompts once
    tokenized = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        tokenized.append((prompt, inputs["input_ids"].to(device)))

    # ── Orig pass ───────────────────────────────────────────────────────
    print(f"  [Sequential] Loading original model: {orig_dir}")
    t0 = time.time()
    orig_model, _ = load_model_and_tokenizer(
        orig_dir, device, dtype, seed=seed, load_tokenizer=False,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

    orig_new_list: list[torch.Tensor] = []
    for i, (prompt, input_ids) in enumerate(tokenized):
        print(f"    [{i + 1}/{len(tokenized)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        with torch.no_grad():
            orig_ids = orig_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        orig_new_list.append(orig_ids[0, input_ids.shape[1]:].cpu())

    del orig_model
    gc.collect()
    empty_cache(device)

    # ── Exp pass ────────────────────────────────────────────────────────
    print(f"  [Sequential] Loading expanded model: {exp_dir}")
    t0 = time.time()
    exp_model, _ = load_model_and_tokenizer(
        exp_dir, device, dtype, seed=seed, load_tokenizer=False,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

    results: list[GenerateResult] = []
    for i, (prompt, input_ids) in enumerate(tokenized):
        print(f"    [{i + 1}/{len(tokenized)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        with torch.no_grad():
            exp_ids = exp_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        exp_new = exp_ids[0, input_ids.shape[1]:].cpu()
        orig_new = orig_new_list[i]
        result = _build_generate_result(prompt, orig_new, exp_new, tokenizer)
        results.append(result)

        if result.match:
            print(f"    ✅ Identical generation ({result.orig_tokens} tokens)")
        else:
            print(f"    ❌ Divergence at token {result.first_divergence}: "
                  f"orig={result.orig_tokens}t, exp={result.exp_tokens}t")
            print(f"      Orig: \"{result.orig_text[:80]}\"")
            print(f"      Exp:  \"{result.exp_text[:80]}\"")

    del exp_model
    gc.collect()
    orig_new_list.clear()
    empty_cache(device)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _print_forward_report(results: list[ForwardResult], atol: float = 1e-5):
    """Pretty-print forward comparison results."""
    n = len(results)
    n_exact = sum(1 for r in results if r.exact_match)
    n_shape_mismatch = sum(1 for r in results if not r.shape_match)
    n_numerical = sum(
        1 for r in results if r.shape_match and not r.exact_match and r.max_abs_diff < atol
    )
    n_mismatch = n - n_exact - n_numerical - n_shape_mismatch

    print(f"\n{'─' * 60}")
    print(f"  Forward Pass Summary ({n} prompts)")
    print(f"{'─' * 60}")
    print(f"  Bit-exact match:      {n_exact}/{n}")
    print(f"  Numerical match:      {n_numerical}/{n}")
    print(f"  Shape mismatch:       {n_shape_mismatch}/{n}")
    print(f"  Significant mismatch: {n_mismatch}/{n}")

    if results:
        max_diffs = [r.max_abs_diff for r in results if r.shape_match]
        if max_diffs:
            print(f"\n  Worst max_abs_diff:   {max(max_diffs):.2e}")
            print(f"  Best max_abs_diff:    {min(max_diffs):.2e}")

        # Per-token detail for the worst prompt
        worst = max(
            (r for r in results if r.shape_match),
            key=lambda r: r.max_abs_diff,
        )
        if worst.per_token_max_diff and len(worst.per_token_max_diff) > 1:
            print(f"\n  Per-token max diff (worst prompt, {len(worst.per_token_max_diff)} tokens):")
            for pos, d in enumerate(worst.per_token_max_diff):
                marker = " ⚠" if d > atol else ""
                print(f"    token {pos:4d}: {d:.2e}{marker}")

    print(f"{'─' * 60}")


def _print_generate_report(results: list[GenerateResult]):
    """Pretty-print generation comparison results."""
    n = len(results)
    n_match = sum(1 for r in results if r.match)

    print(f"\n{'─' * 60}")
    print(f"  Generation Summary ({n} prompts)")
    print(f"{'─' * 60}")
    print(f"  Identical:  {n_match}/{n}")
    print(f"  Divergent:  {n - n_match}/{n}")

    if n - n_match > 0:
        print(f"\n  Divergent cases:")
        for r in results:
            if not r.match:
                print(f"    Prompt: \"{r.prompt[:60]}...\"")
                print(f"    First divergence at token: {r.first_divergence}")
                print(f"    Orig ({r.orig_tokens}t): \"{r.orig_text[:100]}\"")
                print(f"    Exp  ({r.exp_tokens}t):  \"{r.exp_text[:100]}\"")
                print()

    print(f"{'─' * 60}")


def _results_to_json(results_forward, results_generate) -> dict:
    """Convert results to a JSON-serializable dict."""
    data: dict = {}

    if results_forward:
        data["forward"] = []
        for r in results_forward:
            entry = {
                "prompt": r.prompt,
                "num_tokens": r.num_tokens,
                "exact_match": r.exact_match,
                "max_abs_diff": r.max_abs_diff,
                "mean_abs_diff": r.mean_abs_diff,
                "cos_sim": r.cos_sim,
                "shape_match": r.shape_match,
                "orig_shape": list(r.orig_shape),
                "exp_shape": list(r.exp_shape),
            }
            if r.per_token_max_diff:
                entry["per_token_max_diff"] = r.per_token_max_diff
            data["forward"].append(entry)

    if results_generate:
        data["generate"] = []
        for r in results_generate:
            data["generate"].append({
                "prompt": r.prompt,
                "match": r.match,
                "orig_tokens": r.orig_tokens,
                "exp_tokens": r.exp_tokens,
                "orig_text": r.orig_text,
                "exp_text": r.exp_text,
                "first_divergence": r.first_divergence,
            })

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify expanded model output consistency "
                    "(forward pass + generation comparison)"
    )
    parser.add_argument("--orig_dir", required=True,
                        help="Path to original model directory")
    parser.add_argument("--exp_dir", required=True,
                        help="Path to expanded model directory")
    parser.add_argument("--device", default="auto",
                        help="Device: auto (detect), cuda, npu, or cpu")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Max input tokens for forward pass")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max tokens to generate")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (float32 recommended for CPU)")
    parser.add_argument("--mode", default="all",
                        choices=["forward", "generate", "all"],
                        help="Verification mode")
    parser.add_argument("--prompts", default=None,
                        help="Custom prompts file (one per line, or JSON array)")
    parser.add_argument("--prompt", default=None, action="append",
                        dest="extra_prompts",
                        help="Additional prompt (repeatable)")
    parser.add_argument("--json_output", default=None,
                        help="Write results as JSON to this file")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for numerical match")
    parser.add_argument("--sequential", action="store_true", default=None,
                        help="Load one model at a time (use for CPU/large models)")
    parser.add_argument("--no-sequential", action="store_false", dest="sequential",
                        help="Load both models simultaneously (GPU only)")
    parser.add_argument("--skip_generate_if_forward_fails", action="store_true",
                        default=True,
                        help="Skip generation if forward check fails")
    parser.add_argument("--no_skip_generate", action="store_false",
                        dest="skip_generate_if_forward_fails",
                        help="Always run generation even if forward fails")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic MISSING param init (default: 42)")
    parser.add_argument("--cleanup-stale-shards", action="store_true", default=False,
                        help="Remove stale safetensors shards not referenced by "
                             "model.safetensors.index.json in --exp_dir")
    args = parser.parse_args()

    # ── Cleanup stale shards (if requested) ──────────────────────────────
    if args.cleanup_stale_shards:
        print(f"Checking for stale shard files in: {args.exp_dir}")
        _cleanup_stale_shards(args.exp_dir, dry_run=False)
        print()

    # ── Device ──────────────────────────────────────────────────────────
    device = _detect_device(args.device)
    is_cpu = (device == "cpu")

    # Default to sequential on CPU, simultaneous on GPU
    if args.sequential is None:
        args.sequential = is_cpu

    print(f"Device: {device}")
    print(f"Mode: {'sequential' if args.sequential else 'simultaneous'} (one model at a time)")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    if device == "cpu" and dtype != torch.float32:
        print(f"Note: dtype={args.dtype} on CPU. Using float32 for stability.")
        dtype = torch.float32

    # ── Prompts ─────────────────────────────────────────────────────────
    prompts = list(DEFAULT_PROMPTS)
    if args.prompts:
        prompts_path = args.prompts
        if os.path.exists(prompts_path):
            with open(prompts_path) as f:
                content = f.read().strip()
            try:
                prompts = json.loads(content)
            except json.JSONDecodeError:
                prompts = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            print(f"WARNING: prompts file not found: {prompts_path}")
    if args.extra_prompts:
        prompts.extend(args.extra_prompts)

    print(f"Test prompts: {len(prompts)}")

    # ── Load tokenizer once ─────────────────────────────────────────────
    print(f"\nLoading tokenizer from: {args.orig_dir}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.orig_dir, trust_remote_code=True
    )

    results_forward: list[ForwardResult] = []
    results_generate: list[GenerateResult] = []
    exit_code = 0

    try:
        # ── Forward comparison ──────────────────────────────────────────
        if args.mode in ("forward", "all"):
            print(f"\n{'=' * 60}")
            print("  FORWARD PASS COMPARISON")
            print(f"{'=' * 60}")

            if args.sequential:
                results_forward = run_forward_sequential(
                    args.orig_dir, args.exp_dir, tokenizer,
                    prompts, device, dtype, args.max_tokens, args.atol,
                    seed=args.seed,
                )
            else:
                print(f"Loading original model:  {args.orig_dir}")
                t0 = time.time()
                orig_model, _ = load_model_and_tokenizer(
                    args.orig_dir, device, dtype, seed=args.seed,
                    load_tokenizer=False,
                )
                print(f"  Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

                print(f"Loading expanded model:  {args.exp_dir}")
                t0 = time.time()
                exp_model, _ = load_model_and_tokenizer(
                    args.exp_dir, device, dtype, seed=args.seed,
                    load_tokenizer=False,
                )
                print(f"  Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

                results_forward = run_forward_compare(
                    orig_model, exp_model, tokenizer,
                    prompts, device, args.max_tokens, atol=args.atol,
                )
                del orig_model
                del exp_model
                gc.collect()

            _print_forward_report(results_forward, atol=args.atol)

            has_mismatch = any(
                not r.shape_match or (not r.exact_match and r.max_abs_diff >= args.atol)
                for r in results_forward
            )
            if has_mismatch:
                exit_code = 1

        # ── Generation comparison ───────────────────────────────────────
        if args.mode in ("generate", "all"):
            skip = (
                args.skip_generate_if_forward_fails
                and results_forward
                and any(not r.exact_match and r.max_abs_diff >= args.atol
                        for r in results_forward)
            )
            if skip:
                print("\n⚠ Skipping generation: forward pass already showed mismatches")
            else:
                print(f"\n{'=' * 60}")
                print("  GENERATION COMPARISON (greedy)")
                print(f"{'=' * 60}")

                if args.sequential:
                    results_generate = run_generate_sequential(
                        args.orig_dir, args.exp_dir, tokenizer,
                        prompts, device, dtype, args.max_new_tokens,
                        seed=args.seed,
                    )
                else:
                    print(f"Loading original model:  {args.orig_dir}")
                    t0 = time.time()
                    orig_model, _ = load_model_and_tokenizer(
                        args.orig_dir, device, dtype, seed=args.seed,
                        load_tokenizer=False,
                    )
                    print(f"  Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

                    print(f"Loading expanded model:  {args.exp_dir}")
                    t0 = time.time()
                    exp_model, _ = load_model_and_tokenizer(
                        args.exp_dir, device, dtype, seed=args.seed,
                        load_tokenizer=False,
                    )
                    print(f"  Loaded in {time.time() - t0:.1f}s (mem: {_get_memory_usage()})")

                    results_generate = run_generate_compare(
                        orig_model, exp_model, tokenizer,
                        prompts, device, args.max_new_tokens,
                    )
                    del orig_model
                    del exp_model
                    gc.collect()

                _print_generate_report(results_generate)

                if any(not r.match for r in results_generate):
                    exit_code = 1

    finally:
        empty_cache(device)
        gc.collect()

    # ── JSON output ─────────────────────────────────────────────────────
    if args.json_output:
        data = _results_to_json(results_forward, results_generate)
        data["exit_code"] = exit_code
        data["device"] = device
        data["dtype"] = args.dtype
        data["sequential"] = args.sequential
        with open(args.json_output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {args.json_output}")

    # ── Final verdict ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print("  ✅ ALL CHECKS PASSED — outputs are consistent!")
    else:
        print("  ❌ VERIFICATION FAILED — outputs differ!")
    print(f"{'=' * 60}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
