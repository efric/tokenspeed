from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


def _run_mla_decode_case(
    *,
    device: str,
    solution: str,
    dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    cache_seqlens_list: list[int],
    page_size: int,
    return_lse: bool,
    max_seqlen_k: int | None = None,
) -> None:
    batch_size = len(cache_seqlens_list)
    q_len = 1
    qk_nope_head_dim = 128
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    live_max_seqlen_k = max(cache_seqlens_list)
    max_seqlen_k = live_max_seqlen_k if max_seqlen_k is None else max_seqlen_k
    if max_seqlen_k < live_max_seqlen_k:
        raise ValueError("max_seqlen_k must cover every live sequence length")
    max_pages = (max_seqlen_k + page_size - 1) // page_size
    live_max_pages = (live_max_seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * live_max_pages

    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        batch_size,
        q_len,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        kv_cache = kv_cache.to(dtype)

    cache_seqlens = torch.tensor(cache_seqlens_list, device=device, dtype=torch.int32)
    page_table = torch.zeros(
        batch_size,
        max_pages,
        device=device,
        dtype=torch.int32,
    )
    for batch_idx, seqlen in enumerate(cache_seqlens_list):
        page_count = (seqlen + page_size - 1) // page_size
        page_start = batch_idx * live_max_pages
        page_table[batch_idx, :page_count] = torch.arange(
            page_start,
            page_start + page_count,
            device=device,
            dtype=torch.int32,
        )
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    result = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        solution=solution,
    )
    if return_lse:
        out, lse = result
    else:
        assert isinstance(result, torch.Tensor)
        out = result
        lse = None

    refs = []
    ref_lses = []
    for batch_idx in range(batch_size):
        kv_rows = []
        for pos in range(int(cache_seqlens[batch_idx].item())):
            page = page_table[batch_idx, pos // page_size]
            kv_rows.append(kv_cache[page, pos % page_size, 0])
        kv = torch.stack(kv_rows).float()
        scores = torch.einsum("hd,kd->hk", q[batch_idx, 0].float(), kv)
        scores = scores * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.matmul(probs, kv[:, :kv_lora_rank]).unsqueeze(0))
        ref_lses.append(torch.logsumexp(scores, dim=-1).unsqueeze(0))
    out_ref = torch.stack(refs, dim=0)

    assert out.shape == (batch_size, q_len, num_heads, kv_lora_rank)
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    if return_lse:
        assert lse is not None
        lse_ref = torch.stack(ref_lses, dim=0)
        assert lse.shape == (batch_size, q_len, num_heads)
        torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 192, 128, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill(
    device: str,
    solution: str,
    is_causal: bool,
    dtype: torch.dtype,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    require,
) -> None:
    require("attention", "mla_prefill", solution, dtype, "q")

    q_lens = [853, 1045]
    kv_lens = q_lens
    cu_seqlens_q = torch.tensor([0, 853, 1898], device=device, dtype=torch.int32)
    cu_seqlens_kv = cu_seqlens_q
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        sum(q_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    k = torch.randn(
        sum(kv_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    v = torch.randn(
        sum(kv_lens), num_heads, v_head_dim, device=device, dtype=init_dtype
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        return_lse=True,
        solution=solution,
    )

    refs = []
    ref_lses = []
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        q_i = q[q_offset : q_offset + q_len].float()
        k_i = k[kv_offset : kv_offset + kv_len].float()
        v_i = v[kv_offset : kv_offset + kv_len].float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * softmax_scale
        if is_causal:
            q_pos = torch.arange(q_len, device=device) + max(kv_len - q_len, 0)
            k_pos = torch.arange(kv_len, device=device)
            mask = q_pos[:, None] >= k_pos[None, :]
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
        q_offset += q_len
        kv_offset += kv_len
    out_ref = torch.cat(refs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)

    assert out.shape == (q.shape[0], q.shape[1], v.shape[-1])
    assert lse.shape == (q.shape[0], q.shape[1])
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    (
        "solution,dtype,num_heads,cache_seqlens_list,page_size,"
        "return_lse,max_seqlen_k"
    ),
    [
        ("triton", torch.bfloat16, 128, [5, 4], 4, True, None),
        ("triton", platform.fp8e4m3fn.dtype, 128, [5, 4], 4, True, None),
        ("gluon", torch.bfloat16, 16, [65, 64, 129, 1], 64, True, None),
        ("gluon", torch.bfloat16, 128, [65, 64, 129, 1] * 16, 64, True, None),
        ("gluon", torch.bfloat16, 64, [64], 64, True, 64),
        ("gluon", torch.bfloat16, 64, [64], 64, False, 80_000),
        ("gluon", torch.bfloat16, 64, [64], 64, True, 80_000),
        ("gluon", torch.bfloat16, 64, [63, 65], 64, False, 80_000),
        ("gluon", torch.bfloat16, 64, [63, 65], 64, True, 80_000),
        ("gluon", torch.bfloat16, 64, [1, 64, 65, 129], 64, False, 80_000),
        ("gluon", torch.bfloat16, 64, [1, 64, 65, 129], 64, True, 80_000),
    ],
    ids=[
        "triton-bf16",
        "triton-fp8",
        "gluon-bh16bn64",
        "gluon-bh64",
        "gluon-h64-b1-single-split",
        "gluon-h64-b1-output-only",
        "gluon-h64-b1-with-lse",
        "gluon-h64-b2-output-only",
        "gluon-h64-b2-with-lse",
        "gluon-h64-b4-output-only",
        "gluon-h64-b4-with-lse",
    ],
)
def test_mla_decode_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    num_heads: int,
    cache_seqlens_list: list[int],
    page_size: int,
    return_lse: bool,
    max_seqlen_k: int | None,
    require,
) -> None:
    require("attention", "mla_decode_with_kvcache", solution, dtype, "q")

    _run_mla_decode_case(
        device=device,
        solution=solution,
        dtype=dtype,
        num_heads=num_heads,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        cache_seqlens_list=cache_seqlens_list,
        page_size=page_size,
        return_lse=return_lse,
        max_seqlen_k=max_seqlen_k,
    )
