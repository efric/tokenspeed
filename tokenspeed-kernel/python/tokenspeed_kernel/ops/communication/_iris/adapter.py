# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Iris state reuse and capability adapters for public communication operations."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication._dependencies import (
    amd_collectives_available,
    iris_available,
)
from tokenspeed_kernel.ops.communication._selection import select_collective
from tokenspeed_kernel.ops.communication._state import TritonCommState
from tokenspeed_kernel.platform import current_platform

_ALLREDUCE_RESIDUAL_ATTNRES_MAX_TOKENS = 16


def all_reduce_can_run(state: TritonCommState, tensor: torch.Tensor, op=None) -> bool:
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    platform = current_platform()
    return (
        platform.is_amd
        and iris_available()
        and op == torch.distributed.ReduceOp.SUM
        and tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.dtype == torch.bfloat16
        and 0 < tensor.numel() <= state.max_numel
        and state.world_size > 1
    )


def _iris_state_key(state: TritonCommState, dtype: torch.dtype) -> tuple:
    producer_direct_max_numel = state.max_bytes // dtype.itemsize
    return (
        id(state.group),
        state.max_numel,
        producer_direct_max_numel,
        state.attnres_max_numel,
        state.max_token_num,
        state.enable_lamport,
        dtype,
    )


def _iris_state_is_compatible(iris_state, state, dtype: torch.dtype) -> bool:
    return (
        iris_state.group is state.group
        and iris_state.rank_in_group == state.rank_in_group
        and iris_state.device == state.device
        and iris_state.dtype == dtype
        and iris_state.staged_max_numel >= state.max_numel
        and iris_state.producer_direct_max_numel >= state.max_bytes // dtype.itemsize
        and iris_state.attnres_max_numel >= state.attnres_max_numel
        and iris_state.attnres_max_rows >= state.max_token_num
        # AttnRes-only views can share a prepared state regardless of its
        # producer-direct policy; they never dispatch a Lamport reduction.
        and (state.max_bytes == 0 or iris_state.enable_lamport == state.enable_lamport)
    )


def _get_or_create_iris_state(state: TritonCommState, dtype: torch.dtype):
    """Return the Iris state sized for this communication backing buffer."""
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    key = _iris_state_key(state, dtype)
    iris_state = _iris_mod.IRIS_AR_STATES.get(key)
    if iris_state is None:
        iris_state = next(
            (
                candidate
                for candidate in _iris_mod.IRIS_AR_STATES.values()
                if _iris_state_is_compatible(candidate, state, dtype)
            ),
            None,
        )
    if iris_state is None:
        iris_state = _iris_mod.create_iris_state(
            group=state.group,
            rank_in_group=state.rank_in_group,
            staged_max_numel=state.max_numel,
            producer_direct_max_numel=state.max_bytes // dtype.itemsize,
            attnres_max_numel=state.attnres_max_numel,
            attnres_max_rows=state.max_token_num,
            enable_lamport=state.enable_lamport,
            dtype=dtype,
            heap_size=None,
            device=state.device,
        )
    _iris_mod.IRIS_AR_STATES[key] = iris_state
    return iris_state


def initialize_all_reduce_state(
    state: TritonCommState,
    dtype: torch.dtype,
) -> None:
    """Allocate the backend storage described by an all-reduce state.

    Args:
        state: Communication state carrying the requested buffer capacities.
        dtype: Element type used by the all-reduce buffers.

    Returns:
        None.
    """
    _get_or_create_iris_state(state, dtype)


def all_reduce(state: TritonCommState, tensor: torch.Tensor, op=None) -> torch.Tensor:
    assert all_reduce_can_run(state, tensor, op=op)
    platform = current_platform()
    if platform.is_amd:

        iris_state = _get_or_create_iris_state(state, tensor.dtype)
        return select_collective("all_reduce", tensor.dtype, "iris_all_reduce")(
            iris_state,
            tensor,
            op=op,
            safe=False,
            async_op=False,
        )

    raise AssertionError(f"Unsupported platform: {platform}")


def symm_outputs_can_run(
    state: TritonCommState,
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    op=None,
) -> bool:
    """Check whether Iris can reduce producer-owned output buffers.

    Args:
        state: Communication group and symmetric-buffer capacity.
        shapes: Ordered shapes the producer will write.
        dtype: Element type shared by all outputs.
        op: Reduction operation; only SUM is currently supported.

    Returns:
        Whether the request is supported by the CDNA4 producer-direct kernel.
    """
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    numels = tuple(math.prod(shape) for shape in shapes)
    total_numel = sum(numels)
    if (
        not current_platform().is_cdna4
        or not amd_collectives_available()
        or op != torch.distributed.ReduceOp.SUM
        or not numels
        or any(numel <= 0 for numel in numels)
    ):
        return False

    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    return _iris_mod.producer_direct_all_reduce_can_run(
        world_size=state.world_size,
        total_numel=total_numel,
        dtype=dtype,
        max_bytes=state.max_bytes,
    )


def acquire_symm_outputs(
    state: TritonCommState,
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Acquire consecutive Iris views for producer-direct reduction."""
    if not symm_outputs_can_run(state, shapes, dtype):
        raise RuntimeError("unsupported symmetric all-reduce output request")
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    iris_state = _get_or_create_iris_state(state, dtype)
    return _iris_mod.iris_acquire_outputs(iris_state, shapes)


def all_reduce_symm_can_run(
    state: TritonCommState,
    tensors: tuple[torch.Tensor, ...],
    op=None,
) -> bool:
    """Check whether tensors are Iris-owned symmetric all-reduce outputs."""
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    if not current_platform().is_cdna4 or op != torch.distributed.ReduceOp.SUM:
        return False
    if not tensors:
        return False
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    key = _iris_state_key(state, tensors[0].dtype)
    iris_state = _iris_mod.IRIS_AR_STATES.get(key)
    return iris_state is not None and iris_state.owns_outputs(tensors)


def all_reduce_symmetric(
    state: TritonCommState,
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Reduce consecutive Iris producer outputs in one launch."""
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    key = _iris_state_key(state, tensors[0].dtype)
    iris_state = _iris_mod.IRIS_AR_STATES[key]
    return select_collective(
        "all_reduce_symmetric", tensors[0].dtype, "iris_all_reduce_symmetric"
    )(
        iris_state,
        tensors,
    )


def allreduce_residual_attnres_max_tokens(world_size: int) -> int:
    """Return the AttnRes token limit for a communication group.

    Args:
        world_size: Number of ranks participating in the all-reduce.

    Returns:
        The supported token count, or zero when the group size is unsupported.
    """
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    kernel_config = _iris_mod.IRIS_ALL_REDUCE_KERNEL_CONFIG.attnres
    if world_size != kernel_config.world_size:
        return 0
    return _ALLREDUCE_RESIDUAL_ATTNRES_MAX_TOKENS


def _all_reduce_residual_attnres_can_run(
    state: TritonCommState,
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    op=None,
) -> bool:
    """Return whether Iris can reduce and consume a attention partial."""
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    m, s_, acc = scratch
    platform = current_platform()
    if not platform.is_cdna4 or not amd_collectives_available():
        return False

    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    kernel_config = _iris_mod.IRIS_ALL_REDUCE_KERNEL_CONFIG.attnres
    num_tokens = partial.shape[0] if partial.ndim == 2 else 0
    return (
        state.world_size == kernel_config.world_size
        and op == torch.distributed.ReduceOp.SUM
        and 0 < num_tokens <= allreduce_residual_attnres_max_tokens(state.world_size)
        and num_tokens <= state.max_token_num
        and partial.shape == residual.shape == (num_tokens, kernel_config.hidden_size)
        and score_weight.shape == output_weight.shape == (kernel_config.hidden_size,)
        and m.shape == s_.shape == (num_tokens,)
        and acc.shape == (num_tokens, kernel_config.hidden_size)
        and partial.dtype
        == residual.dtype
        == score_weight.dtype
        == output_weight.dtype
        == torch.bfloat16
        and m.dtype == s_.dtype == acc.dtype == torch.float32
        and partial.device
        == residual.device
        == score_weight.device
        == output_weight.device
        == m.device
        == s_.device
        == acc.device
        == state.device
        and all(
            tensor.is_contiguous()
            for tensor in (
                partial,
                residual,
                score_weight,
                output_weight,
                m,
                s_,
                acc,
            )
        )
        and partial.numel() <= state.attnres_max_numel
    )


def _all_reduce_residual_attnres(
    state: TritonCommState,
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a attention partial and finish its AttnRes epilogue.

    Args:
        state: Initialized communication state for the tensor-parallel group.
        partial: Contiguous local BF16 attention partial shaped ``[M, 7168]``.
        residual: Contiguous BF16 residual stream shaped ``[M, 7168]``.
        score_weight: Contiguous BF16 AttnRes score weight shaped ``[7168]``.
        output_weight: Contiguous BF16 output RMSNorm weight shaped ``[7168]``.
        scratch: Contiguous FP32 ``(max_logit, exp_sum, weighted_sum)`` tensors
            for the historical AttnRes candidates, shaped ``[M]``, ``[M]``,
            and ``[M, 7168]``.
        eps: Positive epsilon used for AttnRes scoring and output RMSNorm.
        op: Reduction operation; only ``SUM`` is supported.

    Returns:
        A pair containing the normalized AttnRes mixture and the BF16 sum of
        ``residual`` with the all-reduced attention partial, both shaped
        ``[M, 7168]``.
    """
    assert _all_reduce_residual_attnres_can_run(
        state,
        partial,
        residual,
        score_weight,
        output_weight,
        scratch,
        op=op,
    )
    import tokenspeed_kernel.ops.communication.iris as _iris_mod

    iris_state = _get_or_create_iris_state(state, partial.dtype)
    return _iris_mod.iris_all_reduce_residual_attnres(
        iris_state,
        partial,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
        op=op,
    )


def _attnres_comm_state(
    input_tensor: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
) -> TritonCommState:
    return TritonCommState(
        enable_lamport=False,
        group=group,
        rank_in_group=rank,
        world_size=group.size(),
        device=input_tensor.device,
        attnres_max_numel=input_tensor.numel(),
        max_numel=0,
        max_bytes=0,
        max_token_num=input_tensor.shape[0],
        hidden_dim=0,
        comm_buff=None,
        symm_mem_hdl=None,
    )


def allreduce_residual_attnres_combine_supported(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    op=None,
) -> bool:
    """Return whether the fused Iris collective supports this call.

    The Iris all-reduce transport is node-local. ``local_world_size`` maps the
    process group's global ranks to nodes so multi-node attention groups use
    the model's ordinary collective fallback instead.
    """
    if local_world_size <= 0:
        return False
    group_ranks = dist.get_process_group_ranks(group)
    if len({global_rank // local_world_size for global_rank in group_ranks}) != 1:
        return False
    return _all_reduce_residual_attnres_can_run(
        _attnres_comm_state(input_tensor, rank, group),
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        op=op,
    )


def allreduce_residual_attnres_combine(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    eps: float = 1e-6,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Iris all-reduce and AttnRes epilogue.

    Args:
        input_tensor: Per-rank BF16 attention partial shaped ``[M, 7168]``.
        residual: Running BF16 residual stream shaped ``[M, 7168]``.
        score_weight: Precomputed AttnRes score weight shaped ``[7168]``.
        output_weight: Output RMSNorm weight shaped ``[7168]``.
        scratch: FP32 ``(max_logit, exp_sum, weighted_sum)`` partials.
        rank: Rank within ``group``.
        group: Validated node-local attention process group.
        local_world_size: Number of processes on each node.
        eps: Positive epsilon for AttnRes scoring and output RMSNorm.
        op: Reduction operation. Only ``SUM`` is supported.

    Returns:
        The normalized AttnRes output and accumulated residual, in that order.
    """
    assert allreduce_residual_attnres_combine_supported(
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        rank=rank,
        group=group,
        local_world_size=local_world_size,
        op=op,
    )
    return _all_reduce_residual_attnres(
        _attnres_comm_state(input_tensor, rank, group),
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
        op=op,
    )
