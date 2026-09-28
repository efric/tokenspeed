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


"""Iris solution entry points. Implementation state and protocols are private."""

from typing import TYPE_CHECKING, Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris.all_reduce import IrisAllReduce
    from tokenspeed_kernel.ops.communication._iris.epilogues import (
        IrisAllReduceResidualRMSNorm,
    )
    from tokenspeed_kernel.ops.communication._iris.rsag import IrisRSAG

from tokenspeed_kernel.ops.communication._iris.context import (
    IRIS_AR_RMSNORM_STATES as IRIS_AR_RMSNORM_STATES,
)
from tokenspeed_kernel.ops.communication._iris.context import (
    IRIS_AR_STATES as IRIS_AR_STATES,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    IRIS_ALL_REDUCE_KERNEL_CONFIG as IRIS_ALL_REDUCE_KERNEL_CONFIG,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    AttnResKernelConfig as AttnResKernelConfig,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    IrisAllReduceKernelConfig as IrisAllReduceKernelConfig,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    PackedAllReduceKernelConfig as PackedAllReduceKernelConfig,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    producer_direct_all_reduce_can_run as producer_direct_all_reduce_can_run,
)


def create_iris_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
    moe_tail_max_rows: int,
    enable_lamport: bool,
    dtype: torch.dtype,
    heap_size: int | None,
    device: torch.device | None,
) -> "IrisAllReduce":
    """Create an Iris all-reduce state with separate capacities for each path.

    Args:
        group: Process group used by the collectives.
        rank_in_group: This process's rank within ``group``.
        staged_max_numel: Maximum ordinary staged all-reduce payload.
        producer_direct_max_numel: Maximum producer-direct payload.
        attnres_max_numel: Maximum fused attention/AttnRes payload.
        attnres_max_rows: Maximum fused attention/AttnRes rows.
        moe_tail_max_rows: Capacity of the borrowed token-sharded MoE result.
        enable_lamport: Allow Lamport for eligible producer-direct payloads.
        dtype: Element type for all payload buffers.
        heap_size: Optional symmetric heap size in bytes.
        device: Device on which buffers are allocated.

    Returns:
        The initialized all-reduce state.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import IrisAllReduce

    return IrisAllReduce(
        group=group,
        rank_in_group=rank_in_group,
        staged_max_numel=staged_max_numel,
        producer_direct_max_numel=producer_direct_max_numel,
        attnres_max_numel=attnres_max_numel,
        attnres_max_rows=attnres_max_rows,
        moe_tail_max_rows=moe_tail_max_rows,
        enable_lamport=enable_lamport,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
    )


@register_kernel(
    "communication",
    "all_reduce",
    name="iris_all_reduce",
    solution="iris",
    traits={"implementation": frozenset({"iris_all_reduce"})},
    signatures=format_signatures(
        ("input",), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
)
def iris_all_reduce(
    state: "IrisAllReduce",
    tensor: torch.Tensor,
    op=None,
    safe: bool = True,
    async_op: bool = False,
) -> torch.Tensor:
    """Reduce in place, optionally cloning the result after mutation.

    SUM is the only operation; async_op must be False. safe=True returns an
    independent clone, while safe=False returns the input tensor.
    """
    return state.all_reduce(tensor, op=op, safe=safe, async_op=async_op)


def iris_acquire_outputs(
    state: "IrisAllReduce",
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Borrow consecutive symmetric input views until the next acquisition."""
    return state.acquire_outputs(shapes)


@register_kernel(
    "communication",
    "all_reduce_symmetric",
    name="iris_all_reduce_symmetric",
    solution="iris",
    traits={"implementation": frozenset({"iris_all_reduce_symmetric"})},
    signatures=format_signatures(
        ("input",), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
)
def iris_all_reduce_symmetric(
    state: "IrisAllReduce",
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Return borrowed views of separate local results, reused by the next reduce."""
    return state.all_reduce_symmetric(tensors)


def iris_all_reduce_residual_attnres(
    state: "IrisAllReduce",
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish the exact AttnRes attention reduction and AttnRes mix."""
    from tokenspeed_kernel.ops.communication._iris.epilogues import (
        all_reduce_residual_attnres,
    )

    return all_reduce_residual_attnres(
        state,
        partial,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
        op=op,
    )


def create_iris_rsag_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device = None,
    heap_size: int | None = None,
) -> "IrisRSAG":
    from tokenspeed_kernel.ops.communication._iris.rsag import IrisRSAG

    return IrisRSAG(
        group=group,
        rank_in_group=rank_in_group,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        device=device,
        heap_size=heap_size,
    )


def create_iris_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    heap_size: int | None = None,
    device: torch.device = None,
    persistent: bool = False,
) -> "IrisAllReduceResidualRMSNorm":
    from tokenspeed_kernel.ops.communication._iris.epilogues import (
        IrisAllReduceResidualRMSNorm,
    )

    return IrisAllReduceResidualRMSNorm(
        group=group,
        rank_in_group=rank_in_group,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
        persistent=persistent,
    )


def iris_allreduce_residual_rmsnorm(
    state: "IrisAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return state.fused(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        eps=eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )


def __getattr__(name: str):
    """Retain lazy access to legacy state types without importing Iris eagerly."""
    if name == "IrisAllReduce":
        from tokenspeed_kernel.ops.communication._iris.all_reduce import IrisAllReduce

        return IrisAllReduce
    if name == "IrisRSAG":
        from tokenspeed_kernel.ops.communication._iris.rsag import IrisRSAG

        return IrisRSAG
    if name == "IrisAllReduceResidualRMSNorm":
        from tokenspeed_kernel.ops.communication._iris.epilogues import (
            IrisAllReduceResidualRMSNorm,
        )

        return IrisAllReduceResidualRMSNorm
    raise AttributeError(name)


@register_kernel(
    "communication",
    "all_reduce_attnres",
    name="iris_all_reduce_attnres_epilogue",
    solution="iris",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("input", "dense", {torch.bfloat16}),
    traits={"implementation": frozenset({"iris_all_reduce_attnres_epilogue"})},
)
def iris_all_reduce_attnres_epilogue(state, partial, residual, epilogue):
    """Run a previously selected AttnRes binding, returning residual then mixture."""
    from tokenspeed_kernel.ops.communication._iris.adapter import (
        allreduce_residual_attnres_combine,
    )

    hidden, updated = allreduce_residual_attnres_combine(
        partial,
        residual,
        epilogue.score_product,
        epilogue.output_weight,
        epilogue.partials,
        rank=state.rank,
        group=state.group,
        local_world_size=state.local_world_size,
        eps=epilogue.eps,
    )
    return updated, hidden


@register_kernel(
    "communication",
    "all_reduce_rmsnorm",
    name="iris_all_reduce_rmsnorm_adapter",
    solution="iris",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "input", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    traits={"implementation": frozenset({"iris_all_reduce_rmsnorm_adapter"})},
)
def iris_all_reduce_rmsnorm_adapter(**kwargs):
    from tokenspeed_kernel.ops.communication._iris.epilogues import (
        allreduce_residual_rmsnorm,
    )

    return allreduce_residual_rmsnorm(**kwargs)
