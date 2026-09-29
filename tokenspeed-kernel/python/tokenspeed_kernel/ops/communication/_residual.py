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

"""Semantic residual all-reduce selection, shared by model callers.

A binding is resolved once, before scheduling its operands. It reports whether
it reads AttnRes partials. An absent binding leaves ordinary reduction and
post-join composition to the caller's existing execution path.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication._contracts import select_collective
from tokenspeed_kernel.ops.residual.attnres import AttnResEpilogue
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import SelectedKernel


@dataclass(frozen=True)
class ResidualAllReduceState:
    group: object
    rank: int
    local_world_size: int
    max_tokens: int
    vendor_fusion: bool
    native: object
    unit_weight: torch.Tensor


@dataclass(frozen=True)
class ResidualAllReduceBinding:
    """Selected reduction; borrowed result storage is consumed before reuse."""

    kernel: SelectedKernel
    state: ResidualAllReduceState
    epilogue: AttnResEpilogue | None
    consumes_partials: bool
    prefer_split_partials: bool

    def __call__(self, partial: torch.Tensor, residual: torch.Tensor):
        """Return (updated residual, optional normalized AttnRes mixture)."""
        return self.kernel(self.state, partial, residual, self.epilogue)


def prepare_residual_all_reduce(
    group,
    rank: int,
    local_world_size: int,
    width: int,
    max_tokens: int,
    vendor_fusion: bool,
    device: torch.device,
) -> ResidualAllReduceState:
    """Prepare residual reduction storage and agree native support across ranks.

    Args:
        group: Device group, or None when distributed execution is inactive.
        rank: Rank within that group.
        local_world_size: Processes per node for node-local transport checks.
        width: Residual width.
        max_tokens: Configured fusion admission limit, including disabled values.
        vendor_fusion: Result of the runtime's group-wide fusion preparation.
        device: Device for persistent storage.

    Returns:
        Immutable group state reused by all layers with these requirements.
    """
    from tokenspeed_kernel.ops.communication.cute import prepare_residual_collective

    unit_weight = torch.ones(width, dtype=torch.bfloat16, device=device)
    native = None
    if (
        current_platform().is_nvidia
        and group is not None
        and dist.is_initialized()
        and group.size() > 1
    ):
        native = prepare_residual_collective(
            group,
            rank,
            width,
            enabled=vendor_fusion and max_tokens > 0,
        )
    return ResidualAllReduceState(
        group,
        rank,
        local_world_size,
        max_tokens,
        vendor_fusion,
        native,
        unit_weight,
    )


def bind_residual_all_reduce(
    state: ResidualAllReduceState,
    partial: torch.Tensor,
    residual: torch.Tensor | None,
    epilogue: AttnResEpilogue | None,
    force_deterministic: bool,
) -> ResidualAllReduceBinding | None:
    """Resolve a reduction and its operand-readiness contract without launching.

    Args:
        state: Prepared group state shared across layers.
        partial: This rank's contribution, shaped [rows, width].
        residual: Replicated residual to accumulate, or None.
        epilogue: Optional AttnRes operands, including caller-owned partials.
        force_deterministic: Existing deterministic communication control.
            It excludes Iris AttnRes while retaining the existing native and
            vendor fusion domains.

    Returns:
        A callable binding describing required operands, or None for composition.

    All ranks provide the same shapes, preparation and controls. No pointer
    alignment or tensor contents participate in protocol selection. None means
    the caller runs ordinary all-reduce and combines after its partials join.
    """
    from tokenspeed_kernel.ops.communication.cute import residual_collective_eligible

    rows = partial.shape[0]
    if residual is None or rows <= 0:
        return None
    if residual_collective_eligible(
        armed=state.native is not None,
        has_prefix=True,
        num_tokens=rows,
        fusion_max_tokens=state.max_tokens,
    ):
        name, mode, consumes = "cute_all_reduce_residual", "all_reduce_residual", False
    elif state.vendor_fusion and rows <= state.max_tokens:
        from tokenspeed_kernel.ops.communication import trtllm  # noqa: F401

        name = (
            "trtllm_all_reduce_attnres"
            if epilogue is not None
            else "trtllm_all_reduce_residual"
        )
        mode = "all_reduce_attnres" if epilogue is not None else "all_reduce_residual"
        consumes = epilogue is not None
    else:
        if (
            force_deterministic
            or epilogue is None
            or epilogue.score_product is None
            or epilogue.output_weight is None
            or state.group is None
        ):
            return None
        from tokenspeed_kernel.ops.communication import iris  # noqa: F401
        from tokenspeed_kernel.ops.communication._iris import adapter

        if not adapter.allreduce_residual_attnres_combine_supported(
            partial,
            residual,
            epilogue.score_product,
            epilogue.output_weight,
            epilogue.partials,
            rank=state.rank,
            group=state.group,
            local_world_size=state.local_world_size,
        ):
            return None
        name, mode, consumes = (
            "iris_all_reduce_attnres_epilogue",
            "all_reduce_attnres",
            True,
        )
    return ResidualAllReduceBinding(
        select_collective(mode, partial.dtype, name),
        state,
        epilogue,
        consumes,
        name == "iris_all_reduce_attnres_epilogue",
    )
