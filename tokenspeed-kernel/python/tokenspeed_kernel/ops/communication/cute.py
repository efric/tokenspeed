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

"""NVIDIA native residual all-reduce adapter."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures

RESIDUAL_MAX_ROWS = 8


def residual_collective_eligible(
    *,
    armed: bool,
    has_prefix: bool,
    num_tokens: int,
    fusion_max_tokens: int,
) -> bool:
    """Apply the measured native residual-reduction window, identically on peers."""
    return (
        armed
        and has_prefix
        and 0 < num_tokens <= min(RESIDUAL_MAX_ROWS, fusion_max_tokens)
    )


def prepare_residual_collective(group, rank: int, width: int, enabled: bool):
    """Collectively prepare the native residual implementation before capture."""
    from tokenspeed_kernel.ops.moe.latent_tail import (
        attn_reduce_shape_supported,
        build_attn_reduce_collective,
        multicast_backend_available,
    )

    local_ok = (
        enabled
        and multicast_backend_available(group)
        and attn_reduce_shape_supported(tp_size=group.size(), hidden_size=width)
    )
    vote = torch.tensor([int(local_ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    if not bool(vote.item()):
        return None
    return build_attn_reduce_collective(
        group=group,
        rank=rank,
        tp_size=group.size(),
        hidden_size=width,
        max_tokens=RESIDUAL_MAX_ROWS,
    )


@register_kernel(
    "communication",
    "all_reduce_residual",
    name="cute_all_reduce_residual",
    solution="cute",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("input", "dense", {torch.bfloat16}),
    traits={"implementation": frozenset({"cute_all_reduce_residual"})},
)
def cute_all_reduce_residual(state, partial, residual, epilogue):
    """Return accumulated residual and no mixed output; the norm is discarded."""
    residual_out, _ = state.native(
        partial,
        residual,
        state.unit_weight,
        include_reduce_scatter=False,
        include_routed=True,
    )
    return residual_out, None
