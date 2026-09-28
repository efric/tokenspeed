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

"""Semantic attention-prefill reduction and AttnRes composition."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform


def attention_prefill_projection_supported(rows: int, dtype: torch.dtype) -> bool:
    """Whether a prepared producer is in the supported prefill window.

    Args:
        rows: Number of attention projection rows on every rank.
        dtype: Projection dtype.

    Returns:
        Whether the CDNA4 BF16 producer path covers this shape.
    """
    return (
        current_platform().is_cdna4 and dtype == torch.bfloat16 and 16 <= rows <= 8192
    )


def attention_prefill_sharded_supported(rows: int) -> bool:
    """Whether token rows can be assigned evenly to the eight-rank mixer.

    Args:
        rows: Number of attention projection rows on every rank.

    Returns:
        Whether the mixer may keep the residual token-sharded.
    """
    return 512 <= rows <= 8192 and rows % 8 == 0


def attention_prefill_mix(
    partial: torch.Tensor,
    residual: torch.Tensor | None,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_norm_eps: float,
    num_valid_blocks: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Mix a prepared attention projection into an AttnRes activation.

    Args:
        partial: Prepared local BF16 projection ``[M, 7168]``.
        residual: Replicated BF16 prefix, or None on a block-write layer.
        block_residual: Replicated block-major history ``[K, M, 7168]``.
        res_weight: AttnRes score projection weight.
        rms_weight: AttnRes score RMSNorm weight.
        eps: Score RMSNorm epsilon.
        out_norm_weight: Output RMSNorm weight.
        out_norm_eps: Output RMSNorm epsilon.
        num_valid_blocks: Number of leading history blocks to mix.
        group: The prepared eight-rank attention process group.

    Returns:
        An owned ``[M/8, 7168]`` residual shard and borrowed replicated
        activation, or None so the caller retains ordinary reduction and
        AttnRes. Consume the activation before this group's next prefill mix
        or token-sharded MoE tail overwrites its storage.
    """
    if not current_platform().is_cdna4:
        return None
    from tokenspeed_kernel.ops.communication.iris_prefill import (
        iris_attention_prefill_mix,
    )

    return iris_attention_prefill_mix(
        partial,
        residual,
        block_residual,
        res_weight,
        rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
        num_valid_blocks=num_valid_blocks,
        group=group,
    )
