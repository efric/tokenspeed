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

"""Semantic token-sharded MoE tail, with optional vendor implementations."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform


def token_sharded_moe_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    residual: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    eps: float | None,
    group: dist.ProcessGroup,
) -> torch.Tensor | None:
    """Return a borrowed reduced/projected result, or None for caller fallback.

    The caller retains the ordinary collective and projection path. A selected
    implementation must consume producer-owned partials without mutating them.
    The returned storage is borrowed and must be consumed before its next use.
    """
    if not current_platform().is_cdna4:
        return None
    from tokenspeed_kernel.ops.moe.iris import iris_kimi3_moe_tail

    return iris_kimi3_moe_tail(
        routed_partial,
        shared_partial,
        residual,
        projection_weight,
        norm_weight=norm_weight,
        eps=eps,
        group=group,
    )
