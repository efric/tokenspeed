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


"""Online-softmax AttnRes partials and collective epilogue contracts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttnResRequirement:
    """Maximum rows and width of an all-reduce followed by AttnRes."""

    max_rows: int
    width: int

    def __post_init__(self) -> None:
        if self.max_rows < 0 or self.width <= 0:
            raise ValueError("invalid AttnRes dimensions")


@dataclass(frozen=True)
class AttnResEpilogue:
    """Operands for combining a replicated prefix with online-softmax partials.

    The separate score factors and their precomputed product preserve existing
    implementation-specific rounding. They are not interchangeable encodings.
    """

    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    score_projection: torch.Tensor
    score_norm: torch.Tensor
    score_product: torch.Tensor | None
    output_weight: torch.Tensor | None
    eps: float


def attnres_combine(
    prefix, score_weight, output_weight, eps, partials, out, enable_pdl
):
    """Combine a prefix with partials into out, optionally applying output RMSNorm."""
    from tokenspeed_kernel.ops.activation.triton import attnres_combine as run

    return run(
        prefix, score_weight, output_weight, eps, partials, out, enable_pdl=enable_pdl
    )
