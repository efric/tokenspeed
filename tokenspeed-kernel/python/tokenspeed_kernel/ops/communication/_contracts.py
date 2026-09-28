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


"""Semantic capacity demands, independent of collective implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.residual.attnres import AttnResRequirement


@dataclass(frozen=True)
class AllReduceRequirement:
    """Maximum rows and width of ordinary partial sums."""

    max_rows: int
    width: int

    def __post_init__(self) -> None:
        if self.max_rows < 0 or self.width <= 0:
            raise ValueError("invalid all-reduce dimensions")


@dataclass(frozen=True)
class PackedAllReduceRequirement:
    """Independent partials produced together over tensor/expert group axes."""

    max_rows: int
    widths: tuple[int, ...]
    tensor_parallel_size: int
    expert_parallel_size: int

    def __post_init__(self) -> None:
        if (
            self.max_rows < 0
            or not self.widths
            or min(self.widths) <= 0
            or self.tensor_parallel_size <= 0
            or self.expert_parallel_size <= 0
        ):
            raise ValueError("invalid packed all-reduce dimensions")


@dataclass(frozen=True)
class AllReducePreparation:
    """Operation demands for one group; implementations derive physical storage."""

    dtype: torch.dtype
    operations: tuple[
        AllReduceRequirement | PackedAllReduceRequirement | AttnResRequirement, ...
    ]
