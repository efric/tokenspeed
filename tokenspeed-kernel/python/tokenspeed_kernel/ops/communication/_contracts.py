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
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

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
class MoETailRequirement:
    """Maximum token-sharded MoE rows, with two producer widths and one result."""

    max_rows: int
    routed_width: int
    hidden_width: int

    def __post_init__(self) -> None:
        if self.max_rows < 0 or self.routed_width <= 0 or self.hidden_width <= 0:
            raise ValueError("invalid MoE-tail dimensions")


@dataclass(frozen=True)
class AllReducePreparation:
    """Operation demands for one group; implementations derive physical storage."""

    dtype: torch.dtype
    operations: tuple[
        AllReduceRequirement
        | PackedAllReduceRequirement
        | MoETailRequirement
        | AttnResRequirement,
        ...,
    ]


def select_collective(mode: str, dtype: torch.dtype, name: str) -> SelectedKernel:
    """Select a prepared implementation without allowing overrides to change protocol.

    Args:
        mode: Semantic communication operation.
        dtype: Payload dtype.
        name: Winner of the operation's rank-uniform capability/policy checks.

    Returns:
        The registered callable. An incompatible override raises before launch.

    Collective policy and environment overrides must be identical on every rank.
    There is no timing-based or pointer-based collective selection here.
    """
    selected = select_kernel(
        "communication",
        mode,
        format_signature(input=dense_tensor_format(dtype)),
        traits={"implementation": name},
    )
    # Registry overrides deliberately bypass some trait gates. A collective
    # override cannot bypass the group's prepared protocol or operand contract.
    if selected.name != name:
        raise ValueError(
            f"collective override {selected.name!r} is incompatible with "
            f"the prepared implementation {name!r}"
        )
    return selected
