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


"""Generic runtime adapter for kernel-package all-reduce implementations."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication import (
    DEFAULT_PRODUCER_DIRECT_MAX_BYTES,
    AllReducePreparation,
    acquire_symm_outputs,
    all_reduce,
    all_reduce_can_run,
    all_reduce_capacity,
    all_reduce_symm_can_run,
    all_reduce_symmetric,
    create_all_reduce_handle,
    prepare_all_reduce_handle,
    producer_all_reduce_available,
    symm_outputs_can_run,
)

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


class KernelAllReduceBackend(CommBackend):
    def __init__(self, fallback: CommBackend, producer_direct_max_bytes: int):
        self._fallback = fallback
        self._instances = {}
        self._producer_direct_max_bytes = producer_direct_max_bytes

    @property
    def producer_direct_max_bytes(self) -> int:
        return self._producer_direct_max_bytes

    def _get_or_create(self, group: Group):
        if group not in self._instances:
            self._instances[group] = create_all_reduce_handle(
                group=pg_manager.get_process_group("nccl", group),
                rank_in_group=group.index(dist.get_rank()),
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                producer_direct_max_bytes=self._producer_direct_max_bytes,
            )
        return self._instances[group]

    def prepare_all_reduce_buffers(
        self,
        group: Group,
        *,
        preparation: AllReducePreparation,
    ) -> bool:
        """Prepare opaque kernel storage from group-wide operation demands."""
        state = prepare_all_reduce_handle(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            preparation=preparation,
            previous=self._instances.get(group),
            producer_direct_max_bytes=self._producer_direct_max_bytes,
        )
        if state is None:
            return False
        self._instances[group] = state
        return True

    def can_run(self, tensor: torch.Tensor, group: Group, op=None) -> bool:
        if len(group) <= 1:
            return False
        try:
            return all_reduce_can_run(self._get_or_create(group), tensor, op=op)
        except Exception:
            return False

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            if self.can_reduce_outputs(tensor, group, op=op):
                return all_reduce_symmetric(self._instances[group], tensor)
            return super().all_reduce(tensor, group, op=op)

        state = self._get_or_create(group)
        if all_reduce_can_run(state, tensor, op=op):
            return all_reduce(state, tensor, op=op)
        return self._fallback.all_reduce(tensor, group, op=op)

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire producer outputs when the kernel implementation supports them."""
        if not self.can_acquire_outputs(shapes, like, group, op=op):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)

        # Do not let one rank silently select a different collective protocol.
        state = self._get_or_create(group)
        return acquire_symm_outputs(state, shapes, like.dtype)

    def can_acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Whether the implementation can consume acquired producer storage."""
        return self.can_acquire_outputs(shapes, like, group, op=op)

    def can_acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Check producer-direct eligibility without allocating kernel storage."""
        if not producer_all_reduce_available() or not like.is_cuda:
            return False
        total_bytes = sum(math.prod(shape) for shape in shapes) * like.dtype.itemsize
        state = self._instances.get(group)
        max_bytes = (
            all_reduce_capacity(state)
            if state is not None
            else self._producer_direct_max_bytes
        )
        if total_bytes > max_bytes:
            return False
        if state is None:
            state = self._get_or_create(group)
        return symm_outputs_can_run(state, shapes, like.dtype, op=op)

    def can_reduce_outputs(
        self,
        tensors: tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> bool:
        """Check whether tensors are this group's symmetric outputs."""
        state = self._instances.get(group)
        return state is not None and all_reduce_symm_can_run(state, tensors, op=op)

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        return self._fallback.all_gather(tensor, group, dim)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_gather_single(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._fallback.reduce_scatter(tensor, group)

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_to_all_single(output, input, group)

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")
