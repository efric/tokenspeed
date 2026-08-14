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

"""Triton all-reduce backend for latency-sensitive small AMD tensors."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import (
    acquire_producer_direct_lane as kernel_acquire_producer_direct_lane,
    acquire_symm_outputs,
    all_reduce,
    all_reduce_can_run,
    all_reduce_symm_can_run,
    all_reduce_symmetric,
    create_state,
    symm_outputs_can_run,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

# Preserve the measured ordinary-Iris window while allowing a larger
# producer-direct backing allocation.
_DEFAULT_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
_DEFAULT_ALL_REDUCE_MAX_BYTES = 512 * 1024


class TritonAllReduceBackend(CommBackend):
    def __init__(
        self,
        fallback: CommBackend,
        producer_direct_max_bytes: int = _DEFAULT_PRODUCER_DIRECT_MAX_BYTES,
    ):
        self._fallback = fallback
        self._instances = {}
        self._producer_direct_max_bytes = producer_direct_max_bytes
        self._max_numel = (
            min(producer_direct_max_bytes, _DEFAULT_ALL_REDUCE_MAX_BYTES)
            // torch.empty((), dtype=torch.bfloat16).element_size()
        )

    @property
    def producer_direct_max_bytes(self) -> int:
        return self._producer_direct_max_bytes

    def _get_or_create(self, group: Group):
        if group in self._instances:
            return self._instances[group]

        state = create_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            max_numel=self._max_numel,
            max_bytes=self._producer_direct_max_bytes,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        self._instances[group] = state
        return state

    def can_run(self, tensor: torch.Tensor, group: Group, op=None) -> bool:
        if len(group) <= 1 or not current_platform().is_amd:
            return False
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        if not (
            op == torch.distributed.ReduceOp.SUM
            and tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.dtype == torch.bfloat16
            and 0 < tensor.numel() <= self._max_numel
        ):
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
        """Acquire symmetric outputs when Iris supports the request."""
        if not self.can_acquire_outputs(shapes, like, group, op=op):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)

        # Do not let one rank silently select a different collective protocol.
        state = self._get_or_create(group)
        return acquire_symm_outputs(state, shapes, like.dtype)

    @staticmethod
    def _bounded_status(local_status: int, local_reason: str) -> tuple[int, str]:
        try:
            status = int(local_status)
        except Exception:
            return 9001, "local status is not an integer"
        try:
            reason = str(local_reason).replace("\x00", "")[:512]
        except Exception:
            return 9004, "local reason could not be serialized"
        return status, reason

    def consensus_producer_direct_lane_status(
        self,
        group: Group,
        *,
        stage: str,
        local_status: int = 0,
        local_reason: str = "",
    ) -> None:
        """Gather a bounded status on Gloo before any collective GPU stage."""

        status, reason = self._bounded_status(local_status, local_reason)
        try:
            stage_name = str(stage).replace("\x00", "")[:64]
        except Exception:
            stage_name = "invalid_stage"
            if status == 0:
                status, reason = 9005, "stage name could not be serialized"
        if not stage_name and status == 0:
            status, reason = 9002, "producer-direct stage name is empty"
        payload = (stage_name, status, reason)
        control_group = pg_manager.get_process_group("gloo", group)
        gathered = [None] * len(group)
        dist.all_gather_object(gathered, payload, group=control_group)

        gathered_stages = {item[0] for item in gathered}
        failures = [
            (global_rank, item[1], item[2])
            for global_rank, item in zip(group, gathered)
            if item[1] != 0
        ]
        if len(gathered_stages) != 1:
            failures = [
                (global_rank, 9003, f"stage mismatch: {item[0]!r}")
                for global_rank, item in zip(group, gathered)
            ]
        if failures:
            details = "; ".join(
                f"rank {rank}: status {failure_status}: {failure_reason}"
                for rank, failure_status, failure_reason in failures
            )
            agreed_stage = sorted(gathered_stages)[0] if gathered_stages else stage_name
            raise RuntimeError(
                f"producer-direct {agreed_stage} consensus failed: {details}"
            )

    def _local_producer_direct_admission(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
    ) -> tuple[int, str]:
        try:
            numels = tuple(math.prod(shape) for shape in shapes)
            element_bytes = like.dtype.itemsize
            total_numel = sum(numels)
            if not current_platform().is_cdna4:
                return 1001, "producer-direct lane requires CDNA4"
            if not like.is_cuda:
                return 1002, "producer-direct lane requires a CUDA/ROCm tensor"
            if len(group) not in (2, 4, 8) or len(set(group)) != len(group):
                return 1003, f"unsupported or duplicate process group: {group}"
            if dist.get_rank() not in group:
                return 1004, f"current rank is not in process group: {group}"
            if like.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                return 1005, f"unsupported producer-direct dtype: {like.dtype}"
            if not numels or any(numel <= 0 for numel in numels):
                return 1006, f"invalid producer-direct shapes: {shapes}"
            if 8 % element_bytes or total_numel % (8 // element_bytes):
                return 1007, "producer-direct size is not packed-word aligned"
            if total_numel * element_bytes > self._producer_direct_max_bytes:
                return 1008, "producer-direct request exceeds lane capacity"
        except Exception as exc:
            return 1099, f"producer-direct admission raised {type(exc).__name__}: {exc}"
        return 0, ""

    def acquire_producer_direct_lane(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        *,
        local_status: int = 0,
        local_reason: str = "",
    ) -> object:
        """Collectively borrow the exact cached Iris producer-direct owner."""

        status, reason = self._bounded_status(local_status, local_reason)
        if status == 0:
            status, reason = self._local_producer_direct_admission(shapes, like, group)
        self.consensus_producer_direct_lane_status(
            group,
            stage="admission",
            local_status=status,
            local_reason=reason,
        )
        lane = None
        try:
            state = self._get_or_create(group)
            lane = kernel_acquire_producer_direct_lane(state, shapes, like.dtype)
            acquisition_status, acquisition_reason = 0, ""
        except Exception as exc:
            acquisition_status = 1100
            acquisition_reason = f"lane acquisition raised {type(exc).__name__}: {exc}"
        # This catches asymmetric validation/setup errors after a collective
        # Iris allocation has returned. A rank stuck inside that allocation is
        # still handled by the enclosing startup watchdog.
        self.consensus_producer_direct_lane_status(
            group,
            stage="lane_acquisition",
            local_status=acquisition_status,
            local_reason=acquisition_reason,
        )
        assert lane is not None
        return lane

    def can_acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Check producer-direct eligibility without initializing Iris."""
        if not current_platform().is_cdna4 or not like.is_cuda:
            return False
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

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_gather_into_tensor(output, input, group)

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
