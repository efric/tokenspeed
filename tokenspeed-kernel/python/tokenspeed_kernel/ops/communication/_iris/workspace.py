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


"""Iris workspace ownership; allocation order is part of the symmetric ABI."""

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication._iris.context import (
    _get_available_gpu_memory,
    _get_or_create_iris_context,
    _peer_addresses,
    amd_collectives_available,
)
from tokenspeed_kernel.ops.communication._iris.policy import (
    _PRODUCER_DIRECT_DTYPES,
    IRIS_ALL_REDUCE_KERNEL_CONFIG,
)
from tokenspeed_kernel.platform import current_platform

logger = logging.getLogger(__name__)
_platform = current_platform()


@dataclass(frozen=True)
class StagedWorkspace:
    input: torch.Tensor | None
    flags: torch.Tensor | None
    tunings: dict


@dataclass(frozen=True)
class TwoStageWorkspace:
    input: torch.Tensor | None
    scratch: torch.Tensor | None
    flags: torch.Tensor | None


@dataclass(frozen=True)
class ProducerWorkspace:
    input: torch.Tensor | None
    scratch: torch.Tensor | None
    output: torch.Tensor | None
    flags: torch.Tensor | None


@dataclass(frozen=True)
class LamportWorkspace:
    region: torch.Tensor | None
    epochs: torch.Tensor | None
    peer_addresses: tuple[int, ...] | None


@dataclass(frozen=True)
class AttnResWorkspace:
    inbox: torch.Tensor | None
    epochs: torch.Tensor | None
    flags: torch.Tensor | None
    peer_inboxes: tuple[int, ...] | None


class IrisAllReduceWorkspace:
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        staged_max_numel: int,
        producer_direct_max_numel: int,
        attnres_max_numel: int,
        attnres_max_rows: int,
        enable_lamport: bool,
        dtype: torch.dtype,
        heap_size: int | None,
        device: torch.device | None,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisAllReduce; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisAllReduce currently targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.staged_max_numel = staged_max_numel
        self.producer_direct_max_numel = producer_direct_max_numel
        self.attnres_max_numel = attnres_max_numel
        self.attnres_max_rows = attnres_max_rows
        self.enable_lamport = enable_lamport
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = group.size()
        if (
            min(
                staged_max_numel,
                producer_direct_max_numel,
                attnres_max_numel,
                attnres_max_rows,
            )
            < 0
        ):
            raise ValueError("Iris all-reduce capacities must be non-negative")
        if bool(attnres_max_numel) != bool(attnres_max_rows):
            raise ValueError(
                "AttnRes element and row capacities must both be zero or non-zero"
            )
        self._kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
        producer_config = self._kernel_config.producer_direct
        staged_config = self._kernel_config.staged
        two_stage_config = self._kernel_config.two_stage
        moe_config = self._kernel_config.packed
        self._elements_per_word = (
            self._kernel_config.packed_word_bytes // dtype.itemsize
        )
        # Reserve complete eligible rows. One program owns one tile for every
        # invocation, independently of the pull path's 84-program cap.
        self._lamport_max_numel = (
            min(
                producer_direct_max_numel // moe_config.row_numel,
                moe_config.lamport_max_rows,
            )
            * moe_config.row_numel
            if enable_lamport
            and _platform.is_cdna4
            and self.world_size == moe_config.world_size
            and dtype == torch.bfloat16
            else 0
        )
        self._lamport_max_programs = (
            self._lamport_max_numel // moe_config.lamport_block_elements
        )
        self._producer_direct_two_stage_workspace_required = (
            producer_direct_max_numel > 0
            and producer_config.two_stage_threshold(self.world_size) is not None
            and two_stage_config.supports_world_size(self.world_size)
        )
        self._producer_direct_scratch_numel = (
            two_stage_config.scratch_numel(
                max_numel=producer_direct_max_numel,
                world_size=self.world_size,
            )
            if self._producer_direct_two_stage_workspace_required
            else 0
        )
        self._producer_direct_max_programs = max(
            producer_config.one_stage_max_programs,
            (
                two_stage_config.max_programs
                if self._producer_direct_two_stage_workspace_required
                else 0
            ),
        )
        self._staged_max_programs = staged_config.max_programs(
            max_numel=staged_max_numel
        )
        self._staged_tunings = tuple(
            tuning
            for tuning in staged_config.cdna4_tunings
            if _platform.is_cdna4
            and tuning.world_size == self.world_size
            and tuning.dtype == dtype
            and tuning.numel <= staged_max_numel
        )

        # Whether this state can ever dispatch a two-stage reduction. Platform,
        # group size and element type are all fixed for the life of the state,
        # so deciding once here keeps the buffers, the heap estimate and the
        # dispatch from disagreeing. Only the payload size is per-call, and it
        # stays in _use_two_stage_plain.
        self._staged_two_stage_supported = (
            _platform.is_cdna4
            and amd_collectives_available()
            and staged_max_numel > 0
            and two_stage_config.supports_world_size(self.world_size)
            and dtype in _PRODUCER_DIRECT_DTYPES
        )
        self._staged_two_stage_scratch_numel = (
            two_stage_config.scratch_numel(
                max_numel=staged_max_numel,
                world_size=self.world_size,
            )
            if self._staged_two_stage_supported
            else 0
        )

        if heap_size is None:
            payload_numel = (
                producer_direct_max_numel
                + self._producer_direct_scratch_numel
                + staged_config.input_slots * staged_max_numel
                + staged_config.input_slots
                * sum(tuning.numel for tuning in self._staged_tunings)
                + 2 * self.world_size * attnres_max_numel
                + (staged_max_numel if self._staged_two_stage_supported else 0)
                + self._staged_two_stage_scratch_numel
                + moe_config.lamport_stages * self.world_size * self._lamport_max_numel
            )
            flag_numel = self.world_size * (
                self._staged_max_programs
                + sum(tuning.num_programs() for tuning in self._staged_tunings)
                + 2 * attnres_max_rows
                + (
                    self._producer_direct_max_programs
                    if producer_direct_max_numel
                    else 0
                )
                + (
                    two_stage_config.max_programs
                    if self._staged_two_stage_supported
                    else 0
                )
            )
            heap_size = max(
                1 << 28,
                payload_numel * dtype.itemsize
                + flag_numel * torch.int32.itemsize
                + (16 << 20),
            )

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        group_ranks = dist.get_process_group_ranks(group)
        assert len(group_ranks) == self.world_size
        assert group_ranks[rank_in_group] == dist.get_rank()
        _input_buf = (
            self._ctx.zeros((producer_direct_max_numel,), dtype=dtype)
            if producer_direct_max_numel
            else None
        )
        _attnres_push_inbox = (
            self._ctx.zeros((2, self.world_size, attnres_max_numel), dtype=dtype)
            if attnres_max_numel
            else None
        )
        _attnres_push_epochs = (
            torch.zeros((attnres_max_rows,), dtype=torch.int32, device=self.device)
            if attnres_max_numel
            else None
        )
        _attnres_push_ready_flags = (
            self._ctx.zeros((2, attnres_max_rows, self.world_size), dtype=torch.int32)
            if attnres_max_numel
            else None
        )
        _producer_direct_scratch_buf = (
            self._ctx.zeros((self._producer_direct_scratch_numel,), dtype=dtype)
            if self._producer_direct_scratch_numel
            else None
        )
        _reduced_output_buf = (
            torch.empty(producer_direct_max_numel, dtype=dtype, device=self.device)
            if producer_direct_max_numel
            else None
        )
        _ready_flags = (
            self._ctx.zeros(
                (self._staged_max_programs, self.world_size), dtype=torch.int32
            )
            if staged_max_numel
            else None
        )
        # The staged one-shot rotates across its own slots rather than sharing
        # _input_buf: that buffer is handed out by acquire_outputs for
        # producer-direct reductions, so its layout is not ours to rotate.
        _staged_input_buf = (
            self._ctx.zeros((staged_config.input_slots, staged_max_numel), dtype=dtype)
            if staged_max_numel
            else None
        )
        # Different block geometries cannot share per-block epochs or rotating
        # slots because their block IDs cover overlapping element ranges.
        _staged_tuning_workspaces = {
            tuning: (
                self._ctx.zeros((staged_config.input_slots, tuning.numel), dtype=dtype),
                self._ctx.zeros(
                    (tuning.num_programs(), self.world_size), dtype=torch.int32
                ),
            )
            for tuning in self._staged_tunings
        }
        # Two-stage plain all-reduce. One-shot stages inside its own kernel and
        # picks a slot from its per-block epoch; the two-stage kernel instead
        # reads peers' inputs out of symmetric memory, so the payload has to be
        # staged before launch. That staging cannot share _staged_input_buf --
        # its slot is chosen by one-shot's kernel from a counter we cannot see --
        # nor _input_buf, which acquire_outputs hands to producer-direct.
        #
        # A single buffer, deliberately: rotating slots from the host does not
        # survive graph capture, which records the staging copy's address and
        # replays it unchanged. The kernel's exit barrier is what makes one
        # buffer safe -- see EXIT_BARRIER.
        #
        # Skipped entirely where dispatch could never reach them: a CDNA3 part
        # or a group of some other size would otherwise reserve a payload
        # buffer and a scratch partition it can never use, and an explicit
        # heap_size sized for the previous allocations would fail to fit them.
        if self._staged_two_stage_supported:
            _staged_two_stage_input_buf = self._ctx.zeros(
                (staged_max_numel,), dtype=dtype
            )
            # Each rank reduces only its own partition and peers read it at the
            # same offset, so scratch holds one partition, not the whole payload.
            _staged_two_stage_scratch_buf = self._ctx.zeros(
                (self._staged_two_stage_scratch_numel,), dtype=dtype
            )
        else:
            _staged_two_stage_input_buf = None
            _staged_two_stage_scratch_buf = None
        _producer_direct_ready_flags = (
            self._ctx.zeros(
                (
                    self._producer_direct_max_programs,
                    self.world_size,
                ),
                dtype=torch.int32,
            )
            if producer_direct_max_numel
            else None
        )
        _lamport_region = (
            self._ctx.zeros(
                (
                    moe_config.lamport_stages,
                    self.world_size,
                    self._lamport_max_numel,
                ),
                dtype=dtype,
            )
            if self._lamport_max_numel
            else None
        )
        _lamport_epochs = (
            torch.zeros(
                (self._lamport_max_programs,),
                dtype=torch.int32,
                device=self.device,
            )
            if self._lamport_max_numel
            else None
        )
        if _lamport_region is not None:
            # Alternating +0/-0 halves: every lane's 16-byte pack has sentinel
            # halves. Legitimate -0 inputs are normalized before publication.
            _lamport_region.view(torch.int32).fill_(-2147483648)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
        # Separate epochs from the producer-direct reduce: in a tensor-parallel
        # MoE both collectives run inside one layer, and a shared counter would
        # let one path's epoch satisfy the other's barrier.
        _staged_two_stage_ready_flags = (
            self._ctx.zeros(
                (two_stage_config.max_programs, self.world_size),
                dtype=torch.int32,
            )
            if self._staged_two_stage_supported
            else None
        )
        if _attnres_push_inbox is not None:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
        heap_bases = self._ctx.get_heap_bases()
        self._group_heap_bases = heap_bases[group_ranks].contiguous()
        group_heap_bases = [int(address) for address in self._group_heap_bases.tolist()]
        self._heap_base_addresses = tuple(
            group_heap_bases + [group_heap_bases[-1]] * (8 - self.world_size)
        )
        _lamport_peer_addresses = None
        if _lamport_region is not None:
            heap_offset = (
                _lamport_region.data_ptr() - self._heap_base_addresses[rank_in_group]
            )
            _lamport_peer_addresses = tuple(
                heap_base + heap_offset for heap_base in self._heap_base_addresses
            )
            assert all(
                address % moe_config.lamport_transaction_bytes == 0
                for address in _lamport_peer_addresses
            )
        _attnres_push_peer_inboxes = (
            _peer_addresses(
                _attnres_push_inbox,
                self._heap_base_addresses,
                rank_in_group,
            )
            if _attnres_push_inbox is not None
            else None
        )
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris all-reduce symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = rank_in_group
        self._workspace = None
        # Construct ownership records after allocating in the original symmetric
        # order. Record boundaries must never reorder heap allocations.
        self.staged = StagedWorkspace(
            _staged_input_buf, _ready_flags, _staged_tuning_workspaces
        )
        self.two_stage = TwoStageWorkspace(
            _staged_two_stage_input_buf,
            _staged_two_stage_scratch_buf,
            _staged_two_stage_ready_flags,
        )
        self.producer = ProducerWorkspace(
            _input_buf,
            _producer_direct_scratch_buf,
            _reduced_output_buf,
            _producer_direct_ready_flags,
        )
        self.lamport = LamportWorkspace(
            _lamport_region, _lamport_epochs, _lamport_peer_addresses
        )
        self.attnres = AttnResWorkspace(
            _attnres_push_inbox,
            _attnres_push_epochs,
            _attnres_push_ready_flags,
            _attnres_push_peer_inboxes,
        )
