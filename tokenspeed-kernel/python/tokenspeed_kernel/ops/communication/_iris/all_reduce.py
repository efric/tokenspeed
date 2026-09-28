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


"""Staged and producer-direct Iris all-reduce host launchers."""

import logging
import math

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import gl, triton
from tokenspeed_kernel.ops.communication._iris.policy import (
    _packed_producer_direct_protocol,
    _select_staged_all_reduce_path,
    _use_two_stage_producer_direct,
)
from tokenspeed_kernel.ops.communication._iris.triton import (
    iris_stage_one_shot_allreduce_kernel,
)
from tokenspeed_kernel.ops.communication._iris.workspace import IrisAllReduceWorkspace
from tokenspeed_kernel.platform import current_platform

logger = logging.getLogger(__name__)
_platform = current_platform()


_PRODUCER_DIRECT_GL_DTYPES = {
    torch.bfloat16: gl.bfloat16,
    torch.float16: gl.float16,
    torch.float32: gl.float32,
}


class IrisAllReduce(IrisAllReduceWorkspace):
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op=None,
        safe: bool = True,
        async_op: bool = False,
    ) -> torch.Tensor:
        assert tensor.is_contiguous(), "Iris requires a contiguous input"
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert not async_op, "Iris all-reduce does not support async_op"
        assert tensor.dtype == self.dtype, (
            f"Iris all-reduce dtype mismatch: tensor={tensor.dtype}, "
            f"backend={self.dtype}"
        )
        numel = tensor.numel()
        assert 0 < numel <= self.staged_max_numel, (
            f"tensor numel ({numel}) exceeds iris buffer capacity "
            f"({self.staged_max_numel})"
        )
        kernel_config = self._kernel_config.staged
        tuning, use_two_stage = _select_staged_all_reduce_path(
            numel=numel,
            world_size=self.world_size,
            dtype=self.dtype,
            two_stage_supported=self._staged_two_stage_supported,
        )
        # One-shot has every rank publish the whole payload and read world_size
        # copies of it, so its cost grows with the payload; the two-stage form
        # moves 2x however wide the world is. Measured on gfx950 at world 8,
        # including the staging copy this path pays and one-shot does not:
        # parity below ~16 tokens of hidden 7168, then 1.11x at 16, 1.37x at 32,
        # 1.82x at 64. The predicate is the kernel's partitioning requirement,
        # not a crossover -- see _use_two_stage_plain.
        # The two-stage kernel stores through a CDNA4 buffer intrinsic, so it is
        # gated the same way the producer-direct reduce is; older AMD parts keep
        # the portable one-shot path.
        # Keep an explicitly tuned one-shot shape on that path. On TP4 gfx950,
        # one-shot remains faster for 16x4096 while two-stage wins at 64x4096.
        if use_two_stage:
            return self._all_reduce_two_stage(tensor, numel, safe=safe)
        if tuning is None:
            block_size = kernel_config.block_size
            num_subgroups = kernel_config.num_subgroups
            input_buf = self.staged.input
            ready_flags = self.staged.flags
            slot_stride = self.staged_max_numel
        else:
            block_size = tuning.block_size
            num_subgroups = tuning.num_subgroups
            input_buf, ready_flags = self.staged.tunings[tuning]
            slot_stride = tuning.numel
        assert input_buf is not None and ready_flags is not None
        iris_stage_one_shot_allreduce_kernel[(triton.cdiv(numel, block_size),)](
            tensor.view(-1),
            input_buf.view(-1),
            tensor.view(-1),
            ready_flags,
            self._group_heap_bases,
            numel,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            BLOCK_SIZE=block_size,
            SLOT_STRIDE=slot_stride,
            NUM_SLOTS=kernel_config.input_slots,
            num_warps=num_subgroups,
        )

        return tensor.clone() if safe else tensor

    @staticmethod
    def _views(
        buffer: torch.Tensor,
        shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        views = []
        offset = 0
        for shape in shapes:
            numel = math.prod(shape)
            views.append(buffer.narrow(0, offset, numel).view(shape))
            offset += numel
        return tuple(views)

    def acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return consecutive views of the symmetric Iris input buffer."""
        if not shapes or any(math.prod(shape) <= 0 for shape in shapes):
            raise ValueError("Iris requires non-empty symmetric output shapes")
        if sum(math.prod(shape) for shape in shapes) > self.producer_direct_max_numel:
            raise ValueError("Iris symmetric outputs exceed the input buffer")
        assert self.producer.input is not None
        return self._views(self.producer.input, shapes)

    def owns_outputs(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        """Whether tensors are consecutive views of this symmetric buffer."""
        if not tensors or any(
            tensor.dtype != self.dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            for tensor in tensors
        ):
            return False
        if self.producer.input is None:
            return False
        element_size = self.producer.input.element_size()
        offset = 0
        for tensor in tensors:
            if (
                tensor.data_ptr()
                != self.producer.input.data_ptr() + offset * element_size
            ):
                return False
            offset += tensor.numel()
        return (
            offset % self._elements_per_word == 0
            and offset <= self.producer_direct_max_numel
        )

    def _all_reduce_two_stage(
        self, tensor: torch.Tensor, numel: int, safe: bool
    ) -> torch.Tensor:
        """Reduce ``tensor`` in place via reduce-scatter then all-gather.

        An unaligned destination uses a temporary output and copy-back. That
        repair changes no rank's collective protocol or result ownership.

        Args:
            tensor: Contiguous local contribution; overwritten with the sum.
            numel: ``tensor.numel()``, already checked against the heap capacity.
            safe: Return a copy rather than the caller's tensor, matching the
                one-shot path -- the reduction lands in place either way, so
                without this the result aliases the input.

        Returns:
            The reduction across the group; a clone of ``tensor`` when ``safe``.
        """
        from tokenspeed_kernel_amd.ops.gfx950.communication.all_reduce import (
            iris_reduce_symmetric_two_stage_gluon_kernel,
        )

        staged = self.two_stage.input
        staged[:numel].copy_(tensor.view(-1))

        partition_numel = numel // self.world_size
        partition_words = partition_numel // self._elements_per_word
        kernel_config = self._kernel_config.two_stage
        block_words = kernel_config.block_words(
            world_size=self.world_size,
            subgroup_size=self._kernel_config.subgroup_size,
        )
        num_tiles = triton.cdiv(partition_words, block_words)
        num_programs = min(num_tiles, kernel_config.max_programs)
        output = tensor.view(-1)
        copy_output = output.data_ptr() % self._kernel_config.packed_word_bytes != 0
        if copy_output:
            output = torch.empty_like(output)
        iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
            staged,
            self.two_stage.scratch,
            output,
            self.two_stage.flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            PARTITION_WORDS=partition_words,
            BLOCK_WORDS=block_words,
            NUM_PROGRAMS=num_programs,
            NUM_TILES=num_tiles,
            NUM_WARPS=kernel_config.num_subgroups,
            SUBGROUP_SIZE=self._kernel_config.subgroup_size,
            WORDS_PER_LANE=kernel_config.words_per_lane,
            ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
            ELEMENTS_PER_WORD=self._elements_per_word,
            EXIT_BARRIER=True,
            num_warps=kernel_config.num_subgroups,
        )
        if copy_output:
            tensor.view(-1).copy_(output)
        return tensor.clone() if safe else tensor

    def all_reduce_symmetric(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Reduce consecutive producer outputs from symmetric memory."""
        assert self.owns_outputs(tensors)
        if not _platform.is_cdna4:
            raise RuntimeError("producer-direct Iris all-reduce requires CDNA4")
        kernel_config = self._kernel_config.producer_direct
        if not kernel_config.supports_world_size(self.world_size):
            raise RuntimeError(
                "producer-direct Iris all-reduce does not support group size "
                f"{self.world_size}"
            )
        if self.dtype not in _PRODUCER_DIRECT_GL_DTYPES:
            raise RuntimeError(
                f"producer-direct Iris all-reduce does not support {self.dtype}"
            )

        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        total_numel = sum(tensor.numel() for tensor in tensors)
        assert self.producer.output is not None
        outputs = self._views(
            self.producer.output,
            shapes,
        )
        if (
            self.enable_lamport
            and _packed_producer_direct_protocol(self.world_size, shapes, self.dtype)
            == "lamport"
        ):
            self._all_reduce_symmetric_lamport(total_numel)
        else:
            self._all_reduce_symmetric_pull(total_numel)
        return outputs

    def _all_reduce_symmetric_lamport(self, total_numel: int) -> None:
        from tokenspeed_kernel_amd.ops.gfx950.communication.all_reduce import (
            lamport_all_reduce_bf16,
        )

        config = self._kernel_config.packed
        assert total_numel <= self._lamport_max_numel
        assert self.lamport.region is not None
        assert self.lamport.epochs is not None
        assert self.lamport.peer_addresses is not None
        num_programs = total_numel // config.lamport_block_elements
        lamport_all_reduce_bf16[(num_programs,)](
            self.producer.input,
            self.lamport.region,
            self.producer.output,
            self.lamport.epochs,
            *self.lamport.peer_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            TOTAL_ELEMENTS=total_numel,
            MAX_ELEMENTS=self._lamport_max_numel,
            NUM_STAGES=config.lamport_stages,
            num_warps=config.lamport_num_subgroups,
        )

    def _all_reduce_symmetric_pull(self, total_numel: int) -> None:
        from tokenspeed_kernel_amd.ops.gfx950.communication.all_reduce import (
            iris_reduce_symmetric_gluon_kernel,
            iris_reduce_symmetric_two_stage_gluon_kernel,
        )

        kernel_config = self._kernel_config.producer_direct
        use_two_stage = _use_two_stage_producer_direct(
            world_size=self.world_size,
            total_numel=total_numel,
            dtype=self.dtype,
        )
        if use_two_stage:
            two_stage_config = self._kernel_config.two_stage
            assert self.producer.scratch is not None
            partition_numel = total_numel // self.world_size
            partition_words = partition_numel // self._elements_per_word
            block_words = two_stage_config.block_words(
                world_size=self.world_size,
                subgroup_size=self._kernel_config.subgroup_size,
            )
            num_tiles = triton.cdiv(partition_words, block_words)
            num_programs = min(num_tiles, two_stage_config.max_programs)
            iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
                self.producer.input,
                self.producer.scratch,
                self.producer.output,
                self.producer.flags,
                *self._heap_base_addresses,
                RANK=self._iris_rank,
                WORLD_SIZE=self.world_size,
                PARTITION_WORDS=partition_words,
                BLOCK_WORDS=block_words,
                NUM_PROGRAMS=num_programs,
                NUM_TILES=num_tiles,
                NUM_WARPS=two_stage_config.num_subgroups,
                SUBGROUP_SIZE=self._kernel_config.subgroup_size,
                WORDS_PER_LANE=two_stage_config.words_per_lane,
                ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
                ELEMENTS_PER_WORD=self._elements_per_word,
                EXIT_BARRIER=False,
                num_warps=two_stage_config.num_subgroups,
            )
        else:
            block_size = kernel_config.one_stage_block_size
            num_tiles = triton.cdiv(total_numel, block_size)
            num_programs = min(num_tiles, kernel_config.one_stage_max_programs)
            iris_reduce_symmetric_gluon_kernel[(num_programs,)](
                self.producer.input,
                self.producer.output,
                self.producer.flags,
                *self._heap_base_addresses,
                RANK=self._iris_rank,
                WORLD_SIZE=self.world_size,
                TOTAL_NUMEL=total_numel,
                BLOCK_SIZE=block_size,
                NUM_PROGRAMS=num_programs,
                NUM_TILES=num_tiles,
                NUM_WARPS=kernel_config.one_stage_num_subgroups,
                SUBGROUP_SIZE=self._kernel_config.subgroup_size,
                WORDS_PER_LANE=kernel_config.one_stage_words_per_lane,
                PUBLISH_READY=kernel_config.publish_ready,
                ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
                ELEMENTS_PER_WORD=self._elements_per_word,
                num_warps=kernel_config.one_stage_num_subgroups,
            )
