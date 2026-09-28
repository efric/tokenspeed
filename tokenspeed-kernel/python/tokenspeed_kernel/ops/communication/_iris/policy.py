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


"""Iris implementation capabilities, launch geometry, and measured policy."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()
_PRODUCER_DIRECT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@dataclass(frozen=True)
class IrisCapacities:
    staged_max_numel: int
    producer_direct_max_numel: int
    attnres_max_numel: int
    attnres_max_rows: int
    enable_lamport: bool
    moe_tail_max_rows: int


def resolve_capacities(
    preparation, world_size: int, ordinary_backing_bytes: int
) -> IrisCapacities:
    """Resolve measured admission separately from physical storage demands."""
    from tokenspeed_kernel.ops.communication._contracts import (
        AllReduceRequirement,
        MoETailRequirement,
        PackedAllReduceRequirement,
    )
    from tokenspeed_kernel.ops.residual.attnres import AttnResRequirement

    staged = producer = attnres = rows = moe_tail_rows = 0
    enable_lamport = False
    has_attnres = any(
        isinstance(op, AttnResRequirement) for op in preparation.operations
    )
    for op in preparation.operations:
        max_rows = min(op.max_rows, 8192)
        if isinstance(op, AllReduceRequirement):
            staged = max(staged, max_rows * op.width)
        elif isinstance(op, AttnResRequirement):
            admitted_rows = min(max_rows, 16) if world_size == 8 else 0
            rows = max(rows, admitted_rows)
            attnres = max(attnres, admitted_rows * op.width)
        elif isinstance(op, PackedAllReduceRequirement):
            if op.tensor_parallel_size * op.expert_parallel_size != world_size:
                raise ValueError("producer group axes do not match the reduction group")
            admitted_rows = (
                max_rows if has_attnres and world_size == 8 else min(max_rows, 48)
            )
            producer = max(producer, admitted_rows * sum(op.widths))
            enable_lamport |= (
                has_attnres
                and world_size == 8
                and op.tensor_parallel_size == 8
                and op.expert_parallel_size == 1
            )
        elif isinstance(op, MoETailRequirement):
            config = IRIS_ALL_REDUCE_KERNEL_CONFIG.packed
            if (
                world_size != config.world_size
                or preparation.dtype != torch.bfloat16
                or (op.routed_width, op.hidden_width)
                != (config.routed_hidden_size, config.hidden_size)
            ):
                raise ValueError("unsupported token-sharded MoE-tail requirement")
            admitted_rows = max_rows // world_size * world_size
            if admitted_rows >= 512:
                moe_tail_rows = max(moe_tail_rows, admitted_rows)
                producer = max(producer, admitted_rows * config.row_numel)
        else:
            raise TypeError(f"unsupported collective requirement: {type(op).__name__}")
    staged = min(
        staged, min(ordinary_backing_bytes, 512 * 1024) // preparation.dtype.itemsize
    )
    return IrisCapacities(
        staged, producer, attnres, rows, enable_lamport, moe_tail_rows
    )


@dataclass(frozen=True)
class _StagedAllReduceKernelTuning:
    world_size: int
    dtype: torch.dtype
    numel: int
    block_size: int
    num_subgroups: int

    def __post_init__(self) -> None:
        if (
            self.world_size <= 1
            or self.numel <= 0
            or self.block_size <= 0
            or self.num_subgroups <= 0
        ):
            raise ValueError("invalid staged Iris kernel tuning")

    def num_programs(self) -> int:
        return triton.cdiv(self.numel, self.block_size)


@dataclass(frozen=True)
class _StagedAllReduceKernelConfig:
    block_size: int
    num_subgroups: int
    input_slots: int
    cdna4_tunings: tuple[_StagedAllReduceKernelTuning, ...]

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.num_subgroups <= 0 or self.input_slots < 2:
            raise ValueError("invalid staged Iris kernel configuration")

    def max_programs(self, max_numel: int) -> int:
        return triton.cdiv(max_numel, self.block_size)

    def tuning(
        self,
        numel: int,
        world_size: int,
        dtype: torch.dtype,
        is_cdna4: bool,
    ) -> _StagedAllReduceKernelTuning | None:
        if not is_cdna4:
            return None
        for tuning in self.cdna4_tunings:
            if (
                tuning.world_size == world_size
                and tuning.dtype == dtype
                and tuning.numel == numel
            ):
                return tuning
        return None


@dataclass(frozen=True)
class _ProducerDirectAllReduceKernelConfig:
    supported_world_sizes: tuple[int, ...]
    one_stage_block_size: int
    one_stage_max_programs: int
    one_stage_num_subgroups: int
    one_stage_words_per_lane: int
    two_stage_min_bytes: tuple[tuple[int, int], ...]
    publish_ready: bool

    def __post_init__(self) -> None:
        if (
            not self.supported_world_sizes
            or len(set(self.supported_world_sizes)) != len(self.supported_world_sizes)
            or any(world_size <= 1 for world_size in self.supported_world_sizes)
            or self.one_stage_block_size <= 0
            or self.one_stage_max_programs <= 0
            or self.one_stage_num_subgroups <= 0
            or self.one_stage_words_per_lane <= 0
        ):
            raise ValueError("invalid producer-direct Iris kernel configuration")
        threshold_world_sizes = tuple(
            world_size for world_size, _ in self.two_stage_min_bytes
        )
        if len(set(threshold_world_sizes)) != len(threshold_world_sizes) or any(
            world_size not in self.supported_world_sizes or min_bytes <= 0
            for world_size, min_bytes in self.two_stage_min_bytes
        ):
            raise ValueError("invalid producer-direct Iris two-stage thresholds")

    def supports_world_size(self, world_size: int) -> bool:
        return world_size in self.supported_world_sizes

    def two_stage_threshold(self, world_size: int) -> int | None:
        return dict(self.two_stage_min_bytes).get(world_size)


@dataclass(frozen=True)
class _TwoStageAllReduceKernelConfig:
    supported_world_sizes: tuple[int, ...]
    max_programs: int
    num_subgroups: int
    words_per_lane: int

    def __post_init__(self) -> None:
        if (
            not self.supported_world_sizes
            or len(set(self.supported_world_sizes)) != len(self.supported_world_sizes)
            or self.num_subgroups <= 0
            or any(
                world_size <= 1 or self.num_subgroups % world_size != 0
                for world_size in self.supported_world_sizes
            )
            or self.max_programs <= 0
            or self.words_per_lane <= 0
        ):
            raise ValueError("invalid two-stage Iris kernel configuration")

    def supports_world_size(self, world_size: int) -> bool:
        return world_size in self.supported_world_sizes

    def can_partition(
        self,
        world_size: int,
        total_numel: int,
        elements_per_word: int,
    ) -> bool:
        return (
            self.supports_world_size(world_size)
            and total_numel % (world_size * elements_per_word) == 0
        )

    def scratch_numel(self, max_numel: int, world_size: int) -> int:
        if max_numel == 0 or not self.supports_world_size(world_size):
            return 0
        return triton.cdiv(max_numel, world_size)

    def block_words(self, world_size: int, subgroup_size: int) -> int:
        assert self.supports_world_size(world_size)
        return self.num_subgroups * subgroup_size * self.words_per_lane // world_size


@dataclass(frozen=True)
class AttnResKernelConfig:
    """AttnRes fused attention-TP all-reduce and AttnRes contract.

    Attributes:
        world_size: Required attention tensor-parallel group size.
        hidden_size: AttnRes attention output width.
        num_subgroups: Number of subgroups in each kernel workgroup.
        elements_per_thread: Number of hidden elements processed per thread.
    """

    world_size: int
    hidden_size: int
    num_subgroups: int
    elements_per_thread: int

    def __post_init__(self) -> None:
        if (
            self.world_size <= 1
            or self.hidden_size <= 0
            or self.num_subgroups <= 0
            or self.elements_per_thread <= 0
        ):
            raise ValueError("invalid AttnRes Iris kernel configuration")


@dataclass(frozen=True)
class PackedAllReduceKernelConfig:
    """Shape and mailbox contract for the CDNA4 packed BF16 Lamport all-reduce.

    Attributes:
        world_size: Required communication group size.
        routed_hidden_size: Width of the routed-expert output.
        hidden_size: Width of the shared-expert output.
        lamport_max_rows: Largest row count using Lamport instead of pull.
        lamport_stages: Number of mailbox generations before reuse.
        lamport_block_elements: Elements owned by one workgroup.
        lamport_num_subgroups: Subgroups per workgroup; polling uses one.
        lamport_transaction_bytes: Width of each lane's publication/read.
    """

    world_size: int
    routed_hidden_size: int
    hidden_size: int
    lamport_max_rows: int
    lamport_stages: int
    lamport_block_elements: int
    lamport_num_subgroups: int
    lamport_transaction_bytes: int
    tail_reduce_scatter_programs: int
    tail_gather_programs: int

    def __post_init__(self) -> None:
        if (
            self.world_size != 8
            or self.routed_hidden_size <= 0
            or self.hidden_size <= 0
            or self.lamport_max_rows <= 0
            or self.lamport_stages < 3
            or self.lamport_block_elements != 512
            or self.lamport_num_subgroups != 1
            or self.lamport_transaction_bytes != 16
            or self.row_numel % self.lamport_block_elements
            or self.tail_reduce_scatter_programs <= 0
            or self.tail_gather_programs <= 0
        ):
            raise ValueError("invalid packed Lamport kernel configuration")

    @property
    def row_numel(self) -> int:
        return self.routed_hidden_size + self.hidden_size

    @property
    def lamport_max_numel(self) -> int:
        return self.lamport_max_rows * self.row_numel

    def rows_for_shapes(self, shapes: tuple[tuple[int, ...], ...]) -> int | None:
        if len(shapes) != 2 or any(len(shape) != 2 for shape in shapes):
            return None
        rows = shapes[0][0]
        if (
            rows <= 0
            or shapes[1][0] != rows
            or (shapes[0][1], shapes[1][1])
            not in (
                (self.routed_hidden_size, self.hidden_size),
                (self.hidden_size, self.routed_hidden_size),
            )
        ):
            return None
        return rows


@dataclass(frozen=True)
class IrisAllReduceKernelConfig:
    """Launch and workspace contract for TokenSpeed's Iris all-reduces.

    ``staged`` and ``producer_direct`` are generic AMD all-reduce paths.
    The producer-direct path supports independent outputs over TP and EP groups;
    it is not EP-specific. ``attnres`` is model-specific and operates
    only on AttnRes's attention tensor-parallel group.

    Attributes:
        subgroup_size: Hardware subgroup width used by the Gluon kernels.
        packed_word_bytes: Packed element width used by the symmetric kernels.
        staged: Launch and workspace parameters for staged all-reduce.
        producer_direct: Launch and eligibility parameters for producer-direct
            all-reduce.
        two_stage: Launch and workspace parameters shared by ordinary staged and
            producer-direct two-stage all-reduce.
        packed: Shape, launch, and mailbox parameters for packed Lamport.
        attnres: Launch and shape parameters for AttnRes.
    """

    subgroup_size: int
    packed_word_bytes: int
    staged: _StagedAllReduceKernelConfig
    producer_direct: _ProducerDirectAllReduceKernelConfig
    two_stage: _TwoStageAllReduceKernelConfig
    packed: PackedAllReduceKernelConfig
    attnres: AttnResKernelConfig

    def __post_init__(self) -> None:
        if (
            self.subgroup_size <= 0
            or self.subgroup_size & (self.subgroup_size - 1)
            or self.packed_word_bytes <= 0
        ):
            raise ValueError("invalid Iris all-reduce kernel configuration")
        if any(
            not self.two_stage.supports_world_size(world_size)
            for world_size, _ in self.producer_direct.two_stage_min_bytes
        ):
            raise ValueError(
                "producer-direct two-stage thresholds require kernel support"
            )
        if self.subgroup_size != 64:
            raise ValueError("AttnRes Lamport requires a 64-thread subgroup")


IRIS_ALL_REDUCE_KERNEL_CONFIG = IrisAllReduceKernelConfig(
    subgroup_size=64,
    packed_word_bytes=8,
    staged=_StagedAllReduceKernelConfig(
        block_size=2048,
        num_subgroups=4,
        input_slots=2,
        # This CDNA4 TP4 tuning came from GLM-5.3-Flash decode.
        cdna4_tunings=(
            _StagedAllReduceKernelTuning(
                world_size=4,
                dtype=torch.bfloat16,
                numel=16 * 4096,
                block_size=512,
                num_subgroups=1,
            ),
        ),
    ),
    producer_direct=_ProducerDirectAllReduceKernelConfig(
        supported_world_sizes=(2, 4, 8),
        one_stage_block_size=512,
        one_stage_max_programs=84,
        one_stage_num_subgroups=1,
        one_stage_words_per_lane=2,
        two_stage_min_bytes=((4, 160 << 10), (8, 96 << 10)),
        publish_ready=False,
    ),
    two_stage=_TwoStageAllReduceKernelConfig(
        supported_world_sizes=(4, 8),
        max_programs=84,
        num_subgroups=8,
        words_per_lane=2,
    ),
    packed=PackedAllReduceKernelConfig(
        world_size=8,
        routed_hidden_size=3584,
        hidden_size=7168,
        lamport_max_rows=6,
        lamport_stages=3,
        lamport_block_elements=512,
        lamport_num_subgroups=1,
        lamport_transaction_bytes=16,
        tail_reduce_scatter_programs=24,
        tail_gather_programs=128,
    ),
    attnres=AttnResKernelConfig(
        world_size=8,
        hidden_size=7168,
        num_subgroups=16,
        elements_per_thread=8,
    ),
)


def _packed_producer_direct_protocol(
    world_size: int,
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
) -> str | None:
    config = IRIS_ALL_REDUCE_KERNEL_CONFIG.packed
    if world_size != config.world_size or dtype != torch.bfloat16:
        return None
    rows = config.rows_for_shapes(shapes)
    return "lamport" if rows is not None and rows <= config.lamport_max_rows else None


def producer_direct_all_reduce_can_run(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
    max_bytes: int,
) -> bool:
    """Check the generic AMD producer-direct kernel's payload requirements.

    Args:
        world_size: Number of ranks participating in the all-reduce.
        total_numel: Total number of elements in the payload.
        dtype: Element type of the payload.
        max_bytes: Byte capacity of the producer-direct symmetric input buffer.

    Returns:
        Whether the group size and element type are supported and the payload is
        positive, packed-word aligned, and within the input-buffer capacity.
    """
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    config = kernel_config.producer_direct
    element_bytes = dtype.itemsize
    return (
        config.supports_world_size(world_size)
        and total_numel > 0
        and dtype in _PRODUCER_DIRECT_DTYPES
        and kernel_config.packed_word_bytes % element_bytes == 0
        and total_numel % (kernel_config.packed_word_bytes // element_bytes) == 0
        and total_numel * element_bytes <= max_bytes
    )


def _use_two_stage_producer_direct(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
) -> bool:
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    min_bytes = kernel_config.producer_direct.two_stage_threshold(world_size)
    if min_bytes is None or dtype not in _PRODUCER_DIRECT_DTYPES:
        return False
    elements_per_word = kernel_config.packed_word_bytes // dtype.itemsize
    return total_numel * dtype.itemsize >= min_bytes and (
        kernel_config.two_stage.can_partition(
            world_size=world_size,
            total_numel=total_numel,
            elements_per_word=elements_per_word,
        )
    )


def _use_two_stage_plain(
    world_size: int,
    numel: int,
    dtype: torch.dtype,
) -> bool:
    """Whether a plain all-reduce of ``numel`` should take the two-stage path.

    Args:
        world_size: Ranks participating in the reduction.
        numel: Elements in the tensor being reduced.
        dtype: Element type; sets how many elements pack into a 64-bit word.

    Returns:
        True when the two-stage reduce-scatter/all-gather can run this shape.

    Unlike the producer-direct threshold this carries no minimum size. Measured
    on gfx950 at world 8, the two forms are within noise of each other below
    about 16 tokens of hidden 7168 (one-shot is marginally ahead at some of those
    shapes), and two-stage pulls away above it: 1.11x at 16 tokens, 1.37x at 32,
    1.82x at 64. A minimum would buy nothing at the small end and risks sitting
    in the wrong place as shapes change, so the only condition kept is the
    kernel's structural one -- the payload has to split evenly into per-rank
    partitions of whole 64-bit words.
    """
    # The kernel packs elements into 64-bit words through
    # _PRODUCER_DIRECT_DTYPES; anything outside it (or wider than a word,
    # which would make elements_per_word zero) stays on one-shot.
    if dtype not in _PRODUCER_DIRECT_DTYPES:
        return False
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    elements_per_word = kernel_config.packed_word_bytes // dtype.itemsize
    return kernel_config.two_stage.can_partition(
        world_size=world_size,
        total_numel=numel,
        elements_per_word=elements_per_word,
    )


def _select_staged_all_reduce_path(
    numel: int,
    world_size: int,
    dtype: torch.dtype,
    two_stage_supported: bool,
) -> tuple[_StagedAllReduceKernelTuning | None, bool]:
    """Resolve the tuned one-shot override and two-stage dispatch together."""
    tuning = IRIS_ALL_REDUCE_KERNEL_CONFIG.staged.tuning(
        numel=numel,
        world_size=world_size,
        dtype=dtype,
        is_cdna4=_platform.is_cdna4,
    )
    use_two_stage = (
        tuning is None
        and two_stage_supported
        and _use_two_stage_plain(world_size, numel, dtype)
    )
    return tuning, use_two_stage
