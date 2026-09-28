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


"""All-reduce with a residual and online-softmax AttnRes epilogue."""

from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.communication._common import (
    _iris_drain_subgroup_vmem,
    _iris_heap_base,
    _iris_sync_rank_epoch,
    _iris_sync_rank_token,
    collective_launch_metadata,
)


@gluon.jit
def _iris_attnres_epilogue(
    reduced,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    weight_offsets,
    mask,
    row,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(
        residual_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        row_offsets,
        mask=mask,
    )
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    mixed = (
        ((block_acc * block_correction + prefix_weight * prefix) * inverse_sum)
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        row_offsets,
        mask=mask,
    )


@gluon.jit(launch_metadata=collective_launch_metadata)
def iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel(
    partial_ptr,
    residual_ptr,
    input_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    ready_flags,
    consumed_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    M: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    INPUT_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """AttnRes attention AR, residual, and split AttnRes combine."""
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    offset = gl.arange(0, BLOCK, layout=layout)
    mask = offset < HIDDEN
    offset_i32 = (row * HIDDEN + offset).to(gl.int32)
    weight_offset_i32 = offset.to(gl.int32)

    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    local_ready = ready_flags + row * WORLD_SIZE + RANK
    epoch = gl.load(local_ready).to(gl.int32) + 1
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    reuse_epoch = gl.maximum(epoch - 2, 0)
    _iris_sync_rank_epoch(
        consumed_flags,
        row,
        reuse_epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=False,
    )

    input_slot_ptr = input_sym_ptr + (epoch & 1) * INPUT_SLOT_STRIDE
    gl.amd.cdna4.buffer_store(
        local.to(input_slot_ptr.dtype.element_ty),
        input_slot_ptr,
        offset_i32,
        mask=mask,
        cache=".wt",
    )

    gl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")
    _iris_sync_rank_epoch(
        ready_flags,
        row,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    input_heap_offset = tl.cast(input_slot_ptr, gl.uint64) - local_heap
    reduced = local
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            peer_heap = _iris_heap_base(
                peer,
                heap_base_0,
                heap_base_1,
                heap_base_2,
                heap_base_3,
                heap_base_4,
                heap_base_5,
                heap_base_6,
                heap_base_7,
            )
            peer_input = tl.cast(
                peer_heap + input_heap_offset,
                partial_ptr.dtype,
            )
            reduced += gl.amd.cdna4.buffer_load(
                peer_input,
                offset_i32,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    # Publish consumption without serializing this epilogue. Reuse waits only
    # when a later invocation wraps back to the same staging slot.
    consumed = consumed_flags + row * WORLD_SIZE + RANK
    gl.atomic_xchg(consumed, epoch, sem="release", scope="sys")

    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        offset_i32,
        weight_offset_i32,
        mask,
        row,
        HIDDEN,
        EPS,
    )


@gluon.jit(launch_metadata=collective_launch_metadata)
def iris_push_one_shot_allreduce_residual_attnres_gluon_kernel(
    partial_ptr,
    residual_ptr,
    inbox_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    generations,
    ready_flags,
    inbox_0: gl.pointer_type(gl.bfloat16),
    inbox_1: gl.pointer_type(gl.bfloat16),
    inbox_2: gl.pointer_type(gl.bfloat16),
    inbox_3: gl.pointer_type(gl.bfloat16),
    inbox_4: gl.pointer_type(gl.bfloat16),
    inbox_5: gl.pointer_type(gl.bfloat16),
    inbox_6: gl.pointer_type(gl.bfloat16),
    inbox_7: gl.pointer_type(gl.bfloat16),
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    READY_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Push AttnRes attention rows into two-slot rank-ordered inboxes.

    Peer inboxes are host-computed byte addresses. Explicit BF16 pointer
    annotations preserve their pointer ABI and element-wise device offsets
    without a Python pointer wrapper. The Iris context owns the mappings.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    )
    generation = gl.load(generations + row).to(gl.int32) + 1
    slot = generation & 1
    inbox_slot_offset = slot * WORLD_SIZE * MAX_ELEMENTS
    sync_ready_flags = ready_flags + slot * READY_SLOT_STRIDE
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    for peer_delta in gl.static_range(1, WORLD_SIZE):
        destination = (RANK + peer_delta) % WORLD_SIZE
        destination_inbox = _iris_heap_base(
            destination,
            inbox_0,
            inbox_1,
            inbox_2,
            inbox_3,
            inbox_4,
            inbox_5,
            inbox_6,
            inbox_7,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination_inbox + inbox_slot_offset + RANK * MAX_ELEMENTS,
            row_offsets,
            mask=mask,
            cache=".wt",
        )
    _iris_drain_subgroup_vmem()
    # The drain is subgroup-local. Join all producer subgroups before the
    # control subgroup publishes the generation to peer ranks.
    gl.barrier()

    _iris_sync_rank_token(
        sync_ready_flags,
        row,
        generation,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
    )
    local_inbox = inbox_sym_ptr + inbox_slot_offset

    if RANK == 0:
        reduced = local.to(gl.float32)
    else:
        reduced = gl.amd.cdna4.buffer_load(
            local_inbox,
            row_offsets,
            mask=mask,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
    for source in gl.static_range(1, WORLD_SIZE):
        if source == RANK:
            reduced += local.to(gl.float32)
        else:
            reduced += gl.amd.cdna4.buffer_load(
                local_inbox + source * MAX_ELEMENTS,
                row_offsets,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    gl.store(generations + row, generation)
    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        weight_offsets,
        mask,
        row,
        HIDDEN,
        EPS,
    )
