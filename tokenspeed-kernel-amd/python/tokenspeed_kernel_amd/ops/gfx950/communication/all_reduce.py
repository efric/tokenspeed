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


"""Packed pull and Lamport all-reduce protocols for gfx950."""

from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.communication._common import (
    _iris_heap_base,
    _iris_sync_rank_epoch,
    _pack_word,
    _unpack_word,
    collective_launch_metadata,
)


@gluon.jit
def _iris_sanitize_lamport_bf16(values):
    bits = values.to(gl.uint16, bitcast=True)
    return gl.where(bits == 0x8000, 0, bits).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _iris_wait_lamport_peers(
    region,
    generation,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    values = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        values += (gl.full([64, 8], 0, gl.bfloat16, LAYOUT),)
    active = valid
    while gl.max(active.to(gl.int32), 0) != 0:
        loaded = ()
        # A lane reloads every peer until all seven packs are ready. Issue the
        # independent reads before checking them; retired lanes keep their data.
        # .cv controls hardware caches, not compiler volatility. The compiler
        # regression test checks that these reads remain in the polling cycle.
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            pointer = (
                region + generation * WORLD_SIZE * MAX_ELEMENTS + peer * MAX_ELEMENTS
            )
            loaded += (
                gl.amd.cdna4.buffer_load(
                    pointer,
                    offsets,
                    mask=active[:, None],
                    other=values[delta - 1],
                    cache=".cv",
                ),
            )
        active = gl.full([64], False, gl.int1, gl.SliceLayout(1, LAYOUT))
        for delta in gl.static_range(0, WORLD_SIZE - 1):
            active |= valid & (
                gl.max(
                    (loaded[delta].to(gl.uint16, bitcast=True) == 0x8000).to(gl.int32),
                    1,
                )
                != 0
            )
        values = loaded
    return values


@gluon.jit(launch_metadata=collective_launch_metadata)
def lamport_all_reduce_bf16(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """Push packed BF16 tiles and poll all peer packs with one subgroup per tile."""
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0)
    gl.static_assert(TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = _iris_heap_base(
            peer,
            region_0,
            region_1,
            region_2,
            region_3,
            region_4,
            region_5,
            region_6,
            region_7,
        )
        # Preserve 16-byte publication stores after casting integer addresses.
        destination = gl.multiple_of(destination.to(gl.pointer_type(gl.bfloat16)), 16)
        destination += generation * stride + RANK * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(local, destination, offsets, mask=mask, cache=".wt")

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < TOTAL_ELEMENTS,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask
    )

    # Clear only this tile's consumed generation. Skipped tiles keep their
    # own epochs, so mixed row counts need no global counter or tail clearing.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(
            sentinel, destination, offsets, mask=mask, cache=".wt"
        )
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit(launch_metadata=collective_launch_metadata)
def iris_reduce_symmetric_gluon_kernel(
    input_sym_ptr,
    output_ptr,
    ready_flags,
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
    TOTAL_NUMEL,
    BLOCK_SIZE: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    PUBLISH_READY: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
):
    """Reduce producer outputs placed consecutively in symmetric memory."""
    block_id = gl.program_id(0)
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
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
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
        PUBLISH=PUBLISH_READY,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    layout: gl.constexpr = gl.BlockedLayout(
        [WORDS_PER_LANE], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    lane = gl.arange(0, BLOCK_SIZE // ELEMENTS_PER_WORD, layout=layout)
    total_packed = TOTAL_NUMEL // ELEMENTS_PER_WORD
    tile_id = block_id
    while tile_id < NUM_TILES:
        packed_offset = tile_id * (BLOCK_SIZE // ELEMENTS_PER_WORD) + lane
        mask = packed_offset < total_packed
        local_packed = gl.amd.cdna4.buffer_load(
            tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
            other=0,
        )
        acc_0, acc_1, acc_2, acc_3 = _unpack_word(
            local_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
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
                    peer_heap + input_heap_offset, gl.pointer_type(gl.uint64)
                )
                peer_packed = gl.amd.cdna4.buffer_load(
                    peer_input,
                    packed_offset.to(gl.int32),
                    mask=mask,
                    other=0,
                    cache=".cg",
                )
                peer_0, peer_1, peer_2, peer_3 = _unpack_word(
                    peer_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
                )
                acc_0 += peer_0
                acc_1 += peer_1
                acc_2 += peer_2
                acc_3 += peer_3

        packed_output = _pack_word(
            acc_0,
            acc_1,
            acc_2,
            acc_3,
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            packed_output,
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
        )
        tile_id += NUM_PROGRAMS

    # Do not return while a peer program can still be reading this rank's
    # input. The next producer reuses the same symmetric buffer.
    completion_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        completion_epoch,
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
        PUBLISH=PUBLISH_READY,
    )


@gluon.jit(launch_metadata=collective_launch_metadata)
def iris_reduce_symmetric_two_stage_gluon_kernel(
    input_sym_ptr,
    scratch_sym_ptr,
    output_ptr,
    ready_flags,
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
    PARTITION_WORDS,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
):
    """Reduce-scatter producer outputs, then all-gather the rank partitions."""
    block_id = gl.program_id(0)
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
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
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

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    scratch_heap_offset = tl.cast(scratch_sym_ptr, gl.uint64) - local_heap
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS_PER_LANE],
        [1, SUBGROUP_SIZE],
        [WORLD_SIZE, NUM_WARPS // WORLD_SIZE],
        [1, 0],
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [WORLD_SIZE, 1], [1, SUBGROUP_SIZE], [1, NUM_WARPS], [0, 1]
    )
    peer_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    reduce_word_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    peer_ids = gl.arange(0, WORLD_SIZE, layout=peer_layout)
    words = gl.arange(0, BLOCK_WORDS, layout=word_layout)
    reduce_words = gl.arange(0, BLOCK_WORDS, layout=reduce_word_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    peer_inputs = tl.cast(
        gl.expand_dims(peer_heaps, 1) + input_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    peer_scratch = tl.cast(
        gl.expand_dims(peer_heaps, 1) + scratch_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[32, 4]],
        [WORLD_SIZE, BLOCK_WORDS],
        [1, 0],
    )
    peer_values = gl.allocate_shared_memory(
        gl.uint64,
        [WORLD_SIZE, BLOCK_WORDS],
        shared_layout,
    )
    rank_start = RANK * PARTITION_WORDS

    # Reduce only this rank's partition of the full input into symmetric scratch.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        input_offset = rank_start + partition_offset
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_inputs + gl.expand_dims(input_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        peer_values.store(values)

        packed = peer_values.load(reduce_layout)
        value_0, value_1, value_2, value_3 = _unpack_word(
            packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        reduced = _pack_word(
            gl.sum(value_0, axis=0),
            gl.sum(value_1, axis=0),
            gl.sum(value_2, axis=0),
            gl.sum(value_3, axis=0),
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            reduced,
            tl.cast(scratch_sym_ptr, gl.pointer_type(gl.uint64)),
            (tile_id * BLOCK_WORDS + reduce_words).to(gl.int32),
            mask=tile_id * BLOCK_WORDS + reduce_words < PARTITION_WORDS,
            cache=".wt",
        )
        tile_id += NUM_PROGRAMS

    partitions_ready = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        partitions_ready,
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

    # Gather one reduced partition from every rank into the local output.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_scratch + gl.expand_dims(partition_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        output_offset = gl.expand_dims(peer_ids * PARTITION_WORDS, 1) + gl.expand_dims(
            partition_offset, 0
        )
        gl.store(
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)) + output_offset,
            values,
            mask=gl.expand_dims(mask, 0),
        )
        tile_id += NUM_PROGRAMS

    if EXIT_BARRIER:
        # Callers that stage into the symmetric input before launching cannot
        # rotate buffers safely: under graph capture the staging copy records a
        # fixed address and replays it, so a rank one invocation ahead would
        # overwrite an input its slower peers are still reducing. Holding the
        # kernel until every peer has finished reading makes a single staging
        # buffer correct by construction, at the price of one more rendezvous.
        # Producer-direct callers own their input and pass False.
        reads_done = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
        _iris_sync_rank_epoch(
            ready_flags,
            block_id,
            reads_done,
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
