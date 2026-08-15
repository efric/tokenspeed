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

"""Raw cooperative gfx950 Kimi K3 MegaMoE persistent kernel.

The entry point intentionally has a flat tensor ABI.  The vendor-neutral shim
owns Iris objects and immutable plans; this package sees only graph-stable
tensors and specialization scalars.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.admission import (
    LDS_BYTES,
    PROGRAMS,
    QUALIFIED_TIMEOUT_NS,
    admit_kimi_k3_megamoe_compiled_kernel,
    preflight_kimi_k3_megamoe_runtime,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.workspace import (
    validate_workspace_tensors,
)

_PROGRAMS = gl.constexpr(PROGRAMS)
_LDS_BYTES = gl.constexpr(LDS_BYTES)
_WORLD_SIZE = gl.constexpr(8)
_XCDS = gl.constexpr(8)
_WORKERS_PER_XCD = gl.constexpr(30)
_EXPERT_WORKGROUPS = gl.constexpr(226)
_W2_TASKS = gl.constexpr(448)
_HIDDEN = gl.constexpr(7168)
_ROUTER = gl.constexpr(896)
_LATENT = gl.constexpr(3584)
_SHARED = gl.constexpr(768)
_PHASE0_BLOCK_K = gl.constexpr(1024)


@gluon.jit
def _drain_subgroup_vmem_before_barrier():
    """Drain each producer subgroup's write-through payload stores."""

    # Every mutable payload store is write-through. A workgroup barrier still
    # cannot complete another subgroup's outstanding VMEM, so keep this impure
    # and operand-free and execute it in every producer subgroup before the
    # workgroup elects a release publisher. DRAIN_ONLY publication was stressed
    # across 240 local workgroups and eight cross-NUMA ranks for 4096 epochs.
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _wait_subgroup_vmem_ack():
    """Complete any vector-memory acknowledgment pending in this subgroup."""

    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _poll_exact_generation(flag, generation, TIMEOUT_NS: gl.constexpr):
    """Acquire one exact system generation for subgroup-zero Iris payload."""

    matched = gl.atomic_poll(
        flag,
        generation,
        sem="acquire",
        scope="sys",
        timeout_ns=TIMEOUT_NS,
    )
    matched = matched != 0
    observed = generation
    if not matched:
        # Failure-only diagnostics do not participate in progress. Split
        # ready/completion planes ensure this cannot be a future generation.
        observed = gl.load(flag, cache_modifier=".cv")
    # Scalar atomic_poll elects lane 0 of subgroup 0 and retains its compiler-
    # owned acquire invalidation plus workgroup rendezvous.  Every dependent
    # Iris payload load is also owned entirely by subgroup 0; completion protects
    # reuse by a later dispatch.  Fresh-binary, skewed world-eight
    # stress qualified this exact gfx950 path for 4096 generations without a
    # second source-owned VMEM acknowledgment or workgroup barrier.
    return matched, observed


@gluon.jit
def _fatal_epoch_is_clear(fatal_epoch):
    """Relaxed-check process-lifetime sticky fatal state without mutation."""

    # The sticky value carries no payload consumed after this check. System
    # scope retains remote poison visibility; atomic_poll's built-in LDS
    # broadcast and workgroup rendezvous make the early-return decision
    # convergent without an acquire invalidation or a second source barrier.
    return (
        gl.atomic_poll(
            fatal_epoch,
            0,
            sem="relaxed",
            scope="sys",
            timeout_ns=0,
        )
        != 0
    )


@gluon.jit
def _group_heap_base(group_rank: gl.constexpr, heap_bases, group_global_ranks):
    global_rank = gl.load(group_global_ranks + group_rank).to(gl.int64)
    # Iris exposes signed INT64 storage. Preserve all address bits rather than
    # numerically narrowing it before pointer-width arithmetic.
    return gl.load(heap_bases + global_rank).to(gl.uint64, bitcast=True)


@gluon.jit
def _translate_symmetric_ptr(
    local_ptr,
    local_heap,
    peer_heap,
    element_ty: gl.constexpr,
):
    byte_offset = tl.cast(local_ptr, gl.uint64) - local_heap
    return tl.cast(peer_heap + byte_offset, gl.pointer_type(element_ty))


@gluon.jit
def _poison_all_ranks(
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    generation,
    pid,
    PHASE: gl.constexpr,
    xcc,
    expected,
    observed,
    RANK: gl.constexpr,
):
    """System-release a fatal generation and retain one local diagnostic."""

    local_heap = _group_heap_base(RANK, heap_bases, group_global_ranks)
    for peer in gl.static_range(0, _WORLD_SIZE):
        peer_heap = _group_heap_base(peer, heap_bases, group_global_ranks)
        peer_fatal = _translate_symmetric_ptr(
            fatal_epoch,
            local_heap,
            peer_heap,
            gl.int64,
        )
        # Fatal state is process-lifetime sticky/nonzero, and every producer
        # for one failed launch writes the same generation. A one-shot swap
        # avoids the unbounded compiler CAS retry loop used for INT64 max.
        gl.atomic_xchg(peer_fatal, generation, sem="release", scope="sys")
    previous = gl.atomic_cas(
        fail_diagnostics,
        gl.full((), 0, gl.int64),
        generation,
        sem="acq_rel",
        scope="gpu",
    )
    if previous == 0:
        gl.store(
            fail_diagnostics + 1,
            pid.to(gl.int64),
            cache_modifier=".wt",
        )
        gl.store(fail_diagnostics + 2, PHASE, cache_modifier=".wt")
        gl.store(fail_diagnostics + 3, generation, cache_modifier=".wt")
        gl.store(
            fail_diagnostics + 4,
            xcc.to(gl.int64),
            cache_modifier=".wt",
        )
        gl.store(fail_diagnostics + 5, expected.to(gl.int64), cache_modifier=".wt")
        gl.store(fail_diagnostics + 6, observed.to(gl.int64), cache_modifier=".wt")
        gl.store(fail_diagnostics + 7, RANK, cache_modifier=".wt")


@gluon.jit
def _poll_local_or_poison(
    gate,
    expected,
    generation,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    pid,
    PHASE: gl.constexpr,
    xcc,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    matched = gl.atomic_poll(
        gate,
        expected,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    if not matched:
        observed = gl.atomic_add(gate, 0, sem="acquire", scope="gpu")
        _poison_all_ranks(
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            generation,
            pid,
            PHASE,
            xcc,
            expected,
            observed,
            RANK,
        )
    # ``atomic_poll`` invalidates through VMEM in its elected subgroup.  The
    # acknowledgment must complete before the convergent rendezvous allows
    # any subgroup to consume payload guarded by ``gate``.
    _wait_subgroup_vmem_ack()
    gl.barrier()
    return matched != 0


@gluon.jit
def _poll_topology_or_poison(
    gate,
    expected,
    generation,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    pid,
    PHASE: gl.constexpr,
    xcc,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Acquire the merged phase/topology publication or fail closed."""

    # The exact topology proof also publishes phase-zero payload written by all
    # 240 workgroups. Complete the acquire invalidation in every subgroup before
    # the workgroup converges on top-k or shared-expert payload consumption.
    matched = gl.atomic_poll(
        gate,
        expected,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    if not matched:
        observed = gl.atomic_add(gate, 0, sem="acquire", scope="gpu")
        _poison_all_ranks(
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            generation,
            pid,
            PHASE,
            xcc,
            expected,
            observed,
            RANK,
        )
    _wait_subgroup_vmem_ack()
    gl.barrier()
    return matched != 0


@gluon.jit
def _arrive_topology_ticket(
    generation,
    xcc,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
):
    """Claim one exact XCD-local rank and hierarchically publish topology."""

    # The admitted gfx950 launch has one resident workgroup per compute unit and
    # 32 active compute units per XCD.  An exact generation contributes at most
    # once per dense hardware XCC ID.  With P=240, observing all eight rank-29
    # tickets therefore proves an exact 30-workgroup population on every XCD.
    # Keep the atomic inside the uniform bounds branch: HW_REG_XCC_ID is four
    # bits wide, while the cumulative ticket allocation intentionally has only
    # the eight dense IDs admitted by this specialization.
    local_rank = gl.full((), -1, gl.int32)
    if xcc < _XCDS:
        ticket = gl.atomic_add(
            xcc_ticket + xcc,
            1,
            sem="relaxed",
            scope="gpu",
        )
        local_rank = (ticket % _WORKERS_PER_XCD).to(gl.int32)
        ticket_generation = ticket // _WORKERS_PER_XCD + 1
        if (ticket_generation == generation) & (local_rank == _WORKERS_PER_XCD - 1):
            xcd_ticket = gl.atomic_add(
                xcd_arrival,
                1,
                sem="relaxed",
                scope="gpu",
            )
            topology_generation = xcd_ticket // _XCDS + 1
            if (xcd_ticket % _XCDS == _XCDS - 1) & (topology_generation == generation):
                gl.atomic_xchg(
                    topology_gate,
                    generation,
                    sem="release",
                    scope="gpu",
                )
    return local_rank


@gluon.jit
def _phase0_router_pair(
    pair,
    hidden_states,
    router_weight,
    router_logits,
):
    """Project one pair of four-row router tiles in two subgroup cohorts."""

    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, _PHASE0_BLOCK_K // 64],
        [1, 1, 64],
        [2, 4, 1],
        [2, 1, 0],
    )
    cn_layout: gl.constexpr = gl.SliceLayout(2, layout)
    c_layout: gl.constexpr = gl.SliceLayout(1, cn_layout)
    n_layout: gl.constexpr = gl.SliceLayout(0, cn_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
    cohort = gl.arange(0, 2, layout=c_layout)
    lane_n = gl.arange(0, 4, layout=n_layout)
    task = pair + cohort * 112
    row = task[:, None] * 4 + lane_n[None, :]

    acc = gl.zeros([2, 4], gl.float32, layout=cn_layout)
    for k0 in range(0, _HIDDEN, _PHASE0_BLOCK_K):
        offs_k = k0 + gl.arange(0, _PHASE0_BLOCK_K, layout=k_layout)
        activation = gl.amd.cdna4.buffer_load(
            hidden_states,
            offs_k.to(gl.int32),
            cache=".ca",
        ).to(gl.float32)
        weight_offsets = (
            row[:, :, None].to(gl.int32) * _HIDDEN + offs_k[None, None, :].to(gl.int32)
        ).to(gl.int32)
        weight = gl.amd.cdna4.buffer_load(
            router_weight,
            weight_offsets,
            cache=".cs",
        )
        activation = gl.convert_layout(activation[None, None, :], layout)
        acc += gl.sum(weight.to(gl.float32) * activation, axis=2)
    gl.store(router_logits + row, acc, cache_modifier=".wt")


@gluon.jit
def _phase0_routed_pair(
    pair,
    hidden_states,
    routed_down_weight,
    routed_input,
):
    """Project one pair of sixteen-row routed tiles in two subgroup cohorts."""

    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, _PHASE0_BLOCK_K // 64],
        [1, 1, 64],
        [2, 4, 1],
        [2, 1, 0],
    )
    cn_layout: gl.constexpr = gl.SliceLayout(2, layout)
    c_layout: gl.constexpr = gl.SliceLayout(1, cn_layout)
    n_layout: gl.constexpr = gl.SliceLayout(0, cn_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
    cohort = gl.arange(0, 2, layout=c_layout)
    lane_n = gl.arange(0, 16, layout=n_layout)
    task = pair + cohort * 112
    row = task[:, None] * 16 + lane_n[None, :]

    acc = gl.zeros([2, 16], gl.float32, layout=cn_layout)
    for k0 in range(0, _HIDDEN, _PHASE0_BLOCK_K):
        offs_k = k0 + gl.arange(0, _PHASE0_BLOCK_K, layout=k_layout)
        activation = gl.amd.cdna4.buffer_load(
            hidden_states,
            offs_k.to(gl.int32),
            cache=".ca",
        ).to(gl.float32)
        weight_offsets = (
            row[:, :, None].to(gl.int32) * _HIDDEN + offs_k[None, None, :].to(gl.int32)
        ).to(gl.int32)
        weight = gl.amd.cdna4.buffer_load(
            routed_down_weight,
            weight_offsets,
            cache=".cs",
        )
        activation = gl.convert_layout(activation[None, None, :], layout)
        acc += gl.sum(weight.to(gl.float32) * activation, axis=2)
    gl.store(routed_input + row, acc.to(gl.bfloat16), cache_modifier=".wt")


@gluon.jit
def _phase0_shared_pair(
    pair,
    hidden_states,
    shared_gate_up_weight,
    shared_input,
    beta,
    linear_beta,
):
    """Project one pair of four-row shared SiTU tiles in two cohorts."""

    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, _PHASE0_BLOCK_K // 64],
        [1, 1, 64],
        [2, 4, 1],
        [2, 1, 0],
    )
    cn_layout: gl.constexpr = gl.SliceLayout(2, layout)
    c_layout: gl.constexpr = gl.SliceLayout(1, cn_layout)
    n_layout: gl.constexpr = gl.SliceLayout(0, cn_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
    cohort = gl.arange(0, 2, layout=c_layout)
    lane_n = gl.arange(0, 4, layout=n_layout)
    task = pair + cohort * 96
    row = task[:, None] * 4 + lane_n[None, :]

    gate_acc = gl.zeros([2, 4], gl.float32, layout=cn_layout)
    up_acc = gl.zeros([2, 4], gl.float32, layout=cn_layout)
    for k0 in range(0, _HIDDEN, _PHASE0_BLOCK_K):
        offs_k = k0 + gl.arange(0, _PHASE0_BLOCK_K, layout=k_layout)
        activation = gl.amd.cdna4.buffer_load(
            hidden_states,
            offs_k.to(gl.int32),
            cache=".ca",
        ).to(gl.float32)
        gate_offsets = (
            row[:, :, None].to(gl.int32) * _HIDDEN + offs_k[None, None, :].to(gl.int32)
        ).to(gl.int32)
        up_offsets = gate_offsets + _SHARED * _HIDDEN
        gate_weight = gl.amd.cdna4.buffer_load(
            shared_gate_up_weight,
            gate_offsets,
            cache=".cs",
        )
        up_weight = gl.amd.cdna4.buffer_load(
            shared_gate_up_weight,
            up_offsets,
            cache=".cs",
        )
        activation = gl.convert_layout(activation[None, None, :], layout)
        gate_acc += gl.sum(gate_weight.to(gl.float32) * activation, axis=2)
        up_acc += gl.sum(up_weight.to(gl.float32) * activation, axis=2)

    # Preserve both BF16 projection outputs before the FP32 SiTU epilogue.
    gate_raw = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    gate = beta * gl.extra.libdevice.tanh(gate_raw / beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
    gl.store(shared_input + row, (gate * up).to(gl.bfloat16), cache_modifier=".wt")


@gluon.jit
def _phase0_projection(
    pid,
    hidden_states,
    router_weight,
    routed_down_weight,
    shared_gate_up_weight,
    router_logits,
    routed_input,
    shared_input,
    beta,
    linear_beta,
):
    """Run the fixed P=240 paired phase-zero projection schedule."""

    if pid < 112:
        _phase0_routed_pair(
            pid,
            hidden_states,
            routed_down_weight,
            routed_input,
        )
    else:
        if pid < 208:
            _phase0_shared_pair(
                pid - 112,
                hidden_states,
                shared_gate_up_weight,
                shared_input,
                beta,
                linear_beta,
            )
            if pid < 192:
                _phase0_router_pair(
                    pid - 80,
                    hidden_states,
                    router_weight,
                    router_logits,
                )
        else:
            _phase0_router_pair(
                pid - 208,
                hidden_states,
                router_weight,
                router_logits,
            )


@gluon.jit
def _sigmoid_bias_top16(
    pid,
    router_logits,
    correction_bias,
    topk_ids,
    topk_weights,
):
    """Produce the exact FP32 K3 sigmoid/bias route on program P-2."""

    if pid == _PROGRAMS - 2:
        neg: gl.constexpr = float("-inf")
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 2],
            [1, 64],
            [1, 8],
            [1, 0],
        )
        col = gl.expand_dims(
            gl.arange(0, 1024, layout=gl.SliceLayout(0, layout)),
            0,
        )
        mask = col < _ROUTER
        logits = gl.load(
            router_logits + col,
            mask=mask,
            other=neg,
            cache_modifier=".cv",
        ).to(gl.float32)
        scores = gl.fdiv(1.0, 1.0 + gl.exp(-logits))
        bias = gl.load(correction_bias + col, mask=mask, other=0.0).to(gl.float32)
        choice = gl.where(mask, scores + bias, neg)

        tcol = gl.expand_dims(
            gl.arange(0, 16, layout=gl.SliceLayout(0, layout)),
            0,
        )
        indices = gl.zeros([1, 16], gl.int32, layout=layout)
        live = mask
        sign = gl.full([1, 1024], 0x80000000, gl.uint32, layout=layout)
        all_bits = gl.full([1, 1024], 0xFFFFFFFF, gl.uint32, layout=layout)
        zeros64 = gl.full([1, 1024], 0, gl.uint64, layout=layout)
        raw = choice.to(gl.uint32, bitcast=True)
        value_key = raw ^ gl.where((raw & sign) != 0, all_bits, sign)
        index_key = (1024 - col).to(gl.uint32)
        packed_key = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
        for slot in gl.static_range(0, 16):
            packed = gl.where(live, packed_key, zeros64)
            best = gl.max(packed, axis=1, keep_dims=True)
            selected_id = (1024 - (best & 0xFFFF).to(gl.int32)).to(gl.int32)
            select_slot = tcol == slot
            indices = gl.where(select_slot, selected_id, indices)
            live &= col != selected_id

        # Materialize IDs once, then reload them in the same compact 1-D layout
        # as current Triton's post-topk conversion. The global round trip avoids
        # keeping the large selection layout live across normalization.
        output_row = gl.arange(0, 1, layout=gl.SliceLayout(1, layout))
        zero_i = gl.zeros([1, 16], gl.int32, layout=layout)
        for slot in gl.static_range(0, 16):
            selected_id = gl.sum(
                gl.where(tcol == slot, indices, zero_i),
                axis=1,
            )
            gl.store(
                topk_ids + slot + output_row * 0,
                selected_id,
                cache_modifier=".wt",
            )
        # Selection stores may be owned by multiple subgroups, while the
        # compact reload uses a different layout. Publish every producer's
        # VMEM before the convergent layout-changing reload.
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()

        compact_layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        compact_offs = gl.arange(0, 16, layout=compact_layout)
        compact_ids = gl.load(topk_ids + compact_offs, cache_modifier=".cv")
        selected_logits = gl.load(
            router_logits + compact_ids,
            cache_modifier=".cv",
        ).to(gl.float32)
        values = gl.fdiv(1.0, 1.0 + gl.exp(-selected_logits))
        denominator = gl.sum(values, axis=0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        values = gl.fdiv(values, denominator)
        gl.store(topk_weights + compact_offs, values, cache_modifier=".wt")
        # The same workgroup reloads weights after topology-gate acquire; make
        # its final top-k stores visible before that intervening invalidation.
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()


@gluon.jit
def _publish_route_plan(
    pid,
    generation,
    topk_ids,
    topk_weights,
    local_route_ids,
    local_route_slots,
    local_route_weights,
    local_route_count,
    route_xcd_start,
    route_xcd_count,
    route_worker_count,
    w13_target,
    w2_target,
    w13_arrival,
    w2_arrival,
    route_plan_gate,
    EXPERT_START: gl.constexpr,
):
    """Compact owned routes and publish deterministic XCD intervals/targets."""

    if pid == _PROGRAMS - 2:
        count = gl.full((), 0, gl.int32)
        for slot in gl.static_range(0, 16):
            expert = gl.load(topk_ids + slot, cache_modifier=".cv")
            weight = gl.load(topk_weights + slot, cache_modifier=".cv")
            owned = (expert >= EXPERT_START) & (expert < EXPERT_START + 112)
            if owned:
                gl.store(
                    local_route_ids + count,
                    expert - EXPERT_START,
                    cache_modifier=".wt",
                )
                gl.store(
                    local_route_slots + count,
                    slot,
                    cache_modifier=".wt",
                )
                gl.store(
                    local_route_weights + count,
                    weight,
                    cache_modifier=".wt",
                )
                count += 1
        gl.store(local_route_count, count, cache_modifier=".wt")

        for route in gl.static_range(0, 16):
            active = route < count
            start = gl.full((), 0, gl.int32)
            xcd_count = gl.full((), 0, gl.int32)
            if active:
                if count <= 8:
                    base = 8 // count
                    extra = 8 % count
                    xcd_count = base + (route < extra).to(gl.int32)
                    start = route * base + gl.minimum(route, extra)
                else:
                    start = route % 8
                    xcd_count = 1

            worker_count = gl.full((), 0, gl.int32)
            for xcd_ordinal in gl.static_range(0, 8):
                in_interval = (
                    active & (xcd_ordinal >= start) & (xcd_ordinal < start + xcd_count)
                )
                # comm indices 0..13 alone leave expert work. For G=8 this is
                # two low local ranks on XCDs 0..5 and one on XCDs 6..7.
                expert_count = 28 if xcd_ordinal < 6 else 29
                worker_count += gl.where(in_interval, expert_count, 0)
            gl.store(route_xcd_start + route, start, cache_modifier=".wt")
            gl.store(route_xcd_count + route, xcd_count, cache_modifier=".wt")
            gl.store(
                route_worker_count + route,
                worker_count,
                cache_modifier=".wt",
            )

        # W13 and fused W2/combine each use one exact 226-expert-workgroup
        # rendezvous, independent of the number of compact routes. Arrivals
        # are quiescent after prior-kernel stream completion, and current
        # expert work cannot start before route_plan_gate. L=0 deliberately
        # leaves every expert counter unchanged and takes the seven-owner zero
        # producer path instead.
        if count > 0:
            current_w13 = gl.load(w13_arrival, cache_modifier=".cv")
            current_w2 = gl.load(w2_arrival, cache_modifier=".cv")
            gl.store(
                w13_target,
                current_w13 + _EXPERT_WORKGROUPS,
                cache_modifier=".wt",
            )
            gl.store(
                w2_target,
                current_w2 + _EXPERT_WORKGROUPS,
                cache_modifier=".wt",
            )
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()
        gl.atomic_xchg(route_plan_gate, generation, sem="release", scope="gpu")


@gluon.jit
def _shared_down_produce(
    pid,
    generation,
    shared_input,
    shared_down_weight,
    symmetric_producer,
    shared_arrival,
    shared_gate,
):
    """Produce four exact N8/K768 tiles and publish one 16-worker stripe."""

    if pid < 224:
        stripe = pid // 16
        stripe_rank = pid % 16
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 8],
            [1, 1, 64],
            [2, 4, 1],
            [2, 1, 0],
        )
        cn_layout: gl.constexpr = gl.SliceLayout(2, layout)
        c_layout: gl.constexpr = gl.SliceLayout(1, cn_layout)
        n_layout: gl.constexpr = gl.SliceLayout(0, cn_layout)
        k_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
        cohort = gl.arange(0, 2, layout=c_layout)
        lane_n = gl.arange(0, 8, layout=n_layout)
        for round_id in gl.static_range(0, 2):
            task = stripe_rank + cohort * 16 + round_id * 32
            output_col = stripe * 512 + task[:, None] * 8 + lane_n[None, :]
            acc = gl.zeros([2, 8], gl.float32, layout=cn_layout)
            for k0 in range(0, 1024, 512):
                offs_k = k0 + gl.arange(0, 512, layout=k_layout)
                k_mask = offs_k < _SHARED
                activation = gl.amd.cdna4.buffer_load(
                    shared_input,
                    offs_k.to(gl.int32),
                    mask=k_mask,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
                weight = gl.load(
                    shared_down_weight
                    + output_col[:, :, None].to(gl.int64) * _SHARED
                    + offs_k[None, None, :].to(gl.int64),
                    mask=k_mask[None, None, :],
                    other=0.0,
                    cache_modifier=".cs",
                )
                activation = gl.convert_layout(activation[None, None, :], layout)
                acc += gl.sum(weight.to(gl.float32) * activation, axis=2)
            gl.store(
                symmetric_producer + output_col,
                acc.to(gl.bfloat16),
                cache_modifier=".wt",
            )
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()
        ticket = gl.atomic_add(
            shared_arrival + stripe,
            1,
            sem="relaxed",
            scope="gpu",
        )
        if ticket % 16 == 15:
            gl.atomic_xchg(
                shared_gate + stripe,
                generation,
                sem="release",
                scope="gpu",
            )


@gluon.jit
def _w13_route_tiles(
    route,
    worker_rank,
    worker_count,
    expert,
    routed_input,
    w13_weight,
    w13_weight_scale,
    w13_intermediate,
    beta,
    linear_beta,
):
    """Produce statically assigned linear-MXFP4 W13 N8 tiles."""

    packed_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2, 16],
        [1, 1, 64],
        [2, 4, 1],
        [2, 1, 0],
    )
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2, 32],
        [1, 1, 64],
        [2, 4, 1],
        [2, 1, 0],
    )
    packed_cn_layout: gl.constexpr = gl.SliceLayout(2, packed_layout)
    packed_c_layout: gl.constexpr = gl.SliceLayout(1, packed_cn_layout)
    packed_n_layout: gl.constexpr = gl.SliceLayout(0, packed_cn_layout)
    packed_k_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.SliceLayout(1, packed_layout),
    )
    expanded_cn_layout: gl.constexpr = gl.SliceLayout(2, expanded_layout)
    expanded_c_layout: gl.constexpr = gl.SliceLayout(1, expanded_cn_layout)
    expanded_n_layout: gl.constexpr = gl.SliceLayout(0, expanded_cn_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.SliceLayout(1, expanded_layout),
    )
    cohort = gl.arange(0, 2, layout=packed_c_layout)
    packed_lane_n = gl.arange(0, 8, layout=packed_n_layout)
    expanded_lane_n = gl.arange(0, 8, layout=expanded_n_layout)
    task = worker_rank + cohort * worker_count
    while gl.min(task, axis=0) < 384:
        valid_task = task < 384
        gate_col = task[:, None] * 8 + packed_lane_n[None, :]
        up_col = 3072 + gate_col
        expanded_task = gl.convert_layout(task, expanded_c_layout)
        expanded_valid_task = expanded_task < 384
        expanded_gate_col = expanded_task[:, None] * 8 + expanded_lane_n[None, :]
        expanded_up_col = 3072 + expanded_gate_col
        gate_acc = gl.zeros([2, 8], gl.float32, layout=expanded_cn_layout)
        up_acc = gl.zeros([2, 8], gl.float32, layout=expanded_cn_layout)
        for kb0 in range(0, 3584, 2048):
            offs_kb = kb0 // 2 + gl.arange(
                0,
                1024,
                layout=packed_k_layout,
            )
            expanded_k = kb0 + gl.arange(
                0,
                2048,
                layout=expanded_k_layout,
            )
            packed_valid = offs_kb < 1792
            expanded_valid = expanded_k < 3584
            activation = gl.amd.cdna4.buffer_load(
                routed_input,
                expanded_k.to(gl.int32),
                mask=expanded_valid,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            expert_weight = expert.to(gl.int64) * 11010048
            gate_offsets = (
                expert_weight
                + gate_col[:, :, None].to(gl.int64) * 1792
                + offs_kb[None, None, :].to(gl.int64)
            )
            up_offsets = (
                expert_weight
                + up_col[:, :, None].to(gl.int64) * 1792
                + offs_kb[None, None, :].to(gl.int64)
            )
            expert_scale = expert.to(gl.int64) * 688128
            gate_scale_offsets = (
                expert_scale
                + expanded_gate_col[:, :, None].to(gl.int64) * 112
                + (expanded_k[None, None, :] // 32).to(gl.int64)
            )
            up_scale_offsets = (
                expert_scale
                + expanded_up_col[:, :, None].to(gl.int64) * 112
                + (expanded_k[None, None, :] // 32).to(gl.int64)
            )
            packed_mask = valid_task[:, None, None] & packed_valid[None, None, :]
            expanded_mask = (
                expanded_valid_task[:, None, None] & expanded_valid[None, None, :]
            )
            gate_packed = gl.amd.cdna4.buffer_load(
                w13_weight,
                gate_offsets.to(gl.int32),
                mask=packed_mask,
                other=0,
                cache=".cs",
            )
            up_packed = gl.amd.cdna4.buffer_load(
                w13_weight,
                up_offsets.to(gl.int32),
                mask=packed_mask,
                other=0,
                cache=".cs",
            )
            gate_scale = gl.amd.cdna4.buffer_load(
                w13_weight_scale,
                gate_scale_offsets.to(gl.int32),
                mask=expanded_mask,
                other=0,
                cache=".cs",
            )
            up_scale = gl.amd.cdna4.buffer_load(
                w13_weight_scale,
                up_scale_offsets.to(gl.int32),
                mask=expanded_mask,
                other=0,
                cache=".cs",
            )
            gate_weight = gl.amd.cdna4.scaled_upcast(
                gate_packed,
                gate_scale,
                gl.bfloat16,
                axis=2,
            )
            up_weight = gl.amd.cdna4.scaled_upcast(
                up_packed,
                up_scale,
                gl.bfloat16,
                axis=2,
            )
            activation = gl.convert_layout(
                activation[None, None, :],
                expanded_layout,
            )
            gate_acc += gl.sum(gate_weight.to(gl.float32) * activation, axis=2)
            up_acc += gl.sum(up_weight.to(gl.float32) * activation, axis=2)

        gate = gate_acc.to(gl.bfloat16).to(gl.float32)
        up = up_acc.to(gl.bfloat16).to(gl.float32)
        gate = (
            beta * gl.extra.libdevice.tanh(gate / beta) * (1.0 / (1.0 + gl.exp(-gate)))
        )
        up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
        gl.store(
            w13_intermediate + route * 3072 + expanded_gate_col.to(gl.int64),
            (gate * up).to(gl.bfloat16),
            mask=expanded_valid_task[:, None],
            cache_modifier=".wt",
        )
        task += 2 * worker_count


@gluon.jit
def _w2_route_tiles(
    route,
    worker_rank,
    worker_count,
    expert,
    w13_intermediate,
    w2_weight,
    w2_weight_scale,
    w2_route_output,
):
    """Produce statically assigned linear-MXFP4 W2 N8 tiles."""

    packed_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [1, 64],
        [8, 1],
        [1, 0],
    )
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [1, 64],
        [8, 1],
        [1, 0],
    )
    packed_n_layout: gl.constexpr = gl.SliceLayout(1, packed_layout)
    packed_k_layout: gl.constexpr = gl.SliceLayout(0, packed_layout)
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    packed_lane_n = gl.arange(0, 8, layout=packed_n_layout)
    expanded_lane_n = gl.arange(0, 8, layout=expanded_n_layout)
    task = worker_rank
    while task < 448:
        output_col = task * 8 + packed_lane_n
        expanded_output_col = task * 8 + expanded_lane_n
        acc = gl.zeros([8], gl.float32, layout=expanded_n_layout)
        for kb0 in range(0, 1536, 512):
            offs_kb = kb0 + gl.arange(0, 512, layout=packed_k_layout)
            expanded_k = 2 * kb0 + gl.arange(
                0,
                1024,
                layout=expanded_k_layout,
            )
            intermediate = gl.amd.cdna4.buffer_load(
                w13_intermediate + route * 3072,
                expanded_k.to(gl.int32),
                cache=".cv",
            ).to(gl.float32)
            weight_offsets = (
                expert.to(gl.int64) * 5505024
                + output_col[:, None].to(gl.int64) * 1536
                + offs_kb[None, :].to(gl.int64)
            )
            scale_offsets = (
                expert.to(gl.int64) * 344064
                + expanded_output_col[:, None].to(gl.int64) * 96
                + (expanded_k[None, :] // 32).to(gl.int64)
            )
            packed = gl.amd.cdna4.buffer_load(
                w2_weight,
                weight_offsets.to(gl.int32),
                cache=".cs",
            )
            scale = gl.amd.cdna4.buffer_load(
                w2_weight_scale,
                scale_offsets.to(gl.int32),
                cache=".cs",
            )
            weight = gl.amd.cdna4.scaled_upcast(
                packed,
                scale,
                gl.bfloat16,
                axis=1,
            )
            intermediate = gl.convert_layout(
                intermediate[None, :],
                expanded_layout,
            )
            acc += gl.sum(weight.to(gl.float32) * intermediate, axis=1)
        gl.store(
            w2_route_output + route * _LATENT + expanded_output_col.to(gl.int64),
            acc.to(gl.bfloat16),
            cache_modifier=".wt",
        )
        task += worker_count


@gluon.jit
def _expert_route_phase(
    route_index,
    active,
    xcd_ordinal,
    expert_local_rank,
    local_route_ids,
    route_xcd_start,
    route_xcd_count,
    route_worker_count,
    routed_input,
    w13_weight,
    w13_weight_scale,
    w13_intermediate,
    beta,
    linear_beta,
):
    """Produce one dynamically selected compact route's W13 intermediate."""

    start = gl.load(route_xcd_start + route_index, cache_modifier=".cv")
    xcd_count = gl.load(route_xcd_count + route_index, cache_modifier=".cv")
    member = (
        active
        & (expert_local_rank >= 0)
        & (xcd_ordinal >= start)
        & (xcd_ordinal < start + xcd_count)
    )
    if member:
        worker_rank = expert_local_rank
        for prior_xcd in gl.static_range(0, 8):
            prior_in_group = (prior_xcd >= start) & (prior_xcd < xcd_ordinal)
            expert_count = 28 if prior_xcd < 6 else 29
            worker_rank += gl.where(prior_in_group, expert_count, 0)
        worker_count = gl.load(
            route_worker_count + route_index,
            cache_modifier=".cv",
        )
        expert = gl.load(
            local_route_ids + route_index,
            cache_modifier=".cv",
        )
        _w13_route_tiles(
            route_index,
            worker_rank,
            worker_count,
            expert,
            routed_input,
            w13_weight,
            w13_weight_scale,
            w13_intermediate,
            beta,
            linear_beta,
        )


@gluon.jit
def _expert_grid_arrive(target_ptr, arrival, gate):
    """Publish one payload generation from all 226 expert workgroups."""

    _drain_subgroup_vmem_before_barrier()
    gl.barrier()
    target = gl.load(target_ptr, cache_modifier=".cv")
    ticket = gl.atomic_add(arrival, 1, sem="relaxed", scope="gpu")
    if ticket + 1 == target:
        gl.atomic_xchg(gate, target, sem="release", scope="gpu")
    return target


@gluon.jit
def _output_centric_w2_combine(
    expert_rank,
    route_count,
    local_route_ids,
    local_route_weights,
    w13_intermediate,
    w2_weight,
    w2_weight_scale,
    symmetric_producer,
):
    """Compute each N8 W2 output across routes in exact compact order."""

    packed_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [1, 64],
        [8, 1],
        [1, 0],
    )
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [1, 64],
        [8, 1],
        [1, 0],
    )
    packed_n_layout: gl.constexpr = gl.SliceLayout(1, packed_layout)
    packed_k_layout: gl.constexpr = gl.SliceLayout(0, packed_layout)
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    packed_lane_n = gl.arange(0, 8, layout=packed_n_layout)
    expanded_lane_n = gl.arange(0, 8, layout=expanded_n_layout)

    task = expert_rank
    while task < _W2_TASKS:
        output_col = task * 8 + packed_lane_n
        expanded_output_col = task * 8 + expanded_lane_n
        combine_acc = gl.zeros([8], gl.float32, layout=expanded_n_layout)
        route = gl.full((), 0, gl.int32)
        while route < route_count:
            expert = gl.load(local_route_ids + route, cache_modifier=".cv")
            route_acc = gl.zeros([8], gl.float32, layout=expanded_n_layout)
            for kb0 in range(0, 1536, 512):
                offs_kb = kb0 + gl.arange(0, 512, layout=packed_k_layout)
                expanded_k = 2 * kb0 + gl.arange(
                    0,
                    1024,
                    layout=expanded_k_layout,
                )
                intermediate = gl.amd.cdna4.buffer_load(
                    w13_intermediate + route * 3072,
                    expanded_k.to(gl.int32),
                    cache=".cv",
                ).to(gl.float32)
                weight_offsets = (
                    expert.to(gl.int64) * 5505024
                    + output_col[:, None].to(gl.int64) * 1536
                    + offs_kb[None, :].to(gl.int64)
                )
                scale_offsets = (
                    expert.to(gl.int64) * 344064
                    + expanded_output_col[:, None].to(gl.int64) * 96
                    + (expanded_k[None, :] // 32).to(gl.int64)
                )
                packed = gl.amd.cdna4.buffer_load(
                    w2_weight,
                    weight_offsets.to(gl.int32),
                    cache=".cs",
                )
                scale = gl.amd.cdna4.buffer_load(
                    w2_weight_scale,
                    scale_offsets.to(gl.int32),
                    cache=".cs",
                )
                weight = gl.amd.cdna4.scaled_upcast(
                    packed,
                    scale,
                    gl.bfloat16,
                    axis=1,
                )
                intermediate = gl.convert_layout(
                    intermediate[None, :],
                    expanded_layout,
                )
                route_acc += gl.sum(
                    weight.to(gl.float32) * intermediate,
                    axis=1,
                )

            # Match the old route-output BF16 store/reload before applying the
            # FP32 route weight, then retain ascending compact-route addition.
            route_value = route_acc.to(gl.bfloat16).to(gl.float32)
            route_weight = gl.load(
                local_route_weights + route,
                cache_modifier=".cv",
            ).to(gl.float32)
            combine_acc += route_value * route_weight
            route += 1

        gl.amd.cdna4.buffer_store(
            combine_acc.to(gl.bfloat16),
            symmetric_producer + _HIDDEN,
            expanded_output_col.to(gl.int32),
            cache=".wt",
        )
        task += _EXPERT_WORKGROUPS


@gluon.jit
def _expert_phases(
    pid,
    generation,
    xcd_ordinal,
    expert_local_rank,
    comm_index,
    local_route_count,
    local_route_ids,
    local_route_weights,
    route_xcd_start,
    route_xcd_count,
    route_worker_count,
    w13_target,
    w2_target,
    routed_input,
    w13_weight,
    w13_weight_scale,
    w2_weight,
    w2_weight_scale,
    w13_intermediate,
    symmetric_producer,
    w13_arrival,
    w13_gate,
    w2_arrival,
    w2_gate,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    xcc,
    beta,
    linear_beta,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Produce W13, then output-centric fused W2/combine, on expert workers."""

    route_count = gl.load(local_route_count, cache_modifier=".cv")
    ready = gl.full((), True, gl.int1)
    if (route_count > 0) & (expert_local_rank >= 0):
        first_route = xcd_ordinal
        if route_count <= 8:
            base = 8 // route_count
            extra = 8 % route_count
            long = base + 1
            long_xcds = extra * long
            first_route = gl.where(
                xcd_ordinal < long_xcds,
                xcd_ordinal // long,
                extra + (xcd_ordinal - long_xcds) // base,
            )
        _expert_route_phase(
            first_route,
            route_count > 0,
            xcd_ordinal,
            expert_local_rank,
            local_route_ids,
            route_xcd_start,
            route_xcd_count,
            route_worker_count,
            routed_input,
            w13_weight,
            w13_weight_scale,
            w13_intermediate,
            beta,
            linear_beta,
        )
        second_route = xcd_ordinal + 8
        _expert_route_phase(
            second_route,
            (route_count > 8) & (second_route < route_count),
            xcd_ordinal,
            expert_local_rank,
            local_route_ids,
            route_xcd_start,
            route_xcd_count,
            route_worker_count,
            routed_input,
            w13_weight,
            w13_weight_scale,
            w13_intermediate,
            beta,
            linear_beta,
        )

        expected_w13 = _expert_grid_arrive(
            w13_target,
            w13_arrival,
            w13_gate,
        )
        w13_ready = _poll_local_or_poison(
            w13_gate,
            expected_w13,
            generation,
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            pid,
            5,
            xcc,
            RANK,
            TIMEOUT_NS,
        )
        if w13_ready:
            expert_rank = expert_local_rank
            for prior_xcd in gl.static_range(0, 8):
                expert_count = 28 if prior_xcd < 6 else 29
                expert_rank += gl.where(
                    prior_xcd < xcd_ordinal,
                    expert_count,
                    0,
                )
            _output_centric_w2_combine(
                expert_rank,
                route_count,
                local_route_ids,
                local_route_weights,
                w13_intermediate,
                w2_weight,
                w2_weight_scale,
                symmetric_producer,
            )
            expected_w2 = _expert_grid_arrive(
                w2_target,
                w2_arrival,
                w2_gate,
            )
            w2_owner = (comm_index >= 14) & (comm_index < 21)
            if w2_owner:
                ready = _poll_local_or_poison(
                    w2_gate,
                    expected_w2,
                    generation,
                    fatal_epoch,
                    heap_bases,
                    group_global_ranks,
                    fail_diagnostics,
                    pid,
                    7,
                    xcc,
                    RANK,
                    TIMEOUT_NS,
                )
        else:
            ready = gl.full((), False, gl.int1)
    return ready


@gluon.jit
def _zero_routed_producer(
    comm_index,
    local_route_count,
    symmetric_producer,
):
    """Materialize the L=0 routed stripe without expert-grid rendezvous."""

    route_count = gl.load(local_route_count, cache_modifier=".cv")
    if (route_count == 0) & (comm_index >= 14) & (comm_index < 21):
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        lane = gl.arange(0, 512, layout=layout)
        gl.amd.cdna4.buffer_store(
            gl.zeros([512], gl.bfloat16, layout=layout),
            symmetric_producer,
            (comm_index * 512 + lane).to(gl.int32),
            cache=".wt",
        )
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()


@gluon.jit
def _unpack_bf16x4(packed):
    value0 = (packed & 0xFFFF).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    value1 = ((packed >> 16) & 0xFFFF).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    value2 = ((packed >> 32) & 0xFFFF).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    value3 = ((packed >> 48) & 0xFFFF).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return (
        value0.to(gl.float32),
        value1.to(gl.float32),
        value2.to(gl.float32),
        value3.to(gl.float32),
    )


@gluon.jit
def _pack_bf16x4(value0, value1, value2, value3):
    bits0 = value0.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits1 = value1.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits2 = value2.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits3 = value3.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits0 | (bits1 << 16) | (bits2 << 32) | (bits3 << 48)


@gluon.jit
def _wait_peer_epoch_or_poison(
    peer_flag,
    expected,
    generation,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    pid,
    PHASE: gl.constexpr,
    xcc,
    peer: gl.constexpr,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Bounded exact peer wait with sticky fail-stop propagation."""

    ready, acquired = _poll_exact_generation(
        peer_flag,
        expected,
        TIMEOUT_NS,
    )
    if not ready:
        _poison_all_ranks(
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            generation,
            pid,
            PHASE,
            xcc,
            expected,
            acquired,
            RANK,
        )
    return ready, acquired


@gluon.jit
def _routed_combine(
    pid,
    generation,
    comm_index,
    local_route_count,
    local_route_weights,
    w2_target,
    w2_gate,
    w2_route_output,
    symmetric_producer,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    xcc,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Combine compact routes in original slot order into one routed stripe."""

    if (comm_index >= 14) & (comm_index < 21):
        route_count = gl.load(local_route_count, cache_modifier=".cv")
        ready = gl.full((), True, gl.int1)
        for route in gl.static_range(0, 16):
            if route < route_count:
                target = gl.load(w2_target + route, cache_modifier=".cv")
                route_ready = _poll_local_or_poison(
                    w2_gate + route,
                    target,
                    generation,
                    fatal_epoch,
                    heap_bases,
                    group_global_ranks,
                    fail_diagnostics,
                    pid,
                    7,
                    xcc,
                    RANK,
                    TIMEOUT_NS,
                )
                ready &= route_ready
        if ready:
            stripe = comm_index - 14
            layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
            lane = gl.arange(0, 512, layout=layout)
            acc = gl.zeros([512], gl.float32, layout=layout)
            for route in gl.static_range(0, 16):
                if route < route_count:
                    value = gl.amd.cdna4.buffer_load(
                        w2_route_output + route * _LATENT,
                        (stripe * 512 + lane).to(gl.int32),
                        cache=".cv",
                    ).to(gl.float32)
                    weight = gl.load(
                        local_route_weights + route,
                        cache_modifier=".cv",
                    ).to(gl.float32)
                    acc += value * weight
            gl.amd.cdna4.buffer_store(
                acc.to(gl.bfloat16),
                symmetric_producer,
                (comm_index * 512 + lane).to(gl.int32),
                cache=".wt",
            )
        return ready
    return gl.full((), True, gl.int1)


@gluon.jit
def _iris_communication_tile(
    pid,
    generation,
    comm_index,
    shared_gate,
    symmetric_producer,
    symmetric_reduced,
    iris_epoch_flags,
    comm_arrival,
    comm_gate,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    fail_diagnostics,
    xcc,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Run one exact 512-element producer-direct Iris stripe."""

    if comm_index < 21:
        producer_ready = gl.full((), True, gl.int1)
        if comm_index < 14:
            producer_ready = _poll_local_or_poison(
                shared_gate + comm_index,
                generation,
                generation,
                fatal_epoch,
                heap_bases,
                group_global_ranks,
                fail_diagnostics,
                pid,
                8,
                xcc,
                RANK,
                TIMEOUT_NS,
            )
        if not producer_ready:
            return

        # Complete routed-producer `.wt` stores before the normal ready release.
        # Shared producers were already published by their local gate and
        # safely repeat the same drain here.
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()
        if not _fatal_epoch_is_clear(fatal_epoch):
            return

        layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
        lane = gl.arange(0, 512, layout=layout)
        consumer_lane = lane < 64
        producer_u64 = tl.cast(symmetric_producer, gl.pointer_type(gl.uint64))

        local_ready_flag = iris_epoch_flags + comm_index * _WORLD_SIZE + RANK
        completion_plane: gl.constexpr = 21 * 8
        local_completion_flag = local_ready_flag + completion_plane
        previous_ready = gl.atomic_xchg(
            local_ready_flag,
            generation,
            sem="release",
            scope="sys",
        )
        if previous_ready != generation - 1:
            _poison_all_ranks(
                fatal_epoch,
                heap_bases,
                group_global_ranks,
                fail_diagnostics,
                generation,
                pid,
                9,
                xcc,
                generation - 1,
                previous_ready,
                RANK,
            )
            return
        local_heap = _group_heap_base(RANK, heap_bases, group_global_ranks)
        flags_offset = tl.cast(iris_epoch_flags, gl.uint64) - local_heap
        local_ready, _ = _wait_peer_epoch_or_poison(
            local_ready_flag,
            generation,
            generation,
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            pid,
            9,
            xcc,
            RANK,
            RANK,
            TIMEOUT_NS,
        )
        if not local_ready:
            return

        peers_ready = gl.full((), True, gl.int1)
        for distance in gl.static_range(1, _WORLD_SIZE):
            # XOR rounds give every rank one unique flag partner per round.
            # Payload accumulation below deliberately remains rank-ascending.
            peer = RANK ^ distance
            if peer != RANK:
                peer_heap = _group_heap_base(peer, heap_bases, group_global_ranks)
                peer_flags = tl.cast(
                    peer_heap + flags_offset,
                    gl.pointer_type(gl.int64),
                )
                peer_flag = peer_flags + comm_index * _WORLD_SIZE + peer
                peer_ready, _ = _wait_peer_epoch_or_poison(
                    peer_flag,
                    generation,
                    generation,
                    fatal_epoch,
                    heap_bases,
                    group_global_ranks,
                    fail_diagnostics,
                    pid,
                    9,
                    xcc,
                    peer,
                    RANK,
                    TIMEOUT_NS,
                )
                peers_ready &= peer_ready
        if not peers_ready:
            return

        reduced_u64 = tl.cast(symmetric_reduced, gl.pointer_type(gl.uint64))
        producer_offset = tl.cast(symmetric_producer, gl.uint64) - local_heap
        # Keep all 128 packed words in the scalar acquire's subgroup.  Two
        # words per lane reproduce the standalone-Iris communication shape and
        # avoid relying on cache invalidation propagating to sibling subgroups.
        for word_group in gl.static_range(0, 2):
            word = lane + word_group * 64
            packed_offset = comm_index * 128 + word
            local_packed = gl.amd.cdna4.buffer_load(
                producer_u64,
                packed_offset.to(gl.int32),
                mask=consumer_lane,
                other=0,
                cache=".cv",
            )
            acc0, acc1, acc2, acc3 = _unpack_bf16x4(local_packed)
            for peer in gl.static_range(0, _WORLD_SIZE):
                if peer != RANK:
                    peer_heap = _group_heap_base(
                        peer,
                        heap_bases,
                        group_global_ranks,
                    )
                    peer_producer = tl.cast(
                        peer_heap + producer_offset,
                        gl.pointer_type(gl.uint64),
                    )
                    peer_packed = gl.amd.cdna4.buffer_load(
                        peer_producer,
                        packed_offset.to(gl.int32),
                        mask=consumer_lane,
                        other=0,
                        cache=".cv",
                    )
                    peer0, peer1, peer2, peer3 = _unpack_bf16x4(peer_packed)
                    acc0 += peer0
                    acc1 += peer1
                    acc2 += peer2
                    acc3 += peer3
            gl.amd.cdna4.buffer_store(
                _pack_bf16x4(acc0, acc1, acc2, acc3),
                reduced_u64,
                packed_offset.to(gl.int32),
                mask=consumer_lane,
                cache=".wt",
            )

        # Publish completion only after subgroup 0 has drained all 128 packed
        # write-through stores in the local reduced tile.
        _drain_subgroup_vmem_before_barrier()
        gl.barrier()
        previous_completion = gl.atomic_xchg(
            local_completion_flag,
            generation,
            sem="release",
            scope="sys",
        )
        if previous_completion != generation - 1:
            _poison_all_ranks(
                fatal_epoch,
                heap_bases,
                group_global_ranks,
                fail_diagnostics,
                generation,
                pid,
                10,
                xcc,
                generation - 1,
                previous_completion,
                RANK,
            )
            return
        peers_complete = gl.full((), True, gl.int1)
        for distance in gl.static_range(1, _WORLD_SIZE):
            peer = RANK ^ distance
            if peer != RANK:
                peer_heap = _group_heap_base(peer, heap_bases, group_global_ranks)
                peer_flags = tl.cast(
                    peer_heap + flags_offset,
                    gl.pointer_type(gl.int64),
                )
                peer_flag = (
                    peer_flags + completion_plane + comm_index * _WORLD_SIZE + peer
                )
                peer_complete, _ = _wait_peer_epoch_or_poison(
                    peer_flag,
                    generation,
                    generation,
                    fatal_epoch,
                    heap_bases,
                    group_global_ranks,
                    fail_diagnostics,
                    pid,
                    10,
                    xcc,
                    peer,
                    RANK,
                    TIMEOUT_NS,
                )
                peers_complete &= peer_complete
        if peers_complete:
            ticket = gl.atomic_add(
                comm_arrival,
                1,
                sem="relaxed",
                scope="gpu",
            )
            if ticket % 21 == 20:
                gl.atomic_xchg(
                    comm_gate,
                    generation,
                    sem="release",
                    scope="gpu",
                )


@gluon.jit
def _final_rmsnorm_linear_add(
    pid,
    prefix_sum,
    routed_norm_weight,
    routed_up_weight,
    symmetric_reduced,
    layer_output,
    rms_eps,
):
    """Finish the reduced routed latent and shared residual in N32 slices."""

    if pid < 224:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 8],
            [1, 64],
            [8, 1],
            [1, 0],
        )
        n_layout: gl.constexpr = gl.SliceLayout(1, layout)
        k_layout: gl.constexpr = gl.SliceLayout(0, layout)
        offs_n = pid * 32 + gl.arange(0, 32, layout=n_layout)
        routed = symmetric_reduced + _HIDDEN
        square_sum = gl.full((), 0.0, gl.float32)
        for k0 in range(0, 4096, 512):
            offs_k = k0 + gl.arange(0, 512, layout=k_layout)
            mask = offs_k < _LATENT
            latent = gl.amd.cdna4.buffer_load(
                routed,
                offs_k.to(gl.int32),
                mask=mask,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            square_sum += gl.sum(latent * latent, axis=0)
        inverse_rms = gl.rsqrt(square_sum / _LATENT + rms_eps)

        acc = gl.zeros([32], gl.float32, layout=n_layout)
        for k0 in range(0, 4096, 512):
            offs_k = k0 + gl.arange(0, 512, layout=k_layout)
            mask = offs_k < _LATENT
            latent = gl.amd.cdna4.buffer_load(
                routed,
                offs_k.to(gl.int32),
                mask=mask,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            norm_weight = gl.amd.cdna4.buffer_load(
                routed_norm_weight,
                offs_k.to(gl.int32),
                mask=mask,
                other=0.0,
                cache=".ca",
            ).to(gl.float32)
            normalized = (latent * inverse_rms * norm_weight).to(gl.bfloat16)
            projection = gl.load(
                routed_up_weight
                + offs_n[:, None].to(gl.int64) * _LATENT
                + offs_k[None, :].to(gl.int64),
                mask=mask[None, :],
                other=0.0,
                cache_modifier=".cs",
            )
            normalized = gl.convert_layout(normalized[None, :], layout)
            acc += gl.sum(
                projection.to(gl.float32) * normalized.to(gl.float32),
                axis=1,
            )
        acc = acc.to(gl.bfloat16).to(gl.float32)
        acc += gl.amd.cdna4.buffer_load(
            prefix_sum,
            offs_n.to(gl.int32),
            cache=".ca",
        ).to(gl.float32)
        acc += gl.amd.cdna4.buffer_load(
            symmetric_reduced,
            offs_n.to(gl.int32),
            cache=".cv",
        ).to(gl.float32)
        gl.store(
            layer_output + offs_n,
            acc.to(gl.bfloat16),
            cache_modifier=".wt",
        )


@gluon.jit
def _kimi_k3_megamoe_kernel(
    hidden_states,
    prefix_sum,
    layer_output,
    router_weight,
    routed_down_weight,
    shared_gate_up_weight,
    shared_down_weight,
    routed_norm_weight,
    routed_up_weight,
    correction_bias,
    w13_weight,
    w13_weight_scale,
    w2_weight,
    w2_weight_scale,
    router_logits,
    routed_input,
    shared_input,
    topk_ids,
    topk_weights,
    local_route_ids,
    local_route_slots,
    local_route_weights,
    local_route_count,
    route_xcd_start,
    route_xcd_count,
    route_worker_count,
    w13_target,
    w2_target,
    w13_intermediate,
    w2_route_output,
    xcc_ticket,
    xcd_arrival,
    phase_arrival,
    phase_gate,
    topology_gate,
    route_plan_gate,
    shared_arrival,
    shared_gate,
    w13_arrival,
    w13_gate,
    w2_arrival,
    w2_gate,
    comm_arrival,
    comm_gate,
    fail_diagnostics,
    symmetric_producer,
    symmetric_reduced,
    iris_epoch_flags,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    beta,
    linear_beta,
    rms_eps,
    EXPERT_START: gl.constexpr,
    RANK: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Execute the compiled MegaMoE phase state machine."""

    # ``fatal_epoch`` is process-lifetime sticky. Graphs contain successive
    # layer nodes that share this workspace and Iris lane, so a node after a
    # poisoned predecessor must not advance arrivals, overwrite scratch, or
    # touch its distinct layer output. The relaxed system poll carries no
    # payload dependency; atomic_poll's LDS broadcast and internal workgroup
    # rendezvous make the early return convergent. Keep this before the first
    # application-state store/arrival.
    if not _fatal_epoch_is_clear(fatal_epoch):
        return

    # The preceding retained launch leaves one stable exact topology
    # generation. Every resident workgroup reads that common predecessor before
    # phase zero; the successor cannot publish until all 240 workgroups later
    # contribute to the hierarchical ticket proof.
    generation = gl.load(topology_gate, cache_modifier=".cv") + 1
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, 512, layout=layout)

    # Materially retain the entire allocation. This is dynamic LDS and the
    # loaded metadata/sharedMemBytes value is checked before first launch.
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    residency = gl.allocate_shared_memory(gl.uint8, [_LDS_BYTES], shared_layout)
    residency_tile = residency.slice(0, 512)
    residency_tile.store((lane % 251).to(gl.uint8))
    gl.barrier()
    residency_token = residency_tile.load(layout)

    xcc = (
        gl.inline_asm_elementwise(
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
            "=s",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        & 0xF
    )
    # Keep the LDS read semantically live without changing a valid XCC value.
    if gl.sum(residency_token.to(gl.int32), axis=0) < 0:
        gl.store(xcc_ticket, -1, cache_modifier=".wt")

    _phase0_projection(
        pid,
        hidden_states,
        router_weight,
        routed_down_weight,
        shared_gate_up_weight,
        router_logits,
        routed_input,
        shared_input,
        beta,
        linear_beta,
    )
    # Phase-zero stores are distributed over all eight subgroups. The relaxed
    # XCD tickets may represent this workgroup only after each subgroup drains
    # its write-through payload and converges at the workgroup barrier.
    _drain_subgroup_vmem_before_barrier()
    gl.barrier()
    local_rank = _arrive_topology_ticket(
        generation,
        xcc,
        xcc_ticket,
        xcd_arrival,
        topology_gate,
    )
    topology_ready = _poll_topology_or_poison(
        topology_gate,
        generation,
        generation,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        fail_diagnostics,
        pid,
        3,
        xcc,
        RANK,
        TIMEOUT_NS,
    )
    if not topology_ready:
        return
    _sigmoid_bias_top16(
        pid,
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
    )
    _shared_down_produce(
        pid,
        generation,
        shared_input,
        shared_down_weight,
        symmetric_producer,
        shared_arrival,
        shared_gate,
    )
    _publish_route_plan(
        pid,
        generation,
        topk_ids,
        topk_weights,
        local_route_ids,
        local_route_slots,
        local_route_weights,
        local_route_count,
        route_xcd_start,
        route_xcd_count,
        route_worker_count,
        w13_target,
        w2_target,
        w13_arrival,
        w2_arrival,
        route_plan_gate,
        EXPERT_START,
    )
    route_ready = _poll_local_or_poison(
        route_plan_gate,
        generation,
        generation,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        fail_diagnostics,
        pid,
        4,
        xcc,
        RANK,
        TIMEOUT_NS,
    )
    if not route_ready:
        return
    xcd_ordinal = xcc
    comm_index = local_rank * _XCDS + xcd_ordinal
    expert_local_rank = local_rank - gl.where(xcd_ordinal < 6, 2, 1)
    expert_ready = _expert_phases(
        pid,
        generation,
        xcd_ordinal,
        expert_local_rank,
        comm_index,
        local_route_count,
        local_route_ids,
        local_route_weights,
        route_xcd_start,
        route_xcd_count,
        route_worker_count,
        w13_target,
        w2_target,
        routed_input,
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
        w13_intermediate,
        symmetric_producer,
        w13_arrival,
        w13_gate,
        w2_arrival,
        w2_gate,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        fail_diagnostics,
        xcc,
        beta,
        linear_beta,
        RANK,
        TIMEOUT_NS,
    )
    if not expert_ready:
        return
    _zero_routed_producer(
        comm_index,
        local_route_count,
        symmetric_producer,
    )
    _iris_communication_tile(
        pid,
        generation,
        comm_index,
        shared_gate,
        symmetric_producer,
        symmetric_reduced,
        iris_epoch_flags,
        comm_arrival,
        comm_gate,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        fail_diagnostics,
        xcc,
        RANK,
        TIMEOUT_NS,
    )
    if pid < 224:
        comm_ready = _poll_local_or_poison(
            comm_gate,
            generation,
            generation,
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            fail_diagnostics,
            pid,
            11,
            xcc,
            RANK,
            TIMEOUT_NS,
        )
        if not comm_ready:
            return
        _final_rmsnorm_linear_add(
            pid,
            prefix_sum,
            routed_norm_weight,
            routed_up_weight,
            symmetric_reduced,
            layer_output,
            rms_eps,
        )


_WORKSPACE_NAMES = (
    "router_logits",
    "routed_input",
    "shared_input",
    "topk_ids",
    "topk_weights",
    "local_route_ids",
    "local_route_slots",
    "local_route_weights",
    "local_route_count",
    "route_xcd_start",
    "route_xcd_count",
    "route_worker_count",
    "w13_target",
    "w2_target",
    "w13_intermediate",
    "w2_route_output",
    "xcc_ticket",
    "xcd_arrival",
    "phase_arrival",
    "phase_gate",
    "topology_gate",
    "route_plan_gate",
    "shared_arrival",
    "shared_gate",
    "w13_arrival",
    "w13_gate",
    "w2_arrival",
    "w2_gate",
    "comm_arrival",
    "comm_gate",
    "fail_diagnostics",
)

KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES = (
    "hidden_states",
    "prefix_sum",
    "layer_output",
    "router_weight",
    "routed_down_weight",
    "shared_gate_up_weight",
    "shared_down_weight",
    "routed_norm_weight",
    "routed_up_weight",
    "correction_bias",
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
    *_WORKSPACE_NAMES,
    "symmetric_producer",
    "symmetric_reduced",
    "iris_epoch_flags",
    "fatal_epoch",
    "heap_bases",
    "group_global_ranks",
)


@dataclass(frozen=True)
class PreparedKimiK3MegaMoEKernel:
    """One loaded and admitted specialization ready for direct dispatch.

    ``runner`` is created from the admitted :class:`CompiledKernel` exactly
    once.  Calling it cannot enter Gluon's JIT or repeat code-object admission,
    which makes the direct path suitable for CUDA/HIP graph capture.
    """

    compiled: object
    runner: Callable[..., None]
    admission: object
    device: torch.device
    expert_start: int
    group_rank: int
    timeout_ns: int
    beta: float
    linear_beta: float
    rms_eps: float


def _require_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Kimi K3 MegaMoE {name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Kimi K3 MegaMoE {name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"Kimi K3 MegaMoE {name} must have dtype {dtype}")
    if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"Kimi K3 MegaMoE {name} must be contiguous on {device}")
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"Kimi K3 MegaMoE {name} must be 16-byte aligned")


def _validate_direct_tensor_contract(
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
) -> None:
    """Check the pointer properties baked into the admitted specialization."""

    if len(tensors) != len(KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES):
        raise ValueError(
            "Kimi K3 MegaMoE raw ABI requires "
            f"{len(KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES)} tensors, got {len(tensors)}"
        )
    for name, tensor in zip(
        KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES,
        tensors,
        strict=True,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Kimi K3 MegaMoE {name} must be a tensor")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"Kimi K3 MegaMoE {name} must be contiguous on {device}")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"Kimi K3 MegaMoE {name} must be 16-byte aligned")


def _validate_raw_tensors(
    tensors: tuple[torch.Tensor, ...],
    workspace: tuple[torch.Tensor, ...],
    iris: tuple[torch.Tensor, ...],
    *,
    expert_start: int,
    group_rank: int,
    timeout_ns: int,
    beta: float,
    linear_beta: float,
    rms_eps: float,
) -> None:
    hidden_states = tensors[0]
    if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
        raise ValueError("Kimi K3 MegaMoE hidden_states must be a CUDA tensor")
    device = hidden_states.device
    dense_specs = (
        ("hidden_states", (1, 7168), torch.bfloat16),
        ("prefix_sum", (1, 7168), torch.bfloat16),
        ("layer_output", (1, 7168), torch.bfloat16),
        ("router_weight", (896, 7168), torch.bfloat16),
        ("routed_down_weight", (3584, 7168), torch.bfloat16),
        ("shared_gate_up_weight", (1536, 7168), torch.bfloat16),
        ("shared_down_weight", (7168, 768), torch.bfloat16),
        ("routed_norm_weight", (3584,), torch.bfloat16),
        ("routed_up_weight", (7168, 3584), torch.bfloat16),
        ("correction_bias", (896,), torch.float32),
        ("w13_weight", (112, 6144, 1792), torch.uint8),
        ("w13_weight_scale", (112, 6144, 112), torch.uint8),
        ("w2_weight", (112, 3584, 1536), torch.uint8),
        ("w2_weight_scale", (112, 3584, 96), torch.uint8),
    )
    for tensor, (name, shape, dtype) in zip(tensors, dense_specs, strict=True):
        _require_tensor(tensor, name, shape, dtype, device)
    if tuple(tensors[10].stride()) != (11010048, 1792, 1):
        raise ValueError("Kimi K3 MegaMoE w13_weight has the wrong linear strides")
    if tuple(tensors[11].stride()) != (688128, 112, 1):
        raise ValueError(
            "Kimi K3 MegaMoE w13_weight_scale has the wrong linear strides"
        )
    if tuple(tensors[12].stride()) != (5505024, 1536, 1):
        raise ValueError("Kimi K3 MegaMoE w2_weight has the wrong linear strides")
    if tuple(tensors[13].stride()) != (344064, 96, 1):
        raise ValueError("Kimi K3 MegaMoE w2_weight_scale has the wrong linear strides")
    validate_workspace_tensors(workspace, device)

    iris_specs = (
        ("symmetric_producer", (10752,), torch.bfloat16),
        ("symmetric_reduced", (10752,), torch.bfloat16),
        ("iris_epoch_flags", (2, 21, 8), torch.int64),
        ("fatal_epoch", (1,), torch.int64),
    )
    for tensor, (name, shape, dtype) in zip(iris[:4], iris_specs, strict=True):
        _require_tensor(tensor, name, shape, dtype, device)
    heap_bases, group_global_ranks = iris[4:]
    if (
        heap_bases.dtype != torch.int64
        or heap_bases.ndim != 1
        or not heap_bases.is_cuda
        or heap_bases.device != device
        or not heap_bases.is_contiguous()
    ):
        raise TypeError("Kimi K3 MegaMoE heap_bases must be contiguous CUDA INT64")
    _require_tensor(
        group_global_ranks,
        "group_global_ranks",
        (8,),
        torch.int64,
        device,
    )
    if expert_start not in range(0, 896, 112):
        raise ValueError("Kimi K3 MegaMoE expert_start must identify an EP8 shard")
    if group_rank not in range(8):
        raise ValueError("Kimi K3 MegaMoE group_rank must be in [0, 8)")
    if timeout_ns != QUALIFIED_TIMEOUT_NS:
        raise ValueError(
            "Kimi K3 MegaMoE timeout_ns must equal the qualified value "
            f"{QUALIFIED_TIMEOUT_NS}"
        )
    if expert_start != 112 * group_rank:
        raise ValueError("Kimi K3 MegaMoE expert_start must equal 112 * group_rank")
    if beta != 4.0 or linear_beta != 25.0 or rms_eps != 1.0e-5:
        raise ValueError("Kimi K3 MegaMoE requires beta=4, linear_beta=25, eps=1e-5")
    if len({tensor.data_ptr() for tensor in tensors[:3]}) != 3:
        raise ValueError("Kimi K3 MegaMoE input, prefix, and output may not alias")
    _validate_direct_tensor_contract(tensors + workspace + iris, device)


def _raw_arguments(
    hidden_states,
    prefix_sum,
    layer_output,
    router_weight,
    routed_down_weight,
    shared_gate_up_weight,
    shared_down_weight,
    routed_norm_weight,
    routed_up_weight,
    correction_bias,
    w13_weight,
    w13_weight_scale,
    w2_weight,
    w2_weight_scale,
    router_logits,
    routed_input,
    shared_input,
    topk_ids,
    topk_weights,
    local_route_ids,
    local_route_slots,
    local_route_weights,
    local_route_count,
    route_xcd_start,
    route_xcd_count,
    route_worker_count,
    w13_target,
    w2_target,
    w13_intermediate,
    w2_route_output,
    xcc_ticket,
    xcd_arrival,
    phase_arrival,
    phase_gate,
    topology_gate,
    route_plan_gate,
    shared_arrival,
    shared_gate,
    w13_arrival,
    w13_gate,
    w2_arrival,
    w2_gate,
    comm_arrival,
    comm_gate,
    fail_diagnostics,
    symmetric_producer,
    symmetric_reduced,
    iris_epoch_flags,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
):
    return (
        hidden_states,
        prefix_sum,
        layer_output,
        router_weight,
        routed_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        routed_norm_weight,
        routed_up_weight,
        correction_bias,
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
        router_logits,
        routed_input,
        shared_input,
        topk_ids,
        topk_weights,
        local_route_ids,
        local_route_slots,
        local_route_weights,
        local_route_count,
        route_xcd_start,
        route_xcd_count,
        route_worker_count,
        w13_target,
        w2_target,
        w13_intermediate,
        w2_route_output,
        xcc_ticket,
        xcd_arrival,
        phase_arrival,
        phase_gate,
        topology_gate,
        route_plan_gate,
        shared_arrival,
        shared_gate,
        w13_arrival,
        w13_gate,
        w2_arrival,
        w2_gate,
        comm_arrival,
        comm_gate,
        fail_diagnostics,
        symmetric_producer,
        symmetric_reduced,
        iris_epoch_flags,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
    )


def compile_kimi_k3_megamoe_gfx950(*args, **kwargs):
    """Warm up the exact specialization without dispatching it.

    The returned loaded kernel must be passed to
    :func:`admit_kimi_k3_megamoe_compiled_kernel` before any real launch.
    """

    beta = float(kwargs.pop("beta", 4.0))
    linear_beta = float(kwargs.pop("linear_beta", 25.0))
    rms_eps = float(kwargs.pop("rms_eps", 1.0e-5))
    expert_start = int(kwargs.pop("expert_start"))
    group_rank = int(kwargs.pop("group_rank"))
    timeout_ns = int(kwargs.pop("timeout_ns"))
    if kwargs:
        raise TypeError(f"unexpected MegaMoE keyword arguments: {tuple(kwargs)}")
    if len(args) != len(KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES):
        raise ValueError(
            "Kimi K3 MegaMoE raw ABI requires "
            f"{len(KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES)} tensors, got {len(args)}"
        )
    workspace_end = 14 + len(_WORKSPACE_NAMES)
    _validate_raw_tensors(
        args[:14],
        args[14:workspace_end],
        args[workspace_end:],
        expert_start=expert_start,
        group_rank=group_rank,
        timeout_ns=timeout_ns,
        beta=beta,
        linear_beta=linear_beta,
        rms_eps=rms_eps,
    )
    preflight_kimi_k3_megamoe_runtime()
    return _kimi_k3_megamoe_kernel.warmup(
        *args,
        beta,
        linear_beta,
        rms_eps,
        EXPERT_START=expert_start,
        RANK=group_rank,
        TIMEOUT_NS=timeout_ns,
        grid=(PROGRAMS,),
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=False,
    )


def prepare_kimi_k3_megamoe_gfx950(*args, **kwargs) -> PreparedKimiK3MegaMoEKernel:
    """Compile, load, admit, and bind one specialization outside capture.

    This is the only raw API that may enter the compiler.  The returned object
    retains both the loaded ``CompiledKernel`` and the direct runner produced
    by ``compiled[(240, 1, 1)]``.
    """

    beta = float(kwargs.get("beta", 4.0))
    linear_beta = float(kwargs.get("linear_beta", 25.0))
    rms_eps = float(kwargs.get("rms_eps", 1.0e-5))
    expert_start = int(kwargs["expert_start"])
    group_rank = int(kwargs["group_rank"])
    timeout_ns = int(kwargs["timeout_ns"])
    compiled = compile_kimi_k3_megamoe_gfx950(*args, **kwargs)
    report = admit_kimi_k3_megamoe_compiled_kernel(
        compiled,
        group_rank=group_rank,
        expert_start=expert_start,
        timeout_ns=timeout_ns,
    )
    # Materialize the runner now. CompiledKernel.__getitem__ initializes the
    # module handle and returns a closure over this exact loaded specialization.
    runner = compiled[(PROGRAMS, 1, 1)]
    return PreparedKimiK3MegaMoEKernel(
        compiled=compiled,
        runner=runner,
        admission=report,
        device=args[0].device,
        expert_start=expert_start,
        group_rank=group_rank,
        timeout_ns=timeout_ns,
        beta=beta,
        linear_beta=linear_beta,
        rms_eps=rms_eps,
    )


def launch_prepared_kimi_k3_megamoe_gfx950(
    prepared: PreparedKimiK3MegaMoEKernel,
    *args,
    stream=None,
) -> None:
    """Dispatch an already admitted kernel without JIT or admission work.

    Args:
        prepared: Object returned by :func:`prepare_kimi_k3_megamoe_gfx950`.
        *args: The tensors named by
            :data:`KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES`, in that exact order.
        stream: Optional raw stream accepted by Triton's loaded-kernel runner.

    Returns:
        ``None``. Dispatch is asynchronous on ``stream``.
    """

    if not isinstance(prepared, PreparedKimiK3MegaMoEKernel):
        raise TypeError("Kimi K3 MegaMoE direct launch requires a prepared kernel")
    _validate_direct_tensor_contract(args, prepared.device)
    prepared.runner(
        *args,
        prepared.beta,
        prepared.linear_beta,
        prepared.rms_eps,
        stream=stream,
    )


def launch_kimi_k3_megamoe_gfx950(
    prepared: PreparedKimiK3MegaMoEKernel,
    *args,
    stream=None,
) -> None:
    """Launch only through a retained, already admitted specialization.

    This alias deliberately cannot warm up, compile, or admit a kernel. Call
    :func:`prepare_kimi_k3_megamoe_gfx950` before the first collective and
    before graph capture.
    """

    launch_prepared_kimi_k3_megamoe_gfx950(
        prepared,
        *args,
        stream=stream,
    )


__all__ = [
    "KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES",
    "PreparedKimiK3MegaMoEKernel",
    "compile_kimi_k3_megamoe_gfx950",
    "launch_kimi_k3_megamoe_gfx950",
    "launch_prepared_kimi_k3_megamoe_gfx950",
    "prepare_kimi_k3_megamoe_gfx950",
]
