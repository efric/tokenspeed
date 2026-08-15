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

"""Bitwise phase oracles for the raw gfx950 Kimi K3 MegaMoE body."""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable

import pytest
import torch
from tokenspeed_kernel.ops.communication.iris import (
    iris_reduce_symmetric_gluon_kernel,
)
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.rmsnorm_linear_add import (
    gluon_rmsnorm_linear_add_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_decode import (
    _latent_input_decode_kernel,
    gluon_latent_input_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
    _final_rmsnorm_linear_add,
    _iris_communication_tile,
    _phase0_projection,
    _publish_route_plan,
    _routed_combine,
    _shared_down_produce,
    _w2_route_tiles,
    _w13_route_tiles,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (
    _stage1_a16w4_situ_warp_gemv,
    _stage2_a16w4_warp_gemv_combine,
)


def _has_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return "gfx950" in str(getattr(properties, "gcnArchName", ""))


pytestmark = [
    pytest.mark.skipif(
        os.getenv("TOKENSPEED_TEST_MEGAMOE_NUMERICS") != "1",
        reason="requires explicit opt-in for the large gfx950 numerical oracles",
    ),
    pytest.mark.skipif(not _has_gfx950(), reason="requires an AMD gfx950 GPU"),
]

_PHASE_TIMING_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TIMING"
_PHASE_TIMING_REPLAYS = 200
_PHASE_TIMING_WARMUPS = 20


@gluon.jit
def _phase0_probe(
    hidden, router_weight, routed_weight, shared_weight, router, routed, shared
):
    _phase0_projection(
        gl.program_id(0),
        hidden,
        router_weight,
        routed_weight,
        shared_weight,
        router,
        routed,
        shared,
        4.0,
        25.0,
    )


@gluon.jit
def _route_plan_probe(
    ids,
    weights,
    local_ids,
    slots,
    local_weights,
    count,
    starts,
    xcd_counts,
    worker_counts,
    w13_targets,
    w2_targets,
    w13_arrivals,
    w2_arrivals,
    gate,
):
    generation = gl.full((), 1, gl.int64)
    _publish_route_plan(
        gl.program_id(0),
        generation,
        ids,
        weights,
        local_ids,
        slots,
        local_weights,
        count,
        starts,
        xcd_counts,
        worker_counts,
        w13_targets,
        w2_targets,
        w13_arrivals,
        w2_arrivals,
        gate,
        112,
    )


@gluon.jit
def _shared_down_probe(shared_input, weight, output, arrivals, gates):
    generation = gl.full((), 1, gl.int64)
    _shared_down_produce(
        gl.program_id(0),
        generation,
        shared_input,
        weight,
        output,
        arrivals,
        gates,
    )


@gluon.jit
def _w13_probe(routed_input, weight, scale, intermediate, WORKERS: gl.constexpr):
    route = gl.full((), 0, gl.int32)
    expert = gl.full((), 0, gl.int32)
    _w13_route_tiles(
        route,
        gl.program_id(0),
        WORKERS,
        expert,
        routed_input,
        weight,
        scale,
        intermediate,
        4.0,
        25.0,
    )


@gluon.jit
def _w2_probe(
    intermediate,
    weight,
    scale,
    route_output,
    route_ptr,
    expert_ptr,
    WORKERS: gl.constexpr,
):
    route = gl.load(route_ptr)
    expert = gl.load(expert_ptr)
    _w2_route_tiles(
        route,
        gl.program_id(0),
        WORKERS,
        expert,
        intermediate,
        weight,
        scale,
        route_output,
    )


@gluon.jit
def _combine_probe(
    route_count,
    weights,
    targets,
    gates,
    route_output,
    symmetric_output,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    diagnostics,
):
    pid = gl.program_id(0)
    generation = gl.full((), 1, gl.int64)
    xcc = gl.full((), 0, gl.int32)
    _routed_combine(
        pid,
        generation,
        14 + pid,
        route_count,
        weights,
        targets,
        gates,
        route_output,
        symmetric_output,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        xcc,
        0,
        1_000_000_000,
    )


@gluon.jit
def _iris_probe(
    shared_gates,
    producer,
    reduced,
    flags,
    arrival,
    gate,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    diagnostics,
):
    pid = gl.program_id(0)
    generation = gl.full((), 1, gl.int64)
    xcc = gl.full((), 0, gl.int32)
    _iris_communication_tile(
        pid,
        generation,
        pid,
        shared_gates,
        producer,
        reduced,
        flags,
        arrival,
        gate,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        xcc,
        0,
        1_000_000_000,
    )


@gluon.jit
def _final_probe(prefix, norm_weight, projection_weight, reduced, output):
    _final_rmsnorm_linear_add(
        gl.program_id(0),
        prefix,
        norm_weight,
        projection_weight,
        reduced,
        output,
        1.0e-5,
    )


def _assert_bitwise_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> None:
    assert actual.dtype == expected.dtype
    if actual.dtype == torch.bfloat16:
        actual_bits = actual.contiguous().view(torch.int16)
        expected_bits = expected.contiguous().view(torch.int16)
    elif actual.dtype == torch.float32:
        actual_bits = actual.contiguous().view(torch.int32)
        expected_bits = expected.contiguous().view(torch.int32)
    else:
        actual_bits = actual
        expected_bits = expected
    differing = actual_bits != expected_bits
    assert not torch.any(
        differing
    ), f"{label} differs at {int(differing.sum())} elements"


def _current_stage2(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    routes = ids.shape[1]
    output = torch.empty((1, 3584), dtype=torch.bfloat16, device=intermediate.device)
    _stage2_a16w4_warp_gemv_combine[(448,)](
        intermediate,
        weight,
        scale,
        output,
        ids,
        route_weights,
        intermediate,
        output,
        output,
        3072,
        3072,
        intermediate.stride(0),
        intermediate.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(0),
        output.stride(1),
        ids.stride(0),
        ids.stride(1),
        route_weights.stride(0),
        route_weights.stride(1),
        FUSE_SHARED_DOWN=False,
        NUM_ROUTED_PROGRAMS=448,
        TOP_K=routes,
        EXPERT_START=0,
        NUM_LOCAL_EXPERTS=routes,
        LINEAR_WEIGHTS=True,
        NUM_PID_N=448,
        BLOCK_N=8,
        BLOCK_KB=512,
        NUM_WARPS=8,
        num_warps=8,
    )
    return output


def _time_retained_launch(
    name: str,
    launch: Callable[[], None],
    *,
    replays: int = _PHASE_TIMING_REPLAYS,
) -> dict[str, float | int | str]:
    """Return device-event timing for one already-warmed launch closure."""

    if replays < 100:
        raise ValueError("MegaMoE phase timing requires at least 100 replays")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()
    microseconds = [
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    ]
    return {
        "name": name,
        "replays": replays,
        "p50_us": statistics.median(microseconds),
        "min_us": min(microseconds),
        "max_us": max(microseconds),
    }


def test_amd_megamoe_phase0_is_bitwise_current_decode() -> None:
    torch.manual_seed(90210)
    device = "cuda:0"
    hidden = (
        torch.randn(7168, device=device, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    router_weight = (
        torch.randn(896, 7168, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    routed_weight = (
        torch.randn(3584, 7168, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    shared_weight = (
        torch.randn(1536, 7168, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    expected = gluon_latent_input_decode_gfx950(
        hidden.reshape(1, -1),
        router_weight,
        routed_weight,
        shared_weight,
        beta=4.0,
        linear_beta=25.0,
    )
    router = torch.full((896,), float("nan"), dtype=torch.float32, device=device)
    routed = torch.full((3584,), float("nan"), dtype=torch.bfloat16, device=device)
    shared = torch.full((768,), float("nan"), dtype=torch.bfloat16, device=device)

    _phase0_probe[(240,)](
        hidden,
        router_weight,
        routed_weight,
        shared_weight,
        router,
        routed,
        shared,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    _assert_bitwise_equal(router, expected[0].reshape(-1), "router projection")
    _assert_bitwise_equal(routed, expected[1].reshape(-1), "routed projection")
    _assert_bitwise_equal(shared, expected[2].reshape(-1), "shared SiTU projection")


def test_amd_megamoe_route_plan_preserves_slot_order_and_weight_bits() -> None:
    device = "cuda:0"
    ids = torch.tensor(
        [0, 112, 115, 50, 220, 111, 119, 400, 223, 224, 150, 700, 113, 895, 300, 180],
        dtype=torch.int32,
        device=device,
    )
    weights = (
        torch.arange(16, dtype=torch.float32, device=device)
        .mul_(0.03125)
        .add_(0.0078125)
    )
    local_ids = torch.full((16,), -1, dtype=torch.int32, device=device)
    slots = torch.full_like(local_ids, -1)
    local_weights = torch.full((16,), float("nan"), dtype=torch.float32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    starts = torch.zeros(16, dtype=torch.int32, device=device)
    xcd_counts = torch.zeros_like(starts)
    worker_counts = torch.zeros_like(starts)
    w13_targets = torch.zeros(16, dtype=torch.int64, device=device)
    w2_targets = torch.zeros_like(w13_targets)
    w13_arrivals = torch.zeros_like(w13_targets)
    w2_arrivals = torch.zeros_like(w13_targets)
    gate = torch.zeros(1, dtype=torch.int64, device=device)

    _route_plan_probe[(240,)](
        ids,
        weights,
        local_ids,
        slots,
        local_weights,
        count,
        starts,
        xcd_counts,
        worker_counts,
        w13_targets,
        w2_targets,
        w13_arrivals,
        w2_arrivals,
        gate,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    owned_slots = torch.tensor([1, 2, 4, 6, 8, 10, 12, 15], device=device)
    assert count.item() == owned_slots.numel()
    assert gate.item() == 1
    assert torch.equal(slots[: count.item()], owned_slots.to(torch.int32))
    assert torch.equal(local_ids[: count.item()], ids[owned_slots] - 112)
    _assert_bitwise_equal(
        local_weights[: count.item()],
        weights[owned_slots],
        "compacted route weights",
    )
    expected_targets = torch.zeros_like(w13_targets)
    expected_targets[0] = 226
    assert torch.equal(w13_targets, expected_targets)
    assert torch.equal(w2_targets, expected_targets)


def test_amd_megamoe_shared_down_is_bitwise_current_joint_decode() -> None:
    torch.manual_seed(161803)
    device = "cuda:0"
    shared_input = (
        torch.randn(1, 768, device=device, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    weight = (
        torch.randn(7168, 768, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    expected = torch.empty((1, 7168), dtype=torch.bfloat16, device=device)
    dummy_intermediate = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
    dummy_uint8 = torch.empty((1, 1, 1), dtype=torch.uint8, device=device)
    dummy_ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
    dummy_weights = torch.ones((1, 1), dtype=torch.float32, device=device)
    _stage2_a16w4_warp_gemv_combine[(896,)](
        dummy_intermediate,
        dummy_uint8,
        dummy_uint8,
        expected,
        dummy_ids,
        dummy_weights,
        shared_input,
        weight,
        expected,
        3584,
        3072,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        expected.stride(0),
        expected.stride(1),
        1,
        1,
        1,
        1,
        FUSE_SHARED_DOWN=True,
        NUM_ROUTED_PROGRAMS=0,
        TOP_K=1,
        EXPERT_START=0,
        NUM_LOCAL_EXPERTS=1,
        LINEAR_WEIGHTS=True,
        NUM_PID_N=448,
        BLOCK_N=8,
        BLOCK_KB=512,
        NUM_WARPS=8,
        num_warps=8,
    )
    producer = torch.full((10752,), float("nan"), dtype=torch.bfloat16, device=device)
    arrivals = torch.zeros(14, dtype=torch.int64, device=device)
    gates = torch.zeros(14, dtype=torch.int64, device=device)

    _shared_down_probe[(224,)](
        shared_input.reshape(-1),
        weight,
        producer,
        arrivals,
        gates,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    _assert_bitwise_equal(producer[:7168], expected.reshape(-1), "shared down")
    assert torch.equal(gates, torch.ones_like(gates))


def test_amd_megamoe_w13_is_bitwise_current_decode() -> None:
    torch.manual_seed(8128)
    device = "cuda:0"
    routed_input = (
        torch.randn(1, 3584, device=device, dtype=torch.bfloat16) * 0.25
    ).contiguous()
    weight = torch.randint(0, 256, (1, 6144, 1792), dtype=torch.uint8, device=device)
    scale = torch.randint(123, 130, (1, 6144, 112), dtype=torch.uint8, device=device)
    ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
    expected = torch.empty((1, 3072), dtype=torch.bfloat16, device=device)
    _stage1_a16w4_situ_warp_gemv[(384,)](
        routed_input,
        weight,
        scale,
        expected,
        ids,
        3584,
        3072,
        1,
        routed_input.stride(0),
        routed_input.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        expected.stride(0),
        expected.stride(1),
        ids.stride(0),
        ids.stride(1),
        SITU_BETA=4.0,
        SITU_LINEAR_BETA=25.0,
        HAS_LINEAR_BETA=True,
        EXPERT_START=0,
        NUM_LOCAL_EXPERTS=1,
        LINEAR_WEIGHTS=True,
        W13_INTERLEAVED=False,
        NUM_PID_N=384,
        BLOCK_N=8,
        BLOCK_KB=1024,
        NUM_WARPS=4,
        MASK_K_TAIL=True,
        num_warps=4,
    )
    actual = torch.full_like(expected, float("nan"))
    _w13_probe[(31,)](
        routed_input.reshape(-1),
        weight,
        scale,
        actual.reshape(-1),
        WORKERS=31,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    _assert_bitwise_equal(actual, expected, "W13 BF16 intermediate")
    assert not torch.isnan(actual).any()


def test_amd_megamoe_w2_and_route_combine_are_bitwise_current_decode() -> None:
    torch.manual_seed(314159)
    device = "cuda:0"
    routes = 4
    intermediate = (
        torch.randn(routes, 3072, device=device, dtype=torch.bfloat16) * 0.25
    ).contiguous()
    weight = torch.randint(
        0,
        256,
        (routes, 3584, 1536),
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.randint(
        123,
        130,
        (routes, 3584, 96),
        dtype=torch.uint8,
        device=device,
    )
    ids = torch.arange(routes, dtype=torch.int32, device=device).reshape(1, routes)
    route_weights = torch.rand((1, routes), dtype=torch.float32, device=device)
    route_weights /= route_weights.sum()
    expected = _current_stage2(intermediate, weight, scale, ids, route_weights)

    route_output = torch.full(
        (16, 3584),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    coordinates = torch.arange(routes, dtype=torch.int32, device=device)
    for route in range(routes):
        _w2_probe[(31,)](
            intermediate,
            weight,
            scale,
            route_output,
            coordinates[route : route + 1],
            coordinates[route : route + 1],
            WORKERS=31,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    route_count = torch.tensor([routes], dtype=torch.int32, device=device)
    targets = torch.ones(16, dtype=torch.int64, device=device)
    gates = torch.ones_like(targets)
    symmetric_output = torch.full(
        (10752,),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
    heap_bases = torch.zeros(8, dtype=torch.int64, device=device)
    group_global_ranks = torch.arange(8, dtype=torch.int64, device=device)
    diagnostics = torch.zeros(8, dtype=torch.int64, device=device)
    _combine_probe[(7,)](
        route_count,
        route_weights.reshape(-1),
        targets,
        gates,
        route_output,
        symmetric_output,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    actual = symmetric_output[7168:]
    _assert_bitwise_equal(actual, expected.reshape(-1), "W2 and route combine")
    assert not torch.isnan(actual).any()


def test_amd_megamoe_iris_math_is_bitwise_current_producer_direct() -> None:
    """Alias all peer heaps locally to isolate the inherited arithmetic/order."""

    torch.manual_seed(141421)
    device = "cuda:0"
    producer = (
        torch.randn(10752, device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    heap_bases = torch.zeros(8, dtype=torch.int64, device=device)
    group_global_ranks = torch.arange(8, dtype=torch.int64, device=device)

    def reference_ready_flags() -> torch.Tensor:
        flags = torch.full((21, 8), 2, dtype=torch.int32, device=device)
        flags[:, 0] = 0
        return flags

    def megamoe_generation_flags() -> torch.Tensor:
        flags = torch.ones((2, 21, 8), dtype=torch.int64, device=device)
        flags[:, :, 0] = 0
        return flags

    expected = torch.full_like(producer, float("nan"))
    reference_flags = reference_ready_flags()
    iris_reduce_symmetric_gluon_kernel[(21,)](
        producer,
        expected,
        reference_flags,
        *([0] * 8),
        RANK=0,
        WORLD_SIZE=8,
        TOTAL_NUMEL=10752,
        BLOCK_SIZE=512,
        NUM_PROGRAMS=21,
        NUM_TILES=21,
        NUM_WARPS=1,
        ELEMENT_DTYPE=gl.bfloat16,
        ELEMENTS_PER_WORD=4,
        num_warps=1,
    )

    actual = torch.full_like(producer, float("nan"))
    flags = megamoe_generation_flags()
    shared_gates = torch.ones(14, dtype=torch.int64, device=device)
    arrival = torch.zeros(1, dtype=torch.int64, device=device)
    gate = torch.zeros(1, dtype=torch.int64, device=device)
    fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
    diagnostics = torch.zeros(8, dtype=torch.int64, device=device)
    _iris_probe[(21,)](
        shared_gates,
        producer,
        actual,
        flags,
        arrival,
        gate,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    _assert_bitwise_equal(actual, expected, "Iris producer-direct reduction")
    assert gate.item() == 1
    assert torch.equal(flags, torch.ones_like(flags))


def test_amd_megamoe_final_is_bitwise_current_decode() -> None:
    torch.manual_seed(57721)
    device = "cuda:0"
    latent = (
        torch.randn(1, 3584, device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    shared = (
        torch.randn(1, 7168, device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    prefix = (
        torch.randn(1, 7168, device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    norm_weight = (
        torch.randn(3584, device=device, dtype=torch.bfloat16) * 0.1
    ).contiguous()
    projection_weight = (
        torch.randn(7168, 3584, device=device, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    expected = gluon_rmsnorm_linear_add_gfx950(
        latent,
        norm_weight,
        projection_weight,
        prefix,
        shared,
        eps=1.0e-5,
    )
    reduced = torch.cat((shared, latent), dim=1).contiguous().reshape(-1)
    actual = torch.full_like(prefix, float("nan"))
    _final_probe[(224,)](
        prefix.reshape(-1),
        norm_weight,
        projection_weight,
        reduced,
        actual.reshape(-1),
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    _assert_bitwise_equal(actual, expected, "final RMSNorm/latent-up/add")
    assert not torch.isnan(actual).any()


@pytest.mark.skipif(
    os.getenv(_PHASE_TIMING_ENV) != "1",
    reason=f"set {_PHASE_TIMING_ENV}=1 for the GPU0 phase diagnostic",
)
def test_amd_megamoe_warmed_phase_timing_diagnostic() -> None:
    """Time warmed MegaMoE phase probes and production decode kernels."""

    from tokenspeed_kernel._triton import triton

    device = "cuda:0"
    hidden = torch.full((7168,), 0.01, dtype=torch.bfloat16, device=device)
    router_weight = torch.full((896, 7168), 0.001, dtype=torch.bfloat16, device=device)
    routed_weight = torch.full((3584, 7168), 0.001, dtype=torch.bfloat16, device=device)
    shared_gate_up_weight = torch.full(
        (1536, 7168), 0.001, dtype=torch.bfloat16, device=device
    )
    router = torch.empty(896, dtype=torch.float32, device=device)
    routed_input = torch.empty(3584, dtype=torch.bfloat16, device=device)
    shared_input = torch.empty(768, dtype=torch.bfloat16, device=device)

    shared_down_weight = torch.full(
        (7168, 768), 0.001, dtype=torch.bfloat16, device=device
    )
    symmetric_output = torch.empty(10_752, dtype=torch.bfloat16, device=device)
    shared_arrivals = torch.zeros(14, dtype=torch.int64, device=device)
    shared_gates = torch.zeros(14, dtype=torch.int64, device=device)

    # A balanced L=2 rank has XCD populations [30] * 8. Removing shared
    # communication owners gives 112 expert workers on XCDs 0..3 and 114 on
    # XCDs 4..7. Time both exact worker counts.
    w13_weight = torch.full((2, 6144, 1792), 0x22, dtype=torch.uint8, device=device)
    w13_scale = torch.full((2, 6144, 112), 120, dtype=torch.uint8, device=device)
    intermediate = torch.empty((16, 3072), dtype=torch.bfloat16, device=device)
    w2_weight = torch.full((2, 3584, 1536), 0x22, dtype=torch.uint8, device=device)
    w2_scale = torch.full((2, 3584, 96), 120, dtype=torch.uint8, device=device)
    route_output = torch.zeros((16, 3584), dtype=torch.bfloat16, device=device)
    route_coordinate = torch.zeros(1, dtype=torch.int32, device=device)
    route_count = torch.tensor([2], dtype=torch.int32, device=device)
    route_weights = torch.full((16,), 0.5, dtype=torch.float32, device=device)
    route_targets = torch.ones(16, dtype=torch.int64, device=device)
    route_gates = torch.ones(16, dtype=torch.int64, device=device)
    fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
    heap_bases = torch.zeros(8, dtype=torch.int64, device=device)
    group_global_ranks = torch.arange(8, dtype=torch.int64, device=device)
    diagnostics = torch.zeros(8, dtype=torch.int64, device=device)

    prefix = torch.full((7168,), 0.125, dtype=torch.bfloat16, device=device)
    norm_weight = torch.ones(3584, dtype=torch.bfloat16, device=device)
    projection_weight = torch.full(
        (7168, 3584), 0.001, dtype=torch.bfloat16, device=device
    )
    reduced = torch.full((10_752,), 0.01, dtype=torch.bfloat16, device=device)
    final_output = torch.empty(7168, dtype=torch.bfloat16, device=device)

    local_ids = torch.full((1, 16), -1, dtype=torch.int32, device=device)
    local_ids[0, :2] = torch.arange(2, dtype=torch.int32, device=device)
    production_intermediate = torch.empty_like(intermediate)
    production_output = torch.empty((1, 3584), dtype=torch.bfloat16, device=device)
    production_shared_output = torch.empty(
        (1, 7168), dtype=torch.bfloat16, device=device
    )
    production_weights = torch.full(
        (1, 16), 1.0 / 16.0, dtype=torch.float32, device=device
    )

    def phase0() -> None:
        _phase0_probe[(240,)](
            hidden,
            router_weight,
            routed_weight,
            shared_gate_up_weight,
            router,
            routed_input,
            shared_input,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def shared_down() -> None:
        _shared_down_probe[(224,)](
            shared_input,
            shared_down_weight,
            symmetric_output,
            shared_arrivals,
            shared_gates,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def production_phase0() -> None:
        _latent_input_decode_kernel[(640,)](
            hidden,
            router_weight,
            routed_weight,
            shared_gate_up_weight,
            router,
            routed_input,
            shared_input,
            4.0,
            25.0,
            HAS_LINEAR_BETA=True,
            num_warps=4,
            num_stages=1,
            waves_per_eu=0,
        )

    def mega_w13(workers: int) -> None:
        _w13_probe[(workers,)](
            routed_input,
            w13_weight,
            w13_scale,
            intermediate,
            WORKERS=workers,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def mega_w2(workers: int) -> None:
        _w2_probe[(workers,)](
            intermediate,
            w2_weight,
            w2_scale,
            route_output,
            route_coordinate,
            route_coordinate,
            WORKERS=workers,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def route_combine() -> None:
        _combine_probe[(7,)](
            route_count,
            route_weights,
            route_targets,
            route_gates,
            route_output,
            symmetric_output,
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            diagnostics,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def final() -> None:
        _final_probe[(224,)](
            prefix,
            norm_weight,
            projection_weight,
            reduced,
            final_output,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )

    def production_w13() -> None:
        _stage1_a16w4_situ_warp_gemv[(16 * 384,)](
            routed_input.reshape(1, -1),
            w13_weight,
            w13_scale,
            production_intermediate,
            local_ids,
            3584,
            3072,
            16,
            3584,
            1,
            w13_weight.stride(0),
            w13_weight.stride(2),
            w13_weight.stride(1),
            w13_scale.stride(0),
            w13_scale.stride(1),
            w13_scale.stride(2),
            production_intermediate.stride(0),
            production_intermediate.stride(1),
            local_ids.stride(0),
            local_ids.stride(1),
            SITU_BETA=4.0,
            SITU_LINEAR_BETA=25.0,
            HAS_LINEAR_BETA=True,
            EXPERT_START=0,
            NUM_LOCAL_EXPERTS=2,
            LINEAR_WEIGHTS=True,
            W13_INTERLEAVED=False,
            NUM_PID_N=384,
            BLOCK_N=8,
            BLOCK_KB=1024,
            NUM_WARPS=4,
            MASK_K_TAIL=True,
            num_warps=4,
        )

    def production_w2(*, fuse_shared: bool) -> None:
        routed_programs = 448
        total_programs = routed_programs + (896 if fuse_shared else 0)
        _stage2_a16w4_warp_gemv_combine[(total_programs,)](
            production_intermediate,
            w2_weight,
            w2_scale,
            production_output,
            local_ids,
            production_weights,
            shared_input.reshape(1, -1),
            shared_down_weight,
            production_shared_output,
            3584,
            3072,
            production_intermediate.stride(0),
            production_intermediate.stride(1),
            w2_weight.stride(0),
            w2_weight.stride(2),
            w2_weight.stride(1),
            w2_scale.stride(0),
            w2_scale.stride(1),
            w2_scale.stride(2),
            production_output.stride(0),
            production_output.stride(1),
            local_ids.stride(0),
            local_ids.stride(1),
            production_weights.stride(0),
            production_weights.stride(1),
            FUSE_SHARED_DOWN=fuse_shared,
            NUM_ROUTED_PROGRAMS=routed_programs,
            TOP_K=16,
            EXPERT_START=0,
            NUM_LOCAL_EXPERTS=2,
            LINEAR_WEIGHTS=True,
            NUM_PID_N=448,
            BLOCK_N=8,
            BLOCK_KB=512,
            NUM_WARPS=8,
            num_warps=8,
        )

    launches: tuple[tuple[str, str, Callable[[], None]], ...] = (
        ("mega_phase0", "grid=240,subgroups=8", phase0),
        ("production_phase0", "grid=640,subgroups=4", production_phase0),
        ("mega_shared_down", "grid=224,subgroups=8", shared_down),
        ("mega_w13_w112", "grid=112,subgroups=8", lambda: mega_w13(112)),
        ("mega_w13_w114", "grid=114,subgroups=8", lambda: mega_w13(114)),
        ("mega_w2_w112", "grid=112,subgroups=8", lambda: mega_w2(112)),
        ("mega_w2_w114", "grid=114,subgroups=8", lambda: mega_w2(114)),
        ("mega_route_combine", "grid=7,subgroups=8,L=2", route_combine),
        ("mega_final", "grid=224,subgroups=8", final),
        ("production_w13", "grid=6144,subgroups=4,L=2/16", production_w13),
        (
            "production_w2",
            "grid=448,subgroups=8,L=2/16",
            lambda: production_w2(fuse_shared=False),
        ),
        (
            "production_w2_shared",
            "grid=1344,subgroups=8,L=2/16",
            lambda: production_w2(fuse_shared=True),
        ),
    )

    for _ in range(_PHASE_TIMING_WARMUPS):
        for _, _, launch in launches:
            launch()
    torch.cuda.synchronize()

    previous_hook = triton.knobs.runtime.jit_post_compile_hook

    def reject_late_compile(*_args, **_kwargs) -> None:
        raise AssertionError("phase timing replay attempted to compile")

    triton.knobs.runtime.jit_post_compile_hook = reject_late_compile
    try:
        results = [
            (shape, _time_retained_launch(name, launch))
            for name, shape, launch in launches
        ]
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous_hook

    for shape, result in results:
        assert float(result["p50_us"]) > 0.0
        print(
            "MEGAMOE_PHASE_TIMING "
            f"name={result['name']} {shape} "
            f"replays={result['replays']} "
            f"p50_us={float(result['p50_us']):.3f} "
            f"min_us={float(result['min_us']):.3f} "
            f"max_us={float(result['max_us']):.3f}",
            flush=True,
        )
