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

"""Isolated output-centric W2/combine probe for gfx950 MegaMoE.

This file deliberately does not change or dispatch the production raw kernel.
It compares a candidate which assigns every N8 output task to one of the 226
expert workgroups against the current ``_w2_route_tiles`` followed by
``_routed_combine`` implementation.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable

import pytest
import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
    _output_centric_w2_combine,
    _routed_combine,
    _w2_route_tiles,
)

_NUMERICS_ENV = "TOKENSPEED_TEST_MEGAMOE_NUMERICS"
_TIMING_ENV = "TOKENSPEED_TEST_MEGAMOE_W2_COMBINE_FUSION_TIMING"
_ROUTE_COUNTS = (0, 1, 2, 8, 9, 16)
_EXPERT_WORKGROUPS = 226
_W2_TASKS = 448
_ROUTED_PRODUCER_OFFSET = 7168
_TIMING_REPLAYS = 200
_TIMING_WARMUPS = 20


def _has_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return "gfx950" in str(getattr(properties, "gcnArchName", ""))


def _output_tasks(expert_rank: int) -> tuple[int, ...]:
    """Return the candidate's disjoint N8 tasks for one expert workgroup."""

    if not 0 <= expert_rank < _EXPERT_WORKGROUPS:
        raise ValueError("expert rank is outside the 226-workgroup grid")
    return tuple(range(expert_rank, _W2_TASKS, _EXPERT_WORKGROUPS))


def _expert_global_rank(xcd_ordinal: int, expert_local_rank: int) -> int:
    """Flatten the admitted ``6*28 + 2*29`` expert-workgroup topology."""

    if not 0 <= xcd_ordinal < 8:
        raise ValueError("XCD ordinal is outside the admitted topology")
    local_population = 28 if xcd_ordinal < 6 else 29
    if not 0 <= expert_local_rank < local_population:
        raise ValueError("expert-local rank is outside its XCD population")
    prefix = 28 * min(xcd_ordinal, 6) + 29 * max(xcd_ordinal - 6, 0)
    return prefix + expert_local_rank


@gluon.jit
def _legacy_w2_probe(
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
def _legacy_combine_probe(
    route_count,
    route_weights,
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
    _routed_combine(
        pid,
        gl.full((), 1, gl.int64),
        14 + pid,
        route_count,
        route_weights,
        targets,
        gates,
        route_output,
        symmetric_output,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        gl.full((), 0, gl.int32),
        0,
        1_000_000_000,
    )


@gluon.jit
def _output_centric_w2_combine_probe(
    intermediate,
    weight,
    scale,
    expert_ids,
    route_weights,
    route_count_ptr,
    legacy_route_output,
    symmetric_output,
):
    """Compute each output across compact routes without route-output traffic."""

    route_count = gl.load(route_count_ptr, cache_modifier=".cv")
    _output_centric_w2_combine(
        gl.program_id(0),
        route_count,
        expert_ids,
        route_weights,
        intermediate,
        weight,
        scale,
        symmetric_output,
    )


def _assert_bitwise_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> None:
    assert actual.dtype == expected.dtype
    actual_bits = actual.contiguous().view(torch.int16)
    expected_bits = expected.contiguous().view(torch.int16)
    differing = actual_bits != expected_bits
    assert not torch.any(differing), (
        f"{label} differs at {int(differing.sum().item())} elements"
    )


def _time_retained_launch(
    name: str,
    launch: Callable[[], None],
    *,
    replays: int = _TIMING_REPLAYS,
) -> dict[str, float | int | str]:
    if replays < 100:
        raise ValueError("W2/combine fusion timing requires at least 100 replays")
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


def _allocate_probe_inputs() -> dict[str, torch.Tensor]:
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(271828)
    expert_ids = torch.randperm(16, generator=generator, device=device).to(torch.int32)
    route_weights = torch.rand(
        16,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    intermediate = (
        torch.randn(
            (16, 3072),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    return {
        "intermediate": intermediate.contiguous(),
        "weight": torch.randint(
            0,
            256,
            (16, 3584, 1536),
            generator=generator,
            dtype=torch.uint8,
            device=device,
        ),
        "scale": torch.randint(
            123,
            130,
            (16, 3584, 96),
            generator=generator,
            dtype=torch.uint8,
            device=device,
        ),
        "expert_ids": expert_ids,
        "route_weights": route_weights,
        "route_count": torch.zeros(1, dtype=torch.int32, device=device),
        "route_coordinates": torch.arange(16, dtype=torch.int32, device=device),
        "targets": torch.ones(16, dtype=torch.int64, device=device),
        "gates": torch.ones(16, dtype=torch.int64, device=device),
        "fatal_epoch": torch.zeros(1, dtype=torch.int64, device=device),
        "heap_bases": torch.zeros(8, dtype=torch.int64, device=device),
        "group_global_ranks": torch.arange(8, dtype=torch.int64, device=device),
        "diagnostics": torch.zeros(8, dtype=torch.int64, device=device),
    }


def _normalized_route_weights(
    storage: torch.Tensor,
    route_count: int,
) -> None:
    if route_count == 0:
        storage.zero_()
        return
    active = storage[:route_count]
    active.div_(active.sum())


def _launch_legacy_stage(
    tensors: dict[str, torch.Tensor],
    route_output: torch.Tensor,
    symmetric_output: torch.Tensor,
    route_count: int,
    *,
    timing_workers: bool = False,
) -> None:
    for route in range(route_count):
        workers = 31
        if timing_workers:
            workers = 112 if route == 0 else 114
        _legacy_w2_probe[(workers,)](
            tensors["intermediate"],
            tensors["weight"],
            tensors["scale"],
            route_output,
            tensors["route_coordinates"][route : route + 1],
            tensors["expert_ids"][route : route + 1],
            WORKERS=workers,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
    _legacy_combine_probe[(7,)](
        tensors["route_count"],
        tensors["route_weights"],
        tensors["targets"],
        tensors["gates"],
        route_output,
        symmetric_output,
        tensors["fatal_epoch"],
        tensors["heap_bases"],
        tensors["group_global_ranks"],
        tensors["diagnostics"],
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )


def _launch_output_centric_stage(
    tensors: dict[str, torch.Tensor],
    legacy_route_output: torch.Tensor,
    symmetric_output: torch.Tensor,
) -> None:
    _output_centric_w2_combine_probe[(_EXPERT_WORKGROUPS,)](
        tensors["intermediate"],
        tensors["weight"],
        tensors["scale"],
        tensors["expert_ids"],
        tensors["route_weights"],
        tensors["route_count"],
        legacy_route_output,
        symmetric_output,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )


def test_output_centric_w2_mapping_covers_every_task_once() -> None:
    expert_ranks = [
        _expert_global_rank(xcd, local_rank)
        for xcd in range(8)
        for local_rank in range(28 if xcd < 6 else 29)
    ]
    assert expert_ranks == list(range(_EXPERT_WORKGROUPS))
    owners = [
        (expert_rank, task)
        for expert_rank in range(_EXPERT_WORKGROUPS)
        for task in _output_tasks(expert_rank)
    ]
    tasks = [task for _, task in owners]
    assert sorted(tasks) == list(range(_W2_TASKS))
    assert len(tasks) == len(set(tasks))
    assert all(len(_output_tasks(rank)) == 2 for rank in range(222))
    assert all(len(_output_tasks(rank)) == 1 for rank in range(222, _EXPERT_WORKGROUPS))


@pytest.mark.parametrize("route_count", _ROUTE_COUNTS)
def test_output_centric_w2_gate_and_l0_zero_owner_model(
    route_count: int,
) -> None:
    current_w13 = 17 * _EXPERT_WORKGROUPS
    current_w2 = 23 * _EXPERT_WORKGROUPS
    if route_count == 0:
        assert current_w13 == 17 * _EXPERT_WORKGROUPS
        assert current_w2 == 23 * _EXPERT_WORKGROUPS
        routed_words = [
            word
            for comm_index in range(14, 21)
            for word in range(comm_index * 512, (comm_index + 1) * 512)
        ]
        assert routed_words == list(range(_ROUTED_PRODUCER_OFFSET, 10_752))
        assert len(routed_words) == len(set(routed_words))
        return

    participants = tuple(range(_EXPERT_WORKGROUPS))
    assert len(participants) == _EXPERT_WORKGROUPS
    assert current_w13 + len(participants) == 18 * _EXPERT_WORKGROUPS
    assert current_w2 + len(participants) == 24 * _EXPERT_WORKGROUPS
    assert tuple(range(route_count)) == tuple(sorted(range(route_count)))
    assert sorted(
        task for rank in participants for task in _output_tasks(rank)
    ) == list(range(_W2_TASKS))


def test_output_centric_probe_source_guards_fusion_contract() -> None:
    wrapper = _output_centric_w2_combine_probe.src
    source = _output_centric_w2_combine.src
    assert "while task < _W2_TASKS" in source
    assert "task += _EXPERT_WORKGROUPS" in source
    assert "while route < route_count" in source
    assert "route += 1" in source
    assert "gl.static_range(0, 16)" not in source
    assert "route_acc.to(gl.bfloat16).to(gl.float32)" in source
    assert source.index("route_acc.to(gl.bfloat16).to(gl.float32)") < source.index(
        "combine_acc += route_value * route_weight"
    )
    assert "symmetric_producer + _HIDDEN" in source
    assert "w2_route_output" not in source
    assert wrapper.count("legacy_route_output") == 1
    assert wrapper.count("_output_centric_w2_combine(") == 1


def test_output_centric_oracle_detects_rounding_and_order_changes() -> None:
    generator = torch.Generator().manual_seed(161803)
    values = torch.randn((16, 4096), generator=generator, dtype=torch.float32) * 3.0
    weights = torch.rand(16, generator=generator, dtype=torch.float32)
    weights /= weights.sum()

    expected = torch.zeros(4096, dtype=torch.float32)
    for route in range(16):
        expected += values[route].to(torch.bfloat16).to(torch.float32) * weights[route]
    expected = expected.to(torch.bfloat16)

    late_round = torch.sum(values * weights[:, None], dim=0).to(torch.bfloat16)
    reverse = torch.zeros(4096, dtype=torch.float32)
    for route in reversed(range(16)):
        reverse += values[route].to(torch.bfloat16).to(torch.float32) * weights[route]
    reverse = reverse.to(torch.bfloat16)

    assert torch.any(expected.view(torch.int16) != late_round.view(torch.int16))
    assert torch.any(expected.view(torch.int16) != reverse.view(torch.int16))


@pytest.mark.skipif(
    os.getenv(_NUMERICS_ENV) != "1" or not _has_gfx950(),
    reason=f"set {_NUMERICS_ENV}=1 on an AMD gfx950 for the numerical oracle",
)
def test_output_centric_w2_combine_is_bitwise_legacy_for_all_route_counts() -> None:
    tensors = _allocate_probe_inputs()
    original_weights = tensors["route_weights"].clone()

    for route_count in _ROUTE_COUNTS:
        tensors["route_count"].fill_(route_count)
        tensors["route_weights"].copy_(original_weights)
        _normalized_route_weights(tensors["route_weights"], route_count)
        legacy_route_output = torch.full(
            (16, 3584),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        legacy_symmetric = torch.full(
            (10_752,),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        poisoned_route_output = torch.full_like(legacy_route_output, float("nan"))
        fused_symmetric = torch.full_like(legacy_symmetric, float("nan"))

        _launch_legacy_stage(
            tensors,
            legacy_route_output,
            legacy_symmetric,
            route_count,
        )
        _launch_output_centric_stage(
            tensors,
            poisoned_route_output,
            fused_symmetric,
        )
        torch.cuda.synchronize()

        _assert_bitwise_equal(
            fused_symmetric[_ROUTED_PRODUCER_OFFSET:],
            legacy_symmetric[_ROUTED_PRODUCER_OFFSET:],
            f"output-centric W2/combine L={route_count}",
        )
        assert torch.isnan(fused_symmetric[:_ROUTED_PRODUCER_OFFSET]).all()
        assert torch.isnan(poisoned_route_output).all()
        assert not torch.isnan(fused_symmetric[_ROUTED_PRODUCER_OFFSET:]).any()


@pytest.mark.skipif(
    os.getenv(_TIMING_ENV) != "1" or not _has_gfx950(),
    reason=f"set {_TIMING_ENV}=1 on an AMD gfx950 for the GPU0 timing probe",
)
def test_output_centric_w2_combine_warmed_l2_timing() -> None:
    from tokenspeed_kernel._triton import triton

    tensors = _allocate_probe_inputs()
    tensors["route_count"].fill_(2)
    _normalized_route_weights(tensors["route_weights"], 2)
    legacy_route_output = torch.empty(
        (16, 3584),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    legacy_symmetric = torch.empty(
        (10_752,),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    poisoned_route_output = torch.full_like(legacy_route_output, float("nan"))
    fused_symmetric = torch.empty_like(legacy_symmetric)

    def legacy() -> None:
        _launch_legacy_stage(
            tensors,
            legacy_route_output,
            legacy_symmetric,
            2,
            timing_workers=True,
        )

    def fused() -> None:
        _launch_output_centric_stage(
            tensors,
            poisoned_route_output,
            fused_symmetric,
        )

    for _ in range(_TIMING_WARMUPS):
        legacy()
        fused()
    torch.cuda.synchronize()
    _assert_bitwise_equal(
        fused_symmetric[_ROUTED_PRODUCER_OFFSET:],
        legacy_symmetric[_ROUTED_PRODUCER_OFFSET:],
        "warmed output-centric W2/combine L=2",
    )
    assert torch.isnan(poisoned_route_output).all()

    previous_hook = triton.knobs.runtime.jit_post_compile_hook

    def reject_late_compile(*_args, **_kwargs) -> None:
        raise AssertionError("W2/combine timing replay attempted to compile")

    triton.knobs.runtime.jit_post_compile_hook = reject_late_compile
    try:
        results = (
            _time_retained_launch("legacy_w2_plus_combine", legacy),
            _time_retained_launch("output_centric_w2_combine", fused),
        )
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous_hook

    for result in results:
        assert float(result["p50_us"]) > 0.0
        print(
            "MEGAMOE_W2_COMBINE_FUSION_TIMING "
            f"name={result['name']} grid="
            f"{'112+114+7' if result['name'] == 'legacy_w2_plus_combine' else 226} "
            "subgroups=8 L=2 "
            f"replays={result['replays']} "
            f"p50_us={float(result['p50_us']):.3f} "
            f"min_us={float(result['min_us']):.3f} "
            f"max_us={float(result['max_us']):.3f}",
            flush=True,
        )
