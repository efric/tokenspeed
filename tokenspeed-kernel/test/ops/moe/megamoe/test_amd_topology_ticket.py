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

"""Opt-in gfx950 stress for the cumulative XCD topology ticket fast path.

Run on an otherwise idle GPU 0 with the admitted system ROCr stack::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_TOPOLOGY_TICKET=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_topology_ticket.py

The synthetic probe checks many cumulative generations and deliberate invalid
populations.  The hardware probe adds the raw kernel's 128-KiB LDS residency
constraint and reads ``HW_REG_XCC_ID`` to validate the physical 30-per-XCD
mapping assumed by the admitted P=240 cooperative launch.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
    _arrive_topology_ticket,
    _poll_topology_or_poison,
)


_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_TOPOLOGY_TICKET"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_TOPOLOGY_TICKET_ITERATIONS"
_PROGRAMS = 240
_XCDS = 8
_WORKERS_PER_XCD = 30
_LDS_BYTES = gl.constexpr(128 * 1024)


@gluon.jit
def _synthetic_ticket_probe(
    xcc_by_pid,
    generation_ptr,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
    local_rank_by_pid,
):
    pid = gl.program_id(0)
    generation = gl.load(generation_ptr, cache_modifier=".cv")
    xcc = gl.load(xcc_by_pid + pid, cache_modifier=".cv")
    local_rank = _arrive_topology_ticket(
        generation,
        xcc,
        xcc_ticket,
        xcd_arrival,
        topology_gate,
    )
    gl.store(local_rank_by_pid + pid, local_rank, cache_modifier=".wt")


@gluon.jit
def _synthetic_ticket_fail_closed_probe(
    xcc_by_pid,
    generation_ptr,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
    local_rank_by_pid,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    diagnostics,
    TIMEOUT_NS: gl.constexpr,
):
    """Exercise the production topology timeout on an invalid population."""

    pid = gl.program_id(0)
    generation = gl.load(generation_ptr, cache_modifier=".cv")
    xcc = gl.load(xcc_by_pid + pid, cache_modifier=".cv")
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
        diagnostics,
        pid,
        3,
        xcc,
        0,
        TIMEOUT_NS,
    )
    gl.store(
        local_rank_by_pid + pid,
        gl.where(topology_ready, local_rank, -2),
        cache_modifier=".wt",
    )


@gluon.jit
def _hardware_ticket_probe(
    generation_ptr,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
    xcc_by_pid,
    local_rank_by_pid,
    fatal_epoch,
    heap_bases,
    group_global_ranks,
    diagnostics,
    TIMEOUT_NS: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, 512, layout=layout)

    # Match the production kernel's one-workgroup-per-compute-unit resource
    # contract rather than relying on an unconstrained scheduler sample.
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
    if gl.sum(residency_token.to(gl.int32), axis=0) < 0:
        gl.store(topology_gate, -1, cache_modifier=".wt")

    generation = gl.load(generation_ptr, cache_modifier=".cv")
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
        diagnostics,
        pid,
        3,
        xcc,
        0,
        TIMEOUT_NS,
    )
    if not topology_ready:
        return
    gl.store(
        xcc_by_pid + pid + lane * 0,
        xcc,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.store(
        local_rank_by_pid + pid + lane * 0,
        local_rank,
        mask=lane == 0,
        cache_modifier=".wt",
    )


def _has_qualified_gpu0() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return (
        architecture.startswith("gfx950")
        and int(getattr(properties, "multi_processor_count", 0)) == 256
    )


pytestmark = [
    pytest.mark.skipif(
        os.getenv(_GATE_ENV) != "1",
        reason=f"set {_GATE_ENV}=1 for the topology-ticket stress",
    ),
    pytest.mark.skipif(not _has_qualified_gpu0(), reason="requires a 256-CU gfx950"),
]


def _assert_exact_mapping(
    xcc_by_generation: torch.Tensor,
    local_rank_by_generation: torch.Tensor,
) -> None:
    for xccs, ranks in zip(
        xcc_by_generation.cpu(),
        local_rank_by_generation.cpu(),
        strict=True,
    ):
        communication_indices = []
        for xcc in range(_XCDS):
            owned = ranks[xccs == xcc]
            assert torch.equal(
                owned.sort().values,
                torch.arange(_WORKERS_PER_XCD, dtype=owned.dtype),
            )
            communication_indices.extend((owned * _XCDS + xcc).tolist())
        assert sorted(communication_indices) == list(range(_PROGRAMS))
        assert sum(index < 14 for index in communication_indices) == 14
        assert sum(index >= 14 for index in communication_indices) == 226


def _assert_ticket_probe_isa(assembly: str) -> None:
    """Require relaxed tickets plus merged release/acquire publication."""

    lines = assembly.lower().splitlines()

    def has_fence(index: int) -> bool:
        neighborhood = "\n".join(lines[max(0, index - 6) : min(len(lines), index + 7)])
        return "buffer_wbl2" in neighborhood or "buffer_inv" in neighborhood

    relaxed_adds = [
        index
        for index, line in enumerate(lines)
        if "global_atomic_add_x2" in line and not has_fence(index)
    ]
    assert len(relaxed_adds) == 2
    assert all(lines[index].rstrip().endswith("sc0") for index in relaxed_adds)

    release_swaps = [
        index
        for index, line in enumerate(lines)
        if "global_atomic_swap_x2" in line and has_fence(index)
    ]
    assert len(release_swaps) == 1
    assert " sc0" not in lines[release_swaps[0]]
    assert " sc1" not in lines[release_swaps[0]]

    polls = re.findall(
        r"\bs_memrealtime\b"
        r"(?:(?!\bs_memrealtime\b).)*?"
        r"\bglobal_load_dwordx2\b[^\n]*\bsc1\s*$"
        r"(?:(?!\bds_write_b8\b).)*?\bbuffer_inv\b[^\n]*\bsc1\s*$"
        r"(?:(?!\bds_write_b8\b).)*?"
        r"\bds_write_b8\b"
        r"(?:(?!\bds_read_u8\b).)*?\bs_barrier\b"
        r"(?:(?!\bds_read_u8\b).)*?\bds_read_u8\b"
        r"(?:(?!;;#asmstart).)*?"
        r";;#asmstart\s+s_waitcnt vmcnt\(0\)\s+;;#asmend"
        r"(?:(?!\bs_barrier\b).)*?\bs_barrier\b",
        assembly.lower(),
        flags=re.DOTALL | re.MULTILINE,
    )
    assert len(polls) == 1
    assert polls[0].count("s_barrier") == 2


def test_synthetic_topology_ticket_generations_and_fail_closed_populations() -> None:
    device = "cuda:0"
    iterations = int(os.getenv(_ITERATIONS_ENV, "256"))
    if iterations < 2:
        raise ValueError(f"{_ITERATIONS_ENV} must be at least 2")

    generator = torch.Generator(device=device)
    generator.manual_seed(20260814)
    base = torch.arange(_XCDS, dtype=torch.int32, device=device).repeat_interleave(
        _WORKERS_PER_XCD
    )
    xcc_by_generation = torch.stack(
        [
            base[torch.randperm(_PROGRAMS, generator=generator, device=device)]
            for _ in range(iterations)
        ]
    )
    generation_index = torch.arange(1, iterations + 1, dtype=torch.int64, device=device)
    generations = torch.stack(
        (generation_index, torch.zeros_like(generation_index)), dim=1
    )
    local_ranks = torch.full_like(xcc_by_generation, -1)
    xcc_ticket = torch.zeros(_XCDS, dtype=torch.int64, device=device)
    xcd_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    topology_gate = torch.zeros(1, dtype=torch.int64, device=device)

    for index in range(iterations):
        _synthetic_ticket_probe[(_PROGRAMS,)](
            xcc_by_generation[index],
            generations[index : index + 1],
            xcc_ticket,
            xcd_arrival,
            topology_gate,
            local_ranks[index],
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
    torch.cuda.synchronize()

    assert torch.equal(
        xcc_ticket.cpu(),
        torch.full((_XCDS,), iterations * _WORKERS_PER_XCD, dtype=torch.int64),
    )
    assert xcd_arrival.item() == iterations * _XCDS
    assert topology_gate.item() == iterations
    _assert_exact_mapping(xcc_by_generation, local_ranks)

    # Begin from a clean predecessor and deliberately transfer one workgroup
    # from XCD 1 to XCD 0. XCD 0 claims one future-generation rank while XCD 1
    # never claims rank 29, so only seven XCDs arrive and the gate stays closed.
    unbalanced = base.clone()
    unbalanced[_WORKERS_PER_XCD] = 0
    sentinel_tickets = torch.full((_XCDS + 2,), -777, dtype=torch.int64, device=device)
    sentinel_tickets[1:-1].zero_()
    invalid_ranks = torch.full((_PROGRAMS,), -1, dtype=torch.int32, device=device)
    invalid_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    invalid_gate = torch.zeros(1, dtype=torch.int64, device=device)
    fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
    heap_bases = torch.zeros(_XCDS, dtype=torch.int64, device=device)
    group_global_ranks = torch.arange(_XCDS, dtype=torch.int64, device=device)
    diagnostics = torch.zeros(8, dtype=torch.int64, device=device)
    generation_one = generations[:1]
    _synthetic_ticket_fail_closed_probe[(_PROGRAMS,)](
        unbalanced,
        generation_one,
        sentinel_tickets[1:-1],
        invalid_arrival,
        invalid_gate,
        invalid_ranks,
        fatal_epoch,
        heap_bases,
        group_global_ranks,
        diagnostics,
        TIMEOUT_NS=10_000_000,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        sentinel_tickets.cpu(),
        torch.tensor([-777, 31, 29, 30, 30, 30, 30, 30, 30, -777]),
    )
    assert invalid_arrival.item() == 7
    assert invalid_gate.item() == 0
    assert fatal_epoch.item() == 1
    assert diagnostics[0].item() == 1
    assert diagnostics[2].item() == 3
    assert diagnostics[3].item() == 1
    assert diagnostics[5].item() == 1
    assert diagnostics[6].item() == 0
    assert diagnostics[7].item() == 0

    # A raw four-bit XCC ID outside [0, 8) cannot address the compact ticket
    # allocation and likewise removes the eighth exact contribution.
    invalid_xcc = base.clone()
    invalid_xcc[0] = 8
    sentinel_tickets.fill_(-777)
    sentinel_tickets[1:-1].zero_()
    invalid_arrival.zero_()
    invalid_gate.zero_()
    _synthetic_ticket_probe[(_PROGRAMS,)](
        invalid_xcc,
        generation_one,
        sentinel_tickets[1:-1],
        invalid_arrival,
        invalid_gate,
        invalid_ranks,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()
    assert sentinel_tickets[0].item() == -777
    assert sentinel_tickets[-1].item() == -777
    assert invalid_arrival.item() == 7
    assert invalid_gate.item() == 0


def test_hardware_topology_ticket_matches_p240_cooperative_residency() -> None:
    device = "cuda:0"
    iterations = int(os.getenv(_ITERATIONS_ENV, "256"))
    if iterations < 2:
        raise ValueError(f"{_ITERATIONS_ENV} must be at least 2")
    generation_index = torch.arange(1, iterations + 1, dtype=torch.int64, device=device)
    generations = torch.stack(
        (generation_index, torch.zeros_like(generation_index)), dim=1
    )
    xcc_by_generation = torch.full(
        (iterations, _PROGRAMS), -1, dtype=torch.int32, device=device
    )
    local_ranks = torch.full_like(xcc_by_generation, -1)
    xcc_ticket = torch.zeros(_XCDS, dtype=torch.int64, device=device)
    xcd_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    topology_gate = torch.zeros(1, dtype=torch.int64, device=device)
    fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
    heap_bases = torch.zeros(_XCDS, dtype=torch.int64, device=device)
    group_global_ranks = torch.arange(_XCDS, dtype=torch.int64, device=device)
    diagnostics = torch.zeros(8, dtype=torch.int64, device=device)

    def arguments(index: int) -> tuple[torch.Tensor, ...]:
        return (
            generations[index : index + 1],
            xcc_ticket,
            xcd_arrival,
            topology_gate,
            xcc_by_generation[index],
            local_ranks[index],
            fatal_epoch,
            heap_bases,
            group_global_ranks,
            diagnostics,
        )

    compiled = _hardware_ticket_probe.warmup(
        *arguments(0),
        TIMEOUT_NS=1_000_000_000,
        grid=(_PROGRAMS,),
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )
    _assert_ticket_probe_isa(compiled.asm["amdgcn"])
    with patch.object(
        _hardware_ticket_probe,
        "_do_compile",
        side_effect=AssertionError("topology generation replay attempted to recompile"),
    ):
        for index in range(iterations):
            _hardware_ticket_probe[(_PROGRAMS,)](
                *arguments(index),
                TIMEOUT_NS=1_000_000_000,
                num_warps=8,
                num_stages=1,
                waves_per_eu=2,
                launch_cooperative_grid=True,
            )
    torch.cuda.synchronize()

    assert torch.equal(
        xcc_ticket.cpu(),
        torch.full((_XCDS,), iterations * _WORKERS_PER_XCD, dtype=torch.int64),
    )
    assert xcd_arrival.item() == iterations * _XCDS
    assert topology_gate.item() == iterations
    assert fatal_epoch.item() == 0
    assert not torch.count_nonzero(diagnostics)
    _assert_exact_mapping(xcc_by_generation, local_ranks)
