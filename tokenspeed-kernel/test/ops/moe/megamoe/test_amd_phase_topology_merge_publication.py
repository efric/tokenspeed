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

"""Opt-in gfx950 litmus for merging phase publication with topology tickets.

This test is deliberately self-contained: it does not import or modify the raw
MegaMoE kernel, admission manifest, or implementation marker.  It isolates the
proposed local-GPU publication chain::

    all eight subgroups write generation-reused payload with ``.wt``
        -> every subgroup drains VMEM
            -> workgroup barrier
                -> relaxed cumulative ``xcc_ticket[xcc]`` RMW
                    -> exact local rank 29 only
                        -> relaxed cumulative ``xcd_arrival`` RMW
                            -> exact eighth contribution only
                                -> release the merged phase/topology gate
                                    -> exact acquire poll
                                        -> every-subgroup VMEM ACK
                                            -> workgroup barrier
                                                -> ``.cv`` payload consumers

Every workgroup derives its generation by coherently loading the prior merged
gate and adding one.  A gate release therefore both proves the admitted
30-workgroup population on all eight XCDs and publishes the phase payload.
The cumulative counters reject a 31/29 population, an invalid four-bit XCC ID,
and counters that are either one generation stale or one generation ahead.

The hardware mode uses 240 cooperative workgroups, eight subgroups per
workgroup, and a 128-KiB LDS allocation so that gfx950 can retain at most one
workgroup per compute unit.  It defaults to 4096 fresh dispatches of the same
compiled runner and rejects any replay that enters JIT compilation.  Payload
words vary by generation and are reused at the same addresses.  A bijective
consumer permutation makes every subgroup load another producer's payload and
the host verifies that the observed physical mapping contains cross-XCD edges.

Run only on a reserved 256-CU gfx950 after explicit authorization::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/\
test_amd_phase_topology_merge_publication.py

Production source remains unchanged regardless of whether this opt-in litmus
passes.  The language-level transitive release chain is intentionally absent
from the two relaxed arrival levels; qualification is specific to the loaded
gfx950 code object and its recorded AMDGCN hash.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from tokenspeed_kernel_amd._triton import gl, gluon

_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE_WATCHDOG_SECONDS"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_PHASE_TOPOLOGY_MERGE_DUMP_AMDGCN"

_PROGRAMS = 240
_XCDS = 8
_WORKERS_PER_XCD = 30
_SUBGROUPS = 8
_LANES = 512
_SOURCE_SHIFT = 113
_LDS_BYTES = gl.constexpr(128 * 1024)
_KERNEL_PROGRAMS = gl.constexpr(_PROGRAMS)
_KERNEL_XCDS = gl.constexpr(_XCDS)
_KERNEL_WORKERS_PER_XCD = gl.constexpr(_WORKERS_PER_XCD)
_KERNEL_SUBGROUPS = gl.constexpr(_SUBGROUPS)
_KERNEL_LANES = gl.constexpr(_LANES)
_KERNEL_SOURCE_SHIFT = gl.constexpr(_SOURCE_SHIFT)


@gluon.jit
def _wait_subgroup_vmem_ack():
    """Complete preceding VMEM in every executing subgroup."""

    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _deliberate_workgroup_skew(pid, generation):
    """Vary ticket order across retained generations without memory traffic."""

    skew_class = (pid * 5 + generation * 3) % 8
    for skew_round in gl.static_range(0, 7):
        if skew_class > skew_round:
            gl.inline_asm_elementwise(
                "s_sleep 4",
                "=r",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )


@gluon.jit
def _hardware_xcc_id():
    """Read the dense four-bit XCC identifier used by the admitted launch."""

    return (
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


@gluon.jit
def _arrive_hierarchical_phase(
    generation,
    xcc,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
):
    """Publish only the exact rank-29/eighth-contribution generation."""

    ticket = gl.full((), -1, gl.int64)
    local_rank = gl.full((), -1, gl.int32)
    xcd_ticket = gl.full((), -1, gl.int64)
    released = gl.full((), 0, gl.int32)
    if (xcc >= 0) & (xcc < _KERNEL_XCDS):
        ticket = gl.atomic_add(
            xcc_ticket + xcc,
            1,
            sem="relaxed",
            scope="gpu",
        )
        local_rank = (ticket % _KERNEL_WORKERS_PER_XCD).to(gl.int32)
        ticket_generation = ticket // _KERNEL_WORKERS_PER_XCD + 1
        if (ticket_generation == generation) & (
            local_rank == _KERNEL_WORKERS_PER_XCD - 1
        ):
            xcd_ticket = gl.atomic_add(
                xcd_arrival,
                1,
                sem="relaxed",
                scope="gpu",
            )
            topology_generation = xcd_ticket // _KERNEL_XCDS + 1
            if (xcd_ticket % _KERNEL_XCDS == _KERNEL_XCDS - 1) & (
                topology_generation == generation
            ):
                gl.atomic_xchg(
                    topology_gate,
                    generation,
                    sem="release",
                    scope="gpu",
                )
                released = 1
    return ticket, local_rank, xcd_ticket, released


@gluon.jit
def _phase_topology_merge_litmus_kernel(
    payload,
    xcc_override,
    xcc_ticket,
    xcd_arrival,
    topology_gate,
    ticket_out,
    local_rank_out,
    xcd_ticket_out,
    xcc_out,
    generation_out,
    release_out,
    consumer_visits,
    failure_generation,
    failure_observed,
    fatal_generation_by_pid,
    timeout_generation,
    timeout_observed,
    timeout_marker,
    TIMEOUT_NS: gl.constexpr,
    USE_HARDWARE_XCC: gl.constexpr,
):
    """Publish and consume one merged phase/topology generation."""

    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [_KERNEL_SUBGROUPS], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup = lane // 64
    subgroup_leader = lane % 64 == 0

    # Every workgroup reads the same predecessor before any workgroup can
    # release its successor: release requires a ticket from all 240 producers.
    prior_generation = gl.load(topology_gate, cache_modifier=".cv")
    generation = prior_generation + 1

    # Match the persistent kernel's one-workgroup-per-compute-unit contract.
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    residency = gl.allocate_shared_memory(gl.uint8, [_LDS_BYTES], shared_layout)
    residency_tile = residency.slice(0, _KERNEL_LANES)
    residency_tile.store((lane % 251).to(gl.uint8))
    gl.barrier()
    residency_token = residency_tile.load(layout).to(gl.int64)

    if USE_HARDWARE_XCC:
        xcc = _hardware_xcc_id()
    else:
        xcc = gl.load(xcc_override + pid, cache_modifier=".cv")
    gl.store(
        xcc_out + pid + lane * 0,
        xcc,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.store(
        generation_out + pid + lane * 0,
        generation,
        mask=lane == 0,
        cache_modifier=".wt",
    )

    _deliberate_workgroup_skew(pid, generation)
    payload_index = pid * _KERNEL_LANES + lane
    expected = generation * 1_000_000_000 + payload_index.to(gl.int64)
    expected += residency_token
    gl.amd.cdna4.buffer_store(
        expected,
        payload,
        payload_index.to(gl.int32),
        cache=".wt",
    )

    # All eight producer subgroups complete their `.wt` stores before the
    # scalar workgroup ticket can represent their payload contribution.
    _wait_subgroup_vmem_ack()
    gl.barrier()
    ticket, local_rank, xcd_ticket_value, released = _arrive_hierarchical_phase(
        generation,
        xcc,
        xcc_ticket,
        xcd_arrival,
        topology_gate,
    )
    gl.store(
        ticket_out + pid + lane * 0,
        ticket,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.store(
        local_rank_out + pid + lane * 0,
        local_rank,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.store(
        xcd_ticket_out + pid + lane * 0,
        xcd_ticket_value,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.store(
        release_out + pid + lane * 0,
        released,
        mask=lane == 0,
        cache_modifier=".wt",
    )

    ready = gl.atomic_poll(
        topology_gate,
        generation,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    # The acquire's BUFFER_INV completes through VMEM.  Every subgroup must
    # acknowledge it before the convergent payload-consumption barrier.
    _wait_subgroup_vmem_ack()
    gl.barrier()

    if not ready:
        observed = gl.load(topology_gate, cache_modifier=".cv")
        gl.store(
            fatal_generation_by_pid + pid + lane * 0,
            generation,
            mask=lane == 0,
            cache_modifier=".wt",
        )
        diagnostic_index = pid * _KERNEL_SUBGROUPS + subgroup
        gl.store(
            timeout_generation + diagnostic_index,
            generation,
            mask=subgroup_leader,
            cache_modifier=".wt",
        )
        gl.store(
            timeout_observed + diagnostic_index,
            observed,
            mask=subgroup_leader,
            cache_modifier=".wt",
        )
        gl.store(
            timeout_marker + diagnostic_index,
            1,
            mask=subgroup_leader,
            cache_modifier=".wt",
        )
        return

    source_pid = (pid + _KERNEL_SOURCE_SHIFT) % _KERNEL_PROGRAMS
    source_index = source_pid * _KERNEL_LANES + lane
    expected_source = (
        generation * 1_000_000_000
        + source_index.to(gl.int64)
        + (lane % 251).to(gl.int64)
    )
    observed = gl.amd.cdna4.buffer_load(
        payload,
        source_index.to(gl.int32),
        cache=".cv",
    )
    mismatch = observed != expected_source
    gl.store(
        failure_generation + pid * _KERNEL_LANES + lane,
        generation,
        mask=mismatch,
        cache_modifier=".wt",
    )
    gl.store(
        failure_observed + pid * _KERNEL_LANES + lane,
        observed,
        mask=mismatch,
        cache_modifier=".wt",
    )

    # Each subgroup leader is the only writer of its visit counter.
    visit_ptr = consumer_visits + pid * _KERNEL_SUBGROUPS + subgroup
    visits = gl.load(
        visit_ptr,
        mask=subgroup_leader,
        other=0,
        cache_modifier=".cv",
    )
    gl.store(
        visit_ptr,
        visits + 1,
        mask=subgroup_leader,
        cache_modifier=".wt",
    )


def _loaded_kernel_cache_identity(
    kernel,
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    """Identify the exact per-device runners retained by a Gluon JIT function."""

    return tuple(
        sorted(
            (
                repr(device),
                tuple(
                    sorted(
                        (repr(key), id(runner))
                        for key, runner in device_cache[0].items()
                    )
                ),
            )
            for device, device_cache in kernel.device_caches.items()
        )
    )


def _instruction_count(assembly: str, mnemonic: str) -> int:
    return len(
        re.findall(
            rf"^\s*{re.escape(mnemonic)}\b",
            assembly,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    )


def _assert_loaded_phase_topology_merge_isa(assembly: str) -> dict[str, object]:
    """Pin the loaded gfx950 message-passing and ticket instruction graph."""

    lowered = assembly.lower()
    lines = lowered.splitlines()
    isa_sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"phase-topology-merge-{isa_sha256}.amdgcn").write_text(
            assembly,
            encoding="utf-8",
        )

    add_indices = [
        index for index, line in enumerate(lines) if "global_atomic_add_x2" in line
    ]
    swap_indices = [
        index for index, line in enumerate(lines) if "global_atomic_swap_x2" in line
    ]
    assert len(add_indices) == 2
    assert len(swap_indices) == 1
    for index in add_indices:
        neighborhood = "\n".join(lines[max(0, index - 6) : index + 7])
        assert "buffer_wbl2" not in neighborhood
        assert "buffer_inv" not in neighborhood
        assert lines[index].rstrip().endswith("sc0")

    # The only swap is the merged gate release.  It must retain the compiler's
    # writeback and acknowledgement before the release RMW.
    release_index = swap_indices[0]
    before_release = "\n".join(lines[max(0, release_index - 12) : release_index])
    assert re.search(r"buffer_wbl2.*s_waitcnt[^\n]*vmcnt\(0\)", before_release, re.S)

    # Producer `.wt` -> explicit all-subgroup drain -> workgroup barrier ->
    # first relaxed ticket.  No cache-wide WBL belongs to this source-owned
    # publication prefix.
    producer = re.search(
        r"\bbuffer_store_dwordx2\b[^\n]*\bsc0 sc1\b"
        r"(?:(?!\bglobal_atomic_add_x2\b).)*?"
        r";;#asmstart\s+s_waitcnt vmcnt\(0\)\s+;;#asmend"
        r"(?:(?!\bglobal_atomic_add_x2\b).)*?\bs_barrier\b"
        r"(?:(?!\bs_barrier\b).)*?\bglobal_atomic_add_x2\b",
        lowered,
        flags=re.DOTALL,
    )
    assert producer is not None
    assert "buffer_wbl2" not in producer.group(0).split("s_barrier", 1)[0]

    # Consumer acquire invalidation -> explicit VMEM ACK -> workgroup barrier
    # -> coherent payload load.  This is the payload-bearing distinction from
    # the production topology-only relaxed poll.
    consumer = re.search(
        r"\bbuffer_inv\b"
        r"(?:(?!\bbuffer_inv\b|\bbuffer_load_dwordx2\b).)*?"
        r";;#asmstart\s+s_waitcnt vmcnt\(0\)\s+;;#asmend"
        r"(?:(?!\bbuffer_inv\b|\bbuffer_load_dwordx2\b).)*?\bs_barrier\b"
        r"(?:(?!\bbuffer_inv\b|\bbuffer_load_dwordx2\b).)*?"
        r"\bbuffer_load_dwordx2\b[^\n]*\bsc0 sc1\b",
        lowered,
        flags=re.DOTALL,
    )
    assert consumer is not None

    return {
        "isa_sha256": isa_sha256,
        "global_atomic_add_x2": len(add_indices),
        "global_atomic_swap_x2": len(swap_indices),
        "buffer_wbl2": _instruction_count(lowered, "buffer_wbl2"),
        "buffer_inv": _instruction_count(lowered, "buffer_inv"),
        "s_barrier": _instruction_count(lowered, "s_barrier"),
    }


def _simulate_hierarchical_arrival(
    xcc_by_pid: list[int],
    *,
    prior_generation: int,
    xcc_tickets: list[int],
    xcd_arrival: int,
) -> dict[str, object]:
    """Pure-host oracle for the exact cumulative hierarchy in the kernel."""

    if len(xcc_by_pid) != _PROGRAMS:
        raise ValueError(f"expected {_PROGRAMS} synthetic workgroups")
    if len(xcc_tickets) != _XCDS:
        raise ValueError(f"expected {_XCDS} cumulative XCC counters")
    generation = prior_generation + 1
    counters = list(xcc_tickets)
    local_ranks = [-1] * _PROGRAMS
    tickets = [-1] * _PROGRAMS
    xcd_tickets = [-1] * _PROGRAMS
    release_pids: list[int] = []
    gate = prior_generation

    for pid, xcc in enumerate(xcc_by_pid):
        if not 0 <= xcc < _XCDS:
            continue
        ticket = counters[xcc]
        counters[xcc] += 1
        tickets[pid] = ticket
        local_rank = ticket % _WORKERS_PER_XCD
        local_ranks[pid] = local_rank
        ticket_generation = ticket // _WORKERS_PER_XCD + 1
        if ticket_generation != generation or local_rank != _WORKERS_PER_XCD - 1:
            continue
        xcd_ticket = xcd_arrival
        xcd_arrival += 1
        xcd_tickets[pid] = xcd_ticket
        topology_generation = xcd_ticket // _XCDS + 1
        if xcd_ticket % _XCDS == _XCDS - 1 and topology_generation == generation:
            gate = generation
            release_pids.append(pid)

    return {
        "generation": generation,
        "gate": gate,
        "xcc_tickets": counters,
        "xcd_arrival": xcd_arrival,
        "tickets": tickets,
        "local_ranks": local_ranks,
        "xcd_tickets": xcd_tickets,
        "release_pids": release_pids,
    }


def _balanced_xccs(rotation: int = 0) -> list[int]:
    xccs = [xcc for xcc in range(_XCDS) for _ in range(_WORKERS_PER_XCD)]
    rotation %= _PROGRAMS
    return xccs[rotation:] + xccs[:rotation]


def _assert_exact_host_mapping(xccs: list[int], local_ranks: list[int]) -> None:
    communication_indices: list[int] = []
    for xcc in range(_XCDS):
        owned = sorted(
            rank for owner, rank in zip(xccs, local_ranks, strict=True) if owner == xcc
        )
        assert owned == list(range(_WORKERS_PER_XCD))
        communication_indices.extend(rank * _XCDS + xcc for rank in owned)
    assert sorted(communication_indices) == list(range(_PROGRAMS))


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _qualified_gpu0() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    return (
        architecture.startswith("gfx950")
        and int(getattr(properties, "multi_processor_count", 0)) == 256
    )


def _new_outputs(
    device: torch.device,
    rows: int,
) -> dict[str, torch.Tensor]:
    return {
        "tickets": torch.full((rows, _PROGRAMS), -1, dtype=torch.int64, device=device),
        "local_ranks": torch.full(
            (rows, _PROGRAMS), -1, dtype=torch.int32, device=device
        ),
        "xcd_tickets": torch.full(
            (rows, _PROGRAMS), -1, dtype=torch.int64, device=device
        ),
        "xccs": torch.full((rows, _PROGRAMS), -1, dtype=torch.int32, device=device),
        "generations": torch.full(
            (rows, _PROGRAMS), -1, dtype=torch.int64, device=device
        ),
        "releases": torch.zeros((rows, _PROGRAMS), dtype=torch.int32, device=device),
    }


def _new_diagnostics(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "failure_generation": torch.zeros(
            (_PROGRAMS, _LANES), dtype=torch.int64, device=device
        ),
        "failure_observed": torch.full(
            (_PROGRAMS, _LANES), -1, dtype=torch.int64, device=device
        ),
        "fatal_generation_by_pid": torch.zeros(
            _PROGRAMS, dtype=torch.int64, device=device
        ),
        "timeout_generation": torch.zeros(
            (_PROGRAMS, _SUBGROUPS), dtype=torch.int64, device=device
        ),
        "timeout_observed": torch.full(
            (_PROGRAMS, _SUBGROUPS), -1, dtype=torch.int64, device=device
        ),
        "timeout_marker": torch.zeros(
            (_PROGRAMS, _SUBGROUPS), dtype=torch.int32, device=device
        ),
    }


def _launch_litmus(
    *,
    payload: torch.Tensor,
    xcc_override: torch.Tensor,
    xcc_ticket: torch.Tensor,
    xcd_arrival: torch.Tensor,
    topology_gate: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    output_row: int,
    consumer_visits: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    timeout_ns: int,
    use_hardware_xcc: bool,
) -> None:
    _phase_topology_merge_litmus_kernel[(_PROGRAMS,)](
        payload,
        xcc_override,
        xcc_ticket,
        xcd_arrival,
        topology_gate,
        outputs["tickets"][output_row],
        outputs["local_ranks"][output_row],
        outputs["xcd_tickets"][output_row],
        outputs["xccs"][output_row],
        outputs["generations"][output_row],
        outputs["releases"][output_row],
        consumer_visits,
        diagnostics["failure_generation"],
        diagnostics["failure_observed"],
        diagnostics["fatal_generation_by_pid"],
        diagnostics["timeout_generation"],
        diagnostics["timeout_observed"],
        diagnostics["timeout_marker"],
        TIMEOUT_NS=timeout_ns,
        USE_HARDWARE_XCC=use_hardware_xcc,
        num_warps=_SUBGROUPS,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )


def _warm_litmus(
    *,
    payload: torch.Tensor,
    xcc_override: torch.Tensor,
    xcc_ticket: torch.Tensor,
    xcd_arrival: torch.Tensor,
    topology_gate: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    consumer_visits: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    timeout_ns: int,
    use_hardware_xcc: bool,
):
    return _phase_topology_merge_litmus_kernel.warmup(
        payload,
        xcc_override,
        xcc_ticket,
        xcd_arrival,
        topology_gate,
        outputs["tickets"][0],
        outputs["local_ranks"][0],
        outputs["xcd_tickets"][0],
        outputs["xccs"][0],
        outputs["generations"][0],
        outputs["releases"][0],
        consumer_visits,
        diagnostics["failure_generation"],
        diagnostics["failure_observed"],
        diagnostics["fatal_generation_by_pid"],
        diagnostics["timeout_generation"],
        diagnostics["timeout_observed"],
        diagnostics["timeout_marker"],
        TIMEOUT_NS=timeout_ns,
        USE_HARDWARE_XCC=use_hardware_xcc,
        grid=(_PROGRAMS,),
        num_warps=_SUBGROUPS,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )


def _assert_success_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    iterations: int,
) -> dict[str, object]:
    xccs = outputs["xccs"].cpu().to(torch.int64)
    local_ranks = outputs["local_ranks"].cpu().to(torch.int64)
    tickets = outputs["tickets"].cpu()
    xcd_tickets = outputs["xcd_tickets"].cpu()
    generations = outputs["generations"].cpu()
    releases = outputs["releases"].cpu()
    minimum_cross_xcd_edges = _PROGRAMS

    for index in range(iterations):
        generation = index + 1
        assert torch.all(generations[index] == generation)
        communication_indices: list[int] = []
        for xcc in range(_XCDS):
            mask = xccs[index] == xcc
            assert int(torch.count_nonzero(mask)) == _WORKERS_PER_XCD
            assert torch.equal(
                local_ranks[index][mask].sort().values,
                torch.arange(_WORKERS_PER_XCD, dtype=torch.int64),
            )
            expected_tickets = torch.arange(
                (generation - 1) * _WORKERS_PER_XCD,
                generation * _WORKERS_PER_XCD,
                dtype=torch.int64,
            )
            assert torch.equal(tickets[index][mask].sort().values, expected_tickets)
            communication_indices.extend(
                (local_ranks[index][mask] * _XCDS + xcc).tolist()
            )
        assert sorted(communication_indices) == list(range(_PROGRAMS))
        contributors = xcd_tickets[index][xcd_tickets[index] >= 0].sort().values
        assert torch.equal(
            contributors,
            torch.arange(
                (generation - 1) * _XCDS,
                generation * _XCDS,
                dtype=torch.int64,
            ),
        )
        assert int(releases[index].sum()) == 1
        sources = (torch.arange(_PROGRAMS) + _SOURCE_SHIFT) % _PROGRAMS
        cross_xcd = int(torch.count_nonzero(xccs[index] != xccs[index][sources]))
        minimum_cross_xcd_edges = min(minimum_cross_xcd_edges, cross_xcd)

    return {"minimum_cross_xcd_consumer_edges": minimum_cross_xcd_edges}


def _run_retained_hardware_success(
    device: torch.device,
    *,
    iterations: int,
    timeout_ns: int,
) -> dict[str, object]:
    payload = torch.full((_PROGRAMS, _LANES), -1, dtype=torch.int64, device=device)
    xcc_override = torch.zeros(_PROGRAMS, dtype=torch.int32, device=device)
    xcc_ticket = torch.zeros(_XCDS, dtype=torch.int64, device=device)
    xcd_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    topology_gate = torch.zeros(1, dtype=torch.int64, device=device)
    outputs = _new_outputs(device, iterations)
    consumer_visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS), dtype=torch.int32, device=device
    )
    diagnostics = _new_diagnostics(device)

    compiled = _warm_litmus(
        payload=payload,
        xcc_override=xcc_override,
        xcc_ticket=xcc_ticket,
        xcd_arrival=xcd_arrival,
        topology_gate=topology_gate,
        outputs=outputs,
        consumer_visits=consumer_visits,
        diagnostics=diagnostics,
        timeout_ns=timeout_ns,
        use_hardware_xcc=True,
    )
    isa = _assert_loaded_phase_topology_merge_isa(compiled.asm["amdgcn"])
    assert int(compiled.metadata.shared) == 128 * 1024
    hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    retained_identity = _loaded_kernel_cache_identity(
        _phase_topology_merge_litmus_kernel
    )
    assert retained_identity

    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with patch.object(
        _phase_topology_merge_litmus_kernel,
        "_do_compile",
        side_effect=AssertionError("retained phase/topology replay attempted to JIT"),
    ):
        for index in range(iterations):
            _launch_litmus(
                payload=payload,
                xcc_override=xcc_override,
                xcc_ticket=xcc_ticket,
                xcd_arrival=xcd_arrival,
                topology_gate=topology_gate,
                outputs=outputs,
                output_row=index,
                consumer_visits=consumer_visits,
                diagnostics=diagnostics,
                timeout_ns=timeout_ns,
                use_hardware_xcc=True,
            )
    assert retained_identity == _loaded_kernel_cache_identity(
        _phase_topology_merge_litmus_kernel
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    mapping = _assert_success_outputs(outputs, iterations=iterations)
    assert int(topology_gate.item()) == iterations
    assert int(xcd_arrival.item()) == iterations * _XCDS
    assert torch.equal(
        xcc_ticket.cpu(),
        torch.full((_XCDS,), iterations * _WORKERS_PER_XCD, dtype=torch.int64),
    )
    assert not torch.count_nonzero(diagnostics["failure_generation"])
    assert not torch.count_nonzero(diagnostics["fatal_generation_by_pid"])
    assert not torch.count_nonzero(diagnostics["timeout_marker"])
    assert torch.all(consumer_visits == iterations)
    assert mapping["minimum_cross_xcd_consumer_edges"] > 0
    return {
        "iterations": iterations,
        "elapsed_seconds": elapsed_seconds,
        "us_per_generation": elapsed_seconds * 1_000_000 / iterations,
        "gate": int(topology_gate.item()),
        "xcc_ticket": tuple(int(value) for value in xcc_ticket.cpu().tolist()),
        "xcd_arrival": int(xcd_arrival.item()),
        "minimum_cross_xcd_consumer_edges": mapping["minimum_cross_xcd_consumer_edges"],
        "isa": isa,
        "hsaco_sha256": hsaco_sha256,
    }


def _run_one_synthetic_case(
    device: torch.device,
    *,
    name: str,
    xccs: list[int],
    prior_generation: int,
    initial_xcc_tickets: list[int],
    initial_xcd_arrival: int,
    expect_success: bool,
    timeout_ns: int,
) -> dict[str, object]:
    payload = torch.full((_PROGRAMS, _LANES), -1, dtype=torch.int64, device=device)
    xcc_override = torch.tensor(xccs, dtype=torch.int32, device=device)
    sentinel_tickets = torch.full((_XCDS + 2,), -777, dtype=torch.int64, device=device)
    sentinel_tickets[1:-1] = torch.tensor(
        initial_xcc_tickets, dtype=torch.int64, device=device
    )
    xcc_ticket = sentinel_tickets[1:-1]
    xcd_arrival = torch.tensor([initial_xcd_arrival], dtype=torch.int64, device=device)
    topology_gate = torch.tensor([prior_generation], dtype=torch.int64, device=device)
    outputs = _new_outputs(device, 1)
    consumer_visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS), dtype=torch.int32, device=device
    )
    diagnostics = _new_diagnostics(device)
    _launch_litmus(
        payload=payload,
        xcc_override=xcc_override,
        xcc_ticket=xcc_ticket,
        xcd_arrival=xcd_arrival,
        topology_gate=topology_gate,
        outputs=outputs,
        output_row=0,
        consumer_visits=consumer_visits,
        diagnostics=diagnostics,
        timeout_ns=timeout_ns,
        use_hardware_xcc=False,
    )
    torch.cuda.synchronize(device)

    oracle = _simulate_hierarchical_arrival(
        xccs,
        prior_generation=prior_generation,
        xcc_tickets=initial_xcc_tickets,
        xcd_arrival=initial_xcd_arrival,
    )
    assert sentinel_tickets[0].item() == -777
    assert sentinel_tickets[-1].item() == -777
    assert tuple(xcc_ticket.cpu().tolist()) == tuple(oracle["xcc_tickets"])
    assert int(xcd_arrival.item()) == oracle["xcd_arrival"]
    assert int(topology_gate.item()) == oracle["gate"]
    observed_xccs = outputs["xccs"][0].cpu().to(torch.int64)
    assert torch.equal(observed_xccs, torch.tensor(xccs, dtype=torch.int64))
    observed_tickets = outputs["tickets"][0].cpu()
    observed_local_ranks = outputs["local_ranks"][0].cpu().to(torch.int64)
    for xcc in range(_XCDS):
        mask = observed_xccs == xcc
        population = int(torch.count_nonzero(mask))
        expected_tickets = torch.arange(
            initial_xcc_tickets[xcc],
            initial_xcc_tickets[xcc] + population,
            dtype=torch.int64,
        )
        assert torch.equal(observed_tickets[mask].sort().values, expected_tickets)
        assert torch.equal(
            observed_local_ranks[mask], observed_tickets[mask] % _WORKERS_PER_XCD
        )
    invalid_mask = (observed_xccs < 0) | (observed_xccs >= _XCDS)
    assert torch.all(observed_tickets[invalid_mask] == -1)
    assert torch.all(observed_local_ranks[invalid_mask] == -1)
    observed_xcd_tickets = outputs["xcd_tickets"][0].cpu()
    contributors = observed_xcd_tickets[observed_xcd_tickets >= 0].sort().values
    assert torch.equal(
        contributors,
        torch.arange(
            initial_xcd_arrival,
            int(oracle["xcd_arrival"]),
            dtype=torch.int64,
        ),
    )
    assert int(outputs["releases"].sum()) == len(oracle["release_pids"])
    generation = prior_generation + 1
    assert torch.all(outputs["generations"] == generation)

    if expect_success:
        assert int(topology_gate.item()) == generation
        assert not torch.count_nonzero(diagnostics["failure_generation"])
        assert not torch.count_nonzero(diagnostics["fatal_generation_by_pid"])
        assert not torch.count_nonzero(diagnostics["timeout_marker"])
        assert torch.all(consumer_visits == 1)
    else:
        assert int(topology_gate.item()) == prior_generation
        assert torch.all(diagnostics["fatal_generation_by_pid"] == generation)
        assert torch.all(diagnostics["timeout_generation"] == generation)
        assert torch.all(diagnostics["timeout_observed"] == prior_generation)
        assert torch.all(diagnostics["timeout_marker"] == 1)
        assert not torch.count_nonzero(consumer_visits)
    return {
        "name": name,
        "generation": generation,
        "gate": int(topology_gate.item()),
        "xcc_ticket": tuple(int(value) for value in xcc_ticket.cpu().tolist()),
        "xcd_arrival": int(xcd_arrival.item()),
        "release_count": int(outputs["releases"].sum()),
        "fatal_diagnostics_uniform": bool(
            torch.all(diagnostics["fatal_generation_by_pid"] == generation)
        ),
        "timeout_diagnostics_uniform": bool(
            torch.all(diagnostics["timeout_generation"] == generation)
            and torch.all(diagnostics["timeout_observed"] == prior_generation)
        ),
    }


def _run_synthetic_matrix(device: torch.device, *, timeout_ns: int) -> list[dict]:
    balanced = _balanced_xccs()
    outputs = _new_outputs(device, 1)
    diagnostics = _new_diagnostics(device)
    payload = torch.full((_PROGRAMS, _LANES), -1, dtype=torch.int64, device=device)
    xcc_override = torch.tensor(balanced, dtype=torch.int32, device=device)
    # Match the sentinel-guarded failure cases' exact pointer alignment so the
    # no-JIT assertion covers their retained runner rather than a distinct
    # alignment specialization.
    sentinel_tickets = torch.zeros(_XCDS + 2, dtype=torch.int64, device=device)
    xcc_ticket = sentinel_tickets[1:-1]
    xcd_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    topology_gate = torch.zeros(1, dtype=torch.int64, device=device)
    consumer_visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS), dtype=torch.int32, device=device
    )
    compiled = _warm_litmus(
        payload=payload,
        xcc_override=xcc_override,
        xcc_ticket=xcc_ticket,
        xcd_arrival=xcd_arrival,
        topology_gate=topology_gate,
        outputs=outputs,
        consumer_visits=consumer_visits,
        diagnostics=diagnostics,
        timeout_ns=timeout_ns,
        use_hardware_xcc=False,
    )
    _assert_loaded_phase_topology_merge_isa(compiled.asm["amdgcn"])
    assert int(compiled.metadata.shared) == 128 * 1024

    unbalanced = balanced.copy()
    unbalanced[_WORKERS_PER_XCD] = 0
    invalid_xcc = balanced.copy()
    invalid_xcc[0] = 8
    prior = 7
    exact_xcc = [prior * _WORKERS_PER_XCD] * _XCDS
    exact_xcd = prior * _XCDS
    cases = (
        {
            "name": "balanced_30x8",
            "xccs": balanced,
            "prior_generation": 0,
            "initial_xcc_tickets": [0] * _XCDS,
            "initial_xcd_arrival": 0,
            "expect_success": True,
        },
        {
            "name": "synthetic_31_29",
            "xccs": unbalanced,
            "prior_generation": 0,
            "initial_xcc_tickets": [0] * _XCDS,
            "initial_xcd_arrival": 0,
            "expect_success": False,
        },
        {
            "name": "invalid_xcc_8",
            "xccs": invalid_xcc,
            "prior_generation": 0,
            "initial_xcc_tickets": [0] * _XCDS,
            "initial_xcd_arrival": 0,
            "expect_success": False,
        },
        {
            "name": "stale_xcc_counters",
            "xccs": balanced,
            "prior_generation": prior,
            "initial_xcc_tickets": [(prior - 1) * _WORKERS_PER_XCD] * _XCDS,
            "initial_xcd_arrival": exact_xcd,
            "expect_success": False,
        },
        {
            "name": "ahead_xcc_counters",
            "xccs": balanced,
            "prior_generation": prior,
            "initial_xcc_tickets": [(prior + 1) * _WORKERS_PER_XCD] * _XCDS,
            "initial_xcd_arrival": exact_xcd,
            "expect_success": False,
        },
        {
            "name": "stale_xcd_counter",
            "xccs": balanced,
            "prior_generation": prior,
            "initial_xcc_tickets": exact_xcc,
            "initial_xcd_arrival": (prior - 1) * _XCDS,
            "expect_success": False,
        },
        {
            "name": "ahead_xcd_counter",
            "xccs": balanced,
            "prior_generation": prior,
            "initial_xcc_tickets": exact_xcc,
            "initial_xcd_arrival": (prior + 1) * _XCDS,
            "expect_success": False,
        },
    )
    retained_identity = _loaded_kernel_cache_identity(
        _phase_topology_merge_litmus_kernel
    )
    assert retained_identity
    with patch.object(
        _phase_topology_merge_litmus_kernel,
        "_do_compile",
        side_effect=AssertionError("synthetic phase/topology case attempted to JIT"),
    ):
        reports = [
            _run_one_synthetic_case(device, timeout_ns=timeout_ns, **case)
            for case in cases
        ]
    assert retained_identity == _loaded_kernel_cache_identity(
        _phase_topology_merge_litmus_kernel
    )
    return reports


def _worker_main(_worker_index: int, iterations: int, timeout_ns: int) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    report = {
        "retained_hardware": _run_retained_hardware_success(
            device,
            iterations=iterations,
            timeout_ns=timeout_ns,
        ),
        "synthetic_matrix": _run_synthetic_matrix(device, timeout_ns=timeout_ns),
    }
    print("MegaMoE merged phase/topology report:", json.dumps(report), flush=True)


def _terminate_processes(context) -> None:
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5)
    for process in context.processes:
        if process.is_alive():
            process.kill()
    for process in context.processes:
        process.join(timeout=5)


def test_amd_phase_topology_merge_source_protocol() -> None:
    """Pin the test-only publication chain before hardware authorization."""

    kernel = inspect.getsource(_phase_topology_merge_litmus_kernel.fn)
    arrival = inspect.getsource(_arrive_hierarchical_phase.fn)
    module = Path(__file__).read_text(encoding="utf-8")

    generation_load = kernel.index(
        'prior_generation = gl.load(topology_gate, cache_modifier=".cv")'
    )
    generation_increment = kernel.index(
        "generation = prior_generation + 1", generation_load
    )
    payload_store = kernel.index("gl.amd.cdna4.buffer_store(", generation_increment)
    producer_drain = kernel.index("_wait_subgroup_vmem_ack()", payload_store)
    producer_barrier = kernel.index("gl.barrier()", producer_drain)
    hierarchy = kernel.index("_arrive_hierarchical_phase(", producer_barrier)
    acquire_poll = kernel.index("ready = gl.atomic_poll(", hierarchy)
    consumer_drain = kernel.index("_wait_subgroup_vmem_ack()", acquire_poll)
    consumer_barrier = kernel.index("gl.barrier()", consumer_drain)
    timeout_branch = kernel.index("if not ready:", consumer_barrier)
    timeout_return = kernel.index("return", timeout_branch)
    payload_load = kernel.index("gl.amd.cdna4.buffer_load(", timeout_return)
    assert (
        generation_load
        < generation_increment
        < payload_store
        < producer_drain
        < producer_barrier
        < hierarchy
        < acquire_poll
        < consumer_drain
        < consumer_barrier
        < timeout_branch
        < timeout_return
        < payload_load
    )
    assert 'cache=".wt"' in kernel[payload_store:producer_drain]
    assert 'sem="acquire"' in kernel[acquire_poll:consumer_drain]
    assert 'cache=".cv"' in kernel[payload_load:]
    assert "[_LDS_BYTES]" in kernel
    assert '"s_sleep 4"' in inspect.getsource(_deliberate_workgroup_skew.fn)

    assert arrival.count("gl.atomic_add(") == 2
    assert arrival.count('sem="relaxed"') == 2
    assert arrival.count("gl.atomic_xchg(") == 1
    assert arrival.count('sem="release"') == 1
    assert "ticket_generation == generation" in arrival
    assert "local_rank == _KERNEL_WORKERS_PER_XCD - 1" in arrival
    assert "topology_generation == generation" in arrival
    assert "xcd_ticket % _KERNEL_XCDS == _KERNEL_XCDS - 1" in arrival
    assert "if (xcc >= 0) & (xcc < _KERNEL_XCDS):" in arrival

    assert (
        re.search(
            r"^from tokenspeed_kernel_amd\.ops\.gfx950\.moe\.megamoe\.kernel import",
            module,
            flags=re.MULTILINE,
        )
        is None
    )
    assert (
        re.search(
            r"^from tokenspeed_kernel_amd\.ops\.gfx950\.moe\.megamoe\.admission import",
            module,
            flags=re.MULTILINE,
        )
        is None
    )


def test_amd_phase_topology_merge_host_counter_oracle() -> None:
    """Prove retained success and fail-closed cumulative counter states."""

    xcc_tickets = [0] * _XCDS
    xcd_arrival = 0
    gate = 0
    # This is the same cumulative range exercised by the eventual retained
    # 4096-dispatch hardware runner, without compiling or touching a GPU.
    for generation in range(1, 4097):
        xccs = _balanced_xccs(rotation=(generation * 17) % _PROGRAMS)
        result = _simulate_hierarchical_arrival(
            xccs,
            prior_generation=gate,
            xcc_tickets=xcc_tickets,
            xcd_arrival=xcd_arrival,
        )
        assert result["generation"] == generation
        assert result["gate"] == generation
        assert len(result["release_pids"]) == 1
        assert sorted(
            ticket for ticket in result["xcd_tickets"] if ticket >= 0
        ) == list(range((generation - 1) * _XCDS, generation * _XCDS))
        _assert_exact_host_mapping(xccs, result["local_ranks"])
        xcc_tickets = result["xcc_tickets"]
        xcd_arrival = result["xcd_arrival"]
        gate = result["gate"]
    assert xcc_tickets == [4096 * _WORKERS_PER_XCD] * _XCDS
    assert xcd_arrival == 4096 * _XCDS

    balanced = _balanced_xccs()
    unbalanced = balanced.copy()
    unbalanced[_WORKERS_PER_XCD] = 0
    invalid_xcc = balanced.copy()
    invalid_xcc[0] = 8
    prior = 7
    exact_xcc = [prior * _WORKERS_PER_XCD] * _XCDS
    exact_xcd = prior * _XCDS
    failures = (
        (unbalanced, 0, [0] * _XCDS, 0),
        (invalid_xcc, 0, [0] * _XCDS, 0),
        (
            balanced,
            prior,
            [(prior - 1) * _WORKERS_PER_XCD] * _XCDS,
            exact_xcd,
        ),
        (
            balanced,
            prior,
            [(prior + 1) * _WORKERS_PER_XCD] * _XCDS,
            exact_xcd,
        ),
        (balanced, prior, exact_xcc, (prior - 1) * _XCDS),
        (balanced, prior, exact_xcc, (prior + 1) * _XCDS),
    )
    for xccs, prior_generation, counters, arrival in failures:
        result = _simulate_hierarchical_arrival(
            xccs,
            prior_generation=prior_generation,
            xcc_tickets=counters,
            xcd_arrival=arrival,
        )
        assert result["gate"] == prior_generation
        assert not result["release_pids"]

    malformed = _simulate_hierarchical_arrival(
        unbalanced,
        prior_generation=0,
        xcc_tickets=[0] * _XCDS,
        xcd_arrival=0,
    )
    assert malformed["xcc_tickets"] == [31, 29, 30, 30, 30, 30, 30, 30]
    assert malformed["xcd_arrival"] == 7
    invalid = _simulate_hierarchical_arrival(
        invalid_xcc,
        prior_generation=0,
        xcc_tickets=[0] * _XCDS,
        xcd_arrival=0,
    )
    assert invalid["xcc_tickets"] == [29, 30, 30, 30, 30, 30, 30, 30]
    assert invalid["tickets"][0] == -1
    assert invalid["local_ranks"][0] == -1
    assert invalid["xcd_arrival"] == 7


def test_amd_phase_topology_merge_publication_hardware() -> None:
    """Run the 4096-generation retained binary and negative matrix on gfx950."""

    if not _env_enabled(_GATE_ENV):
        pytest.skip(f"set {_GATE_ENV}=1 after reserving and authorizing GPU 0")
    if not _qualified_gpu0():
        pytest.skip("the merged phase/topology litmus requires a 256-CU gfx950")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the merged phase/topology litmus requires system ROCr")

    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "300"))
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if timeout_ns <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")

    context = mp.spawn(
        _worker_main,
        args=(iterations, timeout_ns),
        nprocs=1,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    "gfx950 merged phase/topology litmus exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise
