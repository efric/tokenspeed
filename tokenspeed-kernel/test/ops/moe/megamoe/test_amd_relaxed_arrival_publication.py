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

"""Opt-in gfx950 multi-workgroup relaxed-arrival publication litmus.

This test isolates the local GPU-scope arrival publication primitive used by
MegaMoE's shared-down, W13, W2, and communication gates. The standalone
240-workgroup shape also models the former flat phase gate; production now
publishes phase payload through the separate hierarchical topology tickets. All
240 cooperative
workgroups write disjoint, generation-reused payload slices with all eight
subgroups, drain VMEM, and converge before contributing an exact arrival
ticket.  The last ticket releases one gate; every subgroup then completes an
acquire poll and invalidate acknowledgement before coherently loading a
different producer's payload slice.

Three otherwise identical specializations are compared:

* scalar ``acq_rel`` arrival, matching the current production protocol;
* scalar ``relaxed`` arrival; and
* globally lane-0-masked tensor ``relaxed`` arrival and release, avoiding the
  scalar ticket-result LDS broadcast when lowering permits it.

The language-level transitive release chain is intentionally absent from the
relaxed variants.  This is a gfx950 hardware litmus for the narrower contract
formed by write-through payload stores, per-subgroup VMEM completion, exact
atomic modification order, the last workgroup's release, and the consumer's
acquire/invalidate.  It is not portable evidence for weakening arbitrary
atomics.

Each generation is a fresh launch of one retained binary.  A 128-KiB LDS
allocation constrains residency to one workgroup per compute unit, and the
test records all eight hardware XCC IDs for every generation.  Generation-
dependent scalar sleeps deliberately perturb arrival order.  The consumer
permutation is bijective and normally crosses XCDs.

Run on one reserved 256-CU gfx950 with the system ROCr runtime::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/\
test_amd_relaxed_arrival_publication.py

The default is 4096 retained generations for each of the three variants.
Production kernel sources are not imported or modified by this litmus.
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

_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL_WATCHDOG_SECONDS"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL_DUMP_AMDGCN"
_VARIANTS_ENV = "TOKENSPEED_TEST_MEGAMOE_RELAXED_ARRIVAL_VARIANTS"

_PROGRAMS = 240
_XCDS = 8
_SUBGROUPS = 8
_LANES = 512
_SOURCE_SHIFT = 113
_LDS_BYTES = gl.constexpr(128 * 1024)
_KERNEL_PROGRAMS = gl.constexpr(_PROGRAMS)
_KERNEL_SUBGROUPS = gl.constexpr(_SUBGROUPS)
_KERNEL_LANES = gl.constexpr(_LANES)
_KERNEL_SOURCE_SHIFT = gl.constexpr(_SOURCE_SHIFT)

_SCALAR_ACQ_REL = 0
_SCALAR_RELAXED = 1
_LANE0_RELAXED = 2
_VARIANTS = (
    ("scalar_acq_rel", _SCALAR_ACQ_REL),
    ("scalar_relaxed", _SCALAR_RELAXED),
    ("lane0_relaxed", _LANE0_RELAXED),
)


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
    """Perturb the arrival order without adding memory transactions."""

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
def _arrive_scalar_acq_rel(arrival, gate, generation):
    """Current scalar publication: fenced arrival plus last release."""

    target = generation * _KERNEL_PROGRAMS
    ticket = gl.atomic_add(arrival, 1, sem="acq_rel", scope="gpu")
    if ticket + 1 == target:
        gl.atomic_xchg(gate, generation, sem="release", scope="gpu")
    return ticket


@gluon.jit
def _arrive_scalar_relaxed(arrival, gate, generation):
    """Fence-free scalar arrival while retaining the last release."""

    target = generation * _KERNEL_PROGRAMS
    ticket = gl.atomic_add(arrival, 1, sem="relaxed", scope="gpu")
    if ticket + 1 == target:
        gl.atomic_xchg(gate, generation, sem="release", scope="gpu")
    return ticket


@gluon.jit
def _arrive_lane0_relaxed(arrival, gate, generation, lane):
    """Use only global lane 0 and keep the ticket result in its tensor lane."""

    lane0 = lane == 0
    target = generation * _KERNEL_PROGRAMS
    ticket = gl.atomic_add(
        arrival + lane * 0,
        1,
        mask=lane0,
        sem="relaxed",
        scope="gpu",
    )
    gl.atomic_xchg(
        gate + lane * 0,
        generation,
        mask=lane0 & (ticket + 1 == target),
        sem="release",
        scope="gpu",
    )
    return ticket


@gluon.jit
def _arrival_publication_litmus_kernel(
    payload,
    arrival,
    gate,
    generation_ptr,
    ticket_by_generation,
    xcc_by_generation,
    consumer_visits,
    failure_generation,
    failure_observed,
    control_failures,
    TIMEOUT_NS: gl.constexpr,
    ARRIVAL_MODE: gl.constexpr,
):
    """Publish and consume one exact generation in a fresh dispatch."""

    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [_KERNEL_SUBGROUPS], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup = lane // 64
    subgroup_leader = lane % 64 == 0
    generation = gl.load(generation_ptr, cache_modifier=".cv")

    # Match the persistent kernel's one-workgroup-per-compute-unit residency.
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    residency = gl.allocate_shared_memory(gl.uint8, [_LDS_BYTES], shared_layout)
    residency_tile = residency.slice(0, _KERNEL_LANES)
    residency_tile.store((lane % 251).to(gl.uint8))
    gl.barrier()
    residency_token = residency_tile.load(layout).to(gl.int64)

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
    generation_pid = (generation - 1) * _KERNEL_PROGRAMS + pid
    gl.store(
        xcc_by_generation + generation_pid + lane * 0,
        xcc,
        mask=lane == 0,
        cache_modifier=".wt",
    )

    _deliberate_workgroup_skew(pid, generation)
    payload_index = pid * _KERNEL_LANES + lane
    expected = generation * 1_000_000_000 + payload_index.to(gl.int64) + residency_token
    gl.amd.cdna4.buffer_store(
        expected,
        payload,
        payload_index.to(gl.int32),
        cache=".wt",
    )

    # All eight producer subgroups drain before one arrival can represent the
    # workgroup.  There is deliberately no cache-wide WBL in this source block.
    _wait_subgroup_vmem_ack()
    gl.barrier()
    if ARRIVAL_MODE == 0:
        ticket = _arrive_scalar_acq_rel(arrival, gate, generation)
    elif ARRIVAL_MODE == 1:
        ticket = _arrive_scalar_relaxed(arrival, gate, generation)
    else:
        ticket = _arrive_lane0_relaxed(arrival, gate, generation, lane)
    gl.store(
        ticket_by_generation + generation_pid + lane * 0,
        ticket,
        mask=lane == 0,
        cache_modifier=".wt",
    )

    ready = gl.atomic_poll(
        gate,
        generation,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    # BUFFER_INV acknowledges through VMCNT; a workgroup barrier alone does
    # not complete it.  Every subgroup waits before converging on payload use.
    _wait_subgroup_vmem_ack()
    gl.barrier()

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
    gl.store(
        control_failures + pid * _KERNEL_SUBGROUPS + subgroup,
        generation,
        mask=subgroup_leader & (ready == 0),
        cache_modifier=".wt",
    )

    # Each subgroup owns one counter, so this non-atomic load/store proves all
    # eight consuming subgroups completed every retained generation.
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


def _assert_loaded_arrival_isa(
    assembly: str,
    *,
    name: str,
    mode: int,
) -> dict[str, object]:
    """Prove the exact loaded publication graph for one specialization."""

    lowered = assembly.lower()
    lines = lowered.splitlines()
    sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}-{sha256}.amdgcn").write_text(assembly, encoding="utf-8")

    add_indices = [
        index for index, line in enumerate(lines) if "global_atomic_add_x2" in line
    ]
    swap_indices = [
        index for index, line in enumerate(lines) if "global_atomic_swap_x2" in line
    ]
    assert len(add_indices) == 1
    assert len(swap_indices) == 1
    arrival_index = add_indices[0]
    release_index = swap_indices[0]

    arrival_neighborhood = "\n".join(
        lines[max(0, arrival_index - 6) : arrival_index + 7]
    )
    if mode == _SCALAR_ACQ_REL:
        assert "buffer_wbl2" in arrival_neighborhood
        assert "buffer_inv" in arrival_neighborhood
    else:
        assert "buffer_wbl2" not in arrival_neighborhood
        assert "buffer_inv" not in arrival_neighborhood

    # The sole swap is the last-arrival release and must retain the compiler's
    # WBL plus its VMEM acknowledgement before the atomic instruction.
    before_release = "\n".join(lines[max(0, release_index - 10) : release_index])
    assert re.search(r"buffer_wbl2.*s_waitcnt[^\n]*vmcnt\(0\)", before_release, re.S)

    # Producer `.wt` -> every-subgroup VMEM drain -> workgroup barrier -> RMW.
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
    source_publication = producer.group(0).split("s_barrier", 1)[0]
    assert "buffer_wbl2" not in source_publication

    # Consumer acquire INV -> explicit ACK -> convergent barrier -> `.cv` load.
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

    poll_start = next(
        index
        for index in range(arrival_index + 1, len(lines))
        if "s_memrealtime" in lines[index]
    )
    arrival_to_poll = "\n".join(lines[arrival_index + 1 : poll_start])
    scalar_ticket_broadcast = re.search(
        r"\bds_write_b64\b.*?\bs_barrier\b.*?\bds_read_b64\b",
        arrival_to_poll,
        flags=re.DOTALL,
    )
    arrival_to_poll_barriers = _instruction_count(arrival_to_poll, "s_barrier")
    if mode == _LANE0_RELAXED:
        assert scalar_ticket_broadcast is None
        # The generic masked-atomic lowering does not elect the known-unique
        # lane directly.  It compacts an arbitrary tensor mask with DPP/DS
        # shuffles and one rendezvous before issuing the release.  Retain this
        # negative result rather than treating the absent b64 broadcast as a
        # clean barrier elimination.
        assert arrival_to_poll_barriers == 1
        assert "v_add_u32_dpp" in arrival_to_poll
        assert "ds_permute_b32" in arrival_to_poll
        assert "ds_bpermute_b32" in arrival_to_poll
    else:
        assert scalar_ticket_broadcast is not None
        assert scalar_ticket_broadcast.group(0).count("s_barrier") == 1

    return {
        "isa_sha256": sha256,
        "global_atomic_add_x2": len(add_indices),
        "global_atomic_swap_x2": len(swap_indices),
        "buffer_wbl2": _instruction_count(lowered, "buffer_wbl2"),
        "buffer_inv": _instruction_count(lowered, "buffer_inv"),
        "s_barrier": _instruction_count(lowered, "s_barrier"),
        "ds_write_b64": _instruction_count(lowered, "ds_write_b64"),
        "ds_read_b64": _instruction_count(lowered, "ds_read_b64"),
        "arrival_result_rendezvous": int(scalar_ticket_broadcast is not None),
        "arrival_to_poll_barriers": arrival_to_poll_barriers,
        "masked_tensor_compaction": mode == _LANE0_RELAXED,
    }


def _assert_isa_comparison(reports: tuple[dict[str, object], ...]) -> None:
    """Require the intended fence and rendezvous deltas, not only validity."""

    by_name = {str(report["name"]): report for report in reports}
    baseline = by_name["scalar_acq_rel"]["isa"]
    scalar = by_name["scalar_relaxed"]["isa"]
    lane0_report = by_name.get("lane0_relaxed")
    lane0 = lane0_report["isa"] if lane0_report is not None else None
    assert isinstance(baseline, dict)
    assert isinstance(scalar, dict)
    assert lane0 is None or isinstance(lane0, dict)

    assert baseline["buffer_wbl2"] == scalar["buffer_wbl2"] + 1
    assert baseline["buffer_inv"] == scalar["buffer_inv"] + 1
    assert baseline["s_barrier"] == scalar["s_barrier"] + 1
    assert baseline["arrival_result_rendezvous"] == 1
    assert scalar["arrival_result_rendezvous"] == 1
    if lane0 is not None:
        assert scalar["buffer_wbl2"] == lane0["buffer_wbl2"]
        assert scalar["buffer_inv"] == lane0["buffer_inv"]
        assert lane0["arrival_result_rendezvous"] == 0
        assert lane0["arrival_to_poll_barriers"] == 1
        assert lane0["masked_tensor_compaction"]
        assert scalar["s_barrier"] == lane0["s_barrier"] + 1


def _production_arrival_counts(local_routes: int) -> dict[str, int]:
    """Return dynamic arrival RMW counts for one successful MegaMoE layer."""

    if not 0 <= local_routes <= 16:
        raise ValueError("local route count must be in [0, 16]")
    # The fused production path has one 226-workgroup W13 rendezvous and one
    # 226-workgroup output-centric W2/combine rendezvous for every nonempty
    # local route set. L=0 bypasses both expert-grid gates entirely. Phase-zero
    # payload uses the already-relaxed hierarchical topology control chain, not
    # a separate flat payload arrival.
    expert_workers = _PROGRAMS - 14 if local_routes > 0 else 0
    return {
        "phase": 0,
        "shared": 224,
        "w13": expert_workers,
        "w2": expert_workers,
        "comm": 21,
    }


def _run_variant(
    device: torch.device,
    *,
    name: str,
    mode: int,
    iterations: int,
    timeout_ns: int,
) -> dict[str, object]:
    payload = torch.full(
        (_PROGRAMS, _LANES),
        -1,
        dtype=torch.int64,
        device=device,
    )
    arrival = torch.zeros(1, dtype=torch.int64, device=device)
    gate = torch.zeros(1, dtype=torch.int64, device=device)
    ticket_by_generation = torch.full(
        (iterations, _PROGRAMS),
        -1,
        dtype=torch.int64,
        device=device,
    )
    xcc_by_generation = torch.full(
        (iterations, _PROGRAMS),
        -1,
        dtype=torch.int32,
        device=device,
    )
    consumer_visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS),
        dtype=torch.int32,
        device=device,
    )
    failure_generation = torch.zeros(
        (_PROGRAMS, _LANES),
        dtype=torch.int64,
        device=device,
    )
    failure_observed = torch.full_like(failure_generation, -1)
    control_failures = torch.zeros(
        (_PROGRAMS, _SUBGROUPS),
        dtype=torch.int64,
        device=device,
    )
    generation_index = torch.arange(
        1,
        iterations + 1,
        dtype=torch.int64,
        device=device,
    )
    generation_values = torch.stack(
        (generation_index, torch.zeros_like(generation_index)),
        dim=1,
    )
    arguments = (
        payload,
        arrival,
        gate,
        generation_values[0],
        ticket_by_generation,
        xcc_by_generation,
        consumer_visits,
        failure_generation,
        failure_observed,
        control_failures,
    )
    compiled = _arrival_publication_litmus_kernel.warmup(
        *arguments,
        TIMEOUT_NS=timeout_ns,
        ARRIVAL_MODE=mode,
        grid=(_PROGRAMS,),
        num_warps=_SUBGROUPS,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )
    isa = _assert_loaded_arrival_isa(compiled.asm["amdgcn"], name=name, mode=mode)
    hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    retained_cache_identity = _loaded_kernel_cache_identity(
        _arrival_publication_litmus_kernel
    )
    assert retained_cache_identity
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with patch.object(
        _arrival_publication_litmus_kernel,
        "_do_compile",
        side_effect=AssertionError("generation replay attempted to recompile"),
    ):
        for generation_index_value in range(iterations):
            _arrival_publication_litmus_kernel[(_PROGRAMS,)](
                payload,
                arrival,
                gate,
                generation_values[generation_index_value],
                ticket_by_generation,
                xcc_by_generation,
                consumer_visits,
                failure_generation,
                failure_observed,
                control_failures,
                TIMEOUT_NS=timeout_ns,
                ARRIVAL_MODE=mode,
                num_warps=_SUBGROUPS,
                num_stages=1,
                waves_per_eu=2,
                launch_cooperative_grid=True,
            )
    assert retained_cache_identity == _loaded_kernel_cache_identity(
        _arrival_publication_litmus_kernel
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    tickets = ticket_by_generation.cpu()
    expected_tickets = torch.arange(
        iterations * _PROGRAMS,
        dtype=torch.int64,
    ).reshape(iterations, _PROGRAMS)
    exact_tickets = torch.equal(tickets.sort(dim=1).values, expected_tickets)

    xccs = xcc_by_generation.cpu().to(torch.int64)
    populations = torch.stack(
        [torch.bincount(row, minlength=_XCDS)[:_XCDS] for row in xccs]
    )
    exact_xcd_population = bool(torch.all(populations == 30))
    sources = (torch.arange(_PROGRAMS) + _SOURCE_SHIFT) % _PROGRAMS
    cross_xcd_edges = torch.count_nonzero(xccs != xccs[:, sources], dim=1)

    failures = failure_generation.cpu()
    bad = torch.nonzero(failures != 0)
    observed = failure_observed.cpu()
    first_failures = tuple(
        {
            "consumer_program": int(program),
            "lane": int(lane),
            "generation": int(failures[program, lane]),
            "observed": int(observed[program, lane]),
        }
        for program, lane in bad[:16].tolist()
    )
    visits = consumer_visits.cpu()
    controls = control_failures.cpu()
    return {
        "name": name,
        "iterations": iterations,
        "retained_launch_elapsed_seconds": elapsed_seconds,
        "retained_launch_us_per_generation": (elapsed_seconds * 1_000_000 / iterations),
        "payload_errors": int(torch.count_nonzero(failures)),
        "bad_words_by_consumer_subgroup": tuple(
            int(torch.count_nonzero(failures[:, subgroup * 64 : (subgroup + 1) * 64]))
            for subgroup in range(_SUBGROUPS)
        ),
        "first_failures": first_failures,
        "control_failures": int(torch.count_nonzero(controls)),
        "control_failures_by_subgroup": tuple(
            int(torch.count_nonzero(controls[:, subgroup]))
            for subgroup in range(_SUBGROUPS)
        ),
        "all_consumer_subgroup_visits": bool(torch.all(visits == iterations)),
        "arrival": int(arrival.item()),
        "gate": int(gate.item()),
        "exact_tickets": exact_tickets,
        "exact_xcd_population_every_generation": exact_xcd_population,
        "minimum_cross_xcd_consumer_edges": int(cross_xcd_edges.min()),
        "maximum_cross_xcd_consumer_edges": int(cross_xcd_edges.max()),
        "isa": isa,
        "hsaco_sha256": hsaco_sha256,
    }


def _worker_main(
    _worker_index: int,
    iterations: int,
    timeout_ns: int,
    variants: tuple[tuple[str, int], ...],
) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    reports = tuple(
        _run_variant(
            device,
            name=name,
            mode=mode,
            iterations=iterations,
            timeout_ns=timeout_ns,
        )
        for name, mode in variants
    )
    _assert_isa_comparison(reports)
    for report in reports:
        assert report["payload_errors"] == 0, report
        assert report["control_failures"] == 0, report
        assert report["all_consumer_subgroup_visits"], report
        assert report["arrival"] == iterations * _PROGRAMS, report
        assert report["gate"] == iterations, report
        assert report["exact_tickets"], report
        assert report["exact_xcd_population_every_generation"], report
        assert report["minimum_cross_xcd_consumer_edges"] > 0, report
    print(
        "MegaMoE relaxed-arrival publication reports:",
        json.dumps(reports),
        flush=True,
    )
    savings = {
        routes: {
            "arrivals": counts,
            "scalar_relaxed_wbl_savings": sum(counts.values()),
            "scalar_relaxed_inv_savings": sum(counts.values()),
            "scalar_relaxed_barrier_savings": sum(counts.values()),
            "lane0_clean_result_rendezvous_savings": 0,
            "lane0_rejected_mask_compaction_barriers": sum(counts.values()),
        }
        for routes in (0, 1, 8, 16)
        for counts in (_production_arrival_counts(routes),)
    }
    print("MegaMoE dynamic arrival savings:", json.dumps(savings), flush=True)


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


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def test_amd_relaxed_arrival_source_protocol() -> None:
    """Pin the test-only protocol before any opt-in hardware execution."""

    kernel = inspect.getsource(_arrival_publication_litmus_kernel.fn)
    acq_rel = inspect.getsource(_arrive_scalar_acq_rel.fn)
    relaxed = inspect.getsource(_arrive_scalar_relaxed.fn)
    lane0 = inspect.getsource(_arrive_lane0_relaxed.fn)

    payload_store = kernel.index("gl.amd.cdna4.buffer_store(")
    producer_drain = kernel.index("_wait_subgroup_vmem_ack()", payload_store)
    producer_barrier = kernel.index("gl.barrier()", producer_drain)
    arrival_dispatch = kernel.index("if ARRIVAL_MODE", producer_barrier)
    acquire_poll = kernel.index("ready = gl.atomic_poll(", arrival_dispatch)
    consumer_drain = kernel.index("_wait_subgroup_vmem_ack()", acquire_poll)
    consumer_barrier = kernel.index("gl.barrier()", consumer_drain)
    payload_load = kernel.index(
        "observed = gl.amd.cdna4.buffer_load(", consumer_barrier
    )
    assert (
        payload_store
        < producer_drain
        < producer_barrier
        < arrival_dispatch
        < acquire_poll
        < consumer_drain
        < consumer_barrier
        < payload_load
    )
    assert 'cache=".wt"' in kernel[payload_store:producer_drain]
    assert 'sem="acquire"' in kernel[acquire_poll:consumer_drain]
    assert 'cache=".cv"' in kernel[payload_load:]
    assert '"s_sleep 4"' in inspect.getsource(_deliberate_workgroup_skew.fn)

    assert 'sem="acq_rel"' in acq_rel
    assert 'sem="relaxed"' not in acq_rel
    assert 'sem="relaxed"' in relaxed
    assert 'sem="acq_rel"' not in relaxed
    for source in (acq_rel, relaxed, lane0):
        assert 'sem="release"' in source
    assert lane0.count("mask=lane0") == 2
    assert "mask=lane0 & (ticket + 1 == target)" in lane0
    assert "if ticket" not in lane0


def test_amd_relaxed_arrival_dynamic_savings_oracle() -> None:
    """Quantify one fence pair and optional rendezvous per production arrival."""

    no_routes = _production_arrival_counts(0)
    assert no_routes == {
        "phase": 0,
        "shared": 224,
        "w13": 0,
        "w2": 0,
        "comm": 21,
    }
    assert sum(no_routes.values()) == 245
    for local_routes in range(1, 17):
        counts = _production_arrival_counts(local_routes)
        assert counts["w13"] == 226
        assert counts["w2"] == 226
        assert sum(counts.values()) == 697


def test_amd_relaxed_arrival_multi_workgroup_publication() -> None:
    """Stress all three arrival forms over 4096 retained generations."""

    if not _env_enabled(_GATE_ENV):
        pytest.skip(f"set {_GATE_ENV}=1 for the gfx950 relaxed-arrival litmus")
    if not torch.cuda.is_available():
        pytest.skip("the relaxed-arrival litmus requires a CUDA device")
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    if not architecture.startswith("gfx950"):
        pytest.skip(f"the relaxed-arrival litmus requires gfx950, got {architecture}")
    if int(getattr(properties, "multi_processor_count", 0)) != 256:
        pytest.skip("the relaxed-arrival litmus requires the 256-CU target")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the relaxed-arrival litmus requires system ROCr in LD_PRELOAD")

    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "300"))
    variant_selection = os.getenv(_VARIANTS_ENV, "all").strip().lower()
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if timeout_ns <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")
    if variant_selection == "all":
        variants = _VARIANTS
    elif variant_selection == "scalar":
        variants = _VARIANTS[:2]
    else:
        raise ValueError(f'{_VARIANTS_ENV} must be "all" or "scalar"')

    context = mp.spawn(
        _worker_main,
        args=(iterations, timeout_ns, variants),
        nprocs=1,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    "gfx950 relaxed-arrival litmus exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise
