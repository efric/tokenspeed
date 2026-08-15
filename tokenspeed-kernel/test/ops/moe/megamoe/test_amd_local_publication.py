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

"""Opt-in gfx950 local gate-to-reused-payload publication litmus.

One producer workgroup rewrites a 512-word payload while a cooperative grid of
240 workgroups consumes it across all eight XCDs. Every generation uses a fresh
kernel invocation while preserving payload, gates, arrivals, and diagnostics;
this avoids treating a compiler-elected subgroup's scalar-loop re-entry as
whole-workgroup participation. A second gate prevents payload reuse until every
consumer has finished. The specializations compare:

* one compiler-elected GPU-acquire poll followed by an explicit invalidate-ACK
  wait and convergent workgroup barrier; and
* the same bounded control poll plus a live-result GPU-acquire in every
  consuming subgroup, followed by the same ACK wait and barrier; and
* the elected protocol with subgroup 0 alone loading all 512 words at eight
  words per lane, proving ownership transfer from all producer subgroups.

The second protocol is diagnostic, not presumed stronger: equivalent
system-scope per-subgroup atomics failed the separate Iris cross-NUMA litmus.
This test establishes whether the cheaper elected protocol suffices for local
scratch reused across XCDs before MegaMoE expands local-gate synchronization.

Run only on one reserved gfx950 with the pinned cooperative ROCr stack::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_local_publication.py

The test-only drain-only ablation keeps every producer subgroup's ``.wt``
store and VMEM drain but omits the explicit cache-wide WBL. It is separately
gated and runs only the all-subgroup-producer/subgroup-0-consumer shape::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION_DRAIN_ONLY=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_local_publication.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from tokenspeed_kernel_amd._triton import gl, gluon

_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION"
_DRAIN_ONLY_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION_DRAIN_ONLY"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_LOCAL_PUBLICATION_WATCHDOG_SECONDS"
_PROGRAMS = 240
_LANES = 512
_KERNEL_PROGRAMS = gl.constexpr(_PROGRAMS)
_KERNEL_LANES = gl.constexpr(_LANES)


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


@gluon.jit
def _wait_subgroup_vmem_ack():
    """Wait in every subgroup for any preceding invalidate acknowledgement."""

    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _publish_subgroup_vmem_system():
    """Make every calling producer subgroup's payload system-visible."""

    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\nbuffer_wbl2 sc0 sc1\ns_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _acquire_local_payload(
    ready_gate,
    generation,
    lane,
    TIMEOUT_NS: gl.constexpr,
    PER_SUBGROUP: gl.constexpr,
):
    """Bound control progress, complete invalidation, then converge consumers."""

    ready = gl.atomic_poll(
        ready_gate,
        generation,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    ready = ready != 0
    if PER_SUBGROUP:
        subgroup_leader = lane % 64 == 0
        acquired = gl.atomic_add(
            ready_gate + lane * 0,
            0,
            mask=subgroup_leader,
            sem="acquire",
            scope="gpu",
        )
        acquired_here = gl.where(subgroup_leader, acquired >= generation, True)
        ready &= gl.min(acquired_here, axis=0)

    # AMD's BUFFER_INV returns an ACK through VMCNT. A workgroup barrier does
    # not wait on that counter, so complete it explicitly before any subgroup
    # may issue a payload load.
    _wait_subgroup_vmem_ack()
    gl.barrier()
    return ready != 0


@gluon.jit
def _local_publication_kernel(
    payload,
    ready_gate,
    completion_arrival,
    completion_gate,
    lane_error_counts,
    first_failure_generation,
    first_failure_observed,
    control_failures,
    worker_xcc,
    generation_ptr,
    TIMEOUT_NS: gl.constexpr,
    PER_SUBGROUP: gl.constexpr,
    SUBGROUP_ZERO_CONSUMER: gl.constexpr,
    DRAIN_ONLY: gl.constexpr,
):
    """Publish and consume one generation in a fresh cooperative dispatch."""

    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup = lane // 64
    subgroup_leader = lane % 64 == 0
    generation = gl.load(generation_ptr, cache_modifier=".cv")

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
    gl.store(
        worker_xcc + pid + lane * 0,
        xcc,
        mask=lane == 0,
        cache_modifier=".wt",
    )

    expected = generation * 1_000_000 + lane.to(gl.int64)
    if pid == 0:
        gl.amd.cdna4.buffer_store(
            expected,
            payload,
            lane.to(gl.int32),
            cache=".wt",
        )
        if DRAIN_ONLY:
            _wait_subgroup_vmem_ack()
        else:
            _publish_subgroup_vmem_system()
        gl.barrier()
        gl.atomic_xchg(
            ready_gate,
            generation,
            sem="release",
            scope="gpu",
        )

    ready = _acquire_local_payload(
        ready_gate,
        generation,
        lane,
        TIMEOUT_NS,
        PER_SUBGROUP,
    )
    if SUBGROUP_ZERO_CONSUMER:
        # Production-faithful ownership transfer: all eight producer
        # subgroups write, but subgroup 0 alone consumes every word.
        consumer_lane = lane < 64
        for word_group in gl.static_range(0, 8):
            word = lane + word_group * 64
            expected_word = generation * 1_000_000 + word.to(gl.int64)
            observed = gl.amd.cdna4.buffer_load(
                payload,
                word.to(gl.int32),
                mask=consumer_lane,
                other=-1,
                cache=".cv",
            )
            mismatch = consumer_lane & (observed != expected_word)
            output_offset = pid * _KERNEL_LANES + word
            gl.atomic_add(
                lane_error_counts + output_offset,
                1,
                mask=mismatch,
                sem="relaxed",
                scope="gpu",
            )
            previous_generation = gl.load(
                first_failure_generation + output_offset,
                mask=consumer_lane,
                other=0,
                cache_modifier=".cv",
            )
            capture = mismatch & (previous_generation < 0)
            gl.store(
                first_failure_generation + output_offset,
                generation,
                mask=capture,
                cache_modifier=".wt",
            )
            gl.store(
                first_failure_observed + output_offset,
                observed,
                mask=capture,
                cache_modifier=".wt",
            )
    else:
        observed = gl.amd.cdna4.buffer_load(
            payload,
            lane.to(gl.int32),
            cache=".cv",
        )
        mismatch = observed != expected
        output_offset = pid * _KERNEL_LANES + lane
        gl.atomic_add(
            lane_error_counts + output_offset,
            1,
            mask=mismatch,
            sem="relaxed",
            scope="gpu",
        )
        previous_generation = gl.load(
            first_failure_generation + output_offset,
            cache_modifier=".cv",
        )
        capture = mismatch & (previous_generation < 0)
        gl.store(
            first_failure_generation + output_offset,
            generation,
            mask=capture,
            cache_modifier=".wt",
        )
        gl.store(
            first_failure_observed + output_offset,
            observed,
            mask=capture,
            cache_modifier=".wt",
        )
    gl.atomic_add(
        control_failures + pid * 8 + subgroup,
        1,
        mask=subgroup_leader & (not ready),
        sem="relaxed",
        scope="gpu",
    )

    # Do not let the producer reuse payload until every consuming subgroup has
    # completed its loads for this generation.
    _wait_subgroup_vmem_ack()
    gl.barrier()
    ticket = gl.atomic_add(
        completion_arrival,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if ticket + 1 == generation * _KERNEL_PROGRAMS:
        gl.atomic_xchg(
            completion_gate,
            generation,
            sem="release",
            scope="gpu",
        )
    complete = gl.atomic_poll(
        completion_gate,
        generation,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )
    complete = complete != 0
    gl.atomic_add(
        control_failures + pid * 8 + subgroup,
        1,
        mask=subgroup_leader & (not complete),
        sem="relaxed",
        scope="gpu",
    )


def _assert_loaded_local_publication_isa(
    assembly: str,
    *,
    per_subgroup: bool,
    subgroup_zero_consumer: bool,
    drain_only: bool,
) -> str:
    """Retain the source protocol in the loaded gfx950 object."""

    if drain_only:
        publication = re.findall(
            r"\bbuffer_store_dwordx2\b[^\n]*\bsc0 sc1\b"
            r"(?:(?!;;#ASMSTART).)*?"
            r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
            r"(?P<before_barrier>"
            r"(?:(?!\bbuffer_wbl2\b|\bs_barrier\b).)*?)"
            r"\bs_barrier\b",
            assembly,
            flags=re.DOTALL,
        )
    else:
        publication = re.findall(
            r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+"
            r"buffer_wbl2 sc0 sc1\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
            r"(?P<before_barrier>.*?)\bs_barrier\b",
            assembly,
            flags=re.DOTALL,
        )
    assert len(publication) == 1
    assert all("atomic" not in body and "store" not in body for body in publication)
    assert re.search(r"\bbuffer_store_dwordx2\b[^\n]*\bsc0 sc1\b", assembly)
    assert re.search(r"\bbuffer_load_dwordx2\b[^\n]*\bsc0 sc1\b", assembly)
    assert re.search(
        r"\bbuffer_inv\b[^\n]*"
        r"(?:(?!\bbuffer_load_dwordx2\b).)*?"
        r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
        r"(?:(?!\bbuffer_load_dwordx2\b).)*?\bs_barrier\b"
        r"(?:(?!\bbuffer_inv\b).)*?\bbuffer_load_dwordx2\b",
        assembly,
        flags=re.DOTALL,
    )
    if per_subgroup:
        assert re.search(
            r"\bglobal_load_dwordx2\b[^\n]*\bsc1\b"
            r"[^\n]*\n\s*s_waitcnt vmcnt\(0\)\s*\n\s*buffer_inv\b",
            assembly,
        )
        assert re.search(
            r"\bbuffer_inv\b"
            r"(?:(?!\bbuffer_load_dwordx2\b).)*?\bv_cmp_ge_i64_e(?:32|64)\b"
            r"(?:(?!\bbuffer_load_dwordx2\b).)*?"
            r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
            r"(?:(?!\bbuffer_inv\b).)*?\bs_barrier\b"
            r"(?:(?!\bbuffer_inv\b).)*?\bbuffer_load_dwordx2\b",
            assembly,
            flags=re.DOTALL,
        )
    if subgroup_zero_consumer:
        assert assembly.count("buffer_load_dwordx2") >= 8
    return hashlib.sha256(assembly.encode("utf-8")).hexdigest()


def _run_variant(
    device: torch.device,
    *,
    iterations: int,
    timeout_ns: int,
    per_subgroup: bool,
    subgroup_zero_consumer: bool,
    drain_only: bool,
) -> dict[str, object]:
    payload = torch.full((_LANES,), -1, dtype=torch.int64, device=device)
    ready_gate = torch.zeros(1, dtype=torch.int64, device=device)
    completion_arrival = torch.zeros(1, dtype=torch.int64, device=device)
    completion_gate = torch.zeros(1, dtype=torch.int64, device=device)
    lane_error_counts = torch.zeros(
        (_PROGRAMS, _LANES),
        dtype=torch.int32,
        device=device,
    )
    first_failure_generation = torch.full(
        (_PROGRAMS, _LANES),
        -1,
        dtype=torch.int64,
        device=device,
    )
    first_failure_observed = torch.full_like(first_failure_generation, -1)
    control_failures = torch.zeros(
        (_PROGRAMS, 8),
        dtype=torch.int32,
        device=device,
    )
    worker_xcc = torch.full((_PROGRAMS,), -1, dtype=torch.int32, device=device)
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
    args = (
        payload,
        ready_gate,
        completion_arrival,
        completion_gate,
        lane_error_counts,
        first_failure_generation,
        first_failure_observed,
        control_failures,
        worker_xcc,
        generation_values[:1],
    )
    compiled = _local_publication_kernel.warmup(
        *args,
        TIMEOUT_NS=timeout_ns,
        PER_SUBGROUP=per_subgroup,
        SUBGROUP_ZERO_CONSUMER=subgroup_zero_consumer,
        DRAIN_ONLY=drain_only,
        grid=(_PROGRAMS,),
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )
    isa_sha256 = _assert_loaded_local_publication_isa(
        compiled.asm["amdgcn"],
        per_subgroup=per_subgroup,
        subgroup_zero_consumer=subgroup_zero_consumer,
        drain_only=drain_only,
    )
    hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    retained_cache_identity = _loaded_kernel_cache_identity(_local_publication_kernel)
    assert retained_cache_identity
    with patch.object(
        _local_publication_kernel,
        "_do_compile",
        side_effect=AssertionError("generation replay attempted to recompile"),
    ):
        for generation_value in range(iterations):
            _local_publication_kernel[(_PROGRAMS,)](
                *args[:-1],
                generation_values[generation_value],
                TIMEOUT_NS=timeout_ns,
                PER_SUBGROUP=per_subgroup,
                SUBGROUP_ZERO_CONSUMER=subgroup_zero_consumer,
                DRAIN_ONLY=drain_only,
                num_warps=8,
                num_stages=1,
                waves_per_eu=2,
                launch_cooperative_grid=True,
            )
    assert retained_cache_identity == _loaded_kernel_cache_identity(
        _local_publication_kernel
    )
    torch.cuda.synchronize(device)

    errors = lane_error_counts.cpu()
    first_generation = first_failure_generation.cpu()
    first_observed = first_failure_observed.cpu()
    bad = torch.nonzero(first_generation >= 0)
    first_failures = tuple(
        {
            "program": int(program),
            "lane": int(lane),
            "generation": int(first_generation[program, lane]),
            "observed": int(first_observed[program, lane]),
        }
        for program, lane in bad[:16].tolist()
    )
    xcc_values = worker_xcc.cpu()
    xcc_ids, xcc_counts = torch.unique(xcc_values, return_counts=True)
    controls = control_failures.cpu()
    return {
        "protocol": (
            "drain_only_subgroup0_all_words"
            if drain_only
            else (
                "elected_ack_wait_subgroup0_all_words"
                if subgroup_zero_consumer
                else "per_subgroup_live" if per_subgroup else "elected_ack_wait"
            )
        ),
        "iterations": iterations,
        "payload_errors": int(errors.sum()),
        "bad_lanes_by_subgroup": tuple(
            int(torch.count_nonzero(errors[:, subgroup * 64 : (subgroup + 1) * 64]))
            for subgroup in range(8)
        ),
        "bad_words_by_producer_subgroup": tuple(
            int(torch.count_nonzero(errors[:, subgroup * 64 : (subgroup + 1) * 64]))
            for subgroup in range(8)
        ),
        "first_failures": first_failures,
        "control_failures": int(controls.sum()),
        "control_failures_by_subgroup": tuple(
            int(controls[:, subgroup].sum()) for subgroup in range(8)
        ),
        "ready_gate": int(ready_gate.item()),
        "completion_gate": int(completion_gate.item()),
        "completion_arrival": int(completion_arrival.item()),
        "xcc_population": {
            int(xcc_id): int(count)
            for xcc_id, count in zip(xcc_ids.tolist(), xcc_counts.tolist())
        },
        "isa_sha256": isa_sha256,
        "hsaco_sha256": hsaco_sha256,
    }


def _worker_main(
    _worker_index: int,
    iterations: int,
    timeout_ns: int,
    drain_only: bool,
) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    variants = (
        ((False, True),)
        if drain_only
        else (
            (False, False),
            (True, False),
            (False, True),
        )
    )
    reports = tuple(
        _run_variant(
            device,
            iterations=iterations,
            timeout_ns=timeout_ns,
            per_subgroup=per_subgroup,
            subgroup_zero_consumer=subgroup_zero_consumer,
            drain_only=drain_only,
        )
        for per_subgroup, subgroup_zero_consumer in variants
    )
    print("MegaMoE local publication reports:", json.dumps(reports), flush=True)

    elected = reports[0]
    assert elected["control_failures"] == 0, elected
    assert elected["payload_errors"] == 0, elected
    assert elected["ready_gate"] == iterations, elected
    assert elected["completion_gate"] == iterations, elected
    assert elected["completion_arrival"] == iterations * _PROGRAMS, elected
    assert len(elected["xcc_population"]) == 8, elected

    ownership_transfer = reports[0] if drain_only else reports[2]
    assert ownership_transfer["control_failures"] == 0, ownership_transfer
    assert ownership_transfer["payload_errors"] == 0, ownership_transfer
    assert ownership_transfer["ready_gate"] == iterations, ownership_transfer
    assert ownership_transfer["completion_gate"] == iterations, ownership_transfer
    assert (
        ownership_transfer["completion_arrival"] == iterations * _PROGRAMS
    ), ownership_transfer
    assert len(ownership_transfer["xcc_population"]) == 8, ownership_transfer


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


def _run_test(*, drain_only: bool) -> None:
    gate_env = _DRAIN_ONLY_GATE_ENV if drain_only else _GATE_ENV
    if not _env_enabled(gate_env):
        pytest.skip(f"set {gate_env}=1 for the gfx950 local publication litmus")
    if not torch.cuda.is_available():
        pytest.skip("the local publication litmus requires a CUDA device")
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    if not architecture.startswith("gfx950"):
        pytest.skip(f"the local publication litmus requires gfx950, got {architecture}")
    if int(getattr(properties, "multi_processor_count", 0)) != 256:
        pytest.skip("the local publication litmus requires the 256-CU target")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the local publication litmus requires system ROCr in LD_PRELOAD")

    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "180"))
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if timeout_ns <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")

    context = mp.spawn(
        _worker_main,
        args=(iterations, timeout_ns, drain_only),
        nprocs=1,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    "gfx950 local publication litmus exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


def test_amd_local_gate_reused_payload_publication() -> None:
    """Compare local acquire protocols across the full persistent grid."""

    _run_test(drain_only=False)


def test_amd_local_gate_reused_payload_publication_drain_only() -> None:
    """Ablate only the explicit producer WBL in the ownership-transfer case."""

    _run_test(drain_only=True)
