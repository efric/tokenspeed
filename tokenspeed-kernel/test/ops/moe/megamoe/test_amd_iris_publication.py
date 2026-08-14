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

"""Opt-in gfx950 Iris payload-publication memory-ordering litmus.

The test isolates the production-faithful Iris publication path with one
eight-subgroup workgroup per rank and exact ready/completion generations:

* all eight producer subgroups own 16 packed words and publish their own VMEM;
* a system-release exchange writes one exact generation to the ready plane;
* local and XOR-ordered peer waits use bounded system-acquire atomic polling;
* every acquire is followed by an explicit VMEM acknowledgment and barrier;
* subgroup 0 coherently consumes all 128 self and peer words; and
* a separate completion plane prevents the next generation from overwriting
  either exact flag until every rank has finished its payload loads;
* litmus-only outputs expose every wait result to all eight subgroup leaders
  at every static wait site; and
* each generation uses a fresh dispatch of one retained compiled binary with
  host-side anti-cascade checks before the next launch.

Run two ranks or all eight ranks with the pinned system ROCr, for example::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION=2,8 \
      TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_BURST=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_iris_publication.py

The test-only drain-only ablation keeps the ``.wt`` producer stores and every
producer subgroup's VMEM drain, but omits the explicit cache-wide WBL before
the ready release.  It always uses burst replay and has a separate opt-in gate::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_DRAIN_ONLY=2,8 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_iris_publication.py

Iterations default to 4096 and the per-poll timeout defaults to one second.
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import socket
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel_amd._triton import gl, gluon, tl


_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION"
_DRAIN_ONLY_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_DRAIN_ONLY"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_WATCHDOG_SECONDS"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_DUMP_AMDGCN_DIR"
_BURST_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_BURST"
_BURST_SKEW_US_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_PUBLICATION_BURST_SKEW_US"
_LANES = 512
_SUBGROUPS = 8
_PAYLOAD_WORDS = 128
_WAIT_CONTROL_FIELDS = 4
_KERNEL_LANES = gl.constexpr(_LANES)
_KERNEL_SUBGROUPS = gl.constexpr(_SUBGROUPS)
_KERNEL_WAIT_CONTROL_FIELDS = gl.constexpr(_WAIT_CONTROL_FIELDS)


@gluon.jit
def _drain_subgroup_vmem():
    """Drain every calling subgroup's outstanding vector-memory operations."""

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
def _poll_exact_generation(
    flag,
    generation,
    wait_control,
    iteration,
    lane,
    WAIT_SITE: gl.constexpr,
    WAIT_SITES: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Acquire-poll one exact, non-overwritable generation flag."""

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
        # Failure-only diagnostics do not participate in progress. The exact
        # flag must still contain either the previous generation or the target;
        # completion synchronization forbids a future overwrite.
        observed = gl.load(flag, cache_modifier=".cv")

    # Litmus-only instrumentation: every subgroup leader records the control
    # tuple it sees at every static wait site. The host checks all eight slots
    # after every fresh dispatch, including the generation tag so a subgroup
    # that skipped this path cannot be hidden by a prior launch's value.
    subgroup = lane // 64
    control_slot = (
        (iteration * WAIT_SITES + WAIT_SITE) * _KERNEL_SUBGROUPS + subgroup
    ) * _KERNEL_WAIT_CONTROL_FIELDS
    subgroup_leader = lane % 64 == 0
    gl.store(
        wait_control + control_slot,
        iteration + 1,
        mask=subgroup_leader,
        cache_modifier=".wt",
    )
    gl.store(
        wait_control + control_slot + 1,
        observed,
        mask=subgroup_leader,
        cache_modifier=".wt",
    )
    gl.store(
        wait_control + control_slot + 2,
        gl.where(matched, 1, 0),
        mask=subgroup_leader,
        cache_modifier=".wt",
    )
    gl.store(
        wait_control + control_slot + 3,
        gl.where(matched, 0, 1),
        mask=subgroup_leader,
        cache_modifier=".wt",
    )
    return matched, observed


@gluon.jit
def _iris_publication_litmus_kernel(
    payload,
    generation_flags,
    error_count,
    remote_payload_error_count,
    self_payload_error_count,
    wait_control,
    timeout_iteration,
    failure_phase,
    failure_peer,
    failure_seen,
    failure_acquired,
    payload_failure_iteration,
    payload_failure_peer,
    payload_failure_observed,
    heap_bases,
    start_iteration_ptr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
    DRAIN_ONLY: gl.constexpr,
):
    """Stress distributed payload -> exact ready/complete flag planes."""

    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    lane_error_count = error_count + lane * 0
    local_heap = gl.load(heap_bases + RANK).to(gl.uint64, bitcast=True)
    payload_offset = tl.cast(payload, gl.uint64) - local_heap
    flag_offset = tl.cast(generation_flags, gl.uint64) - local_heap
    local_ready_flag = generation_flags + RANK
    local_completion_flag = generation_flags + WORLD_SIZE + RANK
    iteration = gl.load(start_iteration_ptr)
    generation = iteration + 1

    # One fresh workgroup invocation executes exactly one generation. This
    # compile-time-only scope keeps the generation body visually grouped; it
    # emits no scalar generation loop or runtime control edge.
    if True:
        consumer_lane = lane < 64
        # The first 16 lanes of every subgroup own 16 distinct words. This is
        # the production-faithful multi-subgroup producer shape.
        subgroup_lane = lane % 64
        word = (lane // 64) * 16 + subgroup_lane
        expected_local = (
            generation * 1_000_000 + (RANK + 1) * 10_000 + word.to(gl.int64)
        )
        gl.amd.cdna4.buffer_store(
            expected_local,
            payload,
            word.to(gl.int32),
            mask=subgroup_lane < 16,
            cache=".wt",
        )

        # A scalar publisher's release cannot write back dirty payload lines
        # owned by the other seven subgroups.  Every producer must publish its
        # own prior stores before the workgroup elects the epoch publisher.
        if DRAIN_ONLY:
            _drain_subgroup_vmem()
        else:
            _publish_subgroup_vmem_system()
        gl.barrier()
        previous_ready = gl.atomic_xchg(
            local_ready_flag,
            generation,
            sem="release",
            scope="sys",
        )
        ready_publication_valid = previous_ready == iteration
        if not ready_publication_valid:
            gl.atomic_add(error_count, 1, sem="relaxed", scope="gpu")
            first_publication_failure = gl.atomic_cas(
                failure_peer,
                -1,
                RANK,
                sem="relaxed",
                scope="gpu",
            )
            if first_publication_failure == -1:
                gl.store(failure_seen, previous_ready)
                gl.store(failure_acquired, iteration)
            gl.atomic_xchg(
                timeout_iteration,
                generation,
                sem="relaxed",
                scope="gpu",
            )
            gl.atomic_xchg(failure_phase, 1, sem="relaxed", scope="gpu")

        self_ready, self_seen = _poll_exact_generation(
            local_ready_flag,
            generation,
            wait_control,
            iteration,
            lane,
            WAIT_SITE=0,
            WAIT_SITES=2 * WORLD_SIZE - 1,
            TIMEOUT_NS=TIMEOUT_NS,
        )
        # Complete atomic_poll's system-acquire invalidation before any
        # subgroup can cross into coherent payload loads.
        _drain_subgroup_vmem()
        gl.barrier()
        if not self_ready:
            gl.atomic_add(error_count, 1, sem="relaxed", scope="gpu")
            gl.atomic_xchg(timeout_iteration, generation, sem="relaxed", scope="gpu")
            gl.atomic_xchg(failure_phase, 5, sem="relaxed", scope="gpu")
            gl.atomic_xchg(failure_peer, RANK, sem="relaxed", scope="gpu")
            gl.store(failure_seen, self_seen)
            gl.store(failure_acquired, generation)

        all_ready = gl.full((), True, gl.int1)
        # XOR rounds are contention-free. Distances 1..3 stay within each
        # four-GPU NUMA half; distance 4 starts matched cross-NUMA polling.
        for distance in gl.static_range(1, WORLD_SIZE):
            peer = RANK ^ distance
            peer_heap = gl.load(heap_bases + peer).to(
                gl.uint64,
                bitcast=True,
            )
            peer_flags = tl.cast(
                peer_heap + flag_offset,
                gl.pointer_type(gl.int64),
            )
            peer_ready, peer_seen = _poll_exact_generation(
                peer_flags + peer,
                generation,
                wait_control,
                iteration,
                lane,
                WAIT_SITE=distance,
                WAIT_SITES=2 * WORLD_SIZE - 1,
                TIMEOUT_NS=TIMEOUT_NS,
            )
            _drain_subgroup_vmem()
            gl.barrier()
            if not peer_ready:
                previous_peer = gl.atomic_cas(
                    failure_peer,
                    -1,
                    peer,
                    sem="relaxed",
                    scope="gpu",
                )
                if previous_peer == -1:
                    gl.store(failure_seen, peer_seen)
                    gl.store(failure_acquired, generation)
            all_ready &= peer_ready
        if not all_ready:
            gl.atomic_add(error_count, 1, sem="relaxed", scope="gpu")
            gl.atomic_xchg(timeout_iteration, generation, sem="relaxed", scope="gpu")
            gl.atomic_xchg(failure_phase, 2, sem="relaxed", scope="gpu")

        # Payload traversal remains canonical rank order, matching the
        # production accumulation order even though flag polling is XOR-ordered.
        for peer in gl.static_range(0, WORLD_SIZE):
            if peer == RANK:
                peer_payload = payload
                payload_error_count = self_payload_error_count + lane * 0
            else:
                peer_heap = gl.load(heap_bases + peer).to(
                    gl.uint64,
                    bitcast=True,
                )
                peer_payload = tl.cast(
                    peer_heap + payload_offset,
                    gl.pointer_type(gl.int64),
                )
                payload_error_count = remote_payload_error_count + lane * 0
            for word_group in gl.static_range(0, 2):
                word = lane + word_group * 64
                observed = gl.amd.cdna4.buffer_load(
                    peer_payload,
                    word.to(gl.int32),
                    mask=consumer_lane,
                    other=-1,
                    cache=".cv",
                )
                expected_peer = (
                    generation * 1_000_000 + (peer + 1) * 10_000 + word.to(gl.int64)
                )
                payload_mismatch = consumer_lane & (observed != expected_peer)
                uncaptured = (
                    gl.load(
                        payload_failure_iteration + word,
                        mask=consumer_lane,
                        other=0,
                    )
                    < 0
                )
                capture = payload_mismatch & uncaptured
                gl.store(
                    payload_failure_iteration + word,
                    generation,
                    mask=capture,
                )
                gl.store(payload_failure_peer + word, peer, mask=capture)
                gl.store(
                    payload_failure_observed + word,
                    observed,
                    mask=capture,
                )
                gl.atomic_add(
                    lane_error_count,
                    1,
                    mask=payload_mismatch,
                    sem="relaxed",
                    scope="gpu",
                )
                gl.atomic_add(
                    payload_error_count,
                    1,
                    mask=payload_mismatch,
                    sem="relaxed",
                    scope="gpu",
                )

        # Publish completion only after every payload load is acknowledged and
        # every subgroup has crossed the workgroup rendezvous. Separate flag
        # planes make both atomic_poll waits exact and non-overwritable.
        _drain_subgroup_vmem()
        gl.barrier()
        previous_completion = gl.atomic_xchg(
            local_completion_flag,
            generation,
            sem="release",
            scope="sys",
        )
        if previous_completion != iteration:
            gl.atomic_add(error_count, 1, sem="relaxed", scope="gpu")
            gl.atomic_xchg(timeout_iteration, generation, sem="relaxed", scope="gpu")
            gl.atomic_xchg(failure_phase, 3, sem="relaxed", scope="gpu")

        all_complete = gl.full((), True, gl.int1)
        for distance in gl.static_range(1, WORLD_SIZE):
            peer = RANK ^ distance
            peer_heap = gl.load(heap_bases + peer).to(
                gl.uint64,
                bitcast=True,
            )
            peer_flags = tl.cast(
                peer_heap + flag_offset,
                gl.pointer_type(gl.int64),
            )
            peer_complete, peer_seen = _poll_exact_generation(
                peer_flags + WORLD_SIZE + peer,
                generation,
                wait_control,
                iteration,
                lane,
                WAIT_SITE=WORLD_SIZE - 1 + distance,
                WAIT_SITES=2 * WORLD_SIZE - 1,
                TIMEOUT_NS=TIMEOUT_NS,
            )
            _drain_subgroup_vmem()
            gl.barrier()
            if not peer_complete:
                previous_peer = gl.atomic_cas(
                    failure_peer,
                    -1,
                    peer,
                    sem="relaxed",
                    scope="gpu",
                )
                if previous_peer == -1:
                    gl.store(failure_seen, peer_seen)
                    gl.store(failure_acquired, generation)
            all_complete &= peer_complete
        if not all_complete:
            gl.atomic_add(error_count, 1, sem="relaxed", scope="gpu")
            gl.atomic_xchg(timeout_iteration, generation, sem="relaxed", scope="gpu")
            gl.atomic_xchg(failure_phase, 4, sem="relaxed", scope="gpu")


def _selected_world_sizes(gate_env: str) -> set[int]:
    raw = os.getenv(gate_env, "")
    try:
        selected = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(
            f"{gate_env} must be a comma-separated subset of 2,8"
        ) from error
    if not selected.issubset({2, 8}):
        raise ValueError(f"{gate_env} must be a comma-separated subset of 2,8")
    return selected


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _require_parent_gate(world_size: int, gate_env: str) -> None:
    if world_size not in _selected_world_sizes(gate_env):
        pytest.skip(f"include {world_size} in {gate_env} to run this hardware litmus")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"the Iris publication litmus requires {world_size} GPUs")
    arches = tuple(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        for index in range(world_size)
    )
    if any(not arch.startswith("gfx950") for arch in arches):
        pytest.skip(f"all {world_size} visible devices must be gfx950, got {arches}")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the Iris publication litmus requires system ROCr in LD_PRELOAD")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _assert_loaded_publication_isa(
    assembly: str,
    world_size: int,
    *,
    drain_only: bool,
) -> str:
    """Prove the payload publication contract in the loaded gfx950 object."""

    isa_sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / f"world{world_size}-{isa_sha256}.amdgcn").write_text(assembly)

    wait_sites = 2 * world_size - 1

    # Each bounded atomic_poll must compare one exact INT64 generation. Its
    # success path performs a system-scope invalidate and broadcasts the i1
    # result through LDS. An at-least comparison would reintroduce the
    # overshoot race that the separate ready/completion planes eliminate.
    poll_graphs = list(
        re.finditer(
            r"\bs_memrealtime\b"
            r"(?:(?!\bs_memrealtime\b).)*?"
            r"\bglobal_load_dwordx2\b[^\n]*\bsc0 sc1\b"
            r"(?:(?!\bs_memrealtime\b).)*?"
            r"\bv_cmp_ne_u64(?:_e(?:32|64))?\b[^\n]*"
            r"(?:(?!\bs_memrealtime\b).)*?"
            r"\bs_memrealtime\b"
            r"(?:(?!\bbuffer_inv\b).)*?"
            r"\bv_cmp_gt_u64(?:_e(?:32|64))?\b[^\n]*"
            r"(?:(?!\bbuffer_inv\b).)*?"
            r"\bbuffer_inv sc0 sc1\b"
            r"(?:(?!\bds_write_b8\b).)*?\bds_write_b8\b"
            r"(?:(?!\bs_barrier\b).)*?\bs_barrier\b"
            r"(?:(?!\bds_read_u8\b).)*?\bds_read_u8\b",
            assembly,
            flags=re.DOTALL,
        )
    )
    assert len(poll_graphs) == wait_sites, (len(poll_graphs), wait_sites)
    assert assembly.count("s_memrealtime") == 2 * wait_sites
    assert assembly.count("buffer_inv sc0 sc1") == wait_sites
    assert assembly.count("ds_write_b8") == wait_sites
    assert assembly.count("ds_read_u8") == wait_sites

    # In addition to atomic_poll's internal result rendezvous, every local or
    # peer acquire has a source-owned VMCNT ACK and workgroup barrier. One more
    # drain/barrier precedes completion publication after all payload loads.
    all_explicit_drains = list(
        re.finditer(
            r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
            r"(?:(?!;;#ASMSTART).)*?\bs_barrier\b",
            assembly,
            flags=re.DOTALL,
        )
    )
    if drain_only:
        # The first source-owned drain is the producer publication ablation;
        # the remaining sites acknowledge exact ready/completion acquires.
        assert len(all_explicit_drains) == 2 * world_size + 1, (
            len(all_explicit_drains),
            2 * world_size + 1,
        )
        producer_drain = all_explicit_drains[0]
        explicit_drains = all_explicit_drains[1:]
    else:
        producer_drain = None
        explicit_drains = all_explicit_drains
    assert len(explicit_drains) == 2 * world_size, (
        len(explicit_drains),
        2 * world_size,
    )

    # Ready and completion are two distinct system-scope INT64 locations. The
    # release lowering writes back prior traffic, swaps the exact generation,
    # and broadcasts the returned previous generation for the no-overwrite
    # assertion. Their byte offsets differ by exactly one WORLD_SIZE plane.
    release_graphs = list(
        re.finditer(
            r"\bbuffer_wbl2 sc0 sc1\b\s+"
            r"s_waitcnt vmcnt\(0\)\s+"
            r"(?P<swap>global_atomic_swap_x2[^\n]*\bsc0 sc1)\s+"
            r"s_waitcnt vmcnt\(0\)\s+"
            r"ds_write_b64"
            r"(?:(?!\bglobal_atomic_swap_x2\b).)*?"
            r"\bs_barrier\b\s+ds_read_b64",
            assembly,
            flags=re.DOTALL,
        )
    )
    assert len(release_graphs) == 2
    release_offsets = tuple(
        int(offset.group(1))
        if (offset := re.search(r"offset:(\d+)", match.group("swap")))
        else 0
        for match in release_graphs
    )
    assert release_offsets[1] - release_offsets[0] == world_size * 8, release_offsets

    # The distributed producer has exactly one .wt store site owned by all
    # eight subgroups, followed by a per-subgroup drain/WBL/drain and a
    # workgroup barrier before the elected ready release.
    if drain_only:
        producer_publications = list(
            re.finditer(
                r"\bbuffer_store_dwordx2\b[^\n]*\bsc0 sc1\b"
                r"(?:(?!;;#ASMSTART).)*?"
                r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
                r"(?P<before_barrier>"
                r"(?:(?!\bbuffer_wbl2\b|\bs_barrier\b).)*?)"
                r"\bs_barrier\b"
                r"(?:(?!\bglobal_atomic_swap_x2\b).)*?"
                r"\bglobal_atomic_swap_x2\b[^\n]*\bsc0 sc1\b",
                assembly,
                flags=re.DOTALL,
            )
        )
        assert len(producer_publications) == 1
        assert producer_drain is not None
        assert producer_drain.start() < release_graphs[0].start()
    else:
        producer_publications = list(
            re.finditer(
                r"\bbuffer_store_dwordx2\b[^\n]*\bsc0 sc1\b"
                r"(?:(?!;;#ASMSTART).)*?"
                r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+"
                r"buffer_wbl2 sc0 sc1\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND"
                r"(?:(?!\bglobal_atomic_swap_x2\b).)*?\bs_barrier\b"
                r"(?:(?!\bglobal_atomic_swap_x2\b).)*?"
                r"\bglobal_atomic_swap_x2\b[^\n]*\bsc0 sc1\b",
                assembly,
                flags=re.DOTALL,
            )
        )
        assert len(producer_publications) == 1
    payload_loads = list(
        re.finditer(
            r"\bbuffer_load_dwordx2\b[^\n]*\bsc0 sc1\b",
            assembly,
        )
    )
    assert len(payload_loads) == 2 * world_size

    # The static sequence must be:
    # ready release -> local/peer ready polls -> payload -> drain/completion
    # release -> peer completion polls. This makes both equality polls
    # non-overwritable and keeps payload traversal canonical.
    ready_polls = poll_graphs[:world_size]
    completion_polls = poll_graphs[world_size:]
    assert release_graphs[0].end() < ready_polls[0].start()
    for index, poll in enumerate(ready_polls):
        assert poll.end() < explicit_drains[index].start()
        if index + 1 < len(ready_polls):
            assert explicit_drains[index].end() < ready_polls[index + 1].start()
    assert explicit_drains[world_size - 1].end() < payload_loads[0].start()
    assert payload_loads[-1].end() < explicit_drains[world_size].start()
    assert explicit_drains[world_size].end() < release_graphs[1].start()
    assert release_graphs[1].end() < completion_polls[0].start()
    for index, poll in enumerate(completion_polls):
        drain = explicit_drains[world_size + 1 + index]
        assert poll.end() < drain.start()
        if index + 1 < len(completion_polls):
            assert drain.end() < completion_polls[index + 1].start()

    # Litmus-only control output has four INT64 fields for every generation,
    # static wait site, and subgroup. Loaded ISA retains all four field stores;
    # runtime history checks prove the dynamic generation/site base and require
    # every subgroup slot to carry generation, matched=1, timed_out=0.
    control_stores = re.findall(
        r"^\s*(global_store_dwordx2[^\n]*\bsc0 sc1)\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    control_offsets = tuple(
        int(offset.group(1)) if (offset := re.search(r"offset:(\d+)", store)) else 0
        for store in control_stores
    )
    assert len(control_stores) == 4 * wait_sites
    assert set(control_offsets) == {0, 8, 16, 24}
    assert all(control_offsets.count(field * 8) == wait_sites for field in range(4))
    subgroup_lane = re.search(
        r"\bv_and_b32_e32 v(?P<local_lane>\d+), 63, v0\b"
        r".*?"
        r"\bv_cmp_eq_u32_e64 s\[\d+:\d+\], 0, "
        r"v(?P=local_lane)\b",
        assembly,
        flags=re.DOTALL,
    )
    assert subgroup_lane

    assert assembly.count("buffer_wbl2 sc0 sc1") == (2 if drain_only else 3)
    assert "global_atomic_add_x2" not in assembly
    assert "global_atomic_cmpswap_x2" not in assembly
    return isa_sha256


def _first_wait_control_failure(
    controls: torch.Tensor,
    expected_generations: torch.Tensor,
) -> dict[str, object] | None:
    """Return the first generation whose eight subgroup controls disagree."""

    if controls.ndim == 3:
        controls = controls.unsqueeze(0)
    expected = expected_generations.reshape(-1, 1, 1)
    generations = controls[:, :, :, 0]
    observations = controls[:, :, :, 1]
    ready = controls[:, :, :, 2]
    timed_out = controls[:, :, :, 3]
    valid_by_generation = (
        torch.all(generations == expected, dim=(1, 2))
        & torch.all(observations == expected, dim=(1, 2))
        & torch.all(ready == 1, dim=(1, 2))
        & torch.all(timed_out == 0, dim=(1, 2))
    )
    failed = torch.nonzero(~valid_by_generation).flatten()
    if not failed.numel():
        return None
    index = int(failed[0].item())
    return {
        "iteration": int(expected_generations[index].item()),
        "generations": tuple(tuple(row) for row in generations[index].tolist()),
        "observations": tuple(tuple(row) for row in observations[index].tolist()),
        "ready": tuple(tuple(row) for row in ready[index].tolist()),
        "timed_out": tuple(tuple(row) for row in timed_out[index].tolist()),
    }


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    iterations: int,
    timeout_ns: int,
    burst: bool,
    burst_skew_us: int,
    drain_only: bool,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    shmem = None
    try:
        import iris

        shmem = iris.iris(1 << 20)
        assert shmem.get_rank() == rank
        assert shmem.get_num_ranks() == world_size
        payload = shmem.zeros(_PAYLOAD_WORDS, dtype=torch.int64, device=device)
        generation_flags = shmem.zeros(
            2 * world_size,
            dtype=torch.int64,
            device=device,
        )
        error_count = torch.zeros(1, dtype=torch.int32, device=device)
        remote_payload_error_count = torch.zeros(1, dtype=torch.int32, device=device)
        self_payload_error_count = torch.zeros(1, dtype=torch.int32, device=device)
        wait_sites = 2 * world_size - 1
        wait_control = torch.full(
            (iterations, wait_sites, _SUBGROUPS, _WAIT_CONTROL_FIELDS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        timeout_iteration = torch.zeros(1, dtype=torch.int64, device=device)
        failure_phase = torch.zeros(1, dtype=torch.int32, device=device)
        failure_peer = torch.full((1,), -1, dtype=torch.int32, device=device)
        failure_seen = torch.full((1,), -1, dtype=torch.int64, device=device)
        failure_acquired = torch.full((1,), -1, dtype=torch.int64, device=device)
        payload_failure_iteration = torch.full(
            (_PAYLOAD_WORDS,), -1, dtype=torch.int64, device=device
        )
        payload_failure_peer = torch.full(
            (_PAYLOAD_WORDS,), -1, dtype=torch.int32, device=device
        )
        payload_failure_observed = torch.full(
            (_PAYLOAD_WORDS,), -1, dtype=torch.int64, device=device
        )
        heap_bases = shmem.get_heap_bases()
        iteration_index = torch.arange(iterations, dtype=torch.int64, device=device)
        iteration_values = torch.stack(
            (iteration_index, torch.zeros_like(iteration_index)),
            dim=1,
        )

        compiled = _iris_publication_litmus_kernel.warmup(
            payload,
            generation_flags,
            error_count,
            remote_payload_error_count,
            self_payload_error_count,
            wait_control,
            timeout_iteration,
            failure_phase,
            failure_peer,
            failure_seen,
            failure_acquired,
            payload_failure_iteration,
            payload_failure_peer,
            payload_failure_observed,
            heap_bases,
            iteration_values[:1],
            RANK=rank,
            WORLD_SIZE=world_size,
            TIMEOUT_NS=timeout_ns,
            DRAIN_ONLY=drain_only,
            grid=(1,),
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
        isa_sha256 = _assert_loaded_publication_isa(
            compiled.asm["amdgcn"],
            world_size,
            drain_only=drain_only,
        )
        hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
        retained_cache_identity = _loaded_kernel_cache_identity(
            _iris_publication_litmus_kernel
        )
        assert retained_cache_identity
        retained_runner_ids = tuple(
            runner_id
            for _, cache_entries in retained_cache_identity
            for _, runner_id in cache_entries
        )
        assert len(retained_runner_ids) == 1
        retained_cache_identity_sha256 = hashlib.sha256(
            repr(retained_cache_identity).encode("utf-8")
        ).hexdigest()

        # Prime every HIP queue before coordinating the first launch. Serving
        # reaches MegaMoE after substantial GPU work, so first-use queue skew
        # is not part of the publication contract this litmus measures.
        queue_prime = torch.zeros(1, dtype=torch.int32, device=device)
        queue_prime.add_(1)
        torch.cuda.synchronize(device)
        assert int(queue_prime.item()) == 1

        # No rank starts polling while another rank is still compiling or
        # initializing its local HIP queue.
        dist.barrier()
        failure_vote = torch.zeros(1, dtype=torch.int32)
        launched_iterations = 0
        max_kernel_elapsed_ns = 0
        host_epoch_mismatch: tuple[int, int, int] | None = None
        first_wait_control_failure: dict[str, object] | None = None
        with patch.object(
            _iris_publication_litmus_kernel,
            "_do_compile",
            side_effect=AssertionError("generation replay attempted to recompile"),
        ):

            def launch_generation(start_iteration: int) -> None:
                _iris_publication_litmus_kernel[(1,)](
                    payload,
                    generation_flags,
                    error_count,
                    remote_payload_error_count,
                    self_payload_error_count,
                    wait_control,
                    timeout_iteration,
                    failure_phase,
                    failure_peer,
                    failure_seen,
                    failure_acquired,
                    payload_failure_iteration,
                    payload_failure_peer,
                    payload_failure_observed,
                    heap_bases,
                    iteration_values[start_iteration],
                    RANK=rank,
                    WORLD_SIZE=world_size,
                    TIMEOUT_NS=timeout_ns,
                    DRAIN_ONLY=drain_only,
                    num_warps=8,
                    num_stages=1,
                    waves_per_eu=2,
                )

            if burst:
                burst_start_ns = time.monotonic_ns()
                for start_iteration in range(iterations):
                    # Repeated rank-dependent host launch delays create device
                    # stream skew without synchronizing ranks between exact
                    # generations. Each stream still preserves local order.
                    if rank and start_iteration % 64 == 0:
                        time.sleep(rank * burst_skew_us / 1_000_000)
                    launch_generation(start_iteration)
                torch.cuda.synchronize(device)
                max_kernel_elapsed_ns = time.monotonic_ns() - burst_start_ns
                launched_iterations = iterations
                observed_ready_generation = int(generation_flags[rank].item())
                observed_completion_generation = int(
                    generation_flags[world_size + rank].item()
                )
                if (
                    observed_ready_generation != iterations
                    or observed_completion_generation != iterations
                ):
                    host_epoch_mismatch = (
                        iterations,
                        observed_ready_generation,
                        observed_completion_generation,
                    )
                first_wait_control_failure = _first_wait_control_failure(
                    wait_control.cpu(),
                    torch.arange(1, iterations + 1, dtype=torch.int64),
                )
                local_failed = int(
                    error_count.item() != 0
                    or host_epoch_mismatch is not None
                    or first_wait_control_failure is not None
                )
                failure_vote.fill_(local_failed)
                dist.all_reduce(failure_vote, op=dist.ReduceOp.MAX)
            else:
                for start_iteration in range(iterations):
                    generation_start_ns = time.monotonic_ns()
                    launch_generation(start_iteration)
                    # Do not enqueue a later generation after any rank reports
                    # a failure. This diagnostic mode prevents stale-payload
                    # cascades and identifies the first divergent generation.
                    torch.cuda.synchronize(device)
                    max_kernel_elapsed_ns = max(
                        max_kernel_elapsed_ns,
                        time.monotonic_ns() - generation_start_ns,
                    )
                    launched_iterations = start_iteration + 1
                    observed_ready_generation = int(generation_flags[rank].item())
                    observed_completion_generation = int(
                        generation_flags[world_size + rank].item()
                    )
                    expected_generation = launched_iterations
                    epoch_matches = (
                        observed_ready_generation == expected_generation
                        and observed_completion_generation == expected_generation
                    )
                    if not epoch_matches and host_epoch_mismatch is None:
                        host_epoch_mismatch = (
                            launched_iterations,
                            observed_ready_generation,
                            observed_completion_generation,
                        )
                    first_wait_control_failure = _first_wait_control_failure(
                        wait_control[start_iteration].cpu(),
                        torch.tensor([launched_iterations], dtype=torch.int64),
                    )
                    local_failed = int(
                        error_count.item() != 0
                        or not epoch_matches
                        or first_wait_control_failure is not None
                    )
                    failure_vote.fill_(local_failed)
                    dist.all_reduce(failure_vote, op=dist.ReduceOp.MAX)
                    if failure_vote.item() != 0:
                        break
        assert retained_cache_identity == _loaded_kernel_cache_identity(
            _iris_publication_litmus_kernel
        )
        torch.cuda.synchronize(device)

        local_flags = generation_flags.cpu()
        bad_iterations = payload_failure_iteration.cpu()
        bad_peers = payload_failure_peer.cpu()
        bad_observed = payload_failure_observed.cpu()
        bad_words = torch.nonzero(bad_iterations >= 0).flatten().tolist()
        visible_devices = os.getenv("ROCR_VISIBLE_DEVICES", "")
        physical_devices = (
            tuple(device.strip() for device in visible_devices.split(","))
            if visible_devices
            else tuple(str(device) for device in range(world_size))
        )
        first_payload_failures = tuple(
            {
                "word": word_index,
                "iteration": int(bad_iterations[word_index].item()),
                "peer": int(bad_peers[word_index].item()),
                "observed": int(bad_observed[word_index].item()),
                "expected": int(bad_iterations[word_index].item()) * 1_000_000
                + (int(bad_peers[word_index].item()) + 1) * 10_000
                + word_index,
            }
            for word_index in sorted(
                bad_words,
                key=lambda index: (int(bad_iterations[index].item()), index),
            )[:16]
        )
        report = {
            "rank": rank,
            "logical_device": rank,
            "physical_device": physical_devices[rank],
            "producer_assignment": "distributed_16_words_per_subgroup",
            "publication_protocol": (
                "drain_only" if drain_only else "drain_wbl2_system"
            ),
            "burst": burst,
            "burst_skew_us": burst_skew_us,
            "errors": int(error_count.item()),
            "remote_payload_errors": int(remote_payload_error_count.item()),
            "self_payload_errors": int(self_payload_error_count.item()),
            "max_kernel_elapsed_ns": max_kernel_elapsed_ns,
            "timeout_iteration": int(timeout_iteration.item()),
            "failure_phase": int(failure_phase.item()),
            "failure_peer": int(failure_peer.item()),
            "failure_seen": int(failure_seen.item()),
            "failure_acquired": int(failure_acquired.item()),
            "launched_iterations": launched_iterations,
            "host_epoch_mismatch": host_epoch_mismatch,
            "first_wait_control_failure": first_wait_control_failure,
            "final_ready_generation": int(local_flags[rank].item()),
            "final_completion_generation": int(local_flags[world_size + rank].item()),
            "untouched_ready_slots": tuple(
                int(local_flags[peer].item())
                for peer in range(world_size)
                if peer != rank
            ),
            "untouched_completion_slots": tuple(
                int(local_flags[world_size + peer].item())
                for peer in range(world_size)
                if peer != rank
            ),
            "bad_words_by_64_word_chunk": tuple(
                sum(
                    1
                    for word_index in bad_words
                    if chunk * 64 <= word_index < (chunk + 1) * 64
                )
                for chunk in range(2)
            ),
            "first_payload_failures": first_payload_failures,
            "isa_sha256": isa_sha256,
            "hsaco_sha256": hsaco_sha256,
            "retained_runner_ids": retained_runner_ids,
            "retained_cache_identity_sha256": retained_cache_identity_sha256,
        }
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, report)
        if rank == 0:
            assert all(item is not None for item in gathered)
            print("Iris publication rank reports:", gathered, flush=True)
            for item in gathered:
                assert item is not None
                assert item["errors"] == 0, item
                assert item["remote_payload_errors"] == 0, item
                assert item["self_payload_errors"] == 0, item
                assert item["timeout_iteration"] == 0, item
                assert item["failure_phase"] == 0, item
                assert item["launched_iterations"] == iterations, item
                assert item["host_epoch_mismatch"] is None, item
                assert item["first_wait_control_failure"] is None, item
                assert item["final_ready_generation"] == iterations, item
                assert item["final_completion_generation"] == iterations, item
                assert not any(item["untouched_ready_slots"]), item
                assert not any(item["untouched_completion_slots"]), item
                assert len(item["isa_sha256"]) == 64, item
                assert len(item["hsaco_sha256"]) == 64, item
                assert len(item["retained_runner_ids"]) == 1, item
                assert len(item["retained_cache_identity_sha256"]) == 64, item
            print(
                "MegaMoE Iris publication litmus PASS:",
                {
                    "world_size": world_size,
                    "iterations": iterations,
                    "burst": burst,
                    "burst_skew_us": burst_skew_us,
                    "producer_assignment": "distributed_16_words_per_subgroup",
                    "publication_protocol": (
                        "drain_only" if drain_only else "drain_wbl2_system"
                    ),
                    "remote_word_checks": iterations
                    * world_size
                    * (world_size - 1)
                    * _PAYLOAD_WORDS,
                    "self_word_checks": iterations * world_size * _PAYLOAD_WORDS,
                    "subgroup_control_checks": iterations
                    * world_size
                    * (2 * world_size - 1)
                    * _SUBGROUPS,
                    "rank_specialized_isa_sha256": tuple(
                        item["isa_sha256"] for item in gathered if item
                    ),
                    "rank_specialized_hsaco_sha256": tuple(
                        item["hsaco_sha256"] for item in gathered if item
                    ),
                },
                flush=True,
            )
        dist.barrier()
    finally:
        if shmem is not None:
            del shmem
            gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


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


def _run_litmus(
    world_size: int,
    *,
    drain_only: bool,
) -> None:
    gate_env = _DRAIN_ONLY_GATE_ENV if drain_only else _GATE_ENV
    _require_parent_gate(world_size, gate_env)
    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "240"))
    burst_value = os.getenv(_BURST_ENV, "0").strip().lower()
    if burst_value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{_BURST_ENV} must be 0, 1, false, or true")
    burst = drain_only or burst_value in {"1", "true"}
    burst_skew_us = int(os.getenv(_BURST_SKEW_US_ENV, "1000"))
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if timeout_ns <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")
    if burst_skew_us < 0:
        raise ValueError(f"{_BURST_SKEW_US_ENV} must be non-negative")

    context = mp.spawn(
        _worker_main,
        args=(
            world_size,
            _open_port(),
            iterations,
            timeout_ns,
            burst,
            burst_skew_us,
            drain_only,
        ),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    f"world{world_size} Iris publication litmus exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


@pytest.mark.parametrize("world_size", [2, 8])
def test_amd_iris_publication_distributed_producers(world_size: int) -> None:
    _run_litmus(world_size, drain_only=False)


@pytest.mark.parametrize("world_size", [2, 8])
def test_amd_iris_publication_distributed_producers_drain_only(
    world_size: int,
) -> None:
    """Ablate only the explicit producer WBL while retaining exact flags."""

    _run_litmus(world_size, drain_only=True)
