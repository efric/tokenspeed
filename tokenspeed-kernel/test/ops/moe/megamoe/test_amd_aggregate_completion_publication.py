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

"""Test-only gfx950 aggregate-completion publication litmus.

This retained-binary world-eight test isolates a proposed reduction of Iris's
21 per-tile completion handshakes to one exact system completion flag per rank.
It deliberately leaves production MegaMoE, Iris ownership, admission, and the
implementation-complete marker untouched.

Every fresh dispatch executes one reused generation with 21 workgroups:

* all eight producer subgroups write one 128-word tile and drain VMEM;
* every tile publishes and acquire-polls its exact ready flag on all ranks;
* subgroup zero checks all eight rank-distinct payloads and writes the reduced
  tile through the cache hierarchy;
* all 21 workgroups drain, then relaxed-arrive at one GPU-local counter;
* the last arrival releases a local completion gate;
* tile zero acquire-polls that gate, system-releases one exact rank flag, then
  relaxed-polls the seven peer aggregate flags without cache invalidation; and
* tile zero releases a final local gate before the stream may dispatch the next
  generation.

The other 20 workgroups do not poll the final gate. Their operations are
drained before arrival, and a subsequent fresh dispatch cannot start until the
whole prior kernel, including tile zero, has returned. This mirrors the
production reuse argument without imposing an extra all-21 acquire.

Run with all eight gfx950 devices reserved and the pinned system ROCr::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/\
test_amd_aggregate_completion_publication.py

The default is 4096 stream-serialized fresh dispatches with periodic per-rank
host launch skew and a one-second bounded poll timeout.

A separate GPU0-only world-one smoke compiles the same protocol shape and runs
16 retained generations before the eight-rank qualification is attempted::

    TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_SMOKE=1 \
      pytest -q -s \
      test_amd_aggregate_completion_publication.py::\
test_amd_aggregate_completion_world1_smoke
"""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
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

_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION"
_SMOKE_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_SMOKE"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_WATCHDOG_SECONDS"
_SKEW_US_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_SKEW_US"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_AGGREGATE_COMPLETION_DUMP_AMDGCN_DIR"
_WORLD_SIZE = 8
_NUM_TILES = 21
_SUBGROUPS = 8
_LANES = 512
_WORDS_PER_TILE = 128
_KERNEL_LANES = gl.constexpr(_LANES)
_KERNEL_SUBGROUPS = gl.constexpr(_SUBGROUPS)
_KERNEL_NUM_TILES = gl.constexpr(_NUM_TILES)
_KERNEL_WORDS_PER_TILE = gl.constexpr(_WORDS_PER_TILE)


@gluon.jit
def _drain_subgroup_vmem():
    """Drain outstanding vector memory in every calling subgroup."""

    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _poll_ready_acquire(flag, generation, TIMEOUT_NS: gl.constexpr):
    """Acquire one payload-publishing exact system generation."""

    return gl.atomic_poll(
        flag,
        generation,
        sem="acquire",
        scope="sys",
        timeout_ns=TIMEOUT_NS,
    )


@gluon.jit
def _poll_local_completion_acquire(flag, generation, TIMEOUT_NS: gl.constexpr):
    """Acquire the last of the 21 drained local arrivals."""

    return gl.atomic_poll(
        flag,
        generation,
        sem="acquire",
        scope="gpu",
        timeout_ns=TIMEOUT_NS,
    )


@gluon.jit
def _poll_aggregate_completion_relaxed(
    flag,
    generation,
    TIMEOUT_NS: gl.constexpr,
):
    """Observe peer read completion without acquiring any peer payload."""

    return gl.atomic_poll(
        flag,
        generation,
        sem="relaxed",
        scope="sys",
        timeout_ns=TIMEOUT_NS,
    )


@gluon.jit
def _aggregate_completion_litmus_kernel(
    payload,
    reduced,
    ready_flags,
    aggregate_flags,
    local_arrival,
    local_completion_gate,
    final_gate,
    payload_errors,
    reduced_errors,
    payload_checks,
    reduced_checks,
    ready_control,
    aggregate_control,
    arrival_tickets,
    publication_errors,
    ready_timeouts,
    local_timeouts,
    aggregate_timeouts,
    heap_bases,
    start_iteration_ptr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Run one generation of 21-tile ready and aggregate completion."""

    tile = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup = lane // 64
    subgroup_lane = lane % 64
    subgroup_leader = subgroup_lane == 0
    consumer_lane = lane < 64
    iteration = gl.load(start_iteration_ptr)
    generation = iteration + 1

    local_heap = gl.load(heap_bases + RANK).to(gl.uint64, bitcast=True)
    payload_heap_offset = tl.cast(payload, gl.uint64) - local_heap
    ready_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    aggregate_heap_offset = tl.cast(aggregate_flags, gl.uint64) - local_heap

    # The first 16 lanes in each subgroup own disjoint words. Values differ by
    # generation, rank, tile, and word so stale, swapped, or duplicate reads
    # cannot hide behind a symmetric payload.
    producer_word = subgroup * 16 + subgroup_lane
    local_payload_offset = tile * _KERNEL_WORDS_PER_TILE + producer_word
    expected_local = (
        generation * 10_000_000
        + (RANK + 1) * 1_000_000
        + tile * 10_000
        + producer_word.to(gl.int64)
    )
    gl.amd.cdna4.buffer_store(
        expected_local,
        payload,
        local_payload_offset.to(gl.int32),
        mask=subgroup_lane < 16,
        cache=".wt",
    )
    _drain_subgroup_vmem()
    gl.barrier()

    local_ready_flag = ready_flags + tile * WORLD_SIZE + RANK
    previous_ready = gl.atomic_xchg(
        local_ready_flag,
        generation,
        sem="release",
        scope="sys",
    )
    if previous_ready != iteration:
        gl.atomic_add(publication_errors, 1, sem="relaxed", scope="gpu")

    all_ready = _poll_ready_acquire(
        local_ready_flag,
        generation,
        TIMEOUT_NS,
    )
    for distance in gl.static_range(1, WORLD_SIZE):
        ready_peer = RANK ^ distance
        peer_heap = gl.load(heap_bases + ready_peer).to(gl.uint64, bitcast=True)
        peer_ready_flags = tl.cast(
            peer_heap + ready_heap_offset,
            gl.pointer_type(gl.int64),
        )
        peer_ready = _poll_ready_acquire(
            peer_ready_flags + tile * WORLD_SIZE + ready_peer,
            generation,
            TIMEOUT_NS,
        )
        all_ready &= peer_ready

    control_offset = (
        iteration * _KERNEL_NUM_TILES + tile
    ) * _KERNEL_SUBGROUPS + subgroup
    gl.store(
        ready_control + control_offset,
        gl.where(all_ready, generation, -generation),
        mask=subgroup_leader,
        cache_modifier=".wt",
    )
    if not all_ready:
        gl.atomic_add(ready_timeouts, 1, sem="relaxed", scope="gpu")
        return

    # Payload traversal remains rank-ascending. Subgroup zero owns all 128
    # coherent loads and writes the exact local reduced tile.
    for word_group in gl.static_range(0, 2):
        word = lane + word_group * 64
        packed_offset = tile * _KERNEL_WORDS_PER_TILE + word
        total = gl.zeros([_KERNEL_LANES], gl.int64, layout=layout)
        for peer in gl.static_range(0, WORLD_SIZE):
            if peer == RANK:
                peer_payload = payload
            else:
                peer_heap = gl.load(heap_bases + peer).to(
                    gl.uint64,
                    bitcast=True,
                )
                peer_payload = tl.cast(
                    peer_heap + payload_heap_offset,
                    gl.pointer_type(gl.int64),
                )
            observed = gl.amd.cdna4.buffer_load(
                peer_payload,
                packed_offset.to(gl.int32),
                mask=consumer_lane,
                other=0,
                cache=".cv",
            )
            expected_peer = (
                generation * 10_000_000
                + (peer + 1) * 1_000_000
                + tile * 10_000
                + word.to(gl.int64)
            )
            mismatch = consumer_lane & (observed != expected_peer)
            gl.atomic_add(
                payload_errors + tile + lane * 0,
                1,
                mask=mismatch,
                sem="relaxed",
                scope="gpu",
            )
            total += observed

        expected_total = (
            WORLD_SIZE * generation * 10_000_000
            + (WORLD_SIZE * (WORLD_SIZE + 1) // 2) * 1_000_000
            + WORLD_SIZE * tile * 10_000
            + WORLD_SIZE * word.to(gl.int64)
        )
        reduced_mismatch = consumer_lane & (total != expected_total)
        gl.atomic_add(
            reduced_errors + tile + lane * 0,
            1,
            mask=reduced_mismatch,
            sem="relaxed",
            scope="gpu",
        )
        gl.amd.cdna4.buffer_store(
            total,
            reduced,
            packed_offset.to(gl.int32),
            mask=consumer_lane,
            cache=".wt",
        )

    gl.atomic_add(
        payload_checks + tile + lane * 0,
        WORLD_SIZE * _KERNEL_WORDS_PER_TILE,
        mask=lane == 0,
        sem="relaxed",
        scope="gpu",
    )
    gl.atomic_add(
        reduced_checks + tile + lane * 0,
        _KERNEL_WORDS_PER_TILE,
        mask=lane == 0,
        sem="relaxed",
        scope="gpu",
    )

    # Every tile drains peer loads and local reduced stores before contributing
    # to the relaxed GPU-local arrival modification order.
    _drain_subgroup_vmem()
    gl.barrier()
    ticket = gl.atomic_add(
        local_arrival,
        1,
        sem="relaxed",
        scope="gpu",
    )
    gl.store(
        arrival_tickets + iteration * _KERNEL_NUM_TILES + tile + lane * 0,
        ticket,
        mask=lane == 0,
        cache_modifier=".wt",
    )
    if ticket // _KERNEL_NUM_TILES != iteration:
        gl.atomic_add(publication_errors, 1, sem="relaxed", scope="gpu")
    if ticket % _KERNEL_NUM_TILES == _KERNEL_NUM_TILES - 1:
        previous_local_gate = gl.atomic_xchg(
            local_completion_gate,
            generation,
            sem="release",
            scope="gpu",
        )
        if previous_local_gate != iteration:
            gl.atomic_add(publication_errors, 1, sem="relaxed", scope="gpu")

    # Only tile zero controls the rank-level system completion handshake. The
    # other 20 workgroups have no operations left after their drained arrival.
    if tile == 0:
        local_complete = _poll_local_completion_acquire(
            local_completion_gate,
            generation,
            TIMEOUT_NS,
        )
        if not local_complete:
            gl.atomic_add(local_timeouts, 1, sem="relaxed", scope="gpu")

        all_complete = local_complete
        if local_complete:
            local_aggregate_flag = aggregate_flags + RANK
            previous_aggregate = gl.atomic_xchg(
                local_aggregate_flag,
                generation,
                sem="release",
                scope="sys",
            )
            if previous_aggregate != iteration:
                gl.atomic_add(publication_errors, 1, sem="relaxed", scope="gpu")

            for distance in gl.static_range(1, WORLD_SIZE):
                completion_peer = RANK ^ distance
                peer_heap = gl.load(heap_bases + completion_peer).to(
                    gl.uint64,
                    bitcast=True,
                )
                peer_aggregate_flags = tl.cast(
                    peer_heap + aggregate_heap_offset,
                    gl.pointer_type(gl.int64),
                )
                peer_complete = _poll_aggregate_completion_relaxed(
                    peer_aggregate_flags + completion_peer,
                    generation,
                    TIMEOUT_NS,
                )
                all_complete &= peer_complete

        gl.store(
            aggregate_control + iteration * _KERNEL_SUBGROUPS + subgroup,
            gl.where(all_complete, generation, -generation),
            mask=subgroup_leader,
            cache_modifier=".wt",
        )
        if not all_complete:
            gl.atomic_add(aggregate_timeouts, 1, sem="relaxed", scope="gpu")
            return

        previous_final = gl.atomic_xchg(
            final_gate,
            generation,
            sem="release",
            scope="gpu",
        )
        if previous_final != iteration:
            gl.atomic_add(publication_errors, 1, sem="relaxed", scope="gpu")


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _loaded_kernel_cache_identity(
    kernel,
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    """Identify the exact per-device runners retained by one JIT function."""

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


def _atomic_poll_source_line(kernel, helper_name: str) -> int:
    source_path = Path(__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    call = next(
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    assert kernel is not None
    return call.lineno


def _poll_graphs_for_source_line(
    assembly: str,
    source_line: int,
) -> list[str]:
    """Find bounded atomic-poll graphs carrying one source provenance."""

    lines = assembly.splitlines()
    source_marker = f"{Path(__file__).name}:{source_line}:"
    graphs: list[str] = []
    for start, line in enumerate(lines):
        if "s_memrealtime" not in line:
            continue
        provenance = "\n".join(lines[max(0, start - 80) : start + 1])
        if source_marker not in provenance:
            continue
        end = next(
            (
                index
                for index in range(start + 1, min(len(lines), start + 320))
                if "ds_read_u8" in lines[index]
            ),
            None,
        )
        if end is None:
            continue
        graph = "\n".join(lines[start : end + 1])
        if "global_load_dwordx2" not in graph:
            continue
        graphs.append(graph)
    return graphs


def _assert_loaded_aggregate_completion_isa(
    assembly: str,
    world_size: int,
) -> str:
    """Classify acquire-ready and relaxed aggregate poll graphs."""

    assembly = assembly.lower()
    isa_sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / f"world{world_size}-{isa_sha256}.amdgcn").write_text(assembly)

    ready_line = _atomic_poll_source_line(
        _poll_ready_acquire,
        "_poll_ready_acquire",
    )
    local_line = _atomic_poll_source_line(
        _poll_local_completion_acquire,
        "_poll_local_completion_acquire",
    )
    aggregate_line = _atomic_poll_source_line(
        _poll_aggregate_completion_relaxed,
        "_poll_aggregate_completion_relaxed",
    )
    ready_graphs = _poll_graphs_for_source_line(assembly, ready_line)
    local_graphs = _poll_graphs_for_source_line(assembly, local_line)
    aggregate_graphs = _poll_graphs_for_source_line(assembly, aggregate_line)
    assert len(ready_graphs) == world_size
    assert len(local_graphs) == 1
    assert len(aggregate_graphs) == world_size - 1

    for graph in ready_graphs:
        assert "global_load_dwordx2" in graph and "sc0 sc1" in graph
        assert re.search(r"v_cmp_(?:eq|ne)_u64", graph)
        assert graph.count("buffer_inv sc0 sc1") == 1
        assert graph.count("s_barrier") == 1
        assert "ds_write_b8" in graph and "ds_read_u8" in graph
    for graph in local_graphs:
        assert re.search(r"v_cmp_(?:eq|ne)_u64", graph)
        assert "buffer_inv" in graph
        assert graph.count("s_barrier") == 1
    for graph in aggregate_graphs:
        assert "global_load_dwordx2" in graph and "sc0 sc1" in graph
        assert re.search(r"v_cmp_(?:eq|ne)_u64", graph)
        assert "buffer_inv" not in graph
        assert graph.count("s_barrier") == 1
        assert "ds_write_b8" in graph and "ds_read_u8" in graph

    system_swaps = re.findall(
        r"^\s*global_atomic_swap_x2[^\n]*sc0 sc1\s*$",
        assembly,
        re.MULTILINE,
    )
    assert len(system_swaps) == 2  # dynamic-tile ready and aggregate completion
    return isa_sha256


def _expected_payload(generation: int, rank: int, tile: int, word: int) -> int:
    return generation * 10_000_000 + (rank + 1) * 1_000_000 + tile * 10_000 + word


def _expected_reduced(
    generation: int,
    tile: int,
    word: int,
    world_size: int = _WORLD_SIZE,
) -> int:
    return sum(
        _expected_payload(generation, rank, tile, word) for rank in range(world_size)
    )


def _require_hardware_gate(world_size: int, gate_env: str) -> None:
    if os.getenv(gate_env, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"set {gate_env}=1 for the world{world_size} aggregate litmus")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(
            f"the aggregate-completion litmus requires {world_size} visible GPUs"
        )
    arches = tuple(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        for index in range(world_size)
    )
    if any(not arch.startswith("gfx950") for arch in arches):
        pytest.skip(f"all {world_size} visible devices must be gfx950, got {arches}")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the aggregate-completion litmus requires pinned system ROCr")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    iterations: int,
    timeout_ns: int,
    skew_us: int,
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
        total_words = _NUM_TILES * _WORDS_PER_TILE
        payload = shmem.zeros(total_words, dtype=torch.int64, device=device)
        ready_flags = shmem.zeros(
            _NUM_TILES * world_size,
            dtype=torch.int64,
            device=device,
        )
        aggregate_flags = shmem.zeros(
            world_size,
            dtype=torch.int64,
            device=device,
        )
        reduced = torch.full((total_words,), -1, dtype=torch.int64, device=device)
        local_arrival = torch.zeros(1, dtype=torch.int64, device=device)
        local_completion_gate = torch.zeros(1, dtype=torch.int64, device=device)
        final_gate = torch.zeros(1, dtype=torch.int64, device=device)
        payload_errors = torch.zeros(_NUM_TILES, dtype=torch.int64, device=device)
        reduced_errors = torch.zeros(_NUM_TILES, dtype=torch.int64, device=device)
        payload_checks = torch.zeros(_NUM_TILES, dtype=torch.int64, device=device)
        reduced_checks = torch.zeros(_NUM_TILES, dtype=torch.int64, device=device)
        ready_control = torch.full(
            (iterations, _NUM_TILES, _SUBGROUPS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        aggregate_control = torch.full(
            (iterations, _SUBGROUPS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        arrival_tickets = torch.full(
            (iterations, _NUM_TILES),
            -1,
            dtype=torch.int64,
            device=device,
        )
        publication_errors = torch.zeros(1, dtype=torch.int64, device=device)
        ready_timeouts = torch.zeros(1, dtype=torch.int64, device=device)
        local_timeouts = torch.zeros(1, dtype=torch.int64, device=device)
        aggregate_timeouts = torch.zeros(1, dtype=torch.int64, device=device)
        heap_bases = shmem.get_heap_bases()
        iteration_index = torch.arange(iterations, dtype=torch.int64, device=device)
        iteration_values = torch.stack(
            (iteration_index, torch.zeros_like(iteration_index)),
            dim=1,
        )

        compiled = _aggregate_completion_litmus_kernel.warmup(
            payload,
            reduced,
            ready_flags,
            aggregate_flags,
            local_arrival,
            local_completion_gate,
            final_gate,
            payload_errors,
            reduced_errors,
            payload_checks,
            reduced_checks,
            ready_control,
            aggregate_control,
            arrival_tickets,
            publication_errors,
            ready_timeouts,
            local_timeouts,
            aggregate_timeouts,
            heap_bases,
            iteration_values[:1],
            RANK=rank,
            WORLD_SIZE=world_size,
            TIMEOUT_NS=timeout_ns,
            grid=(_NUM_TILES,),
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
        isa_sha256 = _assert_loaded_aggregate_completion_isa(
            compiled.asm["amdgcn"],
            world_size,
        )
        hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
        retained_identity = _loaded_kernel_cache_identity(
            _aggregate_completion_litmus_kernel
        )
        assert retained_identity

        queue_prime = torch.zeros(1, dtype=torch.int32, device=device)
        queue_prime.add_(1)
        torch.cuda.synchronize(device)
        assert int(queue_prime.item()) == 1
        dist.barrier()

        start_ns = time.monotonic_ns()
        with patch.object(
            _aggregate_completion_litmus_kernel,
            "_do_compile",
            side_effect=AssertionError("fresh dispatch attempted to recompile"),
        ):
            for iteration in range(iterations):
                if rank and iteration % 64 == 0:
                    time.sleep(rank * skew_us / 1_000_000)
                _aggregate_completion_litmus_kernel[(_NUM_TILES,)](
                    payload,
                    reduced,
                    ready_flags,
                    aggregate_flags,
                    local_arrival,
                    local_completion_gate,
                    final_gate,
                    payload_errors,
                    reduced_errors,
                    payload_checks,
                    reduced_checks,
                    ready_control,
                    aggregate_control,
                    arrival_tickets,
                    publication_errors,
                    ready_timeouts,
                    local_timeouts,
                    aggregate_timeouts,
                    heap_bases,
                    iteration_values[iteration],
                    RANK=rank,
                    WORLD_SIZE=world_size,
                    TIMEOUT_NS=timeout_ns,
                    num_warps=8,
                    num_stages=1,
                    waves_per_eu=2,
                )
        torch.cuda.synchronize(device)
        elapsed_ns = time.monotonic_ns() - start_ns
        assert retained_identity == _loaded_kernel_cache_identity(
            _aggregate_completion_litmus_kernel
        )

        expected_generation = torch.arange(
            1,
            iterations + 1,
            dtype=torch.int64,
        )
        ready_control_cpu = ready_control.cpu()
        aggregate_control_cpu = aggregate_control.cpu()
        expected_ready_control = expected_generation[:, None, None].expand(
            iterations,
            _NUM_TILES,
            _SUBGROUPS,
        )
        expected_aggregate_control = expected_generation[:, None].expand(
            iterations,
            _SUBGROUPS,
        )
        tickets_cpu = arrival_tickets.cpu()
        expected_tickets = torch.arange(
            iterations * _NUM_TILES,
            dtype=torch.int64,
        ).view(iterations, _NUM_TILES)
        final_generation = iterations
        tile_ids = torch.arange(_NUM_TILES, dtype=torch.int64)[:, None]
        words = torch.arange(_WORDS_PER_TILE, dtype=torch.int64)[None, :]
        expected_local_payload = (
            final_generation * 10_000_000
            + (rank + 1) * 1_000_000
            + tile_ids * 10_000
            + words
        )
        expected_reduced_payload = (
            world_size * final_generation * 10_000_000
            + (world_size * (world_size + 1) // 2) * 1_000_000
            + world_size * tile_ids * 10_000
            + world_size * words
        )

        local_ready = ready_flags.view(_NUM_TILES, world_size).cpu()
        local_aggregate = aggregate_flags.cpu()
        report = {
            "rank": rank,
            "iterations": iterations,
            "elapsed_ns": elapsed_ns,
            "publication_errors": int(publication_errors.item()),
            "ready_timeouts": int(ready_timeouts.item()),
            "local_timeouts": int(local_timeouts.item()),
            "aggregate_timeouts": int(aggregate_timeouts.item()),
            "payload_errors": tuple(int(value) for value in payload_errors.cpu()),
            "reduced_errors": tuple(int(value) for value in reduced_errors.cpu()),
            "payload_checks": tuple(int(value) for value in payload_checks.cpu()),
            "reduced_checks": tuple(int(value) for value in reduced_checks.cpu()),
            "local_arrival": int(local_arrival.item()),
            "local_completion_gate": int(local_completion_gate.item()),
            "final_gate": int(final_gate.item()),
            "ready_control_exact": bool(
                torch.equal(ready_control_cpu, expected_ready_control)
            ),
            "aggregate_control_exact": bool(
                torch.equal(aggregate_control_cpu, expected_aggregate_control)
            ),
            "arrival_tickets_exact": bool(
                torch.equal(torch.sort(tickets_cpu, dim=1).values, expected_tickets)
            ),
            "payload_exact": bool(
                torch.equal(
                    payload.cpu().view(_NUM_TILES, _WORDS_PER_TILE),
                    expected_local_payload,
                )
            ),
            "reduced_exact": bool(
                torch.equal(
                    reduced.cpu().view(_NUM_TILES, _WORDS_PER_TILE),
                    expected_reduced_payload,
                )
            ),
            "local_ready_exact": bool(
                torch.all(local_ready[:, rank] == iterations).item()
            ),
            "untouched_ready_zero": bool(
                torch.count_nonzero(
                    local_ready[
                        :,
                        [peer for peer in range(world_size) if peer != rank],
                    ]
                ).item()
                == 0
            ),
            "local_aggregate_exact": int(local_aggregate[rank].item()) == iterations,
            "untouched_aggregate_zero": bool(
                torch.count_nonzero(
                    local_aggregate[
                        [peer for peer in range(world_size) if peer != rank]
                    ]
                ).item()
                == 0
            ),
            "isa_sha256": isa_sha256,
            "hsaco_sha256": hsaco_sha256,
            "retained_identity_sha256": hashlib.sha256(
                repr(retained_identity).encode("utf-8")
            ).hexdigest(),
        }
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, report)
        if rank == 0:
            assert all(item is not None for item in gathered)
            for item in gathered:
                assert item is not None
                assert item["publication_errors"] == 0, item
                assert item["ready_timeouts"] == 0, item
                assert item["local_timeouts"] == 0, item
                assert item["aggregate_timeouts"] == 0, item
                assert not any(item["payload_errors"]), item
                assert not any(item["reduced_errors"]), item
                assert (
                    item["payload_checks"]
                    == (iterations * world_size * _WORDS_PER_TILE,) * _NUM_TILES
                ), item
                assert (
                    item["reduced_checks"]
                    == (iterations * _WORDS_PER_TILE,) * _NUM_TILES
                ), item
                assert item["local_arrival"] == iterations * _NUM_TILES, item
                assert item["local_completion_gate"] == iterations, item
                assert item["final_gate"] == iterations, item
                assert item["ready_control_exact"], item
                assert item["aggregate_control_exact"], item
                assert item["arrival_tickets_exact"], item
                assert item["payload_exact"], item
                assert item["reduced_exact"], item
                assert item["local_ready_exact"], item
                assert item["untouched_ready_zero"], item
                assert item["local_aggregate_exact"], item
                assert item["untouched_aggregate_zero"], item
            print(
                "MegaMoE aggregate completion litmus PASS:",
                {
                    "world_size": world_size,
                    "iterations": iterations,
                    "tiles": _NUM_TILES,
                    "payload_word_checks": iterations
                    * world_size
                    * _NUM_TILES
                    * world_size
                    * _WORDS_PER_TILE,
                    "reduced_word_checks": iterations
                    * world_size
                    * _NUM_TILES
                    * _WORDS_PER_TILE,
                    "rank_reports": gathered,
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
    processes = tuple(context.processes)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    survivors = tuple(process for process in processes if process.is_alive())
    for process in survivors:
        process.kill()
    for process in survivors:
        process.join(timeout=5)
    remaining = tuple(process.pid for process in survivors if process.is_alive())
    if remaining:
        raise RuntimeError(f"aggregate litmus workers survived cleanup: {remaining}")


def _run_litmus(
    world_size: int,
    gate_env: str,
    *,
    default_iterations: int,
) -> None:
    _require_hardware_gate(world_size, gate_env)
    iterations = int(os.getenv(_ITERATIONS_ENV, str(default_iterations)))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "300"))
    skew_us = int(os.getenv(_SKEW_US_ENV, "1000"))
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if timeout_ns <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")
    if skew_us < 0:
        raise ValueError(f"{_SKEW_US_ENV} must be nonnegative")

    context = mp.spawn(
        _worker_main,
        args=(world_size, _open_port(), iterations, timeout_ns, skew_us),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    f"world{world_size} aggregate completion litmus exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


def _run_world8_litmus() -> None:
    _run_litmus(
        _WORLD_SIZE,
        _GATE_ENV,
        default_iterations=4096,
    )


def _literal_keyword(call: ast.Call, name: str) -> object:
    keyword = next(keyword for keyword in call.keywords if keyword.arg == name)
    return ast.literal_eval(keyword.value)


def test_aggregate_completion_source_protocol_is_isolated_and_exact() -> None:
    """Guard the proposed memory orders without importing production raw code."""

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    ready_poll = next(
        node
        for node in ast.walk(functions["_poll_ready_acquire"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    local_poll = next(
        node
        for node in ast.walk(functions["_poll_local_completion_acquire"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    aggregate_poll = next(
        node
        for node in ast.walk(functions["_poll_aggregate_completion_relaxed"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    assert (
        _literal_keyword(ready_poll, "sem"),
        _literal_keyword(ready_poll, "scope"),
    ) == (
        "acquire",
        "sys",
    )
    assert (
        _literal_keyword(local_poll, "sem"),
        _literal_keyword(local_poll, "scope"),
    ) == (
        "acquire",
        "gpu",
    )
    assert (
        _literal_keyword(aggregate_poll, "sem"),
        _literal_keyword(aggregate_poll, "scope"),
    ) == ("relaxed", "sys")

    kernel = functions["_aggregate_completion_litmus_kernel"]
    relaxed_adds = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_add"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "local_arrival"
    ]
    assert len(relaxed_adds) == 1
    assert (
        _literal_keyword(relaxed_adds[0], "sem"),
        _literal_keyword(relaxed_adds[0], "scope"),
    ) == ("relaxed", "gpu")
    release_swaps = {
        ast.unparse(node.args[0]): node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_xchg"
        and node.args
    }
    assert set(release_swaps) == {
        "local_ready_flag",
        "local_completion_gate",
        "local_aggregate_flag",
        "final_gate",
    }
    for pointer, expected_scope in (
        ("local_ready_flag", "sys"),
        ("local_completion_gate", "gpu"),
        ("local_aggregate_flag", "sys"),
        ("final_gate", "gpu"),
    ):
        call = release_swaps[pointer]
        assert _literal_keyword(call, "sem") == "release"
        assert _literal_keyword(call, "scope") == expected_scope
    controller_branches = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "tile == 0"
    ]
    assert len(controller_branches) == 1
    controller_source = ast.unparse(controller_branches[0])
    assert "_poll_local_completion_acquire" in controller_source
    assert "_poll_aggregate_completion_relaxed" in controller_source
    assert "gl.atomic_xchg(final_gate" in controller_source
    assert "_poll_ready_acquire" not in controller_source
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel" not in imported_modules


def test_aggregate_completion_schedule_and_payload_model() -> None:
    """Cover all 21 tiles, 128 words, eight ranks, and exact ticket reuse."""

    producer_words = {
        subgroup * 16 + subgroup_lane
        for subgroup in range(_SUBGROUPS)
        for subgroup_lane in range(16)
    }
    assert producer_words == set(range(_WORDS_PER_TILE))
    for generation in (1, 2, 4096):
        for tile in range(_NUM_TILES):
            for word in (0, 63, 64, 127):
                values = tuple(
                    _expected_payload(generation, rank, tile, word)
                    for rank in range(_WORLD_SIZE)
                )
                assert len(set(values)) == _WORLD_SIZE
                assert sum(values) == _expected_reduced(generation, tile, word)
        tickets = range(
            (generation - 1) * _NUM_TILES,
            generation * _NUM_TILES,
        )
        assert sum(ticket % _NUM_TILES == _NUM_TILES - 1 for ticket in tickets) == 1


def test_aggregate_completion_retained_fresh_dispatch_contract() -> None:
    """Keep the 4096-dispatch, no-JIT, controller-only reuse proof visible."""

    source = inspect.getsource(_worker_main)
    assert "patch.object" in source
    assert '"_do_compile"' in source
    assert "for iteration in range(iterations)" in source
    assert "iteration_values[iteration]" in source
    assert "torch.cuda.synchronize(device)" in source
    assert "dist.barrier()" in source
    assert "_aggregate_completion_litmus_kernel[(_NUM_TILES,)]" in source
    runner_source = inspect.getsource(_run_world8_litmus)
    assert "default_iterations=4096" in runner_source


def test_amd_aggregate_completion_world1_smoke() -> None:
    _run_litmus(
        1,
        _SMOKE_GATE_ENV,
        default_iterations=16,
    )


def test_amd_aggregate_completion_world8_retained_binary() -> None:
    _run_world8_litmus()
