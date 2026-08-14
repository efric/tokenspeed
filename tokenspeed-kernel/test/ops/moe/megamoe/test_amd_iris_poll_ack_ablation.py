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

"""Opt-in gfx950 Iris system-poll acknowledgment ablation.

This test-only kernel isolates the two split Iris epoch planes used by MegaMoE.
It changes one thing relative to the qualified drain-only publication litmus:
there is no source-owned ``s_waitcnt vmcnt(0)`` plus workgroup barrier after an
exact system-acquire ``atomic_poll``.  The compiler-owned acquire invalidate and
the poll's built-in workgroup rendezvous remain intact.

The ablation is deliberately narrower than a general polling primitive:

* ``atomic_poll`` elects lane 0 of subgroup 0 for its scalar flag load;
* only subgroup 0 owns the coherent payload loads guarded by ready polls;
* only a later dispatch can overwrite payload after completion polls; and
* producer publication and pre-completion payload-load acknowledgment retain
  their existing per-subgroup VMEM drain plus workgroup barrier.

There is no wait-control tensor or masked diagnostic store between a poll and
the payload.  Such instrumentation can add compiler waits and would make the
loaded-ISA question circular.  Runtime checking instead accumulates exact
payload mismatches after the payload loads and records one status per fresh
dispatch after all completion polls.

World size 1 is only a compile/smoke prequalification.  World size 8 is the
cross-NUMA evidence.  Both modes replay one retained loaded binary across fresh
dispatches, and world size 8 injects rank-dependent host launch skew::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION=8 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/\
test_amd_iris_poll_ack_ablation.py

Iterations default to 4096.  This file does not alter the production raw
kernel, admission hashes, or the implementation-complete marker.
"""

from __future__ import annotations

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


_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION_ITERATIONS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION_TIMEOUT_NS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION_WATCHDOG_SECONDS"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION_DUMP_AMDGCN_DIR"
_BURST_SKEW_US_ENV = "TOKENSPEED_TEST_MEGAMOE_IRIS_POLL_ACK_ABLATION_BURST_SKEW_US"
_LANES = 512
_PAYLOAD_WORDS = 128
_KERNEL_LANES = gl.constexpr(_LANES)


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
def _poll_exact_generation_no_source_ack(
    flag,
    generation,
    TIMEOUT_NS: gl.constexpr,
):
    """Acquire one exact generation with only atomic_poll's own rendezvous."""

    return (
        gl.atomic_poll(
            flag,
            generation,
            sem="acquire",
            scope="sys",
            timeout_ns=TIMEOUT_NS,
        )
        != 0
    )


@gluon.jit
def _iris_poll_ack_ablation_kernel(
    payload,
    generation_flags,
    payload_error_count,
    protocol_error_count,
    generation_status,
    heap_bases,
    start_iteration_ptr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Stress split ready/completion planes without post-poll source ACKs."""

    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup_lane = lane % 64
    consumer_lane = lane < 64
    local_heap = gl.load(heap_bases + RANK).to(gl.uint64, bitcast=True)
    payload_offset = tl.cast(payload, gl.uint64) - local_heap
    flag_offset = tl.cast(generation_flags, gl.uint64) - local_heap
    local_ready_flag = generation_flags + RANK
    local_completion_flag = generation_flags + WORLD_SIZE + RANK
    iteration = gl.load(start_iteration_ptr)
    generation = iteration + 1

    # All eight subgroups retain the production producer shape. Write-through
    # stores plus one drain in every subgroup precede the ready release.
    word = (lane // 64) * 16 + subgroup_lane
    expected_local = generation * 1_000_000 + (RANK + 1) * 10_000 + word.to(gl.int64)
    gl.amd.cdna4.buffer_store(
        expected_local,
        payload,
        word.to(gl.int32),
        mask=subgroup_lane < 16,
        cache=".wt",
    )
    _drain_subgroup_vmem()
    gl.barrier()

    previous_ready = gl.atomic_xchg(
        local_ready_flag,
        generation,
        sem="release",
        scope="sys",
    )
    protocol_ok = previous_ready == iteration

    # Every subgroup executes each scalar poll because atomic_poll contains its
    # own workgroup rendezvous. The scalar election is lane 0 of subgroup 0, the
    # same subgroup that owns every dependent payload load below. Do not add
    # diagnostics, a VMEM ACK, or a second barrier in this interval.
    all_ready = _poll_exact_generation_no_source_ack(
        local_ready_flag,
        generation,
        TIMEOUT_NS,
    )
    for distance in gl.static_range(1, WORLD_SIZE):
        peer = RANK ^ distance
        peer_heap = gl.load(heap_bases + peer).to(gl.uint64, bitcast=True)
        peer_flags = tl.cast(
            peer_heap + flag_offset,
            gl.pointer_type(gl.int64),
        )
        all_ready &= _poll_exact_generation_no_source_ack(
            peer_flags + peer,
            generation,
            TIMEOUT_NS,
        )
    protocol_ok &= all_ready

    # This is the first memory operation after the final ready poll. Only the
    # first subgroup's 64 lanes consume two words apiece, in canonical rank
    # order, with the same coherent cache policy as production MegaMoE.
    lane_mismatches = gl.zeros([_KERNEL_LANES], gl.int32, layout=layout)
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            peer_payload = payload
        else:
            peer_heap = gl.load(heap_bases + peer).to(
                gl.uint64,
                bitcast=True,
            )
            peer_payload = tl.cast(
                peer_heap + payload_offset,
                gl.pointer_type(gl.int64),
            )
        for word_group in gl.static_range(0, 2):
            payload_word = lane + word_group * 64
            observed = gl.amd.cdna4.buffer_load(
                peer_payload,
                payload_word.to(gl.int32),
                mask=consumer_lane,
                other=-1,
                cache=".cv",
            )
            expected_peer = (
                generation * 1_000_000 + (peer + 1) * 10_000 + payload_word.to(gl.int64)
            )
            lane_mismatches += gl.where(
                consumer_lane & (observed != expected_peer),
                1,
                0,
            )

    gl.atomic_add(
        payload_error_count + lane * 0,
        lane_mismatches,
        mask=consumer_lane & (lane_mismatches != 0),
        sem="relaxed",
        scope="gpu",
    )

    # Retain the production load acknowledgment before completion publication.
    # This is not a post-poll ACK: it orders payload consumption before reuse.
    _drain_subgroup_vmem()
    gl.barrier()
    previous_completion = gl.atomic_xchg(
        local_completion_flag,
        generation,
        sem="release",
        scope="sys",
    )
    protocol_ok &= previous_completion == iteration

    # Completion waits guard reuse by a later dispatch on the same stream. They
    # carry no same-dispatch payload dependency and retain only atomic_poll's
    # built-in workgroup rendezvous.
    all_complete = gl.full((), True, gl.int1)
    for distance in gl.static_range(1, WORLD_SIZE):
        peer = RANK ^ distance
        peer_heap = gl.load(heap_bases + peer).to(gl.uint64, bitcast=True)
        peer_flags = tl.cast(
            peer_heap + flag_offset,
            gl.pointer_type(gl.int64),
        )
        all_complete &= _poll_exact_generation_no_source_ack(
            peer_flags + WORLD_SIZE + peer,
            generation,
            TIMEOUT_NS,
        )
    protocol_ok &= all_complete

    # End-only instrumentation cannot add an acknowledgment between a poll and
    # its dependent payload access. The host requires every fresh dispatch to
    # overwrite its initially negative slot with exactly one.
    gl.store(
        generation_status + iteration + lane * 0,
        gl.where(protocol_ok, 1, 0),
        mask=lane == 0,
        cache_modifier=".wt",
    )
    gl.atomic_add(
        protocol_error_count + lane * 0,
        1,
        mask=(lane == 0) & (not protocol_ok),
        sem="relaxed",
        scope="gpu",
    )


def _selected_world_sizes() -> set[int]:
    raw = os.getenv(_GATE_ENV, "")
    try:
        selected = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as error:
        raise ValueError(
            f"{_GATE_ENV} must be a comma-separated subset of 1,8"
        ) from error
    if not selected.issubset({1, 8}):
        raise ValueError(f"{_GATE_ENV} must be a comma-separated subset of 1,8")
    return selected


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _loaded_kernel_cache_identity(
    kernel,
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    """Identify the exact per-device runners retained by a Gluon function."""

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


def _require_parent_gate(world_size: int) -> None:
    if world_size not in _selected_world_sizes():
        pytest.skip(f"include {world_size} in {_GATE_ENV} to run this litmus")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"the poll-ACK ablation requires {world_size} GPUs")
    architectures = tuple(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        for index in range(world_size)
    )
    if any(not architecture.startswith("gfx950") for architecture in architectures):
        pytest.skip(f"all {world_size} devices must be gfx950, got {architectures}")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("the poll-ACK ablation requires system ROCr in LD_PRELOAD")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _assert_loaded_poll_ack_ablation_isa(
    assembly: str,
    world_size: int,
) -> str:
    """Prove the exact no-source-ACK path in the loaded gfx950 object."""

    isa_sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / f"world{world_size}-{isa_sha256}.amdgcn").write_text(assembly)

    wait_sites = 2 * world_size - 1
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

    # Source-owned drains remain only at producer publication and between the
    # payload loads and completion release. In particular, neither ready nor
    # completion polling has a third ACK/barrier sequence hidden in inline asm.
    explicit_drains = list(
        re.finditer(
            r";;#ASMSTART\s+s_waitcnt vmcnt\(0\)\s+;;#ASMEND",
            assembly,
        )
    )
    assert len(explicit_drains) == 2, len(explicit_drains)
    assert assembly.count(";;#ASMSTART") == 2
    assert assembly.count(";;#ASMEND") == 2

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
        if (offset := re.search(r"offset:(\d+)", graph.group("swap")))
        else 0
        for graph in release_graphs
    )
    assert release_offsets[1] - release_offsets[0] == world_size * 8

    payload_loads = list(
        re.finditer(
            r"\bbuffer_load_dwordx2\b[^\n]*\bsc0 sc1\b",
            assembly,
        )
    )
    assert len(payload_loads) == 2 * world_size

    ready_polls = poll_graphs[:world_size]
    completion_polls = poll_graphs[world_size:]
    assert explicit_drains[0].end() < release_graphs[0].start()
    assert release_graphs[0].end() < ready_polls[0].start()
    # LLVM may schedule the independent payload loads after atomic_poll's
    # acquire invalidate and internal barrier but before its LDS result read.
    # The latter feeds only end-of-dispatch diagnostics, not payload addresses.
    # Therefore order against the final ready poll's invalidate, rather than
    # treating the compiler-owned result read as part of the memory dependency.
    last_ready_invalidate = ready_polls[-1].start() + ready_polls[-1].group(0).index(
        "buffer_inv sc0 sc1"
    )
    assert last_ready_invalidate < payload_loads[0].start()
    assert payload_loads[-1].end() < explicit_drains[1].start()
    assert explicit_drains[1].end() < release_graphs[1].start()
    if completion_polls:
        assert release_graphs[1].end() < completion_polls[0].start()

    ready_to_payload = assembly[release_graphs[0].end() : payload_loads[0].start()]
    completion_tail = assembly[release_graphs[1].end() :]
    assert ";;#ASMSTART" not in ready_to_payload
    assert "global_store" not in ready_to_payload
    assert ";;#ASMSTART" not in completion_tail

    # A scalar atomic_poll has every lane/subgroup dimension redundant. The AMD
    # lowering elects workgroup lane 0: the loaded ISA must retain both the
    # lane-within-subgroup and subgroup-base predicates. The payload mask then
    # selects lanes [0, 64), so the acquire flag load/invalidate and coherent
    # payload loads are owned by the same subgroup.
    lane_id = re.search(
        r"\bv_and_b32_e32 v(?P<lane>\d+), 63, v0\b",
        assembly,
    )
    subgroup_base = re.search(
        r"\bv_readfirstlane_b32 s(?P<base>\d+), v0\b",
        assembly,
    )
    assert lane_id
    assert subgroup_base
    subgroup_id = re.search(
        rf"\bs_lshr_b32 s(?P<subgroup>\d+), "
        rf"s{subgroup_base.group('base')}, 6\b",
        assembly,
    )
    assert subgroup_id
    election = re.search(
        rf"\bv_or_b32_e32 v(?P<linear>\d+), "
        rf"s{subgroup_id.group('subgroup')}, v{lane_id.group('lane')}\b"
        rf"(?:(?!\bv_cmp_eq_u32).)*?"
        rf"\bv_cmp_eq_u32_e64 s\[(?P<mask_lo>\d+):(?P<mask_hi>\d+)\], "
        rf"0, v(?P=linear)\b",
        assembly,
        flags=re.DOTALL,
    )
    assert election
    election_mask = f"s[{election.group('mask_lo')}:{election.group('mask_hi')}]"
    for poll in poll_graphs:
        poll_prefix = assembly[max(0, poll.start() - 1024) : poll.start()]
        assert re.search(
            rf"\bs_and_saveexec_b64 s\[\d+:\d+\], "
            rf"{re.escape(election_mask)}",
            poll_prefix,
        )
    payload_prefix = assembly[ready_polls[-1].start() : payload_loads[0].start()]
    assert re.search(
        rf"\bs_cmp_eq_u32 s{subgroup_id.group('subgroup')}, 0\b",
        payload_prefix,
    )

    # This region is the essential ablation: an exact scalar system flag load,
    # successful acquire invalidate, compiler-owned LDS rendezvous, then the
    # first coherent payload load, with no source-owned ACK in between.
    last_ready_to_payload = assembly[ready_polls[-1].start() : payload_loads[0].end()]
    assert re.search(
        r"\bglobal_load_dwordx2\b[^\n]*\bsc0 sc1\b"
        r"(?:(?!;;#ASMSTART).)*?"
        r"\bbuffer_inv sc0 sc1\b"
        r"(?:(?!;;#ASMSTART).)*?"
        r"\bbuffer_load_dwordx2\b[^\n]*\bsc0 sc1\b",
        last_ready_to_payload,
        flags=re.DOTALL,
    )

    assert assembly.count("buffer_wbl2 sc0 sc1") == 2
    assert "global_atomic_add_x2" not in assembly
    assert "global_atomic_cmpswap_x2" not in assembly
    return isa_sha256


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    iterations: int,
    timeout_ns: int,
    burst_skew_us: int,
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
        payload_error_count = torch.zeros(1, dtype=torch.int32, device=device)
        protocol_error_count = torch.zeros(1, dtype=torch.int32, device=device)
        generation_status = torch.full(
            (iterations,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        heap_bases = shmem.get_heap_bases()
        iteration_index = torch.arange(iterations, dtype=torch.int64, device=device)
        iteration_values = torch.stack(
            (iteration_index, torch.zeros_like(iteration_index)),
            dim=1,
        )

        compiled = _iris_poll_ack_ablation_kernel.warmup(
            payload,
            generation_flags,
            payload_error_count,
            protocol_error_count,
            generation_status,
            heap_bases,
            iteration_values[:1],
            RANK=rank,
            WORLD_SIZE=world_size,
            TIMEOUT_NS=timeout_ns,
            grid=(1,),
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
        isa_sha256 = _assert_loaded_poll_ack_ablation_isa(
            compiled.asm["amdgcn"],
            world_size,
        )
        hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
        retained_cache_identity = _loaded_kernel_cache_identity(
            _iris_poll_ack_ablation_kernel
        )
        assert retained_cache_identity
        retained_runner_ids = tuple(
            runner_id
            for _, cache_entries in retained_cache_identity
            for _, runner_id in cache_entries
        )
        assert len(retained_runner_ids) == 1

        queue_prime = torch.zeros(1, dtype=torch.int32, device=device)
        queue_prime.add_(1)
        torch.cuda.synchronize(device)
        assert int(queue_prime.item()) == 1
        dist.barrier()

        start_ns = time.monotonic_ns()
        with patch.object(
            _iris_poll_ack_ablation_kernel,
            "_do_compile",
            side_effect=AssertionError("generation replay attempted to recompile"),
        ):
            for start_iteration in range(iterations):
                if rank and start_iteration % 64 == 0:
                    time.sleep(rank * burst_skew_us / 1_000_000)
                _iris_poll_ack_ablation_kernel[(1,)](
                    payload,
                    generation_flags,
                    payload_error_count,
                    protocol_error_count,
                    generation_status,
                    heap_bases,
                    iteration_values[start_iteration],
                    RANK=rank,
                    WORLD_SIZE=world_size,
                    TIMEOUT_NS=timeout_ns,
                    num_warps=8,
                    num_stages=1,
                    waves_per_eu=2,
                )
            torch.cuda.synchronize(device)
        elapsed_ns = time.monotonic_ns() - start_ns

        assert retained_cache_identity == _loaded_kernel_cache_identity(
            _iris_poll_ack_ablation_kernel
        )
        local_flags = generation_flags.cpu()
        statuses = generation_status.cpu()
        report = {
            "rank": rank,
            "world_size": world_size,
            "iterations": iterations,
            "payload_errors": int(payload_error_count.item()),
            "protocol_errors": int(protocol_error_count.item()),
            "bad_status_indices": tuple(
                torch.nonzero(statuses != 1).flatten().tolist()[:16]
            ),
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
            "elapsed_ns": elapsed_ns,
            "isa_sha256": isa_sha256,
            "hsaco_sha256": hsaco_sha256,
            "retained_runner_ids": retained_runner_ids,
        }
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, report)
        if rank == 0:
            assert all(item is not None for item in gathered)
            print("Iris poll-ACK ablation rank reports:", gathered, flush=True)
            for item in gathered:
                assert item is not None
                assert item["payload_errors"] == 0, item
                assert item["protocol_errors"] == 0, item
                assert not item["bad_status_indices"], item
                assert item["final_ready_generation"] == iterations, item
                assert item["final_completion_generation"] == iterations, item
                assert not any(item["untouched_ready_slots"]), item
                assert not any(item["untouched_completion_slots"]), item
                assert len(item["isa_sha256"]) == 64, item
                assert len(item["hsaco_sha256"]) == 64, item
                assert len(item["retained_runner_ids"]) == 1, item
            print(
                "MegaMoE Iris poll-ACK ablation PASS:",
                {
                    "world_size": world_size,
                    "iterations": iterations,
                    "burst_skew_us": burst_skew_us,
                    "remote_word_checks": iterations
                    * world_size
                    * (world_size - 1)
                    * _PAYLOAD_WORDS,
                    "self_word_checks": iterations * world_size * _PAYLOAD_WORDS,
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


def _run_litmus(world_size: int) -> None:
    _require_parent_gate(world_size)
    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    timeout_ns = int(os.getenv(_TIMEOUT_ENV, "1000000000"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "240"))
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
            burst_skew_us,
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
                    f"world{world_size} Iris poll-ACK ablation exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


def test_amd_iris_poll_ack_ablation_source_contract() -> None:
    """Keep the test source isolated from the production/local-poll cases."""

    poll = inspect.getsource(_poll_exact_generation_no_source_ack.fn)
    kernel = inspect.getsource(_iris_poll_ack_ablation_kernel.fn)
    assert poll.count("gl.atomic_poll(") == 1
    assert 'sem="acquire"' in poll
    assert 'scope="sys"' in poll
    assert "_wait_subgroup_vmem_ack" not in poll
    assert "gl.barrier" not in poll
    assert "gl.load" not in poll
    assert "wait_control" not in poll
    assert "wait_control" not in kernel
    assert kernel.count("_drain_subgroup_vmem()") == 2
    assert kernel.count("gl.barrier()") == 2

    ready_release = kernel.index("previous_ready = gl.atomic_xchg(")
    self_ready = kernel.index("all_ready = _poll_exact_generation_no_source_ack(")
    ready_peer = kernel.index(
        "all_ready &= _poll_exact_generation_no_source_ack(",
        self_ready,
    )
    payload_load = kernel.index("observed = gl.amd.cdna4.buffer_load(")
    payload_drain = kernel.index("_drain_subgroup_vmem()", payload_load)
    completion_release = kernel.index("previous_completion = gl.atomic_xchg(")
    completion_peer = kernel.index(
        "all_complete &= _poll_exact_generation_no_source_ack(",
    )
    end_instrumentation = kernel.index("generation_status + iteration")
    assert (
        ready_release
        < self_ready
        < ready_peer
        < payload_load
        < payload_drain
        < completion_release
        < completion_peer
        < end_instrumentation
    )
    ready_to_payload = kernel[ready_release:payload_load]
    assert "_drain_subgroup_vmem" not in ready_to_payload
    assert "gl.barrier" not in ready_to_payload
    assert "gl.store" not in ready_to_payload
    assert "gl.atomic_add" not in ready_to_payload
    assert "mask=consumer_lane" in kernel[payload_load:payload_drain]
    completion_tail = kernel[completion_release:completion_peer]
    assert "_drain_subgroup_vmem" not in completion_tail
    assert "gl.barrier" not in completion_tail


@pytest.mark.parametrize("world_size", [1, 8])
def test_amd_iris_poll_ack_ablation(world_size: int) -> None:
    _run_litmus(world_size)
