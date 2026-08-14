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

"""Opt-in gfx950 sticky-fatal entry-poll fast-path litmus.

This test deliberately does not import or modify the raw MegaMoE helper.  It
isolates the proposed ``RELAXED_NO_POST`` entry protocol:

* exact-poll process-lifetime sticky fatal state for zero with a system-scope
  relaxed atomic load and a zero timeout;
* use ``atomic_poll``'s built-in LDS result broadcast/workgroup rendezvous;
* issue no acquire invalidation, source VMEM wait, or second workgroup barrier;
* let all eight subgroups of all 240 workgroups record healthy participation;
  and
* capture two guarded nodes with a system-release poison between them, proving
  that the second node and every later replay leave application state intact.

The single-GPU gate uses an ordinary local fatal allocation.  The world-eight
gate uses one Iris symmetric fatal slot per rank and makes every rank publish
the same sticky generation to every peer between the two guarded graph nodes.
Both modes retain one compiled guard binary across all fresh launches/replays.

Run the focused single-GPU qualification with the pinned system ROCr, for
example::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_fatal_entry_fastpath.py

Run the eight-rank symmetric qualification with::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_WORLD8=1 \
      pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_fatal_entry_fastpath.py

Fresh-launch iterations default to 4096 and sticky graph replays default to 64.
"""

from __future__ import annotations

import gc
import hashlib
import json
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

_SINGLE_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH"
_WORLD8_GATE_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_WORLD8"
_ITERATIONS_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_ITERATIONS"
_GRAPH_REPLAYS_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_REPLAYS"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_WATCHDOG_SECONDS"
_DUMP_AMDGCN_ENV = "TOKENSPEED_TEST_MEGAMOE_FATAL_ENTRY_FASTPATH_DUMP_AMDGCN_DIR"
_PROGRAMS = 240
_SUBGROUPS = 8
_LANES = 512
_POISON_GENERATION = 1
_KERNEL_SUBGROUPS = gl.constexpr(_SUBGROUPS)
_KERNEL_LANES = gl.constexpr(_LANES)
_KERNEL_POISON_GENERATION = gl.constexpr(_POISON_GENERATION)


@gluon.jit
def _fatal_epoch_is_clear_relaxed_no_post(fatal_epoch):
    """Exact-check sticky fatal control without acquiring unrelated payload."""

    # A nonzero sticky value carries no payload that the guarded node consumes.
    # System scope is retained for remote poison visibility; relaxed semantics
    # avoid BUFFER_INV, and atomic_poll already broadcasts its scalar result
    # through LDS before its built-in workgroup rendezvous returns.
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
def _fatal_entry_relaxed_no_post_kernel(fatal_epoch, visits, node_output):
    """Record one healthy entry in every subgroup, or return without writes."""

    if not _fatal_epoch_is_clear_relaxed_no_post(fatal_epoch):
        return

    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [_KERNEL_SUBGROUPS], [0])
    lane = gl.arange(0, _KERNEL_LANES, layout=layout)
    subgroup = lane // 64
    subgroup_leader = lane % 64 == 0
    slot = pid * _KERNEL_SUBGROUPS + subgroup
    gl.atomic_add(
        visits + slot,
        1,
        mask=subgroup_leader,
        sem="relaxed",
        scope="gpu",
    )
    gl.store(
        node_output + slot,
        1,
        mask=subgroup_leader,
        cache_modifier=".wt",
    )


@gluon.jit
def _poison_local_fatal_kernel(fatal_epoch):
    """System-release one sticky generation to this rank's fatal slot."""

    gl.atomic_xchg(
        fatal_epoch,
        _KERNEL_POISON_GENERATION,
        sem="release",
        scope="sys",
    )


@gluon.jit
def _poison_all_symmetric_fatal_kernel(
    fatal_epoch,
    heap_bases,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
):
    """System-release one sticky generation to every symmetric fatal slot."""

    local_heap = gl.load(heap_bases + RANK).to(gl.uint64, bitcast=True)
    byte_offset = tl.cast(fatal_epoch, gl.uint64) - local_heap
    for peer in gl.static_range(0, WORLD_SIZE):
        peer_heap = gl.load(heap_bases + peer).to(gl.uint64, bitcast=True)
        peer_fatal = tl.cast(
            peer_heap + byte_offset,
            gl.pointer_type(gl.int64),
        )
        gl.atomic_xchg(
            peer_fatal,
            _KERNEL_POISON_GENERATION,
            sem="release",
            scope="sys",
        )


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _assert_loaded_relaxed_no_post_isa(
    assembly: str,
    *,
    rank: int,
    world_size: int,
) -> str:
    """Prove exact relaxed system polling and the sole LDS rendezvous."""

    isa_sha256 = hashlib.sha256(assembly.encode("utf-8")).hexdigest()
    if dump_dir := os.getenv(_DUMP_AMDGCN_ENV):
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"world{world_size}-rank{rank}-{isa_sha256}.amdgcn").write_text(
            assembly,
            encoding="utf-8",
        )

    # Timeout zero still executes one exact coherent system-scope load.  The
    # constant timeout folds away the duration comparison, so prove the exact
    # equality against zero directly rather than matching the nonzero-timeout
    # loop shape used by the ready/completion publication litmuses.  Relaxed
    # semantics must not emit BUFFER_INV.
    poll_graphs = list(
        re.finditer(
            r"\bs_memrealtime\b"
            r"(?:(?!\bs_memrealtime\b).)*?"
            r"\bglobal_load_dwordx2\b[^\n]*\bsc0 sc1\b"
            r"(?:(?!\bds_write_b8\b).)*?"
            r"\bv_cmp_eq_u64(?:_e(?:32|64))?\b[^\n]*\b0,\s*v\["
            r"(?:(?!\bds_write_b8\b).)*?"
            r"\bs_memrealtime\b"
            r"(?:(?!\bds_write_b8\b).)*?"
            r"\bds_write_b8\b"
            r"(?:(?!\bs_barrier\b).)*?\bs_barrier\b"
            r"(?:(?!\bds_read_u8\b).)*?\bds_read_u8\b",
            assembly,
            flags=re.DOTALL,
        )
    )
    assert len(poll_graphs) == 1
    assert assembly.count("s_memrealtime") == 2
    assert assembly.count("global_load_dwordx2") == 1
    assert assembly.count("ds_write_b8") == 1
    assert assembly.count("ds_read_u8") == 1
    assert assembly.count("s_barrier") == 1
    assert "buffer_inv" not in assembly
    assert ";;#ASMSTART" not in assembly.upper()

    # The first application-state operation follows the poll result without a
    # source-owned VMEM wait or second workgroup rendezvous.
    after_poll = assembly[poll_graphs[0].end() :]
    first_application_op = re.search(
        r"\b(?:global_atomic_add|global_store_dword)\b",
        after_poll,
    )
    assert first_application_op is not None
    prefix = after_poll[: first_application_op.start()]
    assert "s_barrier" not in prefix
    assert re.search(r"\bs_waitcnt\b[^\n]*\bvmcnt\(", prefix) is None
    assert ";;#ASMSTART" not in prefix.upper()
    return isa_sha256


def _guard_launch(
    fatal_epoch: torch.Tensor,
    visits: torch.Tensor,
    node_output: torch.Tensor,
) -> None:
    _fatal_entry_relaxed_no_post_kernel[(_PROGRAMS,)](
        fatal_epoch,
        visits,
        node_output,
        num_warps=_SUBGROUPS,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )


def _run_healthy_fresh_launches(
    device: torch.device,
    fatal_epoch: torch.Tensor,
    *,
    iterations: int,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Replay one retained guard binary over healthy fresh dispatches."""

    visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS),
        dtype=torch.int32,
        device=device,
    )
    node_output = torch.full_like(visits, -1)
    compiled = _fatal_entry_relaxed_no_post_kernel.warmup(
        fatal_epoch,
        visits,
        node_output,
        grid=(_PROGRAMS,),
        num_warps=_SUBGROUPS,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=True,
    )
    device_index = device.index if device.index is not None else 0
    architecture = str(
        getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    )
    if not architecture.startswith("gfx950"):
        raise AssertionError(f"fatal-entry litmus requires gfx950, got {architecture}")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    isa_sha256 = _assert_loaded_relaxed_no_post_isa(
        compiled.asm["amdgcn"],
        rank=rank,
        world_size=world_size,
    )
    hsaco_sha256 = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    retained_identity = _loaded_kernel_cache_identity(
        _fatal_entry_relaxed_no_post_kernel
    )
    assert retained_identity

    stream = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with patch.object(
        _fatal_entry_relaxed_no_post_kernel,
        "_do_compile",
        side_effect=AssertionError("fresh guard launch attempted to recompile"),
    ):
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(iterations):
                _guard_launch(fatal_epoch, visits, node_output)
            end.record()
        end.synchronize()

    assert retained_identity == _loaded_kernel_cache_identity(
        _fatal_entry_relaxed_no_post_kernel
    )
    torch.testing.assert_close(
        visits,
        torch.full_like(visits, iterations),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        node_output,
        torch.ones_like(node_output),
        rtol=0,
        atol=0,
    )
    assert int(fatal_epoch.item()) == 0
    elapsed_ms = start.elapsed_time(end)
    return (
        {
            "healthy_iterations": iterations,
            "healthy_elapsed_ms": elapsed_ms,
            "healthy_us_per_dispatch": elapsed_ms * 1000.0 / iterations,
            "isa_sha256": isa_sha256,
            "hsaco_sha256": hsaco_sha256,
        },
        visits,
        node_output,
    )


def _warmup_poison_kernel(
    fatal_epoch: torch.Tensor,
    heap_bases: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    if heap_bases is None:
        _poison_local_fatal_kernel.warmup(
            fatal_epoch,
            grid=(1,),
            num_warps=1,
            num_stages=1,
        )
        return
    _poison_all_symmetric_fatal_kernel.warmup(
        fatal_epoch,
        heap_bases,
        RANK=rank,
        WORLD_SIZE=world_size,
        grid=(1,),
        num_warps=1,
        num_stages=1,
    )


def _poison_launch(
    fatal_epoch: torch.Tensor,
    heap_bases: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    if heap_bases is None:
        _poison_local_fatal_kernel[(1,)](
            fatal_epoch,
            num_warps=1,
            num_stages=1,
        )
        return
    _poison_all_symmetric_fatal_kernel[(1,)](
        fatal_epoch,
        heap_bases,
        RANK=rank,
        WORLD_SIZE=world_size,
        num_warps=1,
        num_stages=1,
    )


def _run_two_node_sticky_graph(
    device: torch.device,
    fatal_epoch: torch.Tensor,
    heap_bases: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    replays: int,
) -> dict[str, object]:
    """Capture healthy node -> poison -> guarded node, then replay sticky state."""

    first_visits = torch.zeros(
        (_PROGRAMS, _SUBGROUPS),
        dtype=torch.int32,
        device=device,
    )
    first_output = torch.full_like(first_visits, -1)
    second_visits = torch.zeros_like(first_visits)
    second_output = torch.full_like(first_output, -2)
    _warmup_poison_kernel(
        fatal_epoch,
        heap_bases,
        rank=rank,
        world_size=world_size,
    )

    stream = torch.cuda.Stream(device=device)
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with patch.object(
        _fatal_entry_relaxed_no_post_kernel,
        "_do_compile",
        side_effect=AssertionError("sticky graph guard attempted to recompile"),
    ):
        with torch.cuda.graph(graph, stream=stream):
            _guard_launch(fatal_epoch, first_visits, first_output)
            _poison_launch(
                fatal_epoch,
                heap_bases,
                rank=rank,
                world_size=world_size,
            )
            _guard_launch(fatal_epoch, second_visits, second_output)
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()

    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    assert int(fatal_epoch.item()) == _POISON_GENERATION
    if world_size == 1:
        torch.testing.assert_close(
            first_visits, torch.ones_like(first_visits), rtol=0, atol=0
        )
        torch.testing.assert_close(
            first_output, torch.ones_like(first_output), rtol=0, atol=0
        )
    else:
        # Every rank orders its first guard before its poison, but a faster
        # peer may poison this rank while that first cooperative grid is still
        # running.  Partial first-plan state is therefore the fail-stop skew
        # being exercised, not a failure: each workgroup either records all of
        # its subgroup leaders or returns before any application-state write.
        assert bool(torch.all((first_visits == 0) | (first_visits == 1)).item())
        assert bool(torch.all(first_visits == first_visits[:, :1]).item())
        expected_first_output = torch.where(
            first_visits != 0,
            torch.ones_like(first_output),
            torch.full_like(first_output, -1),
        )
        torch.testing.assert_close(first_output, expected_first_output, rtol=0, atol=0)
    torch.testing.assert_close(
        second_visits, torch.zeros_like(second_visits), rtol=0, atol=0
    )
    torch.testing.assert_close(
        second_output, torch.full_like(second_output, -2), rtol=0, atol=0
    )
    first_snapshot = first_visits.clone()
    first_output_snapshot = first_output.clone()
    second_snapshot = second_visits.clone()
    second_output_snapshot = second_output.clone()

    # The first replay establishes poison. Every subsequent replay must abort
    # both guarded nodes before either touches its distinct application state.
    with torch.cuda.stream(stream):
        for _ in range(replays - 1):
            graph.replay()
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    assert int(fatal_epoch.item()) == _POISON_GENERATION
    torch.testing.assert_close(first_visits, first_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(first_output, first_output_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(second_visits, second_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(second_output, second_output_snapshot, rtol=0, atol=0)
    return {
        "sticky_graph_replays": replays,
        "fatal_generation": int(fatal_epoch.item()),
        "first_node_visits": int(first_visits.sum().item()),
        "second_node_visits": int(second_visits.sum().item()),
    }


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    iterations: int,
    replays: int,
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
        if world_size == 1:
            fatal_epoch = torch.zeros(1, dtype=torch.int64, device=device)
            heap_bases = None
        else:
            import iris

            shmem = iris.iris(1 << 20)
            assert shmem.get_rank() == rank
            assert shmem.get_num_ranks() == world_size
            fatal_epoch = shmem.zeros(1, dtype=torch.int64, device=device)
            heap_bases = shmem.get_heap_bases()

        healthy, _, _ = _run_healthy_fresh_launches(
            device,
            fatal_epoch,
            iterations=iterations,
        )
        sticky = _run_two_node_sticky_graph(
            device,
            fatal_epoch,
            heap_bases,
            rank=rank,
            world_size=world_size,
            replays=replays,
        )
        report = {"rank": rank, "world_size": world_size, **healthy, **sticky}
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, report)
        if rank == 0:
            assert all(item is not None for item in gathered)
            for item in gathered:
                assert item is not None
                assert item["healthy_iterations"] == iterations
                assert item["fatal_generation"] == _POISON_GENERATION
                assert item["second_node_visits"] == 0
            first_node_visits = sum(
                int(item["first_node_visits"]) for item in gathered if item is not None
            )
            if world_size == 1:
                assert first_node_visits == _PROGRAMS * _SUBGROUPS
            else:
                # The rank issuing the globally earliest poison necessarily
                # completed its stream-ordered first guard before that poison.
                assert first_node_visits >= _PROGRAMS * _SUBGROUPS
                assert first_node_visits <= world_size * _PROGRAMS * _SUBGROUPS
            print(
                "MegaMoE fatal-entry RELAXED_NO_POST reports:",
                json.dumps(gathered),
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


def _run_hardware_litmus(world_size: int) -> None:
    gate_env = _SINGLE_GATE_ENV if world_size == 1 else _WORLD8_GATE_ENV
    if not _env_enabled(gate_env):
        pytest.skip(f"set {gate_env}=1 for the gfx950 fatal-entry fast-path litmus")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"fatal-entry litmus requires {world_size} visible GPUs")
    properties = tuple(
        torch.cuda.get_device_properties(index) for index in range(world_size)
    )
    architectures = tuple(
        str(getattr(device, "gcnArchName", "")) for device in properties
    )
    if any(not architecture.startswith("gfx950") for architecture in architectures):
        pytest.skip(f"all visible devices must be gfx950, got {architectures}")
    if any(
        int(getattr(device, "multi_processor_count", 0)) != 256 for device in properties
    ):
        pytest.skip("fatal-entry litmus requires the 256-CU gfx950 target")
    if "libhsa-runtime64.so" not in os.getenv("LD_PRELOAD", ""):
        pytest.skip("fatal-entry litmus requires the pinned system ROCr")
    if world_size > 1:
        try:
            import iris  # noqa: F401
        except ImportError:
            pytest.skip("world-eight fatal-entry litmus requires Iris")

    iterations = int(os.getenv(_ITERATIONS_ENV, "4096"))
    replays = int(os.getenv(_GRAPH_REPLAYS_ENV, "64"))
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "240"))
    if iterations <= 0:
        raise ValueError(f"{_ITERATIONS_ENV} must be positive")
    if replays <= 0:
        raise ValueError(f"{_GRAPH_REPLAYS_ENV} must be positive")
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")

    context = mp.spawn(
        _worker_main,
        args=(world_size, _open_port(), iterations, replays),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                _terminate_processes(context)
                pytest.fail(
                    f"world{world_size} fatal-entry fast-path litmus exceeded "
                    f"the {watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


def test_amd_fatal_entry_relaxed_no_post_source_contract() -> None:
    """Keep the ablation isolated from raw MegaMoE and free of post-poll sync."""

    source = _fatal_epoch_is_clear_relaxed_no_post.src
    assert "gl.atomic_poll(" in source
    assert 'sem="relaxed"' in source
    assert 'scope="sys"' in source
    assert "timeout_ns=0" in source
    assert "buffer_inv" not in source
    assert "inline_asm" not in source
    assert "barrier" not in source


def test_amd_fatal_entry_relaxed_no_post_gpu0() -> None:
    """Stress healthy fresh dispatch and sticky two-node replay on GPU0."""

    _run_hardware_litmus(1)


def test_amd_fatal_entry_relaxed_no_post_world8() -> None:
    """Repeat the sticky protocol with symmetric all-rank poison publication."""

    _run_hardware_litmus(8)
