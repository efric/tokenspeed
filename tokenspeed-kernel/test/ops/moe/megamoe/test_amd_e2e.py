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

"""Opt-in eight-rank gfx950 MegaMoE integration and fail-stop harness.

Run the successful-generation test only with all eight gfx950 devices reserved::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      TOKENSPEED_TEST_MEGAMOE_8GPU=1 pytest -q -s \
      tokenspeed-kernel/test/ops/moe/megamoe/test_amd_e2e.py

The missing-rank graph-replay fault is a separate opt-in because it deliberately
poisons the process-lifetime Iris lane::

    TOKENSPEED_TEST_MEGAMOE_FAILSTOP=1  # in addition to the variables above

Both tests check the implementation-complete marker in the parent process
before spawning a single worker.  This is intentional: compiling or launching
only some ranks while the raw kernel is incomplete can strand a collective.

This harness is a retained-binary control/liveness integration gate, not a
standalone numerical or publication golden. Its finite/change/cross-rank checks
can miss a coherent but wrong result. Qualification also requires the focused
fresh-launch payload-publication litmuses and comparison with the existing K3
MoE numerical path described in the design document.
"""

from __future__ import annotations

import os
import socket
import time
import traceback
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 8
_LANE_SHAPES = ((1, 7168), (1, 3584))
_TOTAL_LANE_ELEMENTS = 10_752
_PROGRAMS = 240
_COMM_PROGRAMS = 21
_E2E_ENV = "TOKENSPEED_TEST_MEGAMOE_8GPU"
_FAILSTOP_ENV = "TOKENSPEED_TEST_MEGAMOE_FAILSTOP"
_WATCHDOG_ENV = "TOKENSPEED_TEST_MEGAMOE_WATCHDOG_SECONDS"
_TIMEOUT_ENV = "TOKENSPEED_TEST_MEGAMOE_TIMEOUT_NS"
_TIMING_REPLAYS_ENV = "TOKENSPEED_TEST_MEGAMOE_TIMING_REPLAYS"
_QUALIFIED_TIMEOUT_NS = 1_000_000_000


def _writable_state(
    state: dict[str, Any],
    first_plan,
    second_plan,
) -> dict[str, torch.Tensor]:
    """Return every tensor that a post-poison graph node must leave unchanged."""

    writable = {
        f"workspace.{name}": tensor for name, tensor in first_plan.workspace.items()
    }
    writable.update(
        {
            "lane.symmetric_producer": state["lane"].symmetric_producer,
            "lane.symmetric_reduced": state["lane"].symmetric_reduced,
            "lane.iris_epoch_flags": state["lane"].iris_epoch_flags,
            "lane.topology_status": state["lane"].topology_status,
            "lane.fatal_epoch": state["lane"].fatal_epoch,
            "first_plan.layer_output": first_plan.layer_output,
            "second_plan.layer_output": second_plan.layer_output,
        }
    )
    return writable


def _assert_snapshots_equal(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    """Prove that a captured graph node performed no observable tensor write."""

    assert before.keys() == after.keys()
    for name in before:
        before_bytes = before[name].view(torch.uint8)
        after_bytes = after[name].view(torch.uint8)
        if not torch.equal(after_bytes, before_bytes):
            raise AssertionError(f"post-poison graph node changed {name}")


def _snapshot_expert_phase_state(
    state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Capture cumulative expert gates before one successful generation."""

    workspace = state["workspace"]
    return {
        name: workspace[name].detach().cpu().clone()
        for phase in ("w13", "w2")
        for name in (
            f"{phase}_target",
            f"{phase}_arrival",
            f"{phase}_gate",
        )
    }


def _assert_success_phase_state(
    generation: int,
    workspace,
    expert_phase_before: dict[str, torch.Tensor] | None,
) -> None:
    """Validate local generations and the exact expert-grid rendezvous."""

    shared_arrival = workspace["shared_arrival"].cpu()
    shared_gate = workspace["shared_gate"].cpu()
    torch.testing.assert_close(
        shared_arrival,
        torch.full_like(shared_arrival, generation * 16),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        shared_gate,
        torch.full_like(shared_gate, generation),
        atol=0,
        rtol=0,
    )

    route_count = int(workspace["local_route_count"].item())
    assert 0 <= route_count <= 16
    if expert_phase_before is None:
        assert generation == 1
        expert_phase_before = {
            name: torch.zeros_like(workspace[name], device="cpu")
            for phase in ("w13", "w2")
            for name in (
                f"{phase}_target",
                f"{phase}_arrival",
                f"{phase}_gate",
            )
        }

    for phase in ("w13", "w2"):
        arrival_name = f"{phase}_arrival"
        arrival = workspace[arrival_name].cpu()
        target = workspace[f"{phase}_target"].cpu()
        gate = workspace[f"{phase}_gate"].cpu()
        tensors = {
            f"{phase}_target": target,
            arrival_name: arrival,
            f"{phase}_gate": gate,
        }
        if route_count > 0:
            expected = expert_phase_before[arrival_name][0] + 226
            assert int(arrival[0].item()) == int(expected.item())
            assert int(target[0].item()) == int(expected.item())
            assert int(gate[0].item()) == int(expected.item())
            for name, tensor in tensors.items():
                if not torch.equal(tensor[1:], expert_phase_before[name][1:]):
                    raise AssertionError(f"{phase} inactive control slots changed")
        else:
            for name, tensor in tensors.items():
                if not torch.equal(tensor, expert_phase_before[name]):
                    raise AssertionError(f"{phase} L=0 controls changed")


def _poison_symmetric_lane_before_replay(
    state: dict[str, Any],
    device: torch.device,
) -> None:
    """Expose incomplete producer or reduction writes in a graph replay."""

    state["lane"].symmetric_producer.fill_(float("nan"))
    state["lane"].symmetric_reduced.fill_(float("nan"))
    torch.cuda.synchronize(device)
    dist.barrier()


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _qualified_timeout_ns() -> int:
    """Return the only timeout embedded in admitted EP8 specializations."""

    timeout_ns = int(os.getenv(_TIMEOUT_ENV, str(_QUALIFIED_TIMEOUT_NS)))
    if timeout_ns != _QUALIFIED_TIMEOUT_NS:
        raise ValueError(
            f"{_TIMEOUT_ENV} must equal the qualified value {_QUALIFIED_TIMEOUT_NS}"
        )
    return timeout_ns


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _require_parent_gate(*, failstop: bool) -> None:
    if not _env_enabled(_E2E_ENV):
        pytest.skip(f"set {_E2E_ENV}=1 for the eight-rank gfx950 integration test")
    if failstop and not _env_enabled(_FAILSTOP_ENV):
        pytest.skip(
            f"set {_FAILSTOP_ENV}=1 to run the process-poisoning fault injection"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip("the MegaMoE integration test requires eight visible GPUs")

    arches = tuple(
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        for index in range(_WORLD_SIZE)
    )
    if any(not arch.startswith("gfx950") for arch in arches):
        pytest.skip(f"all eight visible devices must be gfx950, got {arches}")

    # ROCr is selected at process startup, before torch is imported.  The raw
    # preflight performs the authoritative dladdr check inside every worker;
    # this parent check avoids spawning eight processes for a known-bad setup.
    preload = os.getenv("LD_PRELOAD", "")
    if "libhsa-runtime64.so" not in preload:
        pytest.skip("the test requires the pinned system ROCr in LD_PRELOAD")

    from tokenspeed_kernel_amd.ops.gfx950.moe import megamoe

    if not megamoe.KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE:
        pytest.skip(
            "MegaMoE completion marker is false; no collective worker was spawned"
        )

    # Fail before spawning if the retained raw API and direct launcher have not
    # landed together.  A marker-true build may never fall back to a JIT launch.
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe import kernel

    for symbol in (
        "PreparedKimiK3MegaMoEKernel",
        "prepare_kimi_k3_megamoe_gfx950",
        "launch_kimi_k3_megamoe_gfx950",
    ):
        if not hasattr(kernel, symbol):
            pytest.fail(f"completion marker is true but raw API lacks {symbol}")


def _join_processes_until(processes: tuple, deadline: float) -> None:
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))


def _terminate_processes(context) -> None:
    processes = tuple(context.processes)
    for process in processes:
        if process.is_alive():
            process.terminate()
    _join_processes_until(processes, time.monotonic() + 5.0)

    # A rank blocked in ROCr/HIP teardown may ignore SIGTERM. Escalate so a
    # failed collective cannot leave a live process holding a GPU context and
    # poison the next qualification attempt.
    survivors = tuple(process for process in processes if process.is_alive())
    for process in survivors:
        process.kill()
    _join_processes_until(survivors, time.monotonic() + 5.0)

    remaining = tuple(process.pid for process in survivors if process.is_alive())
    if remaining:
        raise RuntimeError(
            f"MegaMoE watchdog could not kill worker processes {remaining}"
        )


def _spawn_with_watchdog(mode: str) -> None:
    watchdog_seconds = float(os.getenv(_WATCHDOG_ENV, "300"))
    if watchdog_seconds <= 0:
        raise ValueError(f"{_WATCHDOG_ENV} must be positive")

    context = mp.spawn(
        _worker_main,
        args=(mode, _open_port()),
        nprocs=_WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + watchdog_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(
                    f"eight-rank MegaMoE {mode} scenario exceeded the "
                    f"{watchdog_seconds:.0f}s watchdog"
                )
    except BaseException:
        _terminate_processes(context)
        raise


def _selected_global_experts(route_distribution: str) -> tuple[int, ...]:
    """Return one deterministic top-16 ownership pattern for the EP8 fixture."""

    if route_distribution == "balanced":
        return tuple(
            expert
            for owner in range(_WORLD_SIZE)
            for expert in (112 * owner, 112 * owner + 1)
        )
    if route_distribution == "rank0_concentrated":
        return tuple(range(16))
    raise ValueError(f"unknown route distribution: {route_distribution}")


def _expected_local_route_ids(
    route_distribution: str,
    rank: int,
) -> tuple[int, ...]:
    expert_start = 112 * rank
    return tuple(
        expert - expert_start
        for expert in _selected_global_experts(route_distribution)
        if expert_start <= expert < expert_start + 112
    )


def _initialize_layer_tensors(
    rank: int,
    device: torch.device,
    route_distribution: str,
) -> tuple:
    """Create one real layer with the requested deterministic EP ownership."""

    hidden_states = torch.empty((1, 7168), dtype=torch.bfloat16, device=device)
    prefix_sum = torch.empty_like(hidden_states)

    router_weight = torch.zeros((896, 7168), dtype=torch.bfloat16, device=device)
    routed_down_weight = torch.zeros((3584, 7168), dtype=torch.bfloat16, device=device)
    shared_gate_up_weight = torch.zeros(
        (1536, 7168), dtype=torch.bfloat16, device=device
    )
    shared_down_weight = torch.zeros((7168, 768), dtype=torch.bfloat16, device=device)
    routed_norm_weight = torch.ones((3584,), dtype=torch.bfloat16, device=device)
    routed_up_weight = torch.zeros((7168, 3584), dtype=torch.bfloat16, device=device)

    # Every rank observes the same top-16. The balanced case gives each shard
    # two routes; the concentrated case gives rank zero all sixteen and every
    # other shard zero, covering both expert bodies and the empty-route path.
    correction_bias = torch.full((896,), -32.0, dtype=torch.float32, device=device)
    selected = torch.tensor(
        _selected_global_experts(route_distribution),
        dtype=torch.int64,
        device=device,
    )
    correction_bias[selected] = 32.0

    diagonal_3584 = torch.arange(3584, device=device)
    routed_down_weight[diagonal_3584, diagonal_3584] = 0.125
    routed_up_weight[diagonal_3584, diagonal_3584] = 0.125
    diagonal_768 = torch.arange(768, device=device)
    shared_gate_up_weight[diagonal_768, diagonal_768] = 0.125
    shared_gate_up_weight[768 + diagonal_768, diagonal_768] = 0.125
    shared_down_weight[diagonal_768, diagonal_768] = 0.125

    # E2M1 nibbles and E8M0 scale bytes are deliberately rank-distinct.  Scale
    # byte 120 keeps the full two-GEMM route finite while every rank contributes
    # a different packed expert shard.
    nibble = (1, 2, 3, 4, 5, 6, 7, 9)[rank]
    packed = nibble | (nibble << 4)
    w13_weight = torch.full((112, 6144, 1792), packed, dtype=torch.uint8, device=device)
    w13_weight_scale = torch.full(
        (112, 6144, 112), 120, dtype=torch.uint8, device=device
    )
    w2_weight = torch.full((112, 3584, 1536), packed, dtype=torch.uint8, device=device)
    w2_weight_scale = torch.full((112, 3584, 96), 120, dtype=torch.uint8, device=device)

    return (
        hidden_states,
        prefix_sum,
        router_weight,
        routed_down_weight,
        shared_gate_up_weight,
        shared_down_weight,
        routed_norm_weight,
        routed_up_weight,
        correction_bias,
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
    )


def _build_state(
    rank: int,
    device: torch.device,
    route_distribution: str,
) -> dict[str, Any]:
    from tokenspeed_kernel.ops.communication import unpack_producer_direct_lane
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_acquire_producer_direct_lane,
    )
    from tokenspeed_kernel.ops.moe.megamoe.types import (
        KimiK3MegaMoELayerSpec,
    )

    iris_state = create_iris_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_numel=_TOTAL_LANE_ELEMENTS,
        dtype=torch.bfloat16,
        device=device,
    )
    opaque_lane = iris_acquire_producer_direct_lane(iris_state, _LANE_SHAPES)
    unpacked = unpack_producer_direct_lane(opaque_lane)
    tensors = _initialize_layer_tensors(rank, device, route_distribution)
    spec = KimiK3MegaMoELayerSpec(
        router_weight=tensors[2],
        routed_down_weight=tensors[3],
        shared_gate_up_weight=tensors[4],
        shared_down_weight=tensors[5],
        routed_norm_weight=tensors[6],
        routed_up_weight=tensors[7],
        correction_bias=tensors[8],
        w13_weight=tensors[9],
        w13_weight_scale=tensors[10],
        w2_weight=tensors[11],
        w2_weight_scale=tensors[12],
        expert_start=112 * rank,
    )
    return {
        "iris_state": iris_state,
        "opaque_lane": opaque_lane,
        "lane": unpacked,
        "spec": spec,
        "hidden": tensors[0],
        "prefix": tensors[1],
        "packed_expert_byte": int(tensors[9].flatten()[0].item()),
        "route_distribution": route_distribution,
    }


def _uniform_prepare(state: dict[str, Any], timeout_ns: int):
    from tokenspeed_kernel.ops.moe.megamoe.api import prepare_kimi_k3_megamoe
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe import kernel

    plans = None
    local_error = ""
    consensus_stages: list[str] = []

    def consensus(stage: str, local_status: int, local_reason: str) -> None:
        statuses: list[tuple[int, str] | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(statuses, (int(local_status), local_reason))
        failures = [
            (peer, status, reason)
            for peer, item in enumerate(statuses)
            if item is not None
            for status, reason in (item,)
            if status != 0
        ]
        if failures:
            details = "; ".join(
                f"rank {peer}: status={status}, reason={reason}"
                for peer, status, reason in failures
            )
            raise RuntimeError(f"rank-uniform {stage} consensus failed: {details}")
        consensus_stages.append(stage)

    try:
        plans = prepare_kimi_k3_megamoe(
            (state["spec"],) * 92,
            state["opaque_lane"],
            like=state["hidden"],
            consensus=consensus,
            timeout_ns=timeout_ns,
        )
    except BaseException:
        local_error = traceback.format_exc()[-4096:]

    gathered: list[str] = [""] * _WORLD_SIZE
    dist.all_gather_object(gathered, local_error)
    failures = [index for index, error in enumerate(gathered) if error]
    if failures:
        details = "\n".join(f"rank {index}: {gathered[index]}" for index in failures)
        raise RuntimeError(f"rank-uniform public preparation failed:\n{details}")
    if plans is None:
        raise AssertionError("public preparation returned no retained plans")
    assert consensus_stages == [
        "host_prepare",
        "compile_resource",
        "collective_warmup",
    ]
    assert len(plans) == 92
    retained = plans[0].prepared_kernel
    assert all(plan.prepared_kernel is retained for plan in plans)
    assert all(plan.lane is state["opaque_lane"] for plan in plans)
    return plans, kernel


def _assert_rank_ownership(rank: int, state: dict[str, Any], prepared) -> None:
    workspace = state["workspace"]
    local_count = int(workspace["local_route_count"].item())
    local_ids = tuple(
        int(value)
        for value in workspace["local_route_ids"][:local_count].cpu().tolist()
    )
    ownership = (
        int(prepared.group_rank),
        int(prepared.expert_start),
        local_count,
        local_ids,
        state["packed_expert_byte"],
    )
    gathered: list[tuple | None] = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, ownership)
    assert [item[0] for item in gathered] == list(range(_WORLD_SIZE))
    assert [item[1] for item in gathered] == [112 * owner for owner in range(8)]
    expected = [
        _expected_local_route_ids(state["route_distribution"], owner)
        for owner in range(_WORLD_SIZE)
    ]
    assert [item[2] for item in gathered] == [len(route_ids) for route_ids in expected]
    assert [item[3] for item in gathered] == expected
    assert len({item[4] for item in gathered}) == _WORLD_SIZE
    assert ownership[0] == rank


def _assert_success_generation(
    rank: int,
    generation: int,
    state: dict[str, Any],
    *,
    expert_phase_before: dict[str, torch.Tensor] | None = None,
    require_routed_signal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    lane = state["lane"]
    workspace = state["workspace"]
    assert int(lane.fatal_epoch.item()) == 0
    assert tuple(workspace["fail_diagnostics"].shape) == (8,)
    assert int(workspace["phase_arrival"].item()) == 0
    assert int(workspace["phase_gate"].item()) == 0
    assert int(workspace["topology_gate"].item()) == generation
    assert int(workspace["route_plan_gate"].item()) == generation
    assert int(workspace["comm_arrival"].item()) == generation * _COMM_PROGRAMS
    assert int(workspace["comm_gate"].item()) == generation
    torch.testing.assert_close(
        workspace["xcc_ticket"].cpu(),
        torch.full((8,), generation * 30, dtype=torch.int64),
        atol=0,
        rtol=0,
    )
    assert int(workspace["xcd_arrival"].item()) == generation * 8
    assert not torch.count_nonzero(lane.topology_status)
    _assert_success_phase_state(
        generation,
        workspace,
        expert_phase_before,
    )

    flags = lane.iris_epoch_flags.cpu()
    torch.testing.assert_close(
        flags[:, :, rank],
        torch.full((2, _COMM_PROGRAMS), generation, dtype=torch.int64),
        atol=0,
        rtol=0,
    )
    other_columns = [column for column in range(_WORLD_SIZE) if column != rank]
    assert not torch.count_nonzero(flags[:, :, other_columns])

    output = state["output"].float().cpu()
    assert bool(torch.isfinite(output).all())
    producer = lane.symmetric_producer.float().cpu()
    reduced = lane.symmetric_reduced.float().cpu()
    for tensor in (
        producer[:7168],
        producer[7168:],
        reduced[:7168],
        reduced[7168:],
    ):
        assert bool(torch.isfinite(tensor).all())
    routed_reduced = reduced[7168:].clone()
    if require_routed_signal:
        assert bool(torch.count_nonzero(routed_reduced))
    gathered = [torch.empty_like(output) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, output)
    for peer_output in gathered[1:]:
        torch.testing.assert_close(peer_output, gathered[0], atol=0.25, rtol=0.02)
    return output, routed_reduced


def _install_no_late_jit_guard(kernel_module):
    """Make every post-admission compiler/admission entry fail."""

    from tokenspeed_kernel._triton import triton
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe import admission

    previous_hook = triton.knobs.runtime.jit_post_compile_hook
    previous_compile = kernel_module.compile_kimi_k3_megamoe_gfx950
    previous_kernel_admit = kernel_module.admit_kimi_k3_megamoe_compiled_kernel
    previous_module_admit = admission.admit_kimi_k3_megamoe_compiled_kernel

    def late_compile_or_admit(*_args, **_kwargs):
        raise AssertionError(
            "MegaMoE entered JIT/admission after retained-binary preparation"
        )

    triton.knobs.runtime.jit_post_compile_hook = late_compile_or_admit
    kernel_module.compile_kimi_k3_megamoe_gfx950 = late_compile_or_admit
    kernel_module.admit_kimi_k3_megamoe_compiled_kernel = late_compile_or_admit
    admission.admit_kimi_k3_megamoe_compiled_kernel = late_compile_or_admit
    return (
        triton,
        previous_hook,
        previous_compile,
        previous_kernel_admit,
        admission,
        previous_module_admit,
    )


def _capture_graph(
    decode,
    state: dict[str, Any],
    plan,
    device: torch.device,
    execution_stream: torch.cuda.Stream,
):
    dist.barrier()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=execution_stream):
        decode(state["hidden"], state["prefix"], plan)
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph


def _capture_two_plan_failstop_graph(
    decode,
    state: dict[str, Any],
    first_plan,
    second_plan,
    device: torch.device,
    execution_stream: torch.cuda.Stream,
):
    """Capture poison source plus a later layer sharing the same state."""

    assert first_plan.workspace.keys() == second_plan.workspace.keys()
    assert all(
        first_plan.workspace[name].data_ptr() == second_plan.workspace[name].data_ptr()
        for name in first_plan.workspace
    )
    assert first_plan.layer_output.data_ptr() != second_plan.layer_output.data_ptr()

    writable = _writable_state(state, first_plan, second_plan)
    before_second = {
        name: torch.empty_like(tensor) for name, tensor in writable.items()
    }
    after_second = {name: torch.empty_like(tensor) for name, tensor in writable.items()}

    dist.barrier()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=execution_stream):
        decode(state["hidden"], state["prefix"], first_plan)
        for name, tensor in writable.items():
            before_second[name].copy_(tensor)
        decode(state["hidden"], state["prefix"], second_plan)
        for name, tensor in writable.items():
            after_second[name].copy_(tensor)
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph, before_second, after_second


def _assert_misaligned_input_rejected_before_dispatch(
    decode,
    state: dict[str, Any],
    plan,
) -> None:
    """Exercise public alignment validation without entering the collective."""

    storage = torch.empty(
        (1, 7169), dtype=torch.bfloat16, device=state["hidden"].device
    )
    misaligned = storage[:, 1:]
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0
    topology_generation_before = int(plan.workspace["topology_gate"].item())
    phase_arrival_before = int(plan.workspace["phase_arrival"].item())
    phase_gate_before = int(plan.workspace["phase_gate"].item())
    try:
        decode(misaligned, state["prefix"], plan)
    except ValueError as error:
        assert "16-byte-aligned" in str(error)
    else:
        raise AssertionError("misaligned MegaMoE input reached collective dispatch")
    assert int(plan.workspace["topology_gate"].item()) == topology_generation_before
    assert int(plan.workspace["phase_arrival"].item()) == phase_arrival_before
    assert int(plan.workspace["phase_gate"].item()) == phase_gate_before


def _worker_main(rank: int, mode: str, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    try:
        from tokenspeed_kernel_amd.ops.gfx950.moe import megamoe

        if not megamoe.KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE:
            raise RuntimeError("completion marker changed after the parent gate")

        timeout_ns = _qualified_timeout_ns()
        route_distribution = (
            "rank0_concentrated" if mode == "success_concentrated" else "balanced"
        )
        state = _build_state(rank, device, route_distribution)
        torch.cuda.synchronize(device)
        dist.barrier()

        plans, kernel_module = _uniform_prepare(state, timeout_ns)
        plan = plans[0]
        execution_stream = torch.cuda.Stream(device=device)
        prepared = plan.prepared_kernel
        state["workspace"] = plan.workspace
        state["output"] = plan.layer_output
        assert prepared.compiled is not None
        assert callable(prepared.runner)
        assert prepared.admission.occupancy == 1
        assert prepared.admission.programs == _PROGRAMS
        assert prepared.admission.subgroups == 8
        assert prepared.admission.code_object_sha256
        # Public preparation collectively warmed generation one.
        _assert_success_generation(rank, 1, state)
        _assert_rank_ownership(rank, state, prepared)

        from tokenspeed_kernel.ops.moe.megamoe.api import kimi_k3_megamoe_decode

        _assert_misaligned_input_rejected_before_dispatch(
            kimi_k3_megamoe_decode,
            state,
            plan,
        )
        dist.barrier()
        retained_compiled = prepared.compiled
        retained_runner = prepared.runner
        guard = _install_no_late_jit_guard(kernel_module)
        (
            triton,
            previous_hook,
            previous_compile,
            previous_kernel_admit,
            admission_module,
            previous_module_admit,
        ) = guard
        try:
            with torch.cuda.stream(execution_stream):
                state["hidden"].fill_(0.01)
                state["prefix"].fill_(0.125)
            expert_phase_before = _snapshot_expert_phase_state(state)
            dist.barrier()
            with torch.cuda.stream(execution_stream):
                output = kimi_k3_megamoe_decode(
                    state["hidden"],
                    state["prefix"],
                    plan,
                )
            assert output is plan.layer_output
            torch.cuda.synchronize(device)
            dist.barrier()
            assert plan._stream_owner.stream_handle == int(execution_stream.cuda_stream)
            first_output, first_routed_reduced = _assert_success_generation(
                rank,
                2,
                state,
                expert_phase_before=expert_phase_before,
            )
            _assert_rank_ownership(rank, state, prepared)

            second_plan = plans[1]
            if mode == "failstop":
                # Give node two a recognizable unique output. Its captured
                # launch follows the node that times out and poisons the Iris
                # lane, and must leave this value and all shared state intact.
                second_plan.layer_output.fill_(rank + 0.75)
                graph, before_second, after_second = _capture_two_plan_failstop_graph(
                    kimi_k3_megamoe_decode,
                    state,
                    plan,
                    second_plan,
                    device,
                    execution_stream,
                )
            else:
                graph = _capture_graph(
                    kimi_k3_megamoe_decode,
                    state,
                    plan,
                    device,
                    execution_stream,
                )
            # Capture records but does not execute the collective.
            assert int(state["workspace"]["topology_gate"].item()) == 2
            assert int(state["workspace"]["phase_arrival"].item()) == 0
            assert int(state["workspace"]["phase_gate"].item()) == 0
            assert int(state["lane"].fatal_epoch.item()) == 0

            if mode in {"success", "success_concentrated"}:
                previous_output = first_output
                previous_routed_reduced = first_routed_reduced
                replay_count = int(
                    os.getenv("TOKENSPEED_TEST_MEGAMOE_GRAPH_REPLAYS", "3")
                )
                if replay_count < 2:
                    raise ValueError(
                        "TOKENSPEED_TEST_MEGAMOE_GRAPH_REPLAYS must be at least 2"
                    )
                for replay in range(replay_count):
                    with torch.cuda.stream(execution_stream):
                        state["hidden"].fill_(0.015 + replay * 0.005)
                        state["prefix"].fill_(0.25 + replay * 0.125)
                    expert_phase_before = _snapshot_expert_phase_state(state)
                    _poison_symmetric_lane_before_replay(state, device)
                    with torch.cuda.stream(execution_stream):
                        graph.replay()
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    generation = replay + 3
                    output, routed_reduced = _assert_success_generation(
                        rank,
                        generation,
                        state,
                        expert_phase_before=expert_phase_before,
                        require_routed_signal=True,
                    )
                    assert not torch.equal(output, previous_output)
                    assert not torch.equal(
                        routed_reduced,
                        previous_routed_reduced,
                    )
                    previous_output = output
                    previous_routed_reduced = routed_reduced

                timing_replays = (
                    0
                    if mode == "success_concentrated"
                    else int(os.getenv(_TIMING_REPLAYS_ENV, "0"))
                )
                if timing_replays < 0:
                    raise ValueError(f"{_TIMING_REPLAYS_ENV} must be nonnegative")
                if timing_replays:
                    # The correctness replays above warm the retained graph.
                    # Time only the graph nodes on the owned execution stream;
                    # enqueue every generation without an intermediate host or
                    # Gloo synchronization so this measures the production
                    # exact-generation protocol, including rank skew.
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(execution_stream):
                        start.record()
                        for _ in range(timing_replays):
                            graph.replay()
                        end.record()
                    end.synchronize()
                    elapsed_ms = start.elapsed_time(end)
                    milliseconds_per_layer = elapsed_ms / timing_replays
                    timings: list[float | None] = [None] * _WORLD_SIZE
                    dist.all_gather_object(timings, milliseconds_per_layer)
                    if rank == 0:
                        values = [
                            float(value) for value in timings if value is not None
                        ]
                        print(
                            "MEGAMOE_LAYER_TIMING_MS "
                            f"replays={timing_replays} "
                            f"min={min(values):.6f} "
                            f"max={max(values):.6f} "
                            f"per_rank={values}",
                            flush=True,
                        )
                    expected_generation = 2 + replay_count + timing_replays
                    assert int(state["workspace"]["topology_gate"].item()) == (
                        expected_generation
                    )
                    assert int(state["workspace"]["phase_arrival"].item()) == 0
                    assert int(state["workspace"]["phase_gate"].item()) == 0
                    assert int(state["lane"].fatal_epoch.item()) == 0
                assert prepared.compiled is retained_compiled
                assert prepared.runner is retained_runner
            elif mode == "failstop":
                next_generation = 3
                with torch.cuda.stream(execution_stream):
                    state["hidden"].fill_(0.02)
                    state["prefix"].fill_(0.5)
                dist.barrier()
                # Rank zero deliberately omits one captured collective replay.
                # The other seven ranks must time out on its stale topology
                # generation, system-release poison all eight fatal slots, and
                # return from every resident cooperative workgroup.
                if rank != 0:
                    with torch.cuda.stream(execution_stream):
                        graph.replay()
                torch.cuda.synchronize(device)
                dist.barrier()

                fatal = int(state["lane"].fatal_epoch.item())
                gathered_fatal: list[int | None] = [None] * _WORLD_SIZE
                dist.all_gather_object(gathered_fatal, fatal)
                assert gathered_fatal == [next_generation] * _WORLD_SIZE
                if rank != 0:
                    assert (
                        int(before_second["lane.fatal_epoch"].item()) == next_generation
                    )
                    _assert_snapshots_equal(before_second, after_second)
                assert prepared.compiled is retained_compiled
                assert prepared.runner is retained_runner
                if rank == 0:
                    assert int(state["workspace"]["topology_gate"].item()) == 2
                else:
                    assert int(state["workspace"]["topology_gate"].item()) == 3
                assert int(state["workspace"]["phase_arrival"].item()) == 0
                assert int(state["workspace"]["phase_gate"].item()) == 0
            else:
                raise ValueError(f"unknown MegaMoE integration mode: {mode}")
        finally:
            triton.knobs.runtime.jit_post_compile_hook = previous_hook
            kernel_module.compile_kimi_k3_megamoe_gfx950 = previous_compile
            kernel_module.admit_kimi_k3_megamoe_compiled_kernel = previous_kernel_admit
            admission_module.admit_kimi_k3_megamoe_compiled_kernel = (
                previous_module_admit
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_gfx950_megamoe_eight_rank_eager_and_graph_generations() -> None:
    """Exercise retained-binary execution and reusable Iris generations on EP8.

    Focused publication litmuses and the K3 numerical oracle remain separate
    mandatory qualification gates.
    """

    _require_parent_gate(failstop=False)
    _spawn_with_watchdog("success")


def test_gfx950_megamoe_eight_rank_concentrated_routes() -> None:
    """Exercise rank0 L16 and the seven L0 paths through graph replay."""

    _require_parent_gate(failstop=False)
    _spawn_with_watchdog("success_concentrated")


def test_gfx950_megamoe_eight_rank_graph_failstop_is_uniform() -> None:
    """Prove a missing rank poisons every lane and all peer kernels return."""

    _require_parent_gate(failstop=True)
    _spawn_with_watchdog("failstop")


def test_failstop_snapshot_guard_detects_a_post_poison_write() -> None:
    """Keep the exact no-write assertion active without requiring EP8."""

    before = {
        "workspace.phase_arrival": torch.tensor([0], dtype=torch.int64),
        "workspace.phase_gate": torch.tensor([0], dtype=torch.int64),
        "workspace.topology_gate": torch.tensor([3], dtype=torch.int64),
        "second_plan.layer_output": torch.full((1, 4), 0.75, dtype=torch.bfloat16),
    }
    after = {name: tensor.clone() for name, tensor in before.items()}
    _assert_snapshots_equal(before, after)

    after["workspace.topology_gate"].add_(1)
    with pytest.raises(AssertionError, match="workspace.topology_gate"):
        _assert_snapshots_equal(before, after)


def test_watchdog_escalates_a_stubborn_worker_to_kill() -> None:
    """A SIGTERM-resistant rank must not survive harness cleanup."""

    class StubbornProcess:
        pid = 17

        def __init__(self) -> None:
            self.alive = True
            self.terminated = False
            self.killed = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.alive = False

        def join(self, timeout: float) -> None:
            assert timeout >= 0

    process = StubbornProcess()
    _terminate_processes(SimpleNamespace(processes=(process,)))
    assert process.terminated
    assert process.killed
    assert not process.is_alive()


def test_ep8_timeout_is_exactly_qualified(monkeypatch) -> None:
    """Reject a harness override that would compile a different binary."""

    monkeypatch.delenv(_TIMEOUT_ENV, raising=False)
    assert _qualified_timeout_ns() == _QUALIFIED_TIMEOUT_NS

    monkeypatch.setenv(_TIMEOUT_ENV, str(_QUALIFIED_TIMEOUT_NS - 1))
    with pytest.raises(ValueError, match="must equal the qualified value"):
        _qualified_timeout_ns()


def test_ep8_route_distributions_cover_two_body_and_zero_route_paths() -> None:
    """Keep both retained-object success ownership patterns host-visible."""

    assert _selected_global_experts("balanced") == tuple(
        expert
        for owner in range(_WORLD_SIZE)
        for expert in (112 * owner, 112 * owner + 1)
    )
    assert all(
        _expected_local_route_ids("balanced", rank) == (0, 1)
        for rank in range(_WORLD_SIZE)
    )
    assert _selected_global_experts("rank0_concentrated") == tuple(range(16))
    assert _expected_local_route_ids("rank0_concentrated", 0) == tuple(range(16))
    assert all(
        _expected_local_route_ids("rank0_concentrated", rank) == ()
        for rank in range(1, _WORLD_SIZE)
    )


def test_success_phase_guard_detects_an_inactive_control_slot() -> None:
    """Keep the one-slot expert rendezvous checks live without EP8."""

    generation = 3
    route_count = 2
    workspace = {
        "shared_arrival": torch.full((14,), generation * 16, dtype=torch.int64),
        "shared_gate": torch.full((14,), generation, dtype=torch.int64),
        "local_route_count": torch.tensor([route_count], dtype=torch.int32),
    }
    before: dict[str, torch.Tensor] = {}
    for phase in ("w13", "w2"):
        previous = torch.arange(16, dtype=torch.int64)
        previous[0] = 452
        for suffix in ("target", "arrival", "gate"):
            name = f"{phase}_{suffix}"
            before[name] = previous.clone()
            workspace[name] = previous.clone()
            workspace[name][0] += 226

    _assert_success_phase_state(generation, workspace, before)
    workspace["w13_arrival"][1] += 1
    with pytest.raises(AssertionError, match="w13 inactive control slots changed"):
        _assert_success_phase_state(generation, workspace, before)


def test_success_phase_guard_requires_l0_controls_to_stay_unchanged() -> None:
    """L=0 uses seven zero-stripe owners and no expert-grid counter write."""

    generation = 5
    workspace = {
        "shared_arrival": torch.full((14,), generation * 16, dtype=torch.int64),
        "shared_gate": torch.full((14,), generation, dtype=torch.int64),
        "local_route_count": torch.tensor([0], dtype=torch.int32),
    }
    before: dict[str, torch.Tensor] = {}
    for phase in ("w13", "w2"):
        for suffix in ("target", "arrival", "gate"):
            name = f"{phase}_{suffix}"
            before[name] = torch.arange(16, dtype=torch.int64) + generation * 226
            workspace[name] = before[name].clone()

    _assert_success_phase_state(generation, workspace, before)
    workspace["w2_target"][0] += 1
    with pytest.raises(AssertionError, match="w2 L=0 controls changed"):
        _assert_success_phase_state(generation, workspace, before)
