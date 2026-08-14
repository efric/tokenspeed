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


import math
import socket
import traceback
from types import SimpleNamespace
from typing import List, Tuple
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_producer_direct_admission_supported_world_sizes(
    monkeypatch,
    world_size,
    dtype,
):
    from tokenspeed_kernel.ops.communication import triton as triton_ops

    monkeypatch.setattr(
        triton_ops,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    state = SimpleNamespace(world_size=world_size, max_bytes=64)

    assert triton_ops.symm_outputs_can_run(
        state,
        ((3, 5), (1, 1)),
        dtype,
    )


@pytest.mark.parametrize(
    ("dtype", "shapes"),
    [
        (torch.bfloat16, ((32,),)),
        (torch.float16, ((32,),)),
        (torch.float32, ((16,),)),
    ],
)
def test_producer_direct_admission_uses_byte_capacity(monkeypatch, dtype, shapes):
    from tokenspeed_kernel.ops.communication import triton as triton_ops

    monkeypatch.setattr(
        triton_ops,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    state = SimpleNamespace(world_size=8, max_bytes=64)

    assert triton_ops.symm_outputs_can_run(state, shapes, dtype)


@pytest.mark.parametrize(
    ("world_size", "shapes", "dtype", "op"),
    [
        (1, ((4,),), torch.bfloat16, dist.ReduceOp.SUM),
        (8, ((3,),), torch.bfloat16, dist.ReduceOp.SUM),
        (8, ((3,),), torch.float32, dist.ReduceOp.SUM),
        (8, ((36,),), torch.bfloat16, dist.ReduceOp.SUM),
        (8, ((18,),), torch.float32, dist.ReduceOp.SUM),
        (8, ((4,),), torch.float64, dist.ReduceOp.SUM),
        (8, ((4,),), torch.bfloat16, dist.ReduceOp.PRODUCT),
    ],
)
def test_producer_direct_admission_rejects_unsupported_requests(
    monkeypatch,
    world_size,
    shapes,
    dtype,
    op,
):
    from tokenspeed_kernel.ops.communication import triton as triton_ops

    monkeypatch.setattr(
        triton_ops,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
    )
    state = SimpleNamespace(world_size=world_size, max_bytes=64)

    assert not triton_ops.symm_outputs_can_run(state, shapes, dtype, op)


def test_producer_direct_admission_is_cdna4_only(monkeypatch):
    from tokenspeed_kernel.ops.communication import triton as triton_ops

    monkeypatch.setattr(
        triton_ops,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=False),
    )
    state = SimpleNamespace(world_size=8, max_bytes=64)

    assert not triton_ops.symm_outputs_can_run(
        state,
        ((4,),),
        torch.bfloat16,
    )


def test_ordinary_all_reduce_initializes_full_producer_direct_capacity(monkeypatch):
    from tokenspeed_kernel.ops.communication import iris as iris_ops
    from tokenspeed_kernel.ops.communication import triton as triton_ops

    platform = SimpleNamespace(is_amd=True, is_cdna4=True)
    monkeypatch.setattr(triton_ops, "current_platform", lambda: platform)
    monkeypatch.setattr(triton_ops, "all_reduce_can_run", lambda *_args, **_kw: True)
    monkeypatch.setattr(iris_ops, "IRIS_AR_STATES", {})

    created_state = SimpleNamespace(max_numel=512)
    create_iris_state = mock.Mock(return_value=created_state)
    monkeypatch.setattr(iris_ops, "create_iris_state", create_iris_state)
    monkeypatch.setattr(iris_ops, "iris_all_reduce", lambda *_args, **_kw: object())

    def acquire_outputs(state, shapes):
        assert sum(math.prod(shape) for shape in shapes) <= state.max_numel
        return (object(),)

    monkeypatch.setattr(iris_ops, "iris_acquire_outputs", acquire_outputs)
    producer_lane = object()
    acquire_lane = mock.Mock(return_value=producer_lane)
    monkeypatch.setattr(
        iris_ops,
        "iris_acquire_producer_direct_lane",
        acquire_lane,
    )

    group = object()
    state = SimpleNamespace(
        group=group,
        rank_in_group=0,
        world_size=8,
        max_numel=256,
        max_bytes=1024,
        device=torch.device("cpu"),
    )
    tensor = SimpleNamespace(dtype=torch.bfloat16)

    triton_ops.all_reduce(state, tensor)
    triton_ops.acquire_symm_outputs(state, ((300,),), torch.bfloat16)
    assert (
        triton_ops.acquire_producer_direct_lane(
            state,
            ((300,),),
            torch.bfloat16,
        )
        is producer_lane
    )

    create_iris_state.assert_called_once_with(
        group=group,
        rank_in_group=0,
        max_numel=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    acquire_lane.assert_called_once_with(created_state, ((300,),))


def test_producer_direct_lane_exposes_graph_stable_flat_abi():
    from tokenspeed_kernel.ops.communication import iris as iris_ops

    class FakeContext:
        def __init__(self, heap_bases):
            self.heap_bases = heap_bases

        def get_heap_bases(self):
            return self.heap_bases

    shapes = ((1, 7168), (1, 3584))
    total_numel = sum(math.prod(shape) for shape in shapes)
    owner = object.__new__(iris_ops.IrisAllReduce)
    owner._ctx = FakeContext(torch.arange(8, dtype=torch.int64))
    owner._input_buf = torch.zeros(total_numel, dtype=torch.bfloat16)
    owner._producer_direct_output_buf = torch.empty_like(owner._input_buf)
    owner._producer_direct_lane_input_buf = torch.zeros_like(owner._input_buf)
    owner._producer_direct_ready_flags = torch.zeros(21, 8, dtype=torch.int32)
    owner._producer_direct_lane_ready_flags = torch.zeros(
        2, 21, 8, dtype=torch.int64
    )
    owner._producer_direct_topology_status = torch.zeros(2, dtype=torch.int64)
    owner._producer_direct_fatal_epoch = torch.zeros(1, dtype=torch.int64)
    owner._heap_bases = owner._ctx.get_heap_bases()
    owner._group_global_rank_tuple = tuple(range(8))
    owner._group_global_ranks = torch.arange(8, dtype=torch.int64)
    owner._producer_direct_lanes = {}
    owner._producer_direct_block_size = 512
    owner._producer_direct_max_programs = 21
    owner._producer_direct_lane_capacity = 512 * 21
    owner._iris_rank = 3
    owner.world_size = 8
    owner.max_numel = total_numel
    owner.dtype = torch.bfloat16
    owner.device = torch.device("cpu")

    lane = owner.acquire_producer_direct_lane(shapes)
    unpacked = iris_ops.unpack_iris_producer_direct_lane(lane)

    assert unpacked is lane
    assert (
        lane.symmetric_producer.data_ptr()
        == owner._producer_direct_lane_input_buf.data_ptr()
    )
    assert lane.symmetric_producer.data_ptr() != owner._input_buf.data_ptr()
    producer_end = lane.symmetric_producer.data_ptr() + (
        lane.symmetric_producer.numel() * lane.symmetric_producer.element_size()
    )
    ordinary_end = owner._input_buf.data_ptr() + (
        owner._input_buf.numel() * owner._input_buf.element_size()
    )
    assert (
        producer_end <= owner._input_buf.data_ptr()
        or ordinary_end <= lane.symmetric_producer.data_ptr()
    )
    lane.symmetric_producer.fill_(1)
    assert torch.count_nonzero(owner._input_buf).item() == 0
    assert (
        lane.symmetric_reduced.data_ptr()
        == owner._producer_direct_output_buf.data_ptr()
    )
    assert lane.iris_epoch_flags is owner._producer_direct_lane_ready_flags
    assert lane.iris_epoch_flags.shape == (2, 21, 8)
    assert owner._producer_direct_ready_flags.dtype == torch.int32
    assert (
        lane.iris_epoch_flags.data_ptr()
        != owner._producer_direct_ready_flags.data_ptr()
    )
    lane.iris_epoch_flags.fill_(1)
    assert torch.count_nonzero(owner._producer_direct_ready_flags).item() == 0
    assert lane.topology_status is owner._producer_direct_topology_status
    assert lane.topology_status.shape == (2,)
    assert lane.fatal_epoch is owner._producer_direct_fatal_epoch
    assert lane.heap_bases is owner._heap_bases
    assert lane.group_global_ranks is owner._group_global_ranks
    assert lane.group_rank == 3
    assert lane.protocol_version == iris_ops.IRIS_PRODUCER_DIRECT_PROTOCOL_VERSION == 3
    assert owner.acquire_producer_direct_lane(shapes) is lane

    dedicated_input = owner._producer_direct_lane_input_buf
    owner._producer_direct_lane_input_buf = dedicated_input.clone()
    with pytest.raises(RuntimeError, match="producer-direct storage was replaced"):
        lane.validate()
    owner._producer_direct_lane_input_buf = dedicated_input

    owner._ctx.heap_bases = owner._heap_bases.clone()
    with pytest.raises(RuntimeError, match="heap-base tensor was replaced"):
        lane.validate()


# ---------------------------------------------------------------------------
# Suite 1: iris_all_reduce
# ---------------------------------------------------------------------------


def _ar_shape_cases() -> List[Tuple[int, ...]]:
    """Shapes covering small, vector, and 2-D cases."""
    return [
        (8,),
        (16, 64),
        (4, 7, 32),
    ]


def _ar_output_shape_cases() -> List[Tuple[Tuple[int, ...], ...]]:
    """Producer-direct collections spanning one, two, and three outputs."""
    return [
        ((1, 7168), (1, 3584)),
        ((2, 7168), (2, 3584)),
        ((4, 7168), (4, 3584)),
        ((8, 7168), (8, 3584)),
        ((16, 7168), (16, 3584)),
        ((3, 20), (2, 12)),
        ((3, 5), (1, 1)),
        ((2, 16),),
        ((3, 20), (2, 12), (4, 4)),
    ]


def _ar_worker_fn(rank, world_size, port, error_dict):
    try:
        _ar_worker_main(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _ar_worker_main(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Iris's example uses gloo because heap-base exchange is host-side; nccl
    # also works, but gloo avoids contending with the iris-managed device
    # memory and matches the upstream example.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Importing inside the worker avoids pulling iris into the parent
        # process (which has no distributed context).
        from tokenspeed_kernel.ops.communication.iris import create_iris_state

        output_shape_cases = _ar_output_shape_cases()
        max_numel = max(
            max(int(torch.tensor(s).prod()) for s in _ar_shape_cases()),
            max(
                sum(int(torch.tensor(shape).prod()) for shape in shapes)
                for shapes in output_shape_cases
            ),
        )
        state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_numel=max_numel,
            dtype=torch.bfloat16,
        )
        for shape in _ar_shape_cases():
            _check_all_reduce(state, rank, world_size, shape, device)
        for shapes in output_shape_cases:
            _check_all_reduce_symmetric_outputs(
                state,
                rank,
                world_size,
                shapes,
                device,
            )

        fp16_state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_numel=max_numel,
            dtype=torch.float16,
        )
        _check_all_reduce_symmetric_outputs(
            fp16_state,
            rank,
            world_size,
            ((3, 5), (1, 1)),
            device,
        )

        fp32_state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_numel=max_numel,
            dtype=torch.float32,
        )
        _check_all_reduce_symmetric_outputs(
            fp32_state,
            rank,
            world_size,
            ((3, 5), (1, 1)),
            device,
        )
        if world_size == 8:
            _check_all_reduce_residual_attnres(state, rank, device)
    finally:
        dist.destroy_process_group()


def _check_all_reduce(state, rank: int, world_size: int, shape, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

    # Each rank contributes a tensor filled with ``rank + 1``; the reduction
    # is therefore ``sum(1..world_size) = world_size*(world_size+1)/2``.
    local = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)

    result = iris_all_reduce(state, local)

    expected_value = world_size * (world_size + 1) // 2
    expected = torch.full(shape, expected_value, dtype=torch.bfloat16, device=device)

    assert result.shape == expected.shape, (
        f"shape mismatch: {result.shape} vs {expected.shape}"
    )
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _check_all_reduce_symmetric_outputs(
    state,
    rank: int,
    world_size: int,
    shapes,
    device,
) -> None:
    if not current_platform().is_cdna4:
        return

    from tokenspeed_kernel.ops.communication.iris import (
        iris_acquire_outputs,
        iris_all_reduce_symmetric,
    )

    outputs = iris_acquire_outputs(state, shapes)
    assert state.owns_outputs(outputs)
    assert not state.owns_outputs(tuple(torch.empty_like(output) for output in outputs))
    for index, output in enumerate(outputs, start=1):
        output.fill_(index * (rank + 1))
    results = iris_all_reduce_symmetric(state, outputs)
    expected_value = world_size * (world_size + 1) // 2
    for index, (output, result) in enumerate(zip(outputs, results), start=1):
        torch.testing.assert_close(
            result,
            torch.full_like(output, index * expected_value),
            atol=0,
            rtol=0,
        )

    snapshots = []
    for scale in range(1, 5):
        for index, output in enumerate(outputs, start=1):
            output.fill_(scale * index * (rank + 1))
        results = iris_all_reduce_symmetric(state, outputs)
        snapshots.append(tuple(result.clone() for result in results))
    torch.cuda.synchronize()
    for scale, results in enumerate(snapshots, start=1):
        for index, (output, result) in enumerate(zip(outputs, results), start=1):
            torch.testing.assert_close(
                result,
                torch.full_like(output, scale * index * expected_value),
                atol=0,
                rtol=0,
            )

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index, output in enumerate(outputs, start=1):
            output.fill_(index * (rank + 1))
        graph_results = iris_all_reduce_symmetric(state, outputs)
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    for index, (output, result) in enumerate(zip(outputs, graph_results), start=1):
        torch.testing.assert_close(
            result,
            torch.full_like(output, index * expected_value),
            atol=0,
            rtol=0,
        )


def _check_all_reduce_residual_attnres(state, rank: int, device) -> None:
    from tokenspeed_kernel.ops.activation.triton import (
        attnres_combine,
        attnres_partial,
    )
    from tokenspeed_kernel.ops.communication.iris import (
        iris_all_reduce,
    )
    from tokenspeed_kernel.ops.communication.triton import (
        allreduce_residual_attnres_combine,
        allreduce_residual_attnres_combine_supported,
    )

    torch.manual_seed(101)
    hidden = 7168
    blocks = (torch.randn(4, 1, hidden, device=device) * 0.1).to(torch.bfloat16)
    score_weight = (torch.randn(hidden, device=device) * 0.02).to(torch.bfloat16)
    output_weight = (1.0 + torch.randn(hidden, device=device) * 0.02).to(torch.bfloat16)
    residual = (torch.randn(1, hidden, device=device) * 0.1).to(torch.bfloat16)
    local = (torch.randn(1, hidden, device=device) * 0.01 + (rank + 1) * 0.002).to(
        torch.bfloat16
    )
    scratch = (
        torch.empty(1, device=device, dtype=torch.float32),
        torch.empty(1, device=device, dtype=torch.float32),
        torch.empty(1, hidden, device=device, dtype=torch.float32),
    )
    attnres_partial(blocks, score_weight, 1e-6, scratch)

    reduced = iris_all_reduce(state, local.clone(), safe=False)
    expected_residual = residual + reduced
    expected_hidden = attnres_combine(
        expected_residual,
        score_weight,
        output_weight,
        1e-6,
        scratch,
        torch.empty_like(residual),
    )
    assert allreduce_residual_attnres_combine_supported(
        local,
        residual,
        score_weight,
        output_weight,
        scratch,
        rank=rank,
        group=state.group,
        local_world_size=8,
    )
    for _ in range(4):
        actual_hidden, actual_residual = allreduce_residual_attnres_combine(
            local,
            residual,
            score_weight,
            output_weight,
            scratch,
            rank=rank,
            group=state.group,
            local_world_size=8,
            eps=1e-6,
        )
        torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-2, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hidden, graph_residual = allreduce_residual_attnres_combine(
            local,
            residual,
            score_weight,
            output_weight,
            scratch,
            rank=rank,
            group=state.group,
            local_world_size=8,
            eps=1e-6,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(graph_hidden, expected_hidden, atol=2e-2, rtol=2e-2)


def _run_ar_test(world_size: int) -> None:
    _skip_if_unsupported(world_size, "Iris all-reduce tests")
    port = _get_open_port()
    _spawn_and_collect(_ar_worker_fn, (world_size, port), world_size)


def test_iris_all_reduce_correctness_world2():
    _run_ar_test(world_size=2)


def test_iris_all_reduce_correctness_world4():
    _run_ar_test(world_size=4)


def test_iris_all_reduce_correctness_world8():
    _run_ar_test(world_size=8)


def _ar_subgroup_worker_fn(rank, world_size, port, error_dict):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        groups = (
            tuple(range(0, world_size, 2)),
            tuple(range(1, world_size, 2)),
        )
        process_groups = tuple(dist.new_group(ranks) for ranks in groups)
        group_index = rank % 2
        group = process_groups[group_index]
        group_rank = groups[group_index].index(rank)

        from tokenspeed_kernel.ops.communication.iris import create_iris_state

        state = create_iris_state(
            group=group,
            rank_in_group=group_rank,
            max_numel=32,
            dtype=torch.bfloat16,
        )
        _check_all_reduce(
            state,
            group_rank,
            len(groups[group_index]),
            (4, 7),
            device,
        )
        _check_all_reduce_symmetric_outputs(
            state,
            group_rank,
            len(groups[group_index]),
            ((3, 5), (1, 1)),
            device,
        )
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_iris_all_reduce_noncontiguous_subgroups():
    world_size = 8
    _skip_if_unsupported(world_size, "Iris subgroup all-reduce tests")
    port = _get_open_port()
    _spawn_and_collect(_ar_subgroup_worker_fn, (world_size, port), world_size)


# ---------------------------------------------------------------------------
# Suite 2: IrisRSAG (reduce-scatter / all-gather)
# ---------------------------------------------------------------------------


def _rsag_uniform_token_cases(world_size: int) -> List[List[int]]:
    return [
        [8] * world_size,
        [16] * world_size,
        [64] * world_size,
    ]


def _rsag_worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        _rsag_worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _rsag_worker_main(rank: int, world_size: int, port: int, hidden_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Match the upstream iris example - gloo for the host-side rendezvous.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed_kernel.ops.communication.iris import create_iris_rsag_state

        cases = _rsag_uniform_token_cases(world_size)
        max_tokens = max(sum(tokens) for tokens in cases)
        rsag = create_iris_rsag_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
        )

        # The generic ``all_gather`` / ``reduce_scatter`` dispatchers in
        # ``communication.triton`` route AMD calls to ``amd_rsag_*`` (which
        # require ``state.symm_mem_hdl``); we deliberately bypass that
        # dispatcher and call the iris RSAG state directly. ``rsag`` IS the
        # IrisRSAG instance now (no TritonCommState wrapper).
        ag_fn = lambda state, t, **kw: rsag.all_gather(t, **kw)  # noqa: E731
        rs_fn = lambda state, t, **kw: rsag.reduce_scatter(t, **kw)  # noqa: E731

        for tokens in cases:
            _check_all_gather(
                rsag, rank, world_size, tokens, hidden_size, device, ag_fn
            )
            _check_reduce_scatter(
                rsag, rank, world_size, tokens, hidden_size, device, rs_fn
            )
    finally:
        dist.destroy_process_group()


def _check_all_gather(rsag, rank, world_size, tokens, hidden_size, device, all_gather):
    local_tokens = tokens[rank]
    local = torch.full(
        (local_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = all_gather(rsag, local, token_list_in_group=tokens)

    expected = torch.empty(
        (sum(tokens), hidden_size), dtype=torch.bfloat16, device=device
    )
    offset = 0
    for peer, peer_tokens in enumerate(tokens):
        expected[offset : offset + peer_tokens].fill_(peer + 1)
        offset += peer_tokens

    assert result.shape == expected.shape, f"{result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _check_reduce_scatter(
    rsag, rank, world_size, tokens, hidden_size, device, reduce_scatter
):
    full = torch.full(
        (sum(tokens), hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = reduce_scatter(rsag, full, token_list_in_group=tokens)

    expected_value = world_size * (world_size + 1) // 2
    expected = torch.full(
        (tokens[rank], hidden_size),
        expected_value,
        dtype=torch.bfloat16,
        device=device,
    )

    assert result.shape == expected.shape, f"{result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _run_rsag_test(world_size: int, hidden_size: int) -> None:
    _skip_if_unsupported(world_size, "IrisRSAG tests")
    port = _get_open_port()
    _spawn_and_collect(_rsag_worker_fn, (world_size, port, hidden_size), world_size)


def test_iris_rsag_correctness_world2():
    _run_rsag_test(world_size=2, hidden_size=2880)


def test_iris_rsag_correctness_world4():
    _run_rsag_test(world_size=4, hidden_size=2880)


def test_iris_rsag_correctness_world8():
    _run_rsag_test(world_size=8, hidden_size=2880)


# ---------------------------------------------------------------------------
# Suite 3: fused allreduce + residual + RMSNorm
# ---------------------------------------------------------------------------


# Token shapes spanning decode (1), short/long prefill (256, 1024), and
# the full ``max_token_num`` (8192) so we exercise both the small-M code
# path and the path that walks the full symmetric heap buffer. Hidden=2880
# is the gpt-oss-120b size we use elsewhere.
_ARRMS_TOKEN_CASES: List[int] = [1, 64, 256, 1024, 8192]
_ARRMS_HIDDEN_DIM = 2880
_ARRMS_EPS = 1e-6


def _arrms_worker_fn(rank, world_size, port, persistent, error_dict):
    try:
        _arrms_worker_main(rank, world_size, port, persistent)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _arrms_worker_main(rank: int, world_size: int, port: int, persistent: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # NCCL is fine here — iris's heap-base exchange is host-side and works
    # the same over any default group.
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed_kernel.ops.communication.iris import (
            create_iris_ar_rmsnorm_state,
        )

        max_token_num = max(_ARRMS_TOKEN_CASES)
        state = create_iris_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=_ARRMS_HIDDEN_DIM,
            dtype=torch.bfloat16,
            persistent=persistent,
        )

        # Use a fixed RMSNorm weight that is *not* identity, so a bug in
        # the weight load path would fail the test.
        weight = torch.linspace(
            0.5, 1.5, _ARRMS_HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )

        for tokens in _ARRMS_TOKEN_CASES:
            _check_arrms_one(
                state,
                rank=rank,
                world_size=world_size,
                tokens=tokens,
                weight=weight,
                device=device,
            )
    finally:
        dist.destroy_process_group()


def _check_arrms_one(state, rank, world_size, tokens, weight, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import (
        iris_allreduce_residual_rmsnorm,
    )

    # Each rank contributes ``rank + 1``; sum across ranks is therefore
    # ``world_size * (world_size + 1) / 2``. Residual is non-uniform
    # (linspace) so the kernel can't accidentally short-circuit it.
    x = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM), rank + 1, dtype=torch.bfloat16, device=device
    )
    residual = (
        torch.arange(tokens * _ARRMS_HIDDEN_DIM, dtype=torch.float32, device=device)
        .reshape(tokens, _ARRMS_HIDDEN_DIM)
        .mul_(0.001)
        .to(torch.bfloat16)
    )

    norm_out, residual_out = iris_allreduce_residual_rmsnorm(
        state,
        input_tensor=x,
        residual=residual,
        weight=weight,
        eps=_ARRMS_EPS,
    )

    # Reference: do everything in fp32, mirroring the AMD test exactly so
    # tolerance differences only reflect implementation noise, not
    # reference noise.
    reduced = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _ARRMS_EPS
    )
    ref_norm = ref_norm * weight.float()

    torch.testing.assert_close(residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def _run_arrms_test(world_size: int, persistent: bool) -> None:
    _skip_if_unsupported(world_size, "Iris fused tests")
    port = _get_open_port()
    _spawn_and_collect(_arrms_worker_fn, (world_size, port, persistent), world_size)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world1(persistent: bool):
    # Single-rank smoke test: exercises the inline-barrier self-signal/wait
    # path (rank sends to itself) and the v1 device_barrier no-op case.
    _run_arrms_test(world_size=1, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world2(persistent: bool):
    _run_arrms_test(world_size=2, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world4(persistent: bool):
    _run_arrms_test(world_size=4, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world8(persistent: bool):
    _run_arrms_test(world_size=8, persistent=persistent)
