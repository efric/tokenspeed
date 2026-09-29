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

"""Public collective contracts: storage, selection, capacity and optional imports."""

import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel import selection
from tokenspeed_kernel.ops import communication
from tokenspeed_kernel.ops.communication import _residual
from tokenspeed_kernel.ops.communication._contracts import select_collective
from tokenspeed_kernel.ops.communication._iris import adapter, context, policy
from tokenspeed_kernel.ops.residual.attnres import AttnResEpilogue, AttnResRequirement
from tokenspeed_kernel.platform import ArchVersion, current_platform


def test_preparation_reuses_storage_and_rejects_late_growth(monkeypatch):
    monkeypatch.setattr(adapter, "producer_all_reduce_available", lambda: True)
    initialize = Mock()
    monkeypatch.setattr(adapter, "initialize_all_reduce_state", initialize)
    group = SimpleNamespace(size=lambda: 8)
    args = dict(
        group=group,
        rank_in_group=0,
        device=torch.device("cpu"),
        producer_direct_max_bytes=1024 * 1024,
    )
    demand = communication.AllReducePreparation(
        torch.bfloat16,
        (
            communication.AllReduceRequirement(8192, 7168),
            AttnResRequirement(8192, 7168),
            communication.PackedAllReduceRequirement(8192, (3584, 7168), 8, 1),
        ),
    )
    handle = communication.prepare_all_reduce_handle(
        preparation=demand, previous=None, **args
    )
    assert type(handle) is adapter.IrisAllReduceHandle
    assert handle.max_numel == 512 * 1024 // 2
    assert communication.all_reduce_capacity(handle) == 8192 * 10752 * 2
    assert handle.attnres_max_numel == 16 * 7168
    assert handle.enable_lamport
    assert (
        communication.prepare_all_reduce_handle(
            preparation=demand, previous=handle, **args
        )
        is handle
    )
    # A separately prepared group lacks the full shared-operation capacity.
    small = communication.AllReducePreparation(
        torch.bfloat16, (communication.AllReduceRequirement(1, 7168),)
    )
    small_handle = communication.prepare_all_reduce_handle(
        preparation=small, previous=None, **args
    )
    with pytest.raises(RuntimeError, match="below requested capacities"):
        communication.prepare_all_reduce_handle(
            preparation=communication.AllReducePreparation(
                torch.bfloat16, (communication.AllReduceRequirement(16, 7168),)
            ),
            previous=small_handle,
            **args,
        )


def test_staged_protocol_does_not_depend_on_local_pointer_alignment(monkeypatch):
    from tokenspeed_kernel.ops.communication._iris import all_reduce as implementation
    from tokenspeed_kernel.ops.communication._iris.all_reduce import IrisAllReduce

    monkeypatch.setattr(policy, "_platform", SimpleNamespace(is_cdna4=True))
    selected = []

    class Launch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: selected.append("one-shot")

    monkeypatch.setattr(
        implementation, "iris_stage_one_shot_allreduce_kernel", Launch()
    )
    state = IrisAllReduce.__new__(IrisAllReduce)
    state.dtype = torch.bfloat16
    state.staged_max_numel = 128
    state.world_size = 4
    state._kernel_config = policy.IRIS_ALL_REDUCE_KERNEL_CONFIG
    state._staged_two_stage_supported = True
    state.staged = SimpleNamespace(input=torch.empty(256), flags=torch.empty(1))
    state._group_heap_bases = None
    state._iris_rank = 0
    state._all_reduce_two_stage = lambda tensor, numel, safe: (
        selected.append("two-stage"),
        tensor,
    )[1]
    aligned = torch.empty(128, dtype=torch.bfloat16)
    unaligned = torch.empty(129, dtype=torch.bfloat16)[1:]
    # Simulate ranks with different pointers: geometry chooses one protocol.
    assert state.all_reduce(aligned, safe=False) is aligned
    assert state.all_reduce(unaligned, safe=False) is unaligned
    assert selected == ["two-stage", "two-stage"]


def _fusion_state(native, vendor_fusion):
    return _residual.ResidualAllReduceState(
        group=object(),
        rank=0,
        local_world_size=8,
        max_tokens=2048,
        vendor_fusion=vendor_fusion,
        native=native,
        unit_weight=torch.ones(64, dtype=torch.bfloat16),
    )


def _epilogue(rows):
    return AttnResEpilogue(
        partials=(torch.empty(rows), torch.empty(rows), torch.empty(rows, 64)),
        score_projection=torch.empty(64, dtype=torch.bfloat16),
        score_norm=torch.empty(64, dtype=torch.bfloat16),
        score_product=torch.empty(64, dtype=torch.bfloat16),
        output_weight=torch.empty(64, dtype=torch.bfloat16),
        eps=1e-6,
    )


def test_attnres_binding_selects_once_and_retains_operands(monkeypatch):
    monkeypatch.setattr(
        selection,
        "current_platform",
        lambda: replace(
            current_platform(), vendor="amd", arch_version=ArchVersion(9, 5)
        ),
    )
    monkeypatch.setattr(
        adapter,
        "allreduce_residual_attnres_combine_supported",
        lambda *args, **kwargs: True,
    )
    partial, residual = (torch.zeros(4, 64, dtype=torch.bfloat16) for _ in range(2))
    epilogue = _epilogue(4)
    run = Mock(return_value=(partial, residual))
    monkeypatch.setattr(adapter, "allreduce_residual_attnres_combine", run)
    binding = communication.bind_residual_all_reduce(
        _fusion_state(None, False), partial, residual, epilogue, False
    )
    assert binding.kernel.name == "iris_all_reduce_attnres_epilogue"
    assert binding.consumes_partials and binding.prefer_split_partials
    monkeypatch.setattr(
        _residual,
        "select_collective",
        Mock(side_effect=AssertionError("must not reselect")),
    )
    updated, hidden = binding(partial, residual)
    assert updated is residual and hidden is partial
    assert run.call_args.args[2] is epilogue.score_product
    assert run.call_args.args[4] is epilogue.partials


@pytest.mark.parametrize(
    "rows,name,consumes",
    [
        (1, "cute_all_reduce_residual", False),
        (8, "cute_all_reduce_residual", False),
        (9, "trtllm_all_reduce_attnres", True),
    ],
)
def test_nvidia_binding_preserves_winner_and_graph_policy(
    monkeypatch, rows, name, consumes
):
    monkeypatch.setattr(
        selection,
        "current_platform",
        lambda: replace(current_platform(), vendor="nvidia"),
    )
    tensor = torch.empty(rows, 64, dtype=torch.bfloat16)
    binding = communication.bind_residual_all_reduce(
        _fusion_state(object(), True), tensor, tensor, _epilogue(rows), False
    )
    assert binding.kernel.name == name
    assert binding.consumes_partials is consumes
    # NVIDIA's existing fused graph remains eligible at its original shapes.
    assert not binding.prefer_split_partials


def test_collective_override_cannot_change_prepared_operand_contract(monkeypatch):
    from tokenspeed_kernel.ops.communication import iris  # noqa: F401
    from tokenspeed_kernel.registry import KernelRegistry

    registry = KernelRegistry.get()
    name = "test_incompatible_collective"
    spec = registry.get_by_name("iris_all_reduce_attnres_epilogue")
    registry.register(replace(spec, name=name), lambda *args: None)

    monkeypatch.setattr(
        selection,
        "current_platform",
        lambda: replace(
            current_platform(), vendor="amd", arch_version=ArchVersion(9, 5)
        ),
    )
    try:
        with selection.kernel_override("communication", "all_reduce_attnres", name):
            with pytest.raises(ValueError, match="incompatible"):
                select_collective(
                    "all_reduce_attnres",
                    torch.bfloat16,
                    "iris_all_reduce_attnres_epilogue",
                )
    finally:
        registry._unregister(name)


def test_absent_optional_package_declines_before_preparation(monkeypatch):
    monkeypatch.setattr(context, "find_spec", lambda name: None)
    monkeypatch.setattr(
        adapter, "current_platform", lambda: SimpleNamespace(is_cdna4=True)
    )
    assert not communication.producer_all_reduce_available()
    assert (
        communication.prepare_all_reduce_handle(
            group=SimpleNamespace(size=lambda: 8),
            rank_in_group=0,
            device=torch.device("cpu"),
            preparation=communication.AllReducePreparation(torch.bfloat16, ()),
            previous=None,
            producer_direct_max_bytes=1024,
        )
        is None
    )


def test_public_import_does_not_load_optional_collective_implementations():
    code = """
import sys
from tokenspeed_kernel.ops import communication
assert 'iris' not in sys.modules
assert 'tokenspeed_kernel_amd.ops.gfx950.communication.all_reduce' not in sys.modules
assert 'tokenspeed_kernel_amd.ops.gfx950.communication.attnres' not in sys.modules
assert 'tokenspeed_kernel.thirdparty.iris' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=os.environ.copy(), timeout=30
    )


@pytest.mark.parametrize("vendor", ["amd", "nvidia"])
@pytest.mark.parametrize("launch_with_pdl", [False, None])
def test_rmsnorm_options_and_unsupported_tuple_survive_public_dispatch(
    monkeypatch, vendor, launch_with_pdl
):
    from tokenspeed_kernel.ops.communication import _rmsnorm, trtllm
    from tokenspeed_kernel.ops.communication._iris import epilogues

    platform = replace(current_platform(), vendor=vendor)
    monkeypatch.setattr(selection, "current_platform", lambda: platform)
    monkeypatch.setattr(_rmsnorm, "current_platform", lambda: platform)
    unsupported = (None, None, None, None)
    run = Mock(return_value=unsupported)
    monkeypatch.setattr(
        epilogues if vendor == "amd" else trtllm,
        "allreduce_residual_rmsnorm",
        run,
    )
    tensor = torch.empty(2, 64, dtype=torch.bfloat16)
    result = communication.allreduce_residual_rmsnorm(
        input_tensor=tensor,
        residual=tensor,
        weight=tensor[0],
        rank=1,
        group=object(),
        eps=1e-5,
        max_token_num=32,
        use_oneshot=False,
        trigger_completion_at_end=True,
        fp32_acc=True,
        block_quant_fp8=True,
        residual_reduce_scattered=True,
        has_partial_norm_out=True,
        max_sm_to_use=7,
        launch_with_pdl=launch_with_pdl,
    )
    assert result is unsupported
    assert run.call_args.kwargs["has_partial_norm_out"]
    assert run.call_args.kwargs["residual_reduce_scattered"]
    assert run.call_args.kwargs["block_quant_fp8"]
    assert run.call_args.kwargs["fp32_acc"]
    assert run.call_args.kwargs["max_sm_to_use"] == 7
    assert run.call_args.kwargs["trigger_completion_at_end"]
    assert run.call_args.kwargs["use_oneshot"] is False
    assert run.call_args.kwargs["launch_with_pdl"] is launch_with_pdl


def test_deterministic_control_keeps_existing_native_and_vendor_domains(monkeypatch):
    tensor = torch.empty(9, 64, dtype=torch.bfloat16)
    epilogue = _epilogue(9)
    assert (
        communication.bind_residual_all_reduce(
            _fusion_state(None, False),
            tensor,
            tensor,
            epilogue,
            True,
        )
        is None
    )
    monkeypatch.setattr(
        selection,
        "current_platform",
        lambda: replace(current_platform(), vendor="nvidia"),
    )
    binding = communication.bind_residual_all_reduce(
        _fusion_state(None, True),
        tensor,
        tensor,
        epilogue,
        True,
    )
    assert binding.kernel.name == "trtllm_all_reduce_attnres"
