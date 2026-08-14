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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed_kernel.ops.moe.megamoe.api import (
    _bind_plan_to_current_stream,
    _normalize_specs,
    kimi_k3_megamoe_decode,
    prepare_kimi_k3_megamoe,
)
from tokenspeed_kernel.ops.moe.megamoe.gluon import (
    _check_warmup_state,
    _raw_arguments,
    _validate_expert_ownership,
)
from tokenspeed_kernel.ops.moe.megamoe.types import (
    KimiK3MegaMoELayerSpec,
    _KimiK3MegaMoEStreamOwner,
)


def _minimal_mapping() -> dict[str, object]:
    tensor = torch.empty(1)
    return {
        "router_weight": tensor,
        "routed_down_weight": tensor,
        "shared_gate_up_weight": tensor,
        "shared_down_weight": tensor,
        "routed_norm_weight": tensor,
        "routed_up_weight": tensor,
        "correction_bias": tensor,
        "w13_weight": tensor,
        "w13_weight_scale": tensor,
        "w2_weight": tensor,
        "w2_weight_scale": tensor,
        "expert_start": 224,
    }


def test_layer_spec_mapping_preserves_exact_order_and_defaults() -> None:
    values = _minimal_mapping()
    spec = KimiK3MegaMoELayerSpec.from_mapping(values)

    assert spec.router_weight is values["router_weight"]
    assert spec.w2_weight_scale is values["w2_weight_scale"]
    assert spec.expert_start == 224
    assert spec.beta == 4.0
    assert spec.linear_beta == 25.0
    assert spec.rms_eps == 1.0e-5
    assert spec.w13_interleaved is False


def test_normalize_specs_requires_the_full_model_layer_set() -> None:
    with pytest.raises(ValueError, match="exactly 92 MoE layers, got 0"):
        _normalize_specs(())


def test_normalize_specs_identifies_the_failing_layer(monkeypatch) -> None:
    specs = tuple(
        KimiK3MegaMoELayerSpec.from_mapping(_minimal_mapping()) for _ in range(92)
    )

    def fail_validation(self) -> None:
        del self
        raise ValueError("sentinel ABI failure")

    monkeypatch.setattr(KimiK3MegaMoELayerSpec, "validate", fail_validation)
    with pytest.raises(ValueError, match="layer 0: sentinel ABI failure"):
        _normalize_specs(specs)


def test_decode_rejects_an_unprepared_plan_before_backend_import() -> None:
    tensor = torch.empty(1)
    with pytest.raises(TypeError, match="requires a prepared layer plan"):
        kimi_k3_megamoe_decode(tensor, tensor, object())


def test_plan_stream_owner_rejects_a_second_host_stream(monkeypatch) -> None:
    owner = _KimiK3MegaMoEStreamOwner()
    plan = SimpleNamespace(
        spec=SimpleNamespace(device=torch.device("cuda:0")),
        _stream_owner=owner,
    )

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=101),
    )
    _bind_plan_to_current_stream(plan)
    _bind_plan_to_current_stream(plan)
    assert owner.stream_handle == 101

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=202),
    )
    with pytest.raises(RuntimeError, match="already bound to another CUDA stream"):
        _bind_plan_to_current_stream(plan)


def test_all_layer_plans_share_one_stream_owner(monkeypatch) -> None:
    from tokenspeed_kernel.ops.moe.megamoe import gluon as gluon_module

    monkeypatch.setattr(
        gluon_module,
        "KimiK3MegaMoELayerPlan",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    specs = tuple(object() for _ in range(92))
    outputs = tuple(object() for _ in range(92))
    plans = gluon_module._make_layer_plans(
        specs,
        object(),
        {},
        outputs,
        object(),
        1_000_000_000,
    )

    assert len(plans) == 92
    assert len({id(plan._stream_owner) for plan in plans}) == 1
    assert tuple(plan.layer_output for plan in plans) == outputs


def test_prepare_exchanges_a_rank_local_host_failure(monkeypatch) -> None:
    stages: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.moe.megamoe.api.kimi_k3_megamoe_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError, match="host consensus returned after a local failure"
    ):
        prepare_kimi_k3_megamoe(
            (),
            object(),
            like=torch.empty(1),
            consensus=lambda stage, status, reason: stages.append(
                (stage, status, reason)
            ),
        )

    assert len(stages) == 1
    stage, status, reason = stages[0]
    assert stage == "host_prepare"
    assert status == 1
    assert "requires the gfx950 Gluon implementation" in reason


def test_vendor_shim_emits_the_flat_51_tensor_abi_in_order() -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe import megamoe as amd_megamoe

    tensors = iter(torch.tensor(index) for index in range(100))
    hidden = next(tensors)
    prefix = next(tensors)
    layer_output = next(tensors)
    spec = SimpleNamespace(
        router_weight=next(tensors),
        routed_down_weight=next(tensors),
        shared_gate_up_weight=next(tensors),
        shared_down_weight=next(tensors),
        routed_norm_weight=next(tensors),
        routed_up_weight=next(tensors),
        correction_bias=next(tensors),
        w13_weight=next(tensors),
        w13_weight_scale=next(tensors),
        w2_weight=next(tensors),
        w2_weight_scale=next(tensors),
    )
    workspace_values = tuple(
        next(tensors) for _ in amd_megamoe.kimi_k3_megamoe_workspace_spec()
    )
    workspace = {
        tensor_spec.name: value
        for tensor_spec, value in zip(
            amd_megamoe.kimi_k3_megamoe_workspace_spec(),
            workspace_values,
            strict=True,
        )
    }
    iris_values = tuple(next(tensors) for _ in range(7))
    lane = SimpleNamespace(
        symmetric_producer=iris_values[0],
        symmetric_reduced=iris_values[1],
        iris_epoch_flags=iris_values[2],
        topology_status=iris_values[3],
        fatal_epoch=iris_values[4],
        heap_bases=iris_values[5],
        group_global_ranks=iris_values[6],
    )
    plan = SimpleNamespace(
        spec=spec,
        workspace=workspace,
        layer_output=layer_output,
    )

    raw = _raw_arguments(hidden, prefix, plan, lane, amd_megamoe)

    expected_iris = iris_values[:3] + iris_values[4:]
    assert len(raw) == 51
    assert raw[:3] == (hidden, prefix, layer_output)
    assert raw[14:45] == workspace_values
    assert raw[45:] == expected_iris


def test_vendor_shim_ties_expert_shard_to_ordered_iris_rank() -> None:
    specs = tuple(SimpleNamespace(expert_start=224) for _ in range(92))
    _validate_expert_ownership(specs, 2)

    mismatched = list(specs)
    mismatched[37] = SimpleNamespace(expert_start=336)
    with pytest.raises(
        RuntimeError,
        match=r"EP rank 2: expected expert_start=224, mismatched layers=\[37\]",
    ):
        _validate_expert_ownership(tuple(mismatched), 2)


def test_warmup_generation_uses_topology_and_keeps_phase_tensors_unused() -> None:
    generation = 3
    workspace = {
        "fail_diagnostics": torch.zeros(8, dtype=torch.int64),
        "phase_arrival": torch.zeros(1, dtype=torch.int64),
        "phase_gate": torch.zeros(1, dtype=torch.int64),
        "xcc_ticket": torch.full((8,), generation * 30, dtype=torch.int64),
        "xcd_arrival": torch.tensor([generation * 8], dtype=torch.int64),
        "topology_gate": torch.tensor([generation], dtype=torch.int64),
    }
    plan = SimpleNamespace(
        workspace=workspace,
        layer_output=torch.zeros(1, dtype=torch.bfloat16),
    )
    lane = SimpleNamespace(fatal_epoch=torch.zeros(1, dtype=torch.int64))

    _check_warmup_state(plan, lane)

    for name in ("phase_arrival", "phase_gate"):
        workspace[name].fill_(1)
        with pytest.raises(
            RuntimeError,
            match="invalid topology ticket generation",
        ):
            _check_warmup_state(plan, lane)
        workspace[name].zero_()
