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

"""Vendor-bound preparation shim for the gfx950 Kimi K3 MegaMoE."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from tokenspeed_kernel.ops.communication import unpack_producer_direct_lane
from tokenspeed_kernel.ops.moe.megamoe.types import (
    KimiK3MegaMoELayerPlan,
    KimiK3MegaMoELayerSpec,
    _KimiK3MegaMoEStreamOwner,
)


_LANE_SHAPES = ((1, 7168), (1, 3584))
_LAST_ADMISSION_REPORT: dict[str, Any] | None = None


def _amd_api():
    from tokenspeed_kernel_amd.ops.gfx950.moe import megamoe as amd_megamoe
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.admission import (
        preflight_kimi_k3_megamoe_runtime,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
        launch_kimi_k3_megamoe_gfx950,
        prepare_kimi_k3_megamoe_gfx950,
    )

    return (
        amd_megamoe,
        preflight_kimi_k3_megamoe_runtime,
        prepare_kimi_k3_megamoe_gfx950,
        launch_kimi_k3_megamoe_gfx950,
    )


def implementation_available() -> bool:
    """Return whether the complete gfx950 implementation passes host preflight."""

    try:
        amd_megamoe, preflight, _, _ = _amd_api()
        if not amd_megamoe.KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE:
            return False
        preflight()
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False
    return True


def _validate_lane(lane: object):
    unpacked = unpack_producer_direct_lane(lane)
    lane_tensors = (
        unpacked.symmetric_producer,
        unpacked.symmetric_reduced,
        unpacked.iris_epoch_flags,
        unpacked.topology_status,
        unpacked.fatal_epoch,
        unpacked.heap_bases,
        unpacked.group_global_ranks,
    )
    if (
        unpacked.shapes != _LANE_SHAPES
        or unpacked.world_size != 8
        or unpacked.flag_dtype != torch.int64
        or unpacked.block_size != 512
        or unpacked.num_programs != 21
        or unpacked.num_tiles != 21
        or unpacked.iris_epoch_flags.shape != (2, 21, 8)
        or unpacked.topology_status.shape != (2,)
        or any(tensor.data_ptr() % 16 != 0 for tensor in lane_tensors)
    ):
        raise RuntimeError("Kimi K3 MegaMoE received an incompatible Iris lane")
    return unpacked


def _validate_expert_ownership(
    specs: Sequence[KimiK3MegaMoELayerSpec], group_rank: int
) -> None:
    """Tie the processed expert shard to the ordered Iris EP rank."""

    expected_start = 112 * group_rank
    mismatched = [
        index for index, spec in enumerate(specs) if spec.expert_start != expected_start
    ]
    if mismatched:
        raise RuntimeError(
            "Kimi K3 MegaMoE expert ownership does not match the ordered "
            f"Iris EP rank {group_rank}: expected expert_start={expected_start}, "
            f"mismatched layers={mismatched[:8]}"
        )


def _workspace_tuple(
    workspace: Mapping[str, torch.Tensor], amd_megamoe
) -> tuple[torch.Tensor, ...]:
    return tuple(
        workspace[tensor_spec.name]
        for tensor_spec in amd_megamoe.kimi_k3_megamoe_workspace_spec()
    )


def _raw_arguments_from_parts(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    layer_output: torch.Tensor,
    spec: KimiK3MegaMoELayerSpec,
    workspace: Mapping[str, torch.Tensor],
    unpacked_lane,
    amd_megamoe,
) -> tuple[torch.Tensor, ...]:
    return (
        hidden_states,
        prefix_sum,
        layer_output,
        spec.router_weight,
        spec.routed_down_weight,
        spec.shared_gate_up_weight,
        spec.shared_down_weight,
        spec.routed_norm_weight,
        spec.routed_up_weight,
        spec.correction_bias,
        spec.w13_weight,
        spec.w13_weight_scale,
        spec.w2_weight,
        spec.w2_weight_scale,
        *_workspace_tuple(workspace, amd_megamoe),
        unpacked_lane.symmetric_producer,
        unpacked_lane.symmetric_reduced,
        unpacked_lane.iris_epoch_flags,
        unpacked_lane.fatal_epoch,
        unpacked_lane.heap_bases,
        unpacked_lane.group_global_ranks,
    )


def _raw_arguments(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    plan: KimiK3MegaMoELayerPlan,
    unpacked_lane,
    amd_megamoe,
) -> tuple[torch.Tensor, ...]:
    return _raw_arguments_from_parts(
        hidden_states,
        prefix_sum,
        plan.layer_output,
        plan.spec,
        plan.workspace,
        unpacked_lane,
        amd_megamoe,
    )


def _bounded_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}".replace("\x00", "")[:512]


def _make_layer_plans(
    specs: Sequence[KimiK3MegaMoELayerSpec],
    lane: object,
    workspace: Mapping[str, torch.Tensor],
    layer_outputs: Sequence[torch.Tensor],
    prepared_kernel: object,
    timeout_ns: int,
) -> tuple[KimiK3MegaMoELayerPlan, ...]:
    """Build all layer views with one stream owner for their shared state."""

    stream_owner = _KimiK3MegaMoEStreamOwner()
    return tuple(
        KimiK3MegaMoELayerPlan(
            spec=spec,
            lane=lane,
            workspace=workspace,
            layer_output=layer_outputs[index],
            prepared_kernel=prepared_kernel,
            timeout_ns=timeout_ns,
            _stream_owner=stream_owner,
        )
        for index, spec in enumerate(specs)
    )


def _check_warmup_state(
    plan: KimiK3MegaMoELayerPlan,
    unpacked_lane,
) -> None:
    workspace = plan.workspace
    fatal_epoch = int(unpacked_lane.fatal_epoch.item())
    if fatal_epoch != 0:
        diagnostics = workspace["fail_diagnostics"].cpu().tolist()
        raise RuntimeError(
            f"MegaMoE warmup poisoned epoch {fatal_epoch}: {diagnostics}"
        )
    generation = int(workspace["topology_gate"].item())
    phase_arrival = int(workspace["phase_arrival"].item())
    phase_gate = int(workspace["phase_gate"].item())
    tickets = workspace["xcc_ticket"].cpu().tolist()
    xcd_arrival = int(workspace["xcd_arrival"].item())
    if (
        generation <= 0
        or phase_arrival != 0
        or phase_gate != 0
        or tickets != [generation * 30] * 8
        or xcd_arrival != generation * 8
    ):
        raise RuntimeError(
            "MegaMoE warmup observed an invalid topology ticket generation: "
            f"generation={generation}, phase_arrival={phase_arrival}, "
            f"phase_gate={phase_gate}, tickets={tickets}, "
            f"xcd_arrival={xcd_arrival}"
        )
    if not bool(torch.isfinite(plan.layer_output).all().item()):
        raise RuntimeError("MegaMoE collective warmup produced non-finite output")


def prepare_gfx950(
    specs: Sequence[KimiK3MegaMoELayerSpec],
    lane: object,
    *,
    like: torch.Tensor,
    consensus: Callable[[str, int, str], None],
    timeout_ns: int,
) -> tuple[KimiK3MegaMoELayerPlan, ...]:
    """Allocate, compile, admit, and collectively warm one model-level plan."""

    local_error: Exception | None = None
    prepared: tuple[KimiK3MegaMoELayerPlan, ...] = ()
    unpacked_lane = None
    amd_megamoe = None
    launch = None
    prepared_kernel = None
    try:
        (
            amd_megamoe,
            preflight,
            prepare_kernel,
            launch,
        ) = _amd_api()
        if not amd_megamoe.KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE:
            raise RuntimeError("gfx950 MegaMoE implementation is not complete")
        runtime_report = preflight()
        unpacked_lane = _validate_lane(lane)
        _validate_expert_ownership(specs, unpacked_lane.group_rank)
        workspace = amd_megamoe.allocate_kimi_k3_megamoe_workspace(like.device)
        layer_outputs = torch.empty(
            (len(specs), 1, 7168),
            dtype=torch.bfloat16,
            device=like.device,
        )
        warm_hidden = torch.zeros_like(layer_outputs[0])
        warm_prefix = torch.zeros_like(layer_outputs[0])
        raw_args = _raw_arguments_from_parts(
            warm_hidden,
            warm_prefix,
            layer_outputs[0],
            specs[0],
            workspace,
            unpacked_lane,
            amd_megamoe,
        )
        raw_kwargs = {
            "expert_start": specs[0].expert_start,
            "group_rank": unpacked_lane.group_rank,
            "timeout_ns": timeout_ns,
            "beta": specs[0].beta,
            "linear_beta": specs[0].linear_beta,
            "rms_eps": specs[0].rms_eps,
        }
        prepared_kernel = prepare_kernel(
            *raw_args,
            **raw_kwargs,
        )
        prepared = _make_layer_plans(
            specs,
            lane,
            workspace,
            layer_outputs,
            prepared_kernel,
            timeout_ns,
        )
        global _LAST_ADMISSION_REPORT
        _LAST_ADMISSION_REPORT = {
            "runtime": runtime_report.as_dict(),
            "code": prepared_kernel.admission.as_dict(),
        }
    except Exception as error:  # Exchanged before any collective kernel launch.
        local_error = error

    consensus(
        "compile_resource",
        int(local_error is not None),
        "" if local_error is None else _bounded_reason(local_error),
    )
    if local_error is not None:
        raise RuntimeError(
            "Kimi K3 MegaMoE compile consensus returned after a local failure"
        ) from local_error
    assert prepared and unpacked_lane is not None and amd_megamoe is not None
    assert launch is not None

    warmup_error: Exception | None = None
    try:
        warm_hidden = torch.zeros_like(prepared[0].layer_output)
        warm_prefix = torch.zeros_like(prepared[0].layer_output)
        launch(
            prepared[0].prepared_kernel,
            *_raw_arguments(
                warm_hidden,
                warm_prefix,
                prepared[0],
                unpacked_lane,
                amd_megamoe,
            ),
        )
        torch.cuda.synchronize(like.device)
        _check_warmup_state(prepared[0], unpacked_lane)
    except Exception as error:  # All ranks compiled before this collective stage.
        warmup_error = error
    consensus(
        "collective_warmup",
        int(warmup_error is not None),
        "" if warmup_error is None else _bounded_reason(warmup_error),
    )
    if warmup_error is not None:
        raise RuntimeError(
            "Kimi K3 MegaMoE warmup consensus returned after a local failure"
        ) from warmup_error
    return prepared


def launch_gfx950(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    plan: KimiK3MegaMoELayerPlan,
) -> None:
    """Launch one already-admitted graph-stable gfx950 plan."""

    amd_megamoe, _, _, launch = _amd_api()
    unpacked_lane = _validate_lane(plan.lane)
    launch(
        plan.prepared_kernel,
        *_raw_arguments(
            hidden_states,
            prefix_sum,
            plan,
            unpacked_lane,
            amd_megamoe,
        ),
    )


def last_admission_report() -> dict[str, Any] | None:
    """Return a shallow copy of the most recent local admission report."""

    return None if _LAST_ADMISSION_REPORT is None else dict(_LAST_ADMISSION_REPORT)


__all__ = [
    "implementation_available",
    "last_admission_report",
    "launch_gfx950",
    "prepare_gfx950",
]
