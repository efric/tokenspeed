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

"""Flat workspace contract for the gfx950 Kimi K3 MegaMoE kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MegaMoETensorSpec:
    """One graph-stable tensor in the raw MegaMoE workspace."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


_WORKSPACE_SPEC = (
    MegaMoETensorSpec("router_logits", (896,), torch.float32),
    MegaMoETensorSpec("routed_input", (3584,), torch.bfloat16),
    MegaMoETensorSpec("shared_input", (768,), torch.bfloat16),
    MegaMoETensorSpec("topk_ids", (16,), torch.int32),
    MegaMoETensorSpec("topk_weights", (16,), torch.float32),
    MegaMoETensorSpec("local_route_ids", (16,), torch.int32),
    MegaMoETensorSpec("local_route_slots", (16,), torch.int32),
    MegaMoETensorSpec("local_route_weights", (16,), torch.float32),
    MegaMoETensorSpec("local_route_count", (1,), torch.int32),
    MegaMoETensorSpec("route_xcd_start", (16,), torch.int32),
    MegaMoETensorSpec("route_xcd_count", (16,), torch.int32),
    MegaMoETensorSpec("route_worker_count", (16,), torch.int32),
    MegaMoETensorSpec("w13_target", (16,), torch.int64),
    MegaMoETensorSpec("w2_target", (16,), torch.int64),
    MegaMoETensorSpec("w13_intermediate", (16, 3072), torch.bfloat16),
    MegaMoETensorSpec("w2_route_output", (16, 3584), torch.bfloat16),
    MegaMoETensorSpec("xcc_ticket", (8,), torch.int64),
    MegaMoETensorSpec("xcd_arrival", (1,), torch.int64),
    MegaMoETensorSpec("phase_arrival", (1,), torch.int64),
    MegaMoETensorSpec("phase_gate", (1,), torch.int64),
    MegaMoETensorSpec("topology_gate", (1,), torch.int64),
    MegaMoETensorSpec("route_plan_gate", (1,), torch.int64),
    MegaMoETensorSpec("shared_arrival", (14,), torch.int64),
    MegaMoETensorSpec("shared_gate", (14,), torch.int64),
    MegaMoETensorSpec("w13_arrival", (16,), torch.int64),
    MegaMoETensorSpec("w13_gate", (16,), torch.int64),
    MegaMoETensorSpec("w2_arrival", (16,), torch.int64),
    MegaMoETensorSpec("w2_gate", (16,), torch.int64),
    MegaMoETensorSpec("comm_arrival", (1,), torch.int64),
    MegaMoETensorSpec("comm_gate", (1,), torch.int64),
    MegaMoETensorSpec("fail_diagnostics", (8,), torch.int64),
)


def kimi_k3_megamoe_workspace_spec() -> tuple[MegaMoETensorSpec, ...]:
    """Return local, nonsymmetric workspace tensors in flat ABI order."""

    return _WORKSPACE_SPEC


def allocate_kimi_k3_megamoe_workspace(
    device: torch.device | str | int,
) -> dict[str, torch.Tensor]:
    """Allocate and zero the graph-stable local workspace.

    Iris producer/reduced buffers and synchronization state are borrowed from
    the communication backend and intentionally excluded.
    """

    return {
        spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=device)
        for spec in _WORKSPACE_SPEC
    }


def validate_workspace_tensors(
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
) -> None:
    """Validate local workspace tensors supplied in the documented ABI order."""

    if len(tensors) != len(_WORKSPACE_SPEC):
        raise ValueError(
            f"MegaMoE requires {len(_WORKSPACE_SPEC)} workspace tensors, got {len(tensors)}"
        )
    for tensor, spec in zip(tensors, _WORKSPACE_SPEC, strict=True):
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"{spec.name} must have shape {spec.shape}")
        if tensor.dtype != spec.dtype:
            raise TypeError(f"{spec.name} must have dtype {spec.dtype}")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{spec.name} must be contiguous on {device}")


__all__ = [
    "MegaMoETensorSpec",
    "allocate_kimi_k3_megamoe_workspace",
    "kimi_k3_megamoe_workspace_spec",
    "validate_workspace_tensors",
]
