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

"""Public experimental Kimi K3 MegaMoE preparation and decode API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from tokenspeed_kernel.ops.moe.megamoe.types import (
    KimiK3MegaMoELayerPlan,
    KimiK3MegaMoELayerSpec,
)
from tokenspeed_kernel.platform import current_platform

_DEFAULT_TIMEOUT_NS = 1_000_000_000


def _bind_plan_to_current_stream(plan: KimiK3MegaMoELayerPlan) -> None:
    """Bind shared plan state before a host-visible direct kernel dispatch."""

    stream = torch.cuda.current_stream(plan.spec.device)
    plan._stream_owner.bind(int(stream.cuda_stream))


def kimi_k3_megamoe_available() -> bool:
    """Return whether the exact gfx950 MegaMoE implementation is importable."""

    if not current_platform().is_cdna4:
        return False
    try:
        from tokenspeed_kernel.ops.moe.megamoe.gluon import implementation_available

        return implementation_available()
    except (ImportError, OSError):
        return False


def _normalize_specs(
    layer_specs: Sequence[KimiK3MegaMoELayerSpec | Mapping[str, Any]],
) -> tuple[KimiK3MegaMoELayerSpec, ...]:
    specs = tuple(
        (
            value
            if isinstance(value, KimiK3MegaMoELayerSpec)
            else KimiK3MegaMoELayerSpec.from_mapping(value)
        )
        for value in layer_specs
    )
    if len(specs) != 92:
        raise ValueError(
            f"Kimi K3 MegaMoE requires exactly 92 MoE layers, got {len(specs)}"
        )
    first_device = specs[0].device
    for index, spec in enumerate(specs):
        try:
            spec.validate()
        except (TypeError, ValueError) as error:
            raise type(error)(f"layer {index}: {error}") from error
        if spec.device != first_device:
            raise ValueError("Kimi K3 MegaMoE layer tensors must share one device")
    return specs


def prepare_kimi_k3_megamoe(
    layer_specs: Sequence[KimiK3MegaMoELayerSpec | Mapping[str, Any]],
    lane: object,
    *,
    like: torch.Tensor,
    consensus: Callable[[str, int, str], None],
    timeout_ns: int = _DEFAULT_TIMEOUT_NS,
) -> tuple[KimiK3MegaMoELayerPlan, ...]:
    """Prepare all graph-stable Kimi K3 MegaMoE layer plans.

    Args:
        layer_specs: Exactly 92 ordered processed-weight records.
        lane: Opaque producer-direct lane acquired from the communication
            backend for shapes ``((1, 7168), (1, 3584))``.
        like: CUDA BF16 tensor providing the target device.
        consensus: Runtime-injected rank-uniform status exchange. Preparation
            invokes it after local compilation/resource admission and before
            any collective warmup; it must raise identically on every rank
            when any rank reports failure.
        timeout_ns: Positive fail-stop timeout for local and peer device polls.

    Returns:
        One immutable layer plan per ordered MoE layer. All plans share one
        local workspace and lane while owning distinct output rows.
    """

    if not callable(consensus):
        raise TypeError("Kimi K3 MegaMoE consensus must be callable")

    # Lane acquisition has already completed, so every rank must reach the
    # injected control-group exchange even if a rank-local host check fails.
    # Otherwise healthy peers could enter the later Iris warmup alone.
    local_error: Exception | None = None
    specs: tuple[KimiK3MegaMoELayerSpec, ...] = ()
    prepare_gfx950 = None
    try:
        if not kimi_k3_megamoe_available():
            raise RuntimeError(
                "Kimi K3 MegaMoE requires the gfx950 Gluon implementation"
            )
        if (
            not isinstance(like, torch.Tensor)
            or not like.is_cuda
            or like.dtype != torch.bfloat16
        ):
            raise ValueError("Kimi K3 MegaMoE like tensor must be CUDA BF16")
        if timeout_ns <= 0:
            raise ValueError("Kimi K3 MegaMoE timeout_ns must be positive")
        specs = _normalize_specs(layer_specs)
        if specs[0].device != like.device:
            raise ValueError("Kimi K3 MegaMoE like tensor must share the layer device")
        from tokenspeed_kernel.ops.moe.megamoe.gluon import (
            prepare_gfx950 as imported_prepare_gfx950,
        )

        prepare_gfx950 = imported_prepare_gfx950
    except Exception as error:  # Exchanged before any collective kernel launch.
        local_error = error

    reason = (
        ""
        if local_error is None
        else f"{type(local_error).__name__}: {local_error}"[:512]
    )
    consensus("host_prepare", int(local_error is not None), reason)
    if local_error is not None:
        raise RuntimeError(
            "Kimi K3 MegaMoE host consensus returned after a local failure"
        ) from local_error
    assert prepare_gfx950 is not None

    return prepare_gfx950(
        specs,
        lane,
        like=like,
        consensus=consensus,
        timeout_ns=int(timeout_ns),
    )


def kimi_k3_megamoe_decode(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    plan: KimiK3MegaMoELayerPlan,
) -> torch.Tensor:
    """Run one complete Kimi K3 MoE sublayer in one persistent kernel.

    This experimental entry point supports TokenSpeed's serialized model
    execution stream. Direct use of the shared plan from another stream, and
    replay of an externally owned graph on another stream, are unsupported.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[1, 7168]``.
        prefix_sum: Contiguous BF16 AttnRes prefix shaped ``[1, 7168]``.
        plan: Prepared immutable plan for the current layer.

    Returns:
        The plan-owned contiguous BF16 output shaped ``[1, 7168]``.
    """

    if not isinstance(plan, KimiK3MegaMoELayerPlan):
        raise TypeError("Kimi K3 MegaMoE requires a prepared layer plan")
    for name, value in (
        ("hidden_states", hidden_states),
        ("prefix_sum", prefix_sum),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (1, 7168)
            or value.dtype != torch.bfloat16
            or value.device != plan.spec.device
            or not value.is_contiguous()
            or value.data_ptr() % 16 != 0
        ):
            raise ValueError(
                f"Kimi K3 MegaMoE {name} must be contiguous colocated "
                "16-byte-aligned CUDA BF16 [1, 7168]"
            )
    addresses = {
        hidden_states.data_ptr(),
        prefix_sum.data_ptr(),
        plan.layer_output.data_ptr(),
    }
    if len(addresses) != 3:
        raise ValueError("Kimi K3 MegaMoE input, prefix, and output may not alias")

    # All 92 plans share one workspace and one Iris lane. Bind before importing
    # or entering the raw launcher so a direct second-stream call fails without
    # advancing an exact-equality gate or overwriting in-flight payload.
    _bind_plan_to_current_stream(plan)

    from tokenspeed_kernel.ops.moe.megamoe.gluon import launch_gfx950

    launch_gfx950(hidden_states, prefix_sum, plan)
    return plan.layer_output


__all__ = [
    "kimi_k3_megamoe_available",
    "kimi_k3_megamoe_decode",
    "prepare_kimi_k3_megamoe",
]
