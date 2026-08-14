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

"""Immutable host-side ABI records for the experimental Kimi K3 MegaMoE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch


class _KimiK3MegaMoEStreamOwner:
    """Bind one shared MegaMoE workspace/lane to one host launch stream.

    The token is deliberately host-only. CUDA/HIP graph replay does not invoke
    Python, so the supported TokenSpeed runtime must replay on the same stream
    that first called the public decode wrapper during graph warmup/capture.
    """

    __slots__ = ("_stream_handle",)

    def __init__(self) -> None:
        self._stream_handle: int | None = None

    @property
    def stream_handle(self) -> int | None:
        """Return the bound raw stream handle, or ``None`` before first decode."""

        return self._stream_handle

    def bind(self, stream_handle: int) -> None:
        """Bind once and reject use of the shared state from another stream."""

        stream_handle = int(stream_handle)
        if self._stream_handle is None:
            self._stream_handle = stream_handle
        elif self._stream_handle != stream_handle:
            raise RuntimeError(
                "Kimi K3 MegaMoE shared state is already bound to another CUDA "
                "stream; direct or external concurrent graph replay is unsupported"
            )


_EXPERT_TENSOR_ABI = {
    "w13_weight": (
        (112, 6144, 1792),
        torch.uint8,
        (11010048, 1792, 1),
    ),
    "w13_weight_scale": (
        (112, 6144, 112),
        torch.uint8,
        (688128, 112, 1),
    ),
    "w2_weight": (
        (112, 3584, 1536),
        torch.uint8,
        (5505024, 1536, 1),
    ),
    "w2_weight_scale": (
        (112, 3584, 96),
        torch.uint8,
        (344064, 96, 1),
    ),
}

_BF16_WEIGHT_SHAPES = {
    "router_weight": (896, 7168),
    "routed_down_weight": (3584, 7168),
    "shared_gate_up_weight": (1536, 7168),
    "shared_down_weight": (7168, 768),
    "routed_norm_weight": (3584,),
    "routed_up_weight": (7168, 3584),
}


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    strides: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Kimi K3 MegaMoE {name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Kimi K3 MegaMoE {name} must have shape {shape}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(
            f"Kimi K3 MegaMoE {name} must have dtype {dtype}, got {tensor.dtype}"
        )
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(
            f"Kimi K3 MegaMoE {name} must be on CUDA device {device}, "
            f"got {tensor.device}"
        )
    if tensor.stride(-1) != 1:
        raise ValueError(f"Kimi K3 MegaMoE {name} must have unit inner stride")
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"Kimi K3 MegaMoE {name} must be 16-byte aligned")
    if strides is None:
        if not tensor.is_contiguous():
            raise ValueError(f"Kimi K3 MegaMoE {name} must be contiguous")
    elif tuple(tensor.stride()) != strides:
        raise ValueError(
            f"Kimi K3 MegaMoE {name} must have strides {strides}, "
            f"got {tuple(tensor.stride())}"
        )


@dataclass(frozen=True, slots=True)
class KimiK3MegaMoELayerSpec:
    """Exact processed-weight ABI for one Kimi K3 MoE layer.

    Every tensor is graph-stable and already processed by the native gfx950
    MXFP4 plan. W13 uses the concatenated gate-then-up row layout.
    """

    router_weight: torch.Tensor
    routed_down_weight: torch.Tensor
    shared_gate_up_weight: torch.Tensor
    shared_down_weight: torch.Tensor
    routed_norm_weight: torch.Tensor
    routed_up_weight: torch.Tensor
    correction_bias: torch.Tensor
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    expert_start: int
    beta: float = 4.0
    linear_beta: float = 25.0
    rms_eps: float = 1.0e-5
    w13_interleaved: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KimiK3MegaMoELayerSpec":
        """Construct a layer ABI record from an ordered runtime mapping."""

        return cls(
            router_weight=values["router_weight"],
            routed_down_weight=values["routed_down_weight"],
            shared_gate_up_weight=values["shared_gate_up_weight"],
            shared_down_weight=values["shared_down_weight"],
            routed_norm_weight=values["routed_norm_weight"],
            routed_up_weight=values["routed_up_weight"],
            correction_bias=values["correction_bias"],
            w13_weight=values["w13_weight"],
            w13_weight_scale=values["w13_weight_scale"],
            w2_weight=values["w2_weight"],
            w2_weight_scale=values["w2_weight_scale"],
            expert_start=int(values["expert_start"]),
            beta=float(values.get("beta", 4.0)),
            linear_beta=float(values.get("linear_beta", 25.0)),
            rms_eps=float(values.get("rms_eps", 1.0e-5)),
            w13_interleaved=bool(values.get("w13_interleaved", False)),
        )

    @property
    def device(self) -> torch.device:
        return self.router_weight.device

    def validate(self) -> None:
        """Validate shapes, types, strides, device, and specialization scalars."""

        device = self.router_weight.device
        for name, shape in _BF16_WEIGHT_SHAPES.items():
            _require_tensor(
                name,
                getattr(self, name),
                shape=shape,
                dtype=torch.bfloat16,
                device=device,
            )
        _require_tensor(
            "correction_bias",
            self.correction_bias,
            shape=(896,),
            dtype=torch.float32,
            device=device,
        )
        for name, (shape, dtype, strides) in _EXPERT_TENSOR_ABI.items():
            _require_tensor(
                name,
                getattr(self, name),
                shape=shape,
                dtype=dtype,
                device=device,
                strides=strides,
            )
        if self.w13_interleaved:
            raise ValueError("Kimi K3 MegaMoE requires concatenated W13 rows")
        if self.expert_start not in range(0, 896, 112):
            raise ValueError(
                "Kimi K3 MegaMoE expert_start must identify a contiguous EP8 "
                f"shard, got {self.expert_start}"
            )
        if self.beta != 4.0 or self.linear_beta != 25.0:
            raise ValueError("Kimi K3 MegaMoE requires SiTU beta=4 and linear beta=25")
        if self.rms_eps != 1.0e-5:
            raise ValueError("Kimi K3 MegaMoE requires rms_eps=1e-5")


@dataclass(frozen=True, slots=True)
class KimiK3MegaMoELayerPlan:
    """Graph-stable prepared plan for one Kimi K3 MegaMoE layer.

    Args:
        spec: Exact processed layer-weight record.
        lane: Opaque producer-direct communication lane owned by
            ``tokenspeed-kernel``.
        workspace: Shared model-level local scratch views keyed by the raw AMD
            ABI names.
        layer_output: Unique contiguous BF16 ``[1, 7168]`` output for this layer.
        prepared_kernel: Strong reference to the one loaded and admitted raw
            specialization shared by every layer on this rank.
        timeout_ns: Bounded device-poll timeout used only for fail-stop recovery.
        _stream_owner: Shared host token that binds all 92 plans, their workspace,
            and their Iris lane to the TokenSpeed model execution stream.
    """

    spec: KimiK3MegaMoELayerSpec
    lane: object
    workspace: Mapping[str, torch.Tensor]
    layer_output: torch.Tensor
    prepared_kernel: object
    timeout_ns: int
    _stream_owner: _KimiK3MegaMoEStreamOwner = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", MappingProxyType(dict(self.workspace)))
        if self.prepared_kernel is None:
            raise ValueError("Kimi K3 MegaMoE requires a prepared raw kernel")
        if self.timeout_ns <= 0:
            raise ValueError("Kimi K3 MegaMoE timeout_ns must be positive")
        if not isinstance(self._stream_owner, _KimiK3MegaMoEStreamOwner):
            raise TypeError("Kimi K3 MegaMoE plans require a shared stream owner")
        _require_tensor(
            "layer_output",
            self.layer_output,
            shape=(1, 7168),
            dtype=torch.bfloat16,
            device=self.spec.device,
        )


__all__ = ["KimiK3MegaMoELayerPlan", "KimiK3MegaMoELayerSpec"]
