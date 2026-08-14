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

"""Bitwise route parity between the MegaMoE Gluon body and current Triton."""

from __future__ import annotations

import os

import pytest
import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
    _sigmoid_bias_top16,
)


@gluon.jit
def _megamoe_topk_probe(router_logits, correction_bias, topk_ids, topk_weights):
    _sigmoid_bias_top16(
        gl.program_id(0),
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
    )


def _assert_matches_current_triton(
    logits: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk,
    )

    reference_weights, reference_ids = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    ids = torch.empty((16,), dtype=torch.int32, device=logits.device)
    weights = torch.empty((16,), dtype=torch.float32, device=logits.device)
    _megamoe_topk_probe[(240,)](
        logits,
        bias,
        ids,
        weights,
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
    )
    torch.cuda.synchronize()

    assert torch.equal(ids, reference_ids[0])
    assert torch.equal(
        weights.view(torch.int32), reference_weights[0].view(torch.int32)
    )
    return ids


@pytest.mark.skipif(
    os.getenv("TOKENSPEED_TEST_MEGAMOE_NUMERICS") != "1",
    reason="requires a gfx950 and the pinned ROCm test environment",
)
@pytest.mark.parametrize("case", ["random_tie", "zero_denominator", "adversarial"])
def test_amd_megamoe_topk_is_bitwise_current_triton(
    case: str,
) -> None:
    if case == "zero_denominator":
        logits = torch.full((1, 896), -float("inf"), device="cuda:0")
        bias = torch.zeros((896,), dtype=torch.float32, device="cuda:0")
    elif case == "adversarial":
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(490771)
        for label in range(6):
            logits = torch.randn(
                (1, 896),
                dtype=torch.float32,
                device="cuda:0",
                generator=generator,
            ) * ((label % 7) + 0.125)
            bias = (
                torch.randn(
                    (896,),
                    dtype=torch.float32,
                    device="cuda:0",
                    generator=generator,
                )
                * 0.5
            )
            if label % 4 == 0:
                logits[0, :32] = float(label % 3 - 1)
            if label % 8 == 0:
                bias[:32] = 0
    else:
        torch.manual_seed(20260814)
        logits = torch.randn((1, 896), dtype=torch.float32, device="cuda:0")
        bias = torch.randn((896,), dtype=torch.float32, device="cuda:0") * 0.05
        # Exercise the packed lower-expert-id tie break.
        logits[0, 0:2] = 8.0
        bias[0:2] = 0.0

    _assert_matches_current_triton(logits, bias)


@pytest.mark.skipif(
    os.getenv("TOKENSPEED_TEST_MEGAMOE_NUMERICS") != "1",
    reason="requires a gfx950 and the pinned ROCm test environment",
)
def test_amd_megamoe_topk_is_bitwise_across_seeds_and_scales() -> None:
    device = "cuda:0"
    for seed in range(64):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        scale = (0.125, 1.0, 5.125, 20.0)[seed % 4]
        logits = (
            torch.randn(
                (1, 896),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * scale
        )
        bias = (
            torch.randn(
                (896,),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.5
        )
        _assert_matches_current_triton(logits, bias)


@pytest.mark.skipif(
    os.getenv("TOKENSPEED_TEST_MEGAMOE_NUMERICS") != "1",
    reason="requires a gfx950 and the pinned ROCm test environment",
)
def test_amd_megamoe_topk_is_bitwise_at_top16_top17_cutoff() -> None:
    device = "cuda:0"
    # sigmoid(0) + 1.25 is exactly 1.75. Both adjacent FP32 bias values
    # therefore remain distinct after the addition and straddle a true tie.
    base = torch.tensor(1.25, dtype=torch.float32)
    below = torch.nextafter(base, torch.tensor(-float("inf"))).item()
    above = torch.nextafter(base, torch.tensor(float("inf"))).item()

    for cutoff, expected_last_id in (
        (below, 15),
        (base.item(), 15),
        (above, 16),
    ):
        logits = torch.full((1, 896), -20.0, dtype=torch.float32, device=device)
        bias = torch.zeros((896,), dtype=torch.float32, device=device)
        logits[0, :15] = torch.linspace(8.0, 4.0, 15, device=device)
        bias[:15] = 2.0
        logits[0, 15:17] = 0.0
        bias[15] = base.item()
        bias[16] = cutoff
        ids = _assert_matches_current_triton(logits, bias)
        assert ids[-1].item() == expected_last_id
