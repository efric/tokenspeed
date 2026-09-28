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

"""Residual RMSNorm selection; vendor option/fallback contracts stay intact."""

import torch
from tokenspeed_kernel.ops.communication._selection import select_collective
from tokenspeed_kernel.platform import current_platform


def run_allreduce_residual_rmsnorm(**kwargs):
    """Preserve vendor tuple results, including unsupported-layout sentinels."""
    platform = current_platform()
    if platform.is_amd or platform.is_nvidia:
        if platform.is_amd:
            from tokenspeed_kernel.ops.communication import iris  # noqa: F401
        else:
            from tokenspeed_kernel.ops.communication import trtllm  # noqa: F401
        name = (
            "iris_all_reduce_rmsnorm_adapter"
            if platform.is_amd
            else "trtllm_all_reduce_rmsnorm_adapter"
        )
        return select_collective(
            "all_reduce_rmsnorm",
            kwargs["input_tensor"].dtype,
            name,
        )(**kwargs)
    from tokenspeed_kernel.ops.layernorm import rmsnorm

    tensor = kwargs["input_tensor"]
    torch.distributed.all_reduce(tensor, group=kwargs["group"])
    output, residual = rmsnorm(
        tensor,
        kwargs["weight"],
        kwargs["eps"],
        residual=kwargs["residual"],
    )
    return output, residual, None
