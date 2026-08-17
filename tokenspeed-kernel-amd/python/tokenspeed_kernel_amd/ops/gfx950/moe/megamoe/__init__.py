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

"""Experimental single-dispatch Kimi K3 MegaMoE kernel for gfx950."""

from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.admission import (
    admit_kimi_k3_megamoe_compiled_kernel,
    preflight_kimi_k3_megamoe_runtime,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel import (
    KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES,
    PreparedKimiK3MegaMoEKernel,
    compile_kimi_k3_megamoe_gfx950,
    launch_kimi_k3_megamoe_gfx950,
    launch_prepared_kimi_k3_megamoe_gfx950,
    prepare_kimi_k3_megamoe_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.workspace import (
    MegaMoETensorSpec,
    allocate_kimi_k3_megamoe_workspace,
    kimi_k3_megamoe_workspace_spec,
)

# Fail closed until projection, routing, both expert GEMVs, Iris, final output,
# and their numerical/liveness tests all pass on the production code object.
KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE = True

__all__ = [
    "KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE",
    "KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES",
    "MegaMoETensorSpec",
    "PreparedKimiK3MegaMoEKernel",
    "admit_kimi_k3_megamoe_compiled_kernel",
    "allocate_kimi_k3_megamoe_workspace",
    "compile_kimi_k3_megamoe_gfx950",
    "kimi_k3_megamoe_workspace_spec",
    "launch_kimi_k3_megamoe_gfx950",
    "launch_prepared_kimi_k3_megamoe_gfx950",
    "preflight_kimi_k3_megamoe_runtime",
    "prepare_kimi_k3_megamoe_gfx950",
]
