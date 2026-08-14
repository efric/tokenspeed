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

import torch

from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe import (
    kimi_k3_megamoe_workspace_spec,
)


def test_amd_workspace_spec_is_flat_and_unique() -> None:
    specs = kimi_k3_megamoe_workspace_spec()
    assert len(specs) == 31
    assert len({spec.name for spec in specs}) == len(specs)
    assert specs[0].name == "router_logits"
    assert specs[0].shape == (896,)
    assert specs[0].dtype == torch.float32
    by_name = {spec.name: spec for spec in specs}
    assert by_name["xcc_ticket"].shape == (8,)
    assert by_name["xcc_ticket"].dtype == torch.int64
    assert by_name["xcd_arrival"].shape == (1,)
    assert by_name["xcd_arrival"].dtype == torch.int64
    assert specs[-1].name == "fail_diagnostics"
    assert specs[-1].shape == (8,)
    assert specs[-1].dtype == torch.int64
