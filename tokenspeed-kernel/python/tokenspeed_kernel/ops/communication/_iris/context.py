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


"""Process-wide Iris heap lifetime and peer address translation."""

from importlib.util import find_spec

import torch

_iris_ctx_singleton = None

IRIS_AR_STATES: dict = {}
IRIS_AR_RMSNORM_STATES: dict = {}


def iris_available() -> bool:
    """Probe the optional Iris package without importing its device code."""
    return find_spec("iris") is not None


def amd_collectives_available() -> bool:
    """Probe the optional dependencies of the Iris gfx950 solution."""
    return iris_available() and find_spec("tokenspeed_kernel_amd") is not None


def _peer_addresses(
    tensor: torch.Tensor,
    heap_bases: tuple[int, ...],
    rank: int,
) -> tuple[int, ...]:
    heap_offset = tensor.data_ptr() - heap_bases[rank]
    return tuple(heap_base + heap_offset for heap_base in heap_bases)


def _get_available_gpu_memory(gpu_id: int, empty_cache: bool = True) -> float:
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_id):
            if empty_cache:
                torch.cuda.empty_cache()
            free_gpu_memory, _ = torch.cuda.mem_get_info()
            return free_gpu_memory / (1 << 30)
    return 0.0


def _get_or_create_iris_context(heap_size: int):
    from tokenspeed_kernel.thirdparty.iris import iris

    global _iris_ctx_singleton
    if _iris_ctx_singleton is None:
        _iris_ctx_singleton = iris.iris(heap_size=heap_size)
    elif heap_size > _iris_ctx_singleton.heap_size:
        raise RuntimeError(
            f"Iris has a {_iris_ctx_singleton.heap_size}-byte symmetric heap, "
            f"but this state requires {heap_size} bytes; prepare the largest "
            "state first"
        )
    return _iris_ctx_singleton
