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


"""Public preparation and execution adapters for registered all-reduces."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication._contracts import AllReducePreparation
from tokenspeed_kernel.ops.communication._dependencies import amd_collectives_available
from tokenspeed_kernel.ops.communication._iris import adapter
from tokenspeed_kernel.ops.communication._state import TritonCommState
from tokenspeed_kernel.platform import current_platform

DEFAULT_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024
ORDINARY_ALL_REDUCE_MAX_BYTES = 512 * 1024


def producer_all_reduce_available() -> bool:
    """Whether the platform has a producer-output collective implementation."""
    return current_platform().is_cdna4 and amd_collectives_available()


def create_all_reduce_handle(
    group: dist.ProcessGroup,
    rank_in_group: int,
    device: torch.device,
    producer_direct_max_bytes: int,
) -> object:
    """Describe default collective storage without allocating a symmetric heap.

    Args:
        group: Device process group.
        rank_in_group: This process's group-local rank.
        device: Device for subsequent workspace preparation.
        producer_direct_max_bytes: Producer storage capacity in bytes.

    Returns:
        An opaque descriptor; ordinary admission remains separately capped.
    """
    return TritonCommState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_numel=min(producer_direct_max_bytes, ORDINARY_ALL_REDUCE_MAX_BYTES)
        // torch.bfloat16.itemsize,
        max_bytes=producer_direct_max_bytes,
        attnres_max_numel=0,
        enable_lamport=False,
        max_token_num=0,
        hidden_dim=0,
        comm_buff=None,
        symm_mem_hdl=None,
    )


def prepare_all_reduce_handle(
    group: dist.ProcessGroup,
    rank_in_group: int,
    device: torch.device,
    preparation: AllReducePreparation,
    previous: object | None,
    producer_direct_max_bytes: int,
) -> object | None:
    """Prepare group storage before capture, preserving existing compatible handles.

    Args:
        group: Device process group.
        rank_in_group: Local rank in group.
        device: Allocation device.
        preparation: Semantic operation demands, identical across group members.
        previous: Already prepared handle, or None on first preparation.
        producer_direct_max_bytes: Default backing capacity; does not widen ordinary admission.

    Returns:
        An opaque handle, or None when this implementation cannot prepare it.
    """
    if (
        group.size() <= 1
        or not producer_all_reduce_available()
        or preparation.dtype != torch.bfloat16
    ):
        return None
    from tokenspeed_kernel.ops.communication._iris.policy import resolve_capacities

    capacity = resolve_capacities(preparation, group.size(), producer_direct_max_bytes)
    requested = (
        capacity.staged_max_numel,
        capacity.producer_direct_max_numel * preparation.dtype.itemsize,
        capacity.attnres_max_numel,
        capacity.attnres_max_rows,
    )
    if not any(requested):
        return None
    if previous is not None:
        if previous.enable_lamport != capacity.enable_lamport:
            raise RuntimeError(
                "all-reduce buffers were initialized with a different Lamport policy"
            )
        available = (
            previous.max_numel,
            previous.max_bytes,
            previous.attnres_max_numel,
            previous.max_token_num,
        )
        if any(have < need for have, need in zip(available, requested)):
            raise RuntimeError(
                f"all-reduce buffers were initialized below requested capacities: {available}, {requested}"
            )
        adapter.initialize_all_reduce_state(previous, preparation.dtype)
        return previous
    state = TritonCommState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_numel=requested[0],
        max_bytes=requested[1],
        attnres_max_numel=requested[2],
        max_token_num=requested[3],
        enable_lamport=capacity.enable_lamport,
        hidden_dim=0,
        comm_buff=None,
        symm_mem_hdl=None,
    )
    adapter.initialize_all_reduce_state(state, preparation.dtype)
    return state


def all_reduce_capacity(handle: object) -> int:
    """Return prepared producer-output capacity in bytes."""
    return handle.max_bytes


# These adapters validate capability before touching the optional implementation.
all_reduce_can_run = adapter.all_reduce_can_run
all_reduce = adapter.all_reduce
symm_outputs_can_run = adapter.symm_outputs_can_run
acquire_symm_outputs = adapter.acquire_symm_outputs
all_reduce_symm_can_run = adapter.all_reduce_symm_can_run
all_reduce_symmetric = adapter.all_reduce_symmetric
