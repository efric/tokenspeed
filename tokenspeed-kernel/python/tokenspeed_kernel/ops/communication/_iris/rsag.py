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


"""Iris CCL reduce-scatter/all-gather host adapter."""

import logging
from typing import List, Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication._iris.context import (
    _get_available_gpu_memory,
    _get_or_create_iris_context,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.iris import IrisConfig as _IrisConfig
from tokenspeed_kernel.thirdparty.iris import all_gather as _iris_all_gather
from tokenspeed_kernel.thirdparty.iris import reduce_scatter as _iris_reduce_scatter

logger = logging.getLogger(__name__)
_platform = current_platform()


class IrisRSAG(object):

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_tokens: int,
        hidden_size: int,
        device: torch.device = None,
        heap_size: int | None = None,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisRSAG; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisRSAG currently targets AMD ROCm; " f"got non-AMD platform: {_platform}"
        )
        assert (
            group == dist.group.WORLD or group.size() == dist.get_world_size()
        ), "iris.ccl all_gather/reduce_scatter do not accept a sub-group."

        self.group = group
        self.rank_in_group = rank_in_group
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = torch.bfloat16
        self.world_size = group.size()

        # Heap holds in/out flat buffers plus iris bookkeeping; over-provision
        # similarly to ``IrisAllReduce`` to leave room for ring/spinlock flags.
        if heap_size is None:
            buf_bytes = max_tokens * hidden_size * self.dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        self._in_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        self._out_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris RSAG symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        assert self._ctx.get_num_ranks() == dist.get_world_size(), (
            f"Iris world size {self._ctx.get_num_ranks()} "
            f"!= torch world size {dist.get_world_size()}"
        )
        assert self.rank_in_group == self._ctx.get_rank(), (
            f"rank mismatch: rank_in_group={self.rank_in_group}, "
            f"iris rank={self._ctx.get_rank()}"
        )

    # -- token-distribution helpers (mirror sibling classes) ----------------

    def get_token_dist(self, total_tokens_in_group: int) -> list:
        token_list_in_group = []
        for rank in range(self.world_size):
            num_tokens_per_rank = total_tokens_in_group // self.world_size + (
                1 if (rank < total_tokens_in_group % self.world_size) else 0
            )
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def get_context(self, token_list_in_group: list) -> Tuple[int, int, int]:
        total_num_tokens = sum(token_list_in_group)
        assert (
            total_num_tokens <= self.max_tokens
        ), f"The inner comm buffer is too small: {total_num_tokens=} is not <= {self.max_tokens=}"
        local_num_tokens = token_list_in_group[self.rank_in_group]
        local_token_offset = sum(token_list_in_group[: self.rank_in_group])
        return total_num_tokens, local_num_tokens, local_token_offset

    # -- internal helpers ---------------------------------------------------

    def _assert_uniform(self, token_list_in_group: List[int]) -> int:
        first = token_list_in_group[0]
        assert all(t == first for t in token_list_in_group), (
            "IrisRSAG requires uniform tokens per rank; got "
            f"token_list_in_group={token_list_in_group}"
        )
        return first

    @staticmethod
    def _pick_block_n(hidden_size: int) -> int:
        # Pick the largest power-of-two block that divides hidden_size, capped
        # at 256. This keeps the iris kernel on its no-mask fast path and
        # still produces enough tiles (world_size * hidden/block_n) to fill
        # ``comm_sms`` SMs on supported AMD chips.
        for cand in (256, 128, 64, 32, 16):
            if hidden_size % cand == 0:
                return cand
        return hidden_size

    def _make_config(self, local_num_tokens: int, hidden_size: int):
        # ``swizzle_size=1`` keeps tile_id ordering row-major in M, which is
        # required so that block-distribution (DISTRIBUTION=1) hands rank r
        # exactly the K tiles spanning rows [r*local, (r+1)*local) in the
        # reduce-scatter kernel. ``all_gather`` is rank-agnostic on tile order
        # so the same config is fine.
        return _IrisConfig(
            block_size_m=local_num_tokens,
            block_size_n=self._pick_block_n(hidden_size),
            swizzle_size=1,
            all_reduce_distribution=1,
        )

    # -- public collective ops ---------------------------------------------

    def reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, local_token_offset = self.get_context(token_list_in_group)
        assert (hidden_states.shape[0] == total_num_tokens) and (
            hidden_states.shape[-1] == self.hidden_size
        ), (
            f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} "
            f"or {hidden_states.shape[-1]=} != {self.hidden_size=} "
            f"{hidden_states.shape=}"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:total_num_tokens, : self.hidden_size]
        out_view = self._out_buff[:total_num_tokens, : self.hidden_size]
        in_view.copy_(hidden_states)

        # Ensure every rank's shared input copy is visible before peer loads begin.
        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, self.hidden_size)
        _iris_reduce_scatter(out_view, in_view, self._ctx, config=config)

        output = out_view[local_token_offset : local_token_offset + local_num_tokens, :]
        return output.clone() if safe else output

    def all_gather(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, _ = self.get_context(token_list_in_group)
        hidden_size = hidden_states.shape[-1]
        assert (hidden_states.shape[0] == local_num_tokens) and (
            hidden_size <= self.hidden_size
        ), (
            f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} "
            "Mismatched shape"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:local_num_tokens, :hidden_size]
        out_view = self._out_buff[:total_num_tokens, :hidden_size]
        in_view.copy_(hidden_states)

        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, hidden_size)
        _iris_all_gather(out_view, in_view, self._ctx, config=config)

        return out_view.clone() if safe else out_view
