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


"""Iris host adapters for operation-owned reduction epilogues."""

import logging
from typing import Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.communication._iris.context import (
    _get_available_gpu_memory,
    _get_or_create_iris_context,
)
from tokenspeed_kernel.platform import current_platform

logger = logging.getLogger(__name__)
_platform = current_platform()


def all_reduce_residual_attnres(
    self,
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a partial, update the residual and finish its AttnRes mix."""
    if op is None:
        op = dist.ReduceOp.SUM
    assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
    from tokenspeed_kernel_amd.ops.gfx950.communication.attnres import (
        iris_push_one_shot_allreduce_residual_attnres_gluon_kernel,
    )

    kernel_config = self._kernel_config.attnres
    assert _platform.is_cdna4 and self.world_size == kernel_config.world_size
    num_tokens = partial.shape[0]
    assert 0 < num_tokens <= self.attnres_max_rows
    expected_shape = (num_tokens, kernel_config.hidden_size)
    assert partial.shape == residual.shape == expected_shape
    assert partial.dtype == residual.dtype == self.dtype == torch.bfloat16
    assert partial.device == residual.device == self.device
    assert partial.is_contiguous() and residual.is_contiguous()
    assert 0 < partial.numel() <= self.attnres_max_numel, (
        f"tensor numel ({partial.numel()}) exceeds iris buffer capacity "
        f"({self.attnres_max_numel})"
    )
    assert self.attnres.inbox is not None
    assert self.attnres.epochs is not None
    assert self.attnres.flags is not None
    assert self.attnres.peer_inboxes is not None
    assert score_weight.shape == output_weight.shape == (kernel_config.hidden_size,)
    assert score_weight.dtype == output_weight.dtype == torch.bfloat16
    assert score_weight.device == output_weight.device == self.device
    assert score_weight.is_contiguous() and output_weight.is_contiguous()
    m, s_, acc = scratch
    assert m.shape == s_.shape == (num_tokens,)
    assert acc.shape == expected_shape
    assert m.dtype == s_.dtype == acc.dtype == torch.float32
    assert m.device == s_.device == acc.device == self.device
    assert m.is_contiguous() and s_.is_contiguous() and acc.is_contiguous()
    assert eps > 0.0

    hidden = torch.empty_like(partial)
    residual_out = torch.empty_like(residual)
    iris_push_one_shot_allreduce_residual_attnres_gluon_kernel[(num_tokens,)](
        partial,
        residual,
        self.attnres.inbox,
        score_weight,
        output_weight,
        m,
        s_,
        acc,
        hidden,
        residual_out,
        self.attnres.epochs,
        self.attnres.flags,
        *self.attnres.peer_inboxes,
        *self._heap_base_addresses,
        RANK=self._iris_rank,
        WORLD_SIZE=self.world_size,
        HIDDEN=kernel_config.hidden_size,
        BLOCK=triton.next_power_of_2(kernel_config.hidden_size),
        MAX_ELEMENTS=self.attnres_max_numel,
        READY_SLOT_STRIDE=self.attnres_max_rows * self.world_size,
        EPS=eps,
        ELEMENTS_PER_THREAD=kernel_config.elements_per_thread,
        NUM_WARPS=kernel_config.num_subgroups,
        SUBGROUP_SIZE=self._kernel_config.subgroup_size,
        num_warps=kernel_config.num_subgroups,
    )
    return hidden, residual_out


class IrisAllReduceResidualRMSNorm(object):

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        heap_size: int | None = None,
        device: torch.device = None,
        persistent: bool = False,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisAllReduceResidualRMSNorm; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisAllReduceResidualRMSNorm currently targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.world_size = group.size()
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        if heap_size is None:
            buf_bytes = max_token_num * hidden_dim * dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))
        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        self._input_buf = self._ctx.zeros((max_token_num, hidden_dim), dtype=dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris AR+RMSNorm symmetric-heap buffer allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = dist.get_rank()

        self.persistent = persistent
        self._num_programs = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if persistent
            else 0
        )

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from tokenspeed_kernel.ops.communication._iris.triton import (
            iris_allreduce_residual_rmsnorm_kernel,
            iris_allreduce_residual_rmsnorm_kernel_persistent,
        )

        assert input_tensor.dtype == self.dtype, (
            f"Iris AR+RMSNorm dtype mismatch: input={input_tensor.dtype}, "
            f"backend={self.dtype}"
        )
        assert input_tensor.dim() == 2, (
            f"input must be 2-D (num_tokens, hidden_dim), got "
            f"shape={input_tensor.shape}"
        )
        assert (
            input_tensor.shape == residual.shape
        ), f"residual shape {residual.shape} != input shape {input_tensor.shape}"
        assert input_tensor.shape[1] == self.hidden_dim, (
            f"hidden_dim mismatch: input={input_tensor.shape[1]} vs "
            f"backend={self.hidden_dim}"
        )
        num_tokens = input_tensor.shape[0]
        assert num_tokens <= self.max_token_num, (
            f"num_tokens ({num_tokens}) exceeds max_token_num "
            f"({self.max_token_num})"
        )
        assert weight.shape == (
            self.hidden_dim,
        ), f"weight shape {weight.shape} != ({self.hidden_dim},)"
        assert input_tensor.is_contiguous() and residual.is_contiguous()

        in_view = self._input_buf[:num_tokens, :]
        in_view.copy_(input_tensor)

        if norm_out is None:
            norm_out = torch.empty_like(input_tensor)
        if residual_out is None:
            residual_out = torch.empty_like(residual)

        self._ctx.device_barrier()

        heap_bases = self._ctx.get_heap_bases()
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        if self.persistent:
            kernel = iris_allreduce_residual_rmsnorm_kernel_persistent
            grid = (min(num_tokens, self._num_programs),)
        else:
            kernel = iris_allreduce_residual_rmsnorm_kernel
            grid = (num_tokens,)
        kernel[grid](
            in_view,
            residual,
            weight,
            norm_out,
            residual_out,
            num_tokens,
            heap_bases,
            iris_rank=self._iris_rank,
            world_size=self.world_size,
            rank_start=self._rank_start,
            rank_stride=self._rank_stride,
            HIDDEN_SIZE=self.hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            EPS=eps,
            num_warps=8,
        )
        # Ensure all peer loads finish before the next call reuses _input_buf.
        self._ctx.device_barrier()
        return norm_out, residual_out


def allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: bool | None = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    residual_reduce_scattered: bool = False,
    has_partial_norm_out: bool = False,
    max_sm_to_use: int | None = None,
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
    platform = current_platform()
    if platform.is_amd:
        if (
            block_quant_fp8
            or residual_reduce_scattered
            or has_partial_norm_out
            or input_tensor.dim() != 2
            or residual is None
        ):
            return None, None, None, None

        token_num, hidden_dim = input_tensor.shape

        import tokenspeed_kernel.ops.communication.iris as _iris_mod
        from tokenspeed_kernel.ops.communication._iris.context import iris_available

        if (
            iris_available()
            and input_tensor.is_cuda
            and residual.is_cuda
            and weight.is_cuda
            and input_tensor.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and input_tensor.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and input_tensor.shape == residual.shape
            and weight.shape == (hidden_dim,)
            and group.size() > 1
            and token_num <= max_token_num
        ):
            key = (id(group), max_token_num, hidden_dim, input_tensor.dtype)
            iris_state = _iris_mod.IRIS_AR_RMSNORM_STATES.get(key)
            if iris_state is None:
                iris_state = _iris_mod.create_iris_ar_rmsnorm_state(
                    group=group,
                    rank_in_group=rank,
                    max_token_num=max_token_num,
                    hidden_dim=hidden_dim,
                    dtype=input_tensor.dtype,
                )
                _iris_mod.IRIS_AR_RMSNORM_STATES[key] = iris_state
            norm_out, residual_out = _iris_mod.iris_allreduce_residual_rmsnorm(
                iris_state,
                input_tensor=input_tensor,
                residual=residual,
                weight=weight,
                eps=eps,
            )
            return norm_out, residual_out, None, None

        from tokenspeed_kernel.ops.communication.triton import (
            allreduce_residual_rmsnorm_can_run,
            allreduce_residual_rmsnorm_get_state,
            amd_allreduce_residual_rmsnorm_kernel,
        )

        state = allreduce_residual_rmsnorm_get_state(
            group=group,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        if not allreduce_residual_rmsnorm_can_run(
            state, input_tensor, residual, weight
        ):
            return None, None, None, None

        state.comm_buff[:token_num, :].copy_(input_tensor)
        norm_out = torch.empty_like(input_tensor)
        residual_out = torch.empty_like(residual)
        amd_allreduce_residual_rmsnorm_kernel[(token_num,)](
            state.symm_mem_hdl.buffer_ptrs_dev,
            state.symm_mem_hdl.signal_pad_ptrs_dev,
            residual,
            weight,
            norm_out,
            residual_out,
            HIDDEN_SIZE=hidden_dim,
            EPS=eps,
            RANK=state.symm_mem_hdl.rank,
            WORLD_SIZE=state.symm_mem_hdl.world_size,
            BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
            num_warps=8,
        )
        return norm_out, residual_out, None, None
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"
        return None, None, None, None
