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

"""Abstract base class for communication backends."""

from abc import ABC, abstractmethod

import torch

from tokenspeed.runtime.distributed.mapping import Group


class CommBackend(ABC):
    """Interface that all communication backends must implement.

    All group parameters are tuples of global ranks, e.g. (0, 1, 2, 3).
    Process groups are looked up from pg_manager, not created here.
    """

    # ---- Collective ops ----

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Reduce one tensor or a collection of independent tensors."""
        if isinstance(tensor, torch.Tensor):
            raise NotImplementedError
        tensors = tensor
        if len(tensors) == 0:
            raise ValueError("all-reduce requires at least one tensor")
        return tuple(self.all_reduce(value, group, op=op) for value in tensors)

    def prepare_all_reduce_lane(self, group: Group, hidden_dim: int) -> bool:
        """Prepare an implementation-specific one-shot lane when supported."""

        return False

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire writable outputs for a later all-reduce.

        Args:
            shapes: Shapes of the outputs the producer will write.
            like: Tensor providing dtype and device for ordinary allocations.
            group: Global ranks participating in every reduction.
            op: Reduction operation.

        Returns:
            Writable outputs accepted by ``all_reduce`` after production.
        """
        if not shapes:
            raise ValueError("all-reduce requires at least one output")
        return tuple(like.new_empty(shape) for shape in shapes)

    def consensus_producer_direct_lane_status(
        self,
        group: Group,
        *,
        stage: str,
        local_status: int = 0,
        local_reason: str = "",
    ) -> None:
        """Reach rank-uniform agreement before a producer-direct stage.

        Args:
            group: Ordered global ranks that must all enter the stage.
            stage: Stable name for the stage being admitted.
            local_status: Zero for local success; nonzero for local failure.
            local_reason: Bounded diagnostic for a nonzero status.

        Raises:
            RuntimeError: If any rank rejects the stage.
        """

        if local_status:
            raise RuntimeError(
                f"producer-direct {stage} rejected locally with status "
                f"{local_status}: {local_reason}"
            )

    def acquire_producer_direct_lane(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        *,
        local_status: int = 0,
        local_reason: str = "",
    ) -> object | None:
        """Collectively acquire opaque state for a fused producer kernel.

        Args:
            shapes: Consecutive producer-output shapes required by the kernel.
            like: Tensor providing the required dtype and device.
            group: Ordered global ranks participating in every invocation.
            local_status: Zero after local admission succeeds; nonzero otherwise.
            local_reason: Bounded diagnostic for a nonzero local status.

        Returns:
            An opaque backend-owned lane, or ``None`` when unsupported.

        Raises:
            RuntimeError: If rank-uniform admission fails.
        """

        self.consensus_producer_direct_lane_status(
            group,
            stage="admission",
            local_status=local_status,
            local_reason=local_reason,
        )
        return None

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor: ...

    @abstractmethod
    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None: ...

    @abstractmethod
    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor: ...

    @abstractmethod
    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        """Even-split all_to_all. output and input must have same numel
        divisible by len(group).
        """
        ...

    # ---- Token-aware ops (uneven token distribution) ----

    @abstractmethod
    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor: ...

    @abstractmethod
    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor: ...
