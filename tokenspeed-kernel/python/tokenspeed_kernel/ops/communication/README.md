# Iris LL128 all-reduce

`create_iris_ll128_state` and `iris_ll128_all_reduce` in `iris.py` provide an
explicit CDNA5 (`gfx1250`) BF16 all-reduce using cooperative atomic load/store
packets. The state uses the existing Iris symmetric heap and group-relative
peer address translation. Inputs and results are ordinary local tensors.

```python
from tokenspeed_kernel.ops.communication.iris import (
    create_iris_ll128_state,
    iris_ll128_all_reduce,
)

state = create_iris_ll128_state(
    group=group,
    rank_in_group=rank,
    max_numel=6 * 10752,
    heap_size=None,
    device=device,
)
result = iris_ll128_all_reduce(state, local_bf16_input)
```

Create the state before graph capture, in the same symmetric-allocation order
on every Iris rank. Each call returns a fresh result with the input shape and
does not modify the input. Two to eight ranks are supported. Calls must have
the same element count and order on every participating rank. Calls sharing a
state must be serialized on each rank; independent concurrent streams need
separate states, coordinated collective order across ranks, and a resource
budget that keeps every polling workgroup resident. The selected Iris mapping
must provide the GPU fabric access required by the cooperative atomic contract.

## Packets and reduction

Each naturally aligned 128-byte packet carries 60 BF16 values (120 bytes) and
an eight-byte epoch. `cooperative_atomic_store` publishes both together with
relaxed system scope. The receiver polls with the matching cooperative load,
checks the complete epoch, and consumes the payload from that checked
snapshot. Packets may arrive in any order. The polling condition covers every
packet from every peer.

The kernel represents four packets as a logical `[4, 64]` tensor of 16-bit
words with `BlockedLayout([1, 8], [4, 8], [1, 1], [1, 0])`. Eight contiguous
lanes cooperatively access each packet, with eight words (16 bytes) per lane.
One wave32 instruction therefore moves four independent 128-byte packets.
Words 0–59 contain payload bits and words 60–63 contain the epoch, least
significant word first. The Gluon API owns the cooperative access; this packet
format belongs to the collective.

Each logical tile always owns the same 512 application elements. Three full
wave batches reserve twelve packets per tile. Nine packets carry the tile's data;
the remaining positions are padding in the initial unmasked implementation.
Application input and output tails use ordinary element masks. All wire
loads and stores access complete allocated packets, including padding.
Publication to every peer precedes polling. The receive loop retains all
accepted data, then each rank folds contributions in ascending **group-rank**
order in FP32 and rounds once to BF16.

The state agrees the minimum compute-unit count across its ranks at creation
and caps the grid at that many one-wave programs. Each program visits tiles
`program_id`, `program_id + num_programs`, and so on in a block-uniform loop.
This keeps the polling workgroups resident independently of message capacity.
The launch needs that residency; do not overlap it with kernels that can occupy
the device while waiting for this collective. Tile IDs, rather than program
IDs, index all mailboxes and epochs when the grid size changes.

No payload bit pattern is reserved. In particular, transport preserves `-0`
and NaN payload bits without normalization. Floating-point addition still
applies its usual numerical rules. Mailboxes are initialized once; consumed
packets are not cleared.

## Epochs and storage reuse

The mailbox shape is `[3, world_size, max_tiles, 12, 64]` in 16-bit words.
Each sender exclusively writes its rank slice in each receiver's mailbox.
Each tile has its own 64-bit device epoch, independent of the tile count in
any particular invocation. A tile skipped by a smaller message keeps its
epoch. All ranks see the same sequence of activations for that tile, so they
continue to agree when the tile next participates.

Epoch zero denotes initialized storage and is never published. The active
slot is `(epoch - 1) % 3`. After `UINT64_MAX`, the next epoch is one;
`UINT64_MAX` is a multiple of three, so slot rotation remains continuous at
rollover. The last three epochs are distinct even at this boundary.

The bidirectional exchange supplies the reuse credit. A rank can publish its
next epoch before another rank has consumed the current epoch. To complete
that next epoch, however, it must receive a next-epoch packet from every
peer. Each peer publishes that packet only after its preceding invocation
has finished consuming the current epoch. Thus a producer cannot progress
far enough to reuse a three-stage slot while a peer still needs it. The
argument applies separately to each stable tile, including when intervening
calls omit the tile. Ordered kernel execution on each rank is a prerequisite;
the mailbox does not synchronize concurrent calls sharing a state.

Relaxed cooperative accesses suffice for these self-contained packets.
Readiness does not publish unrelated memory. A protocol that uses an epoch
to publish another allocation would need the corresponding ordering/fences.

## Relation to RCCL and the paper

RCCL's `CollCommon_ll128.h` supplies the 120-byte payload / eight-byte epoch
format and cooperative transport inspiration. Gluon expresses ownership and
readiness using tensors and block-uniform control flow; it does not require
RCCL's per-thread arrays, register shuffles, or independent warp loops.

The algorithm uses the one-shot exchange and buffering ideas in
[sections III-A1 and IV-3 of the paper](https://arxiv.org/html/2607.16100v1).
Its section IV-4 algorithm additionally requires cache-line atomic **addition**
over NVLink. Cooperative load/store does not provide that operation. This
implementation reduces locally in registers and preserves TokenSpeed's
rank-ordered FP32 contract.

## Validation

`test/amd/ops/test_iris_lamport.py::test_ll128_cooperative_polling_codegen`
compiles the full kernel for `gfx1250` without launching it. It checks the
LLVM cooperative intrinsics, relaxed system scope, full packet loads on the
assembly retry loop, publication stores without mailbox clearing, wave32,
and absence of spills. The LLVM backend must include the vector-memory-type
fix for cooperative atomics from
[LLVM PR #229501](https://github.com/llvm/llvm-project/pull/229501).

`test_ll128_epochs_and_payloads_world8` requires eight CDNA5 GPUs. It checks
rank-ordered numerical results, payload bit preservation in the mailboxes,
partial packets and tiles, epoch rollover, changing sizes and grids without
JIT specialization, and graph replay under rank skew. Offline compilation does
not establish hardware fabric atomicity or distributed liveness; those checks
remain hardware tests.
