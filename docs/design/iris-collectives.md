# Iris collective boundaries

Iris is an optional collective implementation, not the communication API. The
runtime owns process groups, model dependencies, and the ordinary collective
fallback. `tokenspeed-kernel` owns operation contracts, selection, and Iris
workspace lifetime. `tokenspeed-kernel-amd` owns gfx950 device kernels. NVIDIA
solutions implement the same operations with their own resources; they do not
inherit Iris state or import the AMD package.

## Ownership and imports

| Layer | Owns | Must not own |
| --- | --- | --- |
| Model/runtime | Group choice, tensor production, stream ordering, fallback collective | Iris flags, heap offsets, protocol geometry, AttnRes scratch ABI on `CommBackend` |
| Communication operation | Semantic preparation, result/alias contract, implementation binding | Model identities, request scheduling, vendor device code |
| Iris host solution | Optional Iris context, group-to-peer mapping, workspace records, launch policy | NVIDIA collective state, model weights in generic transport |
| gfx950 solution | RMA, synchronization, packed stores, arithmetic | Process groups, heap allocation, runtime imports |

The public communication package and its operation-facing `iris.py` are
lightweight. They do not import core Iris or gfx950 device kernels at module
load. `thirdparty/iris.py` is the sole external Iris compatibility bridge;
`_iris/` imports it only after the implementation is selected or prepared.
The AMD device files import their own `_triton.py` bridge and have no reverse
dependency on `tokenspeed-kernel`.

Only semantic demands and the prepared-implementation guard live in shared
`_contracts.py`. Iris dependency probes, handle preparation and physical
capacities live under `_iris/`. The older `TritonCommState` belongs to the
cross-vendor legacy RS/AG implementation in `triton.py`; the Iris public
all-reduce handle does not carry its RS/AG buffers. Public handle creation
still delays Iris heap and gfx950 kernel imports until preparation or launch.

The ordinary path is:

```text
comm_ops.all_reduce
    -> AutoBackend.all_reduce
        -> KernelAllReduceBackend.all_reduce
            -> communication.all_reduce
                -> _iris.adapter.all_reduce
                    -> iris.iris_all_reduce
                        -> _iris.all_reduce.IrisAllReduce.all_reduce
```

The registry selects a named operation implementation, not a distinct runtime
backend for each Iris protocol. One-shot, two-stage, producer-direct, Lamport,
and fused AttnRes remain private policy choices within the solution. A bound
residual operation also declares whether it consumes AttnRes partials, allowing
the model's existing fork/join to wait only when those operands are required.

## Preparation and storage

Callers submit `AllReduceRequirement`, `PackedAllReduceRequirement`, and
`AttnResRequirement` as maximum logical shapes. The Iris policy derives
physical capacities and measured protocol eligibility. An enlarged backing
buffer does not widen ordinary all-reduce admission beyond 512 KiB. A prepared
context cannot grow its symmetric heap after allocation. Compatible groups
reuse state without changing allocation order or captured pointer lifetimes.

Staged one-shot, staged two-stage, producer-direct, Lamport, and AttnRes each
retain their own workspace and epoch domains. Producer acquisition yields
borrowed symmetric input views; the returned reduced views use separate
reusable local storage. Callers must consume them before that storage is reused.
The existing group-rank translation and context-rank mapping are distinct.

## Preserved execution contracts

- Ordinary all-reduce mutates its input. Its low-level `safe=True` option
  returns a clone after mutation; `safe=False` aliases the input.
- Shape and dtype, not a rank-local pointer, select staged one-shot versus
  two-stage. An unaligned two-stage destination uses a temporary output and
  local copy-back, so mixed pointer offsets cannot split peer protocols.
- Producer-direct reduction returns borrowed result views distinct from its
  symmetric producer inputs. Lamport remains limited to its measured TP8 BF16
  two-output shapes and row window.
- Fused AttnRes retains the existing FP32/BF16 cast points, operand order,
  supported shapes, and residual/result ownership. Unsupported cases retain
  ordinary reduction plus post-join composition.
- Kernel launch geometry, barriers, and device epoch handling must remain
  valid across graph replay and wraparound. Batch-varying geometry is passed
  as runtime arguments where required by the current JIT contract.

Preparation and implementation selection must agree across the group. No
rank-local fallback may switch collective protocols once peer work can launch.
New MoE or attention sharding should extend typed operation demands and private
Iris workspaces; it should not expose new physical buffers through `CommBackend`
or move gfx950 Gluon code into the common solution files.

The token-sharded MoE tail follows this rule: `MoETailRequirement` describes
maximum rows and output widths; `IrisAllReduceWorkspace.moe_tail` owns the
borrowed result and gather flags; `ops/moe/__init__.py` exposes the runtime's
vendor-neutral entry. The Iris MoE solution checks producer ownership and
permitted aliases before loading its gfx950 device implementation. Unsupported
inputs return `None` before launch, leaving the caller's existing reduction and
projection path intact.

Attention prefill uses the same prepared producer input and MoE result workspace,
without another symmetric allocation. `ops/communication/prefill.py` owns the
semantic row windows and optional fallback. `iris_prefill.py` checks the exact
group, producer ownership, tensor geometry, and aliasing before it imports
gfx950 kernels. The supported mixer reduce-scatters the projection, mixes the
local AttnRes history, and gathers only the normalized activation. It returns
an owned residual token shard and a borrowed full activation; the model passes
an explicit `prefix_is_sharded` bit to the MoE tail. A non-Iris MoE tier gathers
that shard before its ordinary projection path. Uneven or short batches keep
the prepared producer reduction but run the ordinary AttnRes mix. A block-write
layer clones a borrowed reduced residual before it can survive a later MoE
collective. The two solution files remain separate import boundaries: model
and general collective imports never load the optional Iris or gfx950 helpers.
