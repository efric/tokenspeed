# Communication operations

Runtime code imports `tokenspeed_kernel.ops.communication`. It supplies process
groups, tensors and semantic capacity requirements. The runtime backend owns the
ordinary collective fallback; it does not own Iris protocols, mailbox geometry,
or epilogue scratch.

## Layout and selection

| Owner | Responsibility |
| --- | --- |
| Public communication API | Preparation, ordinary reduction and bound residual operations |
| `_contracts.py` | Vendor-neutral capacity demands and prepared-implementation selection guard |
| `ops/residual/attnres.py` | AttnRes partial/weight contract and standalone composition |
| `iris.py`, `cute.py`, `trtllm.py` | Registered solution adapters |
| `_iris/adapter.py` | Iris handle, preparation, eligibility and operation adapter |
| `_iris/policy.py` | Supported domains, capacities, measured thresholds and launch geometry |
| `_iris/context.py` | Iris dependency probes, process heap lifetime and peer mapping |
| `_iris/workspace.py` | Separate staged, two-stage, producer, Lamport and AttnRes resource records |
| `_iris/all_reduce.py`, `epilogues.py`, `rsag.py` | Host launch adapters |
| `_iris/triton.py` | Portable RMA device kernels |
| `tokenspeed_kernel_amd/ops/gfx950/communication/` | gfx950 Gluon device implementations |
| `thirdparty/iris.py` | External Iris imports and Triton import compatibility |

The legacy `TritonCommState` stays with `communication/triton.py` because the
Triton RS/AG path uses it on both vendors. Iris's public all-reduce handle is
smaller and remains inside `_iris/adapter.py`; the legacy Triton exports accept
the same state fields. No generic communication module owns Iris package probes
or Iris physical capacities.

The AMD package takes device pointers and launch constants. It imports neither
Iris nor `tokenspeed_kernel`. Public imports do not load Iris or AMD collective
kernels. The legacy `communication.triton` exports remain compatibility aliases;
new callers use the public operation API. The existing legacy residual RMSNorm
and CCL RSAG support restrictions remain unchanged.

Operation policy preserves the measured winners. The existing registry resolves
the callable, and the collective adapter checks that an override cannot change
the prepared protocol or operand contract. All ranks must use the same controls,
operation metadata and package configuration. Selection performs no timing runs,
host vote, allocation, or device-value read in the forward path.

## Preparation and lifetime

An `AllReducePreparation` contains dtype and a tuple of
`AllReduceRequirement`, `PackedAllReduceRequirement`, `MoETailRequirement`,
or `AttnResRequirement`.
The packed requirement describes output widths and TP/EP group axes. It never
requests Lamport. Requirements must match across participants, and preparation
must complete before cache sizing and graph capture.

`MoETailRequirement` reserves a separate borrowed result and gather flags only
for the measured TP8 BF16 token-sharded MoE domain. It extends the prepared
producer capacity but does not widen ordinary all-reduce admission. Its gfx950
kernels are in `communication/moe_prefill.py` in the AMD package; the model
calls the vendor-neutral `ops/moe/__init__.py` entry point.

An opaque handle owns the existing cached solution state. Compatible calls reuse
it; larger demands cannot grow an existing heap. Physical backing capacity does
not widen ordinary all-reduce admission: that remains 512 KiB, independently of
producer-direct and AttnRes capacities. Capacity policy retains the existing
shared-group and baseline windows.

Workspace records retain the original allocation order. Staged one-shot has
rotating input slots and per-geometry flags. Staged two-stage retains
`EXIT_BARRIER=True`; producer-direct two-stage retains `False`. Lamport retains
its three generations, sentinel packs and epochs. The records share the existing
process heap; they do not introduce extra heaps or per-layer allocation.

Ordinary all-reduce mutates its input. The low-level `safe=True` option then
clones the result; `safe=False` returns the input. Producer acquisition returns
borrowed symmetric input views. Producer reduction returns views of a separate,
reusable local result buffer. Neither input nor result may be reused until its
existing stream/remote-reader lifetime ends. Preparation and registration keep
strong ownership throughout graph replay.

## Output storage contract

Ordinary tensor calls keep the existing shape-based one-shot/two-stage choice.
The two-stage kernel needs an 8-byte-aligned output pointer, so an unaligned
contiguous view uses a temporary output and copies the reduction back into the
input. This rank-local output repair does not change the collective protocol;
all ranks use the same shape-based choice. Producer-direct results use their
separate borrowed buffer and never alias the symmetric producer input.

## Residual operations and stream ordering

`prepare_residual_all_reduce` prepares persistent residual storage and native
capability agreement. `bind_residual_all_reduce` returns a binding or `None`.
The binding records both `consumes_partials` and `prefer_split_partials`:
operand readiness and graph preference are separate decisions.

The model resolves the binding after its attention producer returns. If it reads
partials produced on an auxiliary stream, it executes after that stream joins.
Otherwise reduction can overlap partial production. The same binding executes;
the model does not select again after the fork. A missing binding uses ordinary
runtime all-reduce, followed by standalone AttnRes combine after the join.

`AttnResEpilogue` keeps separate score factors and their precomputed BF16
product. NVIDIA and Iris consume their original representations. Iris still
rounds the reduction before residual addition, the prefix before scoring and the
mixture before output normalization. Native residual-only reduction still
returns no mixture. The public residual RMSNorm adapter preserves vendor option
handling and unsupported-result tuples, including quantized/partial layouts.

## Validation

Run the common policy/contract suite, runtime communication and AttnRes suites,
and AMD Iris communication/Lamport suites. The distributed suite includes
allocation-backed output, arbitrary views with rank-dependent offsets, graph
replay, alternating geometries, subgroup mapping and borrowed output lifetimes.
Eight-rank fused AttnRes and Lamport execution need eight available GPUs;
compilation checks alone do not establish distributed synchronization correctness.
