# Kimi K3 routed-MoE EP8 clean-sheet design

Status: host-only research artifact; no production implementation. Selectively
refreshed in worktree `agent/megamoe-cleansheets-main-refresh` onto
`origin/main` `e784229526ce11d272a3c4a0b3f64ab9a8973491` on 2026-08-15. The
original clean-sheet design was frozen at `8129173ab75ce473a04e495f1a55db14af5f0a80`;
the historical P240 evidence came from pre-refresh commit
`be7d5001927c9613314f300a420fb88c4bdd4628`.

## Decision

- Treat the historical ordinary P240 MegaMoE as a measured control, not an
  architecture. Its production package and flag were removed before the
  refreshed base, so it is not the current-checkout baseline.
  Do not retain its static program numbering, 31-tensor local workspace,
  51-tensor raw ABI, global phase barriers, all-worker W13/W2 ownership,
  route-to-XCD mapping, or one all-phase code object.
- Preserve only qualified semantics: exact Kimi K3 arithmetic boundaries,
  contiguous EP8 expert ownership, replicated tokens, deterministic route
  combine, direct symmetric-address translation, and release/acquire
  publication.
- Recommended launch graph is three bounded kernels:

  ```text
  K0 projection + top-k + local schedule
      -> K1 resident event-driven routed/shared core + joint Iris reduction
          -> K2 routed RMSNorm/up + shared/prefix add
  ```

- K1 is the megakernel: its W13 producers, W2 consumers, and communication
  workers have disjoint ownership and exchange medium-grained events. It has no
  whole-grid phase barrier. The starting discriminator is P256, eight subgroups
  per workgroup, with 128/96/32 logical W13/W2/communication workers; every
  number is an A/B knob, not an admission constant.
- B1 remains a vector-GEMV latency pipeline. B8/B16 use a hybrid: vector GEMV
  for singleton/small expert segments and exact-ragged MFMA tiles only when
  repeated token rows can amortize them. Never pad every active expert to 64
  rows merely to enter a grouped kernel.
- XCD affinity is off for B1 expert work. Per-XCD communication batches and
  B8/B16 same-weight M windows are experiments against non-XCD controls, not
  inherited truths. Only enable either when integrated timing and ATT agree.
- Admit a specialization only for a batch where it beats a fresh top-of-main
  baseline. Other batches continue through the native fused path.
- Reuse current TokenSpeed MoE primitives as first-class controls and
  implementation references: the linear EP8 checkpoint layout, route-direct
  W13/SiTU and output-owned W2 GEMV, scaled-upcast addressing, fused
  shared-down B1 composition, and gdot128/MFMA family. Redesign their schedule
  when useful; do not casually reimplement their qualified arithmetic.

## Verified contract

Repository-verified capture-time model path on the refreshed base:

```text
KimiLinearDecoderLayer.forward (python/tokenspeed/runtime/models/kimi_k3.py:2401)
    -> KimiLinearMoE.forward (.../kimi_k3.py:1549; called at :2552,
       or from the fused-AttnRes path at :2354)
        -> B1 native fused composition (:1572-1575)
            -> KimiLinearMoE._forward_fused_decode_pipeline (:1454)
                -> latent_moe_input_projections (:1471)
                -> TopK.forward (:1479)
                -> latent_moe_expert_shared_all_reduce (:1480)
                -> LatentMoELayer.finalize_output (:1496)
        -> B8/B16 native LatentMoELayer.forward (:1576-1581)
```

At graph replay the host replays the captured graph; the Python frames above
describe capture-time construction, not per-token host calls.

- Topology: one node, eight ranks; attention TP8; MoE TP1/EP8; every rank sees
  the same tokens; experts are contiguous 112-expert shards. Therefore this EP8
  path needs no token dispatch/all-to-all. Each rank computes its owned route
  partial and all eight ranks jointly reduce consecutive
  `[shared output, routed latent]` views
  (`python/tokenspeed/runtime/layers/moe/latent.py:187-232`; view order and
  producer-direct reduction are in
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/iris.py:441-513`).
  The EP-only joint-reduce selection is exact at
  `python/tokenspeed/runtime/layers/moe/latent.py:263-294`.
- Dimensions: `H=7168`, routed latent `L=3584`, `E=896`, top-k `K=16`, expert
  intermediate `I=3072`, shared intermediate `S=768`, 92 MoE layers
  (`python/tokenspeed/runtime/configs/kimi_k3_config.py:147-181`; layer
  selection is `python/tokenspeed/runtime/models/kimi_k3.py:2028-2043`).
- The EP8-specialized registry deliberately retains the contiguous linear
  checkpoint/storage layout
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:70-84,99-203`),
  exactly:

  - W13 bytes `[112,6144,1792]`, strides `[11010048,1792,1]`; scales
    `[112,6144,112]`, strides `[688128,112,1]`.
  - W2 bytes `[112,3584,1536]`, strides `[5505024,1536,1]`; scales
    `[112,3584,96]`, strides `[344064,96,1]`.
  - Dense BF16 weights are router `[896,7168]`, routed down `[3584,7168]`,
    shared gate/up `[1536,7168]`, shared down `[7168,768]`, routed norm
    `[3584]`, and routed up `[7168,3584]`
    (`python/tokenspeed/runtime/models/kimi_k3.py:1139-1246`). Raw MXFP4
    allocation formulas are at
    `python/tokenspeed/runtime/layers/moe/weights/mxfp4.py:38-96`.
- Router logits and selection are FP32. Selection uses
  `sigmoid(logit)+correction_bias`, with lower expert ID as the exact tie break;
  route weights use the unbiased sigmoid values, renormalized in FP32
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py:32-73`).
- Routed numerical order is non-negotiable:

  1. W13 dot accumulates FP32 and rounds to BF16.
  2. SiTU computes `4*tanh(g/4)*sigmoid(g)` and `25*tanh(u/25)` in FP32,
     multiplies, and rounds to BF16.
  3. W2 accumulates FP32 and each route rounds to BF16.
  4. Route weight is applied in FP32; local routes accumulate in ascending
     top-k slot order; the local partial rounds to BF16.
  5. The joint reduction preserves the qualified Iris local-then-increasing-peer
     FP32 order and BF16 output before routed RMSNorm/up.

  The reference boundaries are visible in
  `tokenspeed-kernel/test/kimi3_reference.py:9-23,119-184`; the current decode
  stage explicitly rounds W2 before weighting at
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_decode.py:376-438`.
- The shared branch is TP8 partial work. Shared and routed partials join one
  reduction, then routed RMSNorm/up adds the reduced shared result and
  `prefix_sum` (`python/tokenspeed/runtime/models/kimi_k3.py:239-293,1233-1268`
  and
  `python/tokenspeed/runtime/layers/moe/latent.py:581-613`).

## Evidence that constrains the redesign

- External production control, 8x MI350X, real checkpoint, 4k/1k:
  EP8 is 58.98/207.05/299.45 output tok/s at B1/B8/B16. B8 stage1/W2/Iris
  medians are 64.50/31.68/65.18 us per layer; B16 summed per-step work is
  11.41/5.22/10.90 ms for those same groups. See
  [issue #55](https://github.com/raikonenfnu/tokenspeed/issues/55).
- Historical matched batch-one graph evidence has a 17.415 ms replay: 7.932 ms fused MoE,
  9.184 ms non-MoE, 0.300 ms idle. The six MoE leaves are 640-workgroup input,
  one-workgroup top-k, 6144-workgroup W13, 1344-workgroup W2/shared, 21-workgroup
  Iris, and 224-workgroup final up. The ATT dispatch table is reproducible from
  `profile_default/kimi-k3-main-04bc0864-graph-20260813T062945Z/att/capture-final/134581`.
- ATT is diagnostic, not timing authority. Non-B1 grouped mechanism controls
  20608/20609 show W13 led by `s_waitcnt vmcnt(2)`, multiple `s_barrier`, and
  `s_waitcnt vmcnt(1)`, while W2 is led by `s_waitcnt vmcnt(3)`. They reject
  generic “double buffer” claims whose decoded waves remain load -> wait ->
  barrier -> compute, but are not an exact-B1 timeline. The retained capture
  has exact-B1 input/W2/final decoded controls; exact-B1 W13 itself was not
  decoded.
- The pre-refresh ordinary-overlap MegaMoE was still slower: about 25.31 ms unprofiled
  TPOT and 15.644 ms MegaMoE body per 92 layers versus 7.932 ms fused MoE. The
  missing time is inside the body, not launch idle.
- That historical admitted B1 ordinary object is vector GEMV, not MFMA, and is
  already an
  I-cache/control warning: its exact rank-zero `.text` is 80,384 bytes (the
  whole HSACO is 135,840 bytes), with 13,653 static ISA lines, 1,347 `waitcnt`,
  and 332 barriers (branch-expanded static counts from the coordinating lane).
- The same admitted object contains no `s_clause`. A conservative partition at
  waits, barriers, stores, and atomics finds 527 global loads in 514 issue
  groups: 506 groups contain one load and the maximum is four. Static expansion
  cannot prove dynamic issue, but it makes longer independent VMEM groups and
  cross-wave VMEM/VALU overlap an explicit ATT requirement for any replacement
  software pipeline.
- The refreshed native gfx950 EP8 control is already route-direct for every
  target batch: `_ROUTE_DIRECT_DECODE_MAX_TOKENS=16`, and the EP8 registry
  selects the linear-weight warp-GEMV path for B1/B8/B16
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:34-35,99-203`).
  One subgroup
  reduces K for each small output-column block without sorting routes or
  padding each route to BM64; W2 visits original top-k slots and applies its
  BF16 boundary before FP32 weighting
  (`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_decode.py:21-36,71-263,273-438`).
  BM64
  remains a grouped-kernel counterfactual, not an explanation of issue #55.
- The refreshed B1 composition is also stronger than the historical six-leaf
  label suggests: `latent_moe_expert_shared` combines routed W2 and shared-down
  production, then producer-direct Iris reduces their consecutive views before
  the fused final RMSNorm/up/add
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/latent_decode.py:21-77`,
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/latent_shared_decode.py:13-53`,
  `python/tokenspeed/runtime/layers/moe/latent.py:187-232`, and
  `python/tokenspeed/runtime/models/kimi_k3.py:1454-1500`). This exact path is
  the B1 control.
- Four-subgroup expert work was about 24.8% slower than eight subgroups; `.ca`
  peer loads were about 22% slower than the qualified `.cv` candidate; a source
  two-group K-loop was about 54.5% slower; deeper Iris source preloads compiled
  to a maximum issue window of one. These are negative controls, not universal
  proofs about a new resource envelope.
- P256 ordinary launch improved the W242 control modestly (~0.93% cold p50,
  9.42% p95), so P240 has no privileged status. A 64 KiB variant still had one
  resident workgroup/compute unit because registers remained limiting.
- Whole-expert XCD affinity and integrated distributed-XCD roles regressed even
  when isolated XCD probes improved. Keep expert placement global at B1.
- Adding per-route events to the old dual-role P240 graph is decisively useless:
  it adds `226*(local_routes-1)` acquires and gives zero modeled makespan gain
  for local-route counts 1..8, because the final W13 producer still owns the
  complete W2 suffix. Events matter only after producer/consumer ownership is
  separated.
- The removed pre-refresh MegaMoE workspace was B1-only with 16 route slots.
  No `ops/moe/megamoe` production package or enable flag exists on the refreshed
  base. A clean implementation therefore introduces its B1/B8/B16 ABI anew;
  B8/B16 require 128 and 256 route records.

## Source transfers, with exclusions

### Source hierarchy

- Refreshed TokenSpeed MoE kernels are the exact B1/B8/B16 performance and
  numerical controls.
- `/home/ericfeng/distributed/triton` is the primary pure-gfx950 Gluon/compiler
  precedent. Fleet and Iris under `/home/ericfeng/distributed/megakernels` are
  the primary AMD megakernel and communication-mechanism precedents.
- Mixture-of-Kittens is the design-quality target for explicit task formulas,
  stage lifetimes, and machine-checkable invariants; its Blackwell mechanics do
  not transfer.
- The `vroomvroom` lens is only a communication/source-review guardrail for
  Triton-shmem-style concerns. It is neither evidence nor an implementation
  template and cannot override the audited Gluon, Fleet, Iris, or TokenSpeed
  sources.

- Current TokenSpeed MoE is a first-class control, not merely a numerical
  oracle. Transfer the EP8 linear storage contract and local-ID guards from
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:70-203`;
  packed-weight/scale
  addressing, CDNA4 `scaled_upcast`, SiTU, and slot-ordered W2 combine from
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_decode.py:71-438`;
  and the B1 shared-down composition from
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/latent_shared_decode.py:13-53`.
  Retain the split
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/{moe,routing,pipelined_program,pipelined_kernel,medium_decode,warp_decode}.py`
  family as the MFMA/pipeline control for
  batched experiments. Reuse its gdot128 preshuffle and canonical scale layout
  (`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/weight_preprocess.py:161-294,308-423`
  and
  `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/scale_layout.py:21-66,146-197`);
  do not infer that its grouped
  schedule is best for sparse EP8 route ownership.

- Mixture-of-Kittens transfers an explicit task/event DAG, distinct compute and
  communication workers, medium-grained readiness, deterministic schedules,
  and generation-safe ring reuse
  (`/home/ericfeng/distributed/megakernels/mixture-of-kittens/deep-dive.txt:168-187,221-253`;
  loader and consumer roles in
  `/home/ericfeng/distributed/megakernels/mixture-of-kittens/csrc/mok_megakernel.cuh:1490-1785`).
  It does **not**
  transfer Blackwell TMA/TMEM/CLC, 256-row tiles, training backward, or
  dispatch/combine all-to-all: Kimi's tokens are replicated.
- Iris transfers exact symmetric pointer translation and system-scope atomic
  semantics
  (`/home/ericfeng/distributed/iris/iris/mem/triton/context.py:17-28,468-565`).
  Refreshed
  TokenSpeed producer-direct Iris additionally proves the desired local-first,
  increasing-peer recurrence and a second completion epoch before buffer reuse
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/iris.py:759-897`).
  The clean design
  keeps separate 64-bit ready/completion planes for clearer generations and
  fail-stop behavior rather than copying the current shared INT32 epoch plane.
  GPT-OSS shows
  that a producer and consumer owned by one workgroup can remove a phase edge,
  and that fewer resident workgroups can beat a full grid
  (`/home/ericfeng/distributed/megakernels/iris/examples/33_gpt_oss_megakernel/gpt_oss_120b_quantized_megakernel.py:53-56,191-259`).
  Its
  fixed top-k-slot GPU roles are not Kimi EP8 expert ownership.
- Fleet transfers runtime XCC identification, two-level event aggregation, and
  M-fast/windowed traversal of repeated rows against one weight tile
  (`/home/ericfeng/distributed/megakernels/fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/gang_linear_mi300.cuh:23-76`
  and
  `/home/ericfeng/distributed/megakernels/fleet-chiplet-megakernel/include/mirage/persistent_kernel/persistent_kernel.cuh:1214-1245`).
  Its one-expert-per-XCD MoE mapping is a
  rejected B1 control.
- CDNA4 has eight XCDs/256 compute units, 32 KiB vector L1 per compute unit,
  4 MiB L2 per XCD, and 256 MiB MALL. Vector `SC=device/system` bypasses L2 on a
  multi-L2 device; bare/group non-temporal accesses retain L2 hit-stream
  behavior
  (`/home/ericfeng/distributed/amdgpu-isa-manuals/cdna4/README.md:2823-2861`).
  Cache policy does not publish data. Cross-GPU payload requires system
  release/acquire.

### `comms-notes` audit

The communication notes were treated as review evidence, not as a second
implementation:

- `/home/ericfeng/distributed/comms-notes/building-megakernels/README.md:314-339`
  transfers the exact
  payload -> system release -> per-consumer acquire ordering and symmetric
  heap-offset invariant. It rejects Iris GPT-OSS's fixed slot-rank ownership
  and its unqualified resident-grid assumption for Kimi EP8.
- `/home/ericfeng/distributed/comms-notes/comms-libraries-comparison/README.md:31-64,238-260,302-319`
  establishes that cross-GPU barriers are memory protocols, direct peer loads
  have no transport queue to `quiet`, release compiles with L2 writeback, and
  acquire with invalidation. The redesign therefore rejects `quiet` as a
  generic correctness step and does not confuse a transport backend with the
  memory-ordering contract.
- `/home/ericfeng/distributed/comms-notes/kimi2.5comms.txt:96-145,147-167`
  identifies the readiness-only
  overwrite race and the numerical difference between the custom FP32 rank
  accumulation and a built-in pairwise-BF16 collective. The design transfers
  an explicit consumer-completion handshake and rejects both single-phase reuse
  and substitution of a numerically different collective. It does not claim
  alternate payload storage that the workspace does not allocate.
- `/home/ericfeng/distributed/comms-notes/megakernels/cache-policy.txt:33-104,106-155`
  separates
  placement, replacement hints, and publication. It transfers XCD-local
  traversal and optional two-level event aggregation as measured candidates;
  it rejects copying Fleet's device-scope NT weight modifier, assuming NT is a
  release, or assuming B1 has useful same-weight M reuse.

## B1 dataflow

```text
K0 router tiles -> top-k -> compact rank-local routes in slot order
K0 routed tiles -----------------------------> route W13 producers
K0 shared gate/up ---------------------------> shared-down flex workers

W13 route 0 --release--> W2 output-tile consumers --+
W13 route 1 --release--> same consumers -----------+--> routed 512-tile ready
... consumers retain FP32 accumulators in slot order+

shared 512-tile ready ----+
routed 512-tile ready ----+--> dedicated Iris worker -> reduced tile ready
all 7 routed tiles -> one RMS scalar -> K2 output tiles wait matching shared tile
```

- W13 task claims are ordered first by top-k slot and then by output tile so the
  cohort completes early routes rather than striping all 16 routes. The last
  tile arrival for a route publishes its event only after all 3072 BF16
  intermediate values are visible.
- W2 workers own output columns, not routes. Each acquires a completed route,
  computes its W2 output tile, applies the BF16 route boundary and FP32 weight,
  and retains its accumulator while W13 advances to the next route. This is the
  producer/consumer pipeline the old dual-role graph could not express.
- B1 does not materialize `[16,3584]` W2 route outputs. One W2 worker acquires
  each route once and computes one or more fixed output tiles; the starting
  96-worker cohort has 112 N32 tiles.
- W2 workers use otherwise-waiting time for shared-down tiles. Communication
  workers never steal a compute task once a peer tile is eligible; this avoids
  a compute tail delaying the only worker that can unblock all ranks.
- Rank zero-route is ordinary: its W13/W2 cohorts help the shared branch, publish
  zero routed tiles, and participate in all 21 joint-reduction tiles.

## B8/B16 dataflow

- K0 scans token-major/slot-major route records, counts 112 local experts,
  computes a 113-entry prefix, and stably scatters by
  `(local_expert, token, slot)`. No host synchronization or cross-GPU dispatch.
- Uniform-like routing is sparse within each rank. The host oracle's seeded B8
  case has 10-24 routes and 10-23 active experts per rank; B16 has 20-44 routes
  and 18-35 active experts. The current route-direct kernel already executes
  exact rows. A BM64 grouped alternative would expand mostly-singleton experts
  by roughly 50-64x, so it remains only a negative control. The clean-sheet
  opportunity is stable same-expert bucketing for weight reuse and selective
  MFMA, while retaining exact row work.
- Dispatch each expert segment by count:

  - count 1-2: M-fast vector GEMV, eight subgroups, no MFMA/LDS staging tax;
  - count 3-4 initially benchmarks both vector GEMV and masked M4 MFMA;
  - count >=4: exact-tail M4/M8 MFMA candidate, only if ATT proves row reuse;
  - never round a segment to BM64; tail masks preserve every legal route.

- W13 publishes one expert-microbatch event. W2 writes BF16 route outputs in
  stable route-record positions. A lightweight combine task consumes a token's
  local slots in ascending top-k order and publishes a routed 512-column tile.
- Shared and routed communication use `21*B` token/tile events. This is the
  medium granularity: smaller than a whole batch, larger than one output tile.
  It lets early tokens reduce while later expert groups execute without a
  quadratic flag surface.
- XCD M-window experiment: keep repeated `(expert, weight-N-tile)` work on one
  XCD and sweep its M rows before advancing N. Compare against a global queue.
  Do not bind the complete W13->W2 chain to an XCD without measuring the
  cross-XCD intermediate handoff and rank-skew tail.

## K1 cohorts and proposed software pipelines

P256 is valid only after a fresh code-object admission proves all 256
workgroups resident **and a normal timeline proves their physical placement:**
one workgroup per compute unit, 32 per XCD, with no missing/duplicate compute
units and no launch tail before the first polling epoch. Exact 256/256 capacity
has no scheduling slack. The role map must compare PID ranges against XCD-local
16/12/4 W13/W2/communication ranks; the latter is the requested per-XCD Iris
batch experiment, not the correctness path.

| Epoch/body | Subgroup issue plan | Intended overlap | Expected synchronization |
|---|---|---|---|
| B1 W13 vector | 8 subgroups own independent paired gate/up N tiles; each issues packed weight+scale VMEM, scaled-upcast/VALU, then vector FMA | issue K+1 VMEM before K VALU reduction; two subgroup-private buffers | no steady-state workgroup barrier; one payload-drain/barrier/release at route completion |
| B1 W2 control | 8 subgroups own independent output tiles and load the common intermediate through cache | other resident subgroups hide VMEM latency | one acquire per consumer workgroup/route; no K-loop barrier |
| B1 W2 staged A/B | two four-subgroup output cohorts share a ping-pong intermediate K tile in LDS while loading disjoint weights | one cohort consumes K while the other prepares K+1, then roles swap | at most one `waitcnt`+barrier per K stage; reject if either four-subgroup cohort is visibly idle |
| B8/B16 vector | same as B1, but sweep same-expert M rows before advancing N | weight cache reuse across exact rows; no MFMA setup | segment boundary only |
| B8/B16 MFMA | 8 subgroups cooperatively load exact M4/M8 A and MXFP4 B; next A tile enters the second LDS buffer during current MFMA | VMEM/LDS for K+1 overlaps `v_mfma` for K | one async wait/barrier per K stage, not multiple serialized waits |
| Iris direct-acquire | 32 workgroups; eight subgroups split one 512-element joint tile, and every subgroup that issues peer VMEM system-acquires that peer's exact ready generation first | remote VMEM and FP32 VALU additions across eight independently qualified subgroups; compute cohorts remain runnable | seven peer acquires per loading subgroup; final workgroup barrier before one completion release |
| Iris leader-staged A/B | one subgroup system-acquires each peer and loads its full 512-element tile into fresh/versioned LDS; all eight subgroups consume only after a workgroup barrier | leader peer VMEM overlaps follower accumulation of the preceding peer when two LDS stages fit | one acquire per peer by the only subgroup issuing peer VMEM, one wait/barrier per peer stage, final completion release |

Two-stage K loops are promoted only from emitted ISA. Source `num_stages=2` or
two arrays is not evidence. W2 four-plus-four staging saves repeated
intermediate traffic but cannot share disjoint W2 weights; it is a targeted
latency experiment, not the default.

## Communication, visibility, and fail-stop

- Symmetric address is `peer_heap_base + (local_pointer-local_heap_base)`. Direct
  peer-pointer tables are allowed only if built from that exact translation and
  retained graph-stably.
- Local/cross-XCD edge: write payload; device release event; **every consuming
  workgroup** performs acquire before loading, unless a separately proved
  transitive publication copies into fresh/versioned storage. One leader's
  acquire does not invalidate every compute unit.
- Peer edge: finish local payload, system-release exact generation to the peer
  flag, system-acquire every required peer generation, then load payload. Sum
  local first and peer IDs in increasing order to retain Iris arithmetic.
- A system acquire applies only to the subgroup that executes it; it does not
  repair sibling subgroups' cached peer lines. The correctness baseline makes
  every subgroup that issues peer VMEM acquire the exact peer-ready generation.
  The A/B alternative permits one acquiring subgroup to load the full peer tile
  into fresh/versioned LDS, followed by a workgroup barrier; followers then read
  LDS, never the peer pointer. A leader-only acquire followed by sibling peer
  loads is forbidden unless a separate leader-release/follower-acquire
  transitivity litmus proves it on every target placement.
- Payload reuse uses one symmetric producer buffer and two distinct 64-bit
  control planes, `ready[tile, producer_rank]` and
  `completion[tile, consumer_rank]`. Before writing generation `g`, a producer
  has observed its local consumer and system-acquired every remote consumer's
  completion at exact generation `g-1` (generation zero is the initialized
  state). The communication worker device-acquires its local partial event,
  stores the one payload, and system-releases ready `g`. For every remote
  producer, either every peer-loading subgroup acquires ready `g`, or the sole
  peer-loading leader acquires and stages into versioned LDS as specified above.
  The workgroup accumulates local first and remote producer ranks in increasing
  order, then publishes one consumer completion `g` only after all eight
  subgroups finish. A producer cannot overwrite for `g+1` until its local
  consumer and every remote consumer have completed `g`. Flags increase
  monotonically; there is no reset-to-zero protocol, `>=` match, alternate flag
  bank, or second payload slot.
- Qualify both Iris choices with a subgroup-distinct stale-payload litmus. Every
  follower subgroup first caches a different old peer slice; the producer then
  writes alternating generation-specific patterns and releases ready under
  deliberate rank skew. The direct variant must have each loading subgroup
  acquire before rereading its slice. The staged variant must have only the
  leader issue peer VMEM into a newly versioned LDS stage before the barrier.
  Run many generations across every XCD placement and reject any
  mixed-generation tile or completion published before all eight subgroup
  checks pass.
- Hot-loop production polls may use the measured no-timeout/runtime-loop form
  only after resident-grid, graph-replay, and fault-injection qualification.
  Qualification builds retain bounded polling. Any timeout system-releases a
  sticky fatal epoch to all ranks; every cohort suppresses output publication.
  A timeout must fail closed, never return a partial numerical result.
- Cache hints are independently swept. The first controls are the qualified
  `.cv` peer path and ordinary/bare-NT local weight loads; neither substitutes
  for an ordering edge.

## Workspace/API contract

- Compile separate graph-stable B1, B8, and B16 plans. Batch is a specialization,
  not a runtime loop over the old B1 ABI.
- Replace the positional flat workspace with typed local and symmetric regions:

  - local `[B,896]` FP32 logits, `[B,3584]` BF16 routed input,
    `[B,768]` BF16 shared input, `[B,16]` IDs/weights;
  - `B*16` packed route records, 112 counts, 113 offsets;
  - `[B*16,3072]` BF16 W13 intermediate;
  - B8/B16 only `[B*16,3584]` BF16 route output; B1 uses output-centric W2;
  - symmetric producer and local reduced `[B,10752]` BF16, laid out as
    `[shared H | routed L]` to match the refreshed producer-direct Iris views;
  - generation-tagged local task events and distinct symmetric ready/completion
    planes, each `[21*B,8]` 64-bit flags.

- Max B16 payloads are small enough to buy clarity: 1.573 MiB W13 workspace,
  1.835 MiB W2-route workspace, 344,064-byte joint symmetric payload, and
  21,504 bytes each for the ready and completion planes (336x8 64-bit flags per
  plane). The total symmetric control footprint is 43,008 bytes. These buffers
  are model-level reusable, not allocated per layer.
- The host oracle is
  `tools/megamoe/kimi_k3_ep8_cleansheet_oracle.py`. It imports no Torch/ROCm and
  emits exact shapes, layouts, ownership, schedules, combine slots, DAGs, task
  counts, route-direct/grouped/bucketed row-work controls, workspace, and
  traffic bounds:

  ```bash
  /home/ericfeng/distributed/.venvs/tokenspeed/bin/python \
    tools/megamoe/kimi_k3_ep8_cleansheet_oracle.py --check --pretty
  ```

## Quantitative bounds

- Dense BF16 weight bytes per rank/layer: 148,642,816. One cold routed expert is
  11,698,176 bytes W13 + 5,849,088 bytes W2 = 17,547,264 bytes.
- Per route: W13 is 44,040,192 FLOPs; W2 is 22,020,096 FLOPs. The useful dense
  work per token across router/down/shared/up is 148,635,648 FLOPs.
- Joint producer payload is 21,504/172,032/344,064 bytes at B1/B8/B16. Each rank
  remotely reads seven times those payloads: 150,528/1,204,224/2,408,448 bytes.
- Weight traffic for a rank lies between
  `unique_local_experts*17,547,264` (perfect batch reuse) and
  `local_routes*17,547,264` bytes (cold every route). Report both; actual route
  histograms and cache counters decide.
- Service-time model for a sequential route pipeline is symbolic:
  `T = t_w13 + t_w2 + (L-1)*max(t_w13,t_w2)`, not
  `L*(t_w13+t_w2)`. It applies only after a timeline proves separate resident
  producers/consumers. The oracle deliberately does not invent `t_w13/t_w2`.

## Launch-organization decision

| Candidate | Finding |
|---|---|
| One all-phase persistent kernel | Defer. It offers maximum overlap but gives every workgroup the union register/LDS/code footprint. The **historical pre-refresh** 156-VGPR/128-KiB/large-object control and 7.7 ms MoE-body loss make it the highest-risk first step. |
| Persistent compute plus concurrent communication kernel | Reject until a real CU-partition mechanism is qualified. A resident compute grid can starve polling communication, and HIP stream order does not reserve compute units. |
| K0 -> resident K1 core+communication -> K2 | Recommend. It keeps profitable routed/shared/communication events in one resident grid and lets projection/final kernels retain phase-specific resources. The historical six-leaf comparison suggested three fewer boundaries; recount them against the refreshed B1 fused composition before claiming a current launch saving. |
| Fully bounded nonpersistent kernels | Retain as a timed numerical/performance control. It sacrifices tile-ready communication overlap but cannot resident-grid deadlock, and directly tests whether smaller hot loops beat the 80,384-byte persistent control through lower instruction-fetch pressure. |

## CU/SIMD and profiler promotion gate

Nominal occupancy never promotes a candidate. For every B1/B8/B16 binary and
rank specialization, record code-object metadata and disassembly, a normal
all-rank timeline, and targeted ATT for an early, middle, and tail task on the
slowest rank plus at least one workgroup per XCD.

- **Resources/residency:** VGPR, SGPR, LDS, scratch, spills, waves-per-EU, code
  size, and actual active/resident waves. P256 K1 requires one resident
  eight-subgroup workgroup on every compute unit before any polling path is
  legal. The first probe must report the physical `(rank, XCD, compute unit,
  workgroup, cohort)` map and epoch timestamps, not merely the 256-workgroup
  count. Fewer than eight active waves in a ready W13/W2 workgroup, any missing
  compute unit, or a launch tail crossing the polling epoch falsifies the
  latency-hiding/liveness model.
- **Issue evidence:** use ATT counters `SQ_BUSY_CU_CYCLES`,
  `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_ACTIVE_INST_VALU`, `SQ_ACTIVE_INST_LDS`,
  `SQ_ACTIVE_INST_VMEM`, `SQ_ACTIVE_INST_FLAT`, `SQ_ACTIVE_INST_SCA`, and
  `SQ_ACTIVE_INST_MISC`, plus per-instruction `Latency/Stall/Idle` and decoded
  wave state. B1 must show interleaved VMEM and VALU with zero MFMA. B8/B16 MFMA
  tasks must show K+1 VMEM/LDS issued before K MFMA retires. A repeated
  VMEM -> immediate `s_waitcnt` -> `s_barrier` -> MFMA chain falsifies the
  software pipeline even when metadata says `num_stages=2`.
- **VMEM grouping:** the historical pre-refresh object has no `s_clause` and
  506 of 514
  conservative global-load issue groups contain one load. Candidate
  disassembly plus live-wave ATT must show longer independent VMEM issue before
  dependent waits, or show that other ready waves cover those waits with VALU
  or MFMA. If the hot loop reproduces isolated load -> wait chains and
  `SQ_BUSY_CU_CYCLES` falls while ready work exists, the pipeline is rejected.
- **Wait/hazard cadence:** disassembly must show only a loop backedge per K
  stage, no 16-way branch-expanded route body, and the synchronization cadence
  in the cohort table above. `waitcnt`, dependency, clause, or barrier rows
  dominating the same-batch baseline's ATT stalls reject the variant. Inspect
  hazard spacing around scaled-upcast, LDS reads, MFMA/VALU consumers, payload
  stores, and release atomics; source order is insufficient.
- **B1 regime:** compare against the no-MFMA vector control. Any MFMA setup,
  subgroup idleness, or staged-LDS barrier tax without lower normal duration is
  a rejection. W2 four-plus-four must show both cohorts progressing between
  barriers; one cohort waiting through the other's whole K body falsifies it.
- **B8/B16 regime:** stratify ATT by expert count. Vector must win or remain for
  count 1-2. MFMA is admitted for a count bucket only when normal time per useful
  row falls and decoded waves show concurrent MFMA/load progress. A faster hot
  expert case cannot justify MFMA for uniform-like singleton routing.
- **I-cache/control:** K1 must be materially smaller than the historical
  pre-refresh all-phase
  80,384-byte `.text`, 13,653-line/1,347-waitcnt/332-barrier object. A K1 object
  meeting or exceeding any of those static counts is rejected unless
  instruction-fetch evidence and normal timing independently clear it. Compare
  the resident K1 against fully bounded W13/W2/combine kernels on identical
  tasks; ATT must check instruction-fetch stalls and hot-loop residency before
  and after the split. Runtime route/peer loops are preferred to
  branch-expanded copies.
- **Balance:** normal timelines report p50/p95 and slowest rank, per-XCD task
  start/end, and per-compute-unit ready versus busy time. Reject an XCD mapping
  if the same XCD has a persistent ready-work tail or if slowest-XCD completion
  worsens versus PID/global-queue control. ATT samples must not come only from a
  fast XCD.
- **Communication caveat:** never use ATT duration of the spin-wait Iris body as
  serving latency; the reference ATT inflated it to 2.86 ms. Use normal traces
  for duration/skew and ATT only for polling/load/order behavior. Decode one
  wave per subgroup in the direct-acquire variant to prove each peer-loading
  subgroup executes the acquire; in the staged variant, prove only the leader
  issues peer VMEM and followers consume post-barrier LDS.
- **Numerics/liveness:** require bitwise M-specific kernel/model equivalence at
  every defined BF16/FP32 boundary for B1/B8/B16, including zero-route,
  balanced, hot-expert, all-routes-one-rank, random, generation wrap-nearby,
  delayed-peer, and injected-fatal cases. B1 serving retains exact prompt and
  completion identity. The retained B8/B16 baseline is not byte-deterministic,
  so those serving arms require an identical prompt multiset, exact
  4096-to-1024 success/token counts, and retention—not equality—of every
  completion hash. Every batch also needs positive loaded-code/dispatch proof
  that MegaMoE executed instead of the native shape fallback.
- **Final performance:** fresh top-of-main, same checkpoint/prompt/graph and
  three warmed runs; gate on median and p95 output tok/s plus slowest-rank MoE
  critical path. A win at any one of B1/B8/B16 admits only that specialization.

## Smallest post-refresh discriminator

1. The selective refresh onto `e7842295` is complete. Record same-checkout
   B1/B8/B16 output tok/s and per-layer controls before implementation: the
   exact route-direct path for all three batches and the stronger fused
   routed/shared/producer-direct-Iris composition used at B1. Do not recreate
   the removed Mega workspace first.
2. Profile the existing exact route-direct
   `_stage1_a16w4_situ_warp_gemv`/W2 body on saved B8/B16 routing. Stratify by
   expert count and record weight bytes/cache behavior, useful-row service
   time, and rank tail. This attributes the issue-55 slope among cold repeated
   weight traversal, available same-expert reuse, and route scheduling; BM64 is
   not present in this control.
3. First implementation: a standalone ordinary-launch **B8 local K1 core**. It
   consumes routed input/top-k, builds the 128-record stable local schedule,
   uses disjoint P256 W13/W2 cohorts, exact-ragged vector/MFMA count buckets,
   materializes BF16 route outputs, and performs deterministic combine. It has
   no Iris, K0, K2, peer state, or production API.
4. Instrument that first K1 launch to record its physical P256 placement and
   first/last task epoch per compute unit/XCD. Keep a fully bounded
   W13 -> W2 -> combine implementation as a timed control for the
   80,384-byte/I-cache hypothesis; do not infer residency from launch size.
5. Compare both organizations to the same-checkout route-direct kernels on
   identical saved routing for balanced-unique, hot, skewed, and real-checkpoint
   histograms. At B1 also compare the full integrated critical path to the
   refreshed fused expert/shared/Iris composition; a routed-core-only win is not
   an end-to-end claim. Require correctness, lower normal makespan, and the
   batch-specific ATT gate. Only then add the 32-worker joint-reduction cohort
   and integrate K0/K2.

This discriminator tests the two independent clean-sheet claims with the least
surface: reusing weights across stable exact-row expert buckets and making
route-event overlap real via separate ownership. Failure is useful and does not
disturb production.
