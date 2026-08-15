# Kimi K3 TP8/EP1 MegaMoE: clean-sheet design

Status: host-only research artifact; no TP8 production implementation,
compilation, GPU run, or serving-performance claim. Selectively refreshed in
worktree `agent/megamoe-cleansheets-main-refresh` onto `origin/main`
`e784229526ce11d272a3c4a0b3f64ab9a8973491` on 2026-08-15. The original
clean-sheet artifact was frozen at
`19e30183469abc87b6823d25c8c3cdd7b02769f7`; historical control evidence came
from pre-refresh commit `be7d5001927c9613314f300a420fb88c4bdd4628`.
Target: gfx950, TP8/EP1, exact active batches `B in {1, 8, 16}`

## Answer and guardrails

- This is an independently derived TP8 design. It starts from checkpoint
  ownership and the model equations, not the EP8 MegaMoE workgroup assignment,
  route-owner state machine, or XCD mapping. No TP8 implementation exists in
  this artifact, so schedule independence and performance remain future
  implementation evidence rather than conclusions.
- EP8 is used only as a numerical and throughput comparator.
- The pre-refresh production MegaMoE package and enable flag are absent on the
  refreshed base. Historical ordinary-ISA measurements remain evidence only;
  they are not a current-checkout serving baseline.
- The failed `situ_decode` K-divisibility/layout probe rejects that one kernel
  expression. It is not evidence against TP8: logical routed intermediate
  width `384` is MXFP4-group exact and W2 needs no `512` padding.
- Begin with one persistent kernel per rank, while retaining bounded split
  kernels as first-class controls. Correctness, then end-to-end time, decides.
- No XCD affinity is presumed. Every affinity mode remains an experiment until
  a representative timeline and targeted ATT establish locality and balance.
- Eight subgroups per workgroup are retained for the first discriminator, but
  `224`, `240`, and `248` workgroups are a measured sweep, not constants.
- The machine-readable contract is
  [`kimi-k3-tp8-cleansheet-oracle-20260814.json`](probes/kimi-k3-tp8-cleansheet-oracle-20260814.json).
  Its pure-standard-library checker is
  [`verify_kimi_k3_tp8_cleansheet_oracle.py`](../../tools/megamoe/verify_kimi_k3_tp8_cleansheet_oracle.py).

## Audited sources and what transfers

- Model truth:
  - [`kimi_k3.py`](../../python/tokenspeed/runtime/models/kimi_k3.py#L1045)
    fixes FP32 router logits, latent `7168 -> 3584 -> 7168`, Kimi top-16,
    shared TP ownership, SiTU configuration, and the capture-time call paths at
    lines 1045--1289 and 1454--1582.
  - [`latent.py`](../../python/tokenspeed/runtime/layers/moe/latent.py#L235)
    fixes the routed/shared/final order and the construction-time reduction
    policy at lines 235--294 and 507--791.
  - [`latent_input.py`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/latent_input.py#L63)
    fixes FP32/BF16 materialization at the input projections.
  - [`mxfp4.py`](../../python/tokenspeed/runtime/layers/moe/weights/mxfp4.py#L38),
    [`loaders.py`](../../python/tokenspeed/runtime/layers/moe/weights/loaders.py#L41),
    and [`expert.py`](../../python/tokenspeed/runtime/layers/moe/expert.py#L120)
    prove the independent TP row sharding and all-896-experts-per-rank layout.
  - [`kimi3_sigmoid_topk.py`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/kimi3_sigmoid_topk.py#L20),
    [`situ_decode.py`](../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_decode.py#L71),
    and [`rmsnorm_linear_add.py`](../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/gemm/fp16/rmsnorm_linear_add.py#L36)
    fix top-k tie-breaking and the important BF16 boundaries. `situ_decode.py`
    is an EP8 implementation reference, not a TP8-compatible registered path.
  - [`communication/iris.py`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/iris.py#L759)
    fixes the qualified recurrence and reuse handshake: initialize from the
    local rank, add peer group ranks in increasing order while skipping local,
    then complete a second epoch before reusing the producer buffer. This is not
    generic globally rank-ascending arithmetic on ranks 1--7.
- Refreshed current-checkout controls:
  - The portable precomputed
    [`triton/mxfp4.py`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/mxfp4.py#L920)
    accepts SiTU, `supports_ep=False`, and `ispp` aligned to 32/128 at lines
    920--1006. Together with the TP-sharded raw tensors, it is the available
    per-rank TP8 compute control, not a qualified reduced serving baseline.
  - The optimized gfx950 Gluon MXFP4 package is now split into
    `fused/{moe,routing,pipelined_program,pipelined_kernel,launch,gemm_api,
    medium_decode,warp_decode}.py`. Its gdot128 preshuffle, scale layout, ragged
    MFMA, and pipeline structure are component controls. The corresponding
    generic Gluon registry supports SiLU/SwiGLU rather than K3 SiTU
    (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:376-441`),
    so it must not be reported as a current
    TP8 K3 serving path.
  - Top main does **not** supply a complete qualified TP8/EP1 K3 reduction path.
    `Kimi3MoEExecutionPlan.joint_moe_reduce` requires TP1/EP>1
    (`python/tokenspeed/runtime/layers/moe/latent.py:288-293`), while
    `LatentMoELayer` installs a routed reducer only when `experts.ep_size > 1`
    (`python/tokenspeed/runtime/layers/moe/latent.py:539-565`). At TP8/EP1 that
    EP size is
    one even though W13/W2 are row-sharded; shared output is reduced separately,
    and AMD retains the replicated routed-up projection
    (`python/tokenspeed/runtime/models/kimi_k3.py:605-611,1199-1268`). Therefore
    fresh TP8 output tok/s requires
    a corrected reduction orchestration before it can be a correctness or
    performance control.

Repository-verified current TP8/EP1 construction consequence:

```text
KimiLinearMoE.__init__ (python/tokenspeed/runtime/models/kimi_k3.py:1061)
    -> MoELayer(tp_size=8, ep_size=1) (:1159-1192)
        -> TP-row-sharded W13/W2 allocation
           (python/tokenspeed/runtime/layers/moe/weights/mxfp4.py:38-96)
    -> LatentMoELayer(expert_parallel_group=ep_group) (:1248-1268)
        -> auto latent_reduce only if experts.ep_size > 1
           (python/tokenspeed/runtime/layers/moe/latent.py:539-565)
            -> not selected for TP8/EP1
```

The missing edge is repository-verified. A future corrected TP composition and
its performance are proposed, not inferred.
- `ep8vstp8.txt` supplied hypotheses; every shape and ownership claim used here
  was re-derived from the files above. Its old prototype timings remain
  comparator evidence, not a schedule template.
- Iris message passing and `iris/ops/matmul_all_reduce.py` transfer the required
  producer-store/system-release/remote-system-acquire/peer-load protocol. Their
  finite-lifetime examples have no persistent-reuse completion plane; this
  design adds it from the qualified K3 reuse contract and comms-notes audit.
  Cache hints never replace ordering.
- Iris expert-sharded MoE transfers stable expert buckets and formula-derived
  grouped tasks to the optional B8/B16 lane; its routing math does not transfer.
- Fleet transfers hierarchical per-XCD counters and arithmetic task ranges only.
  Fleet's XCD affinity is a hypothesis here because B1 has 16 distinct experts
  and therefore no same-expert weight reuse inside a token.
- Iris GPT-OSS transfers grid-strided phases and empirical grid sizing. Its
  `180` workgroups, four subgroups, and exact barriers are model-specific.
- Mixture-of-Kittens transfers task-count formulas, buffer epochs, and readiness
  attached to independently schedulable consumers. Blackwell TMA/TMEM/cluster
  mechanics and its EP training communication do not transfer to CDNA4.
- HazyResearch MegaKernels transfers explicit loader/consumer/storer lifetimes
  and ping-pong buffers, not CUDA-specific primitives.
- CDNA4 sources establish eight XCDs, up to 256 compute units, 160 KiB LDS per
  compute unit, 32 KiB L1 per compute unit, 4 MiB coherent L2 per XCD, and the
  system-scope cache/atomic rules. They establish feasibility, not speed.
- Historical real-checkpoint 4K numbers are comparator-only:
  [EP8 issue #55](https://github.com/raikonenfnu/tokenspeed/issues/55) reports
  `58.98/207.05/299.45 tok/s` at B1/B8/B16; [TP8 issue #56](https://github.com/raikonenfnu/tokenspeed/issues/56)
  reports `51.17/235.80/374.55 tok/s`. The latter used two reductions and is not
  this design.

### Source hierarchy

- Refreshed TokenSpeed MoE kernels are the exact API, layout, numerical, and
  per-rank compute controls. The portable SiTU path and optimized Gluon fused
  family are both first-class references with the compatibility limits above;
  neither silently fills the missing TP8 routed reduction.
- `/home/ericfeng/distributed/triton` is the primary pure-gfx950 Gluon and
  compiler precedent. Its CDNA4 Gluon tests establish direct global-to-LDS
  copies, scaled-MFMA layouts, and the requirement to inspect emitted TTGIR and
  AMDGCN for a requested pipeline; they do not establish this kernel's tiling.
- Fleet and Iris are the primary AMD megakernel/communication-mechanism
  precedents. Mixture-of-Kittens is the design-quality target for explicit task
  formulas, stage lifetimes, and machine-checkable invariants, while its
  Blackwell mechanisms remain non-transferable.
- The `vroomvroom` review lens is used only as a communication and source-review
  guardrail. It is neither evidence nor an implementation template and cannot
  override the audited repository sources.

### `/home/ericfeng/distributed/comms-notes` audit ledger

- `building-megakernels/kimik3_moe.txt:4-130,132-220`: transfer the exact K3
  branch dependencies, segmented routed/shared reduction, and absence of route
  A2A on replicated tokens. Reject its EP8 ownership and six-launch sequence as
  a TP8 schedule.
- `building-megakernels/kimik3_moe_bs1.txt:5-64`: transfer the graph-replay call
  path and normal-trace B1 kernel medians/workgroup counts as comparator facts.
  Reject ATT duration as timing evidence and reject non-B1 families for B1.
- `megakernels/fleet.txt:61-116`: transfer physical XCC discovery, XCD-local
  queues/counters, M-major traversal, and amortized global publication. Reject a
  dedicated scheduler compute unit and any assumption that its expert affinity
  produces reuse for K3 B1.
- `megakernels/cache-policy.txt:1-31,33-104,106-155`: transfer static XCD
  partitioning, two-level barriers, and the distinction between cache residency
  and visibility. Reject cache modifiers as ordering; reject device-scope NT1
  when the goal is XCD-local L2 reuse; do not copy Fleet constants blindly.
- `megakernels/iris_gpt_oss_megakernel.txt:1-44`: transfer grid-strided phases
  and the empirical barrier/grid tradeoff. Reject fixed 180-workgroup geometry
  and its rank-0/four-rank GPT-OSS transport protocol.
- `megakernels/hazyresearch.txt:180-244`: transfer explicit loader/consumer/
  storer lifetimes, dependency counters, and page reuse after completion.
  Reject CUDA subgroup-role counts and platform-specific async mechanisms.
- Protocol conclusion: local grid visibility, cross-XCD visibility, and
  cross-GPU symmetric-heap visibility are separate domains. This design uses a
  local generation gate for the first, and Iris system release/acquire plus a
  completion plane for the last; neither NT nor invalidate hints replace them.

### Other required local-tree audit ledger

- `/home/ericfeng/distributed/triton`:
  - `python/test/gluon/test_core.py:1669-1703,1837-1897` establishes pure Gluon
    CDNA4 direct-to-LDS ordering and scaled-MFMA operand/scale layouts.
  - `third_party/amd/python/test/test_warp_pipeline_gfx9.py:14-128` establishes
    gfx950 Gluon pipeline-stage lowering and emitted-ISA checks; and
    `test/TritonGPU/amd/amd-block-pingpong-asyncmark-multi-token.mlir:1-71`
    establishes that post-transform async-wait counts must be verified for
    gfx950 rather than inferred from source.
  - Transfer these compiler/API and inspection precedents. Reject CDNA5-only
    TDM/cluster examples and any claim that a source-level two-stage annotation
    proves VMEM/MFMA overlap.

- `/home/ericfeng/distributed/iris`:
  - `iris/ops/matmul_all_reduce.py:57-65,125-144` and
    `iris/mem/triton/context.py:734-840` establish one-shot pull versus two-shot
    owner/scatter and the `.wt` store -> barrier -> system release -> system
    acquire -> peer load chain. Transfer J0/J1; add the completion plane that
    these finite-lifetime examples do not provide for persistent buffer reuse.
  - `examples/06_message_passing/message_passing_device_context.py:24-89`
    establishes remote store/release and acquire-before-load. Transfer the
    memory-order relation, not its one-producer/one-consumer topology.
  - `examples/31_expert_sharded_moe/moe.py:110-153` and
    `grouped_matmul.py:47-98` establish expert-sorted metadata and one
    `(expert,N tile)` task looping its ragged M rows. Transfer this only to G;
    reject its softmax routing/reduction order as K3 numerical semantics.
- `/home/ericfeng/distributed/megakernels`:
  - `fleet-chiplet-megakernel/include/mirage/persistent_kernel/persistent_kernel.cuh:743-769,1082-1109,1214-1245`,
    `fleet-chiplet-megakernel/include/mirage/persistent_kernel/mpk_atoms.cuh:291-331`,
    and
    `fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/gang_linear_mi300.cuh:23-95`
    establish physical
    XCC discovery, local/global counters, publication, and M-major windows.
    Transfer mechanisms, not a presumed K3 affinity or scheduler role.
  - `iris/examples/33_gpt_oss_megakernel/gpt_oss_120b_quantized_megakernel.py:53-56,258-350,389-411`
    establishes empirical grid sizing, phase barriers, vector expert work, and
    staged K loops. Transfer principles, not `NUM_WG=180` or its geometry.
  - `mixture-of-kittens/csrc/mok_megakernel.cuh:1495-1760` establishes
    formula-derived tasks, multi-stage semaphores, and tile-ready consumers.
    Transfer scheduling algebra; reject Blackwell TMA/TMEM/CLC/clusters.
  - `Megakernels/demos/low-latency-llama/matvec_pipeline.cuh:29-241`
    establishes three-stage page/semaphore lifetimes. Transfer bounded
    ping-pong ownership; reject CUDA-specific roles and page machinery.
- `/home/ericfeng/distributed/amdgpu-isa-manuals/cdna4`:
  - `whitepaper.md:35,77-91` establishes eight XCDs, up to 256 compute units,
    160 KiB LDS, 32 KiB L1, 4 MiB/XCD L2, 256 MiB Infinity Cache, and 8 TB/s
    peak HBM.
  - `README.md:411,1094,2833-2870,23305` establishes XCC_ID, vector cache scope,
    atomic scope, and `BUFFER_WBL2`. Transfer hardware limits and ordering
    requirements; reject any inference that capacity or cache bits prove speed.

## Exact topology and tensors

- Constants: hidden `H=7168`, routed latent `L=3584`, experts `E=896`, top-k
  `K_t=16`, routed expert intermediate `I=3072`, TP ranks `P=8`, local routed
  intermediate `I_r=384`, and shared global intermediate `2I=6144` with local
  shared intermediate `768`.
- All tokens and routing decisions are replicated. Every TP rank owns the same
  896 expert IDs but only rows `r*384:(r+1)*384` of each expert's gate/up and
  the matching W2 input columns. There is no route all-to-all.
- Rank-local raw MXFP4 tensors are exact, unpadded:

| tensor | rank-local shape | ownership |
| --- | --- | --- |
| W13 packed | `[896, 768, 1792]` U8 | 384 gate + 384 up rows; 3584/2 packed K |
| W13 scale | `[896, 768, 112]` U8 | one UE8M0 scale per 32 K values |
| W2 packed | `[896, 3584, 192]` U8 | 384/2 packed input columns |
| W2 scale | `[896, 3584, 12]` U8 | one scale per 32 local K values |

- Replicated BF16 weights are router `[896,7168]`, routed down `[3584,7168]`,
  routed norm `[3584]`, and routed up `[7168,3584]`.
- Shared rank-local BF16 weights are gate/up `[1536,7168]` and down
  `[7168,768]`.
- Preprocessing may replace raw storage with an equivalent rank-local layout,
  but must not retain a second full raw/preshuffled expert representation.

## Equations and numerical materialization

For token `b`, route slot `j`, selected expert `e_j`, and rank `r`:

```text
q_b                 = FP32(h_b @ W_router^T)
s_b                 = sigmoid(q_b) in FP32
e_0..e_15           = descending top-16 of s_b + correction_bias
                       with lower expert id winning an exact tie
alpha_bj            = routed_scale * s_b[e_j] / sum_j(s_b[e_j]) in FP32
z_b                 = BF16(h_b @ W_latent_down^T)
g_bjr, u_bjr        = BF16(MXFP4_linear(z_b, G_ejr/U_ejr), FP32 accumulate)
a_bjr               = BF16((4*tanh(g_bjr/4)*sigmoid(g_bjr))
                              * (25*tanh(u_bjr/25)))
d_bjr               = BF16(MXFP4_linear(a_bjr, D_ejr), FP32 accumulate)
R_br                 = BF16(sum_j_in_original_slot_order(alpha_bj*FP32(d_bjr)))
S_br                 = BF16(shared_down_r(BF16(SiTU(shared_gate_up_r(h_b)))))
lane_br              = concat(S_br[H], R_br[L])
reduced_lane_br      = BF16(FP32(lane_br), then add FP32(lane_bp)
                              for peer p=0..7, p!=r, in increasing order)
normalized_br        = BF16(RMSNorm(reduced_lane_br.routed))
routed_up_br         = BF16(normalized_br @ W_latent_up^T)
output_br            = BF16(FP32(routed_up_br) + FP32(prefix_b)
                              + FP32(reduced_lane_br.shared))
```

- `concat` is layout, not an arithmetic sum: the one collective reduces
  corresponding shared and routed lane elements independently.
- Compatibility uses BF16 rank-local W2 and shared producer lanes. An FP32
  producer-lane experiment doubles communication bytes and is admitted only if
  the BF16 TP reference fails tolerance.
- Route accumulation stays in original top-k slot order. Expert sorting may
  reorder compute, never this combine.
- Every destination preserves qualified Iris arithmetic: its local contribution
  initializes FP32 accumulators, then peer group ranks are added in increasing
  order excluding local. This can produce destination-rank-specific last bits;
  it is deliberately not reassociated into a generic all-reduce tree. Direct
  atomics are rejected because they alter order and complicate zeroing/epochs.
- Deliberate choice: local-first is the first TP numerical contract because it
  minimizes risk by preserving qualified K3 Iris arithmetic; it does not import
  any EP task schedule. A global 0--7 contract would make one common BF16 lane
  and simplify J1, but is a new numerical mode and remains a later experiment
  requiring every-rank correctly reduced TP boundary qualification.
- Preserve the qualified repeated RMS calculation per output tile in the first
  discriminator. A once-per-token RMS scratch value is a later, separately
  qualified numerical optimization.

## Independent event graph

The single-rank work is replicated on all eight ranks:

```text
tp8_megamoe(hidden[B,H], prefix[B,H])
    -> packed input projections
        -> router logits FP32 -> stable top-k ids/weights FP32
        -> routed latent z BF16 -> routed W13 -> SiTU BF16 -> routed W2 R_r BF16
        -> shared SiTU input BF16 -> shared down S_r BF16
    -> producer phase gate for both independent lane segments
    -> system-visible publish of concat(S_r, R_r)
    -> one joint TP all-reduce, FP32 local-first/increasing-peers -> BF16 lane
    -> completion plane: every peer reader finished before producer-slot reuse
    -> routed RMSNorm -> routed up -> add prefix + shared -> output BF16
```

- Local grid gates use monotonically increasing generations and a fully
  resident grid. Every producer performs a VMEM drain before the release.
- Remote protocol is store -> system release -> remote system acquire -> load.
- J0, the first choice, publishes a whole ready lane after the producer phase
  gate and performs one-shot pull reduction. Each destination initializes from
  local, then loads peer group ranks 0--7 excluding local. Communication
  workgroups grid-stride 512-element tiles.
- J1, for B8/B16 only, assigns each lane tile to one rank, reduces at the owner
  using owner-local first then increasing peers, then scatters that
  BF16 result. It cuts duplicated peer reads but changes the destination-specific
  J0 recurrence to the owner's recurrence, adds a scatter handoff, and therefore
  needs both numerical-tolerance acceptance and an end-to-end win. An exact J1
  would compute eight differently ordered sums and forfeits its traffic benefit.
- Incremental tile events are allowed only when a ready tile enables a different
  workgroup to perform useful downstream work. Per-route events followed by a
  fixed all-route suffix are forbidden: the EP event oracle added
  `226*(L-1)` waits and changed no makespan.
- B1 therefore begins with phase gates. B8/B16 bucket events are legal only for
  independently stealable expert/output tasks.

## B1 schedule: direct, output-owned, exact K

- Persistent grid: candidate `P in {224,240,248}`, eight subgroups per
  workgroup, all workgroups proven resident before any grid gate.
- Packed H-input projection: `6016 = 896 + 3584 + 1536` output rows,
  376 logical N16 tiles. It produces router FP32, routed latent BF16, and shared
  activated BF16 with their exact boundaries.
- Top-k: one workgroup; no EP logical-to-physical dispatch map.
- Routed W13: 16 routes * 24 logical N16 gate/up tiles = 384 tasks.
  A task computes 16 local intermediate channels and both gate/up rows.
- W13 K loop: four logical tiles `[1024,1024,1024,512]` in a two-buffer software
  pipeline. All eight subgroups compute; they cooperatively stage `z`, preload
  their owned packed weights/scales for stage `t+1`, and compute stage `t`.
  Approximate activation+W13-weight+scale storage is 19,456 bytes/stage and
  38,912 bytes double-buffered.
- Routed W2: 224 N16 output tasks. A workgroup owns 16 latent output columns and
  loops all 16 routes in original slot order, so no `[16,L]` route-output
  materialization is needed.
- W2 cohort experiment: subgroups 0--3 own output N0--7 and subgroups 4--7 own
  N8--15. Both cohorts reuse one staged 384-element route activation. Exact K384
  is `[256,128]`, double buffered. Staging activation, both cohorts' weights,
  and scales costs at most 5,376 bytes; activation-only staging costs 1,024.
- Shared down: 224 N32 tasks. Final routed-up/add: 224 N32 tasks. Joint lane:
  21 tiles of 512 BF16 elements.
- Start with branch overlap enabled inside the persistent task pool: shared work
  and routed-ready work may interleave. Disable it only for a reproduced numeric
  or scheduling failure, not as a precaution.

## B8 and B16 schedules

### D: direct token-slot lane

- Exact routed logical tasks are:

| B | route slots | W13 N16 | W2 token/N16 | joint 512-element tiles |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 128 | 3,072 | 1,792 | 168 |
| 16 | 256 | 6,144 | 3,584 | 336 |

- Use M8/M16 MFMA tiles for replicated dense input/shared/final projections;
  do not repeat B1 GEMV across tokens. Routed W13 remains token-slot tasks, and
  W2 remains token/output-owned because expert weights differ by route.
- D is the first B8/B16 implementation because it needs no sorting and preserves
  the B1 numerical/dataflow contract.

### G: stable expert-bucket lane

- Build a count/histogram over at most 256 route slots and 896 experts, stable in
  `(expert id, token id, original slot)` order.
- Execute tiny ragged expert batches with M1/M2/M4/... tiles. Only real repeated
  experts create weight reuse; use the measured route histogram.
- W2 may materialize `[B*16,L]` BF16 route outputs and then combine in original
  slot order. Extra write+read traffic is 1.75 MiB at B8 and 3.50 MiB at B16.
- Under a uniform independent-token model, expected unique experts are 120.279
  at B8 and 224.412 at B16: duplicate-slot fractions are only 6.03% and 12.34%.
  G must overcome sorting, ragged tails, route-output traffic, and imbalance.
- G is forbidden at B1: 16 distinct selected experts give no within-token expert
  reuse. B1 expert-XCD affinity is therefore not promoted by Fleet analogy.

## XCD mapping experiments

- A0: neutral global grid-stride queue. This is the baseline.
- A1: W2/shared/final output-column affinity, with per-XCD static ranges and a
  local ticket. This can preserve intermediate/output ownership without assuming
  expert reuse.
- A2: expert-id affinity for G at B8/B16 only, with hierarchical local queues
  plus global stealing for tails.
- A3: W13-to-W2 handoff affinity, admitted only if readiness creates independent
  W2 work rather than a fixed suffix.
- Read the physical XCC/XCD ID. Never infer XCD from `program_id`.
- Promote an affinity only when the same input/routes show both a lower critical
  path and balanced XCD/CU activity. Cache-hit anecdotes are insufficient.

## Workspace and proposed API

- Preparation-time plan owns one rank's sole expert representation, graph-stable
  buffers, symmetric Iris addresses, group rank order, task formulas, and the
  selected `{D|G} x {J0|J1} x {A0|A1|A2|A3}` mode.
- Proposed preparation surface:

```text
prepare_kimi_k3_tp8_megamoe(
    weights, tp_group, iris_context, *, max_batch=16
) -> KimiK3TP8MegaMoEPlan
```

- Proposed execution surface:

```text
kimi_k3_tp8_megamoe_decode(
    hidden_states[B,7168], prefix_sum[B,7168], plan,
    *, active_batch: Literal[1,8,16], epoch: int, out[B,7168]
) -> out
```

- Admission is rank-uniform and validates TP8/EP1, exact active batch (not graph
  padded M), all shapes/dtypes/strides, rank ordering, symmetric heap bases,
  generation state, grid residency, and one-storage expert ownership.
- Major B16 payloads per rank, excluding flags/counters/fatal/topology metadata,
  include: logits 57,344 bytes; routed latent
  114,688; shared activation 24,576; routed activation 196,608; and producer and
  reduced joint lanes 344,064 bytes each per ring slot. Two ring slots for both
  lanes total 1,376,256 bytes. G adds 1,835,008 bytes of route-output storage.
- J0 has two distinct INT64 arrays: ready `[ring_slot=2,comm_programs,rank=8]`
  and completion with the same shape. The leading two is payload-ring parity,
  not a ready/completion plane. Per array they cost 2,688/21,504/28,672 bytes;
  together 5,376/43,008/57,344 at B1/B8/B16. Top-k IDs and weights each cost
  64/512/1,024 bytes; fatal epoch is INT64[1]. Communication programs are
  21/168/224 at B1/B8/B16 for P=224. J1 uses per-tile ready and
  scatter-completion generations. The removed pre-refresh fixed 21-program B1
  storage was an ABI capacity limit, not an algorithmic B8/B16 limit.

## Serving-correctness acceptance

- Every batch must positively prove that the intended TP8 MegaMoE HSACO was
  loaded and that its dispatch executed; a fallback-path result cannot qualify
  even when numerically correct.
- B1 uses the paired 4096-token prompt/1024-token completion workload and
  requires exact equality of both the prompt SHA-256 and normalized completion
  SHA-256 between baseline and candidate, with one successful request.
- B8/B16 use identical prompt multisets between arms. Exactly 8/16 requests must
  succeed, and every request must consume 4096 prompt tokens and produce 1024
  completion tokens. Retain every prompt and normalized-completion hash, but do
  not require per-request or cross-arm completion-hash equality: scheduling can
  change batched trajectories.
- B8 and B16 therefore additionally require a separate, deterministic,
  bitwise M-specific layer oracle at each BF16 materialization boundary. The
  serving count/hash ledger and this oracle are both mandatory; neither
  substitutes for the other.
- A correctness and end-to-end win at one target batch admits only that exact
  batch specialization. B1 evidence cannot enable B8/B16, and vice versa.

## Traffic and compute model

- B1 modeled linear-weight traffic is 183,730,176 bytes = 175.219 MiB:
  141.750 MiB fixed dense/shared weights plus 16 * 2.092 MiB selected expert
  shards. This deliberately excludes the 7,168-byte routed-norm vector and
  3,584-byte correction bias. The linear-weight HBM-only lower bound is 22.97 us
  at 8 TB/s or 34.67 us at an effective 5.3 TB/s; neither includes those small
  parameters, activations, latency, compute, synchronization, or communication.
- Rank-local routed-expert storage is 1,965,293,568 bytes = 1,874.25 MiB per
  layer for either TP8 (896 one-eighth shards) or EP8 (112 full experts), before
  allocator/layout overhead. A topology-only memory comparison must start here.
- Per token/rank arithmetic is 280,756,224 linear FLOPs: router 12,845,056;
  routed down/up 51,380,224 each; shared W13/W2 22,020,096/11,010,048; selected
  routed W13/W2 88,080,384/44,040,192.
- Joint BF16 lane sizes are 21,504/172,032/344,064 bytes for B1/B8/B16. J0
  remote reads per rank are seven times those sizes. These bytes are small
  compared with weights; latency, synchronization, and rank arrival skew matter.
- Ideal unique-weight traffic under the uniform expert model is 175.219 MiB at
  B1, 393.350 MiB at B8, and 611.175 MiB at B16, versus naive no-reuse totals of
  175.219, 1,401.750, and 2,803.500 MiB.

## Concrete CU/SIMD evidence gate

Resource metadata is admission evidence only. VGPR, SGPR, LDS, scratch, and
nominal occupancy cannot promote a candidate.

### Predicted hot-loop behavior

- The historical pre-refresh ordinary B1 object is a vector-GEMV regime: zero
  MFMA, 13,653
  branch-expanded static instruction lines, 1,347 static `waitcnt` lines, and
  332 static barriers. It has no `s_clause`; a conservative split at waits,
  barriers, stores, and atomics places 527 global loads in 514 groups, of which
  506 contain one load (maximum four). Rank-0 ELF `.text` is 80,384 bytes and
  the HSACO is 135,840 bytes. These are branch-expanded/code-shape counts, not
  dynamic events. B1 R0 therefore defaults to VALU FMA and must prove longer
  independent VMEM issue plus VMEM/VALU or inter-wave latency hiding; MFMA is
  not a promotion requirement. Front-end/I-cache pressure is first-class.
- K384 alone does not justify MFMA. In D, every route has different expert
  weights, so B8/B16 routed W13/W2 also stay M-fast vector GEMV. Replicated
  dense projections use M8/M16 MFMA. In G, use vector tiles for real expert
  buckets below M8 and consider MFMA only for measured buckets at least M8.

| candidate / batch | subgroup roles and expected issue mix | predicted cadence and risk |
| --- | --- | --- |
| R0 routed-core, B1 | W13: all 8 subgroups compute two logical channels each; cooperative `z` loads. W2: two 4-subgroup N8 cohorts share activation. VMEM weight/scale loads should overlap current VALU-FMA issue; LDS feeds activation. | W13 four K tiles: prologue plus three buffer transitions. W2 each route has K256/K128 and a route transition. Main risks: `vmcnt` immediately after load, LDS barrier bubbles, register growth, and cohort imbalance. |
| D-J0-A0/A1, B1 | Compute phases require all 8 waves; communication's acquire must keep peer consumption in one subgroup, so 7 waves are intentionally idle for 21 communication workgroups. | Expect about 8 resident waves per active compute unit, approximately 2/SIMD. The whole grid must reside. Hot-code breadth and repeated final RMS read are I-cache/VMEM risks. |
| D-J0-A0/A1, B8/B16 | Dense replicated projections issue MFMA while VMEM prefetches the next K tile. Routed direct tasks remain vector GEMV, use all 8 subgroups, and W2 keeps the two cohorts. | Long queues should hide tails. Events are phase-level. Risk shifts to expert-weight VMEM bandwidth and communication arrival skew, not route-owner skew. |
| G-J1-A2, B8/B16 | Subgroups collaborate on ragged expert M tiles; use MFMA only for measured M8+ buckets, with VALU/tiny-GEMV tails. Owner communication uses one acquire subgroup per tile. | Per-bucket readiness is useful only with stealable consumers. Risks: sparse MFMA lanes, hot-expert XCD imbalance, sorting, scatter handoff, and larger code footprint. |

### Required capture set

For every candidate and each of B1/B8/B16:

1. Freeze commit, checkpoint, quantization, TP8/EP1 mapping, graph/eager mode,
   exact active batch, route IDs, input, clocks, and grid.
2. Use 3--5 uninstrumented warmed runs for output tok/s and TPOT. Never use ATT
   duration as the performance number.
3. Capture a normal GPU/Perfetto timeline on all eight ranks for phase duration,
   overlap, workgroup distribution, rank arrival skew, and communication order.
4. Capture rocprof kernel CSV plus code objects for exact grid/workgroup, LDS,
   scratch, VGPR/SGPR, spills, and dispatch identity.
5. Target steady W13 and W2 dispatches with ATT; target communication separately
   because spin/acquire behavior is profiler-perturbed. On the slowest rank,
   capture early, middle, and tail tasks and at least one workgroup on each of
   the eight XCDs. Fast-XCD-only evidence is rejected. Decode wave timelines and
   statistics, and disassemble the exact code object.

### Promotion/rejection predicates

- Residency: every persistent workgroup must be resident before a grid gate.
  In a one-workgroup-per-CU plan, ATT must show eight resident waves on sampled
  active compute units and approximately two per SIMD; normal traces must show
  the intended workgroup count on all eight XCDs. Any missing resident workgroup
  or barrier deadlock rejects the grid.
- Wave activity: in W13/W2 steady intervals, at least six of eight waves must be
  active in at least 80% of sampled non-tail intervals. A lower value is allowed
  only when the candidate is measurably faster and ATT identifies intentional
  specialization rather than dependency starvation.
- Pipeline: in at least 80% of non-tail K transitions, decoded issue history must
  place independent VALU work (B1/direct routed) or MFMA work (B8/B16 dense and
  selected M8+ bucket variants) between the next-tile VMEM issue and its
  consuming wait. Dependency-stall samples must fall at least 10% against the
  one-stage control. Otherwise the two-stage pipeline is rejected.
- Dense B8/B16: MFMA and VMEM issue must coexist in the steady tile window. A
  VALU-only dense path or empty MFMA slots rejects that tiling. B1 routed GEMV
  may be VALU-FMA dominated; zero MFMA is not by itself a failure there.
- Hazards: disassembly must show the required `vmcnt`/`lgkmcnt` waits before
  consumers, legal clause formation, and safe barrier spacing. Any missing
  wait/fence is a correctness rejection. A wait immediately following its load
  in most K stages rejects the claimed overlap.
- Barriers: ATT barrier/idle samples may not exceed 10% of a hot W13/W2 phase;
  a pipeline that converts saved dependency stalls into equal or larger barrier
  stalls is rejected.
- Issue utilization: require a statistically stable improvement in active issue
  fraction and at least 3% lower median hot-phase duration versus the exact
  one-stage/single-cohort control. Full-kernel and end-to-end time must not
  regress outside run-to-run noise.
- Code footprint: record `.text` bytes and hot-loop instruction span. More than
  15% growth requires ATT to show no new fetch/idle signature; any correlated
  fetch stalls or phase slowdown rejects the expanded specialization. Compare
  against the ordinary 80,384-byte rank-0 `.text`, not only source line count.
- XCD balance: task and active-cycle spread across XCDs must satisfy
  `max/mean - min/mean <= 10%` for D, excluding a documented final partial tile.
  A1/A2 must improve its target phase by at least 3% on identical routes and may
  not worsen full-kernel median by more than 1%; otherwise retain A0.
- Communication: separate rank-arrival skew from peer-load time. J1 must lower
  full joint-reduction critical path, not merely peer bytes. Per-event waits with
  no independently issued consumer work reject the event scheme.
- Final promotion requires numerical gates, an uninstrumented end-to-end win at
  any target batch, the all-rank normal timeline, and the CU/SIMD/ISA evidence
  above. No single proxy can substitute for this set.

## Prior TP lane: what remains valid

Valid conclusions:

- Loader-level TP ownership, rank-local shapes, `I_r=384`, and all 896 expert
  shards per rank are exact.
- TP removes EP route-owner rank skew. Historical B8/B16 scaling supports the
  topology, while proving nothing about this schedule.
- Logical K384 is exact; K512 padding adds 33.3% W2 K work and is avoidable.
- Shared and routed row-parallel partials may share one concatenated collective.
- Ideal expert element count per rank equals EP8. A second full model copy is not
  inherent to TP8.
- Historical B1 TP8 lost with two reductions; removing that specific overhead is
  a justified experiment, not a promised B1 win.

Invalid or premature conclusions:

- A Gluon AST/layout failure or the old `intermediate % 256` guard disproves TP8.
- Compile success, nominal occupancy, or resource metadata predicts throughput.
- K512 is required, Fleet-style expert XCD affinity helps B1, or EP8 route events
  should be retained.
- The prototype's roughly 21 GiB/rank memory delta is architectural. It must be
  audited for raw+preshuffled or framework duplication first.
- TP8 beats EP8 at B1. Existing real-checkpoint evidence says the opposite; this
  clean joint-reduction design has not run.

## Smallest independent discriminator after top-of-main refresh

1. The selective refresh onto `e7842295` and loader/API audit are complete.
   Collect fresh EP8 B1/B8/B16 output tok/s and the same-checkout portable TP8
   per-rank compute timings. Do not label TP8 output tok/s a baseline until a
   corrected routed-reduction orchestration is present and qualified.
2. Implement only a TP8 routed-core persistent probe: inputs `z[B,3584]` and
   precomputed top-k; all-896 rank-local TP shards; output rank partial
   `[B,3584]`. Do not copy any EP8 task/state-machine code.
3. First specialization is B1 R0: 384 W13 logical N16 tasks, one grid gate, then
   224 output-owned W2 N16 tasks with exact K384 and two four-subgroup cohorts.
   Compare one-stage/single-cohort controls and `P={224,240,248}`. Also time a
   bounded two-kernel W13 -> W2 control with identical math/tasks. If it wins,
   that attributes the fused loss to persistent control/front-end/code footprint
   rather than TP8 topology and justifies considering the bounded split.
4. Match the eager TP decomposition at every BF16 boundary and reject NaN/Inf.
   Inspect TTIR/TTGIR/LLVM/ISA, run uninstrumented timing, all-rank normal trace,
   then targeted W13/W2 ATT through the evidence gate above.
5. Advance only if routed-core time beats the same-checkout portable per-rank
   TP component at at least one of B1/B8/B16 with no spills or numerical
   regression. Then attach J0 and compare a correctly reduced full lane against
   a corrected non-Mega TP composition. Only that integrated comparison can
   support serving claims. G/J1/XCD affinity are later discriminators, never
   prerequisites.
