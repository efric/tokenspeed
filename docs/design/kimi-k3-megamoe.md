# Kimi K3 gfx950 MegaMoE design

Status: correctness-qualified experiment; active performance tuning; marker false
Target: eight gfx950/CDNA4 GPUs; attention TP8; routed MoE EP8 or TP8/EP1;
decode batches 1, 8, and 16
Experiment switch: `--enable-kimi-k3-megamoe`
Primary metric: matched output tokens/s; a win at batch size 1, 8, or 16 is sufficient

This document is the contract for the experimental Kimi K3 MegaMoE path. It
separates facts verified in the current repository and supplied profiling
artifacts from design choices, and records which correctness, compiler, and
performance gates were accepted or rejected.

Canonical tuning state on 2026-08-14: the accepted correctness protocol is the
restored d5dad per-tile completion protocol. The ordinary-launch production
candidate has kernel source SHA-256
`b0314296919330245d04724afc6e8902c6e2e977936bb101e48e532fb92f5da7`;
its kernel body is unchanged from d5dad and only the host compile metadata is
different.
Each of the 21 communication workgroups publishes its own completion flag,
exact-acquires the corresponding flag from all seven peers, and contributes to
the rank-local communication gate. The experimental implementation marker
remains `False`: the path is not performance-admitted and the existing fused
kernels remain the production path.

## Living tuning ledger -- read this first

This is the concise source of truth for ongoing work. Detailed protocol and
numerical contracts remain in the numbered sections below. Agents must update
this ledger when an experiment changes disposition, and must record its scope,
correctness gate, resource gate, performance result, and retained artifact.
An isolated microbenchmark never overrides an integrated full-body or model
result.

Disposition labels are: **integrated**, **retain**, **rejected**,
**promising/unqualified**, **host-qualified**, **active**, and **deferred**.

### One-minute handoff

- **Success criterion:** beat the fresh same-checkout implementation in matched
  output tokens/s at B1, B8, or B16; report every supported target batch and
  retain its numerical, protocol, fail-stop, and dispatch proof.
- **Measured control:** ordinary-launch P240 plus depth-one scheduler overlap is
  the best qualified B1 MegaMoE control. It is not a clean-sheet architecture
  constraint and is still slower than the fused path inside the MoE body.
- **Keep:** d5dad's direct per-tile Iris protocol and self acquire, broad rather
  than single-XCD expert bandwidth, one joint shared+routed reduction when the
  topology permits it, and exact BF16/FP32 materialization boundaries.
- **Do not repeat unchanged:** cooperative launch/stream forwarding, Iris flag
  batching, global expert-to-XCD affinity, source-only peer preload depths,
  LDS-only occupancy reduction, the broken staged-W2 layout, or route gates on
  the current dual-role ownership graph.
- **Deferred measurement:** the all-rank phase-clock runner has dispatched no
  MegaMoE grid: its executable-identity classifier and then its fresh-cache
  whole-HSACO admission both failed closed. A host-verified final cache plan is
  frozen, but rerun it only after the top-of-main refresh. Phase clocks select a
  representative occurrence; only a normal timeline plus decoded ATT can prove
  instruction-level overlap.
- **Host-qualified redesigns:** independent clean-sheet routed-EP8 and
  routed-TP8/EP1 plans now cover B1/B8/B16. Their first post-refresh probes
  separate W13 producers from output-owned W2 consumers; task rings, new
  workgroup roles, workspace/API versions, and justified bounded splits remain
  available rather than being forced into P240.
- **XCD rule:** affinity is phase-specific. Use topology to balance bandwidth
  and control traffic; never inherit a whole-expert/XCD mapping without a
  phase-local A/B test.
- **Promotion rule:** resources and occupancy are only prerequisites. A serious
  candidate also needs CU/SIMD evidence for live-wave issue overlap, waits,
  barriers, dependencies, instruction fetch, and XCD balance.
- **Integration order:** finish the frozen-base research round and ledger, then
  commit it, fetch/rebase onto top of main, rebuild worktree-bound environments,
  measure fresh B1/B8/B16 controls, and requalify only surviving designs.

### Canonical checkout and execution path

- **Integrated:** the canonical kernel is ordinary P240, eight subgroups,
  128 KiB LDS, 156 VGPRs, 106 SGPRs, 40 SGPR lane-save slots, occupancy one,
  and no scratch/private/VGPR spills. Its source hash is `b0314296...`; its
  admission source hash is `f05e800b...`; it is AST-identical to the
  qualification-time `43874a45...` file after the required formatter pass.
  The 51-tensor ABI and d5dad numerical and memory-order body are unchanged.
  `launch_cooperative_grid=False` is the only kernel-source difference from
  d5dad.
- **Integrated:** scheduler overlap depth one is enabled. The fatal-epoch D2H
  mirror is a two-slot ring, so the current asynchronous copy cannot race the
  host's read of the preceding result. Device replays remain serialized on one
  execution stream; this is CPU/GPU enqueue overlap, not concurrent graph
  execution. Runtime construction now requires at least `depth + 1` host slots,
  and the qualification benchmark forces fatal D2H on and rejects the disabled
  warning rather than accepting an unsafe throughput result.
- **Retain:** the implementation marker remains `False` until a matched
  unprofiled flag-on run beats the current arm at B1, B8, or B16 and that
  batch's correctness/fail-stop gates pass. A win at any one of the three
  target batch sizes is sufficient to validate the megakernel approach; report
  all three whenever the candidate supports them.
- **Design rule:** ordinary P240 is the current measured control, not a
  mandatory template. Clean-sheet EP8 and routed-TP8/EP1 prototypes may replace
  its workgroup numbering, static phase barriers, workspace/API, XCD mapping,
  task ownership, and even single-kernel phase organization. Preserve the
  model equations, qualified BF16/FP32 boundaries, required deterministic
  accumulation order, communication happens-before edges, and fail-stop
  contract; everything else must earn its place experimentally. Target the
  explicit event/task structure and implementation quality of
  Mixture-of-Kittens rather than forcing new mechanisms into P240.
- **Verified call path:**

  ```text
  KimiLinearMoE.forward (python/tokenspeed/runtime/models/kimi_k3.py:1684)
      -> kimi_k3_megamoe_decode (tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/megamoe/api.py:165)
          -> launch_gfx950 (tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/megamoe/gluon.py:353)
              -> launch_prepared_kimi_k3_megamoe_gfx950 (tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/megamoe/kernel.py:2571)
                  -> _kimi_k3_megamoe_kernel (:1879)
                      -> _expert_phases (:1243)
                      -> _iris_communication_tile (:1547)
  ```

### Performance scoreboard

- **External 4k/1k topology controls:** issue
  [#55](https://github.com/raikonenfnu/tokenspeed/issues/55) reports the
  real-checkpoint TP8/EP8 path at **58.98 / 207.05 / 299.45 output tok/s** for
  B1/B8/B16. Issue
  [#56](https://github.com/raikonenfnu/tokenspeed/issues/56) reports its
  real-checkpoint routed-TP8/EP1 prototype at **51.17 / 235.80 / 374.55
  output tok/s**. Thus routed TP loses 13.2% at B1 but wins 13.9% at B8 and
  25.1% at B16. These are topology/mechanism controls on `fb00a5be`, not the
  post-research top-of-main acceptance baseline; refresh all three after the
  current research round.
- **Retained 04bc control:** the older matched checkout measured
  **58.0592 / 193.2465 / 298.7008 output tok/s** at B1/B8/B16. It is useful for
  detecting gross drift only; it must not replace the fresh same-checkout
  flag-off comparator.
- **Reference:** historical fused batch-one throughput is **58.0592 tok/s**
  with 16.72 ms TPOT. In the matched profiler environment, fused warm TPOT is
  18.86 ms and a replay spans 17.415 ms: fused MoE 7.932 ms, non-MoE kernels
  9.184 ms, and intra-replay idle 0.300 ms.
- **Rejected historical candidate:** cooperative d5dad measured 14.1222 tok/s
  and 70.35 ms TPOT in the original paired 4096-to-1024 benchmark. Its normal
  trace spans 73.071 ms: MegaMoE 22.360 ms, non-MoE 37.803 ms, and idle
  13.192 ms.
- **Current best, not yet admitted:** ordinary P240 plus overlap produced the
  same 4096-to-128 text, no fatal event, and 25.31 ms unprofiled TPOT. Across
  64 complete rank-replays its span is 25.277 ms: MegaMoE 15.644 ms, non-MoE
  9.266 ms, and idle 0.377 ms. All 5,888 MegaMoE nodes execute on stream 1.
- **Current shape limit:** the admitted MegaMoE raw ABI is batch one only;
  B8/B16 are baseline and redesign targets, not supported MegaMoE measurements.
  A batch winner needs a distinct compile-time schedule and any required API or
  workspace versioning before it can enter the matched serving benchmark.
- **Current critical gap:** ordinary launch removed the cooperative side-stream
  and whole-model interference. Of the remaining 7.862 ms/replay gap versus
  fused, 7.712 ms (98.1%) is now inside the MegaMoE body. The next acceptance
  B1 run must still use the full matched 4096-to-1024 benchmark; the shorter
  trace is diagnostic evidence, not a throughput win. Separate fresh
  top-of-main output-throughput controls are required for B8 and B16.
- **Static CU/SIMD baseline, not a dynamic conclusion:** the admitted rank-zero
  ordinary body has an 80,384-byte ELF `.text` section and 13,653 instruction
  lines, including 1,347 `s_waitcnt`, 332 `s_barrier`, 604 `s_nop`, 463
  `global_load_dwordx2`, and 78 packed buffer loads; it emits no MFMA
  instruction because this B1 path is a vector GEMV.
  It also emits no `s_clause`: under a conservative static partition at waits,
  barriers, stores, and atomics, 527 global-load instructions form 514 issue
  groups, and 506 of those groups contain only one load (maximum four).
  Those counts include mutually exclusive rank/role and timeout branches, so
  they do not measure executed issue mix. Use phase clocks only to select the
  dominant occurrence, then use a timeline plus decoded ATT to determine
  whether the live waves actually interleave VMEM, LDS, and VALU work or stall
  on waits, barriers, dependency chains, or instruction fetch.
- **Existing fused ATT control, perturbed but dynamic:** decoded `code.json`
  reports aggregate stall/latency ratios of 81.21% for the exact-B1 packed
  input projection, 81.44% for exact-B1 W2/shared, and 74.23% for exact-B1
  final RMS/up. Their largest sampled stalls are VMEM waits at
  `latent_input_decode.py:116,134`, `situ_decode.py:421`, and
  `rmsnorm_linear_add.py:74`. The available W13 wave capture is M16 rather than
  exact B1 and reports 62.25%, so it is ISA/mechanism evidence only. ATT
  serialization perturbs duration—especially Iris—and these ratios are not
  serving-time percentages; use them to define the load/wait/interleaving
  control that a candidate must beat.
- **Target calibration:** the fused graph's W2-plus-combine leaf has median
  9.919 us, not the roughly 24.5 us standalone event-envelope number. Use the
  graph value when prioritizing W2 redesigns.
- **Primary artifact:**
  `profile_default/kimi-k3-megamoe-ordinary-overlap-on-valid-sdk-stack0-20260814T203803Z/analysis.md`.

### Integrated wins and retained protocol decisions

- **Integrated -- ordinary launch:** ordinary and cooperative d5dad compile to
  the same AMDGCN/HSACO and resources apart from launch metadata. Eager and
  captured rank-zero parity, all-rank loaded-code admission, balanced EP8 with
  200 replays, rank-zero-concentrated L16, and missing-rank fail-stop all pass.
  Loaded capacity is 256 workgroups for P240.
- **Integrated -- scheduler overlap:** healthy outputs are deterministic across
  repeated 4096-to-128 requests. A sticky device fatal makes the one already
  enqueued successor self-abort, while the two-slot host mirror preserves
  fail-stop observation.
- **Retain -- direct Iris protocol:** keep 21 per-stripe ready publications,
  21 one-subgroup reducers, per-tile completion, exact peer acquisition, and
  self-acquire. This is currently faster than every tested flag-batching
  hierarchy despite executing more system atomics.
- **Deprioritize -- textbook tree all-reduce:** the current world-eight lane
  performs one-hop peer loads in the required rank order. A tree adds dependent
  hops, changes the numerical recurrence, and does not remove payload bytes;
  the tested hierarchical flag schemes already lost. Reconsider hierarchy only
  as a payload-partitioned pipeline with an explicit critical-path model, not as
  a mechanical replacement for seven direct loads.
- **Retain -- current XCD ownership:** use XCD topology for deterministic local
  rank discovery and balanced work, but do not bind a whole routed expert to
  one XCD. Affinity is a phase-specific hypothesis, not a global invariant.
- **Retain -- eight subgroups for the full body:** four-subgroup and staged
  variants remain useful resource probes, but none has produced an integrated
  win. Resource ceilings are admission observations, not tuning limits; larger
  LDS/register use is acceptable when it wins without spills or loss of
  required residency.

### Completed experiments -- do not repeat without a new discriminator

- **Rejected -- cooperative same-stream forwarding:** forwarding the captured
  stream handle still placed all MegaMoE nodes on capture stream 5. Per-layer
  before/after handoffs remained about 12.2 ms/replay. Ordinary launch, not the
  wrapper forwarding change, fixed placement.
- **Rejected -- aggregate completion:** the isolated publication litmus passed,
  but EP8 latency regressed 0.163426 to 0.170793 ms (+4.51%). The source is
  retained only as negative evidence.
- **Rejected -- Iris flag batching:** world-eight rank-max p50/p95 was
  60.5815/65.4420 us for direct357 system operations, 61.9220/67.3620 us for
  compact199, and 62.6220/67.6810 us for preferred21-with-follower199. Fewer
  flags did not compensate for extra local control. Removing the self acquire
  also lost about 1.10% p50 on the physical-0/4 pair.
- **Rejected for the current B1 control -- inherited whole-expert/XCD
  affinity:** isolated W13 favored
  wider XCD fanout (fanout 1/2/4/8 cold p50 76.082/56.661/44.201/37.301 us).
  A distributed logical role improved an isolated W13 lane by 2.61% cold and
  was neutral in isolated W2, but the integrated A/B/D test reversed it. At L2,
  current A was 126.964 us, distributed B was +0.503%, and W13-only D was
  +2.111%; at L16 B was +0.884% and D +2.211%. That full-body result is rank
  zero with locally aliased Iris, not EP8 communication evidence. Keep A for
  the current B1 body. B8/B16 expert buckets, new cohort layouts, and real
  world-eight phase-local XCD policies remain independent experiments.
- **Rejected -- staged two-cohort W2 prototype:** its first LDS staging layout
  was numerically wrong (3,575/3,584 BF16 elements differed). It is a test bug,
  not evidence against correct staging, but that implementation must not be
  timed or promoted.
- **Rejected -- unchanged P226 subgroup/cache variants:** `.ca` was about 22%
  slower than `.cv` on rotating cold weights; the explicit two-group K-loop
  pipeline was about 54.5% slower; four subgroups were about 24.8% slower.
  Their higher reported occupancy did not offset added barriers, waits, and
  reduced outstanding cold-load overlap.
- **Promising/unqualified -- larger W2 grids:** a broad sweep overstated its
  gain. The narrow retained comparison found P256/W242 only 0.93% better cold
  p50 than W226 but 9.42% better p95. This supports a full-body phase-balance
  test, not a claimed median win.
- **Promising/unqualified -- no-timeout polling:** rank-zero generated AMDGCN
  shrank 35.2%, timers 46 to 4, exchanges 192 to 24, barriers 332 to 115, and
  an immediate-ready paired probe improved about 1.72%. It changes a protocol
  failure from bounded poison to watchdog-recovered hang; no world-eight stale,
  skew, reuse, or model qualification exists.
- **Promising/unqualified -- runtime peer loops:** ready/completion loops and
  rank-ascending payload recurrence are semantically preserved. Rank-zero code
  shrank 20.4% with unchanged 156 VGPR/106 SGPR and no scratch, but LLVM partly
  unrolled/tail-merged the loops and no dispatch/timing/world-eight result
  exists.
- **Rejected as primary explanation -- cold weights:** rotating 92 exact layer
  pointers added 1.1995 us to phase zero and 1.2400 us to final; their sum is
  only 2.98% of the former approximately 82 us isolated gap. A fixed GEMV after
  either phase slowed only 0.26/0.18 us. Coldness exists but is not the main
  current-body explanation.
- **Retain as profiler procedure:** a valid ROCm GPU trace in this environment
  requires the reference-compatible `libtorch_cpu.so` plus system HSA runtime.
  HSA-only and registrar-only attempts produced CPU-only traces. Stack-enabled
  fused export also inflated files and wall time; use stack-disabled capture
  after a one-operation GPU/runtime category gate.
- **Retain -- fatal D2H check:** the healthy 8-byte copy has about 6.36 us host
  API median and is already queued without an inter-replay gap. It is not the
  remaining body bottleneck; the profiling-only D2H-off arm has not been run
  and should not displace phase work.

### Active experiments, in priority order

- **Host-qualified clean-sheet design -- routed MoE EP8:** ordinary P240 is
  only the measured control. The frozen design splits K0 projection/schedule,
  a resident event-driven K1 routed/shared core plus joint reduction, and K2
  final projection. Its first K1 discriminator uses separate W13 producers and
  W2 consumers in a measured P256 placement; B1 is output-centric vector GEMV,
  while B8/B16 compare exact route-direct/vector work with stable exact-ragged
  buckets and MFMA only where measured expert-row reuse exists. One symmetric
  payload has distinct monotonic ready and completion planes. Every subgroup
  issuing peer VMEM acquires for itself; the only alternative is one acquiring
  leader loading fresh/versioned LDS followed by a workgroup barrier. Twelve
  host oracle cases pass, including exact route work and B8/B16 nondeterministic
  serving policy. No kernel has compiled or run. Frozen commits are `8317e6f1`
  plus corrective `8129173a`; final design/oracle hashes are `f7088beb...` and
  `d5a702f7...`. First post-refresh implementation is a standalone B8 local K1
  core against the existing route-direct control and a fully bounded control;
  it must pass normal timeline and targeted CU/SIMD ATT gates before Iris or
  production integration.

- **Deferred until the post-refresh control -- phase critical path:** the first
  bounded all-rank runner compiled production, compile-time-false control, and
  clock arms on all eight ranks, then stopped before its first MegaMoE dispatch
  because the old normalizer retained DWARF and kernel/file symbol identity.
  Artifact review proves all eight production/control pairs have identical ELF
  `.text`, instruction lines, CFG, kernarg layout, and resources. A corrected
  run with a fresh worktree cache again stopped before dispatch because absolute
  source/debug identity changed the whole HSACO and admission correctly rejected
  it. The final host-only plan copies the exact qualified 57-file ordinary cache
  into a disposable cache, pins all eight admitted objects, rejects a production
  cache miss before compiling diagnostics, and bounds the run to 48 generations
  per rank. Its verifier passes; no timestamp, perturbation, or ATT result exists.
  Revalidate that plan after rebasing rather than carrying stale code objects.
- **Rejected -- Iris subgroup-zero payload preload:** depths 2/4/8 preserved
  direct357 correctness and ascending-rank FP32 order, but all compiled to 16
  packed loads with a maximum of one load between waits. Depth1 hot/cold p50
  was 59.762/61.101 us. Depth2 was -1.87%/+2.82%, depth4 -1.67%/+0.07%, and
  depth8 -2.41%/-0.62%; no arm won at least 1% in both regimes. Do not promote
  or run world eight. Mixture-of-Kittens' address-precompute/issue/consume
  pattern needs a lower-level compiler schedule or cross-subgroup staging to
  transfer; source restructuring alone did not create independent VMEM issue.
- **Active -- two-stage compute K loops:** test native CDNA4 asynchronous-copy
  ping-pong through the qualified Gluon `cdna4.async_copy` surface, paired with
  the general gfx950 `warp_pipeline_stage`, while keeping exact reduction order.
  Retain a source-level two-stage loop as the control. Local Triton owns the
  frontend/lowering contract, Mixture-of-Kittens supplies the loader/consumer
  lifetime target, and Iris GPT-OSS supplies an existing staged-GEMV control.
- **Rejected compiler shape -- 80 KiB LDS:** the exact 128 KiB/8-subgroup
  control loaded at occupancy one with canonical resources. A single 80 KiB
  Gluon allocation failed before TTGIR/HSACO because LinearLayout requires the
  allocation's out-dimension size to be a power of two. This says nothing about
  loaded occupancy.
- **Rejected -- LDS-only two-workgroup residency:** legal 64 KiB/8-subgroup
  variants still loaded at one workgroup/compute unit. With the explicit
  `waves_per_eu=2` attribute, the compiler reported two waves/SIMD. Omitting
  the attribute reduced VGPRs 156 to 155 and raised that limit to three
  waves/SIMD, but two eight-subgroup workgroups require four waves/SIMD. Both
  arms had zero scratch/private/VGPR spills. Register/lifetime reduction or a
  real four-subgroup remap is required before P480/P512; smaller LDS alone is
  not a sufficient lever. Retained manifest:
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-lds-grid-resource/profile_default/kimi-k3-resource-grid-64kib-followup-20260814T222459Z/followup-manifest.json`.
- **Active -- grid/resource sweep:** smaller full-body grids
  P224/P226/P232/P240/P248/P256 are a barrier-versus-wavefront axis independent
  of two-workgroup residency; do not assume a larger grid is better. Fixed
  logical work remains phase0=240, shared/final=224, expert=226, Iris=21, so
  every grid needs an exact loop/census oracle.
- **Host-qualified clean-sheet design -- routed MoE TP8/EP1:** the frozen
  design artifact is independently derived rather than being an adaptation of
  the EP8 route-owner/state-machine/grid spine. It derives all 896 per-rank
  expert shards and exact local I=384 from
  checkpoint ownership, uses EP8 only as a numerical/throughput comparator,
  and retains no all-to-all, AITER dependency, FP8 intermediate, or physical
  K512 padding. The first routed-core-only R0 discriminator consumes prepared
  `z`/top-k, runs 384 W13 N16 tasks through K `[1024,1024,1024,512]`, one local
  gate, then 224 output-owned W2 N16 tasks through exact K `[256,128]` with two
  four-subgroup cohorts; sweep P224/P240/P248 against one-stage and bounded
  two-kernel controls. J0 later publishes one joint shared+routed lane and
  preserves the qualified destination-local-first, increasing-peer recurrence
  on every rank. J1 owner/scatter changes that arithmetic and is B8/B16-only
  behind numerical tolerance and a critical-path win. Its 273-check host oracle
  passes; no kernel has compiled or run. Frozen signed commit `19e30183`; final
  design/oracle/checker hashes `fa812161...`/`64c3616b...`/`9af3b08f...`.
  The prior P1 probe was an EP-spine adaptation and cannot reject this design.
  Retain from it only topology measurements, exact K384, the 18.71-GiB K512
  padding diagnosis, and the joint-reduction goal. In that old implementation,
  the first N8/K64 stage-2
  probe failed `ConvertTritonAMDGPUToLLVM`. The later exact N32/K64 probe failed
  earlier, during AST-to-TTIR: `scaled_upcast` returned a builder-inferred
  `DistributedLinearLayout`, while the intermediate had an explicit
  `BlockedLayout([4,2], [1,64], [8,1], [1,0])`; Gluon requires identical
  layout objects for the broadcast multiply. No stage-2 candidate emitted
  TTIR/HSACO or dispatched. The four-packed-values-per-thread arithmetic test
  is therefore necessary but not compile-sufficient. After the main refresh,
  test exactly one compile-only repair first: convert the expanded weight to
  the intermediate layout with `assert_trivial=True`. If that is nontrivial,
  reject it and try an inferred activation layout with only reduced-N
  conversion. That is a narrow compiler discriminator for the old prototype,
  not the starting point or acceptance gate for the clean-sheet design. Exact
  candidates use no K-tail mask; padded controls retain it. Implementation
  independence and performance remain **deferred** because this clean-sheet
  design has not yet produced a kernel.
- **Deferred -- per-route W13 gates in the current static ownership:** the
  existing 16-wide `w13_arrival`, `w13_gate`, and `w13_target` planes can express
  exact route gates for `L<=8`, and the dependency graph is deadlock-free when
  each producer publishes before waiting. It still cannot shorten the modeled
  critical path: all 226 W2 consumers are also W13 producers, so the last W13
  producer retains its full static W2 suffix before the all-226 W2 completion.
  The zero-poll makespan delta is exactly zero for every `L=1..8`, while W13
  acquires rise from 226 to `226*L` (an added `226*(L-1)`). Do not build this
  arm alone. Reopen route/chunk events only in a clean-sheet schedule that
  decouples or redistributes W2 ownership through a consumer cohort, task ring,
  or dynamic queue; then compare the event mechanism factorially against the
  ownership change.
- **Researched proposal -- phase-zero shared/router loop fusion:** workgroups
  112..191 currently call `_phase0_shared_pair` and then
  `_phase0_router_pair`, reloading the same hidden-state K tile in two loops.
  A joint loop can retain separate accumulators and exact stores while issuing
  the activation load once. Accept higher registers if phase-zero timing wins
  and loaded occupancy remains sufficient.
- **Researched proposal -- Iris-to-final RMS handoff:** all 224 final
  workgroups currently rescan and reduce the same 3,584-element routed BF16
  vector before projection. Compute the scalar once at the communication
  frontier and publish it with `comm_gate`, then let final workgroups consume
  it. This is eligible only with a portable/qualified multi-producer
  happens-before chain and bitwise parity of the current RMS reduction; a
  relaxed-arrival inference alone is insufficient.
- **Target batches 1/8/16:** batch one remains latency-sensitive; ordinary P240
  is only the current monolithic control, not a required spine. B8 and B16 may
  use distinct compile-time schedules/API workspace.
  Test M-fast/windowed traversal and reuse the same weight tile across token
  rows before adding route affinity. Do not extrapolate B8/B16 wins to B1, but
  a matched output-throughput win at either target batch is independently
  sufficient.

### Cross-source redesign findings

- **Source hierarchy:** use `/home/ericfeng/distributed/triton` as the primary
  source for pure gfx950 Gluon frontend, layout, pipeline, and lowering
  contracts; use Fleet and Iris as AMD megakernel organization precedents; use
  Mixture-of-Kittens as the event/task-lifetime and implementation-quality bar.
  `vroomvroom` remains useful for symmetric-memory correctness and source-shape
  review, but its Triton-shmem transformations are not an architecture template
  for a pure Gluon compute kernel.
- **Existing optimized TokenSpeed MoE kernels are first-class controls:** the
  fused K3 path packs router/routed/shared input projections only when their
  rows share one allocation and 128-row region boundaries
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/latent_input.py:19-57`),
  then calls one expert/shared operation and one joint reduction
  (`python/tokenspeed/runtime/models/kimi_k3.py:1636-1679`;
  `python/tokenspeed/runtime/layers/moe/latent.py:141-181`). On gfx950, the
  registered routed kernel selects route-direct vector W13/W2 for contiguous
  M1..M16 and otherwise the grouped MFMA implementation
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:125-190`).
  Route-direct avoids sorting and per-expert 64-row padding, preserves the BF16
  W2 boundary, and combines original top-k slots directly
  (`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_decode.py:26-36,72-433`);
  grouped MFMA intentionally pads each expert to 64 rows
  (`.../situ_grouped.py:45-48,187-412`). Every redesign must retain these as
  paired controls and justify changes to routing, weight layout, tile shape,
  reduction, and launch separately. Do not call a new megakernel win when it
  merely compares against an untuned generic fallback.
- **Triton gfx950 Gluon -- transfer with qualification:** the general
  `warp_pipeline_stage` test compiles directly for gfx950 and checks the emitted
  `s_setprio` schedule
  (`/home/ericfeng/distributed/triton/third_party/amd/python/test/test_warp_pipeline_gfx9.py:14-17,20-33,59-85`).
  CDNA4 Gluon exposes direct global/buffer-to-LDS async copy, commit/wait groups,
  and relaxed post-wait LDS loads; the runtime test requires `buffer_load...lds`
  or `global_load_lds` plus `vmcnt(0)`
  (`/home/ericfeng/distributed/triton/python/test/gluon/test_core.py:1669-1703`;
  `.../experimental/gluon/language/amd/cdna4/async_copy.py:16-43,75-103,135-179`).
  These are source precedents, not results from the qualified compiler: the
  test files request gfx950 and define runtime checks, but we have not executed
  them in the qualified stack. With the inherited repository `PYTHONPATH` the
  venv resolves this source tree; with that override removed it resolves a
  substantively different packaged Triton
  (`tokenspeed-triton 3.8.10.post20260721`). After the top-of-main refresh,
  build or bind a worktree-local compiler and record imported Python and native
  extension provenance before compiling or timing. Make these the first
  gfx950 K-loop controls, but promote only emitted TTGIR/LLVM/AMDGPU, loaded
  resources, normal timing, and ATT issue overlap. The visually similar
  `_cdna5.py` TDM/WMMA examples are guarded for gfx1250, not gfx950; source
  presence is not target qualification.
- **Triton gfx950 scheduling limits:** the AMD warp-pipeline lowering groups
  four wave64 subgroups, so two simultaneously live groups require an
  eight-subgroup/512-thread workgroup
  (`third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp:1018-1037`).
  It phase-shifts the same subgroups through stages; it does not provide fixed
  loader/consumer specialization. The pipeliner requires at least two stages
  and inserts waits/barriers (`WarpPipeliner.cpp:208-398`). Pair an
  eight-subgroup staged arm with an ordinary-load control and inspect dynamic
  waits, barriers, set-priority cadence, and ATT; lowering success alone is not
  overlap evidence.
- **Triton gfx950 compute/layout limits:** CDNA4 direct-to-LDS shares VMEM
  completion with ordinary operations, so unrelated loads can lengthen its
  waits; prefer coalesced 128-bit access and test buffer versus 64-bit global
  addressing. Scaled MFMA exposes native square 16x16x128 and 32x32x64 shapes
  (`third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp:262-268`), making
  it a strong regular TP8/dense B16 candidate but only a measured
  counterfactual for B8 or sparse expert rows below 16. Layout conversion can
  emit permutations, LDS traffic, barriers, and pressure; require
  TTGIR/LLVM/ISA proof rather than assuming it is free. No complete pure-gfx950
  sparse-MoE persistent precedent was found, so the new core is a composition
  of independently qualified mechanisms rather than a port.
- **Reject CDNA5 mechanisms as gfx950 precedents:** the local `*_cdna5.py`
  examples use gfx1250-only TDM, WMMA-v3, multi-workgroup launch, and true
  `warp_specialize`. Their task/scheduler algebra may transfer only after TDM
  becomes CDNA4 buffer/global-to-LDS, WMMA becomes MFMA-v4, and specialization
  becomes a separately proved gfx950 phase or manual-subgroup protocol. Start
  with ordinary launch; persistent scheduling and XCD remap require measured
  placement/residency and phase-local A/B tests.
- **`comms-notes` -- transfer and reject:** retain payload store -> system
  release -> per-consuming-workgroup acquire, symmetric byte-offset pointer
  translation, and separate local/cross-XCD/cross-GPU visibility domains
  (`/home/ericfeng/distributed/comms-notes/building-megakernels/README.md:314-335`).
  Readiness alone does not make a reusable payload safe: a fast producer can
  overwrite generation N while a peer still reads N, so a completion plane or
  separately proved ring-lifetime protocol is mandatory
  (`/home/ericfeng/distributed/comms-notes/kimi2.5comms.txt:96-124`). Cache
  modifiers and XCD placement remain performance policy, never publication.
  Reject the notes' fixed EP ownership, scheduler constants, and prototype
  residency assumptions when deriving clean TP8 or new EP cohorts.
- **Standalone Iris -- transfer and reject:** its fused matmul/all-reduce path
  publishes an auxiliary tile with a system-release exchange before one-shot
  or owner-scattered two-shot reduction
  (`/home/ericfeng/distributed/iris/iris/ops/matmul_all_reduce.py:125-144`;
  `/home/ericfeng/distributed/iris/iris/mem/triton/context.py:734-840`). Retain
  direct peer addressing, destination-local-first one-shot accumulation, and
  owner/scatter as a separately qualified B8/B16 option. Do not infer a
  reusable persistent protocol: these examples have a readiness flag but no
  distinct completion plane, and owner/scatter changes the qualified
  destination-local arithmetic.
- **CDNA4 architecture -- constrain, do not predict:** one device has eight
  XCDs and up to 256 compute units; pairs of compute units share a 64 KiB
  instruction cache, each compute unit has 160 KiB LDS at 256 bytes/clock and
  a 32 KiB L1, and each XCD has a coherent 4 MiB L2
  (`/home/ericfeng/distributed/amdgpu-isa-manuals/cdna4/whitepaper.md:35-45,77-91`).
  The ISA notes define the physical `XCC_ID` and the vector load/store/atomic
  scope, non-temporal, writeback, and invalidate controls
  (`/home/ericfeng/distributed/amdgpu-isa-manuals/cdna4/README.md:411,1094,2823-2873`).
  Retain explicit physical IDs and scopes for measurement and correctness; do
  not treat cache-control bits as publication or a performance guarantee.
  Peak 8 TB/s HBM and nominal LDS/register capacity are upper bounds, not proof
  of useful residency, load interleaving, or XCD affinity; loaded occupancy,
  normal timelines, and decoded ATT decide.
- **Mixture-of-Kittens -- transfer:** pull-style direct payload reads,
  medium-grained readiness, explicit producer/consumer roles, multiple loader
  stages, deterministic task order, and early ring-slot reuse
  (`/home/ericfeng/distributed/megakernels/mixture-of-kittens/deep-dive.txt:53-57,68-77,168-187,221-253`).
  Its closest Iris analogue is the seven-stage combine: it precomputes peer
  addresses
  (`/home/ericfeng/distributed/megakernels/mixture-of-kittens/csrc/mok_megakernel.cuh:530-543`),
  issues every valid asynchronous load (`:549-570`), then consumes stages in
  order (`:572-589`). Its GEMM path
  separately uses eight 32-lane NVIDIA warps (256 threads, unlike our eight
  64-lane subgroups), a six-stage input ring, and dedicated loader/consumer
  semaphores (`:20-60,1273-1376,1542-1552`).
  **Do not transfer:** Blackwell CLC/TMA/TMEM/clusters, training minibatches, or
  token dispatch/combine that K3 TP8/EP8 does not need at B1.
- **Iris GPT-OSS -- transfer:** phase fusion when the consumer can use a value
  directly from registers, removal of barriers for deferred residuals, and
  pipelined K loops. It measured 180 workgroups faster than 256 because cheaper
  barriers still filled GEMVs
  (`/home/ericfeng/distributed/megakernels/iris/examples/33_gpt_oss_megakernel/gpt_oss_120b_quantized_megakernel.py:53-56`),
  fuses norm into projections and cache append into attention (`:191-231`),
  and uses a staged K loop
  (`.../common/gemv_fp4.py:96-125`). **Probe first:** smaller P and staged K.
  It is a four-subgroup P180 source precedent, not a loaded-resource proof: the
  checkout records neither an explicit LDS reservation nor an admitted
  occupancy artifact.
  **Do not copy blindly:** redundant top-k in every workgroup (`:255-259`) or a
  single relaxed publisher without transitive payload publication.
- **Fleet -- transfer:** CDNA4 small-tile GEMM pipelines, XCD-local rank
  discovery, M-fast/windowed traversal for B>1, and measured split-K controls
  (`/home/ericfeng/distributed/megakernels/fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/gang_linear_mi300.cuh:23-76`).
  **Rejected policy:** one expert at a time per XCD
  (`.../gang_moe_linear_mi300.cuh:16-28`) is slower for our integrated B1
  shapes. Preserve its pipeline/tile ideas without preserving that affinity.
  Fleet uses four CDNA subgroups and roughly 60 KiB LDS, but its named
  `cp_async` path is a synchronous buffer load plus workgroup fences, not a
  CDNA4 asynchronous-copy precedent. Its MI350 schedule also has 240 workers
  plus eight separate scheduler workgroups, so it is not monolithic-P240
  residency evidence.
  Cross-XCD or XCD-local split-K (`.../persistent_kernel.py:1468-1551`) remains
  a later TP8/dense-phase probe because it adds workspace and atomics.

### Evidence index

- Historical paired d5dad rejection:
  `profile_default/kimi-k3-megamoe-final-d5dad-20260814T120409Z/comparison.json`.
- Cooperative stream diagnosis:
  `profile_default/kimi-k3-megamoe-same-stream-valid-sdk-stack0-20260814T193615Z/analysis.md`.
- Ordinary-launch full-model trace:
  `profile_default/kimi-k3-megamoe-ordinary-overlap-on-valid-sdk-stack0-20260814T203803Z/analysis.md`.
- Ordinary rank-zero executable/ISA control:
  `profile_default/kimi-k3-heavy-noncoop-paired-final-20260814TomdZ8F/paired-code/ordinary.{hsaco,amdgcn}`
  (`.text` 80,384 bytes; assembly file SHA-256 `d49bea71...`).
- Fused-path decoded ATT controls:
  `profile_default/kimi-k3-main-04bc0864-graph-20260813T062945Z/att/per-kernel/`
  (exact B1 input/W2/final; W13 wave behavior is M16-only).
- Phase-clock pre-dispatch object-gate failure:
  `profile_default/kimi-k3-megamoe-phase-bounded-20260814T225021Z/manifest.json`
  (zero MegaMoE dispatches; cleanup evidence in the adjacent
  `cleanup-release.json`; executable/debug distinction in
  `control-identity-classification.json`).
- Phase-clock final host-only cache plan:
  `profile_default/kimi-k3-megamoe-phase-corrected-20260814T231532Z/final-retry-cache-plan.json`
  (SHA-256 `a669386f...`; 57-file qualified-cache tree digest `a28a4e8b...`;
  verifier passed; zero MegaMoE dispatches and no phase timing).
- No-timeout paired rank-zero probe:
  `/home/ericfeng/distributed/megamoe-minimal-sync-r0-hF3Jy5KD/paired-report.json`.
- Runtime-loop rank-zero compile report:
  `/home/ericfeng/distributed/megamoe-runtime-peer-loop-r0-P9bfd0uc/report.json`.
- Subgroup/pipeline/grid probes:
  `/home/ericfeng/distributed/pipeline-grid-artifacts/compile-smoke-20260814/`.
- Rotating-layer cold probe:
  `/home/ericfeng/distributed/megamoe-rotating-phase0-final-xJwUUPBE/report.json`.
- World-eight Iris batching probe:
  `/tmp/megamoe-xcd-batch-world8/staged-retry-20260814/`.
- Integrated XCD A/B/D probe:
  `/tmp/megamoe-full-body-abd-physical2-final.9MhwxZ/full-body-a-b-d-rank0-summary.json`
  (SHA-256 `408bec51...`; rank-zero locally aliased Iris, not EP8 communication
  evidence).
- Iris payload-preload depth sweep:
  `/tmp/megamoe-iris-preload-physical2-final.O5sQc7/` (summary SHA-256
  `a229c937bcee961bd999ca7e82f8edc56eb3833ac6ad6247db5717b69a2e28e1`).
- Static per-route W13 event model:
  `/tmp/kimi-k3-route-pipeline-report.json` (report SHA-256
  `33e688db1fc88a402792f70372abcfaa466902a3646ebf3d01d3477ec72027bb`;
  72 host tests passed); generator:
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-route-pipeline/tools/megamoe/kimi_k3_route_pipeline_probe.py`.
- Routed-TP8 first compiler artifact:
  `profile_default/kimi-k3-routed-tp8-p1-20260814T2136Z-I0ket1/`.
- Routed-TP8 exact-N32 layout failure:
  `profile_default/kimi-k3-routed-tp8-p1-gpu2-20260814TXXXXXXZ-yYZfUD/`.
- Clean-sheet routed-EP8 design/oracle (signed commits
  `8317e6f1e132dca27d5c5c95ad1ec48d93385d3b` and
  `8129173ab75ce473a04e495f1a55db14af5f0a80`; 12 host oracle cases; no
  compile or GPU run):
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-ep8-cleansheet/docs/design/kimi-k3-megamoe-ep8-cleansheet.md`
  (`f7088beb1b02cc043ea55de505d5f2b81a87f06210f48b1a3e2bf0aee8581f2b`),
  `tools/megamoe/kimi_k3_ep8_cleansheet_oracle.py`
  (`d5a702f79dd6cb872c03aae5c8ee8799b0b05aa5d9d0f9a0171cd7fab593b722`),
  `test/cli/test_kimi_k3_ep8_cleansheet_oracle.py`
  (`a69e963bc12863c94f0c21d9fdaa884acc3855752fc007702392a6f23b6bb17b`),
  and `tools/megamoe/README.md`
  (`10a52a59265542be2e66f16f93d787a5c793e3ed8727ac676857edcc79c93e30`).
  The generated `/tmp/kimi_k3_ep8_cleansheet_oracle-v2.json`
  (`b9825be6c26ffcf745e1fb1edbf80a445059fdfcde41fff87338b60ceadecd15`)
  is ephemeral, not a repository artifact.
- Clean-sheet routed-TP8/EP1 design/oracle (signed commit
  `19e30183469abc87b6823d25c8c3cdd7b02769f7`; 273 host checks; no compile
  or GPU run):
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-tp8-cleansheet/docs/design/kimi-k3-tp8-cleansheet.md`
  (`fa812161472c878b829966ae05e088def2f74e58bd3d87d0b39658185de0c32f`),
  `docs/design/probes/kimi-k3-tp8-cleansheet-oracle-20260814.json`
  (`64c3616bfdb3d58567d5e6f15c15a4f9254b8fefe71ba3e3dd1f42af77867c04`),
  and `tools/megamoe/verify_kimi_k3_tp8_cleansheet_oracle.py`
  (`9af3b08fe39e548bb001c45ee0d6d37a804d426032a195ed0381f87f6de4d92e`).
- **Integration warning:** EP initial commit `8317e6f1` contains only its four
  intended files. EP corrective commit `8129173a` and TP commit `19e30183`
  each also contain 21 pre-existing files reformatted by the required
  repository-wide pre-commit run. After refreshing main, transplant only the
  intended design/oracle/test/README artifacts; do not cherry-pick either
  formatter-swept commit wholesale.
- Initial EP8-versus-TP8 hypotheses:
  `ep8vstp8.txt` (SHA-256
  `c6e75d78be409f34df2485afca39474e83ff19b5ef2f337e1d70d34283ee8b85`;
  hypothesis input only; all retained shapes, ownership, and numerical claims
  were re-derived from source).
- LDS/grid compile-load matrix:
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-lds-grid-resource/profile_default/kimi-k3-resource-grid-compile-load-20260814T221720Z/matrix-manifest.json`.
- Legal 64 KiB loaded-resource follow-up:
  `/home/ericfeng/distributed/.worktrees/tokenspeed/agent/megamoe-lds-grid-resource/profile_default/kimi-k3-resource-grid-64kib-followup-20260814T222459Z/followup-manifest.json`.

### External TP evidence and memory decision

- **External evidence, not our qualification:** issue
  [#56](https://github.com/raikonenfnu/tokenspeed/issues/56) reports a
  numerically correct routed-TP arm that improves B8/B16 over issue
  [#55](https://github.com/raikonenfnu/tokenspeed/issues/55) but loses B1 by
  13.2% (51.17 versus 58.98 tok/s), consistent with two rendezvous dominating
  one-token decode. Issue
  [#53](https://github.com/raikonenfnu/tokenspeed/issues/53) is shape/mechanism
  evidence, not a production-correct topology comparison.
- **Verified derivation:** padding logical TP W2 K=384 to physical K=512 plus
  scale padding costs 20,089,667,584 bytes (18.71 GiB) per rank across 896
  experts and 92 layers, explaining most of the reported roughly 21 GB memory
  loss. The MegaMoE TP ABI therefore stays linear K384 or uses a tail-capable
  preshuffle; physical K512 is not accepted merely to satisfy a package tile.

### Promotion and anti-loop rules

- Compile and retain every specialization before timing; reject late JIT.
- Require bitwise equality at the defined BF16/FP32 materialization boundaries,
  exact task census, protocol generations, zero fatal/diagnostic state, and
  no scratch/private/VGPR spill before reading performance.
- Serving correctness is batch-specific. B1 requires one successful
  4096-to-1024 request and exact prompt plus normalized-completion equality
  between baseline and candidate. B8/B16 require exactly 8/16 successful
  requests, every request 4096-to-1024, an identical prompt multiset, all
  completion hashes retained but not required equal, and a separate bitwise
  M-specific kernel/model-boundary oracle. In every batch, logs and loaded code
  must positively prove that MegaMoE—not the
  `hidden_states.shape[0] != 1` fused fallback—executed.
- Compare hot and rotating-92 cold pointers; use balanced AB/BA or cyclic arm
  order and report p50/p95 plus rank maximum.
- A source/ISA reduction in atomics, barriers, code size, or registers is not a
  win until the integrated timing wins. Conversely, do not reject larger
  register/LDS use solely for exceeding an earlier observed maximum.
- Resource metadata is necessary, not sufficient. For every serious winner,
  capture a normal all-rank timeline, choose the slowest rank, then decode ATT
  for representative early, middle, and tail tasks plus at least one
  workgroup from every XCD. Inspect active and resident waves, issue-slot
  utilization, VALU and MFMA overlap, VMEM/LDS latency hiding,
  dependency-chain length, memory clauses, `s_waitcnt`/hazard spacing, barrier
  bubbles, instruction-cache effects, and per-XCD/compute-unit balance. Lower
  VGPRs or higher nominal occupancy is a loss when it serializes loads or adds
  scheduler/wait bubbles, as the rejected explicit pipeline already
  demonstrated.
- Qualify in order: host/source oracle -> rank-zero compile/ISA -> bounded
  numerical microprobe -> full-body rank zero -> world eight skew/reuse/fail
  stop -> matched unprofiled output-throughput benchmark at B1, B8, or B16.
- Finish the current frozen-base research round before pulling top-of-main.
  Then rebase/fix every surviving branch and rerun its source, compiler,
  resource, and numerical gates; stale code-object hashes are never carried
  across that refresh.
- The shared qualified venv is editable-wired to the primary checkout, and its
  scheduler import finder outranks `PYTHONPATH`. A refreshed-worktree benchmark
  must use a dedicated cloned venv with all four repository editables rebound
  using `--no-deps`, then prove every Python module and loaded extension comes
  from that worktree before startup.
- Fresh B1/B8/B16 runs use three repetitions in a balanced batch order and
  report median, range, MAD, CV, TTFT, and TPOT. Spread above 6% or CV above 3%
  triggers two more repetitions; a candidate win inside observed arm variance
  must repeat with reversed flag-on/flag-off order.
- Do not rerun a rejected arm unless a new experiment isolates the prior loss
  or changes a single causal mechanism.

## 1. Outcome and boundaries

Sections 1 through 17 below document the current ordinary-P240 control and its
qualification contract. They are not architectural requirements for the
clean-sheet EP8 or TP8 redesign lanes. A new design may use task queues,
descriptor rings, dedicated communication/compute cohorts, a revised
workspace/API, or a bounded kernel split; it must define and qualify its own
equivalent numerical, publication, progress, and fail-stop contracts.

When explicitly enabled for qualification, the experiment replaces the
complete six-launch batch-size-one Kimi K3 MoE sublayer with one static
persistent Gluon kernel per MoE layer. The kernel covers:

1. router, routed-latent, and shared-input projections;
2. sigmoid plus correction-bias top-16 routing;
3. rank-local routed W13, SiTU, W2, and deterministic route combination;
4. the tensor-parallel shared-expert down projection;
5. one Iris producer-direct EP8 sum of the shared and routed partials;
6. routed RMSNorm and latent-up projection; and
7. addition of the shared result and `prefix_sum`.

The schedule is static in the sense relevant to this experiment: every
workgroup executes a compiled phase state machine and derives its tiles from
`program_id`, observed XCD identity, rank-local route bounds, and fixed formulas.
There is no task queue, descriptor interpreter, work stealing, or permanently
reserved scheduler compute unit. Runtime expert IDs and the number of routes
owned by a rank remain data, not a dynamic scheduling algorithm.

The current control's qualified envelope is deliberately narrow: physical tensor row
count `M=1`, gfx950, TP8/EP8 with MoE TP1 and EP8, 896 experts split
contiguously as 112 per rank, top-16, BF16 activations, the native plan's
linear MXFP4 checkpoint layout, and Iris producer-direct communication.
Requests with `M>1` use the current path. An idle graph replay may still carry
one dummy physical row and therefore execute MegaMoE; it must complete the same
collective protocol on every rank. A requested but structurally incompatible
MegaMoE configuration fails during startup rather than allowing ranks to choose
different collective paths.

Non-goals for the first implementation are token-dispatch all-to-all,
cross-node operation, expert-location remapping/EPLB, prefill, speculative
verify shapes, and a kernel that persists across multiple transformer layers.
The unit of persistence is one complete MoE sublayer.

## 2. Baseline and provenance

The matched baseline is the checked-in artifact
`profile_default/kimi-k3-main-04bc0864-graph-20260813T062945Z`:

- TokenSpeed commit: `04bc08649f3e53e19144ee86564be7c6121c99d2`;
- hardware: eight gfx950 GPUs;
- topology: attention TP8 and routed MoE TP1/EP8;
- graph capture sizes: 1, 2, 4, 8, and 16;
- model: `/data/models/moonshotai-Kimi-K3`;
- batch-size-one median output throughput: **58.0592 tokens/s**;
- raw batch-size-one runs: 56.2040, 59.0891, and 58.0592 tokens/s; and
- batch-size-one median TPOT: 16.72 ms.

The exact server and load-generator commands are in the artifact's
`COMMANDS.md`. The historical arm remains untouched. A valid contemporaneous
comparison reuses every server/eval argument and the same qualified ROCr,
compiler, and profiler-interposition environment; between paired arms only the
experimental MegaMoE switch may change. Uninstrumented runs are the throughput
authority. Perfetto, rocprofv3, and ATT are diagnostic and must not be used as
throughput measurements.

Verified normal-trace medians for the six existing MoE launches are
approximately 19.36, 5.20, 12.72, 9.00, 19.36, and 12.40 microseconds. ATT
serializes and greatly inflates the Iris launch, so its communication duration
is not representative. The six normal medians sum to about 78 microseconds per
MoE layer. K3 has 92 MoE layers, making launch removal material at one-token
decode even if arithmetic bandwidth is unchanged.

The current verified fused-control call path is:

```text
KimiLinearDecoderLayer.forward (python/tokenspeed/runtime/models/kimi_k3.py:2096)
    -> KimiLinearMoE.forward (:1684; called at :2234)
        -> _forward_fused_decode_pipeline (:1636; selected at :1714-1716)
            -> latent_moe_input_projections (:1653)
            -> TopK.forward (:1661)
            -> latent_moe_expert_shared_all_reduce (:1662)
            -> LatentMoELayer.finalize_output (:1678)
```

The corresponding GPU sequence is:

```text
_latent_input_decode_kernel
    -> _kimi3_sigmoid_bias_topk_kernel
        -> _stage1_a16w4_situ_warp_gemv
            -> _stage2_a16w4_warp_gemv_combine
                -> iris_reduce_symmetric_gluon_kernel
                    -> _rmsnorm_linear_add_kernel
```

## 3. Why there is no token dispatch

This is a verified property of the target configuration. The preceding
attention path has already replicated the one-token hidden state across the
eight attention ranks. Each rank independently computes the same router choice
and routed latent. Routed expert ownership is complete-expert EP8: rank `r`
owns experts `[112*r, 112*r+111]`. Each rank computes only its local selected
experts and contributes zero for the other selected experts.

The only MoE communication is therefore the final sum of two independent rank
partials:

```text
shared partial [1, 7168]
    + routed partial [1, 3584]
        -> one consecutive Iris lane [shared 7168 | routed 3584]
            -> EP8 SUM
```

Copying Mixture-of-Kittens token packing/dispatch or the GPT-OSS one-attention
rank plus four-expert-rank protocol would add communication that the current K3
topology does not require.

## 4. Numerical contract

The MegaMoE result must match the existing fused pipeline's materialization and
accumulation boundaries, not merely an algebraically equivalent FP32 model.

Verified K3 dimensions are:

| Quantity | Value |
|---|---:|
| model hidden width | 7168 |
| routed latent width | 3584 |
| routed experts | 896 |
| experts per rank | 112 |
| selected experts | 16 |
| expert intermediate width | 3072 |
| shared TP8 intermediate width per rank | 768 |

The selected native EP8 plan validates, but does not preprocess or swizzle, the
expert tensors. The initial kernel consumes `linear_weights=True` with this
exact contiguous ABI:

| Tensor | Dtype and shape | Element strides |
|---|---|---|
| `w13_weight` | UINT8 `[112, 6144, 1792]` | `[11010048, 1792, 1]` |
| `w13_weight_scale` | UINT8 `[112, 6144, 112]` | `[688128, 112, 1]` |
| `w2_weight` | UINT8 `[112, 3584, 1536]` | `[5505024, 1536, 1]` |
| `w2_weight_scale` | UINT8 `[112, 3584, 96]` | `[344064, 96, 1]` |

`w13_input_layout` is `"concatenated"`; gate rows precede up rows. Startup
checks every shape, dtype, stride, and layout flag. The separate gfx950
K-packed/CDNA4-scale-swizzled representation used by other Gluon plans is not
admitted by the first MegaMoE specialization.

The required boundaries are:

- router logits accumulate and remain FP32;
- routed-down and activated shared input materialize as BF16;
- top-k selection is bit-contract equivalent to the current specialized K3
  route: FP32 sigmoid, FP32 correction-bias addition, ordered-FP32 packed keys,
  inverse expert ID as the lower-ID tie break, and the same descending packed-key
  result order. The correctness-first Gluon implementation performs 16 packed
  maximum reductions with removal; a custom bitonic merge is admitted only as a
  separately tested optimization because Gluon has no source-level `topk`;
- mixture weights reload the selected FP32 logits, apply the unbiased FP32
  sigmoid, sum in the same 16-lane reduction order, divide and scale in the
  same order as the current specialized Triton kernel, with observed bit
  identity required for IDs and stored FP32 weights;
- ties choose the lower expert ID;
- shared gate and up projection accumulators each pass through BF16 before their
  FP32 SiTU, matching `latent_input_decode.py:154-161`;
- routed W13 gate and up accumulators each pass through BF16 before FP32 SiTU;
- the activated W13 intermediate materializes as BF16;
- each route's W2 result passes through BF16 before multiplication by its route
  weight;
- rank-local route results combine in original top-k slot order in FP32, then
  materialize as BF16;
- Iris starts with the local BF16 contribution, then accumulates peer ranks in
  ascending group-rank order while skipping local, in FP32, and stores BF16;
- RMSNorm consumes the reduced BF16 routed latent with `eps=1e-5`;
- normalized latent values presented to latent-up are BF16;
- latent-up accumulates in FP32, converts BF16 then back to FP32, adds
  `prefix_sum`, adds the reduced shared result, and finally stores BF16; and
- the final output is BF16.

The restored d5dad implementation fuses W2 with deterministic route combination
in an output-centric expert grid. Each route accumulator crosses the same BF16
boundary before its FP32 route weight is applied, routes add in compact slot
order, and the final BF16 routed stripe is written directly to the symmetric
producer buffer. Atomic accumulation by completion order is forbidden.

The public operation is deliberately narrow:

```text
kimi_k3_megamoe_decode(
    hidden_states: BF16[1, 7168],
    prefix_sum: BF16[1, 7168],
    plan: KimiK3MegaMoELayerPlan,
) -> BF16[1, 7168]
```

Both row inputs are contiguous, colocated, and nonaliasing with the contiguous
BF16 `layer_output[1,7168]`. The immutable host plan is a logical record, not the
raw AMD argument list. It owns the processed layer-weight spec, borrowed Iris
lane, shared local-workspace views, unique `layer_output`, retained admitted
kernel, qualified timeout, and shared stream owner.

The registration shim emits the exact flat 51-tensor raw ABI:

| Tensor indices | Count | Contents |
| --- | ---: | --- |
| `0..2` | 3 | `hidden_states`, `prefix_sum`, `layer_output` |
| `3..13` | 11 | six BF16 dense weights, FP32 correction bias, and four linear MXFP4 expert weight/scale tensors |
| `14..44` | 31 | local nonsymmetric workspace tensors in the order from section 13 |
| `45..50` | 6 | symmetric producer, local reduced output, split Iris flags, fatal epoch, live heap bases, and group-global ranks |

The runner appends dynamic scalar values `beta=4.0`, `linear_beta=25.0`, and
`rms_eps=1e-5`. `EXPERT_START`, group-local `RANK`, and the qualified
`TIMEOUT_NS` are compile-time constants. Normalized top-k, routed scale one,
gfx950, linear concatenated W13, top-k 16, world size eight, 240 workgroups,
eight subgroups, and 128 KiB LDS are algorithm or launch invariants rather than
additional flat arguments. Invocation generation is the coherent value of
`topology_gate + 1`; it is not supplied by the host. The ABI-reserved
`phase_arrival` and `phase_gate` remain zero. No wrapper may silently reorder
the 51 tensors or infer a different specialization from mutable global state.
Every tensor is on the input device, has unit inner stride, and is disjoint from
mutable workspace except for the plan's named output/scratch views.

Projection output reductions retain the current logical four-subgroup cohorts
even though the persistent workgroup has eight subgroups: two independent
four-subgroup tasks share a workgroup. This preserves router-logit reduction
order. Any alternative eight-subgroup reduction is a separately gated
numerical experiment with adversarial top-16/top-17 cutoff tests and full
decode-trajectory validation.

## 5. Prior-art synthesis

The implementation transfers mechanisms only where their contracts match K3.

### Mixture-of-Kittens

Transfer deterministic route layout, arithmetic task indexing, fine-grained
publication, and ring/epoch reuse discipline. Do not transfer Blackwell CLC,
TMA, TMEM, clusters, warpgroup roles, or NVLink-specific dispatch. Its public
megakernel also excludes routing setup and some epilogue work, so it is not the
scope definition for this experiment.

The narrow source-level transfer is its combine pipeline, not its training
topology.
`/home/ericfeng/distributed/megakernels/mixture-of-kittens/csrc/mok_megakernel.cuh:530-589`
computes a batch of peer addresses,
issues seven independent loads, and then consumes the stages in deterministic
order. MegaMoE tests the same issue/consume separation inside the existing
subgroup-zero Iris reducer at depths 1/2/4/8, preserving exact acquires and
rank-ascending arithmetic. MoK's six-deep GEMM producer/consumer ring instead
depends on TMA, clusters, and register-specialized warpgroups; it is motivation
for a CDNA4 asynchronous-copy probe, not source to port literally.

### Iris GPT-OSS megakernel

Transfer the idea of a per-worker compiled phase program and direct symmetric
memory access. Do not copy its single `pid==0` publisher after a relaxed local
barrier. Stores issued by other workgroups are not automatically ordered by a
later system release from `pid==0`. MegaMoE uses an acquire/release RMW handoff
that makes the communication workgroup a transitive publisher, and every
consumer workgroup performs its own acquire.

Two implementation details are directly testable. Its `NUM_WG=180` was chosen
because a smaller resident grid reduced barrier cost while retaining enough
GEMV parallelism, so MegaMoE must sweep smaller grids as well as proposed
larger grids. Its FP4 GEMV uses `tl.range(..., num_stages=NSTAGES)` with a
default depth of three to overlap the next K tile's loads with the current dot.
The analogous MegaMoE experiment uses Gluon's native CDNA4 asynchronous-copy
commit/wait groups and must classify actual waitcnt/barrier changes. Its fused
norm/projection and register-use examples transfer only where the concrete K3
producer and consumer are the same workgroup; they do not justify deleting a
cross-workgroup publication edge.

### Fleet

Transfer observation of `HW_REG_XCC_ID`, construction of XCD-local worker rank
and population, contiguous XCD-local tile ranges, and hierarchical completion.
Do not reserve one scheduler workgroup per XCD. Fleet's scheduler queues are
separable from its locality formulas.

Fleet's `gang_moe_linear_mi300.cuh` assigns one expert at a time to each XCD
and has all local workers cooperate to protect its weights in L2. That policy
is rejected for current K3 B1: wider W13 XCD fanout won monotonically in the
isolated probe, and both distributed-role variants lost in the integrated
full-body probe. Retain the small 16x64x{128,256} tiles, CK load/compute
pipeline, and B>1 M-fast windowed traversal as independent mechanisms. Its
cross-XCD and XCD-local split-K variants remain controls for TP8 or dense
projections only after accounting for their extra workspace, atomics, and
finalization event.

Fleet's cache prose also needs a CDNA4 correction. On a multi-L2 CDNA4 device,
device-scope vector accesses bypass XCD L2. The source-level initial streaming
candidate for weights is Gluon `.cs`, which currently lowers to `sc0 nt` on
CDNA4; `.cg` currently lowers identically, and `.ca` is the controlled temporal
comparison. The small acquired W13 intermediate uses temporal local loads so
the first consumer on an XCD can populate that XCD's L2. Cache hints do not
establish synchronization.

### Mirage and Hazy

Transfer dependency regions and `reserve -> produce -> publish -> consume ->
release` lifetimes. Compile them into phase formulas. Do not add an interpreter,
controller subgroup, descriptor queue, NVIDIA TMA, dynamic register
redistribution, or Hazy's 213 KiB page pool. CDNA4 has 160 KiB LDS per compute
unit and no gfx950 subgroup specialization facility suitable for those roles.

## 6. Persistent launch and topology bootstrap

The current offline-qualified, deployment-disabled specialization is:

- 240 workgroups;
- eight 64-lane subgroups per workgroup;
- `waves_per_eu=2`;
- `launch_cooperative_grid=False`;
- one explicitly touched 128 KiB LDS residency allocation per workgroup; and
- no private-memory/scratch instructions or VGPR spills. LLVM's distinct
  SGPR-to-VGPR lane save/restore count is recorded and tuned separately.

The active test-only grid sweep is `P in {224,226,232,240,248,256}`. Logical
work remains phase zero 240, shared/final 224, expert 226, and Iris 21, so every
grid requires an exact loop/census oracle, workspace/admission review,
loaded-code inspection, and rank-specialized code-object hashes. The current
raw ABI and production admission accept only P240.

The qualified heavy binaries allocate 106 SGPRs and 156 VGPRs. LLVM reports 40
SGPR-to-VGPR lane-save slots on every rank; the emitted ISA uses
`v_writelane_b32`/`v_readlane_b32` into two reserved VGPRs, reports
`private_segment_fixed_size == 0` and `uses_flat_scratch == 0`, and contains no
`scratch_load*`/`scratch_store*` instructions. This is register rematerialization
traffic, not a hidden memory spill. It remains a measured pressure/code-size
risk: admission records it explicitly, and tuning must not increase it without
an end-to-end win. Any VGPR spill or memory-backed scalar spill remains fatal.

Eight subgroups distribute as two subgroups per execution unit. The compiler
attribute `amdgpu-waves-per-eu=2,2` is a requested compiler occupancy
constraint, not residency proof. The touched LDS allocation makes a second
512-thread workgroup physically impossible on a gfx950 compute unit with 160
KiB LDS. It is a residency reservation first; compute may reuse parts of it only
if the final compiled dynamic-LDS requirement remains at least as large.

Ordinary launch metadata is a deployment-gated requirement. The existing
process-isolation preflight remains intact: the server command must preload
`/opt/rocm/lib/libhsa-runtime64.so.1` before Python imports PyTorch. An enabled
startup uses `dladdr` to require that `hsa_init`, `hsa_shut_down`, and
`hsa_signal_store_screlease` resolve to the pinned `/opt/rocm` object, checks its
realpath/build ID/hash, and requires exactly one loaded `libamdhip64.so`, the
PyTorch copy. PyTorch's RPATH can still map its bundled HSA DSO as a second,
non-resolving object; that dual mapping is a recorded experimental residual
risk, not grounds to mistake symbol provenance. Changing `LD_PRELOAD` inside an
initialized process is too late. Device cooperative capability and Triton's
ten-argument `hipModuleLaunchCooperativeKernel` ABI remain frozen preflight
fingerprints even though MegaMoE now rejects cooperative launch metadata. This
deliberately changes only launch routing, not the qualified runtime envelope.

This requirement comes from a reproduced lifecycle distinction on the target
machine. With PyTorch's bundled HSA object, a correct cooperative barrier
deterministically SIGSEGVs during ROCr `AqlQueue` destruction. With the system
ROCr object preloaded, the same eager kernel plus 1,000 captured graph replays
exits cleanly: XCD IDs 0 through 7, arrival 240240, generation 1001, 15 VGPRs,
zero spills, 131072 bytes LDS, and loaded occupancy one. Loading a second system
HIP runtime merely to query occupancy is also forbidden; the query resolves
through PyTorch's already loaded `libamdhip64.so`.
The reproducible source and pinned result are
`tools/megamoe/gfx950_persistent_grid_probe.py` and
`docs/design/probes/gfx950-persistent-grid-20260814.json`.

A whole-grid wait is admitted only when the actual loaded specialization
satisfies all of these machine-checked and runtime conditions:

- the frozen runtime preflight still reports cooperative-launch support, while
  the loaded MegaMoE metadata records the required ordinary option;
- `compiled.metadata.shared` and the actual launch `sharedMemBytes` argument are
  at least 131072 bytes. Gluon LDS is dynamic and a normal AMD code object can
  report `group_segment_fixed_size == 0`; if static LDS is later added, the
  1280-byte-quantized static-plus-dynamic total must remain at least 131072;
- `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`, queried on Triton's
  loaded `hipFunction_t` with block size 512 and
  `compiled.metadata.shared`, reports exactly one resident workgroup per active
  compute unit;
- admission records the actual runtime compute-unit count and requires grid
  size to be no larger than loaded occupancy times that count. The qualified
  tuple is `1 * 256 = 256 >= 240`; capacity is necessary but is not a general
  ordinary-scheduler contract;
- VGPR, SGPR, LDS, private-segment, scratch-instruction, VGPR-spill, and
  SGPR-to-VGPR lane-spill metadata are recorded. Private segment, flat scratch,
  scratch instructions, and VGPR spills must be zero; scalar lane saves must
  equal the qualified value of 40 on every rank without changing the occupancy
  envelope;
- every auxiliary-stream producer has joined; TokenSpeed injects one model
  execution stream into graph warmup/capture and later eager/replay dispatch,
  and all 92 plan views share a host owner that rejects a public decode call on
  another stream before raw dispatch;
- eager and graph capture/replay retain ordinary launch semantics, with the
  heavy grid on the marker/replay stream rather than a cooperative side stream;
- the ordinary P240 surrogate passes 1,001 generations with exactly 30
  workgroups on each XCD per generation, while the rank-zero exact heavy body
  passes eager and captured/replayed numerical and protocol checks under a
  watchdog. EP8 peer-memory qualification remains required before the false
  implementation marker can change; and
- any barrier, topology, or Iris fault poisons the plan and aborts the process;
  it never falls back after another rank may have entered the protocol.

Qualification has two distinct stages. The offline all-rank compile gate pins
the raw source, emitted AMDGCN, and rank-specialized HSACO hashes and classifies
the required publication and acquire graphs. Runtime admission records the
loaded ROCr path/hash, HIP and PyTorch versions, Triton AMD driver source path,
exact HSACO hash, launch shape, and resource metadata. It then executes one
collective eager warmup and checks fatal state, topology, and output. Serving
startup does not rerun the multi-generation heavy-binary or standalone Iris
litmuses. A change to the compiler stack, source, emitted code, or runtime
envelope invalidates qualification and requires the offline gates to be rerun
before new hashes are pinned.

At one workgroup per compute unit, P240 has 16 spare resident slots on the
256-compute-unit device and exceeds the 224-compute-unit capacity of seven
XCDs. The ordinary surrogate observed exactly 30 workgroups on every XCD in
every one of 1,001 generations, but ordinary launch does not promise this as a
general scheduling contract. Offline qualification therefore keeps a process
watchdog; runtime startup rejects a preflight, loaded-code, rank-consensus, or
eager-warmup failure. Cooperative metadata is rejected rather than retained as
a fallback. An `atomic_poll` timeout is only the fail-stop path for a resident
workgroup; it cannot rescue a workgroup that the scheduler never made resident.

At entry, every workgroup reads XCC ID once with:

```text
s_getreg_b32 ..., hwreg(HW_REG_XCC_ID, 0, 4)
```

The scalar value is masked to four bits. Only IDs below eight may index the
compact cumulative `xcc_ticket: INT64[8]`; an ID in `[8,16)` contributes
nothing and therefore cannot release topology. At entry, every workgroup
coherently reads the preceding `topology_gate` and derives the next generation.
After computing phase zero, draining its stores, and rendezvousing, it performs
one GPU-scope relaxed fetch-add on its XCD bucket:

```text
ticket            = fetch_add(xcc_ticket[xcc], 1, relaxed, gpu)
local_rank         = ticket % 30
ticket_generation  = ticket // 30 + 1
```

The exact rank-29 ticket for the current generation contributes one relaxed
fetch-add to cumulative `xcd_arrival`. The exact eighth contribution exchanges
`generation` into `topology_gate` with release GPU scope. Every workgroup polls
that topology gate before top-k and shared-down work. The gate publishes the
phase-zero payload, so its exact acquire poll retains compiler invalidation and
LDS result broadcast followed by the explicit VMEM acknowledgment and
workgroup rendezvous.

Successful graph reuse is an induction. Before generation `g`, every XCD
bucket equals `30*(g-1)` and `xcd_arrival == 8*(g-1)`. A bucket can contribute
the current rank-29 threshold at most once. If all eight buckets contribute,
each contains at least 30 current workgroups; because the launch has exactly
240 workgroups, every dense XCD has exactly 30 and no invalid ID exists. Thus
after the gate releases:

```text
xcd_ordinal(pid) = xcc(pid)                 # dense IDs 0..7
local_rank(pid)  = ticket(pid) % 30         # a permutation of 0..29 per XCD
local_count      = 30
G                = 8
```

A `31/29` or `32/28` transfer leaves the deficient bucket below rank 29 and
produces at most seven XCD arrivals, so the gate cannot release. Stale or future
cumulative state likewise fails the exact generation checks. The bounded poll
then system-poisons the rank; sticky fatal state forbids a successor, so failed
generations never need counter repair.

The previous 240-entry XCC census, per-workgroup prefix scans, population
table, and symmetric eight-rank topology consensus are absent. Cross-rank
safety remains fail-closed: a rank that cannot pass its local ticket gate never
publishes normal Iris ready epochs, so peer Iris waits time out and propagate
the same sticky fatal generation before output commit. Eager warmup additionally
checks `xcc_ticket == 30*generation`, `xcd_arrival == 8*generation`, and
`topology_gate == generation` on every rank. The borrowed Iris lane retains its
legacy topology-status allocation for protocol compatibility, but the 51-tensor
raw kernel neither receives nor accesses it.

Per successful rank and generation, this removes 240 XCC stores, the leader's
240-entry census and 16 population reductions, 240 prefix scans containing
28,680 earlier-program comparisons, population-table reads in route/expert
ownership, and one system release plus seven peer topology polls. It adds 240
relaxed INT64 bucket RMWs distributed over eight XCD counters and eight relaxed
XCD-arrival RMWs. The merged topology release/poll also publishes the phase-zero
payload and gates top-k/shared work; it retains compiler release writeback,
acquire invalidation, and the source-owned ACK/barrier required by its
workgroup-wide consumers.

## 7. Rank-local route to XCD mapping

Top-k is identical on every rank. Each rank compacts its owned selected routes
in original slot order into at most 16 entries. Let `L` be the rank-local route
count and `i` the compact route index.

For `1 <= L <= G`, the eight admitted XCDs are partitioned into balanced,
contiguous groups:

```text
base  = G // L
extra = G % L
size(i)  = base + (i < extra)
start(i) = i * base + min(i, extra)
```

For `L > G`, route `i` is assigned to XCD ordinal `i % G`; routes sharing an
XCD execute in deterministic compact-route order. For `L == 0`, routed phases
are empty but all workers remain available for shared projection,
communication, and the final projection.

Within a route's XCD group, W13 splits its 3072 intermediate output columns into
contiguous ranges. Each tile performs the full 3584-element K reduction, and
all 226 expert workgroups then rendezvous on one slot-zero W13 gate. For the
common planning case `L=2`, each route receives four XCDs; one route's MXFP4 W13
weights plus scales are approximately 11.16 MiB, or about 2.79 MiB per XCD.

W2 does not reuse those route-XCD groups. Its global output-centric grid divides
448 N8 tasks over all 226 expert workgroups. Each task visits every compact
route in order, preserves the per-route BF16 boundary, applies the FP32 route
weight, and directly writes one combined symmetric output slice before the
single slot-zero W2 gate. The per-route W2 output allocation and synchronization
slots 1 through 15 remain ABI-inactive. Measured routing histograms and cache
behavior, not a uniform-routing approximation, determine whether future
ownership ablations are worthwhile.

Internal ablations will compare flat striding, one-home-XCD, stable two/four-XCD
subsets, and adaptive groups. The public experiment switch selects only the
validated winner.

## 8. Compiled phase schedule

Let `P=240`. A logical four-subgroup task uses cohort `c in {0,1}` inside an
eight-subgroup workgroup. The following table is the complete
correctness-first `program_id -> task -> slice` contract. All N ranges are
half-open, and every listed output has exactly one writer.

| Region | Logical task domain and tile | Ownership and destination | Publication / readers |
|---|---|---|---|
| XCD ticket | one scalar task per `pid` after phase-zero drain | relaxed cumulative claim of one XCD-local rank; exact rank 29 contributes one of eight XCD arrivals | release/acquire `topology_gate`; every workgroup polls before top-k/shared work |
| input router | `q in [0,224)`, BF16 GEMV N4, K7168 | `q=pid+c*P+2*P*k`; FP32 `router[4q:4q+4]` | merged topology generation; top-k leader |
| input routed | `q in [224,448)`, BF16 GEMV N16, K7168 | same traversal; BF16 `routed[16(q-224):+16]` | merged topology generation; every W13 worker |
| input shared | `q in [448,640)`, two BF16 GEMVs N4, K7168 plus SiTU | same traversal; BF16 `shared_in[4(q-448):+4]` | merged topology generation; shared-down producers |
| topology | all `P` workgroups | exact 30-workgroup population on each dense XCD, or bounded fail-stop | merged phase/topology release-acquire with explicit ACK/barrier; no cross-rank status consensus |
| route | leader `pid=P-2`, one task over 896 experts | exact top-16 plus compact owned routes and cumulative targets | poll topology proof, then release scalar `route_plan_gate`; every expert worker acquires |
| shared down | 14 stripes, 64 logical N8/K768 tasks per stripe | `pid in [0,224)`: `s=pid//16`, `r=pid%16`, tasks `j=r+16*k`; BF16 symmetric shared `[512s+8j:+8]` | 16 producer arrivals/stripe; shared communication owner acquires stripe gate |
| W13 expert grid | per-route 384 logical N8/K3584 tasks | each of 226 expert workgroups computes its partition for at most two routes; BF16 `inter[i,8q:8q+8]` | all 226 arrive at slot-zero W13 gate; all 226 acquire it before W2 |
| output-centric W2/combine | 448 logical N8/K3072 tasks | `q=expert_rank+226*k`; each task visits local routes in compact order and writes its unique BF16 symmetric routed N8 | all 226 arrive at slot-zero W2 gate; communication owners 14..20 acquire it |
| Iris shared | 14 exact N512 stripes | shared communication index `s in [0,14)` | wait shared stripe gate, then direct producer-reduction protocol |
| Iris routed | seven exact N512 stripes | routed communication index `s in [0,7)` | follows the acquired W2 gate; for `L=0`, owner writes BF16 zero instead |
| final | 224 N32/K3584 tasks | `pid in [0,224)` recomputes current-order RMS and writes its unique BF16 output slice | acquire 21-producer communication-done gate; kernel completion publishes output |

Phase-0 task classification uses the logical bounds in the table after deriving
`q`; tasks outside `[0,640)` are empty. Every workgroup owns at least two valid
phase-0 tasks. The four-subgroup cohorts use the current K-loop and reduction
layouts. Phase-0 K is exact. W13 retains the current masked 1792-byte packed-K
tail; W2 is an exact three-by-512-byte packed-K traversal; shared down masks the
second K512 tile at K768. Every N tile is exact, and Iris has no tail.

The route plan carries local expert row `e_i`, original top-k slot, FP32 route
weight, XCD interval, and `C(i)`. W13 task `q` reads all
`routed[0:3584]`, gate rows `w13_weight[e_i, 8q:8q+8, :]`, concatenated up rows
`w13_weight[e_i, 3072+8q:3072+8q+8, :]`, and both matching scale slices, then
is the sole writer of `inter[i, 8q:8q+8]`. Output-centric W2 task `q` reads each
local route's complete intermediate and matching expert W2 rows, crosses the
per-route BF16 boundary, applies the FP32 route weight, and accumulates routes
in compact order before directly writing its unique symmetric N8. Shared task
`(s,j)` reads all `shared_in[0:768]` and rows `512s+8j:+8` of the BF16
shared-down weight. Final task `pid` reads the complete reduced routed latent
for current-order RMS, rows `32pid:32pid+32` of latent-up, `prefix_sum` and
reduced shared at the same N32 slice, and alone writes its per-layer output
slice. These offsets remain the source of truth for generated schedule coverage
tests.

After phase 0, every workgroup claims its XCD ticket and exact-acquires the
topology gate that publishes the phase-zero payload. `pid=P-2` then computes
exact top-k while workgroups `0..223` run shared down. `pid=P-2` finishes
rank-local route compaction and the two slot-zero cumulative targets and
release-publishes the route plan. Expert work begins after that plan while any
remaining shared-down work can continue. Workers execute the same compiled
state machine, with phase roles selected by formulas rather than a task queue.

Twenty-one communication workgroups are selected across XCDs after topology
publication:

```text
comm_index(pid) = local_rank(pid) * G + xcd_ordinal(pid)
is_comm(pid)     = comm_index(pid) < 21
```

With admitted `G=8`, this chooses one worker on every XCD, then a second on
every XCD, then a third on the first five: quotas `3,3,3,3,3,2,2,2`. Indices
`0..13` own shared stripes and temporarily leave expert work; indices `14..20`
participate in expert work and later own routed stripes. Expert-local rank and
population exclude only shared communication indices. Exact 30-per-XCD tickets
therefore leave 28 expert workers on XCDs `0..5` and 29 on XCDs `6..7`, for 226
expert-eligible workgroups total.

For route `i`, define `member(pid, i)` as an expert-eligible worker whose XCD
ordinal is in route `i`'s contiguous group when `L <= G`, or equals `i % G`
when `L > G`. Its exact group coordinates are:

```text
C(i) = count(q in [0,P) where member(q, i))
r(pid, i) = count(q < pid where member(q, i))
```

Thus a multi-XCD group has one deterministic, program-ID-ordered worker space;
it is not a separate traversal per XCD. The route leader stores `C(i)` and the
XCD interval in the route plan before publishing it. The implementation does
not scan all program IDs per worker. Because communication selection takes the
lowest XCD-local ranks, it precomputes:

```text
shared_quota(x) = count(k >= 0 where k*G + x < 14)
expert_count(x) = 30 - shared_quota(x)  # 28 on x=0..5, 29 on x=6..7
expert_local_rank(pid) = local_rank(pid) - shared_quota(xcd_ordinal(pid))
C(i) = sum(expert_count(x) for x in route_xcd_interval(i))
r(pid,i) = sum(expert_count(x) for x before pid's XCD in that interval)
           + expert_local_rank(pid)
```

The two definitions are equivalent under the only admitted topology,
`local_count(x) == 30`, and the test oracle checks every route count `L=0..16`.

For each route, every eligible worker in its XCD group has W13 work; the inverse
mapping assigns at most two route bodies to each expert workgroup. W2 ownership
is instead global and output-centric: 448 N8 tasks are divided over all 226
expert workgroups, and each task visits routes in compact order. `L==0` bypasses
both 226-participant gates; the seven routed owners store zero stripes. Empty
tasks still follow every dependency they consume, but do not increment a
producer counter they do not own.

Each shared stripe can enter Iris as soon as its 16 producers publish it,
overlapping peer reduction with routed expert work. Each routed owner enters
Iris after acquiring the single W2 grid gate, or after its `L=0` zero-stripe
publication. After its peer-completion epoch, each of the 21 communication
owners contributes once to `comm_done`; final workgroups acquire the resulting
gate. There is no barrier after the final store.

The reference schedule deliberately retains redundant per-output-workgroup RMS
calculation from the current final kernel. A single-producer inverse RMS would
require another device-wide publication and is only an ablation after measured
evidence.

Later optimization may publish W13 intermediate chunks and begin matching W2 K
chunks early. It is gated on ATT evidence because keeping W2 FP32 accumulators
live across polls can raise the resource maximum of the entire fused kernel.

## 9. Local memory-order protocol

Synchronization storage is graph-stable signed INT64 initialized to zero and is
never host-reset during eager execution or graph replay. Active cumulative
arrivals increase and generation gates advance; ABI-reserved
`phase_arrival`/`phase_gate` and inactive route slots remain zero. Every scalar
synchronization helper is entered convergently by all eight subgroups in a
participating workgroup: one elected lane performs the scalar RMW or exchange
and the whole workgroup rendezvouses before using its result. In particular, no
subgroup conditionally skips `gl.atomic_poll`, whose AMD lowering includes a
workgroup rendezvous.

Every wait entered by a resident workgroup remains finite under the admitted
ordinary launch so a software, topology, or peer failure cannot spin forever.
`gl.atomic_poll(..., timeout_ns=...)` returning false is a fail-stop escape, not
permission to continue. One-shot system-release exchanges publish the same
nonzero invocation generation to every rank's symmetric fatal slot; this avoids
the compiler-generated, potentially unbounded INT64 maximum CAS loop. A local
GPU-scope `atomic_cas(0, generation)` elects exactly one diagnostic writer for
the rank-local `INT64[8]` record
`{claim_generation, pid, phase, generation, XCC, expected, observed, rank}`.
The claim generation occupies slot zero and the seven diagnostic fields occupy
slots one through seven. The workgroup stops all payload accesses and returns.

Every local, peer, topology, and Iris-generation wait is a bounded
`gl.atomic_poll`; there is no hand-written global-clock loop. Successful local
and merged-topology payload polls add `_wait_subgroup_vmem_ack()` plus a second
workgroup barrier because all subgroups may consume the guarded payload. The 15
Iris system polls instead retain `atomic_poll`'s compiler-owned invalidation and
rendezvous with no source ACK or second barrier; all dependent payload loads are
restricted to subgroup zero. An Iris owner checks sticky fatal state before
publishing a normal ready generation. Thus all resident workgroups leave an
abandoned generation and the kernel can terminate.

The symmetric fatal epoch is process-lifetime sticky. Before its first scratch
store or arrival, every workgroup in every MegaMoE node exact-polls the value
zero with system-scope relaxed semantics and returns convergently when the poll
fails. The fatal value is pure control and carries no payload dependency, so
the coherent scalar load and `atomic_poll`'s built-in LDS boolean broadcast and
workgroup rendezvous are sufficient; this path intentionally has no acquire
invalidation, post-poll VMEM acknowledgment, or second source barrier. In a
healthy layer this removes that redundant sequence from all 240 entry guards
and the 21 Iris-owner guards. This entry guard is required because all 92
captured layer nodes share one workspace and lane; nodes already enqueued after
the node that detects a fault must not straddle counter generations or
overwrite their distinct outputs. The overlap scheduler may already have
enqueued the next replay on the same serialized model execution stream. Those
nodes self-abort, and each in-flight result has a distinct pinned fatal-epoch
mirror. The host rejects the failed sampled result before token commit; a
subsequent invocation is never used as an in-kernel recovery mechanism.

The resulting output is poisoned and must never be consumed as a valid token.
On failure, eager startup synchronizes, reads the fatal epoch, and gathers the
local eight-word diagnostic record. Captured decode instead enqueues only the
graph-stable fatal epoch through the sampled-token result path. A nonzero value
raises before token commit; the normal result path does not copy the full
diagnostic record. Normal overlap may have enqueued one later replay before this
check, but sticky fatal state prevents its MegaMoE nodes from advancing shared
state and its result is never committed. The serving supervisor must then
terminate and restart the entire eight-rank job. Automatic coordinated restart
is outside this repository's implementation, and diagnostics beyond the fatal
generation must be collected out of band after abort. No in-process path reuses
counters, continues the request, or enters the existing collective. This does
not make the schedule nonblocking or semantics-preserving on failure; it
converts a protocol failure from an unbounded GPU/rank hang into a bounded fatal
failure.

Startup eager warmup synchronizes and checks fatal state directly. Captured
decode carries the graph-stable fatal value through the existing sampled-token
completion/result path: before the scheduler accepts the sampled token, it
checks the associated fatal generation and rejects the result on any nonzero
value. The check is ordered after graph completion and adds no independent
per-token device synchronization. At most the overlap loop's already-issued
successor can be in flight; sticky fatal state makes its MegaMoE nodes return
without touching application state.

`TOKENSPEED_K3_MEGAMOE_FATAL_EPOCH_D2H=0` is a profiling-only discriminator
that removes the per-result scalar copy and host check while leaving the device
sticky-fatal protocol intact. It is not a serving configuration: a fault would
no longer be surfaced before token commit. Any timing collected with it must be
reported separately and retained only if the same healthy workload preserves
the reference output fingerprint.

All mutable global payload stores use write-through cache policy, and all
post-gate payload reloads use coherent cache policy. Before a relaxed arrival
can represent payload written by multiple subgroups, every producing subgroup
executes the impure sequence

```text
s_waitcnt vmcnt(0)
```

and then the workgroup rendezvouses. The source-owned block is deliberately
drain-only: it contains no `buffer_wbl2`. The last-producer release and consumer
acquire retain their compiler-generated writeback and invalidation as required.
The current source has nine explicit drain sites; specialization emits ten
blocks because the output-centric expert publication helper is instantiated
twice. The offline all-rank loaded-code gate classifies every site and rejects a
store or atomic between the drain and its first barrier. Runtime admission then
requires the exact qualified HSACO hash.

Fixed-producer handoffs use a distinct arrival/gate pair and a compile-time
producer count `C`:

```text
producer stores its complete payload with write-through policy
every producer subgroup drains; workgroup rendezvouses
ticket = atomic_add(arrival, 1, relaxed, gpu)
is_last = ticket % C == C - 1
if is_last:
    atomic_xchg(gate, invocation_generation, release, gpu)
consumer: atomic_poll(gate, expected_generation, acquire, gpu, bounded timeout)
consumer completes acquire VMEM acknowledgment and rendezvouses
consumer coherently loads payload
```

Write-through stores, per-subgroup VMEM drain, and the workgroup rendezvous make
each payload visible before its relaxed arrival. Each shared stripe (`C=16`)
and final communication completion (`C=21`) uses the modulo-last-producer test
and publishes the invocation generation already derived from `topology_gate`.
Their fixed producer counts are protected by one-stream execution and
sticky-fatal no-successor semantics: a generation with a missing or extra
producer cannot recover by advancing the next invocation. Each shared stripe
has a separate counter/gate. Topology and route leaders publish the same
invocation generation with a GPU-scope release `atomic_xchg`, and each dependent
workgroup exact-polls it with GPU-scope acquire.

For `L>0`, W13 and fused W2/combine each use the same fixed grid of 226 expert
workgroups. After the previous kernel has completed, the route leader reads
slot-zero W13 and W2 arrivals, advances each cumulative target by 226, and
publishes those targets with the route plan:

```text
w13_target[0] = coherent_load(w13_arrival[0]) + 226
w2_target[0]  = coherent_load(w2_arrival[0])  + 226
write-through route-plan stores; per-subgroup drain + rendezvous
publish route plan with scalar route_plan_gate release

each expert workgroup computes at most two route-partitioned W13 bodies
target = acquired w13_target[0]
ticket = atomic_add(w13_arrival[0], 1, relaxed, gpu)
if ticket + 1 == target:
    atomic_xchg(w13_gate[0], target, release, gpu)
all 226 expert workgroups acquire w13_gate[0]

each expert workgroup owns output tasks expert_rank + k*226 from 448 N8 tasks
for each task, accumulate every local route in compact slot order
round each route result through BF16 before applying its FP32 route weight
store the combined BF16 result directly in symmetric_producer[7168:]
target = acquired w2_target[0]
ticket = atomic_add(w2_arrival[0], 1, relaxed, gpu)
if ticket + 1 == target:
    atomic_xchg(w2_gate[0], target, release, gpu)
only communication owners 14..20 acquire w2_gate[0] before Iris
```

For `L=0`, all expert workgroups bypass both arrivals and gates. Communication
owners 14 through 20 instead write the routed stripe as BF16 zero, drain and
rendezvous, then enter Iris; expert counters and gates remain unchanged. Kernel
serialization on the admitted stream guarantees that the route leader observes
quiescent prior slot-zero arrivals before publishing the next targets. The
remaining 15 target/arrival/gate slots and `w2_route_output` allocation stay in
the flat ABI but are inactive in the restored d5dad execution path.

Only actual producers participate in these handoffs. Phase-zero completion is
represented by the hierarchical per-XCD topology tickets, not a flat
`phase_arrival`; routing is a single-producer publication; each Iris owner
contributes to the 21-producer completion; and kernel completion is the only
publication after final output stores.

Exact polling is safe locally because a gate cannot advance to the next value
until all consumers in the current kernel have exited and the next kernel has
begun on the same stream. The supported TokenSpeed runtime uses the normal
overlap scheduler while keeping graph warmup, capture, eager dispatch, and
replay on one serialized model execution stream, and binds all 92 public plan
views to that raw stream handle. A direct public decode call on another stream
is rejected before dispatch. Replaying an externally owned CUDA/HIP graph
bypasses the host owner check and is therefore unsupported; such a graph must
not share this workspace or Iris lane. Signed INT64 overflow is outside the
supported process lifetime; no wrap comparison or reset protocol is claimed.

The verified runtime ownership paths are:

```text
ModelExecutor.__init__ (python/tokenspeed/runtime/execution/model_executor.py:494)
    -> CudaGraphWrapper(..., stream=self.execution_stream) (python/tokenspeed/runtime/execution/model_executor.py:495)
        -> CudaGraphWrapper._capture_one (python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:397)
            -> torch.cuda.graph(..., stream=self.stream) (python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:524)
    -> ModelExecutor._autotune (python/tokenspeed/runtime/execution/model_executor.py:565)
        -> with torch.cuda.stream(self.execution_stream) (python/tokenspeed/runtime/execution/model_executor.py:598)
EventLoop.event_loop_overlap (python/tokenspeed/runtime/engine/event_loop.py:1706)
    -> ModelExecutor.execute_forward_op (python/tokenspeed/runtime/execution/model_executor.py:1297)
        -> with torch.cuda.stream(self.execution_stream) (python/tokenspeed/runtime/execution/model_executor.py:1329)
            -> CudaGraphWrapper.__call__ (python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:1160)
                -> captured graph replay (python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:1258)
    -> ModelExecutor.execute_idle_forward (python/tokenspeed/runtime/execution/model_executor.py:1085)
        -> with torch.cuda.stream(self.execution_stream) (python/tokenspeed/runtime/execution/model_executor.py:1100)
            -> eager idle forward or captured graph replay
```

The collective preparation warmup is the sole raw launch before this ownership
domain exists. It runs before graph construction and serving, device-synchronizes
before its control-group result is accepted, and leaves the plan owner unbound;
the first public graph-warmup or eager decode binds it. Direct calls through the
public API are allowed only when serialized on that bound stream. A third-party
call to `CUDAGraph.replay()` invokes no Python owner check, even when it happens
to use the same stream, and is outside this experiment's supported runtime.

The protocol must be confirmed in LLVM IR and AMDGCN. Required evidence includes
the arrival atomic ordering, release writeback, consumer acquire invalidate,
waits, and absence of a compiler-hoisted payload load. A cache modifier or
`gl.barrier()` alone is not a grid publication protocol.

## 10. Iris producer-direct protocol

K3's communication lane is an exact fit for the current Iris specialization:

```text
7168 shared elements = 14 * 512
3584 routed elements =  7 * 512
total                = 21 * 512
```

There is no tail and no 512-element tile crosses the shared/routed boundary.
The 21 topology-selected communication workgroups from section 8 become
temporary communication workers, one per stripe. They remain ordinary compute
workers in other phases. The persistent workgroup has eight subgroups, whereas
the current standalone Iris kernel has one. All eight subgroups enter every
control helper convergently. Payload consumption is deliberately narrower:
subgroup zero's 64 lanes each own two of the stripe's 128 packed UINT64 words,
and the other seven subgroups are masked. Each packed word contains four BF16
values, is loaded exactly once per rank, and is stored exactly once. This
ownership is required by the qualified gfx950 cache-invalidation behavior; a
scalar acquire is not treated as payload permission for sibling subgroups.

The MegaMoE lane uses protocol version 3 and owns symmetric
`iris_epoch_flags: INT64[2, 21, 8]`. Plane zero is ready and plane one is
completion. These are distinct exact-generation flags, not two increments of
one monotone flag. After its shared-stripe handoff, output-centric routed store,
or `L=0` routed-zero store, each communication workgroup is ordered after all
of that stripe's local producers and performs:

```text
local symmetric stripe ready
    -> every producing subgroup executes VMEM drain
    -> workgroup barrier
    -> system-release xchg ready[comm, rank] = generation
       and require returned value == generation - 1
    -> exact system-acquire poll local ready == generation
       with compiler-owned invalidation and rendezvous
    -> exact system-acquire poll every peer ready == generation in XOR order
       with compiler-owned invalidation and rendezvous
    -> subgroup zero loads local and peer payloads coherently
       and reduces peers in ascending rank order in FP32
    -> store local BF16 reduced stripe
    -> all subgroups execute VMEM drain;
       only subgroup zero owns reduced-stripe stores
    -> workgroup barrier
    -> system-release xchg completion[comm, rank] = generation
       and require returned value == generation - 1
    -> exact system-acquire poll every peer completion == generation
       in XOR order, with compiler-owned invalidation and rendezvous
```

The source-owned per-subgroup drain is part of the memory-model contract because
`s_barrier` does not wait sibling-subgroup VMEM counters. Source inline assembly
must show `s_waitcnt vmcnt(0)` before the publication barrier and must not contain
`buffer_wbl2`; system-release lowering supplies the required compiler writeback.
The 15 Iris system polls—local plus seven peer ready polls and seven peer
completion polls—retain compiler-owned acquire invalidation and LDS boolean
broadcast/rendezvous. They deliberately have no source-owned post-poll VMEM ACK
or second barrier because subgroup zero both owns the poll and consumes all 128
packed payload words. This differs from local and merged-topology payload gates:
their consumers span the workgroup, so `_poll_local_or_poison` and
`_poll_topology_or_poison` retain an explicit VMEM ACK and second workgroup
barrier. The exact qualified HSACO SHA-256 is checked at runtime so a
backend/codegen change cannot silently invalidate these properties.

The split completion plane prevents the next layer/invocation from overwriting
symmetric input while a peer can still read it. The no-overwrite argument is:
a rank cannot publish `ready[g+1]` until it has observed every
`completion[g]`; a peer publishes `completion[g]` only after observing that
rank's `ready[g]` and finishing its payload load. Likewise, a rank cannot
publish `completion[g+1]` until all peers publish `ready[g+1]`, and those peers
cannot do so until they observed its `completion[g]`. Exact equality therefore
cannot be skipped in the one-stream protocol. A failure poisons the process and
has no successor generation.

Remote addresses are always computed as the byte offset from this rank's
symmetric heap base added to the peer's authoritative live heap base.
Process-group rank and global rank are not interchangeable. Control peers are
visited in XOR-distance order to avoid every rank concentrating on the same
target; numerical payload accumulation remains ascending by group rank.

MegaMoE's peer helper is:

```text
wait_exact_peer_generation_or_poison(peer, plane, stripe, generation):
    local_peer_slot = flags[plane, stripe, peer_group_rank]
    peer_global = group_global_ranks[peer_group_rank]
    peer_flag = heap_bases[peer_global]
              + byte_offset(local_peer_slot, local_heap_base)
    matched = atomic_poll(peer_flag, generation, acquire, sys, timeout_ns)
    # compiler-owned invalidate plus LDS broadcast/rendezvous; no source ACK
    if not matched:
        poison_all_ranks(generation, peer, generation, coherent_load(peer_flag))
    return matched
```

Each communication workgroup calls this helper for its own ready flag, for
every peer ready flag, and for every peer completion flag. Only after all ready
polls does subgroup zero load payload. Only after all completion polls does the
worker contribute to local `comm_done`. This rank publishes through its own
column; a translated peer address selects the peer-owned column on the peer's
heap.

The MegaMoE lane is isolated from ordinary Iris protocol state. Its dedicated
symmetric producer allocation, INT64 split flags, topology status, and fatal
epoch belong only to protocol v3. Existing standalone and fused Iris calls keep
their ordinary buffers and INT32 flags unchanged; they do not alternate on the
MegaMoE flags. Signed INT64 wrap is outside the supported service lifetime.
gfx950 system-scope exact-poll compilation/disassembly and retained-binary
cross-NUMA/world-eight stress are admission gates, not assumptions.

Every communication workgroup performs its own seven ready acquires and seven
completion acquires. After all 21 workers finish the completion epoch, the
fixed-count local `comm_done` handoff publishes the reduced output to final
workers.

## 11. Iris lane ownership and package boundaries

The communication backend owns one borrowed, graph-stable producer-direct lane
shared by all 92 MoE layers. Its concrete record contains:

- a strong Iris-state owner;
- one dedicated consecutive symmetric MegaMoE producer view `[10752]`, isolated
  from ordinary and fused Iris input buffers, plus the reduced-output view;
- symmetric split exact-generation flags `INT64[2,21,8]`;
- one symmetric topology-status record `INT64[2]` containing epoch and validity;
- one symmetric sticky fatal epoch `INT64[1]`;
- the Iris context's authoritative live device heap-base tensor;
- the ordered process-group global-rank tuple and this rank's group index;
- world size, flag dtype, protocol version 3, and 512-by-21 geometry; and
- graph-stable tensor identities and data pointers for validation.

MegaMoE receives the Iris context's authoritative live heap-base tensor. The
group-global-rank tuple indexes that tensor in-kernel, and peer addresses are
formed by symmetric byte-offset translation. Existing standalone and fused
Iris kernels remain unchanged and still receive their cached Python heap-base
address tuple; they do not share MegaMoE's live-tensor ABI. Lane validation
rejects replacement of the live tensor object or its `data_ptr`. A strong owner
keeps the context, producer buffers, flags, and live base tensor alive for every
captured graph.

Protocol v3 admits no symmetric allocation or peer refresh after lane
acquisition. All Iris allocations and the final peer refresh must complete
before the lane is borrowed and before graph warmup/capture. In-place changes to
heap-base contents have no tracked refresh generation or TokenSpeed cross-rank
stream-ordering handshake and are therefore unsupported. Replacement of the
heap-base tensor or its `data_ptr` invalidates the plan. Supporting a future
post-acquisition refresh requires explicit rank-wide quiescence, a refresh
generation, event ordering onto every future lane stream, failure poisoning,
and graph destruction/recapture rules before this restriction can be relaxed.

The offline qualification and runtime host paths are:

```text
Offline candidate qualification
    -> compile all eight rank specializations
    -> inspect source, AMDGCN publication/acquire graphs, resources, and hashes
    -> run watchdog-protected heavy-binary eager/capture/replay and Iris litmuses
    -> pin the qualified rank-specialized HSACO hashes
Kimi K3 post_quant_warmup
    -> validate every layer before entering a collective
    -> runtime acquire_producer_direct_lane
        -> AutoBackend
            -> TritonAllReduceBackend._get_or_create
                -> tokenspeed-kernel Iris state
                    -> borrowed producer-direct lane
    -> tokenspeed-kernel prepare_kimi_k3_megamoe
        -> allocate one local scratch workspace and per-layer outputs
        -> require the common 16-byte pointer-alignment trait
        -> compile, load, and retain the one raw specialization on every rank
        -> exchange compile and loaded-resource status on the control group
        -> launch one collective eager warmup
        -> synchronize and check fatal state, topology, and finite output
    -> attach immutable per-layer views of the model-level plan
ModelExecutor initialization
    -> create the model execution stream
    -> inject it into CudaGraphWrapper capture
    -> serialize autotune and graph warmup/capture on it
EventLoop overlap step
    -> enqueue the current forward on the model execution stream
    -> synchronize and commit the previous result while the current forward runs
    -> active forward: execute_forward_op on the model execution stream
    -> DP-idle forward: execute_idle_forward on the same stream
        -> public MegaMoE decode binds/checks the shared plan owner
            -> retained raw runner
```

Gluon normally includes pointer alignment in its specialization key. Every
model, lane, workspace, input, prefix, and output pointer is therefore required
to be at least 16-byte aligned, which fixes the same `D` trait used by the
preparation tensors. Preparation retains the admitted `CompiledKernel` and its
materialized `compiled[(240, 1, 1)]` runner in every immutable layer plan.
Collective warmup, graph capture, and decode call only that direct runner; the
steady-state launcher cannot enter the JIT, load another module, or repeat
admission. An offset-but-contiguous pointer with a weaker trait is rejected
before dispatch. The plans also share one host stream owner. Its first public
decode binds the raw model-execution-stream handle; a later public call through
a different stream fails before the retained runner is entered. This guard does
not intercept a third-party graph object's `replay()`, so direct/external graph
ownership is outside the experimental API contract.

The decoder layer passes `ctx.forward_mode.is_decode_or_idle()` explicitly into
`KimiLinearMoE.forward`; `M==1` alone is insufficient because a one-row EXTEND
is prefill and remains a non-goal. The forward path is:

```text
KimiLinearDecoderLayer.forward
    -> KimiLinearMoE.forward
       [M == 1, decode-or-idle, and plan exists]
    -> tokenspeed_kernel.kimi_k3_megamoe_decode
        -> registered gfx950 Gluon wrapper
            -> unpack borrowed lane to raw tensors/scalars
            -> tokenspeed-kernel-amd MegaMoE launcher
                -> one persistent kernel
```

The runtime sees the lane as opaque. `tokenspeed-kernel` owns its concrete type
and validates it. `tokenspeed-kernel-amd` receives only ordinary tensors,
pointers, integers, and compile-time traits; it does not import Iris or
`tokenspeed-kernel`.

Required files are kept under separate MegaMoE directories:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/megamoe/` for the public
  operation, plan, selection, and gfx950 registration shim;
- `tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/megamoe/`
  for the static schedule and Gluon implementation; and
- corresponding tests under `tokenspeed-kernel/test/` and runtime tests under
  `test/runtime/`.

`CommBackend.acquire_producer_direct_lane(shapes, like, group) -> object | None`
is the only new runtime-facing communication API. The runtime treats the return
as opaque; only the TokenSpeed-kernel registration wrapper unpacks it. Lane
acquisition is collective and has no broad exception-to-fallback wrapper. All
ranks acquire in the same order. The lane may not be used concurrently on two
streams, flags are not host-reset, and all ranks must take the same fused or
fallback path. Any common-Iris change lands with standalone producer-direct
regression tests before MegaMoE depends on it. The borrowed lane wraps the
exact `IRIS_AR_STATES` instance used by the current fallback path but owns
separate MegaMoE producer and flag allocations inside that state; constructing
a second Iris all-reduce state is forbidden. Lane validation rejects aliasing
between the dedicated MegaMoE allocations and ordinary Iris buffers or flags.

## 12. Model lifecycle and experimental flag

`ServerArgs` contains `enable_kimi_k3_megamoe: bool = False`, the CLI exposes
`--enable-kimi-k3-megamoe`, and the field is copied into
`global_server_args_dict`. The normalized `load_format` is exported there as
well so model-side admission can enforce the loader contract. No new MoE
backend enum is added: the existing `mega_moe` backend name belongs to a
different DeepSeek implementation.

Preparation occurs in a new top-level K3 `post_quant_warmup()` delegate. The
default loader invokes this hook only after every quant method and module has
run `process_weights_after_loading` (`model_loader/loader.py:415-433`). This is
late enough for K3's projection concatenation and the selected MXFP4 plan's
validation, and early enough for KV-cache memory profiling, workspace freeze,
and CUDA graph capture. Lazy per-layer Iris or workspace allocation in
`forward()` is forbidden.

The first experiment's loader allowlist is exactly normalized `LoadFormat.AUTO`
resolved to `DefaultModelLoader`, matching the artifact command. In
`WeightLoader.load_model`, after constructing `LoadConfig` but before
`get_model()`, an enabled flag checks both values and rejects every other loader
class, including dummy, sharded-state, and extensible loaders. Immediately after
`get_model()` returns, the enabled path asserts
`model._kimi_k3_megamoe_prepared is True`, all 92 layer plans are non-null, and
their lane owner is identical. Missing `post_quant_warmup()` is therefore a
startup error, never a forward-time silent fallback.

Online weight replacement is rejected before any tensor mutation because K3
projection repacking changes backing addresses and invalidates captured graphs.
The pre-mutation guards are exact:

- `AsyncLLM.update_weights_from_disk` rejects before creating/dispatching the
  update request or changing `server_args`;
- `SchedulerControlClient.update_weights_from_tensor` and
  `update_weights_from_distributed` reject before invoking their communicators;
  and
- top-level `KimiK3ForConditionalGeneration.load_weights()` checks an already
  prepared `_kimi_k3_megamoe_prepared` flag before consuming the iterable. This
  is defense in depth for direct worker calls; it is false during the one
  admitted initial load and becomes immutable after preparation.

Later update support requires a quiescent plan teardown, repacking, lane/plan
reconstruction, graph destruction, and graph recapture on all ranks.

Every rank first performs all noncollective validation over all 92 layers. The
exact admission predicate is:

- architecture `KimiK3ForConditionalGeneration` with a language model;
- gfx950 exactly, `world_size == nprocs_per_node == 8`, and `nnodes == 1`;
- attention TP8 with DP1/CP1 and MoE TP1/EP8/DP1;
- MoE EP group and TP-EP group are the same ordered eight local world ranks;
- native execution plan with `use_native` and `joint_moe_reduce` true;
- `all2all_backend == "none"` and deterministic RSAG disabled;
- trivial expert locations, no redundant experts, and EPLB disabled;
- no speculative algorithm and physical decode `M=1` for the fused dispatch;
- 92 MoE layers, 896 experts, 112 contiguous local experts, top-16,
  sigmoid/noaux routing, `n_group == topk_group == 1`, renormalization enabled,
  and routed scaling factor one;
- SiTU beta four, linear beta 25, and latent RMSNorm enabled;
- the six exact BF16 contiguous projection/norm shapes, FP32 correction-bias
  shape, field order, colocated device, strides, and alias rules from section 4;
- the four exact linear MXFP4 tensor shapes, dtypes, strides, concatenated W13
  layout, and `linear_weights=True` from section 4; and
- the persistent resource, stream-exclusivity, and topology predicates from
  section 6.

Startup then follows a staged no-hang protocol:

```text
all ranks finish all local model/layout checks
    -> control-group exchange of fixed status and bounded reason
        -> any failure: every rank raises the same startup error
        -> all success: collectively acquire the Iris lane
            -> exchange acquisition status
                -> any failure: every rank raises
                -> each rank JIT-compiles its one rank-specialized kernel
                   without a collective kernel launch
                    -> exchange compile/resource status
                        -> any failure: every rank raises
                        -> collectively warm the retained specialization once
                            -> exchange eager-warmup status
                                -> any failure: every rank raises
                                -> attach plans on every rank
```

The status exchange itself cannot invoke the unprepared Iris data path. Every
stage has a subprocess watchdog, and no rank enters the next collective until
the control-group consensus succeeds. Separate one-rank fault injection is
mandatory for validation, lane acquisition, compilation/resource checks,
local topology ticket failure, and collective warmup.

Once prepared, decode-or-idle `M=1` must call MegaMoE; it cannot catch an
execution error and enter a different collective. EXTEND/prefill of any shape
and `M!=1` deterministically use the existing K3 path on all ranks. Here `M` is
the physical tensor row count, not HTTP concurrency: an idle graph-size-one
replay may execute MegaMoE on its padded dummy row and must participate in the
same Iris epochs.
With the flag enabled, a missing layer plan on an admitted decode-or-idle row is
an invariant violation that raises before launch; plan existence is not a
fallback selector. The fallback clauses above apply only to unsupported forward
modes/shapes that are chosen uniformly before any MegaMoE protocol begins.

## 13. Scratch layout

One model-level local workspace containing exactly 31 nonsymmetric tensors is
reused sequentially by all MoE layers and graph replays. Its grouped contents
are:

- FP32 router logits `[896]`;
- BF16 routed input `[3584]`;
- BF16 shared input `[768]`;
- top-k IDs and FP32 weights `[16]`;
- compact local route IDs, slots, weights, and count `[16]`;
- route XCD intervals and worker counts `[16]`, plus INT64 W13/W2 cumulative
  target arrays whose slot zero is active and remaining slots are ABI-retained;
- BF16 W13 intermediates `[16, 3072]`;
- ABI-retained, inactive BF16 per-route W2 outputs `[16, 3584]`;
- cumulative INT64 XCD tickets `[8]` and one cumulative eight-XCD arrival;
- ABI-reserved INT64 `phase_arrival` and `phase_gate`, both zero in the restored
  d5dad protocol, plus active `topology_gate`, scalar `route_plan_gate`, 14
  shared-stripe, slot-zero-active `w13_arrival/w13_gate[16]` and
  `w2_arrival/w2_gate[16]`, and communication arrival/gate state;
- fail-stop diagnostics `INT64[8]` containing claim generation, program ID,
  phase, generation, XCC, expected value, observed value, and rank.

The model plan separately allocates BF16 per-layer final outputs
`[92,1,7168]`; they are not members of the 31-tensor scratch workspace.

Iris's dedicated symmetric MegaMoE producer `[10752]`, reduced output `[10752]`,
live heap-base tensor, split INT64 exact-generation flags `[2,21,8]`, symmetric
topology status `[epoch, valid]` with shape `[2]`, and symmetric sticky INT64
fatal epoch `[1]` belong to the borrowed protocol-v3 lane and are not local
MegaMoE scratch. Topology status is retained for lane compatibility but is not
passed to the raw kernel. Ordinary Iris input buffers and INT32 flags remain
distinct.
The reference schedule computes inverse RMS inside each final workgroup and has
no shared inverse-RMS slot.

The typical rank has about two local routes, but buffers cover all 16 so route
skew cannot corrupt memory. Per-layer outputs make the returned prefix live
until its downstream consumers finish; internal scratch is shared because
layers execute serially on the admitted stream. Address/alias canaries verify
that layer `n+1` cannot clobber layer `n` early. All addresses and strides are
graph-stable, and all storage is counted before KV-cache memory profiling.

## 14. Validation gates

Correctness precedes tuning. The experiment remains disabled because its final
performance gate failed. The list below is retained as the qualification
contract; the cited artifacts, rather than presence in the list, establish
which gates passed.

### Static schedule tests

- exact tile coverage with no duplicates for every `L` from 0 through 16;
- balanced XCD interval construction and non-dense observed XCC IDs;
- deterministic compact-route/top-k slot ordering;
- unique W13/W2/shared/combine/output ownership; and
- changing successive producer populations with exact cumulative W13/W2
  targets and no gate lapping;
- topology-aware 21-worker selection and exact coverage of all 128 packed words
  per communication stripe; and
- no scheduler-style fetch-add task assignment.

### Single-GPU kernel tests

- each projection phase against the existing specialized kernel/reference;
- router-logit comparison plus exact K3 sigmoid/bias top-16 IDs and weights,
  including ties and adversarial top-16/top-17 near-cutoff scores;
- zero, one, two, eight, and sixteen local routes;
- W13 BF16/SiTU and per-route W2 BF16 boundaries;
- final inverse RMS and latent-up/add output;
- cross-XCD handoff lapping with distinct payloads and prewarmed stale cache
  lines;
- forced local barrier timeout records exact diagnostics, poisons every
  workgroup, and returns without consuming a partial payload;
- eager repetition and CUDA graph capture/replay under subprocess watchdogs;
- actual heavy-code-object barrier replay for every retained grid; and
- cumulative topology tickets, loaded-function occupancy, resource metadata,
  no scratch or
  private-memory spills, an exact rank-specialized SGPR-to-VGPR lane-save
  expectation, all nine source/10 specialized producer drain sites, all six
  local/topology acquire-plus-explicit-ACK graphs, all 15 Iris compiler-owned
  acquire/rendezvous graphs with no source ACK, the relaxed-arrival and fatal
  control graphs, and an exact qualified HSACO SHA-256 check.

### Eight-GPU Iris tests

- distinct per-rank values across the exact `(7168, 3584)` boundary;
- fused reduction versus current `iris_all_reduce_symmetric`;
- protocol-v3 lane storage is nonaliasing with ordinary Iris producer buffers
  and INT32 flags;
- exact-generation ready and completion planes, including self-ready acquire,
  with compiler-owned acquire invalidation and rendezvous at every system wait,
  no source post-ACK, and subgroup-zero ownership of all dependent payload;
- independently skewed ranks before ready and before completion, proving exact
  generation acquisition for every peer on both planes;
- forced ready- and completion-wait timeouts propagate the fatal generation to
  all ranks and let every kernel return without fallback;
- successive valid then one-rank-invalid topology generations under rank skew,
  proving stale status cannot satisfy the generation-tagged consensus and all
  ranks fail-stop uniformly;
- every final consumer workgroup acquiring `comm_done` itself;
- fresh-dispatch retained-runner producer-to-ready-to-peer-load-to-completion
  stress with subgroup-distinct payloads, proving every producer subgroup is
  system-visible while subgroup zero consumes all 128 packed words;
- 4,096-generation cross-NUMA and world-eight burst replay without
  per-generation device synchronization or control-group collectives, with
  rank-dependent enqueue skew, full generation/site/subgroup histories, and
  exact final ready/completion values;
- gfx950 INT64 system-release/exact-acquire litmus and large nonwrapping
  generation seeds;
- rejection of any symmetric allocation or peer refresh after protocol-v3 lane
  acquisition, before an existing captured plan can replay;
- heap-base tensor or storage replacement invalidating every retained plan and
  requiring graph destruction/recapture;
- borrowed-lane identity with the exact fallback `IRIS_AR_STATES` owner;
- lane-owner garbage-collection lifetime and concurrent-stream rejection;
- explicit rejection of a permuted/non-world process group; and
- output comparison against the current six-launch K3 layer on real processed
  weights.

Equal inputs on all ranks are insufficient because they hide rank swaps,
duplicate contributions, and missing peers.

### End-to-end model tests

- flag-off behavior and kernel sequence are unchanged;
- flag-on startup admission and precise rejection cases;
- one-rank-invalid startup fault injection makes every rank exit rather than
  hang;
- exact expert tensor shape, stride, dtype, and layout rejection;
- unsupported-loader and online-update rejection before mutation;
- `M=1` graph contains exactly 92 MegaMoE nodes and zero occurrences of all six
  replaced kernel names;
- `M=2,4,8,16` uniformly use the existing path;
- one-row EXTEND/prefill uses the existing path despite `M==1`;
- long idle graph-one replay and idle/decode alternation;
- eager and captured timeout injection is observed on the sampled-token result
  path before the scheduler accepts a token, blocks subsequent replay, and
  makes a second captured layer node sharing the workspace leave all counters,
  gates, lane buffers, and both layer outputs unchanged;
- consecutive-layer output-address and alias canaries;
- deterministic greedy decode agrees with the baseline within the existing K3
  tolerance/trajectory contract; and
- long replay does not deadlock or reuse a lane early.

## 15. Compiler and profiling evidence

For every retained persistent candidate, save TTIR, TTGIR, LLVM IR, AMDGCN,
kernel metadata, and raw benchmark results. Inspect:

- actual `s_getreg_b32` XCC extraction;
- active arrival/gate RMW, release, and acquire sequences, with no flat
  phase-arrival RMW;
- system-scope Iris atomics and peer address arithmetic;
- 512-element exact stripe loads with no tail masking;
- weight and peer cache-control bits (`sc0`, `sc1`, and `nt`);
- source-owned `s_waitcnt` drain placement and compiler-inserted release
  `buffer_wbl2`/acquire `buffer_inv` placement;
- peer-loop issue/consume order;
- VGPR, SGPR, LDS, private segment, scratch/spills, and code size; and
- the effective `amdgpu-waves-per-eu=2,2` attribute.

Normal Perfetto traces establish representative layer timing and rank skew.
rocprofv3 establishes launch geometry and code-object metadata. ATT is used only
after dispatch isolation to identify source/ISA stalls; serialized Iris ATT
time is explicitly excluded from performance claims.

Tuning proceeds one variable at a time over:

- persistent grid `P in {224,226,232,240,248,256}`, retaining only candidates
  that preserve the fixed logical task census and satisfy exact loaded
  residency plus eight-XCD gates;
- flat, home-XCD, fixed-subset, and adaptive-XCD ownership;
- input and shared output tile sizes;
- shared-first/routed-first/parity-interleaved phase 3;
- temporal versus XCD-streaming weight loads;
- one-time versus repeated RMS calculation;
- bulk versus chunk-published W13-to-W2 handoff.

A source transformation is rejected if its gain comes from omitted completion,
changes numerical order unintentionally, introduces memory/VGPR spills,
regresses the qualified scalar lane-spill bound without an end-to-end win,
destabilizes graph replay, or wins only an ATT-perturbed measurement.

## 16. Initial d5dad performance rejection and final acceptance rule

This section preserves the first controlled cooperative-d5dad rejection. It is
historical evidence, not the current tuning scoreboard; the living ledger
records the later ordinary-launch and scheduler-overlap results.

Kernel microbenchmarks are necessary but not sufficient. The final paired
benchmark is recorded in
[`comparison.json`](../../profile_default/kimi-k3-megamoe-final-d5dad-20260814T120409Z/comparison.json).
The historical artifact and two contemporaneous arms are reported separately:

1. historical commit `04bc0864`, flag absent: **58.0592 output tokens/s**;
2. the final candidate checkout and environment, flag off; and
3. the identical candidate checkout, build, environment, and commands, flag on.

The causal comparison is arm 3 versus arm 2; arm 3 versus arm 1 answers the
requested historical comparison. Protocol-v3 split INT64 flags and exact
acquires are isolated from ordinary Iris, whose INT32 flags and current kernels
remain unchanged. The contemporaneous flag-off arm isolates the MegaMoE switch
from checkout, compiler, ROCr-preload, and environment drift.

Each current arm used the exact artifact server/evalscope configuration, three
unprofiled repetitions, the same 4096-token input and 1024-token output,
concurrency one, greedy sampling, seed, and graph sizes. Execution order was
fixed rather than alternated: all three flag-off runs preceded all three
flag-on runs. This leaves temporal/order drift as a benchmark caveat, although
the measured loss is large and consistent across repetitions. A fixed warmup
was discarded, and every raw run and median was recorded. Preflight required
the resolved `enable_prefix_caching` value to be `false` for both arms. Both
current arms set
`LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1`; the historical artifact number
is contextual because that original command did not.

| Arm | Raw output throughput (tokens/s) | Median output throughput | Median TPOT | Median TTFT |
| --- | --- | ---: | ---: | ---: |
| flag off | 56.0157, 54.6373, 56.8902 | **56.0157 tokens/s** | 17.35 ms | 544.59 ms |
| flag on, restored d5dad | 14.0183, 14.1753, 14.1222 | **14.1222 tokens/s** | 70.35 ms | 539.43 ms |

The implementation marker was temporarily set to `True` only to collect the
controlled flag-on arm, then restored to `False` after the performance
rejection.

The causal flag-on versus flag-off throughput delta was **-74.7889%**, and flag
on also remained below the historical **58.0592 tokens/s** median. All
six contemporaneous runs used the identical prompt SHA-256
`ac1d4aa0f8c8d03580e7565d297cb3563e04424d0e2fc5559ddee6c248b70106`
and produced the identical normalized completion SHA-256
`4c915ff07d791eb281d5d7bc68273573114e43f7db855dca2d4dfe7b9cef7cc9`.
The comparison therefore establishes output identity for this workload, but it
decisively rejects performance admission.

An aggregate-completion protocol also passed its isolated retained-binary
publication litmus. Integrating it into the production kernel nevertheless
regressed the measured EP8 timing from 0.163426 ms to 0.170793 ms, a 4.51%
increase. That production change was reverted; the rejected aggregate variant
remains preserved only in
`tokenspeed-kernel/test/ops/moe/megamoe/test_amd_aggregate_completion_publication.py`
as negative performance evidence. The restored d5dad kernel retains per-tile
completion publication and acquisition.

TPOT and TTFT are included above. Retained per-layer timing, rank-skew,
compile/resource, and higher-captured-batch fallback probes remain diagnostic
evidence; they do not override the paired output-throughput rejection.

The cooperative d5dad candidate produced an identical normalized completion for
the paired workload but did not beat either baseline. The experiment therefore
remains off, with the implementation marker set to `False`. Ordinary launch
subsequently removed the side-stream/interference component, but has not yet
passed the full matched 4096-to-1024 acceptance benchmark. Phase-attributed
body optimization and that final paired benchmark remain mandatory;
launch-count reduction alone is not a success criterion.

## 17. Extension to target batches 8 and 16

The batch-one schedule must not hard-code an ownership model that blocks B8 or
B16. Those two target batches use separately compiled plans. They group routes
by expert, use stable expert/XCD subsets only when measured, and traverse token
rows M-fast/windowed so a weight tile can serve several rows before eviction.
Their routing, scratch sizing, collective geometry, graph capture, and
residency are separate contracts; they are not enabled merely by relaxing an
`M==1` check.

A matched output-throughput win at B1, B8, or B16 is sufficient for the project
goal. Every reported winner remains scoped to its measured batch: a B8/B16
throughput result does not establish B1 latency, and a B1 result does not prove
weight reuse or tail behavior at B8/B16.

The current optimized kernels remain the correctness and fallback path until a
variant independently passes the same output-throughput and memory-model gates.
