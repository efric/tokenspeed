experimental prototye for a "mega" moe design
only tested on bs=1, 4k/1k -> 42 output tok/s at time of its previous commit (TOM was 58 tok/s)

Design overview:

                             preceding attention / AttnRes
                                          |
                             h [1,7168], p [1,7168]
                                          |
                                          v
    +================================================================================+
    | KERNEL #1: _kimi_k3_megamoe_core_kernel                                      |
    | kernel.py:2012                                                                 |
    |                                                                                |
    | Ordinary persistent P240 launch:                                               |
    |   - 240 long-lived workgroups                                                  |
    |   - 8 subgroups/workgroup                                                      |
    |   - 64 KiB LDS/workgroup                                                       |
    |   - admitted capacity: 256 resident workgroups                                 |
    |                                                                                |
    | The same workgroups move through the phases below.                             |
    +================================================================================+
                                          |
                                          v
    +================================================================================+
    | PHASE A: INPUT PROJECTIONS                                                     |
    | _phase0_projection, kernel.py:502                                              |
    |                                                                                |
    | Logical work is the same as production kernel #1, but packed into P240.        |
    | Two subgroup cohorts let one workgroup execute two projection tasks.           |
    |                                                                                |
    |    pid 0..111              pid 112..191             pid 192..207               |
    |    ROUTED PAIR             SHARED PAIR               SHARED PAIR                |
    |                            + ROUTER PAIR                                         |
    |                                                                                |
    |    32 routed cols/WG       8 shared cols/WG         8 shared cols/WG           |
    |    h @ Wrouted_down.T      h @ Wshared_gate/up.T    h @ Wshared_gate/up.T      |
    |                                                                                |
    |                                                  pid 208..239                  |
    |                                                  ROUTER PAIR                   |
    |                                                  8 router cols/WG              |
    |                                                  h @ Wrouter.T                 |
    |                                                                                |
    | Totals:                                                                        |
    |   112 × 32 = 3584 routed columns  -> xL [1,3584] BF16                          |
    |    96 ×  8 =  768 shared columns  -> uS [1,768] BF16                           |
    |   112 ×  8 =  896 router columns  -> q  [1,896] FP32                           |
    |                                                                                |
    | Shared gate/up keeps the same BF16 boundaries before SiTU.                     |
    +=====================================+==========================================+
                                          |
                             drain all subgroup stores
                                          |
                                          v
    +================================================================================+
    | PHASE B: TOPOLOGY/RESIDENCY GATE                                               |
    | _arrive_topology_ticket                                                        |
    |                                                                                |
    | Every workgroup reads its hardware XCD ID. The launch must establish:          |
    |                                                                                |
    |                       30 workgroups per XCD                                     |
    |                              × 8 XCDs                                           |
    |                              = P240                                             |
    |                                                                                |
    | The 30th workgroup on each XCD publishes an arrival. The eighth XCD releases   |
    | the topology gate. No later phase starts until all phase-A producers finish.   |
    |                                                                                |
    | Important: phase-A roles were selected by program ID. Roles below are remapped |
    | using XCD-local workgroup rank, so they are aligned to the observed hardware   |
    | placement rather than assuming program IDs map directly to XCDs.               |
    +=====================================+==========================================+
                                          |
                    +---------------------+----------------------+
                    |                                            |
                    v                                            v
    +======================================+   +=====================================+
    | PHASE C1: ROUTER + ROUTE PLAN        |   | PHASE C2: SHARED DOWN              |
    |                                      |   | _shared_down_produce, kernel.py:867 |
    | One elected workgroup: pid 238       |   |                                     |
    |                                      |   | pid < 224                           |
    | hierarchical sigmoid/top-16          |   |                                     |
    | kernel.py:643                        |   | 14 stripes × 512 output columns      |
    |                                      |   | 16 producer workgroups/stripe       |
    |   8 subgroups each select local 16   |   |                                     |
    |   candidates from 128 experts        |   | uS @ Wshared_down_rank.T            |
    |              |                       |   |        768 -> 7168                   |
    |     union of 128 candidates          |   |                                     |
    |              |                       |   | directly stores BF16 into:           |
    |       exact global top-16            |   | symmetric_producer[0:7168]           |
    |              |                       |   |                                     |
    |       normalized weights             |   | Each stripe publishes a ready gate  |
    |                                      |   | after all 16 producers arrive.       |
    | _publish_route_plan, kernel.py:765   |   +==================+==================+
    |                                      |                      |
    | Keep only experts owned by this EP8  |                      |
    | rank: local route count L <= 16      |                      |
    |                                      |                      |
    | Assign each local route to one or    |                      |
    | more XCD intervals:                  |                      |
    |   L <= 8: divide 8 XCDs across L     |                      |
    |   L > 8: route i -> XCD (i mod 8)    |                      |
    |                                      |                      |
    | Publish route IDs, original slots,   |                      |
    | weights, worker counts and gates.    |                      |
    +==================+===================+                      |
                       |                                          |
                       v                                          |
    +================================================================================+
    | PHASE D: ROUTED EXPERT COMPUTE                                                 |
    | _expert_phases, kernel.py:1367                                                 |
    |                                                                                |
    | The topology mapping reserves:                                                 |
    |                                                                                |
    |   14 workgroups for shared Iris tiles                                          |
    |   226 workgroups for routed expert work                                        |
    |                                                                                |
    | On XCD 0..5: 28 expert workgroups/XCD                                          |
    | On XCD 6..7: 29 expert workgroups/XCD                                          |
    | Total: 6×28 + 2×29 = 226                                                       |
    |                                                                                |
    | D1: W13                                                                        |
    |                                                                                |
    |   Each compact local route is assigned an XCD interval.                        |
    |   The expert workgroups in that interval divide its N8 output tiles:           |
    |                                                                                |
    |       xL @ W13[expert].T                                                       |
    |           -> BF16 boundary -> SiTU                                             |
    |           -> w13_intermediate[local_route,3072]                                |
    |                                                                                |
    |   All 226 expert workgroups arrive at one W13 gate.                            |
    |                                                                                |
    |                                      |                                         |
    |                                      v                                         |
    | D2: output-centric W2 + route combine                                          |
    |                                                                                |
    |   448 N8 output tasks are divided across the same 226 workgroups.               |
    |   For every owned N8 output tile:                                              |
    |                                                                                |
    |       accumulator = 0                                                          |
    |       for local route in original top-16 slot order:                           |
    |           value = w13_intermediate[route] @ W2[expert].T                       |
    |           value = BF16(value)                                                  |
    |           accumulator += FP32(value) * route_weight                            |
    |       store BF16(accumulator)                                                  |
    |                                                                                |
    |   Unlike the earlier prototype, there is no materialized                       |
    |   [route,3584] W2 output followed by another combine phase.                    |
    |                                                                                |
    |   It writes directly into:                                                     |
    |       symmetric_producer[7168:10752] = r_rank [1,3584]                         |
    |                                                                                |
    |   All 226 workgroups then arrive at the W2 gate.                               |
    +=======================================+========================================+
                                            |
                  +-------------------------+-------------------------+
                  |                                                   |
                  | shared tiles may begin Iris while routed           |
                  | W13/W2 work is still executing                     |
                  v                                                   v
    +================================================================================+
    | PHASE E: PRODUCER-DIRECT IRIS INSIDE THE SAME P240 KERNEL                      |
    | _iris_communication_tile, kernel.py:1671                                      |
    |                                                                                |
    | Exactly 21 topology-selected workgroups become communication owners:           |
    |                                                                                |
    |       comm index 0..13                       comm index 14..20                   |
    |       14 shared stripes                      7 routed stripes                   |
    |       14 × 512 = 7168                        7 × 512 = 3584                     |
    |       wait for shared stripe gate            wait for W2 completion            |
    |                                                                                |
    |                 symmetric producer buffer on each rank                         |
    |       +--------------------------------+---------------------------+            |
    |       | s_rank: 7168 BF16             | r_rank: 3584 BF16        |            |
    |       +--------------------------------+---------------------------+            |
    |                                                                                |
    | Each communication workgroup uses all 8 subgroups:                             |
    |                                                                                |
    |   1. Publish this rank's exact ready epoch with system-release semantics.       |
    |   2. Acquire every peer's ready epoch.                                          |
    |   3. Seed FP32 accumulators with this rank's BF16 values.                       |
    |   4. Load peer payloads in ascending rank order, skipping the local rank.       |
    |   5. Store the reduced 512-element BF16 tile.                                  |
    |   6. Publish completion and wait for every peer's completion epoch.            |
    |                                                                                |
    | Ready protects producer -> reader visibility. Completion prevents a rank from  |
    | reusing its symmetric producer storage while another rank still reads it.      |
    |                                                                                |
    |                  symmetric_reduced on every rank                               |
    |       +--------------------------------+---------------------------+            |
    |       | s = SUM_EP(s_rank), 7168      | r = SUM_EP(r_rank), 3584 |            |
    |       +--------------------------------+---------------------------+            |
    +=======================================+========================================+
                                            |
                            persistent P240 launch completes
                                            |
                    same-stream kernel boundary supplies the final dependency
                                            |
                                            v
    +================================================================================+
    | KERNEL #2: _kimi_k3_megamoe_finalizer_kernel                                  |
    | kernel.py:2273, computation at kernel.py:1925                                  |
    |                                                                                |
    | Separate ordinary P224 launch, 8 subgroups/workgroup.                          |
    | Each workgroup owns one N32 output slice:                                      |
    |                                                                                |
    |       r = symmetric_reduced[7168:10752]                                        |
    |                                                                                |
    |       RMSNorm(r)                                                               |
    |            |                                                                   |
    |       BF16 normalized activation                                               |
    |            |                                                                   |
    |       normalized @ Wlatent_up.T                                                |
    |            |                                                                   |
    |       BF16 routed_up                                                           |
    |            |                                                                   |
    |       y[N32] = p[N32] + routed_up[N32] + s[N32]                               |
    |                                                                                |
    | As in production, this is not another shared-expert GEMM. It loads the already |
    | reduced shared result and joins it with the routed projection and prefix.      |
    +=======================================+========================================+
                                            |
                                            v
                                      y [1,7168]
