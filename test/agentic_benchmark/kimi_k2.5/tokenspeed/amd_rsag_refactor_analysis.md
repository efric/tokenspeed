# AMD token RSAG refactor: root-cause correction, design proof, and review

Date: 2026-07-23

This document analyzes the AMD token all-gather/reduce-scatter (RSAG) path used
by Kimi K2.5 Attention DP8 + MoE TP8 and Attention DP8 + MoE EP8. It covers the
original implementation at TokenSpeed commit
`d4ffe4448c78c15bc0b1df801be04b0fd999be4f`, the one-launch refactor introduced
at `58f0749c9b861bd1d06558d6fdef26102afe36c6`, and the one-wave correctness
hardening on branch `fix-dp-kimi-rsag-hang`.

The most important conclusion is a correction:

> The evidence does not show that the old RSAG kernel caused the observed Kimi
> DP8 startup failure, and the alleged old-kernel hang could not be reproduced.
> The concrete DP8 + TP8 failure was a persistent inference-tensor lifetime
> error. The old RSAG nevertheless had a real performance pathology, a
> source-level cross-wave publication gap, and a cross-device scheduling
> dependency with no forward-progress guarantee. The new design removes those
> structural risks. Restricting each payload workgroup to one AMD wave closes
> the remaining memory-ordering proof without splitting the barrier into a
> second kernel.

This distinction matters. A robust replacement is justified, but it must not be
presented as the experimentally isolated fix for a hang that the investigation
did not reproduce.

## Executive result and confidence ledger

| Question or prior claim | Result | Evidence strength |
| --- | --- | --- |
| Do both Kimi Attention DP8 layouts invoke token RSAG? | Yes. Attention resolves to TP1/DP8, while dense and MoE computation have width eight. | Proven from parallelism resolution and `CommManager.use_all_reduce()`. |
| Did graph capture corrupt the old collective? | No evidence of graph corruption. Four eager forwards occur before capture, and the old kernel completed capture in the archived model run. | Proven for the observed run and call order. |
| Did the old RSAG hang at the first all-gather in the reproduced Kimi run? | No. The engine became ready; the later built-in generation warmup raised an inference-tensor mutation error in reduce-scatter. | Proven by the archived server log. |
| Does the old scalar-CAS lowering let different waves observe different loop results? | No. One thread performs the CAS, writes the old value to LDS, and all waves rendezvous and reload the same value. | Disproved by LLVM IR and AMDGCN. |
| Is an LDS/local fence sufficient to publish peer VMEM stores from every wave? | No. `lgkmcnt(0)` does not wait for other waves' VMEM stores. | Proven from the compiler source and ISA lowering. |
| Did that publication gap produce stale Kimi data in the observed run? | Not established. The cached old artifact also happened to contain an all-wave `vmcnt(0)` before the CAS loop, and all focused value checks passed. | Unproven as an incident cause; real as a source-contract gap. |
| Can the old per-payload-CTA protocol deadlock under independently scheduled peer grids? | The programming model gives it no forward-progress guarantee. A residency cycle is possible if matching CTA IDs are not resident together. | Structural risk, not reproduced on this system. |
| Why does the new protocol make progress? | Only an already-resident first or last local workgroup waits across ranks; no wait depends on an unscheduled local workgroup. | Algorithmic proof under the stated assumptions. |
| Is the committed four-wave refactor fully proved correct? | No. Its last-workgroup release sequence did not by itself aggregate stores from the other waves in each workgroup. | High-severity review finding, resolved here with one wave/workgroup. |
| Is rocSHMEM required? | No for the tested single-node direct-peer path. Standard Triton atomics plus PyTorch symmetric mappings are sufficient. | Implemented and tested. |

## Intuition first

### What token RSAG does

Attention data parallelism gives each rank only its local token rows. The dense
and MoE layers in these Kimi layouts instead divide their computation across all
eight GPUs. RSAG converts between those two layouts:

```text
rank r owns Nr token rows
        |
        | local TP1 attention
        v
token all-gather: every rank obtains all token rows
        |
        | each rank computes its TP8 or EP8 contribution
        v
token reduce-scatter: sum contributions and return rank r's Nr rows
```

The rows are uneven. During a large DP prefill one rank can own 8,192 rows while
the other seven own zero. With Kimi's hidden width of 7,168, the active rank has
58,720,256 elements.

### The old design: every tile is its own international checkpoint

The old kernel assigned one payload tile to each workgroup. After moving its
tile, workgroup `i` on every GPU performed a barrier specifically with
workgroup `i` on every other GPU.

An intuitive analogy is a fleet of couriers on each GPU:

1. Courier `i` delivers one box to every city.
2. Courier `i` refuses to leave until courier `i` in every other city reaches
   the same checkpoint.
3. Every city has a finite number of parking spaces for resident couriers.

This works when the hardware happens to admit matching courier IDs together.
It has no general guarantee when different cities admit different IDs. All
parking spaces can be occupied by couriers waiting for peers that have not yet
been scheduled.

The old host launcher made the imbalance worse by sizing every rank from the
largest rank's payload. For `[8192, 0, ...]`, all eight ranks launched 57,344
workgroups. Seven ranks did no useful data movement, but their 401,408 idle
workgroups still participated in per-tile barriers.

### The new design: workers finish, then one ambassador waits

The replacement separates local completion from cross-rank completion without
creating a second kernel launch:

1. A bounded rank-local grid strides through all local payload tiles.
2. Each worker records local completion and exits.
3. The last local worker becomes that rank's ambassador.
4. Only the eight ambassadors perform the cross-rank barrier.

For reduce-scatter, an already-resident first worker similarly becomes the
entry ambassador before any peer reads begin.

The critical progress property is simple: an ambassador never waits for local
work that is still queued. The entry ambassador is already resident because it
won the election. The exit ambassador cannot be elected until every other
local worker has finished.

## How the Kimi verification was initiated

The end-to-end verification used the dedicated TokenSpeed virtual environment
and the AMD harness changes from branch `fix-dp-kimi-agentic-harness` in the
integration worktree:

```bash
set -o pipefail
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
HF_HUB_CACHE=/data/models/hf/hub \
BENCH_CONFIGS="attn_dp8_moe_tp8 attn_dp8_moe_ep8" \
AGENTIC_MAX_TURNS=3 \
SERVER_PORT=8100 \
DIST_INIT_ADDR=127.0.0.1:4100 \
GPU_POLL_INTERVAL_SECONDS=10 \
./test/agentic_benchmark/kimi_k2.5/tokenspeed/agentic_bench.sh 2>&1 |
  tee /tmp/tokenspeed-kimi-dp8-agentic-centralized-final.log
```

The harness performs three phases per configuration:

1. **Startup:** launch `ts serve`, load target and draft models, run eager graph
   warmups, capture decode graphs, run TokenSpeed's built-in generation warmup,
   and wait for HTTP readiness.
2. **Serving:** run `evalscope perf` with two conversations at concurrency two,
   using dataset offset 68.
3. **Agentic sweep:** run the `(number, parallel)` pairs `(4,1)`, `(8,2)`,
   `(8,4)`, `(16,8)`, and `(32,16)`.

Immediately before each server launch, `wait_for_idle_gpus` refreshes both
utilization/VRAM and KFD ownership with `rocm-smi`. An eight-GPU configuration
does not launch unless it finds eight cards at zero utilization, zero allocated
VRAM, and with no KFD owner. The selected IDs are fixed in both
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

This later RSAG review intentionally did not rerun the full model. Every
eight-GPU kernel process was preceded by:

```bash
rocm-smi --showuse --showmemuse --showpids
```

and launched only when all eight cards were idle with no KFD PIDs. The focused
benchmark command was:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
export PYTHONPATH="$PWD/tokenspeed-kernel/python:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=/tmp/tokenspeed-rsag-one-wave-review-cache
torchrun --standalone --nproc-per-node=8 \
  test/runtime/distributed/benchmark_amd_rsag.py \
  --label one-wave-review-a --warmups 100
```

## Why DP8 + MoE TP8 and DP8 + MoE EP8 select RSAG

`ServerArgs.resolve_parallelism()` resolves `--data-parallel-size 8` on an
eight-rank job to attention TP1/DP8. Dense TP defaults to the full world size.
MoE is either TP8/EP1 or TP1/EP8:

| Configuration | Attention width | Dense width | MoE TP × EP width |
| --- | ---: | ---: | ---: |
| Attention DP8 + MoE TP8 | 1 | 8 | 8 × 1 |
| Attention DP8 + MoE EP8 | 1 | 8 | 1 × 8 |

`CommManager.use_all_reduce()` compares attention TP width with dense TP width
for dense layers and with MoE TP-times-EP width for MoE layers:

```python
if is_moe:
    return attn.tp_size == moe.tp_ep_size
return attn.tp_size == dense.tp_size
```

Both comparisons are `1 != 8`, so `pre_mlp_comm()` calls token all-gather and
`post_mlp_comm()` calls token reduce-scatter. Attention TP8 configurations have
matching widths and use ordinary all-reduce instead; they do not exercise token
RSAG in this target model.

Relevant source locations are:

- `python/tokenspeed/runtime/utils/server_args.py:440-494`
- `python/tokenspeed/runtime/distributed/comm_manager.py:128-131`
- `python/tokenspeed/runtime/distributed/comm_manager.py:200-270`

## Complete Kimi-to-kernel call tree

The model path is:

```text
KimiK25ForConditionalGeneration.forward
  python/tokenspeed/runtime/models/kimi_k25.py:917
  -> BaseCausalLM.forward
     python/tokenspeed/runtime/models/base/causal_lm.py:147
  -> DeepseekV3Model.forward
     python/tokenspeed/runtime/models/deepseek_v3.py:1423
  -> DeepseekV3DecoderLayer.forward
     python/tokenspeed/runtime/models/deepseek_v3.py:1311
  -> DeepseekV3DecoderLayer.forward_mlp
     python/tokenspeed/runtime/models/deepseek_v3.py:1363
```

The all-gather side is:

```text
CommManager.pre_mlp_comm
  -> pre_dense_comm or pre_moe_comm
  -> token_all_gather
     python/tokenspeed/runtime/distributed/comm_ops.py:288
  -> AutoBackend.token_all_gather
     python/tokenspeed/runtime/distributed/comm_backend/auto.py:75
  -> TritonRSAGBackend.token_all_gather
     python/tokenspeed/runtime/distributed/comm_backend/triton_rsag.py:106
  -> tokenspeed_kernel.ops.communication.triton.all_gather
  -> amd_rsag_all_gather
  -> amd_rsag_all_gather_kernel
       grid-stride peer stores
       -> _amd_rsag_complete_grid
       -> _amd_rsag_cross_rank_barrier
```

The return side is:

```text
CommManager.post_mlp_fused
  -> post_dense_comm or post_moe_comm
  -> token_reduce_scatter
     python/tokenspeed/runtime/distributed/comm_ops.py:305
  -> AutoBackend.token_reduce_scatter
     python/tokenspeed/runtime/distributed/comm_backend/auto.py:83
  -> TritonRSAGBackend.token_reduce_scatter
     python/tokenspeed/runtime/distributed/comm_backend/triton_rsag.py:115
  -> tokenspeed_kernel.ops.communication.triton.reduce_scatter
  -> amd_rsag_reduce_scatter
       stream-local copy into symmetric storage
  -> amd_rsag_reduce_scatter_kernel
       -> _amd_rsag_reduce_scatter_entry
       -> _amd_rsag_cross_rank_barrier
       grid-stride peer loads, sum, and local stores
       -> _amd_rsag_complete_grid
       -> _amd_rsag_cross_rank_barrier
```

Kimi's auxiliary MoE stream does not wrap RSAG. `forward_mlp()` performs the
pre-MLP gather before calling the MoE body and the post-MLP scatter after it.
The fork is inside the MoE body at
`python/tokenspeed/runtime/models/deepseek_v3.py:352`, and its join occurs before
that body returns.

During decode graph creation, `CudaGraphWrapper._capture_one()` calls the whole
forward four times eagerly at
`python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:466-475`. Actual
capture begins later at lines 501-502. A failure during this startup interval is
not automatically a graph-capture failure.

## The old algorithm

At `d4ffe44`, all-gather workgroup `i` did:

```text
load local tile i
store tile i into every rank's symmetric buffer
for every peer: CAS its signal[i][my_rank] from 0 to 1
for every peer: CAS my signal[i][peer] from 1 to 0
return
```

Reduce-scatter workgroup `i` did:

```text
per-workgroup entry barrier
load tile i from every peer and sum
store the local result
per-workgroup exit barrier
return
```

The host grid was:

```text
ceil(max(tokens_on_any_rank) * hidden_size / 1024)
```

on every rank. The signal pad therefore required
`grid_size * world_size` words per rank. At Kimi's largest skewed shape this was
57,344 workgroups and 458,752 signal words per rank.

For a grid of `B` workgroups and `P` ranks, a barrier had approximately
`B * P * (P - 1)` successful cross-device signal operations, plus polling. The
new centralized barrier has `2 * (P - 1)` cross-device signal operations per
phase and `B` rank-local completion RMWs.

## What the old scalar CAS actually lowered to

The source loops in `send_signal_to_peers()` and
`wait_signal_from_peers()` used a scalar `tl.atomic_cas` result as their loop
condition. AMD Triton does not let every lane execute that scalar CAS. It:

1. elects thread zero;
2. executes one system-scope compare/exchange;
3. writes the returned old value to LDS;
4. executes a workgroup barrier;
5. lets every lane reload that LDS value.

The generated shape is:

```text
all waves: local/LDS ordering barrier
wave 0, lane 0: system CAS
wave 0, lane 0: write returned old value to LDS
all waves: local/LDS barrier
all waves: read the same LDS value and test the loop condition
```

The lowering is visible in:

- `triton/lib/Conversion/TritonGPUToLLVM/Utility.cpp:1203-1247`
- `triton/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp:2198-2247`

The cached old artifact is:

```text
/home/ericfeng/distributed/.triton-cache/
R6GN22RS2TYOBFFCQ4W246WS2KW6R6RLX4HXDL6X6L3DVR7PADZA/
amd_rsag_all_gather_kernel.{ttir,llir,amdgcn}
```

Its AMDGCN contains `s_barrier`, elected
`global_atomic_cmpswap`, an LDS write, another `s_barrier`, an LDS read, and the
loop branch.

### Why “CAS completion diverges between waves” is wrong

The whole point of the LDS write and workgroup barrier is to make the CAS result
uniform. Every wave reloads the same scalar. Without evidence of a compiler or
hardware failure in that broadcast, divergent loop conditions are not a
defensible root cause.

The repeated barriers are expensive. They are not intrinsically divergent or
deadlocking when every wave reaches them.

## The `lgkmcnt` versus `vmcnt` question

The concern is valid, with an important evidence boundary.

Triton commit `c57bbbd8c1d83a8388baa508cf1286bfdad1695d` added CTA
barriers around scalar release/acquire atomics. However,
`insertAtomicOrderingBarriers()` requests only `AddrSpace::Local`:

```text
release atomic:
    local fence
    workgroup barrier
    local fence
    elected thread performs the system atomic
```

On AMD, the local fences lower to waits for LDS/GDS/scalar operations, typically
`s_waitcnt lgkmcnt(0)`. `LGKMCNT` does not track vector global-memory stores.
Those stores are tracked by `VMCNT`.

A system-release CAS executed by wave 0 can order wave 0's earlier operations.
It does not, by itself, make stores issued by waves 1-3 predecessors of that
release. To publish a multi-wave payload, every wave needs to drain the relevant
global writes, then the workgroup must rendezvous, and only then may the leader
signal:

```text
every wave: wait for its VMEM stores
all waves:  workgroup rendezvous
leader:     system-release signal
```

So, yes: a local-only `lgkmcnt` barrier is not the ordering operation wanted for
peer VMEM stores.

It still does not prove that this caused the alleged old failure. The examined
old all-gather artifact happens to contain `s_waitcnt vmcnt(0)` on every wave
before entering the CAS loop because the compiler also had to resolve nearby
global signal-pointer loads. That incidental instruction drains the earlier
stores in this artifact, but it is not a source-level contract on which the
kernel may rely. Small scheduling or compiler changes could remove or move it.

The precise conclusion is:

> The old source had no explicit, portable proof that all payload waves'
> peer stores happened-before its scalar release signal. This is a correctness
> gap. No stale read attributable to that gap was reproduced in the tested
> artifacts.

## The liveness issue in simpler terms

There are two different “waiting” concerns.

### Inside one workgroup

Think of one clerk polling the remote flag and writing the result on a shared
whiteboard. All four teams stop at the whiteboard, read the same answer, and
either retry together or leave together. This is wasteful because everyone
regroups on every poll, but it is intended to stay uniform.

`tl.atomic_poll` improves this part: one elected thread loops while the others
skip directly to a single rendezvous after polling succeeds. It removes a
whole-workgroup barrier from every unsuccessful iteration.

### Across independently scheduled GPU grids

The more serious structural risk is the courier scenario. Workgroup 17 on GPU 0
waits for workgroup 17 on every peer. If GPU 1's resident slots contain
workgroups 0-16 and 18 while workgroup 17 is queued, workgroup 17 on GPU 0 can
occupy a slot waiting for queued work. A sufficiently unlucky cycle across GPUs
can fill all resident slots with waiters.

The AMD, Triton, and PyTorch programming models do not promise matching
workgroup residency across independent device launches. Therefore the old
algorithm has no general liveness proof even though this hardware repeatedly
scheduled the tested grids compatibly.

Replacing only the CAS loop with `tl.atomic_poll` would not fix this
cross-workgroup dependency. The algorithm must change so cross-rank waiters do
not depend on unscheduled local work.

## What the focused reproduction actually found

The alleged old-kernel hang was tested without a model using both the production
TokenSpeed Triton compiler and the older
`tokenspeed-triton==3.8.10.post20260709` wheel.

The old implementation completed:

- world size two, eight workgroups, four waves, barrier only;
- world size two, eight workgroups, four waves, with peer payload stores;
- asymmetric active/idle ranks;
- 1,000 repeated barrier/payload iterations;
- exact Kimi geometry at world size two:
  `[8192, 0]`, hidden width 7,168, grid 57,344;
- exact Kimi geometry at world size eight:
  `[8192, 0, 0, 0, 0, 0, 0, 0]`, grid 57,344.

All checked values were correct. No old RSAG hang was reproduced.

That does not mathematically prove the old design safe; scheduling-sensitive
progress failures are difficult to force. It does mean that an “exact root cause
of the observed old hang” cannot honestly be supplied from the current
evidence. PC sampling, wave-state capture, or a deterministic failing artifact
would be needed to identify a stalled instruction if such a hang is observed
again.

## What actually failed in Kimi DP8

### DP8 + MoE TP8

An archived launch using the original monolithic RSAG reached:

```text
TokenSpeed engine ready
Generate request WARMUP_...
Prefill batch ...
```

It then failed in `amd_rsag_reduce_scatter()` at the host-side
`state.comm_buff.copy_(hidden_states)`:

```text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

The state had been lazily allocated while model forward ran under inference
mode, then reused by an idle/non-inference runtime path. This is a persistent
buffer lifetime bug, not a barrier hang. It is fixed separately by
`06d4938a1d8c9b52dd3739f7a9e4dbad56e06963` on branch
`fix-dp-kimi-persistent-buffer`.

With only that lifetime fix applied and the old monolithic RSAG left intact,
the same DP8 + TP8 configuration reached engine readiness, completed its
built-in generation warmup, and reported `SERVING`. This is direct
counterevidence to the claim that the old barrier necessarily hung during Kimi
startup.

### DP8 + MoE EP8

EP8 selects the same token RSAG because its combined MoE width is eight. It also
had an independent route-metadata defect: remote expert routes localized to
`-1` were incorrectly counted/sorted as local expert-zero work. That issue is
fixed on branch `fix-dp-kimi-moe-ep-routing`.

The later long-running EP8 failure was a device/access fault, not an RSAG signal
or value failure. The completed startup, serving, and lower-concurrency agentic
phases do demonstrate that the collective implementation is capable, but they
do not identify the later device fault's cause.

### Separate graph-layout issue

`CudaGraphWrapper._capture_one()` also has a separate DP correctness fix at
`python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:403-416`. Capture
must initialize `global_bs` consistently so frozen token offsets do not read
uninitialized symmetric memory during replay. That could produce NaNs, but it is
not the CAS/barrier issue and does not mutate the graph object.

## The new one-launch algorithm

### Signal-pad layout

Each rank reserves five `uint32` words:

| Offset | Meaning | Values |
| ---: | --- | --- |
| 0 | even-generation cross-rank bank | root arrival count or peer release flag |
| 1 | odd-generation cross-rank bank | root arrival count or peer release flag |
| 2 | local barrier phase | monotonically wrapping `uint32` |
| 3 | local payload-workgroup completion count | `0..num_programs` |
| 4 | reduce-scatter entry state | `0=idle`, `1=leader elected`, `2=ready` |

The old signal storage grew with payload size. The new storage is constant per
rank.

### Rank-local bounded grid

The launch size is:

```text
max(1, min(ceil(local_numel / 1024), compute_units))
```

Every workgroup grid-strides over additional tiles. An empty rank still launches
one workgroup so it participates in collective ordering, but its loop has zero
iterations and does not dereference a null payload pointer.

The grid does not need to be fully resident for correctness. Non-final
all-gather/exit workers never wait across ranks, so queued work can continue to
be admitted. The compute-unit cap bounds overhead and was the best tested
one-wave setting.

### All-gather

On each rank:

```text
for my grid-stride payload tiles:
    load local input tile
    store it into every peer's symmetric buffer slice

system-release increment local completion counter
if I am not last:
    return

acquire all earlier completion RMWs
perform one cross-rank barrier as this rank's ambassador
reset local completion state
return
```

The Python wrapper returns the populated prefix of the local symmetric buffer.

### Reduce-scatter entry

The Python wrapper first copies the full local TP/EP contribution into its
symmetric buffer on the same stream.

Inside the single Triton launch:

```text
try CAS(entry_state, idle -> elected)
if I won:
    perform the cross-rank entry barrier
    release-store ready
else:
    acquire-poll ready
```

The leader is necessarily resident because it executed the winning CAS.
Followers may occupy other resident slots, but the leader itself can progress
and every peer has an equivalent resident leader.

### Reduce-scatter payload and exit

After entry:

```text
for my grid-stride output tiles:
    load the matching slice from every peer
    accumulate in FP32
    store my reduced local output

system-release increment local completion counter
if I am not last:
    return

acquire all earlier completion RMWs
perform one cross-rank exit barrier
reset completion count and entry state
return
```

The exit barrier prevents any rank from reusing its symmetric input buffer while
another rank still reads it.

### Centralized double-buffered cross-rank barrier

One ambassador per rank:

1. relaxed-increments its local phase and selects `old_phase & 1`;
2. each non-root rank system-release-increments the selected bank on rank zero;
3. rank zero system-acquire-polls its bank for `world_size - 1`;
4. rank zero clears its bank and system-release-writes `1` to that bank on every
   non-root rank;
5. each non-root rank system-acquire-polls its local bank and clears it.

Two banks prevent an ABA/reuse race. A fast rank can be at most one completed
barrier ahead of a slow rank: rank zero cannot finish generation `N+1` without
the slow rank's `N+1` arrival, and that rank cannot issue the arrival until it
has consumed generation `N`. Therefore generation `N+2` cannot reuse the same
bank before generation `N` is consumed.

Only the low phase bit selects the bank, so `uint32` wrap is harmless. The test
suite initializes phase to `0xffffffff` and checks a complete
all-gather/reduce-scatter cycle across the wrap.

## Why one wave per payload workgroup is required

The initial one-launch refactor used four waves per workgroup and then relied on
each workgroup's completion-counter release to publish all its peer operations.
That claim was too strong.

With four waves:

```text
waves 0-3 issue peer stores
wave 0 lane 0 executes scalar system-release completion atomic
```

The atomic's AMD release sequence includes a `buffer_wbl2` and
`s_waitcnt vmcnt(0)` for the issuing wave. A wave wait is not a workgroup-wide
VMEM wait. The local barriers inserted around a scalar atomic use the local
address space and do not supply the missing global-store aggregation.

With one wave:

```text
the only wave issues every lane's peer stores
the same wave executes the system-release completion atomic
its vmcnt(0) drains every lane's outstanding VMEM operations
```

The proof no longer needs a cross-wave global-memory fence that Triton's current
source API cannot express.

The reviewed one-wave artifact reports `num_warps: 1`. Its AMDGCN replaces
workgroup `s_barrier` operations with wave-barrier markers and, immediately
before the completion atomic, emits:

```text
buffer_wbl2 sc0 sc1
s_waitcnt vmcnt(0)
global_atomic_add ... sc0 sc1
```

This is the exact publication edge needed by the algorithm.

## Memory-ordering proof

The proof assumes all ranks invoke the same collectives in the same order and
that one symmetric state is not used concurrently from independent streams.

### All-gather visibility

For every rank:

```text
same-wave peer stores
  happens-before system-release completion RMW
  happens-before last workgroup's acquire of the RMW release sequence
  happens-before that ambassador's system-release arrival/release
  synchronizes-with the remote ambassador's system acquire
```

For non-root ranks, the chain goes through rank zero:

```text
non-root release arrival
  -> root acquire
  -> root release to peers
  -> peer acquire
```

Rank zero's own payload completion is sequenced before its releases to peers.
When a rank leaves the barrier, every rank's peer stores are visible.

### Reduce-scatter entry visibility

The stream-local copy completes before the Triton kernel starts. The elected
leader's release arrival is therefore ordered after that copy. The cross-rank
acquire/release chain makes every rank's copy visible to every leader. The
leader then release-publishes local entry state `ready`, and follower workgroups
acquire-poll it before reading peer buffers.

### Reduce-scatter exit safety

Peer loads feed the accumulation and output store. The same wave's completion
release occurs after those operations. The last local workgroup acquires all
completion RMWs, and the cross-rank barrier prevents symmetric-buffer reuse
until every rank has finished reading.

## Liveness proof and failure model

The new design avoids the old peer-matched-workgroup dependency:

- **Reduce-scatter entry:** the elected leader is already resident.
- **Payload:** workers do not wait across ranks.
- **Completion:** the last worker is already resident, and all other local
  workers have finished.
- **Cross-rank barrier:** exactly one resident ambassador per rank waits.

Thus no ambassador needs an unscheduled local workgroup to make progress.

This is not fault tolerant. A crashed rank, mismatched collective order, or
device fault leaves peers polling indefinitely. `tl.atomic_poll` supports a
timeout, but the production path currently has no agreed device-to-host error
contract for propagating it. Adding a timeout plus a clear rank/phase diagnostic
is a worthwhile follow-up; silently continuing after timeout would be unsafe.

## Design and implementation review

### Findings resolved in this branch

#### High: four-wave completion publication was not proved

The initial refactor's completion comment claimed that one scalar release
covered all peer operations in a multi-wave workgroup. It did not. The launch is
now explicitly one AMD wave per workgroup, with the invariant documented next
to `_AMD_RSAG_NUM_WARPS`.

#### Medium: signal layout and helper intent were opaque

The single `_AMD_RSAG_STATE_OFFSET` and expressions such as `+ 1` obscured the
five-word protocol. The code now includes a layout table, names the local
offsets and entry-state values, renames `PEER_BLOCK_SIZE` to
`PEER_VECTOR_SIZE`, and renames `_amd_rsag_grid_finish()` to
`_amd_rsag_complete_grid()`.

#### Medium: rank-global launch sizing was pathological

The refactor sizes work from `local_numel`, caps the persistent grid at one
workgroup per compute unit, and retains one synchronization workgroup for empty
ranks.

### Design choices that are sound

- Dynamic entry-leader election is better than assuming program ID zero is
  admitted first.
- A last-arriver completion counter is better than a global sense flag because
  non-final workers can exit.
- System scope on completion RMWs makes the transitive publication proof
  straightforward. Reducing it to GPU scope would require a separate explicit
  system publication step and a careful proof.
- A relaxed phase increment is sufficient because exactly one ambassador per
  rank calls the cross-rank barrier at a time.
- Alternating banks are necessary; one bank is vulnerable to a fast rank
  publishing a later generation before a slow peer consumes the earlier
  release.
- A centralized barrier is reasonable at world size eight: only seven arrivals
  and seven releases cross devices per phase.

### Remaining improvement opportunities

1. **Timeout and diagnostics.** Add a supported collective timeout path that
   reports rank, phase, bank, entry state, and completion count before trapping.
2. **Larger world sizes.** The rank-zero coordinator is `O(P)` and can become a
   hotspot. A tree or hierarchical barrier should be considered only when
   measurements above the current eight-rank target justify it.
3. **Compiler support for multi-wave publication.** A Triton workgroup barrier
   with global-write ordering before scalar release, and global-read ordering
   after acquire, would allow safe multi-wave payload workgroups. The AMD
   lowering already understands global address-space barriers; the atomic
   ordering insertion currently requests only local.
4. **Launch tuning.** One workgroup per CU was the best retained setting.
   Four one-wave workgroups per CU increased contention. Two per CU was not run
   after the GPUs became occupied and benchmarking was deliberately concluded.
5. **Concurrent use contract.** The state is stream ordered, not a concurrent
   multi-collective object. An assertion or API-level statement would make that
   contract more visible.
6. **Fault isolation.** The later EP8 device fault remains outside this
   collective proof and needs its own reproducer if investigated.

## Why the alternatives are not preferable here

### Separate payload and barrier kernels

A dedicated one-wave barrier kernel is easy to reason about and uses the
existing PyTorch symmetric-memory barrier style, but it adds launches at every
MLP boundary. The user requirement is one launch, and the last-arriver design
achieves that without sacrificing the bounded parallel payload grid.

### Replace only the CAS loop with `tl.atomic_poll`

This removes repeated whole-workgroup barriers inside the polling loop. It does
not:

- aggregate multi-wave peer stores before a scalar signal;
- complete the whole local grid;
- remove the peer-matched workgroup residency dependency.

It is a useful primitive inside the new algorithm, not a complete algorithmic
fix by itself.

### PyTorch symmetric-memory `handle.barrier()`

PyTorch's native barrier launches a separate one-workgroup kernel on the current
stream. It is suitable between kernels but cannot be invoked midway through a
Triton payload kernel without splitting the launch.

### rocSHMEM / triton-shmem

rocSHMEM provides single-thread, wave, and workgroup barrier variants.
`barrier_all` also performs `quiet` for rocSHMEM-issued RMA. It still does not
automatically establish completion for an arbitrary multi-workgroup Triton grid,
and rocSHMEM `quiet` does not necessarily track direct stores issued through
PyTorch peer pointers.

A one-launch rocSHMEM version would still need the local first/last-leader
algorithm. For this single-node path, adding a vendor transport would increase
dependencies without closing a remaining semantic gap. rocSHMEM becomes
appropriate if the collective must cross nodes or use a transport where direct
PyTorch peer mappings are unavailable; then the ambassador can call a
rocSHMEM/triton-shmem device collective after local completion.

### One persistent workgroup for the whole payload

This has a simple proof but can underuse the 256-CU device on large Kimi
prefills. The retained bounded grid preserves parallel payload movement while
keeping only one cross-rank waiter.

## Focused correctness and performance evidence

### End-to-end capability evidence

The centralized one-launch algorithm, before the final one-wave hardening, was
used in the bounded end-to-end run described above:

- DP8 + MoE TP8 reached readiness in 130 seconds, completed the two-conversation
  serving phase, and completed agentic concurrency 1, 2, 4, 8, and 16. All
  204/204 turn requests in the second EvalScope phase succeeded.
- DP8 + MoE EP8 reached readiness in 195 seconds, completed the
  two-conversation serving phase, and completed agentic concurrency 1, 2, and 4.
  All 60/60 completed turn requests succeeded. Concurrency 8 had only entered
  its connection probe when the run was stopped, and concurrency 16 was not
  started.

These results show that the new synchronization structure can serve the target
model. They are not used as proof of the final one-wave memory edge: that edge
was checked at kernel level and in generated AMDGCN, and the full model was not
rerun after the launch-width change at the user's requested stopping boundary.

### Correctness coverage already completed

The one-launch implementation passed focused exact-value tests at world sizes
two, four, and eight, including:

- uniform and uneven token distributions;
- all-gather/reduce-scatter round trips;
- exact Kimi hidden width 7,168;
- `[8192, 0, ...]` and `[..., 0, 8192]`, so rank zero need not own payload;
- zero-token input/output pointers;
- alternating CUDA/HIP graph replays over shared persistent state;
- phase wrap from `0xffffffff`;
- old-kernel exact Kimi geometry with both the current and July 9 compiler
  wheels.

The one-wave follow-up was compiled and value-checked by the same world-size
eight benchmark. Its generated AMDGCN was inspected for the publication
sequence described above.

After the readability refactor, active-Kimi and zero-local-work specializations
of both final kernels were also compiled offline for
`GPUTarget("hip", "gfx950", 64)`. All reported `num_warps=1` and compiled
successfully without opening a KFD process. The final all-gather assembly emits
`buffer_wbl2`, `s_waitcnt vmcnt(0)`, then the system-scope completion
`global_atomic_add` at signal offset 12; reduce-scatter emits the same sequence.

### Performance

Times below are max-rank GPU event times on eight `gfx950` GPUs, hidden width
7,168. The old column is the mean of two warmed bracketing runs; the one-wave
column is the focused review run.

| Distribution | Old AG | One-wave AG | Old RS | One-wave RS |
| --- | ---: | ---: | ---: | ---: |
| 1 token/rank | 0.01542 ms | 0.01673 ms | 0.02937 ms | 0.02137 ms |
| 8 tokens/rank | 0.02056 ms | 0.01643 ms | 0.04019 ms | 0.02437 ms |
| 128 tokens/rank | 0.21397 ms | 0.06875 ms | 0.39065 ms | 0.10692 ms |
| `[8192, 0, ...]` | 16.00805 ms | 2.20650 ms | 24.01355 ms | 2.14170 ms |

One-token all-gather alone is approximately 1.3 microseconds slower, but the
all-gather plus reduce-scatter pair is faster in every tested shape. The
Kimi-idle shape improves by about 7.3× for all-gather and 11.2× for
reduce-scatter.

Increasing the one-wave grid to four workgroups per CU was rejected: 128-token
reduce-scatter rose to 0.18109 ms and Kimi-idle reduce-scatter to 3.28732 ms.
The extra workers contended for the same peer-memory bandwidth.

## What was not tested

Benchmarking was intentionally concluded after demonstrating capability. The
following remain explicit gaps:

- a full Kimi model rerun after the final one-wave launch change; prior
  end-to-end results used the same one-launch algorithm with four waves;
- two one-wave workgroups per CU;
- EP8 agentic concurrency 8 and 16 in the final bounded sweep;
- the dataset's normal 10-15 turns rather than the bounded three-turn run;
- captured prefill graphs (the AMD configurations disable prefill capture but
  retain decode capture);
- simultaneous collectives on independent streams sharing one RSAG state;
- multi-node transport;
- GPU architectures other than the tested single-node eight-`gfx950` system;
- rank failure, timeout, and recovery behavior;
- an upstream Triton global-address-space atomic-ordering barrier alternative.

## Final assessment

The old design should be replaced, but the reason must be stated accurately.

The investigation did not reproduce an old RSAG hang or stale data, and the
archived Kimi DP8 + TP8 failure was an inference-tensor lifetime error. The
earlier claims of CAS-result divergence and graph corruption are contradicted
by the lowering and execution order.

The old kernel still depended on incidental generated VMEM waits for multi-wave
publication, imposed a peer-matched-CTA scheduling dependency with no formal
progress guarantee, and performed enormous amounts of useless synchronization
for DP-idle ranks. Those are sufficient engineering reasons to refactor it.

The retained algorithm is general, one launch, and uses only standard Triton
over PyTorch symmetric-memory mappings. Its local first/last-leader protocol
removes the scheduling cycle; its centralized double-buffered barrier provides
reusable system-scope ordering; and its one-wave workgroups make each completion
release cover the entire workgroup payload. Under the documented collective
ordering and single-stream state-use assumptions, the new design has both a
memory-ordering proof and a forward-progress argument that the old design did
not.
