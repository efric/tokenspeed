# AMD Kimi K2.5 DP8 token RSAG verification

This report records the July 23, 2026 verification of Kimi K2.5 on eight
`gfx950` GPUs for:

- Attention DP8 + MoE TP8
- Attention DP8 + MoE EP8

The verification covers server startup before EvalScope, a two-conversation
serving phase, and the second EvalScope agentic sweep. The final-source TP8
sweep completed through concurrency 16. At the requested stopping boundary, the
final-source EP8 sweep had completed through concurrency 4; concurrency 8 and 16
are recorded below as remaining coverage. This report also documents the AMD
token all-gather/reduce-scatter (RSAG) investigation and the resulting
single-launch implementation.

The checkout was based on
`d4ffe4448c78c15bc0b1df801be04b0fd999be4f`. EvalScope outputs are under
`outputs/20260723_125025/`; the combined harness log is
`/tmp/tokenspeed-kimi-dp8-agentic-centralized-final.log`. The harness pinned
EvalScope at `acd09b44384d53174768bb1063f675420f76fae9`.

## Exact invocation and GPU reservation

The run used the TokenSpeed virtual environment and forced imports to come from
this checkout:

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

`AGENTIC_MAX_TURNS=3` was an explicit bound for this verification, not a change
to the dataset. Every conversation still begins with an approximately
50K-token prompt, generates 500 tokens per turn, and performs two cached
follow-up turns. Omitting the override preserves the dataset's normal 10--15
turn depth.

Immediately before the TP8 launch, the harness printed:

```text
card0,0,0,0,N/A
card1,0,0,0,N/A
card2,0,0,0,N/A
card3,0,0,0,N/A
card4,0,0,0,N/A
card5,0,0,0,N/A
card6,0,0,0,N/A
card7,0,0,0,N/A
No KFD PIDs currently running
Confirmed idle GPU(s) 0,1,2,3,4,5,6,7 immediately before launch
```

After TP8 stopped, the first EP8 check still saw 1--13% residual VRAM and did
not launch. Ten seconds later, all eight cards reported 0% utilization and 0%
allocated VRAM, with no KFD owners; only then did the harness launch EP8. The
selected IDs remain fixed through both `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` for the server lifetime.

## Why these two configurations use token RSAG

The runtime resolved the configurations as follows:

| Configuration | Attention | Dense | MoE | Width comparison |
| --- | --- | --- | --- | --- |
| DP8 + MoE TP8 | TP1 / DP8 | TP8 | TP8 / EP1 | `1 != 8` |
| DP8 + MoE EP8 | TP1 / DP8 | TP8 | TP1 / EP8 | `1 != 1*8` |

`CommManager.use_all_reduce()` compares attention TP width with dense TP width
or MoE TP-times-EP width. Both configurations therefore translate between
DP-local token rows and width-eight dense/MoE computation:

```text
DP rank owns Ni rows
  -> local TP1 attention
  -> token all-gather across eight ranks
  -> TP8 or EP8 MoE computation
  -> token reduce-scatter/sum
  -> the DP rank receives its Ni rows
```

The server logs independently confirm `allreduce is forbidden` for attention
TP1 versus dense TP8. In contrast, the Attention TP configurations have equal
attention and dense/MoE group widths and use ordinary all-reduce for this
model.

## What was actually wrong

The original AMD payload kernel made every multi-wave CTA perform its own
symmetric barrier. Its source-level scalar CAS loops lowered to:

```text
s_barrier
elect thread zero
global_atomic_cmpswap
store the returned value in LDS
s_barrier
reload the value and branch back to the loop
```

Thus each polling iteration became a four-wave, whole-workgroup rendezvous.
The old focused eager collective hung without graph capture, while an NCCL
control completed. Cached TTIR, LLVM IR, and AMDGCN all show the CAS/LDS/barrier
structure.

This evidence supports a liveness defect in the per-CTA symmetric-barrier
arrangement. It does **not** uniquely prove that CAS completion diverged between
waves; the compiler intended to broadcast one result through LDS. The exact
stalled-wave/backend event would require PC sampling or wave-state capture.
Also, the old kernel can complete with newer compiler builds, so the failure is
timing/compiler sensitive rather than a deterministic functional mismatch.

Calling this "multi-stream graph corruption" is inaccurate:

- `CudaGraphWrapper._capture_one()` executes four complete eager forwards
  before entering `torch.cuda.graph(...)`.
- The focused failure was also eager.
- Kimi forks an auxiliary stream inside the MoE body, but token all-gather is
  before that body and reduce-scatter is after its join.
- The collective can hang or leave shared data/signal state wrong, but it does
  not rewrite the captured graph object.

There was a separate graph-replay bug in which capture froze a wrong
`global_bs` layout and later read uninitialized symmetric memory. That bug is
documented in `cuda_graph_wrapper.py` and should not be conflated with the
barrier liveness defect.

## Single-launch solution

The replacement remains standard Triton over PyTorch symmetric memory. Each
all-gather or reduce-scatter uses one custom Triton kernel for its payload and
all synchronization; no separate barrier kernel and no rocSHMEM dependency are
required. The public Python path still performs the expected surrounding tensor
copy or safe-result clone when requested.

### Rank-local persistent grid

Each rank launches:

```text
max(1, min(ceil(local_numel / 1024), device_compute_units))
```

CTAs and grid-strides over its payload. An empty rank launches one
synchronization CTA but never dereferences its zero-length input or output.
This replaces the old host-maximum grid, under which a distribution such as
`[8192, 0, ...]` made every idle rank launch 57,344 entirely masked CTAs.

### Reduce-scatter entry

The first resident CTA to atomically change rank-local entry state from 0 to 1
becomes the leader. It performs the cross-rank entry barrier and publishes state
2 with release semantics. Other resident CTAs use `tl.atomic_poll(...,
acquire)` and then begin peer reads. Dynamic election avoids assuming that
program ID zero is scheduled first.

### Payload completion

Every payload CTA contributes an acquire-release increment to a local
completion counter and exits. The CTA that observes
`old_count == num_programs - 1` is necessarily last. Only that CTA performs the
cross-rank completion barrier and resets local state.

This is the key liveness property: a CTA never waits for the same program ID to
be resident on every peer. Cross-device waiting starts only after all other
local payload CTAs have completed and released their resources.

### Reusable cross-rank barrier

One leader per rank uses two alternating 32-bit signal banks and a local phase
word:

1. Atomically advance the local phase and select `old_phase & 1`.
2. Each non-root leader release-increments that bank on rank zero once.
3. Rank zero acquire-polls its bank for `world_size - 1`. The RMW release
   sequence makes this acquire observe every rank's completed local payload.
4. Rank zero resets its bank and release-exchanges `1` into the selected bank
   on all peers with one masked vector atomic.
5. Each peer acquire-polls its local bank and resets it to zero.

The alternate bank prevents generation `N + 2` from overwriting an unconsumed
generation `N` signal. Rank zero cannot finish generation `N + 1` until every
peer has entered it, and entering `N + 1` requires that peer to consume and
reset `N`; therefore a bank is clean before reuse. Only the low phase bit
selects a bank, so uint32 wrap is harmless and is covered by a direct test.

At DP8 this reduces a barrier from 128 cross-device RMWs in the first
all-to-all counter design to 14: seven arrivals and seven release signals. It
also keeps the required transitive memory ordering: release-RMW sequence,
rank-zero acquire, rank-zero release, peer acquire.

The implementation requires `tokenspeed-triton` from July 21, 2026 or later:

- `tl.atomic_poll` elects one thread to spin and rendezvouses the CTA once after
  success. Generated AMDGCN contains only a system-scope load/compare/branch in
  the loop and one `s_barrier` after it.
- Triton commit `c57bbbd8c1` inserts CTA ordering barriers for release/acquire
  atomics. This makes a scalar leader's system-scope release order stores from
  all payload waves, rather than only its own thread.

### Other required DP fixes

The complete model path also required independent fixes:

- Persistent symmetric buffers are explicitly created outside inference mode,
  even when lazy initialization is reached from an inference-mode model
  forward. This prevents a later non-inference `copy_` from failing.
- EP localization sorts invalid `-1` remote routes after valid local experts,
  excludes them from ragged metadata, and zeros their unscheduled intermediate
  rows before separate activation and their output rows before reduction.
- Rank-local launch sizing handles zero-token ranks without null-pointer
  payload launches.

## Alternatives considered

Replacing the handwritten CAS loop with `tl.atomic_poll` was necessary but not
sufficient by itself. Keeping one peer-matched barrier per payload CTA can
still deadlock when independently scheduled cross-device grids do not have
matching CTAs resident. The last-CTA leader algorithm removes that dependency.

A single persistent CTA striding over the entire payload would also make
progress, but leaves substantial bandwidth unused for Kimi's large prefill
geometry. A host-side PyTorch or rocSHMEM barrier around a payload kernel would
restore progress but reintroduce the split launch the change is intended to
avoid. The current `triton-shmem` collective wrappers likewise call
driver-side `barrier_all()` around Triton payload launches. Direct rocSHMEM
device calls could implement a one-launch version, but add a vendor dependency
without solving a remaining semantic gap here.

## Focused correctness and compiler verification

The final focused GPU suite ran with a clean eight-GPU reservation:

```bash
pytest -q \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops_kimi_dp_idle_rank \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops_cuda_graph \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops_inference_initialized_state
```

Result: `10 passed in 133.55s`.

The direct `tokenspeed-kernel` world-size-four test also passed and explicitly
initialized the phase word to `0xffffffff` before an all-gather/reduce-scatter
cycle. It finished with both banks, the local completion count, and the entry
state reset to zero and the wrapped phase equal to two.

Coverage includes:

- exact all-gather and reduce-scatter values at world sizes 2, 4, and 8;
- Kimi's hidden size 7,168 with both `[8192, 0, ...]` and
  `[..., 0, 8192]`, proving rank zero need not own payload;
- three graph capture streams sharing one symmetric workspace;
- 20 alternating graph replays with changed inputs and ragged distributions;
- state created under inference mode and mutated outside it;
- EP route metadata and rank-local grid unit tests.

Generated AMDGCN was also inspected. `atomic_poll` loops contain no
`s_barrier`, LDS broadcast, or global atomic; the one rendezvous occurs only on
success. For the same WS8 all-gather shape, the centralized version contains
12 finite `s_barrier` instructions and 8 global atomics, versus 44 and 36 in
the earlier all-to-all counter implementation. The remaining barriers are
finite ordering/rendezvous points outside poll loops.

## Performance

The benchmark reports minimum, mean, and maximum GPU event time across all
eight ranks so an idle rank cannot disappear behind rank-zero timing. The
table is the mean of two warmed runs' max-rank times:

| Distribution, hidden 7,168 | Original AG | Final AG | Original RS | Final RS |
| --- | ---: | ---: | ---: | ---: |
| 1 token/rank | 0.015419 ms | 0.016622 ms | 0.029367 ms | 0.021228 ms |
| 8 tokens/rank | 0.020558 ms | 0.015659 ms | 0.040185 ms | 0.022897 ms |
| 128 tokens/rank | 0.213965 ms | 0.070680 ms | 0.390645 ms | 0.096799 ms |
| `[8192, 0, ...]` | 16.008052 ms | 2.274622 ms | 24.013551 ms | 2.246297 ms |

The only individual-operation regression is 1.20 microseconds (7.8%) for
one-token all-gather. The all-gather plus reduce-scatter pair that surrounds an
MLP is still 15.5% faster at one token/rank, 36.5% faster at eight, 72.3% faster
at 128, and 88.7% faster for the Kimi idle-rank prefill. A 100-repetition
`[..., 0, 8192]` run measured 2.218 ms all-gather and 2.140 ms
reduce-scatter, showing that the rank-zero coordinator does not need local
payload. The large-shape improvement comes from bounded grid-striding and one
cross-rank leader instead of a barrier in every payload CTA.

## End-to-end results

All values below use three turns per conversation and 500 generated tokens per
turn.

### Attention DP8 + MoE TP8

Startup reached readiness in 130 seconds. It loaded target and draft
checkpoints, captured decode batch sizes 1 and 2 after eager warmups, completed
the built-in generation warmup, and exposed the HTTP endpoint.

The serving phase completed 6/6 requests with zero failures in 518.82 seconds
(600.30 total tokens/s, 5.78 output tokens/s).

| Sweep concurrency | Turn requests | Failed | Duration | Total tok/s | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0 | 1,936.51 s | 321.84 | 3.10 |
| 2 | 24 | 0 | 2,049.00 s | 608.30 | 5.86 |
| 4 | 24 | 0 | 1,161.28 s | 1,073.19 | 10.33 |
| 8 | 48 | 0 | 1,246.69 s | 2,000.10 | 19.25 |
| 16 | 96 | 0 | 1,780.74 s | 2,799.63 | 26.96 |

Across the second EvalScope phase, all 204/204 turn requests succeeded.

### Attention DP8 + MoE EP8

Startup reached readiness in 195 seconds and passed the same load, eager
warmup, decode capture, internal generation, gRPC, and HTTP checks.

The serving phase completed 6/6 requests with zero failures in 493.26 seconds
(631.39 total tokens/s, 6.08 output tokens/s).

| Sweep concurrency | Turn requests | Failed | Duration | Total tok/s | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0 | 1,920.86 s | 324.46 | 3.12 |
| 2 | 24 | 0 | 2,118.47 s | 588.36 | 5.66 |
| 4 | 24 | 0 | 1,081.67 s | 1,152.18 | 11.09 |
| 8 | not completed | -- | -- | -- | -- |
| 16 | not started | -- | -- | -- | -- |

Across the completed EP8 sweep loads, all 60/60 turn requests succeeded. The
harness automatically entered concurrency 8 during EvalScope's five-second
transition after the concurrency-4 summary. It passed the connection probe, but
was intentionally interrupted at `0/16` reported conversations before a valid
benchmark summary; concurrency 16 was never started. These are not counted as
passes.

That requested interrupt gave the outer `tee` pipeline status 130. The shutdown
audit found that the detached server group had outlived the pipeline, so it was
terminated by its exact process-group ID; ports 8100/4100 disappeared, all KFD
owners cleared, and all eight GPUs returned to 0% utilization and 0% VRAM. The
harness now has explicit INT/TERM handlers, tests process-group rather than
leader-PID liveness, and handles the brief pre-`setsid` launch race.

Neither configuration logged a collective hang, NaN, device-access fault, or
failed request during a completed phase. The only server-log errors were HTTP
503 responses from readiness probes before `ts serve ready` and Starlette
`CancelledError` traces during server shutdown.

## Remaining end-to-end coverage

The final centralized implementation has not yet completed:

- EP8 agentic concurrency 8 or 16;
- the dataset's normal 10--15-turn depth (this verification explicitly used
  three turns);
- captured prefill graphs, because the AMD benchmark configuration disables
  prefill capture while retaining decode graphs;
- multi-node execution or a GPU architecture other than this single-node,
  eight-`gfx950` system;
- execution of the modified CI manifest. Its `amd-mi35x-4gpu-test` runner only
  permits the analogous DP4 + MoE TP4 RSAG profile; the YAML was parsed locally
  but the CI job was not submitted from this worktree.

The full TP8 sweep, EP8 startup and serving, EP8 concurrency 1/2/4, focused
world-size 2/4/8 correctness tests, graph replay tests, both Kimi idle-rank
orientations, and the original-versus-final microbenchmark bracket were
completed on the final source.
