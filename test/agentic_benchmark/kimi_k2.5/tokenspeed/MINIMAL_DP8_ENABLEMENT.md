# Minimal AMD Kimi K2.5 attention-DP enablement

## Result

The persistent symmetric-buffer lifetime fix was tested by itself before any
MoE routing or RSAG algorithm change.

- Attention DP8 + MoE TP8 passed startup, serving, and every point of the
  bounded agentic sweep with only that production fix.
- Attention DP8 + MoE EP8 passed startup, serving, and agentic concurrency 1,
  2, and 4 with only that production fix. It then encountered a GPU nil-address
  fault shortly after concurrency 8 began.

This proves that the persistent-buffer patch fixes the original deterministic
DP startup/idle-rank exception in both profiles and is sufficient for the full
bounded TP8 workload. It does not prove completion of the EP8 sweep.

The most minimal production candidate is therefore:

```text
fix-dp-kimi-minimal-enablement
  persistent symmetric-buffer lifetime
  benchmark/configuration support
  focused regression and four-GPU CI proxy
```

It deliberately retains both the original AMD token RSAG algorithm and the
original zero-gated EP route algorithm. The alternatives are isolated:

- `fix-dp-kimi-mxfp4-route-followup` at `893736b` excludes remote MXFP4
  routes from local GEMM work.
- `fix-dp-kimi-rsag-hang` at `441911a` contains the one-launch RSAG refactor
  and its corrected analysis.

Because model benchmarking was intentionally concluded after the latest round,
this document does not claim that the exact final branch passes EP8 concurrency
8 or 16, or the unbounded agentic workload.

## Commit organization

Each concern is an independently signed-off commit:

| Commit | Area | Purpose |
| --- | --- | --- |
| `b5db054` | Runtime correctness | Allocate cached AMD RSAG state outside inference mode. |
| `8a926b3` | Benchmark config | Define AMD DP8 + MoE TP8 and DP8 + MoE EP8 profiles. |
| `574bba0` | Benchmark harness | Pin EvalScope and make selection, ports, checkout, and cleanup reproducible. |
| `3b6c4ce` | GPU reservation | Require the same eight GPUs to remain idle and owner-free before launch. |
| `48b6c1c` | CI config | Exercise the attention-DP token-collective path on the four-GPU AMD runner. |
| `c52c818` | Evidence provenance | Record the exact source revision and clean/dirty state in future benchmark logs. |

Only `b5db054` changes production behavior. No EP routing change, split
synchronization kernel, one-launch RSAG rewrite, rocSHMEM dependency, RCCL
fallback, or direct runtime dependency on another kernel package is present.

### Minimality ledger

| Candidate removal | Evidence | Decision |
| --- | --- | --- |
| Persistent-buffer fix | Without it, cached state created in inference mode is later mutated outside inference mode and PyTorch raises before reduce-scatter launches. Its focused WS2/WS4 regression passes on this branch. | Required for both profiles. |
| EP route exclusion | The old path maps remote `-1` routes to expert zero, but their gate is zero. Both GEMMs write every row and the zero-gated result is scattered before top-k reduction. Old and excluded paths were bitwise equal at exact Kimi geometry; the old path passed 1,000 repeats without an access fault. | Excluded from the minimum; retained as performance and non-finite-value hardening. |
| RSAG refactor | The original RSAG completed the full TP8 bounded run and EP8 startup/serving/lower-concurrency work. Focused old-RSAG Kimi-skew cases completed with correct values, and no observed model failure was localized to it. | Excluded from the minimum; retained as general liveness/performance hardening. |

The production allocation and its regression at branch head are byte-for-byte
identical to the persistent-only model state at `3b6c4ce`. Every later commit
changes only CI, evidence capture, or documentation. Thus the preserved
persistent-only model runs exercise the same production implementation present
at branch head.

## The persistent-buffer defect

Decode-graph startup runs four eager model forwards before graph capture.
`ModelExecutor._forward_step()` runs under `@maybe_inference_mode()`, so the
first token collective used to lazily create its persistent symmetric workspace
while PyTorch inference mode was active:

```text
ts serve
  -> Engine.initialize
  -> Scheduler.initialize
  -> ModelExecutor.initialize
  -> CudaGraphWrapper._capture_one
  -> four eager model forwards
  -> decoder MLP communication
  -> token_all_gather / token_reduce_scatter
  -> TritonRSAGBackend._get_or_create
  -> amd_create_rsag_state
  -> torch.distributed._symmetric_memory.empty
```

That made `comm_buff` an inference tensor. The state was cached beyond startup.
An idle DP rank could later run `execute_idle_forward()` outside inference mode,
where reduce-scatter stages data with:

```python
state.comm_buff[:total_num_tokens, :].copy_(hidden_states)
```

PyTorch rejected the mutation before any Triton kernel launch:

```text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

Commit `b5db054` allocates and rendezvouses the workspace through the existing
`_alloc_symm()` helper. That helper creates persistent mutable state under:

```python
with torch.inference_mode(False), torch.no_grad():
    tensor = symm_mem.empty(...)
```

This changes lifetime semantics only. Shape, process group, peer mappings, and
the original collective algorithm remain unchanged.

## Persistent-only model evidence

Both runs in this section used `3b6c4ce`: the persistent production fix plus
benchmark/configuration commits. They did not contain the EP route fix or an
RSAG algorithm change.

### Attention DP8 + MoE TP8

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
cd /home/ericfeng/distributed/.worktrees/tokenspeed/fix-dp-kimi/test/agentic_benchmark/kimi_k2.5/tokenspeed

BENCH_CONFIGS=attn_dp8_moe_tp8 \
AGENTIC_MAX_TURNS=3 \
GPU_POLL_INTERVAL_SECONDS=10 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8127 \
DIST_INIT_ADDR=127.0.0.1:4217 \
./agentic_bench.sh 2>&1 |
  tee /tmp/tokenspeed-kimi-persistent-only-tp8-full-20260724.log
```

Startup and graph initialization passed. Serving passed 6/6 requests. The
bounded agentic sweep produced:

| Concurrency | Successful | Failed | Duration (s) | Average latency (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0 | 2108.85 | 175.74 |
| 2 | 24 | 0 | 2387.79 | 194.41 |
| 4 | 24 | 0 | 1360.70 | 225.22 |
| 8 | 48 | 0 | 1411.79 | 224.88 |
| 16 | 96 | 0 | 2450.56 | 375.57 |

This is direct model evidence that the persistent-buffer production fix alone
can carry the TP8 profile through the tested startup, serving, and sweep.

### Attention DP8 + MoE EP8

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
cd /home/ericfeng/distributed/.worktrees/tokenspeed/fix-dp-kimi/test/agentic_benchmark/kimi_k2.5/tokenspeed

BENCH_CONFIGS=attn_dp8_moe_ep8 \
AGENTIC_MAX_TURNS=3 \
GPU_POLL_INTERVAL_SECONDS=10 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8128 \
DIST_INIT_ADDR=127.0.0.1:4218 \
./agentic_bench.sh 2>&1 |
  tee /tmp/tokenspeed-kimi-persistent-only-ep8-full-20260724.log
```

Startup passed, serving passed 6/6, and agentic concurrency 1, 2, and 4
completed without request failures. Early in concurrency 8, KFD reported:

```text
Memory access fault by GPU node-2 ... on address (nil)
Fatal Python error: Aborted
```

The inference-tensor mutation exception did not recur. This is evidence that
the persistent defect was resolved, not evidence that the complete EP8 profile
was accepted.

## Why EP route exclusion is a follow-up

In EP8, global top-k expert IDs owned by other ranks localize to `-1`. The old
MXFP4 route builder converted negative IDs to expert zero before sorting and
counting. For example:

```text
localized routes = [[-1, 0, 1], [-1, -1, -1]]
old local counts = [5, 1]
excluded counts  = [1, 1]
```

That metadata is inefficient, but calling it incorrect was too strong. The
complete old path is:

```text
remote route
  -> local id -1 and gate 0
  -> safe id 0 for metadata
  -> both ragged GEMMs compute local expert 0
  -> down projection multiplies the finite result by gate 0
  -> scatter writes zero to the original route
  -> top-k sum is unchanged
```

Because the old expert counts sum to every top-k row, neither GEMM leaves an
unscheduled output row. Explicit clearing becomes necessary only after the
optimization removes remote rows from the ragged schedule. For finite
intermediates, the original and excluded algorithms are mathematically
equivalent. Exclusion additionally prevents a hypothetical non-finite
expert-zero result from surviving `0 * NaN`, but no such value was observed.

The exact Kimi geometry was tested on gfx950:

```text
tokens               8,192
top-k                    8
hidden size          7,168
MoE intermediate     2,048
global experts          384
local experts            48
EP width                  8
```

The excluded path scheduled 8,193 local rows in the deterministic test
distribution, including 171 on expert zero. The original path scheduled all
65,536 top-k rows, including 57,514 on expert zero. Both returned finite,
bitwise-identical tensors. The original geometry then completed 1,000
iterations with bitwise-stable output, a 15.161 ms median, and no memory-access
fault.

This does not prove that no workload can fail, but it disproves the claim that
the old metadata is inherently semantically wrong or that the large
expert-zero slice alone reproduces the node-2 fault. Since algorithmic
inefficiency is acceptable for the minimal enablement, the route exclusion is
kept on `fix-dp-kimi-mxfp4-route-followup`.

## EP8 follow-up run with route exclusion

For attribution only, a later run used the persistent fix plus route exclusion
while retaining the original RSAG:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
cd /home/ericfeng/distributed/.worktrees/tokenspeed/fix-dp-kimi/test/agentic_benchmark/kimi_k2.5/tokenspeed

BENCH_CONFIGS=attn_dp8_moe_ep8 \
AGENTIC_MAX_TURNS=3 \
GPU_POLL_INTERVAL_SECONDS=10 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8129 \
DIST_INIT_ADDR=127.0.0.1:4219 \
./agentic_bench.sh 2>&1 |
  tee /tmp/tokenspeed-kimi-minimal-ep8-route-full-20260724.log
```

Startup and graph initialization passed, serving passed 6/6, and agentic
concurrency 1 through 8 completed:

| Concurrency | Successful | Failed | Duration (s) | Average latency (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0 | 2136.02 | 178.00 |
| 2 | 24 | 0 | 2200.61 | 182.13 |
| 4 | 24 | 0 | 1260.21 | 204.57 |
| 8 | 48 | 0 | 1736.52 | 267.58 |

At concurrency 16, five first-turn requests completed and then device work
stopped retiring. All scheduler processes remained alive and all eight GPUs
stayed busy. There was no KFD fault, Python abort, TCPStore loss, or OOM.
ROCgdb found the rank-zero host waiting for a HIP event, with resident
wavefronts reported in `__amd_rocclr_copyBuffer.kd`.

This proves a device-side progress stop. It does not identify the operation that
created the dependency chain, and in particular does not prove that the
original RSAG was stalled. Focused old-RSAG tests—including exact Kimi-skew
WS2/WS8 cases and repeated value checks—completed correctly, so the RSAG
refactor is not attributed as the fix for this observation.

Likewise, one route-excluded run crossing the earlier concurrency-8 window does
not prove that route exclusion fixed the prior node-2 fault. The old
exact-geometry route stress did not reproduce that fault, and the excluded run
encountered a different progress stop at concurrency 16.

## GPU reservation and benchmark phases

Immediately before each profile launch, `agentic_bench.sh` samples:

```bash
rocm-smi --showuse --showmemuse --csv
rocm-smi --showpidgpus
```

A GPU is eligible only when utilization and allocated VRAM are zero and it has
no KFD owner. The same eight IDs must remain eligible for
`GPU_IDLE_STABLE_CHECKS` consecutive samples. Only then are those exact IDs
passed in both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

The harness prints the repository path, exact Git revision, and clean/dirty
state at the start of every future run so its source provenance is captured in
the benchmark log.

For each selected configuration, the harness runs:

1. `ts serve` startup, eager graph warmups, graph capture, built-in generation
   warmup, and readiness polling;
2. a six-request serving check through `evalscope perf`; and
3. the agentic `(number, parallel)` points `(4,1)`, `(8,2)`, `(8,4)`,
   `(16,8)`, and `(32,16)`.

`AGENTIC_MAX_TURNS=3` bounded the recorded model runs. Omit it for the original
10-to-15-turn dataset.

## Four-GPU CI proxy

The AMD EvalScope CI runner has four GPUs. Its configuration uses attention
TP1/DP4, dense TP4, and MoE TP4. Since attention width one differs from the
dense/MoE width four, it selects token all-gather/reduce-scatter for the same
reason as DP8 + MoE TP8.

This is a semantic-path proxy, not DP8 acceptance. It does not exercise EP
route exclusion, which is intentionally outside this branch.

## Exact-branch focused verification

After removing both follow-up algorithms from this branch, the production
regression was rerun:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m pytest -q \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops_inference_initialized_state
```

Before the GPU command, `rocm-smi --showuse --showmemuse --showpidgpus`
reported all eight GPUs at zero utilization, zero allocated VRAM, and no KFD
owners. Both WS2 and WS4 persistent-state cases passed.

Both configuration scripts and the harness passed `bash -n`, the CI YAML
parsed successfully, and `pre-commit run --all-files` passed. The route
follow-up independently passed its three host tests and gfx950 production-path
test before its two signed-off commits were created.

## Evidence hashes

```text
b0b847ada1ed4b863cb859beaf57d9c5d26850705d8b8d9680ac092a9a893b1a  /tmp/tokenspeed-kimi-persistent-only-tp8-full-20260724.log
5f8366590fd4e569ef71e63bbffa421fabd51ea23f1ed327b2fa7ba10be1b5de  /tmp/tokenspeed_server_attn_dp8_moe_tp8.log
dc165603f0e27a389c2f94d12cb00510cc8b741e261ac13b66f2477029bbbf64  /tmp/tokenspeed-kimi-persistent-only-ep8-full-20260724.log
0dc60cec144970fc9decb9a307359fae8b3e03f341c5f79ac9ee93e8d48b9b11  /tmp/tokenspeed_server_attn_dp8_moe_ep8-persistent-only-fault-20260724.log
b5bff6bbebcaf9efaab0d2eecb7542c910f8106855430db0a9141f2987c6376e  /tmp/tokenspeed-kimi-minimal-ep8-route-full-20260724.log
92a53ca47580d7a7c93a25cf036ca461f7e0bbc37e0f82ca8ac5cad76f836347  /tmp/tokenspeed_server_attn_dp8_moe_ep8.log
```

## Explicit acceptance gap

The following remain untested on the exact final minimal branch:

- EP8 agentic concurrency 8 completion;
- EP8 agentic concurrency 16 completion;
- the uncapped 10-to-15-turn agentic dataset;
- long-duration replay or soak behavior;
- execution of the updated four-GPU CI job; and
- a controlled diagnosis of the EP8 concurrency-16 device progress stop.

TP8 was tested through every bounded point on the production-equivalent
persistent-only state. EP8 persistent-only evidence reaches concurrency 4.
The route-excluded follow-up reaches concurrency 8, but is not used as proof
that this smaller branch reaches that point.

When model benchmarking resumes, the exact acceptance commands are:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
cd /home/ericfeng/distributed/.worktrees/tokenspeed/fix-dp-kimi/test/agentic_benchmark/kimi_k2.5/tokenspeed

BENCH_CONFIGS=attn_dp8_moe_tp8 \
GPU_POLL_INTERVAL_SECONDS=10 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8127 \
DIST_INIT_ADDR=127.0.0.1:4217 \
./agentic_bench.sh

BENCH_CONFIGS=attn_dp8_moe_ep8 \
GPU_POLL_INTERVAL_SECONDS=10 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8128 \
DIST_INIT_ADDR=127.0.0.1:4218 \
./agentic_bench.sh
```

Do not set `AGENTIC_MAX_TURNS` for final unbounded acceptance.
