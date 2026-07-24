# Minimal AMD Kimi K2.5 attention-DP enablement

## Result

The persistent symmetric-buffer lifetime fix was tested by itself before any
MoE routing or RSAG algorithm change.

- Attention DP8 + MoE TP8 passed startup, serving, and every point of the
  bounded agentic sweep with only that production fix.
- Attention DP8 + MoE EP8 passed startup, serving, and agentic concurrency 1,
  2, and 4 with only that production fix. It then encountered a GPU nil-address
  fault shortly after concurrency 8 began.
- With the independent MXFP4 EP route-metadata fix added, EP8 passed
  concurrency 8. At concurrency 16 it later stopped making device progress.

The first item proves that the persistent-buffer fix is sufficient for the
requested TP8 profile and fixes the original deterministic DP startup/idle-rank
exception. The EP8 results show that the persistent defect was fixed there too,
but do not prove a complete EP8 sweep.

The most minimal candidate branch is therefore:

```text
fix-dp-kimi-minimal-enablement
  persistent symmetric-buffer lifetime
  benchmark/configuration support
  MXFP4 invalid EP route exclusion
  focused tests and four-GPU CI proxy
```

It deliberately retains the original AMD token RSAG algorithm. The canonical
isolated one-launch follow-up is `fix-dp-kimi-rsag-hang` at `441911a`; its
analysis is in `amd_rsag_refactor_analysis.md` on that branch. The earlier
integrated snapshot is retained as `fix-dp-kimi-one-launch-candidate` only for
provenance.

Focused investigation found structural forward-progress and publication risks
in the old algorithm, but did not reproduce an old-kernel hang or localize
either observed model failure to an RSAG instruction. It is consequently a
valuable general hardening and performance follow-up, not a proven prerequisite
for this minimal patch.

Because model benchmarking was intentionally concluded after the latest round,
this document does not claim that the exact final branch passes EP8 concurrency
16 or the unbounded agentic workload.

## Commit organization

Each concern is an independently signed-off commit:

| Commit | Area | Purpose |
| --- | --- | --- |
| `b5db054` | Runtime correctness | Allocate cached AMD RSAG state outside inference mode. |
| `8a926b3` | Benchmark config | Define AMD DP8 + MoE TP8 and DP8 + MoE EP8 profiles. |
| `574bba0` | Benchmark harness | Pin EvalScope and make selection, ports, checkout, and cleanup reproducible. |
| `3b6c4ce` | GPU reservation | Require the same eight GPUs to remain idle and owner-free before launch. |
| `f9819a3` | EP correctness | Exclude invalid remote MXFP4 routes and clear unscheduled rows. |
| `85301b5` | EP test | Exercise the production MXFP4 EP path on gfx950. |
| `7744a61` | CI config | Exercise the attention-DP token-collective path on the four-GPU AMD runner. |
| `33ef202` | Evidence provenance | Record the exact source revision and clean/dirty state in future benchmark logs. |

Only `b5db054` and `f9819a3` change production behavior. The latter is
EP-specific and is not needed by the TP8 profile. No split synchronization
kernel, one-launch RSAG rewrite, rocSHMEM dependency, RCCL fallback, or direct
runtime dependency on another kernel package is present.

### Minimality ledger

| Candidate removal | Evidence | Decision |
| --- | --- | --- |
| Persistent-buffer fix | Without it, cached state created in inference mode is later mutated outside inference mode and PyTorch raises before reduce-scatter launches. Its focused WS2/WS4 regression passes on this branch. | Required for both profiles. |
| EP route fix | The old metadata maps remote `-1` routes to expert zero and counts `[5,1]` instead of `[1,1]` in the focused reproducer. The production gfx950 path matches its reference after the fix. | Required for correct EP8 routing; irrelevant to TP8. |
| RSAG refactor | The original RSAG completed the full TP8 bounded run and EP8 through concurrency 8. Focused old-RSAG Kimi-skew cases also completed with correct values. No observed model failure was localized to it. | Excluded from the minimal branch; retained as follow-up hardening. |

The production source at branch head is byte-for-byte identical to `85301b5`.
Later commits change only CI configuration and documentation. The persistent
allocation and regression files are unchanged from the persistent-only model
state at `3b6c4ce`. Thus the preserved model runs exercise the same production
implementations that are present at branch head.

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

## Why the EP route fix is a separate correctness change

In EP8, global top-k expert IDs owned by other ranks localize to `-1`. The old
MXFP4 route builder converted negative IDs to expert zero before sorting and
counting. For example:

```text
localized routes = [[-1, 0, 1], [-1, -1, -1]]
old local counts = [5, 1]
correct counts   = [1, 1]
```

Zero gate weights can hide a small output symptom, but ragged GEMM still treats
those entries as real expert-zero rows. At EP8, most of a rank's route table can
be remote. Worse, rows excluded from the intended local schedule can otherwise
remain uninitialized before later activation, quantization, or top-k reduction.

Commit `f9819a3`:

1. sorts invalid routes after valid local experts;
2. excludes invalid routes from ragged expert counts;
3. clears intermediate rows not written by the local ragged GEMM; and
4. clears invalid original route positions before the top-k sum.

The change is restricted to the MXFP4 precomputed-top-k EP path. It is not an
RSAG change and is not needed for MoE TP8.

Focused commands:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate

python -m pytest -q tokenspeed-kernel/test/test_kernel_api_selection.py \
  -k 'mxfp4_ep_topk_localization_masks_remote_experts or mxfp4_ep_routing_metadata_excludes_remote_routes or mxfp4_ep_apply_zeroes_unscheduled_remote_routes'

python -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_triton_mxfp4_ep_apply_gfx950.py
```

The host metadata suite passed 3 tests. The gfx950 test used packed,
swizzled production weights and matched an FP32 reference.

## EP8 with the route fix and original RSAG

This run used `85301b5`, so it contained both minimal production fixes while
retaining the original RSAG:

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
route metadata, which has focused host and gfx950 coverage.

## Exact-branch focused verification

After removing the RSAG refactor from this branch, the following checks were
rerun:

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m pytest -q \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_token_ops_inference_initialized_state

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_triton_mxfp4_ep_apply_gfx950.py
```

Before each GPU command, `rocm-smi --showuse --showmemuse --showpidgpus`
reported all eight GPUs at zero utilization, zero allocated VRAM, and no KFD
owners. Results were 2 passing WS2/WS4 persistent-state cases and 1 passing
gfx950 MXFP4 EP case.

The three host routing tests also passed, both configuration scripts and the
harness passed `bash -n`, the CI YAML parsed successfully, and
`pre-commit run --all-files` passed.

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

- EP8 agentic concurrency 16 completion;
- the uncapped 10-to-15-turn agentic dataset;
- long-duration replay or soak behavior;
- execution of the updated four-GPU CI job; and
- a controlled diagnosis of the EP8 concurrency-16 device progress stop.

TP8 was tested through every bounded point on the production-equivalent
persistent-only state. EP8 was tested through concurrency 8 with both minimal
production fixes and the original RSAG.

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
