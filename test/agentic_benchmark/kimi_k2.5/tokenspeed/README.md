# Agentic Benchmark — TokenSpeed

Sweep `ts serve` against an agentic, multi-turn workload (SWE-Smith) at a
fixed set of attention/MoE parallelism layouts and report per-config throughput,
latency, and KV-cache hit rate.

Server listens on port **8000** by default. Set `SERVER_PORT` (and, for DP
profiles, `DIST_INIT_ADDR`) when those local ports are already owned by another
service.

## Layout

```
agentic_bench.sh        # main sweep: dataset prep -> for config in CONFIGS: launch, wait, bench, kill
configs/                # one shell script per parallelism layout (each `exec`s ts serve)
collect_outputs.py      # parse a sweep into a flat CSV
outputs/<sweep_ts>/<config>/parallel_<P>_number_<N>/  # per-run evalscope artifacts
```

## Run a sweep

```bash
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
cd test/agentic_benchmark/kimi_k2.5/tokenspeed
./agentic_bench.sh
```

The AMD DP8 verification is initiated through the same harness, narrowed to the
two layouts that select token all-gather/reduce-scatter:

```bash
BENCH_CONFIGS="attn_dp8_moe_tp8 attn_dp8_moe_ep8" \
SERVER_PORT=8100 \
DIST_INIT_ADDR=127.0.0.1:4100 \
./agentic_bench.sh
```

The script (1) installs evalscope at the pinned commit, (2) builds the SWE-Smith
multi-turn dataset, (3) iterates each config in `CONFIGS=()`: launch server, poll
`/readiness` until ready, run `evalscope perf`, kill server, wait for the port to be
free, repeat. It puts this checkout's `python/` and `tokenspeed-kernel/python/`
first on `PYTHONPATH`, so an editable install from another worktree cannot change
what is being verified. The harness aborts the whole sweep on the first failure
(`set -e`).

The readiness timeout is 65 minutes and each config gives the engine 60 minutes
because a gfx950 Gluon MXFP4 startup can spend more than the default 30-minute
engine bound in model-wide weight preshuffling before draft loading and graph
capture. By default, EvalScope uses each dataset conversation's normal 10--15
turn depth. Set `AGENTIC_MAX_TURNS=<N>` for a deliberately bounded verification.

Use the whitespace-separated `BENCH_CONFIGS` environment variable to narrow the
matrix without editing the harness.

## Verification criteria

- **Startup** covers launch through successful readiness: target and draft
  checkpoints load, backend preprocessing and built-in warmup complete, decode
  graph capture completes, and HTTP `/readiness` reports serving.
- **Serving** is the first `evalscope perf` phase. It passes when two concurrent
  SWE-Smith conversations complete through the OpenAI-compatible API.
- **Agentic sweep** is the second `evalscope perf` phase. Its nominal
  `(number, parallel)` pairs are `(4,1)`, `(8,2)`, `(8,4)`, `(16,8)`, and
  `(32,16)`.

The harness prints a fresh `rocm-smi` utilization, VRAM, and KFD-owner snapshot
immediately before every launch. It will not launch an eight-GPU layout unless
it can select eight GPUs with zero utilization, zero allocated VRAM, and no KFD
owner. Four-GPU layouts use the same rule for four cards. GPU visibility is fixed
to the reserved IDs for the lifetime of that server.

The investigation, exact bounded invocation, compiler evidence, and measured
DP8 TP8/EP8 results are recorded in
[`amd_dp8_rsag_verification.md`](amd_dp8_rsag_verification.md).

## Configs

Each `configs/*.sh` `exec`s `ts serve` with the full flag set for one
layout. Key flags:

- `--attn-tp-size`, `--moe-tp-size` *or* `--ep-size`
- `--max-num-seqs`, `--max-prefill-tokens`, `--chunked-prefill-size`
- `--quantization mxfp4` with `amd/Kimi-K2.5-MXFP4`
- `--engine-startup-timeout 3600` for model-wide Gluon preprocessing
- `--disable-prefill-graph`; decode graphs remain enabled
- `--moe-backend gluon` for MoE TP and `--moe-backend triton` for MoE EP
- MLA target attention and MHA Eagle3 draft attention
- Eagle3 spec-dec: `--speculative-algorithm EAGLE3 --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`

The draft checkpoint is `lightseekorg/kimi-k2.5-eagle3`. If the checkpoints are
already downloaded in a non-default hub cache, set `HF_HUB_CACHE` and continue to
use their model IDs. Passing a raw snapshot path breaks tokenizer-relative Python
imports because snapshot entries are content-addressed symlinks.

Naming: `attn_<X>_moe_<Y>` where `X ∈ {tp4,tp8,dp8}` and `Y ∈ {tp4,tp8,ep4,ep8}`.
World size = the number after `attn_(tp|dp)`. All six layouts are included by
default.

To verify the parallelism actually applied, grep the server log:
```bash
grep -A6 "Parallelism configuration" /tmp/tokenspeed_server_<config>.log
```

## Collect results

```bash
python3 collect_outputs.py outputs/<sweep_ts> -o sweep.csv
```

Emits one row per (config, concurrency) with `Conc.`, `Latency (tps/user)`,
`Throughput (tps/gpu)`, `Approx Cache Hit`, `Decoded Tok/Iter`. `tps/gpu` divides
the system-wide `Total Throughput (tok/s)` by the GPU count inferred from the
config name; the other metrics come straight from `benchmark_summary.json`
(same numbers as evalscope's `performance_summary.txt` Request Metrics table).
