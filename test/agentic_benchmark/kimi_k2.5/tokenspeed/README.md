# Agentic Benchmark — TokenSpeed

Sweep `ts serve` against an agentic, multi-turn workload (SWE-Smith) at a
fixed set of attention/MoE parallelism layouts and report per-config throughput,
latency, and KV-cache hit rate.

The server listens on port **8000** by default. The harness prepends the
runtime, portable kernel, and separately packaged AMD kernel roots from the
current checkout to `PYTHONPATH`, so editable installs from another worktree do
not affect the result.

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

The script (1) installs evalscope at the pinned commit, (2) builds the SWE-Smith
multi-turn dataset, (3) iterates each config in `CONFIGS=()`: launch server, poll
`/readiness` until ready, run `evalscope perf`, kill server, wait for the port to be
free, repeat. Before each launch, it requires eight GPUs to remain idle across
several consecutive `rocm-smi` checks. It aborts the whole sweep on the first
failure (`set -e`).

Use `BENCH_CONFIGS` for an isolated profile, and override the HTTP and distributed
initialization ports when the defaults are occupied:

```bash
BENCH_CONFIGS=attn_dp8_moe_ep8 \
GPU_IDLE_STABLE_CHECKS=7 \
SERVER_PORT=8130 \
DIST_INIT_ADDR=127.0.0.1:4230 \
./agentic_bench.sh
```

The generated dataset contains 10–15 turns per conversation. Leave
`AGENTIC_MAX_TURNS` unset for the complete sweep; setting it is intended only
for bounded debugging.

## Configs

Each `configs/*.sh` `exec`s `ts serve` with the full flag set for one
layout. Key flags:

- `--attn-tp-size`, `--moe-tp-size` *or* `--ep-size`
- `--max-num-seqs`, `--max-prefill-tokens`, `--chunked-prefill-size`
- AMD model/quantization: `amd/Kimi-K2.5-MXFP4`, `--quantization mxfp4`
- AMD backends: `--moe-backend triton`, `--attention-backend mla`
- Eagle3 spec-dec: `--speculative-algorithm EAGLE3 --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`

Naming: `attn_<X>_moe_<Y>` where `X ∈ {tp4,tp8,dp8}` and `Y ∈ {tp4,tp8,ep4,ep8}`.
World size = the number after `attn_(tp|dp)`.

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
