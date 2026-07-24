# Agentic Benchmark — TokenSpeed

Sweep `ts serve` against an agentic, multi-turn workload (SWE-Smith) at a
fixed set of attention/MoE parallelism layouts and report per-config throughput,
latency, and KV-cache hit rate.

Server listens on port **8000**.

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

The script (1) installs EvalScope at the pinned commit, (2) builds the SWE-Smith
multi-turn dataset, and (3) iterates each selected configuration: launch the
server, poll `/readiness`, run serving and agentic `evalscope perf` phases,
stop the server, and wait for the port to be free. It aborts the sweep on the
first failure (`set -e`).

The default matrix contains the two AMD attention-DP profiles. Select one
without editing the script:

```bash
BENCH_CONFIGS=attn_dp8_moe_tp8 ./agentic_bench.sh
BENCH_CONFIGS=attn_dp8_moe_ep8 ./agentic_bench.sh
```

Before every launch, the harness requires eight GPUs that remain at zero
utilization, zero allocated VRAM, and without KFD owners across consecutive
`rocm-smi` samples. It exposes only those selected IDs to the server.

See [MINIMAL_DP8_ENABLEMENT.md](MINIMAL_DP8_ENABLEMENT.md) for exact commands,
isolated persistent-only results, patch boundaries, evidence hashes, and the
remaining EP8 acceptance gap.

## Configs

Each `configs/*.sh` `exec`s `ts serve` with the full flag set for one
layout. Key flags:

- `--data-parallel-size 8`, with `--moe-tp-size 8` *or* `--ep-size 8`
- `--max-num-seqs`, `--max-prefill-tokens`, `--chunked-prefill-size`
- `--quantization mxfp4`
- `--attention-backend mla`, `--drafter-attention-backend mha`
- `--moe-backend gluon` for TP8 or `--moe-backend triton` for EP8
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
