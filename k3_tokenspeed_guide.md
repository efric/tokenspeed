# Kimi K3 on TokenSpeed

This guide covers a reproducible development workflow for serving, validating,
benchmarking, and profiling Kimi K3 with TokenSpeed. Commands use placeholders
and do not assume a particular user, checkout location, model cache, GPU index,
or CI runner.

The relevant in-tree references are:

- `docs/guides/getting-started.md`
- `docs/recipes/models.md`
- `python/tokenspeed/bench.py`
- `tokenspeed-kernel/README.md`
- `test/runtime/test_kimi_k3_*.py`
- `tokenspeed-kernel/test/ops/`
- `test/ci/eval/` and `test/ci/perf/`

## 1. Quick Start

### Environment and build

Start from a CUDA or ROCm environment supported by the checkout. Use a virtual
environment rather than installing Python packages globally:

```bash
cd /path/to/tokenspeed
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e ./python --no-build-isolation
python -m pip install -e ./tokenspeed-kernel/python --no-build-isolation

# Kimi K3 is FlatKV-only.
SKBUILD_CMAKE_DEFINE="TOKENSPEED_FLAT_KVCACHE=ON" \
  python -m pip install -e ./tokenspeed-scheduler
```

Verify the installation and scheduler build:

```bash
tokenspeed env
tokenspeed serve --help
python - <<'PY'
import tokenspeed_scheduler
assert tokenspeed_scheduler.FLAT_KVCACHE
print("FlatKV scheduler: enabled")
PY
```

The current NVIDIA KDA path also requires `flash-linear-attention`. AMD uses
the native KDA implementation and does not require that package.

### Checkpoint preparation

Set the checkpoint and writable Transformers module cache:

```bash
export MODEL_DIR=/path/to/flattened/Kimi-K3
export HF_MODULES_CACHE=/path/to/writable/transformers_modules
export HOST=127.0.0.1
export PORT=8000
export BASE_URL="http://${HOST}:${PORT}"
```

`$MODEL_DIR` must be a flattened local checkpoint:

- Config, tokenizer, and checkpoint `*.py` files must be real files.
- Weight files may remain symlinks.
- Do not point the current SMG launch path directly at a Hugging Face snapshot
  whose Python/tokenizer files resolve through `blobs/` symlinks.

The standard single-node recipes below use eight-way text tensor parallelism.
Kimi K3's vision encoder has 12 attention heads, so use
`--mm-encoder-tp-mode data` with text TP8.

### AMD launch

```bash
tokenspeed serve "$MODEL_DIR" \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --attention-backend mla \
  --moe-backend auto \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --disable-kvstore \
  --host "$HOST" \
  --port "$PORT"
```

On gfx950, this selects the specialized Gluon SiTU MoE path when its topology
and shapes are supported.

### NVIDIA launch

```bash
tokenspeed serve "$MODEL_DIR" \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --ep-size 8 \
  --moe-backend flashinfer_trtllm \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --disable-kvstore \
  --host "$HOST" \
  --port "$PORT"
```

The NVIDIA backend and its sidecar have architecture-specific requirements.
Use `--moe-backend triton` as the portable fallback where the specialized
TensorRT-LLM backend is unavailable.

Both commands keep CUDA/HIP graph execution enabled. Do not add
`--enforce-eager` unless eager execution is the behavior being investigated.

### Readiness and chat smoke test

```bash
curl -f "${BASE_URL}/readiness"
curl -s "${BASE_URL}/v1/models"

curl -s "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3",
    "messages": [
      {"role": "user", "content": "Explain why the sky appears blue in two sentences."}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

TokenSpeed automatically selects the Kimi K3 reasoning and tool-call parsers.
The smoke test should be checked for a successful response, plausible text, and
the absence of NaNs or malformed output. It is not a substitute for an accuracy
evaluation.

## 2. Validation and Benchmarking

### Workload controls

TokenSpeed's in-tree serving benchmark is `tokenspeed bench serve`.

| Control | Meaning |
|---|---|
| `--input-len` | Target prompt length. This primarily controls prefill work. |
| `--output-len` | Requested generated length. Long outputs expose steady-state decode. |
| `--num-prompts` | Total requests measured. |
| `--max-concurrency` | Maximum client requests in flight at once. |
| `--request-rate` | Open-loop arrival rate in requests/second. The default `inf` submits as fast as the concurrency limit allows. |
| `--random-range-ratio` | Randomizes input and output lengths around their configured targets instead of using exact lengths. |
| `--random-prefix-len` | Gives generated requests one shared random prefix for prefix-cache experiments. |
| `--max-num-seqs` | Server-side cap on resident sequences. |
| `--chunked-prefill-size` | Server-side maximum scheduled tokens per iteration; `-1` disables chunked prefill. |
| `--max-total-tokens` | Explicit server KV/token-pool capacity. |

Client concurrency is not a fixed GPU batch size. TokenSpeed continuously
forms dynamic batches from active requests. A long-output, infinite-rate run
with concurrency `N` approximates decode batch `N` while all requests remain
active. Simultaneous long prompts exercise batched/chunked prefill, but the
actual per-step token count is still constrained by scheduler budgets.

Keep these fixed when comparing two implementations:

- checkpoint and tokenizer
- parallel topology and backend selection
- graph versus eager mode
- speculative-decoding configuration
- input/output lengths, prompt seed, request count, and concurrency
- prefix-cache state

Do not compare aggregate kernel time from traces with different generated-token
counts. Compare identical requests or per-occurrence kernel statistics.

### Basic serving benchmark

Pass the local tokenizer explicitly because the served name `kimi-k3` is not a
tokenizer path:

```bash
tokenspeed bench serve \
  --base-url "$BASE_URL" \
  --model kimi-k3 \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --input-len 4096 \
  --output-len 1024 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --num-warmups 2 \
  --ignore-eos \
  --extra-body '{"temperature": 0}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --result-dir ./benchmark-results
```

The principal metrics are:

- **TTFT**: prefill plus scheduling latency to the first output token.
- **TPOT/ITL**: steady-state decode latency per token.
- **Completion throughput**: aggregate output tokens/second.
- **E2EL**: complete request latency.
- **TPS/user**: approximately `1000 / TPOT_ms` for a steady decode stream.

### Common workload set

The following set covers the workloads used most often in current public K3
reports and TokenSpeed's in-tree performance recipes:

| Scenario | Input/output | Concurrency | Purpose |
|---|---:|---:|---|
| Fast smoke | 128/32 | 1 | Endpoint and basic generation health |
| Prefill | 8192/1 and 16384/1 | 1 | TTFT and prompt throughput with negligible decode |
| Fixed low-latency decode | 4096/1024 | 1 | Exact-length local baseline |
| Published K3 random recipe | 8192/1024 targets, range ratio 0.8 | 1 | Reproduces the target-only command published by vLLM |
| Random throughput sweep | 1024/1024, 4096/1024, 8192/1024 | 1, 2, 4, 8, 16 | Pareto curve across prompt sizes and concurrency |
| Prefix reuse | Same as above with a shared prefix | 1, 2, 4, 8, 16 | FlatKV prefix-cache behavior |
| Agentic | SWE-smith-style multi-turn traces | 1, 2, 4, 8, 16 | TPS/user versus tokens/minute/GPU |
| Long context | 128K, 256K, 512K, and 1M prompts | 1 | Cache capacity and long-prefill scaling |
| Speculative decode | Low-entropy coding/reasoning prompts, long outputs | 1 first, then a concurrency sweep | Acceptance rate and target-plus-draft throughput |

The public vLLM target-only K3 command uses 8K/1K target lengths,
`--random-range-ratio 0.8`, and concurrency 1. Range-ratio semantics are tied
to the benchmark implementation and version, so preserve the exact benchmark
tool and flag when comparing with that published number. Its speculative SPEED
Bench run uses low-entropy prompts up to roughly 10K input and 1536 output
tokens. TokenSpeed's built-in benchmark currently supports `random` and
`sharegpt`, not SPEED Bench, so use EvalScope or the upstream benchmark for
that dataset.

The TokenSpeed K3 launch report validates OCRBench, MMMU-Pro, and DeepSWE.
Those are the current high-value quality checks when reproducing a performance
change. Run a small subset during iteration and the complete evaluation before
publishing a result.

References:

- [TokenSpeed Kimi K3 launch report](https://lightseek.org/blog/tokenspeed-kimi-k3.html)
- [vLLM Kimi K3 benchmark recipe](https://vllm-project.github.io/2026/07/27/k3.html#reproduce-our-benchmark)
- [TokenSpeed agentic performance methodology](https://lightseek.org/blog/lightseek-tokenspeed.html)

### Concurrency sweep

Use at least several measured requests per concurrency so startup and tail
effects do not dominate:

```bash
for input_len in 1024 4096 8192; do
  for concurrency in 1 2 4 8 16; do
    tokenspeed bench serve \
      --base-url "$BASE_URL" \
      --model kimi-k3 \
      --tokenizer "$MODEL_DIR" \
      --dataset-name random \
      --input-len "$input_len" \
      --output-len 1024 \
      --num-prompts "$((5 * concurrency))" \
      --max-concurrency "$concurrency" \
      --num-warmups "$concurrency" \
      --ignore-eos \
      --extra-body '{"temperature": 0}' \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,95,99 \
      --label "k3-${input_len}-c${concurrency}" \
      --save-result \
      --result-dir ./benchmark-results
  done
done
```

For an open-loop capacity test, hold `--max-concurrency` fixed and sweep
`--request-rate`. Stop increasing the rate when queueing causes TTFT/TPOT to
violate the intended service objective. `--ramp-up-strategy` can perform a
linear or exponential request-rate ramp.

### Runtime and model validation

Cheap K3 configuration and FlatKV checks:

```bash
pytest -q \
  test/runtime/test_kimi_k3_config.py \
  test/runtime/test_kimi_k3_cache_spec.py \
  test/runtime/test_kimi_k3_expert_map.py \
  test/runtime/models/test_kimi_k3_vlm.py
```

Runtime integration, graph, KDA, MLA, and AttnRes checks:

```bash
pytest -q \
  test/runtime/test_kimi_k3_attn_res.py \
  test/runtime/test_kimi_k3_flat_integration.py \
  test/runtime/test_kimi_k3_flat_cudagraph.py \
  test/runtime/test_kimi_k3_flat_kda.py \
  test/runtime/test_kimi_k3_flat_mla.py \
  test/runtime/test_hybrid_mla_kv_nan_sanitize.py
```

Representative kernel tests:

```bash
pytest -q \
  tokenspeed-kernel/test/ops/test_kimi3_dispatch.py \
  tokenspeed-kernel/test/ops/test_kimi3_projection_gfx950.py \
  tokenspeed-kernel/test/ops/test_kimi3_prefill_gluon_gfx950.py \
  tokenspeed-kernel/test/ops/attention/test_kda_fused_decode_amd.py \
  tokenspeed-kernel/test/ops/moe/test_mxfp4_situ_latent_moe_reference.py \
  tokenspeed-kernel/test/ops/moe/test_latent_moe_runtime_gfx950.py
```

Architecture-specific tests skip when their required backend is unavailable.
Do not treat a skipped gfx950 or NVIDIA-specific test as validation on another
architecture.

### EvalScope

`test/ci/eval/` and `test/ci/perf/` contain the in-tree EvalScope command
patterns. There is no checked-in K3-specific EvalScope YAML in the current
tree, so use those manifests as references for a direct EvalScope command.

A direct small accuracy run follows this form:

```bash
export EVAL_DATASET=aime25
export EVAL_LIMIT=4
export EVAL_CONCURRENCY=4

evalscope eval \
  --model kimi-k3 \
  --api-url "${BASE_URL}/v1" \
  --api-key EMPTY_TOKEN \
  --datasets "$EVAL_DATASET" \
  --limit "$EVAL_LIMIT" \
  --eval-batch-size "$EVAL_CONCURRENCY" \
  --stream \
  --generation-config \
    '{"do_sample": false, "temperature": 0.0, "max_tokens": 8192}'
```

For performance, EvalScope's `--parallel` is client concurrency and `--number`
is request count. The in-tree random-performance jobs commonly pair
concurrencies `1 2 4 8 16` with request counts `5 10 20 40 80`.

## 3. Eight-Rank Proton and VizTracer Profiling

Proton records semantic TokenSpeed kernel scopes and GPU activities. VizTracer
records Python activity. They run together because VizTracer is host-side.
Do not combine Proton with the torch `GPU` activity or `CUDA_PROFILER`; those
compete for the same CUPTI/roctracer interface.

### Launch with trace output configured

Use one empty directory per capture:

```bash
export TRACE_DIR=/path/to/traces/kimi-k3-tp8
mkdir -p "$TRACE_DIR"

export TOKENSPEED_PROFILER_DIR="$TRACE_DIR"
export TOKENSPEED_KERNEL_PROFILE_DATA=trace
export TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=chrome_trace
export TOKENSPEED_VIZTRACER_MIN_DURATION_US=20
```

On AMD, Proton requires `ROCR_VISIBLE_DEVICES`; `HIP_VISIBLE_DEVICES` must be
unset before the server starts:

```bash
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Launch the normal TP8 server with those environment variables. Keep graph mode
enabled. For an uncached prefill trace, explicitly disable prefix caching for
that server run with `--no-enable-prefix-caching`.

### Capture prefill and decode separately

Use stage-separated profiling so prefill/extend and decode statistics are
written to different files:

```bash
curl -sS -X POST "$BASE_URL/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{
    "num_steps": 8,
    "activities": ["VIZTRACER", "PROTON"],
    "profile_by_stage": true,
    "profile_id": "kimi-k3-tp8"
  }'
```

Then send one request with enough output tokens to capture the decode steps:

```bash
tokenspeed bench serve \
  --base-url "$BASE_URL" \
  --model kimi-k3 \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --input-len 4096 \
  --output-len 16 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ready-check-timeout-sec 0 \
  --ignore-eos \
  --extra-body '{"temperature": 0}'
```

Each scheduler rank writes a VizTracer and Proton file for each captured stage:

```text
kimi-k3-tp8-TP0-EXTEND.viztracer.json
kimi-k3-tp8-TP0-EXTEND.proton.chrome_trace
...
kimi-k3-tp8-TP7-DECODE.viztracer.json
kimi-k3-tp8-TP7-DECODE.proton.chrome_trace
```

Topologies with data or context parallelism may include `DP` or `CP` fields in
the rank tag. Use the emitted filenames rather than assuming a particular tag.

### Merge all ranks

Merge each stage independently:

```bash
for stage in EXTEND DECODE; do
  merge_args=(--all-ranks)
  for rank in {0..7}; do
    viz="$TRACE_DIR/kimi-k3-tp8-TP${rank}-${stage}.viztracer.json"
    proton="$TRACE_DIR/kimi-k3-tp8-TP${rank}-${stage}.proton.chrome_trace"
    test -f "$viz"
    test -f "$proton"
    merge_args+=(--rank "$rank" "$viz" "$proton")
  done

  output="$TRACE_DIR/kimi-k3-tp8-${stage,,}-merged.json"
  tokenspeed merge-traces "${merge_args[@]}" --output "$output"
  gzip -k "$output"
done
```

The merged JSON is the machine-readable trace artifact. The merge aligns each
rank through the absolute clock anchors recorded by both profilers, making it
suitable for programmatic analysis of launch ordering, stream gaps, and rank
skew. It is not intended for sub-microsecond attribution.

In the merged file:

- Proton rank `r` uses process ID `10000 + r`.
- `cat="metric"` events are semantic scopes such as
  `attention.mla_decode_with_kvcache[selected_kernel]`.
- `cat="kernel"` events are the physical GPU launches shown on GPU stream
  tracks.
- A physical kernel's `args.call_stack` identifies its enclosing semantic
  TokenSpeed operation where one is available.
- VizTracer-to-Proton flow events associate Python scope entry with the
  matching semantic Proton scope.

### Generate textual trace reports

Place the following standalone script in a temporary working directory as
`trace_text_report.py`. It turns the merged Chrome trace into TSV files that an
agent can sort, filter, compare, and correlate with source:

```python
#!/usr/bin/env python3
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
    return values[max(index, 0)]


def clean(value):
    return str(value).replace("\t", " ").replace("\n", " ")


def semantic_parent(event):
    stack = event.get("args", {}).get("call_stack", [])
    return clean(stack[-2]) if len(stack) >= 3 else "unscoped"


def write_tsv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("trace")
parser.add_argument("output_dir")
args = parser.parse_args()

trace_path = Path(args.trace)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
opener = gzip.open if trace_path.suffix == ".gz" else open
with opener(trace_path, "rt", encoding="utf-8") as trace_file:
    trace = json.load(trace_file)

kernels = []
scopes = []
for event in trace["traceEvents"]:
    if event.get("ph") != "X":
        continue
    pid = event.get("pid")
    if not isinstance(pid, int) or not 10000 <= pid < 20000:
        continue
    if event.get("cat") == "kernel":
        kernels.append(event)
    elif event.get("cat") == "metric":
        scopes.append(event)

if not kernels:
    raise SystemExit("no merged Proton kernel events found")

first_timestamp = min(
    float(event["ts"]) for event in [*kernels, *scopes] if "ts" in event
)
samples = collections.defaultdict(list)
for event in kernels:
    rank = event["pid"] - 10000
    key = (rank, semantic_parent(event), clean(event["name"]))
    samples[key].append(float(event["dur"]))

rank_rows = []
rank_stats = {}
for key, durations in samples.items():
    rank, scope, kernel = key
    stats = {
        "count": len(durations),
        "total": sum(durations),
        "mean": sum(durations) / len(durations),
        "p50": percentile(durations, 0.50),
        "p95": percentile(durations, 0.95),
        "max": max(durations),
    }
    rank_stats[key] = stats
    rank_rows.append(
        (
            rank,
            stats["count"],
            f'{stats["total"]:.3f}',
            f'{stats["mean"]:.3f}',
            f'{stats["p50"]:.3f}',
            f'{stats["p95"]:.3f}',
            f'{stats["max"]:.3f}',
            scope,
            kernel,
        )
    )
rank_rows.sort(key=lambda row: (row[0], -float(row[2])))
write_tsv(
    output_dir / "kernels-by-rank.tsv",
    ("rank", "count", "total_us", "mean_us", "p50_us", "p95_us", "max_us",
     "scope", "kernel"),
    rank_rows,
)

cross_rank = collections.defaultdict(list)
for (rank, scope, kernel), stats in rank_stats.items():
    cross_rank[(scope, kernel)].append((rank, stats))

cross_rows = []
for (scope, kernel), members in cross_rank.items():
    counts = [stats["count"] for _, stats in members]
    totals = [stats["total"] for _, stats in members]
    p50s = [stats["p50"] for _, stats in members]
    p95s = [stats["p95"] for _, stats in members]
    critical_rank, critical_stats = max(members, key=lambda item: item[1]["total"])
    cross_rows.append(
        (
            len(members),
            min(counts),
            max(counts),
            f"{min(totals):.3f}",
            f"{critical_stats['total']:.3f}",
            critical_rank,
            f"{min(p50s):.3f}",
            f"{max(p50s):.3f}",
            f"{max(p95s):.3f}",
            scope,
            kernel,
        )
    )
cross_rows.sort(key=lambda row: -float(row[4]))
write_tsv(
    output_dir / "kernels-cross-rank.tsv",
    ("rank_count", "count_min", "count_max", "total_min_us", "total_max_us",
     "critical_rank", "p50_min_us", "p50_max_us", "p95_max_us", "scope",
     "kernel"),
    cross_rows,
)

sequence_rows = []
previous_end = {}
occurrences = collections.Counter()
for event in sorted(kernels, key=lambda item: (item["pid"], item["ts"])):
    rank = event["pid"] - 10000
    stream = event.get("tid", "unknown")
    start = float(event["ts"]) - first_timestamp
    duration = float(event["dur"])
    stream_key = (rank, stream)
    gap = max(0.0, start - previous_end.get(stream_key, start))
    previous_end[stream_key] = max(previous_end.get(stream_key, start), start + duration)
    scope = semantic_parent(event)
    kernel = clean(event["name"])
    occurrence_key = (rank, scope, kernel)
    occurrence = occurrences[occurrence_key]
    occurrences[occurrence_key] += 1
    sequence_rows.append(
        (
            rank,
            f"{start:.3f}",
            f"{duration:.3f}",
            f"{gap:.3f}",
            stream,
            occurrence,
            scope,
            kernel,
        )
    )
write_tsv(
    output_dir / "kernel-sequence.tsv",
    ("rank", "start_us", "duration_us", "gap_before_us", "stream",
     "occurrence", "scope", "kernel"),
    sequence_rows,
)

scope_rows = []
for event in sorted(scopes, key=lambda item: (item["pid"], item["ts"])):
    stack = event.get("args", {}).get("call_stack", [])
    scope = clean(stack[-1] if stack else event["name"])
    metrics = json.dumps(
        event.get("args", {}).get("metrics", {}), sort_keys=True, separators=(",", ":")
    )
    scope_rows.append(
        (
            event["pid"] - 10000,
            f'{float(event["ts"]) - first_timestamp:.3f}',
            f'{float(event["dur"]):.3f}',
            scope,
            metrics,
        )
    )
write_tsv(
    output_dir / "semantic-scopes.tsv",
    ("rank", "start_us", "duration_us", "scope", "metrics"),
    scope_rows,
)
```

Run it as:

```bash
export REPORT_DIR="$TRACE_DIR/text-report"
for stage in extend decode; do
  python trace_text_report.py \
    "$TRACE_DIR/kimi-k3-tp8-${stage}-merged.json.gz" \
    "$REPORT_DIR/$stage"
done
```

The outputs are:

- `kernels-by-rank.tsv`: physical kernel cost under each semantic operation,
  sorted by total time within each rank.
- `kernels-cross-rank.tsv`: the same scope/kernel correlated across ranks,
  sorted by the largest per-rank total.
- `kernel-sequence.tsv`: chronological launches within each rank, with aligned
  timestamps, stream, occurrence, and preceding idle gap.
- `semantic-scopes.tsv`: TokenSpeed operation names, captured shape metrics,
  and host-side scope durations.

Choose the stage being investigated, then start with its cross-rank critical
kernels and rank-0 launch sequence:

```bash
export STAGE_REPORT="$REPORT_DIR/decode"
head -n 41 "$STAGE_REPORT/kernels-cross-rank.tsv"
awk -F '\t' 'NR == 1 || $1 == 0' \
  "$STAGE_REPORT/kernels-by-rank.tsv" | head -n 41
```

Query communication, suspected fusion tails, and real operation shapes:

```bash
rg -i 'iris|all.?reduce|all.?to.?all|nccl|rccl' \
  "$STAGE_REPORT/kernel-sequence.tsv"
rg -i 'copy|cast|transpose|permute|concat|cat|fill|elementwise' \
  "$STAGE_REPORT/kernel-sequence.tsv"
rg -i 'mla|kda|latent_moe|topk|router' \
  "$STAGE_REPORT/semantic-scopes.tsv"
```

Use context lines around a launch to recover its producers and consumers:

```bash
rg -n -C 8 'physical_kernel_name_fragment' \
  "$STAGE_REPORT/kernel-sequence.tsv"
```

For a kernel or collective shared by every rank, group the same occurrence
across ranks and compare `start_us` and `duration_us`:

```bash
export KERNEL_FRAGMENT=iris_stage_one_shot_allreduce_kernel
rg -F "$KERNEL_FRAGMENT" "$STAGE_REPORT/kernel-sequence.tsv" \
  | sort -t $'\t' -k6,6n -k1,1n
```

For synchronized work, the critical time is governed by the slowest rank.
`kernels-cross-rank.tsv` exposes total and percentile spread, while
`kernel-sequence.tsv` exposes late arrival at collectives and stream gaps.

## 4. From Trace to a Focused Optimization

### Select work from the critical path

Use this order:

1. Analyze `text-report/extend` and `text-report/decode` independently.
2. Read `kernels-cross-rank.tsv` from the chosen stage to find operations with
   the largest `total_max_us` and the rank responsible for that total.
3. Read that rank in `kernels-by-rank.tsv`; `total_us` already represents
   `launch count × observed duration`.
4. Locate the kernel in `kernel-sequence.tsv` and inspect neighboring
   collectives, stream gaps, copies, casts, relayouts, and elementwise tails.
5. Query the semantic scope in `semantic-scopes.tsv` to recover its real shape
   envelope and selected implementation.
6. Map the selected implementation to source and reproduce those shapes with
   the focused numerics and benchmark commands below.

For K3, inspect repeated layer blocks by component:

- KDA or MLA attention
- AttnRes
- router/top-k and latent MoE
- shared experts
- TP/EP collectives
- dense projections and residual/norm epilogues

Do not prioritize by one unusually slow first launch. Exclude compilation,
lazy initialization, graph capture, and warmup from steady-state statistics.

### Map a trace event to code

1. Read the `scope` column in the textual report, for example
   `gemm.mm[torch_mm]` or
   `attention.mla_decode_with_kvcache[gluon_...]`.
2. Search the operator family/mode and selected kernel:

   ```bash
   export SELECTED_KERNEL=gluon_mla_decode_fp8xfp8_gfx950_bh16bn128
   rg "mla_decode_with_kvcache|${SELECTED_KERNEL}" \
     python tokenspeed-kernel/python
   ```

3. Start from the highest-level runtime call, then follow the public
   `tokenspeed_kernel` API into registry selection and finally the backend
   implementation.
4. Treat opaque vendor GEMM names such as `Cijk_...` as physical kernels. Their
   semantic Proton parent identifies the TokenSpeed operation that launched
   them.

### Reproduce the real shape

Use the shape metrics attached to the semantic Proton scope. Record all
selection-relevant properties, not only `M/N/K`:

- dtype and quantization format
- tensor shapes, strides, and scale layouts
- batch/token count and context length
- TP/EP size and local expert ownership
- top-k and local routed-token counts
- graph/eager state
- optional outputs, residuals, gates, or fused epilogues

Build a shape envelope around the trace point:

- Decode: `M=1` and active-sequence counts expected from the concurrency sweep.
- Prefill: the chunk/token counts produced by the scheduler, including boundary
  and non-multiple shapes.
- MoE: balanced, empty-local-expert, and skewed routing.
- Attention: short, representative, and maximum intended context lengths.

### Numerics and kernel benchmarking

For an operator already registered with a numerics input generator:

```bash
export KERNEL_NAME=registered_kernel_name
export DTYPE_ROLE=a
export SHAPES='[{"M": 1, "N": 7168, "K": 3584}]'

python -m tokenspeed_kernel.numerics "$KERNEL_NAME" \
  --dtype bf16 \
  --dtype-role "$DTYPE_ROLE" \
  --shapes "$SHAPES" \
  --verbose

python -m tokenspeed_kernel.benchmark "$KERNEL_NAME" \
  --dtype bf16 \
  --dtype-role "$DTYPE_ROLE" \
  --shapes "$SHAPES" \
  --verify \
  --warmup-iters 100 \
  --bench-iters 1000 \
  --export benchmark.json
```

To compare every registered implementation for an operation, replace the
kernel name with `--op family.mode`. For example:

```bash
python -m tokenspeed_kernel.benchmark \
  --op gemm.mm \
  --dtype bf16 \
  --dtype-role a \
  --shapes '[{"M": 1, "N": 7168, "K": 3584}]' \
  --verify \
  --warmup-iters 100 \
  --bench-iters 1000
```

The runner uses GPU events and reports p50, p90, and p99 latency. Use at least
100 warmups and 1000 measured iterations for development comparisons; use more
iterations for very short kernels and final performance claims.

If the operation does not have a registered numerics generator, reproduce the
trace shape in a disposable harness outside the repository. Compare the public
operator or implementation entry point against a framework reference, keep
allocation and setup outside the measured region, and use GPU events or the
repository benchmark utilities for timing.

`kernel_override(family, mode, kernel_name)` can force a specific registered
implementation through the public operator from such a harness.

### Small-subgraph and fusion benchmarks

A fusion benchmark must compare:

```text
unfused producer + every materialization/copy/elementwise op + consumer
```

against:

```text
fused semantic operation
```

Use the same preallocated inputs and outputs, stream, dtype, layout, and shape.
Include synchronization and communication that the fusion changes. Verify the
final materialized outputs against the unfused reference, including any BF16
rounding boundaries that are part of the model semantics.

Validate the candidate across:

- representative and boundary shapes
- non-contiguous or preallocated outputs when the API permits them
- empty/skewed routing for MoE
- graph capture/replay if the operation is used inside decode graphs
- distributed correctness through `torchrun` when communication is involved

Then validate in this order:

1. Focused numerics test.
2. Focused kernel/subgraph benchmark.
3. Dispatch and fallback tests.
4. Relevant K3 runtime tests.
5. One real-checkpoint chat request.
6. Fixed 4K/1K and 8K/1K serving baselines.
7. Concurrency sweep.
8. Accuracy evaluation.
9. A new all-rank trace confirming that the intended kernel is selected and
   that no replacement copies, casts, barriers, or relayout kernels appeared.

An optimization is complete only when its isolated gain survives the identical
end-to-end workload and the output remains numerically and behaviorally valid.
