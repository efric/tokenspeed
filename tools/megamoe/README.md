# Kimi K3 MegaMoE qualification tools

`gfx950_persistent_grid_probe.py` qualifies the cooperative 240-workgroup
persistent-grid mechanism without model weights.

`kimi_k3_e2e_benchmark.py` is the fail-closed end-to-end throughput gate. It
copies the graph-enabled batch-one commands from
`profile_default/kimi-k3-main-04bc0864-graph-20260813T062945Z/COMMANDS.md` and
runs three repetitions for each current-checkout arm:

1. the existing fused path, with the experimental flag absent; and
2. the identical server, environment, and EvalScope workload with only
   `--enable-kimi-k3-megamoe` appended.

Both current-checkout arms additionally pass `--no-enable-prefix-caching`.
EvalScope deliberately regenerates the same seeded 4096-token prompt in every
repetition so the harness can require an identical semantic fingerprint across
all six runs. Disabling local prefix reuse makes each repetition an independent
4096-token prefill followed by a 1024-token decode and prevents repetitions two
and three from measuring a cached-history path that repetition one cannot take.

Before starting a full model, run the non-mutating preflight from the qualified
TokenSpeed environment:

```bash
cd /home/ericfeng/distributed/tokenspeed
source /home/ericfeng/distributed/.venvs/tokenspeed/bin/activate
unset PYTHONPATH
python tools/megamoe/kimi_k3_e2e_benchmark.py --preflight-only
```

The preflight rejects an incomplete implementation marker, a mismatched ROCr
or Triton cooperative-launch ABI, unexpected CLI routing, a different
EvalScope version, occupied ports, or anything other than eight idle gfx950
devices. The benchmark launches all children with exactly
`LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1`.

For a controlled paired measurement, temporarily enable the qualified candidate
marker and run the comparison into a new directory. The honest evidence command
records a loss as data rather than converting it into a harness failure:

```bash
python tools/megamoe/kimi_k3_e2e_benchmark.py \
  --output-root /home/ericfeng/distributed/tokenspeed/profile_default/kimi-k3-megamoe-compare-$(date -u +%Y%m%dT%H%M%SZ)
```

The harness executes the three flag-off repetitions first and then the three
flag-on repetitions, so the result must record fixed-order time drift as a
caveat. Use `--require-win` only for a future candidate being considered for
performance admission; it makes a correct but non-winning comparison return a
nonzero status. Restore the implementation marker to `False` after any rejected
candidate run.

The output contains the exact commands, environment and source fingerprints,
pre-launch `rocm-smi` JSON, server and EvalScope logs, each original EvalScope
report, `all-runs.csv`, `median-of-three.csv`, and `comparison.json`. A startup,
activation-marker, request, fatal-epoch, cleanup, or source-change failure is
recorded in `failure.json` and returns a nonzero status.

Each measured run is also checked against EvalScope's `benchmark_data.db`. The
harness requires exactly one successful 4096-to-1024 row, extracts the JSON
`request.prompt`, decodes EvalScope's response list with a primitive-only
restricted unpickler, and concatenates the streamed `choices[0].text` fragments
in database order. SHA-256 hashes and UTF-8 character/byte counts for those two
semantic strings are recorded in `completed-runs.json`, `all-runs.csv`, and
`comparison.json`. Response IDs, creation timestamps, finish reasons, and other
transport metadata are intentionally excluded.

The comparison fails immediately if measured prompts differ across repetitions
or arms, or if any completion differs. Therefore the default workflow accepts
throughput only after proving one prompt hash and one completion hash across all
six contemporaneous runs.
