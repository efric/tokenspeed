#!/usr/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$SCRIPT_DIR"

# Always exercise this checkout, even when the active venv has editable
# installs pointing at another worktree.
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/tokenspeed-kernel/python${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_PORT=${SERVER_PORT:-8000}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-127.0.0.1:4000}
EVALSCOPE_TURN_ARGS=()
if [[ -n ${AGENTIC_MAX_TURNS:-} ]]; then
    EVALSCOPE_TURN_ARGS=(--max-turns "$AGENTIC_MAX_TURNS")
fi

# Kimi's MLA KV pool does not publish the cache groups required by the Flat KV
# scheduler build. Fail before dataset setup or the lengthy Gluon preshuffle.
python3 - <<'PY'
import tokenspeed_scheduler

if tokenspeed_scheduler.FLAT_KVCACHE:
    raise SystemExit(
        "Kimi MLA requires a radix-built tokenspeed_scheduler; rebuild with "
        "cmake.define.TOKENSPEED_FLAT_KVCACHE=OFF"
    )
PY

# Prepare dataset
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
python3 -m pip install \
    "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

[ -f build_swe_smith_dataset.py ] || wget https://raw.githubusercontent.com/modelscope/evalscope/${EVALSCOPE_COMMIT}/examples/perf/build_swe_smith_dataset.py \
    -O build_swe_smith_dataset.py

# Note: Only 71 conversations can be built
[ -f agentic_dataset.json ] || python3 build_swe_smith_dataset.py \
    --model-path moonshotai/Kimi-K2.5 \
    --first-turn-length 50000 \
    --subsequent-turn-length 800 \
    --min-turns 10 \
    --max-turns 15 \
    --number 128 \
    --output-path agentic_dataset.json \
    --num-workers 32

# Sweep all six layouts by default. BENCH_CONFIGS accepts a whitespace-separated
# subset, for example the two AMD token-RSAG layouts under investigation.
if [[ -n ${BENCH_CONFIGS:-} ]]; then
    read -r -a CONFIGS <<< "$BENCH_CONFIGS"
else
    CONFIGS=(
        attn_tp4_moe_tp4
        attn_tp4_moe_ep4
        attn_tp8_moe_tp8
        attn_tp8_moe_ep8
        attn_dp8_moe_tp8
        attn_dp8_moe_ep8
    )
fi

SERVER_PID=
SERVER_LOG=
SELECTED_GPU_IDS=

gpu_count_for_config() {
    case "$1" in
        attn_tp4_*) echo 4 ;;
        *) echo 8 ;;
    esac
}

wait_for_idle_gpus() {
    local required=$1
    local poll_interval=${GPU_POLL_INTERVAL_SECONDS:-30}

    command -v rocm-smi >/dev/null || {
        echo "rocm-smi is required to verify GPU availability" >&2
        return 1
    }

    while true; do
        local gpu_state
        local pid_state
        if gpu_state=$(rocm-smi --showuse --showmemuse --csv) &&
            pid_state=$(rocm-smi --showpidgpus); then
            printf '%s\n' "$gpu_state"
            printf '%s\n' "$pid_state"

            local busy_gpu_ids
            busy_gpu_ids=$(awk '
                /^PID [0-9]+ is using/ {
                    read_devices = 1
                    next
                }
                read_devices && /^[[:space:]]*[0-9]+([[:space:]]+[0-9]+)*[[:space:]]*$/ {
                    for (i = 1; i <= NF; i++) {
                        print $i
                    }
                    read_devices = 0
                }
            ' <<< "$pid_state" | sort -nu | paste -sd, -)

            local selected
            if selected=$(awk -F, -v required="$required" -v busy="$busy_gpu_ids" '
                function is_busy(id, count, ids, i) {
                    if (busy == "") {
                        return 0
                    }
                    count = split(busy, ids, ",")
                    for (i = 1; i <= count; i++) {
                        if (ids[i] == id) {
                            return 1
                        }
                    }
                    return 0
                }
                NR > 1 {
                    seen++
                    id = $1
                    sub(/^card/, "", id)
                    if (($2 + 0) == 0 && ($3 + 0) == 0 && !is_busy(id)) {
                        free++
                        if (free <= required) {
                            selected = selected (selected == "" ? "" : ",") id
                        }
                    }
                }
                END {
                    # Use any idle subset of the requested width. Busy devices
                    # outside that selected subset do not share this server.
                    if (free < required) {
                        exit 1
                    }
                    print selected
                }
            ' <<< "$gpu_state"); then
                SELECTED_GPU_IDS=$selected
                echo "Confirmed idle GPU(s) ${SELECTED_GPU_IDS} immediately before launch"
                return 0
            fi
        else
            echo "rocm-smi utilization or KFD-process refresh failed; not launching" >&2
        fi

        echo "Waiting ${poll_interval}s for ${required} idle GPU(s)..."
        sleep "$poll_interval"
    done
}

launch_server() {
    local config=$1
    SERVER_LOG=/tmp/tokenspeed_server_${config}.log
    HIP_VISIBLE_DEVICES="$SELECTED_GPU_IDS" \
        CUDA_VISIBLE_DEVICES="$SELECTED_GPU_IDS" \
        TOKENSPEED_PORT="$SERVER_PORT" \
        TOKENSPEED_DIST_INIT_ADDR="$DIST_INIT_ADDR" \
        setsid "${SCRIPT_DIR}/configs/${config}.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    # Do not return until setsid has established the matching process group.
    # stop_server also handles the pre-setsid child-PID case if INT/TERM arrives
    # during this brief loop.
    for _ in {1..100}; do
        kill -0 -- "-$SERVER_PID" 2>/dev/null && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || return 0
        sleep 0.01
    done
    echo "Server process group $SERVER_PID was not established" >&2
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    return 1
}

wait_for_ready() {
    # Keep the outer readiness bound beyond the configs' 60-minute engine
    # bound. A gfx950 Gluon MXFP4 launch can spend over 30 minutes in the
    # model-wide weight preshuffle before draft loading and graph capture.
    local TIMEOUT=3900
    local START=$SECONDS
    until curl -sf -o /dev/null "http://127.0.0.1:${SERVER_PORT}/readiness"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died early. Last log lines:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if grep -qE "CUDA out of memory|OutOfMemory|RuntimeError|Killed" "$SERVER_LOG"; then
            echo "Server hit a fatal error:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if (( SECONDS - START > TIMEOUT )); then
            echo "Timeout after ${TIMEOUT}s waiting for server" >&2
            return 1
        fi
        sleep 5
    done
    echo "Server ready after $((SECONDS - START))s"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
            echo "Stopping ts serve (pgid $SERVER_PID)..."
            kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
            for _ in {1..20}; do
                kill -0 -- "-$SERVER_PID" 2>/dev/null || break
                sleep 1
            done
            kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        elif kill -0 "$SERVER_PID" 2>/dev/null; then
            # An immediate signal can arrive between fork and setsid. At that
            # point the exact child PID is the only scoped cleanup target.
            echo "Stopping pre-setsid server process (pid $SERVER_PID)..."
            kill -TERM "$SERVER_PID" 2>/dev/null || true
            for _ in {1..20}; do
                kill -0 "$SERVER_PID" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_port_free() {
    local port=${1:-8000}
    local timeout=${2:-90}
    local start=$SECONDS
    while ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            echo "Port ${port} still in use after ${timeout}s" >&2
            return 1
        fi
        sleep 1
    done
}

handle_signal() {
    local exit_code=$1
    # Let the EXIT handler perform the scoped process-group cleanup without a
    # second signal re-entering this handler.
    trap - INT TERM
    exit "$exit_code"
}

trap stop_server EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

for CONFIG in "${CONFIGS[@]}"; do
    if [[ ! -x "${SCRIPT_DIR}/configs/${CONFIG}.sh" ]]; then
        echo "Unknown or non-executable benchmark config: ${CONFIG}" >&2
        exit 2
    fi
done

# Preflight: bail out if the selected HTTP port is already in use.
wait_for_port_free "$SERVER_PORT"

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/${SWEEP_TS}"
echo "Sweep outputs: ${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    echo "=== Running $CONFIG ==="
    wait_for_idle_gpus "$(gpu_count_for_config "$CONFIG")"
    launch_server "$CONFIG"

    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    echo "Serving phase..."
    evalscope perf \
        --model amd/Kimi-K2.5-MXFP4 \
        --url "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
        --api openai \
        --dataset swe_smith \
        --dataset-path agentic_dataset.json \
        --max-tokens 500 \
        --multi-turn \
        "${EVALSCOPE_TURN_ARGS[@]}" \
        --number 2 \
        --parallel 2 \
        --extra-args '{"ignore_eos": true}' \
        --dataset-offset 68 \
        --outputs-dir /tmp/outputs

    echo "Agentic sweep..."
    evalscope perf \
        --model amd/Kimi-K2.5-MXFP4 \
        --url "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
        --api openai \
        --dataset swe_smith \
        --dataset-path agentic_dataset.json \
        --max-tokens 500 \
        --multi-turn \
        "${EVALSCOPE_TURN_ARGS[@]}" \
        --number 4 8 8 16 32 \
        --parallel 1 2 4 8 16 \
        --extra-args '{"ignore_eos": true}' \
        --name "$CONFIG" \
        --outputs-dir "$SWEEP_DIR" \
        --no-timestamp

    stop_server
    wait_for_port_free "$SERVER_PORT"
done
