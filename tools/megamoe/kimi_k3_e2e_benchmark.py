# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run the qualified Kimi K3 batch-one MegaMoE throughput comparison.

The benchmark intentionally reproduces the graph-enabled M=1 serving and
EvalScope commands recorded in the August 13, 2026 historical artifact.  It
runs one current-checkout flag-off server and one flag-on server, with three
EvalScope repetitions per arm.  The only server-argv difference is
``--enable-kimi-k3-megamoe``.

The system ROCr object must be loaded before PyTorch.  This orchestrator pins
``LD_PRELOAD`` for every child instead of importing the runtime in its own
process.  It also refuses to launch while the implementation marker is false,
when the exact eight-gfx950 idle gate fails, or when the CLI does not propagate
the experimental flag to the engine exactly as expected.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import pickle
import re
import shlex
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/data/models/moonshotai-Kimi-K3")
VENV = Path("/home/ericfeng/distributed/.venvs/tokenspeed")
TS = VENV / "bin" / "ts"
PYTHON = VENV / "bin" / "python"
EVALSCOPE = Path("/tmp/evalscope-perf/bin/evalscope")
SYSTEM_ROCR = Path("/opt/rocm/lib/libhsa-runtime64.so.1")
ROCM_SMI = Path("/opt/rocm/bin/rocm-smi")
HOST = "127.0.0.1"
PORT = 8000
CONTROL_PORT = 8001
STARTUP_TIMEOUT_SECONDS = 1800
REPETITIONS = 3
CORRECTNESS_NORMALIZATION = "evalscope-sqlite-openai-completions-v1"
_MAX_RESULT_DB_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_MESSAGES = 65536
_MAX_SEMANTIC_TEXT_CHARACTERS = 16 * 1024 * 1024

HISTORICAL_ROOT = (
    REPO_ROOT / "profile_default" / "kimi-k3-main-04bc0864-graph-20260813T062945Z"
)
HISTORICAL_COMMIT = "04bc08649f3e53e19144ee86564be7c6121c99d2"
HISTORICAL_COMMANDS_SHA256 = (
    "8c0dcc3aab45f89574d634ba806503a7fca11118655affd5f9ddc9d51ada3681"
)
HISTORICAL_M1_RUNS = (56.2040, 59.0891, 58.0592)
HISTORICAL_M1_MEDIAN = 58.0592

FLAG = "--enable-kimi-k3-megamoe"
ACTIVATION_MARKER = (
    "Kimi-K3 MegaMoE disabled overlap scheduling so each fatal epoch is "
    "checked before the next graph replay is enqueued"
)
FATAL_LOG_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"Kimi-K3 MegaMoE reported fatal epoch\s+[1-9][0-9]*",
        r"MegaMoE warmup poisoned epoch\s+[1-9][0-9]*",
        r"Kimi K3 MegaMoE compile consensus returned after a local failure",
        r"Kimi K3 MegaMoE warmup consensus returned after a local failure",
        r"Kimi-K3 MegaMoE admission consensus returned unexpectedly",
        r"gfx950 MegaMoE implementation is not complete",
    )
)

SERVER_ARGUMENTS = (
    str(MODEL),
    "--served-model-name",
    "kimi-k3",
    "--tp",
    "8",
    "--enable-expert-parallel",
    "--trust-remote-code",
    "--quantization",
    "compressed-tensors",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--attention-backend",
    "mla",
    "--moe-backend",
    "auto",
    "--sampling-backend",
    "greedy",
    "--seed",
    "1013305063",
    "--gpu-memory-utilization",
    "0.92",
    "--max-model-len",
    "262144",
    "--max-total-tokens",
    "262144",
    "--max-num-seqs",
    "32",
    "--chunked-prefill-size",
    "8192",
    "--disable-kvstore",
    # Every repetition intentionally uses the same seeded prompt so its
    # semantic fingerprint can be compared byte-for-byte across both arms.
    # Disable local prefix reuse to keep each repetition independent and to
    # prevent a later repetition from taking a cached-history attention path
    # that the first repetition did not exercise.
    "--no-enable-prefix-caching",
    "--cudagraph-capture-sizes",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--max-cudagraph-capture-size",
    "16",
    "--disable-prefill-graph",
    "--mm-encoder-tp-mode",
    "data",
    "--host",
    HOST,
    "--port",
    str(PORT),
    "--engine-startup-timeout",
    str(STARTUP_TIMEOUT_SECONDS),
)

EVAL_ARGUMENTS = (
    "perf",
    "--model",
    "kimi-k3",
    "--url",
    f"http://{HOST}:{PORT}/v1/completions",
    "--api",
    "openai",
    "--tokenizer-path",
    str(MODEL),
    "--dataset",
    "random",
    "--min-prompt-length",
    "4096",
    "--max-prompt-length",
    "4096",
    "--min-tokens",
    "1024",
    "--max-tokens",
    "1024",
    "--temperature",
    "0",
    "--parallel",
    "1",
    "--number",
    "1",
    "--warmup-num",
    "1",
    "--stream",
    "--extra-args",
    '{"ignore_eos":true}',
    "--seed",
    "1",
    "--total-timeout",
    "21600",
)


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark admission or execution error."""


@dataclass(frozen=True)
class CorrectnessFingerprint:
    """Normalized semantic content extracted from one measured request."""

    correctness_normalization: str
    prompt_sha256: str
    completion_sha256: str
    prompt_characters: int
    prompt_utf8_bytes: int
    completion_characters: int
    completion_utf8_bytes: int
    response_message_count: int
    completion_fragment_count: int


@dataclass(frozen=True)
class RunMetrics:
    """Validated metrics from one EvalScope M=1 repetition."""

    arm: str
    repetition: int
    output_throughput_tok_s: float
    ttft_ms: float
    tpot_ms: float
    latency_s: float
    input_tokens: float
    output_tokens: float
    correctness_normalization: str
    prompt_sha256: str
    completion_sha256: str
    prompt_characters: int
    prompt_utf8_bytes: int
    completion_characters: int
    completion_utf8_bytes: int
    response_message_count: int
    completion_fragment_count: int


class _PrimitiveUnpickler(pickle.Unpickler):
    """Decode EvalScope's primitive container without importing pickle globals."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"global pickle object is forbidden: {module}.{name}"
        )

    def persistent_load(self, pid: object) -> object:
        raise pickle.UnpicklingError(f"persistent pickle ID is forbidden: {pid!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    roots = (
        REPO_ROOT / "python",
        REPO_ROOT / "tokenspeed-kernel" / "python",
        REPO_ROOT / "tokenspeed-kernel-amd" / "python",
    )
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "little"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _benchmark_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["LD_PRELOAD"] = str(SYSTEM_ROCR)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_server_command(enabled: bool) -> list[str]:
    """Return the exact historical server command plus the optional flag."""

    command = [str(TS), "serve", *SERVER_ARGUMENTS]
    if enabled:
        command.append(FLAG)
    return command


def build_eval_command(output_dir: Path) -> list[str]:
    """Return the exact historical EvalScope M=1 command for one output path."""

    return [
        str(EVALSCOPE),
        *EVAL_ARGUMENTS,
        "--outputs-dir",
        str(output_dir),
        "--no-timestamp",
    ]


def _run_checked(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        raise BenchmarkError(
            f"command failed with rc={error.returncode}: {shlex.join(command)}"
            + (f"\n{stderr}" if stderr else "")
        ) from error


def _require_paths() -> None:
    required_files = (TS, PYTHON, EVALSCOPE, SYSTEM_ROCR, ROCM_SMI)
    for path in required_files:
        if not path.is_file():
            raise BenchmarkError(f"required executable/object is missing: {path}")
    if not MODEL.is_dir():
        raise BenchmarkError(f"Kimi K3 model directory is missing: {MODEL}")
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise BenchmarkError(
            f"run with the qualified TokenSpeed venv {VENV}; sys.prefix={sys.prefix}"
        )


def _verify_historical_artifact() -> dict[str, object]:
    commands = HISTORICAL_ROOT / "COMMANDS.md"
    if _sha256_file(commands) != HISTORICAL_COMMANDS_SHA256:
        raise BenchmarkError(f"historical COMMANDS.md changed unexpectedly: {commands}")
    all_runs = HISTORICAL_ROOT / "benchmark" / "all-runs.csv"
    with all_runs.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["concurrency"] == "1"]
    observed = tuple(float(row["output_throughput_tok_s"]) for row in rows)
    if observed != HISTORICAL_M1_RUNS:
        raise BenchmarkError(
            f"historical M=1 runs changed: expected {HISTORICAL_M1_RUNS}, got {observed}"
        )
    if statistics.median(observed) != HISTORICAL_M1_MEDIAN:
        raise BenchmarkError(
            "historical M=1 median no longer matches the qualified value"
        )
    return {
        "artifact": str(HISTORICAL_ROOT),
        "commit": HISTORICAL_COMMIT,
        "commands_sha256": HISTORICAL_COMMANDS_SHA256,
        "m1_runs": observed,
        "m1_median_output_throughput_tok_s": HISTORICAL_M1_MEDIAN,
    }


def _probe_evalscope_version(env: dict[str, str]) -> str:
    result = _run_checked(
        (str(EVALSCOPE), "--version"),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = "\n".join((result.stdout, result.stderr)).strip()
    if "evalscope 1.9.1" not in output:
        raise BenchmarkError(f"requires EvalScope 1.9.1, got {output!r}")
    return output


def _json_sentinel(output: str, sentinel: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith(sentinel):
            value = json.loads(line.removeprefix(sentinel))
            if not isinstance(value, dict):
                break
            return value
    raise BenchmarkError(f"probe did not emit {sentinel!r}:\n{output[-4000:]}")


def _probe_cli_contract(env: dict[str, str]) -> dict[str, object]:
    code = r"""
import json
import sys
from tokenspeed.cli._argsplit import split_argv
from tokenspeed.runtime.utils.server_args import prepare_server_args

def inspect(argv):
    split = split_argv(argv)
    parsed = prepare_server_args(split.engine)
    return {
        "engine": split.engine,
        "gateway": split.gateway,
        "flag": parsed.enable_kimi_k3_megamoe,
        "disable_overlap_schedule": parsed.disable_overlap_schedule,
        "enable_prefix_caching": parsed.enable_prefix_caching,
        "load_format": parsed.load_format,
        "world_size": parsed.mapping.world_size,
        "moe_tp_size": parsed.mapping.moe.tp_size,
        "moe_ep_size": parsed.mapping.moe.ep_size,
    }

off = json.loads(sys.argv[1])
on = json.loads(sys.argv[2])
print("KIMI_CLI_CONTRACT=" + json.dumps({"flag_off": inspect(off), "flag_on": inspect(on)}))
"""
    off = build_server_command(False)[2:]
    on = build_server_command(True)[2:]
    result = _run_checked(
        (str(PYTHON), "-c", code, json.dumps(off), json.dumps(on)),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    report = _json_sentinel(result.stdout, "KIMI_CLI_CONTRACT=")
    flag_off = report["flag_off"]
    flag_on = report["flag_on"]
    if not isinstance(flag_off, dict) or not isinstance(flag_on, dict):
        raise BenchmarkError("CLI contract probe returned malformed arms")
    if flag_off["gateway"] != flag_on["gateway"]:
        raise BenchmarkError("MegaMoE flag unexpectedly changed gateway arguments")
    expected_on_engine = [*flag_off["engine"], FLAG]
    if flag_on["engine"] != expected_on_engine:
        raise BenchmarkError(
            "enabled engine arguments differ by more than the MegaMoE flag"
        )
    expected = {
        "flag_off": (False, False),
        "flag_on": (True, True),
    }
    for arm, (flag, overlap) in expected.items():
        parsed = report[arm]
        if (
            parsed["flag"] is not flag
            or parsed["disable_overlap_schedule"] is not overlap
            or parsed.get("enable_prefix_caching") is not False
            or parsed["load_format"] != "auto"
            or parsed["world_size"] != 8
            or parsed["moe_tp_size"] != 1
            or parsed["moe_ep_size"] != 8
        ):
            raise BenchmarkError(f"unexpected parsed CLI contract for {arm}: {parsed}")
    return report


def _probe_megamoe_runtime(env: dict[str, str]) -> dict[str, object]:
    code = r"""
import json
from tokenspeed_kernel import kimi_k3_megamoe_available
from tokenspeed_kernel_amd.ops.gfx950.moe import megamoe

runtime = megamoe.preflight_kimi_k3_megamoe_runtime().as_dict()
print("KIMI_MEGAMOE_PREFLIGHT=" + json.dumps({
    "implementation_complete": megamoe.KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE,
    "public_available": kimi_k3_megamoe_available(),
    "runtime": runtime,
}))
"""
    result = _run_checked(
        (str(PYTHON), "-c", code),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return _json_sentinel(result.stdout, "KIMI_MEGAMOE_PREFLIGHT=")


def validate_gpu_report(report: object) -> dict[str, dict[str, str]]:
    """Validate the exact eight-idle-gfx950 hardware gate."""

    if not isinstance(report, dict):
        raise BenchmarkError("rocm-smi JSON is not an object")
    cards = {
        name: value
        for name, value in report.items()
        if re.fullmatch(r"card[0-9]+", str(name))
    }
    if len(cards) != 8 or sorted(cards) != [f"card{index}" for index in range(8)]:
        raise BenchmarkError(f"requires exactly card0..card7, got {sorted(cards)}")
    for name, card in cards.items():
        if not isinstance(card, dict):
            raise BenchmarkError(f"{name} rocm-smi record is malformed")
        if card.get("GFX Version") != "gfx950":
            raise BenchmarkError(f"{name} is not gfx950: {card.get('GFX Version')!r}")
        if int(card.get("GPU use (%)", -1)) != 0:
            raise BenchmarkError(f"{name} is busy: GPU use={card.get('GPU use (%)')}%")
        allocated = card.get("GPU Memory Allocated (VRAM%)")
        if allocated is None:
            raise BenchmarkError(
                f"{name} lacks the VRAM allocation percentage in rocm-smi output"
            )
        if int(allocated) != 0:
            raise BenchmarkError(f"{name} has allocated VRAM: {allocated}%")
    return cards


def _query_gpu_report(env: dict[str, str]) -> dict[str, object]:
    result = _run_checked(
        (
            str(ROCM_SMI),
            "--showproductname",
            "--showuse",
            "--showmemuse",
            "--json",
        ),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"invalid rocm-smi JSON: {error}") from error
    validate_gpu_report(report)
    return report


def _wait_for_idle_gpu_report(
    env: dict[str, str], *, timeout_seconds: float = 120.0
) -> dict[str, object]:
    """Wait out context teardown, then return one fully idle hardware gate."""

    deadline = time.monotonic() + timeout_seconds
    last_error: BenchmarkError | None = None
    while True:
        try:
            return _query_gpu_report(env)
        except BenchmarkError as error:
            # A malformed/non-gfx inventory cannot heal. Utilization and VRAM
            # can remain nonzero briefly after a prior server or probe exits.
            if " is busy:" not in str(error) and " has allocated VRAM:" not in str(
                error
            ):
                raise
            last_error = error
        if time.monotonic() >= deadline:
            assert last_error is not None
            raise BenchmarkError(
                f"GPU idle gate did not clear within {timeout_seconds:.0f}s: {last_error}"
            ) from last_error
        time.sleep(2.0)


def _check_ports_free() -> None:
    for port in (PORT, CONTROL_PORT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, port))
            except OSError as error:
                raise BenchmarkError(
                    f"required port {HOST}:{port} is in use"
                ) from error


def _preflight(include_megamoe: bool) -> dict[str, object]:
    _require_paths()
    env = _benchmark_env()
    report: dict[str, object] = {
        "timestamp_utc": _utc_now(),
        "historical": _verify_historical_artifact(),
        "evalscope_version": _probe_evalscope_version(env),
        "cli_contract": _probe_cli_contract(env),
        "system_rocr": {
            "path": str(SYSTEM_ROCR.resolve()),
            "sha256": _sha256_file(SYSTEM_ROCR),
        },
    }
    if include_megamoe:
        megamoe = _probe_megamoe_runtime(env)
        report["megamoe"] = megamoe
        if megamoe.get("implementation_complete") is not True:
            raise BenchmarkError(
                "refusing full-model launch: "
                "KIMI_K3_MEGAMOE_IMPLEMENTATION_COMPLETE is false"
            )
        if megamoe.get("public_available") is not True:
            raise BenchmarkError(
                "refusing full-model launch: kimi_k3_megamoe_available() is false"
            )
    _check_ports_free()
    report["gpu_gate"] = _wait_for_idle_gpu_report(env)
    return report


def _wait_ready(process: subprocess.Popen, server_log: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    readiness = f"http://{HOST}:{PORT}/readiness"
    next_heartbeat = time.monotonic()
    last_error = "not probed"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise BenchmarkError(
                f"server exited during startup with rc={returncode}\n{tail}"
            )
        try:
            with urllib.request.urlopen(readiness, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        now = time.monotonic()
        if now >= next_heartbeat:
            elapsed = STARTUP_TIMEOUT_SECONDS - max(deadline - now, 0.0)
            print(
                f"waiting for server readiness: {elapsed:.0f}s elapsed ({last_error})",
                flush=True,
            )
            next_heartbeat = now + 30.0
        time.sleep(1.0)
    raise BenchmarkError(
        f"server did not reach {readiness} within {STARTUP_TIMEOUT_SECONDS}s"
    )


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 120.0),
        (signal.SIGTERM, 30.0),
        (signal.SIGKILL, 10.0),
    ):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        raise BenchmarkError(f"could not stop server process group {process.pid}")


@contextlib.contextmanager
def _server(
    arm_dir: Path,
    *,
    enabled: bool,
    env: dict[str, str],
) -> Iterator[tuple[subprocess.Popen, Path]]:
    command = build_server_command(enabled)
    command_path = arm_dir / "server-command.txt"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    server_log = arm_dir / "server.log"
    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(process, server_log)
            yield process, server_log
        finally:
            _stop_process_group(process)


def scan_server_log(server_log: Path, *, enabled: bool) -> None:
    """Check activation and all known MegaMoE fail-stop messages."""

    text = server_log.read_text(encoding="utf-8", errors="replace")
    marker_present = ACTIVATION_MARKER in text
    if marker_present is not enabled:
        expectation = "present" if enabled else "absent"
        raise BenchmarkError(
            f"MegaMoE activation marker must be {expectation} in {server_log}"
        )
    for pattern in FATAL_LOG_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise BenchmarkError(
                f"fatal MegaMoE log marker in {server_log}: {match.group(0)}"
            )


def _utf8_fingerprint(text: str, *, label: str) -> tuple[str, int, int]:
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise BenchmarkError(f"EvalScope {label} is not valid UTF-8 text") from error
    return hashlib.sha256(encoded).hexdigest(), len(text), len(encoded)


def _decode_response_messages(encoded: object) -> list[object]:
    if not isinstance(encoded, str):
        raise BenchmarkError("EvalScope response_messages must be base64 text")
    if len(encoded) > _MAX_RESULT_DB_BYTES * 2:
        raise BenchmarkError("EvalScope response_messages exceeds the safety bound")
    try:
        serialized = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise BenchmarkError("EvalScope response_messages is invalid base64") from error
    if len(serialized) > _MAX_RESULT_DB_BYTES:
        raise BenchmarkError(
            "decoded EvalScope response_messages exceeds the safety bound"
        )
    try:
        messages = _PrimitiveUnpickler(io.BytesIO(serialized)).load()
    except (EOFError, pickle.UnpicklingError, ValueError) as error:
        raise BenchmarkError(
            "EvalScope response_messages is not a primitive-only pickle"
        ) from error
    if not isinstance(messages, list):
        raise BenchmarkError("EvalScope response_messages must decode to a list")
    if not 0 < len(messages) <= _MAX_RESPONSE_MESSAGES:
        raise BenchmarkError(
            f"EvalScope response message count is invalid: {len(messages)}"
        )
    return messages


def fingerprint_evalscope_result_db(path: Path) -> CorrectnessFingerprint:
    """Hash the measured prompt and ordered streamed completion text.

    EvalScope persists transport chunks with request IDs and creation times.
    This normalization deliberately hashes only ``request.prompt`` and the
    ordered concatenation of ``choices[0].text``. IDs, timestamps, model names,
    finish reasons, and other transport metadata do not affect the hashes.
    """

    if not path.is_file():
        raise BenchmarkError(f"EvalScope did not produce result database: {path}")
    if path.stat().st_size > _MAX_RESULT_DB_BYTES:
        raise BenchmarkError(
            f"EvalScope result database exceeds the safety bound: {path}"
        )
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        try:
            rows = connection.execute(
                "SELECT request, response_messages, success, prompt_tokens, "
                "completion_tokens FROM result"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise BenchmarkError(
            f"cannot read EvalScope result database {path}: {error}"
        ) from error
    if len(rows) != 1:
        raise BenchmarkError(
            f"EvalScope result database must contain one measured row, got {len(rows)}"
        )
    request_raw, response_raw, success, prompt_tokens, completion_tokens = rows[0]
    if success != 1 or prompt_tokens != 4096 or completion_tokens != 1024:
        raise BenchmarkError(
            "EvalScope measured row must be successful 4096->1024; got "
            f"success={success}, prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}"
        )
    if not isinstance(request_raw, str):
        raise BenchmarkError("EvalScope request must be JSON text")
    try:
        request = json.loads(request_raw)
    except json.JSONDecodeError as error:
        raise BenchmarkError("EvalScope request is invalid JSON") from error
    if not isinstance(request, dict) or not isinstance(request.get("prompt"), str):
        raise BenchmarkError("EvalScope request.prompt must be text")
    prompt = request["prompt"]
    if len(prompt) > _MAX_SEMANTIC_TEXT_CHARACTERS:
        raise BenchmarkError("EvalScope request.prompt exceeds the safety bound")

    fragments: list[str] = []
    completion_characters = 0
    messages = _decode_response_messages(response_raw)
    for index, raw_message in enumerate(messages):
        message = raw_message
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError as error:
                raise BenchmarkError(
                    f"EvalScope response message {index} is invalid JSON"
                ) from error
        if not isinstance(message, dict):
            raise BenchmarkError(f"EvalScope response message {index} is not an object")
        choices = message.get("choices")
        if choices is None and isinstance(message.get("usage"), dict):
            continue
        if not isinstance(choices, list) or len(choices) != 1:
            raise BenchmarkError(
                f"EvalScope response message {index} must contain one choice"
            )
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise BenchmarkError(
                f"EvalScope response message {index} has an invalid choice"
            )
        fragment = choice.get("text")
        if not isinstance(fragment, str):
            raise BenchmarkError(
                f"EvalScope response message {index} choice.text must be text"
            )
        completion_characters += len(fragment)
        if completion_characters > _MAX_SEMANTIC_TEXT_CHARACTERS:
            raise BenchmarkError("EvalScope completion text exceeds the safety bound")
        fragments.append(fragment)
    if not fragments:
        raise BenchmarkError("EvalScope response contains no completion fragments")
    completion = "".join(fragments)
    if not completion:
        raise BenchmarkError("EvalScope response contains an empty completion")
    prompt_hash, prompt_characters, prompt_bytes = _utf8_fingerprint(
        prompt, label="prompt"
    )
    completion_hash, completion_characters, completion_bytes = _utf8_fingerprint(
        completion, label="completion"
    )
    return CorrectnessFingerprint(
        correctness_normalization=CORRECTNESS_NORMALIZATION,
        prompt_sha256=prompt_hash,
        completion_sha256=completion_hash,
        prompt_characters=prompt_characters,
        prompt_utf8_bytes=prompt_bytes,
        completion_characters=completion_characters,
        completion_utf8_bytes=completion_bytes,
        response_message_count=len(messages),
        completion_fragment_count=len(fragments),
    )


def parse_evalscope_summary(path: Path, *, arm: str, repetition: int) -> RunMetrics:
    """Load one summary and its normalized SQLite correctness fingerprint."""

    if not path.is_file():
        raise BenchmarkError(f"EvalScope did not produce summary: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "Concurrency": 1,
        "Total Requests": 1,
        "Success Requests": 1,
        "Failed Requests": 0,
        "Avg Input Tokens": 4096.0,
        "Avg Output Tokens": 1024.0,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise BenchmarkError(
                f"invalid EvalScope {field}: expected {value}, got {report.get(field)}"
            )
    fingerprint = fingerprint_evalscope_result_db(path.with_name("benchmark_data.db"))
    metrics = RunMetrics(
        arm=arm,
        repetition=repetition,
        output_throughput_tok_s=float(report["Output Throughput (tok/s)"]),
        ttft_ms=float(report["TTFT (ms)"]),
        tpot_ms=float(report["TPOT (ms)"]),
        latency_s=float(report["Avg Latency (s)"]),
        input_tokens=float(report["Avg Input Tokens"]),
        output_tokens=float(report["Avg Output Tokens"]),
        **asdict(fingerprint),
    )
    if metrics.output_throughput_tok_s <= 0 or metrics.tpot_ms <= 0:
        raise BenchmarkError(f"invalid nonpositive EvalScope metrics: {metrics}")
    return metrics


def _run_repetition(
    arm: str,
    repetition: int,
    arm_dir: Path,
    process: subprocess.Popen,
    server_log: Path,
    env: dict[str, str],
) -> RunMetrics:
    output_dir = arm_dir / "benchmark" / f"r{repetition}"
    if output_dir.exists():
        raise BenchmarkError(f"refusing to overwrite repetition: {output_dir}")
    output_dir.mkdir(parents=True)
    command = build_eval_command(output_dir)
    (output_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    run_log = output_dir / "run.log"
    with run_log.open("w", encoding="utf-8") as log:
        log.write(f"BEGIN ARM={arm} REP={repetition} {_utc_now()}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(
            f"END ARM={arm} REP={repetition} STATUS={result.returncode} {_utc_now()}\n"
        )
    if result.returncode != 0:
        raise BenchmarkError(
            f"EvalScope {arm} repetition {repetition} failed with rc={result.returncode}"
        )
    if process.poll() is not None:
        raise BenchmarkError(
            f"server exited during {arm} repetition {repetition} with rc={process.returncode}"
        )
    scan_server_log(server_log, enabled=arm == "flag-on")
    summary = output_dir / "kimi-k3" / "parallel_1_number_1" / "benchmark_summary.json"
    return parse_evalscope_summary(summary, arm=arm, repetition=repetition)


def validate_correctness_identity(runs: Sequence[RunMetrics]) -> dict[str, object]:
    """Require one prompt and completion fingerprint across measured runs."""

    if not runs:
        raise BenchmarkError("correctness identity requires at least one run")
    normalizations = {run.correctness_normalization for run in runs}
    if normalizations != {CORRECTNESS_NORMALIZATION}:
        raise BenchmarkError(
            f"unexpected correctness normalization versions: {sorted(normalizations)}"
        )

    def hashes(field: str) -> dict[str, str]:
        return {
            f"{run.arm}/r{run.repetition}": str(getattr(run, field)) for run in runs
        }

    prompt_hashes = hashes("prompt_sha256")
    if len(set(prompt_hashes.values())) != 1:
        raise BenchmarkError(
            f"measured request prompts differ across runs: {prompt_hashes}"
        )
    completion_hashes = hashes("completion_sha256")
    if len(set(completion_hashes.values())) != 1:
        raise BenchmarkError(
            "measured completions differ between contemporaneous runs: "
            f"{completion_hashes}"
        )
    return {
        "normalization": CORRECTNESS_NORMALIZATION,
        "run_count": len(runs),
        "prompt_identity": True,
        "completion_identity": True,
        "prompt_sha256": next(iter(prompt_hashes.values())),
        "completion_sha256": next(iter(completion_hashes.values())),
        "per_run": {
            key: {
                "prompt_sha256": prompt_hashes[key],
                "completion_sha256": completion_hashes[key],
            }
            for key in prompt_hashes
        },
    }


def summarize_results(runs: Sequence[RunMetrics]) -> dict[str, object]:
    """Compute per-arm medians and historical/causal comparisons."""

    correctness = validate_correctness_identity(runs)
    arms: dict[str, dict[str, object]] = {}
    for arm in ("flag-off", "flag-on"):
        selected = [run for run in runs if run.arm == arm]
        if not selected:
            continue
        if len(selected) != REPETITIONS:
            raise BenchmarkError(
                f"{arm} has {len(selected)} runs, expected {REPETITIONS}"
            )
        throughputs = [run.output_throughput_tok_s for run in selected]
        median = statistics.median(throughputs)
        arms[arm] = {
            "runs_output_throughput_tok_s": throughputs,
            "runs_prompt_sha256": [run.prompt_sha256 for run in selected],
            "runs_completion_sha256": [run.completion_sha256 for run in selected],
            "median_output_throughput_tok_s": median,
            "median_ttft_ms": statistics.median(run.ttft_ms for run in selected),
            "median_tpot_ms": statistics.median(run.tpot_ms for run in selected),
            "historical_median_output_throughput_tok_s": HISTORICAL_M1_MEDIAN,
            "delta_vs_historical_pct": (median / HISTORICAL_M1_MEDIAN - 1.0) * 100.0,
        }
    comparison: dict[str, object] = {
        "historical": {
            "commit": HISTORICAL_COMMIT,
            "runs_output_throughput_tok_s": HISTORICAL_M1_RUNS,
            "median_output_throughput_tok_s": HISTORICAL_M1_MEDIAN,
        },
        "arms": arms,
        "correctness": correctness,
    }
    if "flag-off" in arms and "flag-on" in arms:
        off = float(arms["flag-off"]["median_output_throughput_tok_s"])
        on = float(arms["flag-on"]["median_output_throughput_tok_s"])
        comparison["causal"] = {
            "delta_flag_on_vs_flag_off_pct": (on / off - 1.0) * 100.0,
            "flag_on_beats_flag_off": on > off,
            "flag_on_beats_historical": on > HISTORICAL_M1_MEDIAN,
        }
    return comparison


def _write_csvs(
    output_root: Path, runs: Sequence[RunMetrics], result: dict[str, object]
) -> None:
    with (output_root / "all-runs.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(asdict(runs[0])))
        writer.writeheader()
        writer.writerows(asdict(run) for run in runs)
    arms = result["arms"]
    assert isinstance(arms, dict)
    fields = (
        "arm",
        "median_output_throughput_tok_s",
        "median_ttft_ms",
        "median_tpot_ms",
        "historical_median_output_throughput_tok_s",
        "delta_vs_historical_pct",
    )
    with (output_root / "median-of-three.csv").open(
        "w", newline="", encoding="utf-8"
    ) as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for arm, values in arms.items():
            assert isinstance(values, dict)
            writer.writerow(
                {"arm": arm, **{field: values[field] for field in fields[1:]}}
            )


def _git(command: Sequence[str]) -> str:
    result = _run_checked(
        ("git", *command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _arm_names(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("flag-off", "flag-on")
    return (selection,)


def _manifest(preflight: dict[str, object], arms: Sequence[str]) -> dict[str, object]:
    env = _benchmark_env()
    relevant_environment = {
        name: value
        for name, value in sorted(env.items())
        if name == "LD_PRELOAD"
        or name == "LD_LIBRARY_PATH"
        or name.startswith(
            ("HSA_", "HIP_", "ROCR_", "ROCM_", "TRITON_", "TORCH_", "TOKENSPEED_")
        )
    }
    return {
        "created_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "git_head": _git(("rev-parse", "HEAD")),
        "git_status": _git(("status", "--short")),
        "source_fingerprint_sha256": _source_fingerprint(),
        "preflight": preflight,
        "environment": relevant_environment,
        "arms": list(arms),
        "server_commands": {
            arm: build_server_command(arm == "flag-on") for arm in arms
        },
        "evalscope_arguments": list(EVAL_ARGUMENTS),
        "repetitions": REPETITIONS,
    }


def run_benchmark(
    output_root: Path, arms: Sequence[str], require_win: bool
) -> dict[str, object]:
    include_megamoe = "flag-on" in arms
    preflight = _preflight(include_megamoe)
    if output_root.exists():
        raise BenchmarkError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    manifest = _manifest(preflight, arms)
    _write_json(output_root / "manifest.json", manifest)
    fingerprint = str(manifest["source_fingerprint_sha256"])
    env = _benchmark_env()
    runs: list[RunMetrics] = []
    try:
        for arm in arms:
            if _source_fingerprint() != fingerprint:
                raise BenchmarkError(
                    "served Python sources changed between benchmark arms"
                )
            _check_ports_free()
            arm_dir = output_root / arm
            arm_dir.mkdir()
            gpu_report = _wait_for_idle_gpu_report(env)
            _write_json(arm_dir / "rocm-smi-prelaunch.json", gpu_report)
            enabled = arm == "flag-on"
            print(f"starting {arm} server", flush=True)
            with _server(arm_dir, enabled=enabled, env=env) as (process, server_log):
                scan_server_log(server_log, enabled=enabled)
                for repetition in range(1, REPETITIONS + 1):
                    print(
                        f"running {arm} repetition {repetition}/{REPETITIONS}",
                        flush=True,
                    )
                    run = _run_repetition(
                        arm,
                        repetition,
                        arm_dir,
                        process,
                        server_log,
                        env,
                    )
                    runs.append(run)
                    # Fail on the first divergent prompt or completion. With
                    # the default two-arm workflow this grows to the required
                    # six-run identity proof before throughput is accepted.
                    validate_correctness_identity(runs)
                    _write_json(
                        arm_dir / "completed-runs.json",
                        [asdict(value) for value in runs if value.arm == arm],
                    )
            if _source_fingerprint() != fingerprint:
                raise BenchmarkError(
                    "served Python sources changed during benchmark arm"
                )
        result = summarize_results(runs)
        _write_json(output_root / "comparison.json", result)
        _write_csvs(output_root, runs, result)
        if require_win:
            causal = result.get("causal")
            if not isinstance(causal, dict):
                raise BenchmarkError("--require-win requires both benchmark arms")
            if (
                not causal["flag_on_beats_flag_off"]
                or not causal["flag_on_beats_historical"]
            ):
                raise BenchmarkError(
                    f"MegaMoE did not satisfy the throughput win gate: {causal}"
                )
        return result
    except BaseException as error:
        _write_json(
            output_root / "failure.json",
            {
                "timestamp_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_runs": [asdict(run) for run in runs],
            },
        )
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New directory for logs and comparison artifacts (required for a run).",
    )
    parser.add_argument(
        "--arms",
        choices=("both", "flag-off", "flag-on"),
        default="both",
        help="Default runs the causal flag-off then flag-on comparison.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all gates without starting a full-model server.",
    )
    parser.add_argument(
        "--require-win",
        action="store_true",
        help="Exit nonzero unless flag-on beats both current flag-off and historical M=1.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    arms = _arm_names(args.arms)
    try:
        if args.preflight_only:
            report = _preflight("flag-on" in arms)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.output_root is None:
            raise BenchmarkError(
                "--output-root is required unless --preflight-only is used"
            )
        result = run_benchmark(args.output_root.resolve(), arms, args.require_win)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BenchmarkError as error:
        print(f"benchmark refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
