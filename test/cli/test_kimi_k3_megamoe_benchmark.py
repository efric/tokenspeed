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

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import pickle
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tokenspeed.cli._argsplit import split_argv

_SCRIPT = Path(__file__).resolve().parents[2] / "tools/megamoe/kimi_k3_e2e_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("kimi_k3_e2e_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def _gpu_report() -> dict[str, dict[str, str]]:
    return {
        f"card{index}": {
            "GFX Version": "gfx950",
            "GPU use (%)": "0",
            "GPU Memory Allocated (VRAM%)": "0",
        }
        for index in range(8)
    }


def _summary() -> dict[str, float | int]:
    return {
        "Concurrency": 1,
        "Total Requests": 1,
        "Success Requests": 1,
        "Failed Requests": 0,
        "Avg Input Tokens": 4096.0,
        "Avg Output Tokens": 1024.0,
        "Output Throughput (tok/s)": 61.25,
        "TTFT (ms)": 500.0,
        "TPOT (ms)": 16.0,
        "Avg Latency (s)": 16.72,
    }


def _write_result_db(
    directory: Path,
    *,
    prompt: str = "same measured prompt",
    fragments: tuple[str, ...] = ("hel", "lo", ""),
    response_id: str = "response-a",
    created: int = 1,
    rows: int = 1,
) -> Path:
    path = directory / "benchmark_data.db"
    messages = [
        {
            "id": response_id,
            "created": created,
            "object": "text_completion",
            "choices": [
                {
                    "index": 0,
                    "text": fragment,
                    "finish_reason": "length" if index == len(fragments) - 1 else None,
                }
            ],
        }
        for index, fragment in enumerate(fragments)
    ]
    request = json.dumps(
        {
            "prompt": prompt,
            "model": "kimi-k3",
            "seed": 1,
            "stream": True,
        }
    )
    response = base64.b64encode(pickle.dumps(messages)).decode("ascii")
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE result(request TEXT, response_messages TEXT, "
            "success INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER)"
        )
        connection.executemany(
            "INSERT INTO result VALUES (?, ?, 1, 4096, 1024)",
            [(request, response)] * rows,
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _run_metrics(
    arm: str,
    repetition: int,
    throughput: float,
    *,
    prompt_sha256: str = "1" * 64,
    completion_sha256: str = "2" * 64,
) -> object:
    return benchmark.RunMetrics(
        arm=arm,
        repetition=repetition,
        output_throughput_tok_s=throughput,
        ttft_ms=500.0,
        tpot_ms=16.0,
        latency_s=16.0,
        input_tokens=4096.0,
        output_tokens=1024.0,
        correctness_normalization=benchmark.CORRECTNESS_NORMALIZATION,
        prompt_sha256=prompt_sha256,
        completion_sha256=completion_sha256,
        prompt_characters=20,
        prompt_utf8_bytes=20,
        completion_characters=5,
        completion_utf8_bytes=5,
        response_message_count=3,
        completion_fragment_count=3,
    )


def test_megamoe_flag_routes_only_to_engine() -> None:
    split = split_argv([str(benchmark.MODEL), "--tp", "8", benchmark.FLAG])
    assert benchmark.FLAG in split.engine
    assert benchmark.FLAG not in split.gateway


def test_server_arms_differ_by_only_experimental_flag() -> None:
    flag_off = benchmark.build_server_command(False)
    flag_on = benchmark.build_server_command(True)
    assert flag_on == [*flag_off, benchmark.FLAG]
    assert flag_off[:2] == [str(benchmark.TS), "serve"]
    assert "--sampling-backend" in flag_off
    assert flag_off[flag_off.index("--sampling-backend") + 1] == "greedy"
    assert flag_off[flag_off.index("--tp") + 1] == "8"
    assert "--no-enable-prefix-caching" in flag_off
    assert flag_off[flag_off.index("--cudagraph-capture-sizes") + 1 :][0:5] == [
        "1",
        "2",
        "4",
        "8",
        "16",
    ]


def test_benchmark_env_forces_fatal_epoch_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/wrong-checkout")
    monkeypatch.setenv("TOKENSPEED_K3_MEGAMOE_FATAL_EPOCH_D2H", "0")
    env = benchmark._benchmark_env()
    assert "PYTHONPATH" not in env
    assert env["TOKENSPEED_K3_MEGAMOE_FATAL_EPOCH_D2H"] == "1"


def _cli_contract_report(*, prefix_caching_arm: str | None = None) -> dict:
    flag_off = benchmark.build_server_command(False)[2:]
    flag_on = benchmark.build_server_command(True)[2:]
    report = {
        "flag_off": {
            "engine": flag_off,
            "gateway": ["same-gateway"],
            "flag": False,
            "disable_overlap_schedule": False,
            "enable_prefix_caching": False,
            "load_format": "auto",
            "world_size": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
        },
        "flag_on": {
            "engine": flag_on,
            "gateway": ["same-gateway"],
            "flag": True,
            "disable_overlap_schedule": False,
            "enable_prefix_caching": False,
            "load_format": "auto",
            "world_size": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
        },
    }
    if prefix_caching_arm is not None:
        report[prefix_caching_arm]["enable_prefix_caching"] = True
    return report


def test_cli_contract_records_disabled_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _cli_contract_report()
    monkeypatch.setattr(
        benchmark,
        "_run_checked",
        lambda *args, **kwargs: type("Result", (), {"stdout": ""})(),
    )
    monkeypatch.setattr(benchmark, "_json_sentinel", lambda *args: report)

    result = benchmark._probe_cli_contract({})
    assert result["flag_off"]["enable_prefix_caching"] is False
    assert result["flag_on"]["enable_prefix_caching"] is False


def test_manifest_retains_resolved_prefix_cache_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _cli_contract_report()
    monkeypatch.setattr(benchmark, "_benchmark_env", lambda: {})
    monkeypatch.setattr(benchmark, "_utc_now", lambda: "timestamp")
    monkeypatch.setattr(benchmark, "_git", lambda _args: "git-value")
    monkeypatch.setattr(benchmark, "_source_fingerprint", lambda: "source-hash")

    manifest = benchmark._manifest(
        {"cli_contract": report},
        ("flag-off", "flag-on"),
    )
    recorded = manifest["preflight"]["cli_contract"]
    assert recorded["flag_off"]["enable_prefix_caching"] is False
    assert recorded["flag_on"]["enable_prefix_caching"] is False


@pytest.mark.parametrize("arm", ("flag_off", "flag_on"))
def test_cli_contract_rejects_resolved_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    report = _cli_contract_report(prefix_caching_arm=arm)
    monkeypatch.setattr(
        benchmark,
        "_run_checked",
        lambda *args, **kwargs: type("Result", (), {"stdout": ""})(),
    )
    monkeypatch.setattr(benchmark, "_json_sentinel", lambda *args: report)

    with pytest.raises(
        benchmark.BenchmarkError, match="unexpected parsed CLI contract"
    ):
        benchmark._probe_cli_contract({})


def test_eval_command_is_exact_batch_one_three_run_payload(tmp_path: Path) -> None:
    command = benchmark.build_eval_command(tmp_path)
    assert command[:2] == [str(benchmark.EVALSCOPE), "perf"]
    expected_pairs = {
        "--model": "kimi-k3",
        "--min-prompt-length": "4096",
        "--max-prompt-length": "4096",
        "--min-tokens": "1024",
        "--max-tokens": "1024",
        "--parallel": "1",
        "--number": "1",
        "--warmup-num": "1",
        "--seed": "1",
    }
    for flag, value in expected_pairs.items():
        assert command[command.index(flag) + 1] == value
    assert command[-3:] == ["--outputs-dir", str(tmp_path), "--no-timestamp"]
    assert benchmark.REPETITIONS == 3


def test_gpu_gate_requires_exact_idle_gfx950_set() -> None:
    assert len(benchmark.validate_gpu_report(_gpu_report())) == 8

    busy = _gpu_report()
    busy["card3"]["GPU use (%)"] = "1"
    with pytest.raises(benchmark.BenchmarkError, match="card3 is busy"):
        benchmark.validate_gpu_report(busy)

    missing = _gpu_report()
    del missing["card7"]
    with pytest.raises(benchmark.BenchmarkError, match="exactly card0..card7"):
        benchmark.validate_gpu_report(missing)


def test_gpu_gate_waits_for_prior_context_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idle = _gpu_report()
    results = iter((benchmark.BenchmarkError("card0 is busy: GPU use=1%"), idle))

    def query(_env: dict[str, str]) -> dict[str, object]:
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(benchmark, "_query_gpu_report", query)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    assert benchmark._wait_for_idle_gpu_report({}) is idle


def test_evalscope_summary_fails_closed_on_request_failure(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_summary.json"
    _write_result_db(tmp_path)
    path.write_text(json.dumps(_summary()), encoding="utf-8")
    metrics = benchmark.parse_evalscope_summary(path, arm="flag-on", repetition=1)
    assert metrics.output_throughput_tok_s == 61.25
    assert metrics.output_tokens == 1024.0
    assert metrics.completion_sha256 == hashlib.sha256(b"hello").hexdigest()

    failed = _summary()
    failed["Success Requests"] = 0
    failed["Failed Requests"] = 1
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="Success Requests"):
        benchmark.parse_evalscope_summary(path, arm="flag-on", repetition=1)


def test_correctness_fingerprint_ignores_transport_metadata(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = benchmark.fingerprint_evalscope_result_db(
        _write_result_db(first_dir, response_id="response-a", created=1)
    )
    second = benchmark.fingerprint_evalscope_result_db(
        _write_result_db(second_dir, response_id="response-b", created=999999)
    )
    assert first.prompt_sha256 == second.prompt_sha256
    assert first.completion_sha256 == second.completion_sha256
    assert first.prompt_sha256 == hashlib.sha256(b"same measured prompt").hexdigest()
    assert first.completion_sha256 == hashlib.sha256(b"hello").hexdigest()
    assert first.completion_characters == 5
    assert first.completion_utf8_bytes == 5
    assert first.response_message_count == 3
    assert first.completion_fragment_count == 3


def test_correctness_fingerprint_rejects_multiple_measured_rows(tmp_path: Path) -> None:
    path = _write_result_db(tmp_path, rows=2)
    with pytest.raises(benchmark.BenchmarkError, match="one measured row"):
        benchmark.fingerprint_evalscope_result_db(path)


def test_correctness_decoder_rejects_pickle_globals() -> None:
    encoded = base64.b64encode(pickle.dumps([Path("/tmp/not-loaded")])).decode("ascii")
    with pytest.raises(benchmark.BenchmarkError, match="primitive-only pickle"):
        benchmark._decode_response_messages(encoded)


def test_server_log_requires_activation_and_rejects_fatal_epoch(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(benchmark.ACTIVATION_MARKER, encoding="utf-8")
    benchmark.scan_server_log(server_log, enabled=True)

    server_log.write_text("ordinary server", encoding="utf-8")
    benchmark.scan_server_log(server_log, enabled=False)
    with pytest.raises(benchmark.BenchmarkError, match="activation marker"):
        benchmark.scan_server_log(server_log, enabled=True)

    server_log.write_text(
        benchmark.ACTIVATION_MARKER
        + "\nKimi-K3 MegaMoE reported fatal epoch 7; discard output",
        encoding="utf-8",
    )
    with pytest.raises(benchmark.BenchmarkError, match="fatal MegaMoE log"):
        benchmark.scan_server_log(server_log, enabled=True)

    server_log.write_text(
        benchmark.ACTIVATION_MARKER
        + "\nKimi-K3 MegaMoE per-result fatal-epoch D2H check is disabled",
        encoding="utf-8",
    )
    with pytest.raises(benchmark.BenchmarkError, match="fatal MegaMoE log"):
        benchmark.scan_server_log(server_log, enabled=True)


def test_comparison_reports_causal_and_historical_wins() -> None:
    runs = []
    for arm, values in (
        ("flag-off", (57.0, 58.0, 59.0)),
        ("flag-on", (60.0, 61.0, 62.0)),
    ):
        for repetition, throughput in enumerate(values, 1):
            runs.append(_run_metrics(arm, repetition, throughput))
    result = benchmark.summarize_results(runs)
    assert result["arms"]["flag-off"]["median_output_throughput_tok_s"] == 58.0
    assert result["arms"]["flag-on"]["median_output_throughput_tok_s"] == 61.0
    assert result["causal"]["flag_on_beats_flag_off"] is True
    assert result["causal"]["flag_on_beats_historical"] is True
    assert result["correctness"]["run_count"] == 6
    assert result["correctness"]["prompt_identity"] is True
    assert result["correctness"]["completion_identity"] is True


def test_correctness_identity_rejects_prompt_or_completion_drift() -> None:
    off = _run_metrics("flag-off", 1, 58.0)
    on = _run_metrics("flag-on", 1, 61.0)
    benchmark.validate_correctness_identity((off, on))

    with pytest.raises(benchmark.BenchmarkError, match="prompts differ"):
        benchmark.validate_correctness_identity(
            (off, replace(on, prompt_sha256="3" * 64))
        )
    with pytest.raises(benchmark.BenchmarkError, match="completions differ"):
        benchmark.validate_correctness_identity(
            (off, replace(on, completion_sha256="4" * 64))
        )
