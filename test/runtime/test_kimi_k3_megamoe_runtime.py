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

"""Cheap runtime lifecycle tests for the experimental Kimi-K3 MegaMoE path."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat  # noqa: E402
from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402
from tokenspeed.runtime.engine.scheduler_control_client import (  # noqa: E402
    SchedulerControlClient,
)
from tokenspeed.runtime.execution import weight_loader  # noqa: E402
from tokenspeed.runtime.execution.cuda_graph_wrapper import (  # noqa: E402
    CudaGraphWrapper,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.model_runner import (  # noqa: E402
    ModelRunner,
    validate_kimi_k3_megamoe_fatal_epoch_slots,
)
from tokenspeed.runtime.execution.types import ModelExecutionResult  # noqa: E402
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader  # noqa: E402
from tokenspeed.runtime.models import kimi_k3  # noqa: E402
from tokenspeed.runtime.utils import env as runtime_env  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(argv)


def test_megamoe_flag_defaults_off_and_preserves_overlap_default() -> None:
    assert not _parse(["--model", "test/model"]).enable_kimi_k3_megamoe
    assert _parse(
        ["--model", "test/model", "--enable-kimi-k3-megamoe"]
    ).enable_kimi_k3_megamoe

    with mock.patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs(model="test/model", enable_kimi_k3_megamoe=True)
    assert not server_args.disable_overlap_schedule

    with mock.patch.object(ServerArgs, "__post_init__"):
        disabled = ServerArgs(
            model="test/model",
            enable_kimi_k3_megamoe=True,
            disable_overlap_schedule=True,
        )
    assert disabled.disable_overlap_schedule


def test_model_capture_and_replay_retain_one_execution_stream() -> None:
    constructor = inspect.getsource(ModelExecutor.__init__)
    stream_creation = constructor.index("self.execution_stream = torch.cuda.Stream()")
    wrapper_creation = constructor.index("self.forward_step = CudaGraphWrapper(")
    stream_injection = constructor.index("stream=self.execution_stream")
    assert stream_creation < wrapper_creation < stream_injection
    assert constructor.count("self.execution_stream = torch.cuda.Stream()") == 1
    assert "with torch.cuda.stream(self.execution_stream):" in constructor
    assert "self.forward_step.stream is not self.execution_stream" in constructor

    wrapper_constructor = inspect.getsource(CudaGraphWrapper.__init__)
    assert "self.stream = stream" in wrapper_constructor
    capture = inspect.getsource(CudaGraphWrapper.capture)
    assert "if self.stream is None:" in capture
    assert "self.stream = torch.cuda.Stream()" in capture

    autotune = inspect.getsource(ModelExecutor._autotune)
    assert "self.execution_stream.wait_stream(torch.cuda.current_stream())" in autotune
    assert "with torch.cuda.stream(self.execution_stream):" in autotune

    idle = inspect.getsource(ModelExecutor.execute_idle_forward)
    assert "current_stream.wait_stream(self.execution_stream)" in idle
    assert "self.execution_stream.wait_stream(current_stream)" in idle
    assert "with torch.cuda.stream(self.execution_stream):" in idle
    assert "self._execute_idle_forward_on_execution_stream" in idle


def test_global_args_export_flag_and_normalized_load_format() -> None:
    with mock.patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs(model="test/model", enable_kimi_k3_megamoe=True)
    server_args.load_format = LoadFormat.AUTO
    with mock.patch.dict(runtime_env.global_server_args_dict, {}, clear=False):
        runtime_env.global_server_args_dict_update(server_args)
        assert runtime_env.global_server_args_dict["enable_kimi_k3_megamoe"] is True
        assert runtime_env.global_server_args_dict["load_format"] == "auto"


def test_megamoe_loader_allowlist_and_prepared_postcondition() -> None:
    auto_config = LoadConfig(load_format="auto")
    weight_loader._require_kimi_k3_megamoe_loader(
        auto_config, DefaultModelLoader(auto_config)
    )

    pt_config = LoadConfig(load_format="pt")
    with pytest.raises(RuntimeError, match="load-format auto"):
        weight_loader._require_kimi_k3_megamoe_loader(
            pt_config, DefaultModelLoader(pt_config)
        )
    with pytest.raises(RuntimeError, match="exact DefaultModelLoader"):
        weight_loader._require_kimi_k3_megamoe_loader(auto_config, object())

    lane = object()
    model = SimpleNamespace(
        _kimi_k3_megamoe_prepared=True,
        _kimi_k3_megamoe_layer_plans=tuple(
            SimpleNamespace(lane=lane) for _ in range(92)
        ),
        _kimi_k3_megamoe_lane_owner=lane,
    )
    weight_loader._require_kimi_k3_megamoe_prepared(model)
    model._kimi_k3_megamoe_layer_plans = model._kimi_k3_megamoe_layer_plans[:-1]
    with pytest.raises(RuntimeError, match="exactly 92"):
        weight_loader._require_kimi_k3_megamoe_prepared(model)


def test_weight_loader_uses_exact_resolved_loader_and_checks_preparation() -> None:
    lane = object()
    model = SimpleNamespace(
        _kimi_k3_megamoe_prepared=True,
        _kimi_k3_megamoe_layer_plans=tuple(
            SimpleNamespace(lane=lane) for _ in range(92)
        ),
        _kimi_k3_megamoe_lane_owner=lane,
    )
    loader = DefaultModelLoader(LoadConfig())
    server_args = SimpleNamespace(
        load_format="auto",
        download_dir=None,
        ext_yaml=None,
        weight_loader_prefetch_checkpoints=True,
        weight_loader_prefetch_num_threads=1,
        enable_kimi_k3_megamoe=True,
        kv_cache_dtype="auto",
        quantization_param_path=None,
    )
    adapter = SimpleNamespace(region=mock.Mock(return_value=contextlib.nullcontext()))
    with (
        mock.patch.object(weight_loader, "get_available_gpu_memory", return_value=1.0),
        mock.patch.object(weight_loader, "set_cuda_arch"),
        mock.patch.object(weight_loader.torch, "set_num_threads"),
        mock.patch.object(weight_loader, "get_model_loader", return_value=loader),
        mock.patch.object(loader, "load_model", return_value=model) as load_model,
        mock.patch.object(weight_loader, "get_model") as generic_get_model,
    ):
        got = weight_loader.WeightLoader.load_model(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            server_args=server_args,
            device="cuda",
            gpu_id=0,
            memory_saver_adapter=adapter,
        )
    assert got is model
    load_model.assert_called_once()
    generic_get_model.assert_not_called()

    bad_server_args = SimpleNamespace(**vars(server_args))
    bad_server_args.load_format = "dummy"
    with (
        mock.patch.object(weight_loader, "get_available_gpu_memory", return_value=1.0),
        mock.patch.object(weight_loader, "set_cuda_arch"),
        mock.patch.object(weight_loader.torch, "set_num_threads"),
        mock.patch.object(weight_loader, "get_model_loader") as get_model_loader,
        pytest.raises(RuntimeError, match="load-format auto"),
    ):
        weight_loader.WeightLoader.load_model(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            server_args=bad_server_args,
            device="cuda",
            gpu_id=0,
            memory_saver_adapter=adapter,
        )
    get_model_loader.assert_not_called()


def _bare_moe(*, plan=object()) -> kimi_k3.KimiLinearMoE:
    moe = object.__new__(kimi_k3.KimiLinearMoE)
    nn.Module.__init__(moe)
    moe._kimi_k3_megamoe_enabled = True
    moe._kimi_k3_megamoe_plan = plan
    return moe


def test_m1_decode_dispatches_megamoe_and_missing_plan_is_fatal() -> None:
    plan = object()
    moe = _bare_moe(plan=plan)
    hidden = torch.randn(1, 4)
    prefix = torch.randn(1, 4)
    expected = torch.randn(1, 4)
    decode = mock.Mock(return_value=expected)
    with mock.patch.object(
        kimi_k3,
        "_get_kimi_k3_megamoe_api",
        return_value=(mock.Mock(), mock.Mock(), decode),
    ):
        got = moe(
            hidden,
            prefix,
            num_global_tokens=1,
            max_num_tokens_per_gpu=1,
            is_decode_or_idle=True,
        )
    assert got is expected
    decode.assert_called_once_with(hidden, prefix, plan)

    moe._kimi_k3_megamoe_plan = None
    with pytest.raises(RuntimeError, match="without a prepared layer plan"):
        moe(
            hidden,
            prefix,
            num_global_tokens=1,
            max_num_tokens_per_gpu=1,
            is_decode_or_idle=True,
        )


@pytest.mark.parametrize(
    ("rows", "is_decode_or_idle"),
    [(1, False), (2, True)],
)
def test_extend_or_m_greater_than_one_uses_existing_path(
    rows: int, is_decode_or_idle: bool
) -> None:
    moe = _bare_moe()
    expected = torch.randn(rows, 4)
    native = mock.Mock(return_value=expected)
    moe.native_latent_moe = native
    moe._use_fused_decode_pipeline = False
    hidden = torch.randn(rows, 4)
    prefix = torch.randn(rows, 4)
    got = moe(
        hidden,
        prefix,
        num_global_tokens=rows,
        max_num_tokens_per_gpu=rows,
        is_decode_or_idle=is_decode_or_idle,
    )
    assert got is expected
    native.assert_called_once()


def _bare_top_model() -> tuple[kimi_k3.KimiK3ForConditionalGeneration, list]:
    model = object.__new__(kimi_k3.KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model._kimi_k3_megamoe_enabled = True
    model._kimi_k3_megamoe_prepared = False
    model._kimi_k3_megamoe_layer_plans = ()
    model._kimi_k3_megamoe_lane_owner = None
    like = torch.empty(1)
    moes = [SimpleNamespace(_kimi_k3_megamoe_plan=None) for _ in range(92)]
    layers = [SimpleNamespace(is_moe_layer=True, block_sparse_moe=moe) for moe in moes]
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=layers,
            embed_tokens=SimpleNamespace(weight=like),
        )
    )
    model.mapping = SimpleNamespace(moe=SimpleNamespace(ep_group=tuple(range(8))))
    return model, moes


def test_post_quant_warmup_stages_all_plans_after_consensus() -> None:
    model, moes = _bare_top_model()
    specs = tuple({"layer": i} for i in range(92))
    lane = object()
    plans = tuple(SimpleNamespace(lane=lane) for _ in range(92))
    prepare = mock.Mock(return_value=plans)
    acquire = mock.Mock(return_value=lane)
    consensus = mock.Mock()
    with (
        mock.patch.object(
            kimi_k3,
            "_kimi_k3_megamoe_local_specs",
            return_value=(specs, torch.empty(1)),
        ),
        mock.patch.object(
            kimi_k3,
            "_get_kimi_k3_megamoe_api",
            return_value=(lambda: True, prepare, mock.Mock()),
        ),
        mock.patch.object(kimi_k3, "_acquire_kimi_k3_megamoe_lane", acquire),
        mock.patch.object(
            kimi_k3, "_kimi_k3_megamoe_consensus", return_value=consensus
        ),
    ):
        model.post_quant_warmup()

    assert model._kimi_k3_megamoe_prepared is True
    assert model._kimi_k3_megamoe_layer_plans == plans
    assert model._kimi_k3_megamoe_lane_owner is lane
    assert [moe._kimi_k3_megamoe_plan for moe in moes] == list(plans)
    assert all(moe._kimi_k3_megamoe_enabled for moe in moes)
    acquire.assert_called_once()
    assert acquire.call_args.kwargs == {"local_status": 0, "local_reason": ""}
    prepare.assert_called_once_with(
        specs,
        lane,
        like=mock.ANY,
        consensus=consensus,
    )


def test_post_quant_local_failure_still_enters_uniform_consensus() -> None:
    model, _ = _bare_top_model()
    acquire = mock.Mock(side_effect=RuntimeError("uniform admission failure"))
    prepare = mock.Mock()
    with (
        mock.patch.object(
            kimi_k3,
            "_kimi_k3_megamoe_local_specs",
            side_effect=ValueError("bad local layer"),
        ),
        mock.patch.object(
            kimi_k3,
            "_get_kimi_k3_megamoe_api",
            return_value=(lambda: True, prepare, mock.Mock()),
        ),
        mock.patch.object(kimi_k3, "_acquire_kimi_k3_megamoe_lane", acquire),
        pytest.raises(RuntimeError, match="uniform admission failure"),
    ):
        model.post_quant_warmup()
    assert acquire.call_args.kwargs["local_status"] == 1
    assert "bad local layer" in acquire.call_args.kwargs["local_reason"]
    prepare.assert_not_called()
    assert model._kimi_k3_megamoe_prepared is False


def test_prepared_model_rejects_weight_iterable_before_consumption() -> None:
    consumed = False

    def weights():
        nonlocal consumed
        consumed = True
        yield "weight", torch.empty(1)

    model = SimpleNamespace(_kimi_k3_megamoe_prepared=True)
    with pytest.raises(RuntimeError, match="online weight replacement"):
        kimi_k3.KimiK3ForConditionalGeneration.load_weights(model, weights())
    assert consumed is False


def test_online_update_guards_run_before_dispatch_or_mutation() -> None:
    auto_create = mock.Mock()
    frontend = SimpleNamespace(
        server_args=SimpleNamespace(enable_kimi_k3_megamoe=True),
        auto_create_handle_loop=auto_create,
    )
    with pytest.raises(RuntimeError, match="Online weight replacement"):
        asyncio.run(AsyncLLM.update_weights_from_disk(frontend, object()))
    with pytest.raises(RuntimeError, match="distributed weight replacement"):
        asyncio.run(
            SchedulerControlClient.update_weights_from_distributed(frontend, object())
        )
    with pytest.raises(RuntimeError, match="tensor weight replacement"):
        asyncio.run(
            SchedulerControlClient.update_weights_from_tensor(frontend, object())
        )
    auto_create.assert_not_called()

    worker = SimpleNamespace(server_args=SimpleNamespace(enable_kimi_k3_megamoe=True))
    ok, reason = ModelRunner.update_weights_from_distributed(worker, object())
    assert not ok
    assert "disabled" in reason


def test_model_execution_result_rejects_fatal_epoch_after_event_sync() -> None:
    event = mock.Mock()
    result = ModelExecutionResult(
        output_tokens=torch.zeros(1, dtype=torch.int32),
        copy_event=event,
        kimi_k3_megamoe_fatal_epoch=torch.tensor([7], dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="fatal epoch 7"):
        result.sync()
    event.synchronize.assert_called_once()

    clean = ModelExecutionResult(
        output_tokens=torch.zeros(1, dtype=torch.int32),
        copy_event=mock.Mock(),
        kimi_k3_megamoe_fatal_epoch=torch.zeros(1, dtype=torch.int64),
    )
    clean.sync()


def test_model_runner_enqueues_fatal_epoch_copy_nonblocking() -> None:
    fatal_gpu = object()
    fatal_cpu = (mock.Mock(), mock.Mock())
    runner = SimpleNamespace(
        _kimi_k3_megamoe_fatal_epoch_gpu=fatal_gpu,
        _kimi_k3_megamoe_fatal_epoch_cpu_slots=fatal_cpu,
        _kimi_k3_megamoe_fatal_epoch_copy_index=0,
        _kimi_k3_megamoe_fatal_epoch_d2h_enabled=True,
    )

    first = ModelRunner.enqueue_kimi_k3_megamoe_fatal_epoch_d2h(runner)
    second = ModelRunner.enqueue_kimi_k3_megamoe_fatal_epoch_d2h(runner)

    assert first is fatal_cpu[0]
    assert second is fatal_cpu[1]
    assert runner._kimi_k3_megamoe_fatal_epoch_copy_index == 2
    fatal_cpu[0].copy_.assert_called_once_with(fatal_gpu, non_blocking=True)
    fatal_cpu[1].copy_.assert_called_once_with(fatal_gpu, non_blocking=True)

    runner._kimi_k3_megamoe_fatal_epoch_d2h_enabled = False
    assert ModelRunner.enqueue_kimi_k3_megamoe_fatal_epoch_d2h(runner) is None
    assert runner._kimi_k3_megamoe_fatal_epoch_copy_index == 2


def test_overlap_order_and_fatal_epoch_slots_form_a_safe_depth_one_pipeline() -> None:
    validate_kimi_k3_megamoe_fatal_epoch_slots(overlap_schedule_depth=1, slot_count=2)
    with pytest.raises(RuntimeError, match="needs at least 3 host slots"):
        validate_kimi_k3_megamoe_fatal_epoch_slots(
            overlap_schedule_depth=2, slot_count=2
        )

    source = inspect.getsource(EventLoop.event_loop_overlap)
    dispatch = source.index("curr_results, _ = self._dispatch_forward(")
    commit_previous = source.index(
        "self._commit_forward_results(prev_forward_op, prev_results)", dispatch
    )
    rotate_current = source.index("prev_results = curr_results", commit_previous)
    assert dispatch < commit_previous < rotate_current
