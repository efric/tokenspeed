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

import torch

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.model_loader import get_model, get_model_loader
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
    set_cuda_arch,
)
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def _require_kimi_k3_megamoe_load_format(load_config: LoadConfig) -> None:
    """Reject non-AUTO formats before resolving or constructing a loader."""

    if load_config.load_format is not LoadFormat.AUTO:
        load_format = getattr(load_config.load_format, "value", load_config.load_format)
        raise RuntimeError(
            f"--enable-kimi-k3-megamoe requires --load-format auto; got {load_format!r}"
        )


def _require_kimi_k3_megamoe_loader(load_config: LoadConfig, loader: object) -> None:
    """Enforce the exact initial-loader contract for experimental MegaMoE."""

    _require_kimi_k3_megamoe_load_format(load_config)
    if type(loader) is not DefaultModelLoader:
        raise RuntimeError(
            "--enable-kimi-k3-megamoe requires the exact DefaultModelLoader; "
            f"resolved {type(loader).__name__}"
        )


def _require_kimi_k3_megamoe_prepared(model: torch.nn.Module) -> None:
    """Reject a model whose post-quant MegaMoE preparation did not complete."""

    if getattr(model, "_kimi_k3_megamoe_prepared", False) is not True:
        raise RuntimeError(
            "Kimi-K3 MegaMoE post_quant_warmup did not complete during loading"
        )
    plans = getattr(model, "_kimi_k3_megamoe_layer_plans", ())
    if (
        not isinstance(plans, tuple)
        or len(plans) != 92
        or any(plan is None for plan in plans)
    ):
        raise RuntimeError(
            "Kimi-K3 MegaMoE preparation must publish exactly 92 layer plans"
        )
    lane_owner = getattr(model, "_kimi_k3_megamoe_lane_owner", None)
    if lane_owner is None or any(
        getattr(plan, "lane", None) is not lane_owner for plan in plans
    ):
        raise RuntimeError(
            "Kimi-K3 MegaMoE layer plans must retain one identical lane owner"
        )


class WeightLoader:
    """Handles model weight loading from disk.

    This class is stateless and does not modify external state.
    It returns LoadedModel with all necessary information.
    """

    @staticmethod
    def load_model(
        model_config: ModelConfig,
        server_args: ServerArgs,
        device: str,
        gpu_id: int,
        memory_saver_adapter: TorchMemorySaverAdapter,
    ):
        """Load model from disk.

        Args:
            model_config: Model configuration
            server_args: Server arguments
            device: Device type ("cuda", "cpu")
            gpu_id: GPU ID
            memory_saver_adapter: Memory saver adapter

        Returns:
            LoadedModel with model and dtype
        """
        logger.info(
            "Load weight begin. avail mem=%.2f GB",
            get_available_gpu_memory(device, gpu_id),
        )

        # Reduce thread conflicts during weight loading
        if device != "cpu":
            torch.set_num_threads(1)

        set_cuda_arch()

        # Create load config
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            ext_yaml=server_args.ext_yaml,
            weight_loader_prefetch_checkpoints=server_args.weight_loader_prefetch_checkpoints,
            weight_loader_prefetch_num_threads=server_args.weight_loader_prefetch_num_threads,
        )

        megamoe_loader = None
        if server_args.enable_kimi_k3_megamoe:
            # Resolve and validate the loader before model construction or any
            # checkpoint tensor can mutate model state.  AUTO is intentionally
            # narrow here: extensible and custom loader classes are not admitted.
            _require_kimi_k3_megamoe_load_format(load_config)
            megamoe_loader = get_model_loader(load_config)
            _require_kimi_k3_megamoe_loader(load_config, megamoe_loader)

        # Load model with memory saver context. Tag as "weights" with CPU backup
        # so release_memory_occupation offloads (and restores) them byte-exact.
        with memory_saver_adapter.region(tag="weights", enable_cpu_backup=True):
            if megamoe_loader is not None:
                model = megamoe_loader.load_model(
                    model_config=model_config,
                    device_config=DeviceConfig(device),
                )
            else:
                model = get_model(
                    model_config=model_config,
                    load_config=load_config,
                    device_config=DeviceConfig(device),
                )

        if server_args.enable_kimi_k3_megamoe:
            _require_kimi_k3_megamoe_prepared(model)

        # Load KV cache scaling factors if using FP8
        if server_args.kv_cache_dtype == "fp8_e4m3":
            if server_args.quantization_param_path is not None:
                if callable(getattr(model, "load_kv_cache_scales", None)):
                    model.load_kv_cache_scales(server_args.quantization_param_path)
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        f"model {model.__class__} does not support loading scaling factors."
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors provided. "
                    "Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        dtype = model_config.dtype

        logger.info(
            "Load weight end. type=%s, dtype=%s, avail mem=%.2f GB",
            type(model).__name__,
            dtype,
            get_available_gpu_memory(device, gpu_id),
        )

        return model
