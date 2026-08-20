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

"""Fail-closed deployment and loaded-code admission for gfx950 MegaMoE."""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
from tokenspeed_kernel_amd._triton import triton

PROGRAMS = 240
SUBGROUPS = 8
WORKGROUP_THREADS = 512
LDS_BYTES = 128 * 1024
QUALIFIED_TIMEOUT_NS = 1_000_000_000
_EXPECTED_COMPUTE_UNITS = 256
_SYSTEM_ROCR = Path("/opt/rocm/lib/libhsa-runtime64.so.1")
_SYSTEM_ROCR_SHA256 = "b8cdfe93d343649a35c1daf73a0a3a6840f09379ebeee9be65670461ffea43f4"
_SYSTEM_ROCR_BUILD_ID = "cbe2c420f8c65e4710580d19cfd7950db722ea9f"
_EXPECTED_TORCH_VERSION = "2.13.0+rocm7.2"
_EXPECTED_TORCH_HIP_VERSION = "7.2.53211"
# The exact split ready/completion protocol, sticky entry acquire, and bounded
# fatal path require scalar values to survive across workgroup broadcasts. The
# qualified objects use 40 SGPR lane-save slots without raising the 156-VGPR
# allocation, creating Triton/VGPR/scratch spills, or
# changing occupancy. Pin each rank-specialized HSACO and lane-save count as
# well as the common resource tuple so a compiler result with superficially
# similar metadata cannot enter the launch path.
_QUALIFIED_SGPR_COUNT = 106
_QUALIFIED_VGPR_COUNT = 156
_QUALIFIED_SGPR_LANE_SPILLS = {
    (0, 0): 40,
    (1, 112): 40,
    (2, 224): 40,
    (3, 336): 40,
    (4, 448): 40,
    (5, 560): 40,
    (6, 672): 40,
    (7, 784): 40,
}
_QUALIFIED_CODE_OBJECT_SHA256 = {
    (0, 0): "0290f32a2858776db0987926a0f7a3030366787667308d39555dff6dde39377c",
    (1, 112): "868e6e1aaa87b43d924c840617c513a96051302b1cb0fd8d001a4ebb42cf46f6",
    (2, 224): "862a8fe72ec4ea0bd1f1c83ce2a205e0283b2a565470b35a2ca07ee4d265ea48",
    (3, 336): "e243ea64c0f38d007b562cfd04b3a5c61469eb3f0c3c8317a74a35b8584c2ae2",
    (4, 448): "47d6cded5c8594deb4b2b25b8614797886a31ee46309120f320d208bf8a39cfe",
    (5, 560): "266fe82bf9d903fd66efed64a0e100ebd0a5d856c0b7b75b7df6f1d34909d285",
    (6, 672): "78d352fb62bee24c64f45f3ba6651826068ab65c3fe95ece7a7b002439493083",
    (7, 784): "026150a4ac0c855267dbe1aabd9e501b29d66c154f126657cc357919f94b7986",
}


@dataclass(frozen=True)
class MegaMoERuntimeAdmission:
    """Recorded process/device facts required before compiling MegaMoE."""

    hsa_symbol_objects: tuple[tuple[str, str], ...]
    system_rocr_realpath: str
    system_rocr_sha256: str
    system_rocr_build_id: str
    mapped_hsa_objects: tuple[str, ...]
    mapped_hip_objects: tuple[str, ...]
    hip_object: str
    torch_version: str
    torch_hip_version: str
    triton_driver_source: str
    cooperative_driver_abi_ok: bool
    architecture: str
    compute_units: int
    cooperative_launch_supported: bool
    max_shared_mem: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serializable admission evidence."""

        return asdict(self)


@dataclass(frozen=True)
class MegaMoECodeAdmission:
    """Loaded specialization facts required before ordinary launch."""

    programs: int
    subgroups: int
    threads: int
    shared: int
    launch_cooperative_grid: bool
    waves_per_eu: int
    occupancy: int
    compute_units: int
    resident_capacity: int
    n_regs: int
    n_spills: int
    sgpr_count: int
    vgpr_count: int
    sgpr_spill_count: int
    vgpr_spill_count: int
    private_segment_fixed_size: int
    uses_dynamic_stack: bool
    uses_flat_scratch: bool
    code_object_size: int
    code_object_sha256: str
    has_scratch_instructions: bool
    group_rank: int
    expert_start: int
    timeout_ns: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serializable specialization evidence."""

        return asdict(self)


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elf_build_id(path: str) -> str:
    result = subprocess.run(
        ("readelf", "-n", path),
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"MegaMoE could not read the ROCr build ID from {path}")
    return match.group(1).lower()


def _symbol_object(symbol: str) -> str:
    process = ctypes.CDLL(None)
    try:
        address = ctypes.cast(getattr(process, symbol), ctypes.c_void_p)
    except AttributeError as error:
        raise RuntimeError(
            f"MegaMoE cannot resolve required HSA symbol {symbol}"
        ) from error
    info = _DlInfo()
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    if libdl.dladdr(address, ctypes.byref(info)) == 0 or info.dli_fname is None:
        raise RuntimeError(f"MegaMoE dladdr failed for HSA symbol {symbol}")
    return os.path.realpath(info.dli_fname.decode())


def _mapped_objects(fragment: str) -> tuple[str, ...]:
    paths: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as mappings:
        for line in mappings:
            candidate = line.rsplit(maxsplit=1)[-1]
            if fragment in candidate and candidate.startswith("/"):
                paths.add(os.path.realpath(candidate))
    return tuple(sorted(paths))


def _driver_source() -> Path:
    package = Path(triton.__file__).resolve().parent
    path = package / "backends" / "amd" / "driver.c"
    if not path.is_file():
        raise RuntimeError(
            f"MegaMoE cannot inspect the loaded Triton AMD driver at {path}"
        )
    return path


def _validate_driver_abi(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    declaration = source.find("FOR_EACH_ERR_FN(hipModuleLaunchCooperativeKernel")
    if declaration < 0:
        raise RuntimeError("MegaMoE Triton driver lacks cooperative launch support")
    declaration_window = source[declaration : declaration + 900]
    if (
        "void **kernelParams)" not in declaration_window
        or "void **kernelParams, void **extra)" in declaration_window
    ):
        raise RuntimeError(
            "MegaMoE rejects Triton's stale eleven-argument "
            "hipModuleLaunchCooperativeKernel declaration"
        )
    call = source.rfind("hipSymbolTable.hipModuleLaunchCooperativeKernel(")
    call_window = source[call : call + 500] if call >= 0 else ""
    if "shared_memory, stream, params));" not in call_window:
        raise RuntimeError(
            "MegaMoE requires Triton's installed ten-argument cooperative launcher call"
        )


@lru_cache(maxsize=1)
def preflight_kimi_k3_megamoe_runtime() -> MegaMoERuntimeAdmission:
    """Validate and record the process-wide isolated deployment contract.

    This must run after Python startup but before any MegaMoE compilation or
    launch. It proves that ``LD_PRELOAD`` took effect; setting the environment
    variable after PyTorch initialized ROCr cannot satisfy this check. The
    cooperative capability and launcher-ABI checks remain part of the frozen
    runtime fingerprint even though the admitted MegaMoE dispatch is ordinary.
    """

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Kimi K3 MegaMoE requires a ROCm CUDA device")
    if torch.__version__ != _EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            "Kimi K3 MegaMoE requires the qualified PyTorch build "
            f"{_EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )
    if torch.version.hip != _EXPECTED_TORCH_HIP_VERSION:
        raise RuntimeError(
            "Kimi K3 MegaMoE requires the qualified HIP runtime version "
            f"{_EXPECTED_TORCH_HIP_VERSION}, got {torch.version.hip}"
        )
    device = torch.cuda.current_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    architecture = str(properties.get("arch", "")).split(":", 1)[0]
    compute_units = int(properties.get("multiprocessor_count", 0))
    cooperative = bool(properties.get("cooperativeLaunch", 0))
    max_shared = int(properties.get("max_shared_mem", 0))
    if architecture != "gfx950":
        raise RuntimeError(f"Kimi K3 MegaMoE requires gfx950, got {architecture!r}")
    if compute_units != _EXPECTED_COMPUTE_UNITS:
        raise RuntimeError(
            f"Kimi K3 MegaMoE requires {_EXPECTED_COMPUTE_UNITS} compute units, "
            f"got {compute_units}"
        )
    if not cooperative:
        raise RuntimeError("Kimi K3 MegaMoE device does not support cooperative launch")
    if max_shared < LDS_BYTES:
        raise RuntimeError(
            f"Kimi K3 MegaMoE requires at least {LDS_BYTES} bytes LDS, got {max_shared}"
        )

    expected_rocr = os.path.realpath(_SYSTEM_ROCR)
    if not os.path.isfile(expected_rocr):
        raise RuntimeError(f"Kimi K3 MegaMoE cannot find system ROCr at {_SYSTEM_ROCR}")
    system_sha = _sha256(expected_rocr)
    system_build_id = _elf_build_id(expected_rocr)
    if system_sha != _SYSTEM_ROCR_SHA256 or system_build_id != _SYSTEM_ROCR_BUILD_ID:
        raise RuntimeError(
            "Kimi K3 MegaMoE system ROCr differs from the qualified build: "
            f"sha256={system_sha}, build_id={system_build_id}"
        )

    symbols = tuple(
        (name, _symbol_object(name))
        for name in ("hsa_init", "hsa_shut_down", "hsa_signal_store_screlease")
    )
    for name, object_path in symbols:
        if not os.path.samefile(object_path, expected_rocr):
            raise RuntimeError(
                f"MegaMoE requires {name} from system ROCr {expected_rocr}, "
                f"but dladdr resolved {object_path}; start Python with "
                f"LD_PRELOAD={_SYSTEM_ROCR}"
            )

    mapped_hip = _mapped_objects("libamdhip64.so")
    torch_hip = os.path.realpath(Path(torch.__path__[0]) / "lib" / "libamdhip64.so")
    if mapped_hip != (torch_hip,):
        raise RuntimeError(
            "Kimi K3 MegaMoE requires exactly PyTorch's loaded HIP runtime; "
            f"mapped objects are {mapped_hip}"
        )

    driver_source = _driver_source()
    _validate_driver_abi(driver_source)
    return MegaMoERuntimeAdmission(
        hsa_symbol_objects=symbols,
        system_rocr_realpath=expected_rocr,
        system_rocr_sha256=system_sha,
        system_rocr_build_id=system_build_id,
        mapped_hsa_objects=_mapped_objects("libhsa-runtime64.so"),
        mapped_hip_objects=mapped_hip,
        hip_object=torch_hip,
        torch_version=torch.__version__,
        torch_hip_version=torch.version.hip,
        triton_driver_source=str(driver_source),
        cooperative_driver_abi_ok=True,
        architecture=architecture,
        compute_units=compute_units,
        cooperative_launch_supported=cooperative,
        max_shared_mem=max_shared,
    )


def _module_occupancy(function: int, shared: int) -> int:
    # Opening the already-loaded object returns its process singleton. Never
    # dlopen /opt/rocm's HIP DSO, which would create the disallowed dual runtime.
    hip_path = os.path.realpath(Path(torch.__path__[0]) / "lib" / "libamdhip64.so")
    hip = ctypes.CDLL(hip_path)
    query = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    result = ctypes.c_int()
    status = query(
        ctypes.byref(result),
        ctypes.c_void_p(function),
        WORKGROUP_THREADS,
        shared,
    )
    if status != 0:
        raise RuntimeError(f"MegaMoE HIP occupancy query failed with status {status}")
    return result.value


def _asm_text(compiled: object) -> str:
    assembly = getattr(compiled, "asm", {})
    if not isinstance(assembly, dict):
        return ""
    for name in ("amdgcn", "asm", "sass"):
        value = assembly.get(name)
        if isinstance(value, str):
            return value
    return ""


def _code_object_bytes(compiled: object) -> bytes:
    assembly = getattr(compiled, "asm", {})
    if isinstance(assembly, dict):
        for name in ("hsaco", "cubin"):
            value = assembly.get(name)
            if isinstance(value, bytes):
                return value
    kernel = getattr(compiled, "kernel", b"")
    if isinstance(kernel, bytes):
        return kernel
    return b""


def _amdgcn_metadata_int(assembly: str, field: str) -> int:
    match = re.search(
        rf"^\s*\.{re.escape(field)}:\s*(\d+)\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match is not None else -1


def _amdgcn_metadata_bool(assembly: str, field: str) -> bool | None:
    match = re.search(
        rf"^\s*\.{re.escape(field)}:\s*(true|false)\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    return match.group(1) == "true" if match is not None else None


def admit_kimi_k3_megamoe_compiled_kernel(
    compiled: object,
    *,
    group_rank: int,
    expert_start: int,
    timeout_ns: int,
    prototype_role: str | None = None,
) -> MegaMoECodeAdmission:
    """Validate one exact loaded rank specialization before its first launch."""

    if prototype_role not in (None, "core", "finalizer"):
        raise ValueError(f"unknown MegaMoE prototype role {prototype_role!r}")
    prototype = prototype_role is not None
    if timeout_ns != QUALIFIED_TIMEOUT_NS:
        raise RuntimeError(
            "Kimi K3 MegaMoE rejects unqualified timeout_ns="
            f"{timeout_ns}; expected {QUALIFIED_TIMEOUT_NS}"
        )
    specialization = (int(group_rank), int(expert_start))
    expected_hash = _QUALIFIED_CODE_OBJECT_SHA256.get(specialization)
    expected_lane_spills = _QUALIFIED_SGPR_LANE_SPILLS.get(specialization)
    if (
        expected_hash is None
        or expected_lane_spills is None
        or expert_start != 112 * group_rank
    ):
        raise RuntimeError(
            "Kimi K3 MegaMoE rejects unqualified specialization "
            f"group_rank={group_rank}, expert_start={expert_start}"
        )

    runtime = preflight_kimi_k3_megamoe_runtime()
    # ``warmup`` compiles without dispatch but lazily defers module loading.
    # Loading the module is required to query the exact hipFunction_t. It does
    # not enqueue the persistent grid.
    if getattr(compiled, "function", None) is None:
        compiled._init_handles()
    metadata = compiled.metadata
    shared = int(metadata.shared)
    cooperative = bool(metadata.launch_cooperative_grid)
    waves_per_eu = int(metadata.waves_per_eu)
    occupancy = _module_occupancy(int(compiled.function), shared)
    compute_units = int(runtime.compute_units)
    resident_capacity = occupancy * compute_units
    n_regs = int(compiled.n_regs)
    n_spills = int(compiled.n_spills)
    assembly = _asm_text(compiled).lower()
    scratch = bool(re.search(r"\bscratch_(?:load|store)(?:_[a-z0-9]+)*\b", assembly))
    sgpr_count = _amdgcn_metadata_int(assembly, "sgpr_count")
    vgpr_count = _amdgcn_metadata_int(assembly, "vgpr_count")
    sgpr_spills = _amdgcn_metadata_int(assembly, "sgpr_spill_count")
    vgpr_spills = _amdgcn_metadata_int(assembly, "vgpr_spill_count")
    private_segment = _amdgcn_metadata_int(assembly, "private_segment_fixed_size")
    dynamic_stack = _amdgcn_metadata_bool(assembly, "uses_dynamic_stack")
    flat_scratch_match = re.search(
        r"^\s*\.set\s+\.l_[^.]+\.uses_flat_scratch,\s*(\d+)\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    uses_flat_scratch = (
        int(flat_scratch_match.group(1)) != 0
        if flat_scratch_match is not None
        else True
    )
    code_object = _code_object_bytes(compiled)
    programs = 224 if prototype_role == "finalizer" else PROGRAMS
    report = MegaMoECodeAdmission(
        programs=programs,
        subgroups=SUBGROUPS,
        threads=WORKGROUP_THREADS,
        shared=shared,
        launch_cooperative_grid=cooperative,
        waves_per_eu=waves_per_eu,
        occupancy=occupancy,
        compute_units=compute_units,
        resident_capacity=resident_capacity,
        n_regs=n_regs,
        n_spills=n_spills,
        sgpr_count=sgpr_count,
        vgpr_count=vgpr_count,
        sgpr_spill_count=sgpr_spills,
        vgpr_spill_count=vgpr_spills,
        private_segment_fixed_size=private_segment,
        uses_dynamic_stack=bool(dynamic_stack),
        uses_flat_scratch=uses_flat_scratch,
        code_object_size=len(code_object),
        code_object_sha256=(
            hashlib.sha256(code_object).hexdigest() if code_object else ""
        ),
        has_scratch_instructions=scratch,
        group_rank=group_rank,
        expert_start=expert_start,
        timeout_ns=timeout_ns,
    )
    failures: list[str] = []
    minimum_shared = 0 if prototype_role == "finalizer" else LDS_BYTES
    expected_waves_per_eu = 0 if prototype_role == "finalizer" else 2
    if shared < minimum_shared:
        failures.append(f"dynamic LDS {shared} < {LDS_BYTES}")
    if cooperative:
        failures.append("compiled metadata is cooperative; ordinary launch is required")
    if waves_per_eu != expected_waves_per_eu:
        failures.append(
            f"waves_per_eu is {waves_per_eu}, expected {expected_waves_per_eu}"
        )
    if prototype_role != "finalizer" and occupancy != 1:
        failures.append(f"loaded occupancy is {occupancy}, expected 1")
    if n_spills != 0:
        failures.append(f"loaded kernel reports {n_spills} spills")
    if not assembly:
        failures.append("loaded kernel did not expose AMDGCN assembly")
    if min(sgpr_count, vgpr_count, sgpr_spills, vgpr_spills, private_segment) < 0:
        failures.append("loaded AMDGCN lacks required resource metadata")
    if vgpr_spills != 0:
        failures.append(f"AMDGPU metadata reports {vgpr_spills} VGPR spills")
    if not prototype and sgpr_count != _QUALIFIED_SGPR_COUNT:
        failures.append(
            f"AMDGPU metadata reports {sgpr_count} SGPRs, "
            f"qualified value is {_QUALIFIED_SGPR_COUNT}"
        )
    if not prototype and vgpr_count != _QUALIFIED_VGPR_COUNT:
        failures.append(
            f"AMDGPU metadata reports {vgpr_count} VGPRs, "
            f"qualified value is {_QUALIFIED_VGPR_COUNT}"
        )
    if not prototype and sgpr_spills != expected_lane_spills:
        failures.append(
            f"AMDGPU metadata reports {sgpr_spills} SGPR lane spills, "
            f"qualified value is {expected_lane_spills}"
        )
    if private_segment != 0:
        failures.append(f"private segment is {private_segment} bytes, expected 0")
    if dynamic_stack is not False:
        failures.append("AMDGPU metadata reports or omits a dynamic stack")
    if uses_flat_scratch:
        failures.append("loaded AMDGCN enables flat scratch")
    if scratch:
        failures.append("loaded AMDGCN contains scratch load/store instructions")
    if not code_object:
        failures.append("loaded kernel did not expose code-object bytes")
    elif not prototype and report.code_object_sha256 != expected_hash:
        failures.append(
            "loaded code-object SHA256 is "
            f"{report.code_object_sha256}, expected "
            f"{expected_hash} for group_rank={group_rank}, "
            f"expert_start={expert_start}"
        )
    if prototype_role != "finalizer" and resident_capacity < PROGRAMS:
        failures.append(
            "ordinary persistent grid exceeds the loaded residency envelope: "
            f"{resident_capacity} resident slots < {PROGRAMS} programs"
        )
    if failures:
        raise RuntimeError(
            "Kimi K3 MegaMoE code admission failed: " + "; ".join(failures)
        )
    return report


__all__ = [
    "LDS_BYTES",
    "MegaMoECodeAdmission",
    "MegaMoERuntimeAdmission",
    "PROGRAMS",
    "QUALIFIED_TIMEOUT_NS",
    "SUBGROUPS",
    "WORKGROUP_THREADS",
    "admit_kimi_k3_megamoe_compiled_kernel",
    "preflight_kimi_k3_megamoe_runtime",
]
