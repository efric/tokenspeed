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

"""Qualify the gfx950 MegaMoE cooperative persistent-grid mechanism.

Run this in the TokenSpeed virtual environment. The cooperative lifecycle test
requires the system ROCr object to be interposed before Python starts::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \
      python tools/megamoe/gfx950_persistent_grid_probe.py --replays 1000

The probe is intentionally independent of model weights. The production gate
must repeat the same checks on the final heavy MegaMoE code object.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_PROGRAMS = gl.constexpr(240)
_SOURCE_ELEMENTS = gl.constexpr(1 << 20)
_LDS_BYTES = gl.constexpr(128 * 1024)


@gluon.jit
def _persistent_grid_probe_kernel(source, payload, arrival, gate, xcc):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    lane = gl.arange(0, 512, layout=layout)
    offsets = pid * 512 + lane

    # The full parent allocation is material and caps gfx950 residency at one
    # 512-thread workgroup per compute unit. Touch a workgroup-sized subview so
    # dead-allocation elimination cannot discard the reservation.
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    residency = gl.allocate_shared_memory(gl.uint8, [_LDS_BYTES], shared_layout)
    residency_tile = residency.slice(0, 512)
    residency_tile.store((lane % 251).to(gl.uint8))
    gl.barrier()
    acc = residency_tile.load(layout).to(gl.float32)

    # Keep the binary nontrivial while making every payload address unique.
    for k in range(0, _SOURCE_ELEMENTS, _PROGRAMS * 512):
        index = offsets + k
        acc += gl.load(
            source + index,
            mask=index < _SOURCE_ELEMENTS,
            other=0.0,
        ).to(gl.float32)
    gl.store(payload + offsets, acc)

    xcc_id = gl.inline_asm_elementwise(
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
        "=s",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.store(xcc + pid + lane * 0, xcc_id, mask=lane == 0)

    ticket = gl.atomic_add(arrival, 1, sem="acq_rel", scope="gpu")
    generation = ticket // _PROGRAMS + 1
    if ticket % _PROGRAMS == _PROGRAMS - 1:
        gl.atomic_xchg(gate, generation, sem="release", scope="gpu")
    gl.atomic_poll(gate, generation, sem="acquire", scope="gpu")

    # Consume another workgroup's payload only after the grid acquire.
    peer_offsets = ((pid + 1) % _PROGRAMS) * 512 + gl.arange(
        0, 512, layout=layout
    )
    value = gl.load(payload + peer_offsets, volatile=True)
    gl.store(payload + offsets, value + 1.0)


def _compiled_kernel():
    cache = _persistent_grid_probe_kernel.device_caches[
        torch.cuda.current_device()
    ][0]
    if len(cache) != 1:
        raise RuntimeError(f"expected one compiled specialization, got {len(cache)}")
    return next(iter(cache.values()))


def _module_occupancy(function: int, block_size: int, shared: int) -> int:
    # Use PyTorch's already-loaded HIP runtime. Loading /opt/rocm's HIP DSO as a
    # second runtime corrupts this lifecycle experiment.
    hip_path = os.path.join(torch.__path__[0], "lib", "libamdhip64.so")
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
        block_size,
        shared,
    )
    if status != 0:
        raise RuntimeError(f"HIP occupancy query failed with status {status}")
    return result.value


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _symbol_object(symbol: str) -> str:
    process = ctypes.CDLL(None)
    address = ctypes.cast(getattr(process, symbol), ctypes.c_void_p)
    info = _DlInfo()
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    if libdl.dladdr(address, ctypes.byref(info)) == 0 or info.dli_fname is None:
        raise RuntimeError(f"dladdr failed for {symbol}")
    return os.path.realpath(info.dli_fname.decode())


def _mapped_objects(fragment: str) -> list[str]:
    paths: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as mappings:
        for line in mappings:
            path = line.rsplit(maxsplit=1)[-1]
            if fragment in path and path.startswith("/"):
                paths.add(os.path.realpath(path))
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if args.replays < 1:
        raise ValueError("--replays must be positive")

    source = torch.arange(
        _SOURCE_ELEMENTS.value,
        device="cuda",
        dtype=torch.float32,
    )
    payload = torch.zeros(
        (_PROGRAMS.value * 512,),
        device="cuda",
        dtype=torch.float32,
    )
    arrival = torch.zeros((), device="cuda", dtype=torch.int64)
    gate = torch.zeros((), device="cuda", dtype=torch.int64)
    xcc = torch.full((_PROGRAMS.value,), -1, device="cuda", dtype=torch.int32)

    cooperative = not args.ordinary
    launch_kwargs = dict(
        num_warps=8,
        num_stages=1,
        waves_per_eu=2,
        launch_cooperative_grid=cooperative,
    )
    _persistent_grid_probe_kernel[(_PROGRAMS.value,)](
        source,
        payload,
        arrival,
        gate,
        xcc,
        **launch_kwargs,
    )
    torch.cuda.synchronize()
    eager_xcc = sorted(torch.unique(xcc).cpu().tolist())

    graph = None
    executed_generations = 1
    if not args.no_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _persistent_grid_probe_kernel[(_PROGRAMS.value,)](
                source,
                payload,
                arrival,
                gate,
                xcc,
                **launch_kwargs,
            )
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.synchronize()
        executed_generations += args.replays

    compiled = _compiled_kernel()
    result = {
        "cooperative": cooperative,
        "programs": _PROGRAMS.value,
        "subgroups_per_workgroup": 8,
        "xcc_ids": eager_xcc,
        "arrival": int(arrival.item()),
        "gate": int(gate.item()),
        "expected_generations": executed_generations,
        "n_regs": compiled.n_regs,
        "n_spills": compiled.n_spills,
        "shared": compiled.metadata.shared,
        "waves_per_eu": compiled.metadata.waves_per_eu,
        "launch_metadata_cooperative": compiled.metadata.launch_cooperative_grid,
        "module_occupancy": _module_occupancy(
            compiled.function,
            512,
            compiled.metadata.shared,
        ),
        "hsa_symbol_objects": {
            name: _symbol_object(name)
            for name in (
                "hsa_init",
                "hsa_shut_down",
                "hsa_signal_store_screlease",
            )
        },
        "mapped_hsa_objects": _mapped_objects("libhsa-runtime64.so"),
        "mapped_hip_objects": _mapped_objects("libamdhip64.so"),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
    }
    if graph is not None:
        graph.reset()
        torch.cuda.synchronize()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
