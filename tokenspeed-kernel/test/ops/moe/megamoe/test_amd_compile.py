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

"""Opt-in loaded-code compile gate for the raw gfx950 MegaMoE kernel."""

from __future__ import annotations

import ast
import hashlib
import os
import re
from collections import Counter
from pathlib import Path

import pytest
import torch

_EXPECTED_KERNEL_SHA256 = (
    "b0314296919330245d04724afc6e8902c6e2e977936bb101e48e532fb92f5da7"
)
_EXPECTED_SPECIALIZATIONS = (
    (
        0,
        0,
        "0290f32a2858776db0987926a0f7a3030366787667308d39555dff6dde39377c",
        "bdc0436faae6808bc9338c65f784f72258f471456f63dcbf3465117b25d956df",
        135840,
        40,
    ),
    (
        1,
        112,
        "868e6e1aaa87b43d924c840617c513a96051302b1cb0fd8d001a4ebb42cf46f6",
        "2fd451cbd95d72611e3decca1208b38836e08b8dc71dcae728e99c25866ed1e5",
        135720,
        40,
    ),
    (
        2,
        224,
        "862a8fe72ec4ea0bd1f1c83ce2a205e0283b2a565470b35a2ca07ee4d265ea48",
        "be4490b2fa6cfe34d176122d6aa747fe206b784a840dcc6f1cb39de6c3215a62",
        136720,
        40,
    ),
    (
        3,
        336,
        "e243ea64c0f38d007b562cfd04b3a5c61469eb3f0c3c8317a74a35b8584c2ae2",
        "d2fd49bc0c25ba0ef1a831ee23c0655e6db58d14ebf1051b15542f325f01cf5d",
        136760,
        40,
    ),
    (
        4,
        448,
        "47d6cded5c8594deb4b2b25b8614797886a31ee46309120f320d208bf8a39cfe",
        "1ddd6fd58e5be93db92a226508ee7fc5ea4fbe24872b345d686ef2a6bd1f4efc",
        136712,
        40,
    ),
    (
        5,
        560,
        "266fe82bf9d903fd66efed64a0e100ebd0a5d856c0b7b75b7df6f1d34909d285",
        "57f4a1cc9736cf0b8c6f89a1c237ed023becbbffdf044ed560fa1fe419e2f438",
        136736,
        40,
    ),
    (
        6,
        672,
        "78d352fb62bee24c64f45f3ba6651826068ab65c3fe95ece7a7b002439493083",
        "ef305ef24ce38000fc2df9a56175910c9904f8dd19860ca89ac8e199d469e395",
        136680,
        40,
    ),
    (
        7,
        784,
        "026150a4ac0c855267dbe1aabd9e501b29d66c154f126657cc357919f94b7986",
        "aaacbaa5dc8c5d822cb1c63bd26d8d0a91aa829fd74f8cd560b7baa7a219050c",
        136648,
        40,
    ),
)


def test_amd_compile_uses_qualified_ordinary_launch_shape() -> None:
    """Pin the host compile boundary without initializing a GPU."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as kernel_module

    source = Path(kernel_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    compile_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compile_kimi_k3_megamoe_gfx950"
    )
    warmups = [
        node
        for node in ast.walk(compile_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warmup"
    ]
    assert len(warmups) == 1
    keywords = {keyword.arg: keyword.value for keyword in warmups[0].keywords}
    grid = keywords["grid"]
    assert isinstance(grid, ast.Tuple)
    assert len(grid.elts) == 1
    assert isinstance(grid.elts[0], ast.Name)
    assert grid.elts[0].id == "PROGRAMS"
    for name, expected in (
        ("num_warps", 8),
        ("num_stages", 1),
        ("waves_per_eu", 2),
        ("launch_cooperative_grid", False),
    ):
        value = keywords[name]
        assert isinstance(value, ast.Constant)
        assert value.value == expected

    calls = {
        node.func.id
        for node in ast.walk(compile_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "preflight_kimi_k3_megamoe_runtime" in calls


def _raw_tensors(device: str = "cuda:0") -> tuple[torch.Tensor, ...]:
    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.workspace import (
        allocate_kimi_k3_megamoe_workspace,
    )

    def empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    dense = (
        empty((1, 7168), torch.bfloat16),
        empty((1, 7168), torch.bfloat16),
        empty((1, 7168), torch.bfloat16),
        empty((896, 7168), torch.bfloat16),
        empty((3584, 7168), torch.bfloat16),
        empty((1536, 7168), torch.bfloat16),
        empty((7168, 768), torch.bfloat16),
        empty((3584,), torch.bfloat16),
        empty((7168, 3584), torch.bfloat16),
        empty((896,), torch.float32),
        empty((112, 6144, 1792), torch.uint8),
        empty((112, 6144, 112), torch.uint8),
        empty((112, 3584, 1536), torch.uint8),
        empty((112, 3584, 96), torch.uint8),
    )
    workspace = allocate_kimi_k3_megamoe_workspace(device)
    iris = (
        empty((10752,), torch.bfloat16),
        empty((10752,), torch.bfloat16),
        empty((2, 21, 8), torch.int64),
        empty((1,), torch.int64),
        empty((8,), torch.int64),
        torch.arange(8, device=device, dtype=torch.int64),
    )
    return dense + tuple(workspace.values()) + iris


def _immediate_offset(instruction: str) -> int:
    match = re.search(r"\boffset:(\d+)\b", instruction)
    return int(match.group(1)) if match is not None else 0


def _assert_publication_blocks(
    assembly: str,
    *,
    publish_asm_line: int,
    source_call_lines: set[int],
) -> None:
    # Match only our impure inline-assembly blocks. Compiler-inserted WBLs for
    # release atomics are separate; all-subgroup write-through publication is
    # intentionally DRAIN_ONLY and must not emit WBL in these source blocks.
    blocks = re.findall(
        rf"\.loc\s+1\s+{publish_asm_line}\s+\d+[^\n]*"
        rf"kernel\.py:{publish_asm_line}:\d+\s+@\[\s+"
        r"kernel\.py:(?P<call_line>\d+):\d+[^\n]*\n"
        r"\s*;;#asmstart\s+(?P<asm_body>.*?)\s+;;#asmend"
        r"(?P<before_barrier>.*?)\bs_barrier\b",
        assembly,
        flags=re.DOTALL,
    )
    assert len(blocks) == 10
    multiplicity = Counter(int(call_line) for call_line, _, _ in blocks)
    assert set(multiplicity) == source_call_lines
    # The output-centric path has one source publication helper shared by the
    # global W13 and W2 expert grids, so that site is emitted twice.  Every
    # other source publication occurs once in the specialization.
    assert sorted(multiplicity.values()) == [1] * 8 + [2]
    assert all(body.strip() == "s_waitcnt vmcnt(0)" for _, body, _ in blocks)
    assert all("buffer_wbl2" not in body for _, body, _ in blocks)
    assert all(
        "atomic" not in before_barrier and "store" not in before_barrier
        for _, _, before_barrier in blocks
    )


def _assert_system_poll_graphs(assembly: str) -> None:
    # Eight exact ready waits and seven completion waits retain atomic_poll's
    # elected system acquire/invalidate and its one compiler-owned LDS
    # rendezvous. Iris payload is consumed only by that elected subgroup, so
    # these 15 graphs deliberately have no source-owned ACK/second barrier.
    invalidations = list(re.finditer(r"\bbuffer_inv sc0 sc1\b", assembly))
    assert len(invalidations) == 15
    internal_acquire = re.compile(
        r"buffer_inv sc0 sc1"
        r"(?:(?!buffer_inv sc0 sc1).)*?ds_write_b8"
        r"(?:(?!buffer_inv sc0 sc1).)*?\bs_barrier\b"
        r"(?:(?!buffer_inv sc0 sc1).)*?ds_read_u8",
        flags=re.DOTALL,
    )
    for invalidation in invalidations:
        before = assembly[max(0, invalidation.start() - 1800) : invalidation.start()]
        loads = list(re.finditer(r"global_load_dwordx2[^\n]*sc0 sc1", before))
        assert loads
        assert "v_cmp_ne_u64" in before[loads[-1].end() :]
        graph = internal_acquire.match(assembly, invalidation.start())
        assert graph is not None
        assert graph.group(0).count("s_barrier") == 1
        assert ";;#asmstart" not in graph.group(0)

    # The two process-lifetime fatal call sites carry no payload dependency.
    # Their system-scope relaxed zero poll must retain the exact coherent load
    # and compiler-owned LDS boolean rendezvous, but emit neither BUFFER_INV
    # nor a source-owned VMEM acknowledgment/second workgroup barrier before
    # the poll result drives the guarded branch.
    fatal_polls = list(
        re.finditer(
            r"\bs_memrealtime\b"
            r"(?:(?!\bs_memrealtime\b|\bbuffer_inv\b).)*?"
            r"\bglobal_load_dwordx2\b[^\n]*\bsc0 sc1\b"
            r"(?:(?!\bbuffer_inv\b|\bds_write_b8\b).)*?"
            r"\bv_cmp_eq_u64(?:_e(?:32|64))?\b[^\n]*\b0,\s*v\["
            r"(?:(?!\bbuffer_inv\b|\bds_write_b8\b).)*?"
            r"\bs_memrealtime\b"
            r"(?:(?!\bbuffer_inv\b|\bds_write_b8\b).)*?"
            r"\bds_write_b8\b"
            r"(?:(?!\bs_barrier\b).)*?\bs_barrier\b"
            r"(?:(?!\bds_read_u8\b).)*?\bds_read_u8\b"
            r"(?P<post_poll>(?:(?!\bs_cbranch_).)*?"
            r"\bv_cmp_eq_u32(?:_e(?:32|64))?\b[^\n]*"
            r"(?:(?!\bs_cbranch_).)*?\bs_cbranch_\w+\b)",
            assembly,
            flags=re.DOTALL,
        )
    )
    assert len(fatal_polls) == 2
    for fatal_poll in fatal_polls:
        graph = fatal_poll.group(0)
        assert "buffer_inv" not in graph
        assert len(re.findall(r"\bs_barrier\b", graph)) == 1
        post_poll = fatal_poll.group("post_poll")
        assert "s_barrier" not in post_poll
        assert re.search(r"\bs_waitcnt\b[^\n]*\bvmcnt\(", post_poll) is None
        assert ";;#asmstart" not in post_poll


def _assert_local_poll_ack_blocks(
    assembly: str,
    *,
    wait_asm_line: int,
    source_call_lines: set[int],
    poll_source_lines: set[int],
) -> None:
    # Local GPU gates can publish payload consumed by all eight subgroups. Keep
    # their source-owned VMEM ACK and second barrier distinct from the 15 Iris
    # system polls above. Debug locations prove every retained ACK comes from
    # one of the two allowed helpers: local payload gates or merged topology.
    blocks = re.findall(
        rf"\.loc\s+1\s+{wait_asm_line}\s+\d+[^\n]*"
        rf"kernel\.py:{wait_asm_line}:\d+\s+@\[\s+"
        r"kernel\.py:(?P<call_line>\d+):\d+[^\n]*\n"
        r"\s*;;#asmstart\s+(?P<asm_body>.*?)\s+;;#asmend"
        r"(?P<before_barrier>.*?)\bs_barrier\b",
        assembly,
        flags=re.DOTALL,
    )
    assert len(blocks) == 6
    assert {int(call_line) for call_line, _, _ in blocks} == source_call_lines
    assert all(body.strip() == "s_waitcnt vmcnt(0)" for _, body, _ in blocks)
    assert all(
        "atomic" not in before_barrier and "store" not in before_barrier
        for _, _, before_barrier in blocks
    )

    # Account for every loaded inline ACK block before classifying its source
    # provenance. This prevents a newly inlined exact-system ACK with an
    # unexpected call location from escaping the local/topology allow-list.
    all_wait_blocks = re.findall(
        rf"\.loc\s+1\s+{wait_asm_line}\s+\d+[^\n]*"
        rf"kernel\.py:{wait_asm_line}:\d+\s+@\[\s+"
        r"kernel\.py:\d+:\d+[^\n]*\n"
        r"\s*;;#asmstart\s+s_waitcnt vmcnt\(0\)\s+;;#asmend",
        assembly,
    )
    assert len(all_wait_blocks) == len(blocks)

    poll_line_pattern = "|".join(str(line) for line in sorted(poll_source_lines))
    local_poll_graphs = list(
        re.finditer(
            rf"\.loc\s+1\s+(?P<poll_line>{poll_line_pattern})\s+\d+[^\n]*\n"
            r"\s*buffer_inv sc1"
            r"(?:(?!\bbuffer_inv\b).)*?ds_write_b8"
            r"(?:(?!\bbuffer_inv\b).)*?\bs_barrier\b"
            r"(?:(?!\bbuffer_inv\b).)*?ds_read_u8",
            assembly,
            flags=re.DOTALL,
        )
    )
    assert len(local_poll_graphs) == len(blocks)
    assert {
        int(graph.group("poll_line")) for graph in local_poll_graphs
    } == poll_source_lines
    for graph in local_poll_graphs:
        body = graph.group(0)
        assert body.count("s_barrier") == 1
        assert ";;#asmstart" not in body


def _assert_merged_phase_topology_isa(
    assembly: str,
    *,
    topology_poll_line: int,
    wait_asm_line: int,
    topology_ack_line: int,
    topology_call_line: int,
) -> None:
    """Classify merged phase payload publication through topology tickets."""

    lines = assembly.splitlines()

    def has_fence(index: int, before: int, after: int) -> bool:
        neighborhood = "\n".join(
            lines[max(0, index - before) : min(len(lines), index + after + 1)]
        )
        return "buffer_wbl2" in neighborhood or "buffer_inv" in neighborhood

    # All six scalar arrival additions are intentionally fence-free: the two
    # topology-ticket links, shared-down, W13, W2, and communication. Phase-zero
    # payload now drains directly into the topology hierarchy instead of using
    # a seventh flat arrival. Keep the scalar result and compiler LDS broadcast
    # live; a lane-0 tensor lowering would not establish the required
    # workgroup-wide ticket value.
    add_indices = [
        index for index, line in enumerate(lines) if "global_atomic_add_x2" in line
    ]
    relaxed_adds = [
        index for index in add_indices if not has_fence(index, before=6, after=6)
    ]
    assert len(add_indices) == 6
    assert relaxed_adds == add_indices
    for index in relaxed_adds:
        assert re.search(r"\bglobal_atomic_add_x2\b[^\n]*\bsc0\s*$", lines[index])
        broadcast = "\n".join(lines[index + 1 : index + 40])
        result = re.search(
            r"ds_write_b64.*?\bs_barrier\b.*?ds_read_b64",
            broadcast,
            flags=re.DOTALL,
        )
        assert result is not None
        assert result.group(0).count("s_barrier") == 1

    swap_indices = [
        index for index, line in enumerate(lines) if "global_atomic_swap_x2" in line
    ]
    # Six payload families have a distinct last-workgroup GPU release: merged
    # phase/topology, route-plan, shared-down, W13, W2, and communication.
    # System ready/completion and fatal swaps carry scope modifiers.
    plain_gpu_swaps = [
        index
        for index in swap_indices
        if re.search(r"\bs\[\d+:\d+\]\s*$", lines[index]) is not None
    ]
    assert len(plain_gpu_swaps) == 6
    release_swaps = [
        index for index in plain_gpu_swaps if has_fence(index, before=6, after=0)
    ]
    assert len(release_swaps) == 6
    for index in release_swaps:
        before = "\n".join(lines[max(0, index - 6) : index])
        assert re.search(
            r"buffer_wbl2 sc1\s+s_waitcnt vmcnt\(0\)\s*$",
            before,
        )

    # The exact successful topology poll acquires the phase payload. It keeps
    # atomic_poll's compiler LDS rendezvous, then explicitly acknowledges the
    # invalidation through VMEM before a second workgroup barrier permits top-k
    # or shared payload consumption.
    location = re.compile(
        rf"\.loc\s+1\s+{topology_poll_line}\s+\d+[^\n]*"
        rf"kernel\.py:{topology_poll_line}:\d+\s+@\[\s+"
        rf"kernel\.py:{topology_call_line}:\d+"
    )
    poll_start = next(
        index for index, line in enumerate(lines) if location.search(line)
    )
    compiler_poll_end = next(
        index
        for index in range(poll_start, min(len(lines), poll_start + 200))
        if "ds_read_u8" in lines[index]
    )
    poll = "\n".join(lines[poll_start : compiler_poll_end + 1])
    poll_loads = list(
        re.finditer(
            r"^\s*global_load_dwordx2[^\n]*\bsc1\s*$",
            poll,
            flags=re.MULTILINE,
        )
    )
    assert len(poll_loads) == 1
    compare = re.search(
        r"\bv_cmp_(?:eq|ne)_u64(?:_e(?:32|64))?\b",
        poll[poll_loads[0].end() :],
    )
    assert compare is not None
    load_to_compare = poll[poll_loads[0].start() : poll_loads[0].end() + compare.end()]
    assert "sc0 sc1" not in load_to_compare
    assert "buffer_wbl2" not in load_to_compare
    assert poll.count("buffer_inv sc1") == 1
    assert poll.count("s_barrier") == 1
    post_poll = "\n".join(lines[compiler_poll_end + 1 :])
    acknowledge = re.search(
        rf"\.loc\s+1\s+{wait_asm_line}\s+\d+[^\n]*"
        rf"kernel\.py:{wait_asm_line}:\d+\s+@\[\s+"
        rf"kernel\.py:{topology_ack_line}:\d+\s+@\[\s+"
        rf"kernel\.py:{topology_call_line}:\d+[^\n]*\n"
        r"\s*;;#asmstart\s+s_waitcnt vmcnt\(0\)\s+;;#asmend"
        r"(?:(?!;;#asmstart).)*?\bs_barrier\b",
        post_poll,
        flags=re.DOTALL,
    )
    assert acknowledge is not None


def _assert_split_flags_and_bounded_fatal_isa(assembly: str, rank: int) -> None:
    system_swaps = re.findall(
        r"^\s*global_atomic_swap_x2[^\n]*sc0 sc1\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    assert len(system_swaps) == 2  # ready, completion
    ready_offset = _immediate_offset(system_swaps[0])
    completion_offset = _immediate_offset(system_swaps[1])
    assert ready_offset == rank * 8
    assert completion_offset == 21 * 8 * 8 + rank * 8
    assert completion_offset - ready_offset == 1344

    # Each returned exact-generation publication retains compiler WBL/wait,
    # LDS workgroup broadcast, and a convergent workgroup barrier.
    for instruction in system_swaps:
        position = assembly.index(instruction)
        before = assembly[max(0, position - 160) : position]
        after = assembly[
            position + len(instruction) : position + len(instruction) + 1400
        ]
        assert re.search(r"buffer_wbl2 sc0 sc1\s+s_waitcnt vmcnt\(0\)\s*$", before)
        broadcast = re.search(
            r"s_waitcnt vmcnt\(0\).*?ds_write_b64.*?s_barrier.*?ds_read_b64",
            after,
            flags=re.DOTALL,
        )
        assert broadcast is not None
        assert broadcast.end() < 1200

    # Every inlined poison site emits eight one-shot system-release swaps and
    # one native GPU CAS for diagnostic election. There is no INT64 max/CAS
    # retry loop on the fail-stop path.
    fatal_swaps = re.findall(
        r"^\s*global_atomic_swap_x2[^\n]*sc1\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    diagnostic_cas = re.findall(
        r"^\s*global_atomic_cmpswap_x2[^\n]*sc0\s*$",
        assembly,
        flags=re.MULTILINE,
    )
    # Merging the former phase gate into topology removes one inlined local
    # timeout/poison site. Owner-only W2 retains one statically emitted poll,
    # while the 15 Iris ready/completion polls remain unchanged.
    assert len(fatal_swaps) - len(system_swaps) == 184
    assert len(diagnostic_cas) == 23


@pytest.mark.skipif(
    os.getenv("TOKENSPEED_TEST_MEGAMOE_COMPILE") != "1",
    reason="requires a gfx950 and the pinned ROCr process stack",
)
def test_amd_raw_megamoe_all_rank_loaded_code_contract() -> None:
    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as kernel_module

    kernel_path = Path(kernel_module.__file__)
    kernel_bytes = kernel_path.read_bytes()
    assert hashlib.sha256(kernel_bytes).hexdigest() == _EXPECTED_KERNEL_SHA256
    kernel_source = kernel_bytes.decode()
    source_lines = kernel_source.splitlines()
    source_call_lines = {
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line.strip() == "_drain_subgroup_vmem_before_barrier()"
    }
    assert len(source_call_lines) == 9
    wait_source_call_lines = {
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line.strip() == "_wait_subgroup_vmem_ack()"
    }
    assert len(wait_source_call_lines) == 2
    publish_helper_line = next(
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line.startswith("def _drain_subgroup_vmem_before_barrier()")
    )
    publish_asm_line = next(
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line_number > publish_helper_line
        and line.strip() == "gl.inline_asm_elementwise("
    )
    wait_helper_line = next(
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line.startswith("def _wait_subgroup_vmem_ack()")
    )
    wait_asm_line = next(
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line_number > wait_helper_line
        and line.strip() == "gl.inline_asm_elementwise("
    )
    tree = ast.parse(kernel_source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    topology_poll_line = next(
        node.lineno
        for node in ast.walk(functions["_poll_topology_or_poison"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    local_poll_line = next(
        node.lineno
        for node in ast.walk(functions["_poll_local_or_poison"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_poll"
    )
    topology_call_line = next(
        node.lineno
        for node in ast.walk(functions["_kimi_k3_megamoe_kernel"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_poll_topology_or_poison"
    )
    topology_ack_line = next(
        node.lineno
        for node in ast.walk(functions["_poll_topology_or_poison"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_wait_subgroup_vmem_ack"
    )

    tensors = _raw_tensors()
    for (
        rank,
        expert_start,
        hsaco_sha,
        amdgcn_sha,
        code_size,
        sgpr_lane_spills,
    ) in _EXPECTED_SPECIALIZATIONS:
        prepared = kernel_module.prepare_kimi_k3_megamoe_gfx950(
            *tensors,
            expert_start=expert_start,
            group_rank=rank,
            timeout_ns=1_000_000_000,
        )
        report = prepared.admission
        assert prepared.compiled.function is not None
        assert callable(prepared.runner)
        assert report.programs == 240
        assert report.subgroups == 8
        assert report.shared == 128 * 1024
        assert report.occupancy == 1
        assert report.compute_units == 256
        assert report.resident_capacity == 256
        assert report.resident_capacity >= report.programs
        assert not report.launch_cooperative_grid
        assert report.waves_per_eu == 2
        assert report.n_regs == 156
        assert report.n_spills == 0
        assert report.sgpr_count == 106
        assert report.vgpr_count == 156
        assert report.sgpr_spill_count == sgpr_lane_spills
        assert report.vgpr_spill_count == 0
        assert report.private_segment_fixed_size == 0
        assert not report.uses_dynamic_stack
        assert not report.uses_flat_scratch
        assert not report.has_scratch_instructions
        assert report.group_rank == rank
        assert report.expert_start == expert_start
        assert report.timeout_ns == 1_000_000_000
        assert report.code_object_size == code_size
        assert report.code_object_sha256 == hsaco_sha

        assembly = prepared.compiled.asm["amdgcn"].lower()
        assert hashlib.sha256(assembly.encode()).hexdigest() == amdgcn_sha
        _assert_publication_blocks(
            assembly,
            publish_asm_line=publish_asm_line,
            source_call_lines=source_call_lines,
        )
        _assert_system_poll_graphs(assembly)
        _assert_local_poll_ack_blocks(
            assembly,
            wait_asm_line=wait_asm_line,
            source_call_lines=wait_source_call_lines,
            poll_source_lines={local_poll_line, topology_poll_line},
        )
        _assert_merged_phase_topology_isa(
            assembly,
            topology_poll_line=topology_poll_line,
            wait_asm_line=wait_asm_line,
            topology_ack_line=topology_ack_line,
            topology_call_line=topology_call_line,
        )
        _assert_split_flags_and_bounded_fatal_isa(assembly, rank)
