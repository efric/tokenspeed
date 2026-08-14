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

"""Host-only tests for the admitted direct MegaMoE dispatch boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


def test_amd_system_waits_and_fatal_entry_protocols() -> None:
    """Guard subgroup-zero Iris waits and the relaxed fatal fast path."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    function_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    functions = {
        name: ast.get_source_segment(module_source, node)
        for name, node in function_nodes.items()
    }
    poll = functions["_poll_exact_generation"]
    fatal = functions["_fatal_epoch_is_clear"]
    assert poll is not None
    assert fatal is not None
    assert "matched = gl.atomic_poll(" in poll
    assert "flag,\n        generation," in poll
    assert 'sem="acquire"' in poll
    assert 'scope="sys"' in poll
    assert "timeout_ns=TIMEOUT_NS" in poll
    assert 'observed = gl.load(flag, cache_modifier=".cv")' in poll
    assert "_wait_subgroup_vmem_ack()" not in poll
    assert "gl.barrier()" not in poll
    assert "atomic_add" not in poll
    assert "while " not in poll
    assert "gl.atomic_poll(" in fatal
    assert 'sem="relaxed"' in fatal
    assert 'scope="sys"' in fatal
    assert "timeout_ns=0" in fatal
    assert "_wait_subgroup_vmem_ack()" not in fatal
    assert "gl.barrier()" not in fatal


def test_amd_peer_wait_poison_wraps_exact_atomic_poll() -> None:
    """Keep timeout poison separate from exact cross-NUMA flag progress."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    wait = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_wait_peer_epoch_or_poison"
    )
    source = ast.get_source_segment(module_source, wait)
    assert source is not None
    assert "ready, acquired = _poll_exact_generation(" in source
    assert "peer_flag,\n        expected,\n        TIMEOUT_NS," in source
    assert "atomic_add" not in source
    assert "while " not in source
    assert "_poison_all_ranks(" in source
    assert "return ready, acquired" in source


def test_amd_fatal_poison_uses_only_bounded_one_shot_atomics() -> None:
    """Reject compiler-generated retry loops in the timeout fail-stop path."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    poison = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_poison_all_ranks"
    )
    source = ast.get_source_segment(module_source, poison)
    assert source is not None
    assert "atomic_max" not in source
    assert (
        'gl.atomic_xchg(peer_fatal, generation, sem="release", scope="sys")' in source
    )
    assert "previous = gl.atomic_cas(" in source
    assert (
        "fail_diagnostics,\n        gl.full((), 0, gl.int64),\n        generation,"
        in source
    )
    assert "if previous == 0:" in source


def test_amd_kernel_entry_checks_sticky_fatal_before_any_write() -> None:
    """Guard fatal entry before generation derivation and phase-zero writes."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_kimi_k3_megamoe_kernel"
    )
    source = ast.get_source_segment(module_source, kernel)
    assert source is not None
    entry_check = source.index("if not _fatal_epoch_is_clear(fatal_epoch):")
    early_return = source.index("return", entry_check)
    generation_load = source.index(
        'generation = gl.load(topology_gate, cache_modifier=".cv") + 1'
    )
    first_shared_write = source.index("residency_tile.store(")
    first_phase = source.index("_phase0_projection(")
    assert entry_check < early_return < generation_load < first_shared_write
    assert entry_check < early_return < generation_load < first_phase
    assert source.count("phase_arrival") == 1
    assert source.count("phase_gate") == 1
    assert "_fixed_grid_publish(" not in source


def test_amd_topology_uses_exact_cumulative_xcd_tickets() -> None:
    """Guard the dense ticket proof and merged payload publication."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    topology = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_arrive_topology_ticket"
    )
    topology_poll = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_poll_topology_or_poison"
    )
    source = ast.get_source_segment(module_source, topology)
    poll_source = ast.get_source_segment(module_source, topology_poll)
    assert source is not None
    assert poll_source is not None
    bounds = source.index("if xcc < _XCDS:")
    ticket = source.index("ticket = gl.atomic_add(", bounds)
    assert bounds < ticket
    assert "xcc_ticket + xcc" in source[ticket:]
    assert "ticket % _WORKERS_PER_XCD" in source
    assert "ticket // _WORKERS_PER_XCD + 1" in source
    assert "ticket_generation == generation" in source
    assert "local_rank == _WORKERS_PER_XCD - 1" in source
    assert "xcd_ticket = gl.atomic_add(" in source
    assert "topology_generation = xcd_ticket // _XCDS + 1" in source
    assert "xcd_ticket % _XCDS == _XCDS - 1" in source
    assert "topology_generation == generation" in source
    assert "gl.atomic_xchg(\n                    topology_gate," in source
    assert source.count('sem="relaxed"') == 2
    assert source.count('sem="release"') == 1
    assert 'sem="acquire"' not in source
    assert "matched = gl.atomic_poll(" in poll_source
    assert poll_source.count('sem="acquire"') == 2
    assert 'sem="relaxed"' not in poll_source
    assert 'scope="gpu"' in poll_source
    assert "timeout_ns=TIMEOUT_NS" in poll_source
    assert "_poison_all_ranks(" in poll_source
    assert "_wait_subgroup_vmem_ack()" in poll_source
    assert "gl.barrier()" in poll_source
    assert "worker_xcc" not in module_source
    assert "observed_xcc" not in module_source
    assert "_topology_consensus" not in module_source


def test_amd_publications_drain_each_subgroup_before_barrier() -> None:
    """Guard write-through VMEM completion before elected releases."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    function_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    functions = {
        name: ast.get_source_segment(module_source, node)
        for name, node in function_nodes.items()
    }
    publication = functions["_drain_subgroup_vmem_before_barrier"]
    communication = functions["_iris_communication_tile"]
    assert publication is not None
    assert communication is not None
    assert '"s_waitcnt vmcnt(0)"' in publication
    assert "buffer_wbl2" not in publication
    assert "is_pure=False" in publication

    def publication_call_count(function_name: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_drain_subgroup_vmem_before_barrier"
            for node in ast.walk(function_nodes[function_name])
        )

    assert publication_call_count("_iris_communication_tile") == 2

    expected_local_publications = {
        "_kimi_k3_megamoe_kernel": 1,
        "_sigmoid_bias_top16": 2,
        "_publish_route_plan": 1,
        "_shared_down_produce": 1,
        "_expert_route_phase": 0,
        "_expert_grid_arrive": 1,
        "_zero_routed_producer": 1,
        "_expert_phases": 0,
    }
    for function_name, expected_count in expected_local_publications.items():
        assert publication_call_count(function_name) == expected_count
    assert (
        sum(
            publication_call_count(function_name)
            for function_name in (
                "_kimi_k3_megamoe_kernel",
                "_sigmoid_bias_top16",
                "_publish_route_plan",
                "_shared_down_produce",
                "_expert_grid_arrive",
                "_zero_routed_producer",
                "_iris_communication_tile",
            )
        )
        == 9
    )


def test_amd_payload_arrival_families_are_unmasked_relaxed_rmw() -> None:
    """Keep ordering in drains/releases instead of arrival acquire fences."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for function_name in (
        "_shared_down_produce",
        "_expert_grid_arrive",
        "_iris_communication_tile",
    ):
        arrivals = [
            node
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "atomic_add"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == 1
        ]
        assert len(arrivals) == 1
        keywords = {keyword.arg: keyword.value for keyword in arrivals[0].keywords}
        assert isinstance(keywords["sem"], ast.Constant)
        assert keywords["sem"].value == "relaxed"
        assert isinstance(keywords["scope"], ast.Constant)
        assert keywords["scope"].value == "gpu"
        assert "mask" not in keywords


def test_amd_expert_dispatch_emits_at_most_two_heavy_route_bodies() -> None:
    """Guard the inverse XCD mapping that replaces the 16-body unroll."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    phases = functions["_expert_phases"]
    route = functions["_expert_route_phase"]
    fused = functions["_output_centric_w2_combine"]
    zero = functions["_zero_routed_producer"]
    entry = functions["_kimi_k3_megamoe_kernel"]
    assert phases is not None
    assert route is not None
    assert fused is not None
    assert zero is not None
    assert entry is not None
    assert "gl.static_range(0, 16)" not in phases
    assert phases.count("_expert_route_phase(") == 2
    assert "base = 8 // route_count" in phases
    assert "extra = 8 % route_count" in phases
    assert "long_xcds = extra * long" in phases
    assert "xcd_ordinal // long" in phases
    assert "extra + (xcd_ordinal - long_xcds) // base" in phases
    assert "second_route = xcd_ordinal + 8" in phases
    assert "(route_count > 8) & (second_route < route_count)" in phases
    assert route.count("_w13_route_tiles(") == 1
    assert route.count("_w2_route_tiles(") == 0
    assert "expert_count = 28 if prior_xcd < 6 else 29" in route
    assert "xcc_population" not in route
    assert phases.count("_expert_grid_arrive(") == 2
    assert phases.count("_poll_local_or_poison(") == 2
    assert phases.count("_output_centric_w2_combine(") == 1
    assert "expert_count = 28 if prior_xcd < 6 else 29" in phases
    assert "while task < _W2_TASKS" in fused
    assert "task += _EXPERT_WORKGROUPS" in fused
    assert "while route < route_count" in fused
    assert "route += 1" in fused
    assert "gl.static_range(0, 16)" not in fused
    assert "route_acc.to(gl.bfloat16).to(gl.float32)" in fused
    assert fused.index("route_acc.to(gl.bfloat16).to(gl.float32)") < fused.index(
        "combine_acc += route_value * route_weight"
    )
    assert "symmetric_producer + _HIDDEN" in fused
    assert "w2_route_output" not in fused
    assert "route_count == 0" in zero
    assert "comm_index >= 14" in zero
    assert "comm_index < 21" in zero
    assert "_drain_subgroup_vmem_before_barrier()" in zero
    assert "_routed_combine(" not in entry
    assert "_zero_routed_producer(" in entry
    assert entry.count("w2_route_output") == 1
    plan = functions["_publish_route_plan"]
    assert plan is not None
    assert "expert_count = 28 if xcd_ordinal < 6 else 29" in plan
    assert "xcc_population" not in plan
    assert "if count > 0" in plan
    assert plan.count("current_w13 + _EXPERT_WORKGROUPS") == 1
    assert plan.count("current_w2 + _EXPERT_WORKGROUPS") == 1
    assert "w13_target + route" not in plan
    assert "w2_target + route" not in plan


def test_amd_w2_gate_acquire_owners_match_routed_iris_tiles() -> None:
    """Map the seven routed Iris tiles to seven expert workgroups exactly."""

    owners = []
    for xcc in range(8):
        for local_rank in range(30):
            comm_index = local_rank * 8 + xcc
            expert_local_rank = local_rank - (2 if xcc < 6 else 1)
            if 14 <= comm_index < 21:
                owners.append(
                    (comm_index, xcc, local_rank, expert_local_rank),
                )
    assert sorted(owners) == [
        (14, 6, 1, 0),
        (15, 7, 1, 0),
        (16, 0, 2, 0),
        (17, 1, 2, 0),
        (18, 2, 2, 0),
        (19, 3, 2, 0),
        (20, 4, 2, 0),
    ]
    assert len({comm_index for comm_index, *_ in owners}) == 7


def test_amd_w2_owner_only_acquire_keeps_timeout_order_and_l0_bypass() -> None:
    """Keep both expert arrivals global but W2 acquire on routed owners only."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    function_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    phases_node = function_nodes["_expert_phases"]
    phases = ast.get_source_segment(module_source, phases_node)
    arrive = ast.get_source_segment(
        module_source,
        function_nodes["_expert_grid_arrive"],
    )
    assert phases is not None
    assert arrive is not None

    # L=0 and non-expert workgroups bypass both arrival and timeout paths via
    # the single workgroup-uniform outer guard.
    guarded = [
        node
        for node in phases_node.body
        if isinstance(node, ast.If)
        and ast.get_source_segment(module_source, node.test)
        == "(route_count > 0) & (expert_local_rank >= 0)"
    ]
    assert len(guarded) == 1
    guarded_source = ast.get_source_segment(module_source, guarded[0])
    assert guarded_source is not None
    assert guarded_source.count("_expert_grid_arrive(") == 2
    assert guarded_source.count("_poll_local_or_poison(") == 2

    # Every producer drains and arrives before the last workgroup releases.
    # W13 then acquires on all 226 producers. W2 performs the same global
    # arrival/release, but only the seven uniform routed owners enter its
    # bounded acquire/timeout path before publishing to Iris.
    assert arrive.index("_drain_subgroup_vmem_before_barrier()") < arrive.index(
        "gl.atomic_add("
    )
    assert arrive.index("gl.atomic_add(") < arrive.index("gl.atomic_xchg(")
    assert "_poll_local_or_poison(" not in arrive
    w13_arrive = phases.index("expected_w13 = _expert_grid_arrive(")
    w13_poll = phases.index("w13_ready = _poll_local_or_poison(")
    w2_compute = phases.index("_output_centric_w2_combine(")
    w2_arrive = phases.index("expected_w2 = _expert_grid_arrive(")
    owner_predicate = phases.index("w2_owner = (comm_index >= 14) & (comm_index < 21)")
    owner_branch = phases.index("if w2_owner:")
    w2_poll = phases.index("ready = _poll_local_or_poison(", owner_branch)
    assert (
        w13_arrive
        < w13_poll
        < w2_compute
        < w2_arrive
        < owner_predicate
        < owner_branch
        < w2_poll
    )
    assert "TIMEOUT_NS" in phases[w13_poll:w2_compute]
    assert "TIMEOUT_NS" in phases[w2_poll:]


def test_amd_phase0_uses_three_fixed_base_pair_specializations() -> None:
    """Reject generic pointer selection and masked shared-up accumulation."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    projection = functions["_phase0_projection"]
    router = functions["_phase0_router_pair"]
    routed = functions["_phase0_routed_pair"]
    shared = functions["_phase0_shared_pair"]
    assert projection is not None
    assert router is not None
    assert routed is not None
    assert shared is not None

    assert projection.count("_phase0_router_pair(") == 2
    assert projection.count("_phase0_routed_pair(") == 1
    assert projection.count("_phase0_shared_pair(") == 1
    assert "if pid < 112:" in projection
    assert "if pid < 208:" in projection
    assert "if pid < 192:" in projection
    assert "pid - 80" in projection
    assert "pid - 112" in projection
    assert "pid - 208" in projection
    assert "gl.where" not in projection
    assert "gl.static_range" not in projection

    assert "task = pair + cohort * 112" in router
    assert "lane_n = gl.arange(0, 4" in router
    assert "task = pair + cohort * 112" in routed
    assert "lane_n = gl.arange(0, 16" in routed
    assert "task = pair + cohort * 96" in shared
    assert "lane_n = gl.arange(0, 4" in shared
    for helper, weight_name in (
        (router, "router_weight"),
        (routed, "routed_down_weight"),
        (shared, "shared_gate_up_weight"),
    ):
        assert f"gl.amd.cdna4.buffer_load(\n            {weight_name}," in helper
        assert ").to(gl.int32)" in helper
        assert "weight_base" not in helper
        assert "weight_ptrs" not in helper
        assert "valid_task" not in helper
        assert "width_valid" not in helper
        assert "[2, 4, 1]" in helper
        assert "mask=" not in helper
        assert "gl.int64" not in helper
    assert "up_acc" not in router
    assert "up_acc" not in routed
    assert "up_acc" in shared


def test_amd_local_gate_acquire_ack_converges_before_payload_use() -> None:
    """Guard the qualified elected local-acquire protocol in one helper."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    wait = functions["_wait_subgroup_vmem_ack"]
    poll = functions["_poll_local_or_poison"]
    assert wait is not None
    assert poll is not None
    assert '"s_waitcnt vmcnt(0)"' in wait
    assert "is_pure=False" in wait
    poison = poll.index("_poison_all_ranks(")
    acknowledgment = poll.index("_wait_subgroup_vmem_ack()")
    barrier = poll.index("gl.barrier()", acknowledgment)
    result = poll.index("return matched != 0")
    assert poison < acknowledgment < barrier < result


def test_amd_iris_consumes_peer_payload_in_subgroup_zero() -> None:
    """Guard the proven one-subgroup remote payload consumption shape."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    communication = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_iris_communication_tile"
    )
    source = ast.get_source_segment(module_source, communication)
    assert source is not None
    assert "consumer_lane = lane < 64" in source
    assert "for word_group in gl.static_range(0, 2):" in source

    assert "completion_plane: gl.constexpr = 21 * 8" in source
    ready_release = source.index("previous_ready = gl.atomic_xchg(")
    local_wait = source.index(
        "local_ready, _ = _wait_peer_epoch_or_poison(",
        ready_release,
    )
    local_load = source.index("local_packed = gl.amd.cdna4.buffer_load(")
    assert ready_release < local_wait < local_load
    assert "previous_ready != generation - 1" in source[ready_release:local_wait]

    peer_wait = source.index("peer_ready, _ = _wait_peer_epoch_or_poison(")
    peer_reduce = source.index("peer_packed = gl.amd.cdna4.buffer_load(")
    assert peer_wait < peer_reduce
    peer_region = source[local_load : source.index("# Publish completion")]
    assert "mask=consumer_lane" in peer_region
    assert source.count("for distance in gl.static_range(1, _WORLD_SIZE):") == 2
    assert source.count("peer = RANK ^ distance") == 2
    # Synchronization order is XOR, but FP32 accumulation order remains the
    # reference rank order so the fused result stays bitwise deterministic.
    assert "for peer in gl.static_range(0, _WORLD_SIZE):" in peer_region
    completion_release = source.index("previous_completion = gl.atomic_xchg(")
    completion_wait = source.index(
        "peer_complete, _ = _wait_peer_epoch_or_poison(",
        completion_release,
    )
    assert completion_release < completion_wait
    assert (
        "previous_completion != generation - 1"
        in source[completion_release:completion_wait]
    )


def test_amd_phase_payload_is_acquired_before_topk_and_shared() -> None:
    """Pin phase stores, merged ticket publication, and payload consumption."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_kimi_k3_megamoe_kernel"
    )
    source = ast.get_source_segment(module_source, kernel)
    assert source is not None
    generation = source.index(
        'generation = gl.load(topology_gate, cache_modifier=".cv") + 1'
    )
    phase = source.index("_phase0_projection(", generation)
    drain = source.index("_drain_subgroup_vmem_before_barrier()", phase)
    barrier = source.index("gl.barrier()", drain)
    ticket_arrival = source.index("local_rank = _arrive_topology_ticket(", barrier)
    topology_wait = source.index(
        "topology_ready = _poll_topology_or_poison(", ticket_arrival
    )
    topology_return = source.index("if not topology_ready:", topology_wait)
    topk = source.index("_sigmoid_bias_top16(", topology_return)
    shared = source.index("_shared_down_produce(", topk)
    route_plan = source.index("_publish_route_plan(", shared)
    assert (
        generation
        < phase
        < drain
        < barrier
        < ticket_arrival
        < topology_wait
        < topology_return
        < topk
        < shared
        < route_plan
    )
    assert "_fixed_grid_publish" not in module_source
    assert "topology_status" not in module_source


def test_amd_mutable_payloads_use_system_coherent_cache_controls() -> None:
    """Keep broad consumer invalidation from discarding dirty scratch lines."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    module_source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)

    global_stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gl"
        and node.func.attr == "store"
    ]
    assert global_stores
    for store in global_stores:
        modifiers = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in store.keywords
            if keyword.arg == "cache_modifier"
        }
        assert modifiers == {"cache_modifier": ".wt"}

    buffer_stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "buffer_store"
    ]
    assert buffer_stores
    for store in buffer_stores:
        modifiers = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in store.keywords
            if keyword.arg == "cache"
        }
        assert modifiers == {"cache": ".wt"}

    # Every mutable payload family crossing a gate has an explicit volatile
    # consumer. Immutable inputs and weights intentionally retain .ca/.cs.
    required_volatile_operands = (
        "router_logits",
        "routed_input",
        "shared_input",
        "topk_ids",
        "topk_weights",
        "local_route_count",
        "local_route_ids",
        "local_route_weights",
        "route_xcd_start",
        "route_xcd_count",
        "route_worker_count",
        "target_ptr",
        "w13_intermediate",
        "w2_route_output",
        "producer_u64",
        "peer_producer",
        "symmetric_reduced",
    )
    for operand in required_volatile_operands:
        assert any(
            operand in ast.get_source_segment(module_source, node.args[0])
            and any(
                keyword.arg in {"cache", "cache_modifier"}
                and ast.literal_eval(keyword.value) == ".cv"
                for keyword in node.keywords
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"load", "buffer_load"}
        ), operand


def test_amd_failstop_graph_captures_two_nodes_around_state_snapshots() -> None:
    """Guard the EP8 proof shape without launching a collective kernel."""

    e2e_path = Path(__file__).with_name("test_amd_e2e.py")
    module_source = e2e_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    capture = functions["_capture_two_plan_failstop_graph"]
    assert capture is not None
    first_launch = capture.index('decode(state["hidden"], state["prefix"], first_plan)')
    first_snapshot = capture.index("before_second[name].copy_(tensor)")
    second_launch = capture.index(
        'decode(state["hidden"], state["prefix"], second_plan)'
    )
    second_snapshot = capture.index("after_second[name].copy_(tensor)")
    assert first_launch < first_snapshot < second_launch < second_snapshot
    assert "first_plan.workspace[name].data_ptr()" in capture
    assert "second_plan.workspace[name].data_ptr()" in capture


def test_amd_local_publication_replays_fresh_workgroups_per_generation() -> None:
    """Reject scalar-loop re-entry as evidence of subgroup participation."""

    litmus_path = Path(__file__).with_name("test_amd_local_publication.py")
    module_source = litmus_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    kernel = functions["_local_publication_kernel"]
    assert not any(isinstance(node, ast.While) for node in ast.walk(kernel))
    kernel_source = ast.get_source_segment(module_source, kernel)
    run_source = ast.get_source_segment(module_source, functions["_run_variant"])
    assert kernel_source is not None
    assert run_source is not None
    assert "ITERATIONS" not in kernel_source
    assert 'generation = gl.load(generation_ptr, cache_modifier=".cv")' in kernel_source
    assert "mismatch = consumer_lane & ready" not in kernel_source
    assert "mismatch = ready &" not in kernel_source
    assert "mask=subgroup_leader & (not ready)" in kernel_source
    assert "mask=subgroup_leader & (not complete)" in kernel_source
    assert "for generation_value in range(iterations):" in run_source
    assert "generation_values[generation_value]" in run_source
    assert 'side_effect=AssertionError("generation replay attempted to recompile")' in (
        run_source
    )
    assert "retained_cache_identity == _loaded_kernel_cache_identity(" in run_source
    assert run_source.count("_local_publication_kernel[(_PROGRAMS,)](") == 1


def test_amd_prepared_launch_calls_only_retained_runner(monkeypatch) -> None:
    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    calls: list[tuple[tuple[object, ...], object]] = []

    def runner(*args, stream=None) -> None:
        calls.append((args, stream))

    prepared = module.PreparedKimiK3MegaMoEKernel(
        compiled=object(),
        runner=runner,
        admission=object(),
        device=torch.device("cuda:0"),
        expert_start=0,
        group_rank=0,
        timeout_ns=1_000_000_000,
        beta=4.0,
        linear_beta=25.0,
        rms_eps=1.0e-5,
    )
    raw = tuple(object() for _ in module.KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES)
    sentinel_stream = object()

    monkeypatch.setattr(module, "_validate_direct_tensor_contract", lambda *_: None)
    monkeypatch.setattr(
        module,
        "compile_kimi_k3_megamoe_gfx950",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct dispatch entered the compiler")
        ),
    )
    monkeypatch.setattr(
        module,
        "admit_kimi_k3_megamoe_compiled_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct dispatch repeated admission")
        ),
    )

    module.launch_kimi_k3_megamoe_gfx950(
        prepared,
        *raw,
        stream=sentinel_stream,
    )

    assert calls == [(raw + (4.0, 25.0, 1.0e-5), sentinel_stream)]


@pytest.mark.parametrize(
    ("group_rank", "expert_start", "timeout_ns", "message"),
    (
        (0, 0, 999_999_999, "unqualified timeout_ns"),
        (0, 112, 1_000_000_000, "unqualified specialization"),
        (8, 896, 1_000_000_000, "unqualified specialization"),
        (-1, -112, 1_000_000_000, "unqualified specialization"),
    ),
)
def test_amd_admission_rejects_unqualified_constexpr_before_preflight(
    group_rank: int,
    expert_start: int,
    timeout_ns: int,
    message: str,
) -> None:
    """Reject unsupported binary specializations without touching a device."""

    from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe import admission

    with pytest.raises(RuntimeError, match=message):
        admission.admit_kimi_k3_megamoe_compiled_kernel(
            object(),
            group_rank=group_rank,
            expert_start=expert_start,
            timeout_ns=timeout_ns,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_amd_prepared_launch_rejects_offset_contiguous_input_before_runner() -> None:
    """An offset BF16 view must fail the raw 16-byte pointer contract."""

    import tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.kernel as module

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    storage = torch.empty(7169, dtype=torch.bfloat16, device=device)
    misaligned = storage.narrow(0, 1, 7168).view(1, 7168)
    aligned = torch.empty((1, 7168), dtype=torch.bfloat16, device=device)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0

    def runner(*_args, **_kwargs) -> None:
        raise AssertionError("misaligned input reached the retained runner")

    prepared = module.PreparedKimiK3MegaMoEKernel(
        compiled=object(),
        runner=runner,
        admission=object(),
        device=device,
        expert_start=0,
        group_rank=0,
        timeout_ns=1_000_000_000,
        beta=4.0,
        linear_beta=25.0,
        rms_eps=1.0e-5,
    )
    raw = (misaligned,) + (aligned,) * (
        len(module.KIMI_K3_MEGAMOE_RAW_TENSOR_NAMES) - 1
    )

    with pytest.raises(ValueError, match="hidden_states must be 16-byte aligned"):
        module.launch_kimi_k3_megamoe_gfx950(prepared, *raw)
