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

"""Verify the host-only Kimi K3 TP8/EP1 clean-sheet design oracle.

This intentionally imports neither Torch nor ROCm. It checks checkpoint-derived
shapes, task/traffic/resource formulas, the event DAG, and small deterministic
top-k/BF16 materialization cases.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

_DEFAULT_ORACLE = (
    Path(__file__).resolve().parents[2]
    / "docs/design/probes/kimi-k3-tp8-cleansheet-oracle-20260814.json"
)


class OracleError(AssertionError):
    """A clean-sheet oracle invariant did not hold."""


class Checker:
    """Count named checks and raise useful errors."""

    def __init__(self) -> None:
        self.count = 0

    def equal(self, actual: Any, expected: Any, name: str) -> None:
        self.count += 1
        if actual != expected:
            raise OracleError(f"{name}: expected {expected!r}, got {actual!r}")

    def close(
        self,
        actual: float,
        expected: float,
        name: str,
        *,
        rel_tol: float = 1e-12,
        abs_tol: float = 1e-9,
    ) -> None:
        self.count += 1
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise OracleError(f"{name}: expected {expected!r}, got {actual!r}")

    def true(self, condition: bool, name: str) -> None:
        self.count += 1
        if not condition:
            raise OracleError(f"{name}: condition is false")


def _float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _bf16_bits(value: float) -> int:
    """Round a Python value through IEEE FP32 to BF16, ties to even."""

    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _bf16_float(value: float) -> float:
    bits = _bf16_bits(value) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _fp32_sum_in_order(values: list[float]) -> float:
    acc = _float32(0.0)
    for value in values:
        acc = _float32(acc + _float32(value))
    return acc


def _shape_oracles(data: dict[str, Any], check: Checker) -> None:
    d = data["dimensions"]
    layouts = data["rank_local_layouts"]
    hidden = d["hidden"]
    latent = d["routed_latent"]
    experts = d["experts"]
    tp = d["tp_world"]
    intermediate = d["routed_intermediate_global"]
    local = intermediate // tp
    shared_local = d["shared_intermediate_global"] // tp

    check.equal(d["ep_world"], 1, "EP world")
    check.equal(d["routed_intermediate_local"], local, "local routed I")
    check.equal(d["shared_intermediate_local"], shared_local, "local shared I")
    check.equal(local % d["mxfp4_group"], 0, "K384 MXFP4 grouping")
    check.equal(layouts["router_weight_bf16"], [experts, hidden], "router shape")
    check.equal(
        layouts["routed_down_weight_bf16"], [latent, hidden], "routed down shape"
    )
    check.equal(
        layouts["shared_gate_up_weight_bf16"],
        [2 * shared_local, hidden],
        "shared gate/up shape",
    )
    check.equal(
        layouts["shared_down_weight_bf16"],
        [hidden, shared_local],
        "shared down shape",
    )
    check.equal(layouts["routed_up_weight_bf16"], [hidden, latent], "routed up shape")
    check.equal(
        layouts["w13_packed_u8"],
        [experts, 2 * local, latent // 2],
        "W13 packed shape",
    )
    check.equal(
        layouts["w13_scale_u8"],
        [experts, 2 * local, latent // d["mxfp4_group"]],
        "W13 scale shape",
    )
    check.equal(
        layouts["w2_packed_u8"],
        [experts, latent, local // 2],
        "W2 packed shape",
    )
    check.equal(
        layouts["w2_scale_u8"],
        [experts, latent, local // d["mxfp4_group"]],
        "W2 scale shape",
    )
    check.equal(layouts["allow_k384_to_k512_padding"], False, "no K padding")
    check.equal(
        layouts["allow_raw_plus_preshuffled_full_expert_copy"],
        False,
        "single expert representation",
    )
    materialization = data["materialization"]
    check.equal(
        materialization["joint_lane_order"],
        ["shared_hidden_7168", "routed_latent_3584"],
        "joint lane segment order",
    )
    check.equal(
        materialization["route_combine"],
        "fp32_original_topk_slot_order_then_bf16",
        "route combine order",
    )
    check.equal(
        materialization["joint_reduce"],
        "fp32_local_rank_first_then_increasing_peer_group_rank_excluding_local_then_bf16",
        "rank reduction order",
    )
    check.equal(
        materialization["final_add"],
        "fp32_routed_then_prefix_then_shared_then_bf16",
        "final add order",
    )
    reduction = data["reduction_contract_decision"]
    check.equal(
        reduction["primary"],
        "qualified_k3_iris_destination_local_first_then_increasing_peers",
        "primary reduction contract",
    )
    check.equal(
        reduction["destination_rank_specific_last_bits_possible"],
        True,
        "destination-specific last bits",
    )
    check.equal(
        reduction["global_0_to_7_common_lane"],
        "later_new_numerical_contract_only",
        "global-order status",
    )


def _batch_oracles(data: dict[str, Any], check: Checker) -> None:
    d = data["dimensions"]
    traffic = data["weight_traffic_bytes_per_rank"]
    hidden = d["hidden"]
    latent = d["routed_latent"]
    experts = d["experts"]
    topk = d["topk"]
    world = d["tp_world"]
    local_i = d["routed_intermediate_local"]
    lane_width = hidden + latent
    tile = d["lane_tile_elements"]
    fixed = traffic["fixed_per_batch"]
    expert_bytes = traffic["one_selected_expert_total"]

    for batch in d["target_batches"]:
        b = data["batches"][str(batch)]
        lane_elements = batch * lane_width
        lane_bytes = 2 * lane_elements
        tiles = (lane_elements + tile - 1) // tile
        check.equal(b["route_slots"], batch * topk, f"B{batch} route slots")
        check.equal(b["joint_lane_elements"], lane_elements, f"B{batch} lane elements")
        check.equal(b["joint_lane_bytes_bf16"], lane_bytes, f"B{batch} lane bytes")
        check.equal(b["joint_tiles_512"], tiles, f"B{batch} lane tiles")
        check.equal(
            b["j0_comm_programs_at_p224"], min(tiles, 224), f"B{batch} comm programs"
        )
        comm_programs = min(tiles, 224)
        flag_shape = [2, comm_programs, world]
        flag_bytes = math.prod(flag_shape) * 8
        check.equal(
            b["j0_ready_flag_shape_int64"],
            flag_shape,
            f"B{batch} ready flag shape",
        )
        check.equal(
            b["j0_completion_flag_shape_int64"],
            flag_shape,
            f"B{batch} completion flag shape",
        )
        check.equal(b["j0_ready_flag_bytes"], flag_bytes, f"B{batch} ready flag bytes")
        check.equal(
            b["j0_completion_flag_bytes"],
            flag_bytes,
            f"B{batch} completion flag bytes",
        )
        check.equal(
            b["j0_ready_plus_completion_flag_bytes"],
            2 * flag_bytes,
            f"B{batch} combined flag bytes",
        )
        check.equal(
            b["j0_remote_read_bytes_per_rank"],
            (world - 1) * lane_bytes,
            f"B{batch} remote reads",
        )
        check.equal(
            b["j0_contribution_load_bytes_per_rank"],
            world * lane_bytes,
            f"B{batch} contribution loads",
        )
        check.equal(
            b["j0_global_peer_wire_bytes"],
            world * (world - 1) * lane_bytes,
            f"B{batch} J0 wire bytes",
        )
        check.equal(
            b["j1_global_two_phase_wire_bytes"],
            2 * (world - 1) * lane_bytes,
            f"B{batch} J1 wire bytes",
        )
        check.equal(
            b["input_projection_logical_n16_tiles"],
            (experts + latent + 2 * d["shared_intermediate_local"]) // 16,
            f"B{batch} input projection tiles",
        )
        check.equal(b["topk_ids_int32_bytes"], batch * topk * 4, f"B{batch} top-k IDs")
        check.equal(
            b["topk_weights_fp32_bytes"],
            batch * topk * 4,
            f"B{batch} top-k weights",
        )
        check.equal(
            b["w13_logical_n16_tasks"],
            batch * topk * (local_i // 16),
            f"B{batch} W13 tasks",
        )
        check.equal(
            b["w2_token_n16_tasks"],
            batch * (latent // 16),
            f"B{batch} W2 tasks",
        )
        check.equal(b["logits_bytes"], batch * experts * 4, f"B{batch} logits")
        check.equal(
            b["routed_latent_bytes"], batch * latent * 2, f"B{batch} routed latent"
        )
        check.equal(
            b["shared_activation_bytes"],
            batch * d["shared_intermediate_local"] * 2,
            f"B{batch} shared activation",
        )
        check.equal(
            b["routed_activation_bytes"],
            batch * topk * local_i * 2,
            f"B{batch} routed activation",
        )
        check.equal(
            b["route_output_storage_bytes_if_materialized"],
            batch * topk * latent * 2,
            f"B{batch} route output storage",
        )
        check.equal(
            b["producer_and_reduced_two_slot_ring_bytes"],
            4 * lane_bytes,
            f"B{batch} two-slot lane ring",
        )

        expected_unique = experts * (1.0 - (1.0 - topk / experts) ** batch)
        duplicate = batch * topk - expected_unique
        check.close(
            b["expected_unique_experts_uniform"],
            expected_unique,
            f"B{batch} expected unique experts",
        )
        check.close(
            b["expected_duplicate_slots_uniform"],
            max(0.0, duplicate),
            f"B{batch} duplicate slots",
        )
        check.close(
            b["expected_duplicate_fraction_uniform"],
            max(0.0, duplicate / (batch * topk)),
            f"B{batch} duplicate fraction",
        )
        ideal = fixed + expected_unique * expert_bytes
        check.close(
            b["ideal_unique_weight_bytes_uniform"],
            ideal,
            f"B{batch} ideal unique weight bytes",
            abs_tol=1e-5,
        )
        check.close(
            b["ideal_unique_weight_mib_uniform"],
            ideal / (1 << 20),
            f"B{batch} ideal unique weight MiB",
        )
        check.equal(
            b["naive_no_reuse_weight_bytes"],
            batch * traffic["b1_total"],
            f"B{batch} naive weights",
        )
        if batch >= 8:
            check.equal(
                b["route_output_write_read_bytes_if_materialized"],
                2 * b["route_output_storage_bytes_if_materialized"],
                f"B{batch} route output traffic",
            )


def _traffic_and_flop_oracles(data: dict[str, Any], check: Checker) -> None:
    d = data["dimensions"]
    t = data["weight_traffic_bytes_per_rank"]
    f = data["linear_flops_per_token_per_rank"]
    hidden = d["hidden"]
    latent = d["routed_latent"]
    experts = d["experts"]
    topk = d["topk"]
    local_i = d["routed_intermediate_local"]
    shared_i = d["shared_intermediate_local"]

    check.equal(t["scope"], "linear_weight_payloads_only", "traffic model scope")
    check.equal(
        t["excluded_routed_norm_bf16_bytes"],
        latent * 2,
        "excluded routed norm bytes",
    )
    check.equal(
        t["excluded_correction_bias_fp32_bytes"],
        experts * 4,
        "excluded correction bias bytes",
    )
    check.equal(
        t["excluded_non_linear_parameter_bytes"],
        t["excluded_routed_norm_bf16_bytes"] + t["excluded_correction_bias_fp32_bytes"],
        "excluded non-linear parameter bytes",
    )

    expected_traffic = {
        "router_bf16": experts * hidden * 2,
        "routed_down_bf16": latent * hidden * 2,
        "shared_w13_bf16": 2 * shared_i * hidden * 2,
        "shared_w2_bf16": hidden * shared_i * 2,
        "routed_up_bf16": hidden * latent * 2,
        "one_selected_expert_w13_payload": 2 * local_i * (latent // 2),
        "one_selected_expert_w13_scale": 2 * local_i * (latent // 32),
        "one_selected_expert_w2_payload": latent * (local_i // 2),
        "one_selected_expert_w2_scale": latent * (local_i // 32),
    }
    for name, value in expected_traffic.items():
        check.equal(t[name], value, f"traffic {name}")
    fixed_names = (
        "router_bf16",
        "routed_down_bf16",
        "shared_w13_bf16",
        "shared_w2_bf16",
        "routed_up_bf16",
    )
    check.equal(
        t["fixed_per_batch"], sum(t[name] for name in fixed_names), "fixed bytes"
    )
    expert_names = (
        "one_selected_expert_w13_payload",
        "one_selected_expert_w13_scale",
        "one_selected_expert_w2_payload",
        "one_selected_expert_w2_scale",
    )
    check.equal(
        t["one_selected_expert_total"],
        sum(t[name] for name in expert_names),
        "one expert bytes",
    )
    check.equal(
        t["rank_local_all_expert_storage"],
        experts * t["one_selected_expert_total"],
        "all expert storage",
    )
    check.close(
        t["rank_local_all_expert_storage_mib"],
        t["rank_local_all_expert_storage"] / (1 << 20),
        "all expert storage MiB",
    )
    check.equal(
        t["sixteen_selected_experts_total"],
        topk * t["one_selected_expert_total"],
        "selected expert bytes",
    )
    check.equal(
        t["b1_total"],
        t["fixed_per_batch"] + t["sixteen_selected_experts_total"],
        "B1 total bytes",
    )
    check.close(t["b1_mib"], t["b1_total"] / (1 << 20), "B1 MiB")
    check.close(
        t["b1_hbm_lower_bound_us_at_8tbps"],
        t["b1_total"] / 8e12 * 1e6,
        "8 TB/s lower bound",
    )
    check.close(
        t["b1_hbm_lower_bound_us_at_5_3tbps"],
        t["b1_total"] / 5.3e12 * 1e6,
        "5.3 TB/s lower bound",
    )

    expected_flops = {
        "router": 2 * hidden * experts,
        "routed_down": 2 * hidden * latent,
        "shared_w13": 2 * hidden * 2 * shared_i,
        "shared_w2": 2 * shared_i * hidden,
        "routed_w13_selected16": 2 * topk * latent * 2 * local_i,
        "routed_w2_selected16": 2 * topk * local_i * latent,
        "routed_up": 2 * latent * hidden,
    }
    for name, value in expected_flops.items():
        check.equal(f[name], value, f"FLOPs {name}")
    check.equal(f["total"], sum(expected_flops.values()), "total FLOPs")


def _pipeline_oracles(data: dict[str, Any], check: Checker) -> None:
    p = data["pipeline_resources"]
    w13 = p["w13"]
    w2 = p["w2"]
    dense = p["dense_bf16_n32_k512"]
    check.equal(sum(w13["k_tiles"]), 3584, "W13 exact K")
    check.equal(
        w13["full_stage_total_bytes"],
        w13["full_stage_activation_bytes"]
        + w13["full_stage_packed_weight_bytes"]
        + w13["full_stage_scale_bytes"],
        "W13 stage LDS",
    )
    check.equal(
        w13["double_buffer_bytes"], 2 * w13["full_stage_total_bytes"], "W13 double LDS"
    )
    check.equal(sum(w2["k_tiles"]), 384, "W2 exact K")
    check.equal(w2["cohorts"] * w2["subgroups_per_cohort"], 8, "W2 subgroup cohorts")
    check.equal(
        w2["cohorts"] * w2["output_channels_per_cohort"],
        w2["output_channels_per_workgroup"],
        "W2 output cohorts",
    )
    check.equal(
        w2["k256_stage_total_bytes"],
        w2["k256_activation_bytes"]
        + w2["k256_packed_weight_bytes"]
        + w2["k256_scale_bytes"],
        "W2 stage LDS",
    )
    check.equal(
        w2["full_stage_double_buffer_bytes"],
        2 * w2["k256_stage_total_bytes"],
        "W2 double LDS",
    )
    check.equal(
        dense["stage_total_bytes"],
        dense["stage_activation_bytes"] + dense["stage_weight_bytes"],
        "dense stage LDS",
    )
    check.equal(
        dense["double_buffer_bytes"], 2 * dense["stage_total_bytes"], "dense double LDS"
    )
    for name, section, key in (
        ("W13", w13, "double_buffer_bytes"),
        ("W2", w2, "full_stage_double_buffer_bytes"),
        ("dense", dense, "double_buffer_bytes"),
    ):
        check.true(
            section[key] < p["cdna4_lds_bytes_per_compute_unit"],
            f"{name} double buffer fits LDS",
        )


def _event_oracles(data: dict[str, Any], check: Checker) -> None:
    graph = data["event_graph"]
    nodes = set(graph["nodes"])
    indegree = {node: 0 for node in nodes}
    successors: dict[str, list[str]] = defaultdict(list)
    for source, target in graph["edges"]:
        check.true(
            source in nodes and target in nodes, f"event edge {source}->{target}"
        )
        successors[source].append(target)
        indegree[target] += 1
    ready = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in successors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    check.equal(visited, len(nodes), "current-epoch event DAG")
    check.equal(
        graph["next_epoch_reuse_edge"],
        ["completion_plane_epoch_n", "system_release_epoch_n_plus_1"],
        "completion-before-reuse edge",
    )
    check.equal(
        graph["allow_cache_hint_as_ordering"], False, "cache hints not ordering"
    )
    check.equal(
        graph["allow_per_route_event_without_independent_consumer"],
        False,
        "no useless per-route events",
    )
    check.equal(
        graph["remote_protocol"],
        [
            "producer_vmem_drain",
            "system_release",
            "remote_system_acquire",
            "peer_load",
            "completion_release",
            "remote_completion_acquire_before_reuse",
        ],
        "Iris protocol order",
    )


def _numeric_oracles(data: dict[str, Any], check: Checker) -> None:
    numeric = data["tiny_numeric_oracles"]
    canonical_e2m1 = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    check.equal(
        [_bf16_bits(value) for value in numeric["mxfp4_e2m1_lut"]],
        [_bf16_bits(value) for value in canonical_e2m1],
        "MXFP4 E2M1 LUT",
    )

    topk = numeric["topk_case"]
    ids = sorted(
        range(len(topk["selection_scores"])),
        key=lambda i: (-topk["selection_scores"][i], i),
    )[: topk["k"]]
    check.equal(ids, topk["expected_ids"], "stable top-k ids")
    selected = [topk["unbiased_scores"][expert] for expert in ids]
    denominator = sum(selected)
    normalized = [score / denominator for score in selected]
    for index, (actual, expected) in enumerate(
        zip(normalized, topk["expected_normalized_weights"], strict=True)
    ):
        check.close(actual, expected, f"top-k weight {index}")

    lane = numeric["joint_lane_case"]
    weights = [_float32(value) for value in lane["route_weights"]]
    rank_routed: list[list[float]] = []
    for rank_routes in lane["rank_route_outputs_bf16"]:
        width = len(rank_routes[0])
        rank_values = []
        for column in range(width):
            products = [
                _float32(weights[slot] * _bf16_float(rank_routes[slot][column]))
                for slot in range(len(weights))
            ]
            rank_values.append(_bf16_float(_fp32_sum_in_order(products)))
        rank_routed.append(rank_values)
    check.equal(
        rank_routed,
        lane["expected_rank_routed_partials_bf16"],
        "rank-local slot-order routed combine",
    )

    rank_lanes = []
    for rank, routed in enumerate(rank_routed):
        shared = [
            _bf16_float(value) for value in lane["rank_shared_partials_bf16"][rank]
        ]
        rank_lanes.append(shared + routed)
    local_rank = lane["destination_local_rank"]
    reduction_order = [local_rank] + [
        rank for rank in range(len(rank_lanes)) if rank != local_rank
    ]
    reduced = []
    for column in range(len(rank_lanes[0])):
        reduced.append(
            _bf16_float(
                _fp32_sum_in_order(
                    [rank_lanes[rank][column] for rank in reduction_order]
                )
            )
        )
    check.equal(reduced, lane["expected_reduced_shared_then_routed_bf16"], "joint lane")
    check.equal(
        [f"{_bf16_bits(value):04x}" for value in reduced],
        lane["expected_reduced_bf16_hex"],
        "joint lane BF16 bits",
    )

    rank_order = numeric["local_first_peer_ascending_reduce_case"]
    check.equal(
        len(rank_order["contributions_bf16"]),
        data["dimensions"]["tp_world"],
        "rank-order case world size",
    )
    rank_inputs = [_bf16_float(value) for value in rank_order["contributions_bf16"]]
    actual_by_local_rank = []
    actual_hex_by_local_rank = []
    for local_rank in range(data["dimensions"]["tp_world"]):
        peer_order = [
            peer_rank
            for peer_rank in range(data["dimensions"]["tp_world"])
            if peer_rank != local_rank
        ]
        recurrence = [rank_inputs[local_rank]] + [
            rank_inputs[peer_rank] for peer_rank in peer_order
        ]
        reduced_for_rank = _bf16_float(_fp32_sum_in_order(recurrence))
        actual_by_local_rank.append(reduced_for_rank)
        actual_hex_by_local_rank.append(f"{_bf16_bits(reduced_for_rank):04x}")
    check.equal(
        actual_by_local_rank,
        rank_order["expected_by_local_rank_fp32_then_bf16"],
        "local-first per-rank reduction values",
    )
    check.equal(
        actual_hex_by_local_rank,
        rank_order["expected_by_local_rank_bf16_hex"],
        "local-first per-rank reduction bits",
    )
    check.close(
        math.fsum(rank_inputs),
        rank_order["mathematical_real_sum_after_bf16_inputs"],
        "rank-order mathematical sum",
    )
    check.true(
        len(set(actual_by_local_rank)) > 1,
        "local-first case distinguishes destination ranks",
    )


def _design_guard_oracles(data: dict[str, Any], check: Checker) -> None:
    artifact = data["artifact"]
    check.equal(
        artifact["base_commit"],
        "e784229526ce11d272a3c4a0b3f64ab9a8973491",
        "refreshed base commit",
    )
    check.equal(artifact["date_utc"], "2026-08-15", "refresh date")
    check.equal(
        artifact["original_design_commit"],
        "19e30183469abc87b6823d25c8c3cdd7b02769f7",
        "original design commit",
    )
    check.equal(artifact["topology"], "TP8_EP1", "topology")
    check.equal(artifact["status"], "host_research_only", "host-only status")
    check.equal(artifact["fresh_design"], True, "fresh design")
    check.equal(artifact["ep8_schedule_reused"], False, "no EP8 schedule reuse")
    check.equal(artifact["production_code_changed"], False, "no production edit")
    check.equal(
        artifact["production_code_changed_scope"],
        "no_tp8_implementation_or_semantic_change_required_repository_formatter_edits_excluded",
        "production edit scope",
    )
    check.equal(
        artifact["required_precommit_formatter_changes_retained"],
        True,
        "required formatter changes retained",
    )
    check.equal(artifact["compiled"], False, "not compiled")
    check.equal(artifact["gpu_run"], False, "not GPU-run")
    check.equal(artifact["implementation_present"], False, "no implementation")
    check.equal(
        artifact["schedule_independence_performance_verified"],
        False,
        "schedule independence not performance-verified",
    )
    check.equal(
        artifact["production_megamoe_package_present"],
        False,
        "removed production MegaMoE package",
    )
    check.equal(
        artifact["situ_decode_layout_failure_is_architectural_evidence"],
        False,
        "layout failure conclusion",
    )
    top_main = data["top_main_audit"]
    check.equal(
        top_main["tp8_ep1_model_path_end_to_end_qualified"],
        False,
        "TP8 model path is not end-to-end qualified",
    )
    check.equal(
        top_main["tp8_ep1_routed_rank_reduction_present"],
        False,
        "TP8 routed reduction is absent",
    )
    check.equal(
        top_main["tp8_per_rank_situ_compute_control_available"],
        True,
        "TP8 per-rank compute control",
    )
    check.equal(top_main["ep8_control_batches"], [1, 8, 16], "EP8 controls")
    fixed_control = data["fixed_control_metadata_bytes"]
    check.equal(fixed_control["fatal_epoch_int64_1"], 8, "fatal epoch bytes")
    check.equal(fixed_control["topology_status_int64_2"], 16, "topology status bytes")
    check.true(
        "grid_phase_counters"
        in fixed_control["excluded_from_total_because_design_not_frozen"],
        "unfrozen grid counters excluded from total",
    )
    collectives = data["collective_candidates"]
    check.equal(
        collectives["J0"]["rank_accumulation_order"],
        "destination_local_first_then_group_rank_0_to_7_excluding_local",
        "J0 destination-local recurrence",
    )
    check.equal(
        collectives["J1"]["rank_accumulation_order"],
        "tile_owner_local_first_then_group_rank_0_to_7_excluding_owner",
        "J1 owner-local recurrence",
    )
    check.equal(
        collectives["J1"]["changes_destination_specific_j0_recurrence"],
        True,
        "J1 changes destination-specific J0 arithmetic",
    )
    check.equal(
        collectives["J1"]["requires_numerical_tolerance_acceptance"],
        True,
        "J1 numerical qualification required",
    )
    ledger = data["research_ledger"]
    check.equal(
        set(ledger),
        {"tokenspeed", "triton", "iris", "megakernels", "comms_notes", "cdna4"},
        "required research trees",
    )
    for name in ("triton", "iris", "megakernels", "comms_notes", "cdna4"):
        check.true(
            ledger[name]["root"].startswith("/home/ericfeng/distributed/"),
            f"{name} root",
        )
        check.true(bool(ledger[name]["sources"]), f"{name} sources")
        check.true(bool(ledger[name]["transfer"]), f"{name} transfer")
        check.true(bool(ledger[name]["reject"]), f"{name} reject")
    compute = data["batch_specific_compute_decision"]
    check.equal(compute["logical_k384_alone_justifies_mfma"], False, "K384 MFMA guard")
    check.true("vector_gemv" in compute["B1_routed"], "B1 vector regime")
    check.true("mfma" in compute["B8_B16_dense_replicated"], "B8/B16 dense MFMA")
    check.true("vector_gemv" in compute["B8_B16_direct_routed"], "direct routed vector")
    check.equal(
        compute["top_main_tp8_per_rank_control"],
        "portable_triton_mxfp4_precomputed_situ",
        "top-main per-rank control",
    )
    baseline = data["ordinary_b1_isa_baseline"]
    check.equal(
        baseline["source_commit"],
        artifact["historical_control_commit"],
        "historical ISA commit",
    )
    check.equal(
        baseline["present_on_refreshed_base"], False, "historical ISA not current"
    )
    check.equal(baseline["mfma_instructions"], 0, "ordinary B1 no MFMA")
    check.equal(baseline["s_clause_instructions"], 0, "ordinary B1 no s_clause")
    check.equal(baseline["static_instruction_lines"], 13653, "ordinary B1 ISA lines")
    check.equal(baseline["static_waitcnt_lines"], 1347, "ordinary B1 waitcnt lines")
    check.equal(baseline["static_barrier_lines"], 332, "ordinary B1 barrier lines")
    check.equal(baseline["static_global_load_instructions"], 527, "ordinary B1 loads")
    check.equal(
        baseline["conservative_global_load_issue_groups"],
        514,
        "ordinary B1 load groups",
    )
    check.equal(
        baseline["single_load_issue_groups"], 506, "ordinary B1 singleton groups"
    )
    check.equal(baseline["maximum_loads_per_issue_group"], 4, "ordinary B1 max group")
    check.equal(baseline["rank0_elf_text_bytes"], 80384, "ordinary B1 ELF text")
    check.equal(baseline["rank0_hsaco_bytes"], 135840, "ordinary B1 HSACO")
    check.equal(baseline["counts_are_dynamic"], False, "ordinary counts are static")
    gate = data["cu_simd_evidence_gate"]
    check.true(
        "nominal_occupancy" in gate["metadata_is_non_promoting"],
        "occupancy non-promoting",
    )
    check.equal(
        gate["resident_waves_per_active_compute_unit_expected"], 8, "resident waves/CU"
    )
    check.equal(gate["resident_waves_per_simd_expected"], 2, "resident waves/SIMD")
    check.equal(gate["att_timing_is_performance_evidence"], False, "ATT timing guard")
    check.equal(
        gate["att_sampling_rule"],
        "slowest_rank_early_middle_tail_tasks_and_at_least_one_sampled_workgroup_per_xcd",
        "ATT representative sampling rule",
    )
    check.equal(
        gate["allow_fast_xcd_only_att_evidence"],
        False,
        "reject fast-XCD-only ATT",
    )
    required = set(gate["final_promotion_requires"])
    check.equal(
        required,
        {
            "numerical_gate",
            "uninstrumented_e2e_win_at_any_target_batch",
            "all_rank_normal_timeline",
            "targeted_att_issue_stall_balance_evidence",
            "exact_isa_hazard_and_code_footprint_review",
        },
        "final evidence set",
    )
    advance_after = data["smallest_post_refresh_discriminator"]["advance_after"]
    check.true(
        "corrected_tp8_routed_reduction_before_any_serving_baseline_claim"
        in advance_after,
        "correct reduction precedes serving baseline",
    )


def _serving_acceptance_oracles(data: dict[str, Any], check: Checker) -> None:
    serving = data["serving_correctness_acceptance"]
    workload = serving["workload"]
    check.equal(workload["prompt_tokens_per_request"], 4096, "serving prompt tokens")
    check.equal(
        workload["completion_tokens_per_request"], 1024, "serving completion tokens"
    )
    check.equal(
        workload["positive_megamoe_loaded_code_and_dispatch_proof_required"],
        True,
        "positive MegaMoE dispatch proof",
    )
    check.equal(
        workload["fallback_result_can_qualify"], False, "fallback cannot qualify"
    )
    b1 = serving["B1"]
    check.equal(b1["expected_successful_requests"], 1, "B1 successful requests")
    check.equal(
        b1["baseline_candidate_prompt_sha256_exact_equal"],
        True,
        "B1 prompt identity",
    )
    check.equal(
        b1["baseline_candidate_normalized_completion_sha256_exact_equal"],
        True,
        "B1 completion identity",
    )
    for batch in (8, 16):
        contract = serving[f"B{batch}"]
        check.equal(
            contract["expected_successful_requests"],
            batch,
            f"B{batch} successful requests",
        )
        check.equal(
            contract["identical_prompt_multiset_between_arms"],
            True,
            f"B{batch} prompt multiset",
        )
        check.equal(
            contract["retain_every_prompt_and_completion_hash"],
            True,
            f"B{batch} retain hashes",
        )
        check.equal(
            contract["completion_hash_equality_required"],
            False,
            f"B{batch} completion hash non-equality policy",
        )
        check.equal(
            contract["separate_bitwise_m_specific_boundary_oracle_required"],
            True,
            f"B{batch} bitwise M-specific oracle",
        )
    check.equal(
        serving["admission_scope"],
        "win_and_correctness_at_one_batch_admit_only_that_exact_batch_specialization",
        "batch-local admission scope",
    )


def verify(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    check = Checker()
    check.equal(data["schema_version"], 1, "schema version")
    _design_guard_oracles(data, check)
    _shape_oracles(data, check)
    _batch_oracles(data, check)
    _traffic_and_flop_oracles(data, check)
    _pipeline_oracles(data, check)
    _event_oracles(data, check)
    _numeric_oracles(data, check)
    _serving_acceptance_oracles(data, check)
    return {
        "result": "PASS",
        "checks": check.count,
        "oracle": str(path),
        "base_commit": data["artifact"]["base_commit"],
        "topology": data["artifact"]["topology"],
        "batches": data["dimensions"]["target_batches"],
        "production_code_changed": data["artifact"]["production_code_changed"],
        "compiled": data["artifact"]["compiled"],
        "gpu_run": data["artifact"]["gpu_run"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=_DEFAULT_ORACLE)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = verify(args.oracle.resolve())
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
