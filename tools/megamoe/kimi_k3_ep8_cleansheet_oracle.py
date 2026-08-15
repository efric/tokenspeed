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

"""Host-only structural oracle for a clean-sheet Kimi K3 EP8 MoE core.

The oracle deliberately imports neither Torch nor a ROCm package.  It checks
route ownership, stable expert grouping, deterministic combine order, bounded
workspace formulas, task counts, and byte/flop models for B1, B8, and B16.  It
does not predict GPU time; measured service times and hardware traces remain
the performance authority.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA_VERSION = 2
REFRESHED_BASE_COMMIT = "e784229526ce11d272a3c4a0b3f64ab9a8973491"
ORIGINAL_DESIGN_COMMIT = "8129173ab75ce473a04e495f1a55db14af5f0a80"
HISTORICAL_CONTROL_COMMIT = "be7d5001927c9613314f300a420fb88c4bdd4628"
REFRESH_DATE_UTC = "2026-08-15"
SUPPORTED_BATCHES = (1, 8, 16)
WORLD_SIZE = 8
HIDDEN = 7168
LATENT = 3584
EXPERTS = 896
LOCAL_EXPERTS = EXPERTS // WORLD_SIZE
TOP_K = 16
INTERMEDIATE = 3072
SHARED = 768
MXFP4_GROUP = 32
BF16_BYTES = 2

BF16_WEIGHT_SHAPES = {
    "router_weight": (EXPERTS, HIDDEN),
    "routed_down_weight": (LATENT, HIDDEN),
    "shared_gate_up_weight": (2 * SHARED, HIDDEN),
    "shared_down_weight": (HIDDEN, SHARED),
    "routed_norm_weight": (LATENT,),
    "routed_up_weight": (HIDDEN, LATENT),
}

EXPERT_CHECKPOINT_LAYOUTS = {
    "w13_weight": {
        "shape": (LOCAL_EXPERTS, 2 * INTERMEDIATE, LATENT // 2),
        "strides": (11010048, 1792, 1),
        "dtype": "uint8",
    },
    "w13_weight_scale": {
        "shape": (LOCAL_EXPERTS, 2 * INTERMEDIATE, LATENT // MXFP4_GROUP),
        "strides": (688128, 112, 1),
        "dtype": "uint8",
    },
    "w2_weight": {
        "shape": (LOCAL_EXPERTS, LATENT, INTERMEDIATE // 2),
        "strides": (5505024, 1536, 1),
        "dtype": "uint8",
    },
    "w2_weight_scale": {
        "shape": (LOCAL_EXPERTS, LATENT, INTERMEDIATE // MXFP4_GROUP),
        "strides": (344064, 96, 1),
        "dtype": "uint8",
    },
}

W13_WEIGHT_BYTES_PER_EXPERT = 2 * INTERMEDIATE * (LATENT // 2) + 2 * INTERMEDIATE * (
    LATENT // MXFP4_GROUP
)
W2_WEIGHT_BYTES_PER_EXPERT = LATENT * (INTERMEDIATE // 2) + LATENT * (
    INTERMEDIATE // MXFP4_GROUP
)
EXPERT_WEIGHT_BYTES = W13_WEIGHT_BYTES_PER_EXPERT + W2_WEIGHT_BYTES_PER_EXPERT


@dataclass(frozen=True, slots=True)
class Route:
    """One token-major, top-k-slot-major route."""

    ordinal: int
    token: int
    slot: int
    expert: int
    owner_rank: int
    local_expert: int


def ceil_div(value: int, divisor: int) -> int:
    """Return an exact nonnegative ceiling division."""

    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


def _route(token: int, slot: int, expert: int) -> Route:
    if not 0 <= expert < EXPERTS:
        raise ValueError(f"expert {expert} is outside [0, {EXPERTS})")
    return Route(
        ordinal=token * TOP_K + slot,
        token=token,
        slot=slot,
        expert=expert,
        owner_rank=expert // LOCAL_EXPERTS,
        local_expert=expert % LOCAL_EXPERTS,
    )


def make_routes(batch: int, scenario: str, seed: int = 20260814) -> tuple[Route, ...]:
    """Construct deterministic legal top-k IDs for one stress scenario."""

    if batch not in SUPPORTED_BATCHES:
        raise ValueError(f"batch must be one of {SUPPORTED_BATCHES}")
    if scenario not in {"balanced_unique", "hot_balanced", "rank_skew", "random"}:
        raise ValueError(f"unknown route scenario {scenario!r}")

    rng = random.Random(seed)
    routes: list[Route] = []
    for token in range(batch):
        if scenario == "random":
            experts = rng.sample(range(EXPERTS), TOP_K)
        elif scenario == "rank_skew":
            # A token cannot select the same expert twice.  Reuse the same 16
            # rank-zero experts across tokens to exercise both rank and hotness
            # skew without constructing an illegal top-k row.
            experts = list(range(TOP_K))
        else:
            # Two experts per owner rank.  balanced_unique rotates local IDs
            # between tokens; hot_balanced maximizes cross-token reuse.
            experts = []
            for slot in range(TOP_K):
                rank = slot % WORLD_SIZE
                local = slot // WORLD_SIZE
                if scenario == "balanced_unique":
                    local = (2 * token + local) % LOCAL_EXPERTS
                experts.append(rank * LOCAL_EXPERTS + local)
        if len(set(experts)) != TOP_K:
            raise AssertionError("a top-k row contains duplicate experts")
        routes.extend(
            _route(token, slot, expert) for slot, expert in enumerate(experts)
        )
    return tuple(routes)


def rank_routes(routes: Iterable[Route], rank: int) -> tuple[Route, ...]:
    """Return routes owned by ``rank`` in deterministic combine order."""

    if not 0 <= rank < WORLD_SIZE:
        raise ValueError("invalid rank")
    return tuple(route for route in routes if route.owner_rank == rank)


def stable_expert_schedule(routes: Iterable[Route], rank: int) -> tuple[Route, ...]:
    """Group one rank's routes by local expert, stable within each expert."""

    local = rank_routes(routes, rank)
    return tuple(sorted(local, key=lambda route: (route.local_expert, route.ordinal)))


def expert_offsets(
    schedule: tuple[Route, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return 112 expert counts and the matching 113-entry exclusive prefix."""

    counts = [0] * LOCAL_EXPERTS
    for route in schedule:
        counts[route.local_expert] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return tuple(counts), tuple(offsets)


def combine_slots(routes: Iterable[Route], rank: int, token: int) -> tuple[int, ...]:
    """Return rank-local top-k slots in the required ascending accumulation order."""

    return tuple(
        route.slot for route in rank_routes(routes, rank) if route.token == token
    )


def bucketed_rows(counts: Iterable[int], maximum_bucket: int = 16) -> int:
    """Count rows for a 1/2/4/... exact-tail microtile policy."""

    rows = 0
    for count in counts:
        if count <= 0:
            continue
        bucket = 1
        while bucket < count and bucket < maximum_bucket:
            bucket *= 2
        rows += bucket if bucket >= count else count
    return rows


def _dense_weight_bytes() -> dict[str, int]:
    result = {
        name: BF16_BYTES * _product(shape) for name, shape in BF16_WEIGHT_SHAPES.items()
    }
    result["total"] = sum(result.values())
    return result


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _workspace_model(batch: int) -> dict[str, Any]:
    capacity = batch * TOP_K
    communication_tiles = batch * (ceil_div(LATENT, 512) + ceil_div(HIDDEN, 512))
    local = {
        "router_logits_f32": batch * EXPERTS * 4,
        "routed_input_bf16": batch * LATENT * BF16_BYTES,
        "shared_input_bf16": batch * SHARED * BF16_BYTES,
        "topk_ids_i32": capacity * 4,
        "topk_weights_f32": capacity * 4,
        "packed_route_records_u64": capacity * 8,
        "expert_counts_i32": LOCAL_EXPERTS * 4,
        "expert_offsets_i32": (LOCAL_EXPERTS + 1) * 4,
        "w13_intermediate_bf16": capacity * INTERMEDIATE * BF16_BYTES,
        "w2_route_output_bf16_batched_only": (
            0 if batch == 1 else capacity * LATENT * BF16_BYTES
        ),
        "reduced_joint_bf16": batch * (LATENT + HIDDEN) * BF16_BYTES,
    }
    # One producer payload is protected by distinct ready and completion
    # planes.  Each plane holds one monotonic 64-bit generation per tile/rank.
    epoch_plane = communication_tiles * WORLD_SIZE * 8
    symmetric = {
        "joint_producer_bf16": batch * (LATENT + HIDDEN) * BF16_BYTES,
        "ready_epoch_flags_u64": epoch_plane,
        "completion_epoch_flags_u64": epoch_plane,
        "fatal_epoch_u64": 8,
    }
    return {
        "local_bytes": local,
        "local_total_bytes": sum(local.values()),
        "symmetric_bytes": symmetric,
        "symmetric_total_bytes": sum(symmetric.values()),
        "communication_tiles": communication_tiles,
    }


def _task_census(
    batch: int, local_counts: tuple[int, ...], route_count: int
) -> dict[str, Any]:
    active_counts = tuple(count for count in local_counts if count)
    grouped_chunks_m4 = sum(ceil_div(count, 4) for count in active_counts)
    common = {
        "phase0_n32": batch
        * (ceil_div(EXPERTS, 32) + ceil_div(LATENT, 32) + ceil_div(2 * SHARED, 32)),
        "shared_down_n32": batch * ceil_div(HIDDEN, 32),
        "communication_512": batch * (ceil_div(LATENT, 512) + ceil_div(HIDDEN, 512)),
        "final_up_n32": batch * ceil_div(HIDDEN, 32),
    }
    if batch == 1:
        expert = {
            "w13_route_n16": route_count * ceil_div(INTERMEDIATE, 16),
            # Each task owns one output tile and consumes all local routes in
            # ascending slot order.  Work is task count times route_count.
            "w2_output_centric_n32": ceil_div(LATENT, 32),
            "w2_route_tile_iterations": route_count * ceil_div(LATENT, 32),
            "route_acquires_by_96_consumers": 96 * route_count,
        }
    else:
        expert = {
            "w13_exact_ragged_m4_n64": grouped_chunks_m4 * ceil_div(INTERMEDIATE, 64),
            "w2_exact_ragged_m4_n64": grouped_chunks_m4 * ceil_div(LATENT, 64),
            "deterministic_combine_512": batch * ceil_div(LATENT, 512),
        }
    return {**common, **expert}


def _rank_model(batch: int, routes: tuple[Route, ...], rank: int) -> dict[str, Any]:
    local = rank_routes(routes, rank)
    schedule = stable_expert_schedule(routes, rank)
    counts, offsets = expert_offsets(schedule)
    active = tuple(count for count in counts if count)
    unique = len(active)
    route_count = len(local)
    return {
        "rank": rank,
        "route_count": route_count,
        "unique_local_experts": unique,
        "max_routes_per_expert": max(active, default=0),
        "expert_counts": list(counts),
        "expert_offsets": list(offsets),
        "combine_slots_by_token": [
            list(combine_slots(routes, rank, token)) for token in range(batch)
        ],
        "schedule_ordinals": [route.ordinal for route in schedule],
        "weight_traffic_bounds_bytes": {
            "ideal_unique_expert_floor": unique * EXPERT_WEIGHT_BYTES,
            "cold_per_route_ceiling": route_count * EXPERT_WEIGHT_BYTES,
        },
        "route_storage_live_bytes": {
            "w13_intermediate": route_count * INTERMEDIATE * BF16_BYTES,
            "w2_output_batched_only": (
                0 if batch == 1 else route_count * LATENT * BF16_BYTES
            ),
        },
        "row_work_models": {
            "native_route_direct": route_count,
            "grouped_bm64_counterfactual": unique * 64,
            "stable_bucketed_1_2_4_8_16": bucketed_rows(active),
        },
        "tasks": _task_census(batch, counts, route_count),
    }


def _b1_dag() -> dict[str, Any]:
    return {
        "granularity": "route for W13; latent-512/shared-512 for communication",
        "nodes": [
            {
                "id": "K0.project",
                "depends_on": ["hidden"],
                "publishes": ["router", "routed", "shared_in"],
            },
            {"id": "K0.topk", "depends_on": ["router"], "publishes": ["topk"]},
            {
                "id": "K0.compact",
                "depends_on": ["topk"],
                "publishes": ["route_schedule"],
            },
            {
                "id": "K1.w13[route]",
                "depends_on": ["routed", "route_schedule"],
                "publishes": ["w13_epoch[route]"],
            },
            {
                "id": "K1.shared_down[tile]",
                "depends_on": ["shared_in"],
                "publishes": ["shared_partial_epoch[tile]"],
            },
            {
                "id": "K1.w2_output[tile]",
                "depends_on": ["route_schedule", "w13_epoch[route] in slot order"],
                "publishes": ["routed_partial_epoch[tile]"],
            },
            {
                "id": "K1.reduce[kind,tile]",
                "depends_on": ["local_partial_epoch[kind,tile]", "peer_ready_epochs"],
                "publishes": ["reduced_epoch[kind,tile]", "peer_completion_epoch"],
            },
            {
                "id": "K2.rms[token]",
                "depends_on": ["all routed reduced tiles"],
                "publishes": ["rms_epoch[token]"],
            },
            {
                "id": "K2.up_add[tile]",
                "depends_on": ["rms_epoch[token]", "shared reduced tile", "prefix"],
                "publishes": ["output tile"],
            },
        ],
    }


def _batched_dag() -> dict[str, Any]:
    return {
        "granularity": "stable expert microbatch; token-by-512 output events",
        "nodes": [
            {
                "id": "K0.project[token]",
                "depends_on": ["hidden[token]"],
                "publishes": ["router[token]", "routed[token]", "shared_in[token]"],
            },
            {
                "id": "K0.topk[token]",
                "depends_on": ["router[token]"],
                "publishes": ["topk[token]"],
            },
            {
                "id": "K0.count_prefix_stable_scatter",
                "depends_on": ["all topk rows"],
                "publishes": ["expert_offsets", "stable_routes"],
            },
            {
                "id": "K1.w13[expert,microbatch,n]",
                "depends_on": ["stable_routes", "routed rows"],
                "publishes": ["w13_group_epoch"],
            },
            {
                "id": "K1.w2[expert,microbatch,n]",
                "depends_on": ["w13_group_epoch"],
                "publishes": ["route_output_epoch[token,slot,n512]"],
            },
            {
                "id": "K1.combine[token,n512]",
                "depends_on": ["local route output epochs in ascending slot order"],
                "publishes": ["routed_partial_epoch[token,n512]"],
            },
            {
                "id": "K1.shared_down[token,n512]",
                "depends_on": ["shared_in[token]"],
                "publishes": ["shared_partial_epoch[token,n512]"],
            },
            {
                "id": "K1.reduce[token,kind,n512]",
                "depends_on": ["local partial epoch", "peer ready epochs"],
                "publishes": ["reduced epoch", "peer completion epoch"],
            },
            {
                "id": "K2.rms[token]",
                "depends_on": ["all routed reduced tiles for token"],
                "publishes": ["rms_epoch[token]"],
            },
            {
                "id": "K2.up_add[token,n]",
                "depends_on": [
                    "rms_epoch[token]",
                    "shared reduced tile",
                    "prefix[token]",
                ],
                "publishes": ["output tile"],
            },
        ],
    }


def _architecture() -> dict[str, Any]:
    return {
        "recommended_launch_graph": [
            {
                "kernel": "K0_projection_topk_schedule",
                "reason": "phase-specific dense resources; produces graph-stable local schedule",
            },
            {
                "kernel": "K1_resident_event_moe_and_joint_reduce",
                "reason": "keeps the only profitable compute/communication overlap inside one resident grid",
            },
            {
                "kernel": "K2_rmsnorm_up_add",
                "reason": "phase-specific final GEMV resources and no global in-kernel tail barrier",
            },
        ],
        "k1_candidate": {
            "programs": 256,
            "subgroups_per_workgroup": 8,
            "logical_cohorts": {
                "w13_producer": 128,
                "w2_consumer_and_shared_flex": 96,
                "communication": 32,
            },
            "role_mapping_ab_test": [
                "pid ranges (no XCD role affinity control)",
                "per-XCD local ranks 16/12/4 after a proved dense P256 launch",
            ],
            "physical_placement_gate": {
                "expected_compute_units": 256,
                "expected_workgroups_per_compute_unit": 1,
                "expected_workgroups_per_xcd": 32,
                "report_fields": [
                    "rank",
                    "xcd",
                    "compute_unit",
                    "workgroup",
                    "cohort",
                    "first_task_epoch",
                    "last_task_epoch",
                ],
                "reject_if": [
                    "any compute unit is missing or duplicated",
                    "the launch tail crosses the first polling epoch",
                ],
            },
            "warning": "counts are discriminator starting points, not admitted constants",
        },
        "measured_controls": {
            "native_ep8_route_direct": {
                "organization": "route_direct_warp_gemv",
                "target_batches": list(SUPPORTED_BATCHES),
                "maximum_tokens": 16,
                "sorts_routes": False,
                "pads_routes_to_bm64": False,
                "sources": [
                    (
                        "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/"
                        "gluon/mxfp4.py:34-35,99-203"
                    ),
                    (
                        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/"
                        "gfx950/moe/mxfp4/situ_decode.py:21-36,71-438"
                    ),
                ],
            },
            "native_b1_joint_expert_shared": {
                "organization": "routed_w2_plus_shared_down_then_producer_direct_iris",
                "lane_order": ["shared_hidden", "routed_latent"],
                "sources": [
                    "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/latent_decode.py:21-77",
                    (
                        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/"
                        "gfx950/moe/mxfp4/latent_shared_decode.py:13-53"
                    ),
                    "python/tokenspeed/runtime/layers/moe/latent.py:187-232",
                    "python/tokenspeed/runtime/models/kimi_k3.py:1454-1500",
                ],
            },
            "native_mxfp4_fused_family": {
                "role": "batched MFMA, pipeline, layout, and addressing component control",
                "activation_note": (
                    "the generic gfx950 Gluon registry is not a current K3 SiTU EP8 "
                    "serving path; transfer only separately qualified mechanisms"
                ),
                "sources": [
                    (
                        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/"
                        "gfx950/moe/mxfp4/fused/moe.py"
                    ),
                    (
                        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/"
                        "gfx950/moe/mxfp4/fused/pipelined_kernel.py"
                    ),
                    (
                        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/"
                        "gfx950/moe/mxfp4/weight_preprocess.py:161-294,308-423"
                    ),
                ],
            },
            "historical_ordinary_b1_isa": {
                "source_commit": HISTORICAL_CONTROL_COMMIT,
                "present_on_refreshed_base": False,
                "text_bytes": 80384,
                "hsaco_bytes": 135840,
                "static_instruction_lines": 13653,
                "waitcnt_instructions": 1347,
                "barrier_instructions": 332,
                "s_clause_instructions": 0,
                "global_load_instructions": 527,
                "conservative_load_issue_groups": 514,
                "single_load_issue_groups": 506,
                "maximum_loads_per_issue_group": 4,
            },
            "fully_bounded_kernels": (
                "timed control for instruction-fetch pressure and resident-grid liveness"
            ),
        },
        "communication_contract": {
            "address": "peer_base + (local_pointer - local_base)",
            "peer_protocol": [
                "observe local and system-acquire remote consumer completions at exact generation g-1",
                "device-acquire the local partial event and store the single payload",
                "system-release producer ready at exact generation g",
                "apply the selected subgroup publication choice to every remote producer ready g",
                "accumulate local first then remote producer ids in increasing order",
                "after all subgroups finish, system-release one consumer completion g",
                "do not overwrite until local and every remote consumer completion g is known",
            ],
            "local_protocol": (
                "device release by the last producer and an acquire by every consumer"
            ),
            "peer_load_publication_choices": [
                {
                    "name": "per_subgroup_system_acquire",
                    "contract": (
                        "every subgroup that issues peer VMEM first system-acquires "
                        "that producer's exact ready generation"
                    ),
                },
                {
                    "name": "leader_to_versioned_lds",
                    "contract": (
                        "one subgroup system-acquires and loads the full peer tile "
                        "into fresh/versioned LDS; followers read only after a "
                        "workgroup barrier"
                    ),
                },
            ],
            "subgroup_stale_payload_litmus": [
                "each follower subgroup caches a distinct old peer slice",
                "producer writes an alternating generation pattern then releases ready",
                "direct loaders each acquire before rereading peer VMEM",
                "staged followers read only the leader-filled versioned LDS after barrier",
                "delay ranks and cover every XCD placement for many generations",
                "reject mixed generations or early completion",
            ],
            "cache_policy_ab_tests": [
                "qualified cv peer loads versus alternatives",
                "ordinary local weights versus bare non-temporal weights",
                "global queues versus phase-specific XCD-local traversal",
            ],
            "rejected_assumptions": [
                "quiet is required for direct mapped peer loads",
                "a cache modifier publishes a payload",
                "one consumer acquire repairs every compute unit cache view",
                "one subgroup acquire repairs sibling subgroup peer-cache views",
                "a leader acquire may be followed by unqualified sibling peer VMEM",
                "a readiness flag alone makes a symmetric slot safe to reuse",
                "the workspace contains alternate payload slots",
                "Fleet device-scope NT weight loads retain XCD-local L2 reuse",
            ],
        },
        "serving_correctness_gate": {
            "all_batches": [
                "bitwise M-specific kernel/model oracle at every BF16/FP32 boundary",
                "exact success and token counts",
                "positive loaded-code and MegaMoE dispatch proof",
                "zero fatal and diagnostic state",
            ],
            "batch_1": "identical prompt and completion hashes across repetitions",
            "batch_8_16": (
                "identical prompt multiset; retain every completion hash but do not "
                "require cross-run byte equality because the control is nondeterministic"
            ),
        },
        "rejected_by_default": {
            "monolithic_all_phases": "inherits the union register/LDS/I-cache envelope that already made P240 occupancy-one and slow",
            "concurrent_compute_plus_communication_kernels": "can starve a polling communication grid without a qualified CU partition mechanism",
            "old_dual_role_route_gates": "226*(L-1) acquires and zero modeled makespan gain for L=1..8",
        },
    }


def build_oracle(batch: int, scenario: str, seed: int = 20260814) -> dict[str, Any]:
    """Build one JSON-serializable structural oracle."""

    routes = make_routes(batch, scenario, seed)
    ranks = [_rank_model(batch, routes, rank) for rank in range(WORLD_SIZE)]
    counts = [rank["route_count"] for rank in ranks]
    return {
        "batch": batch,
        "scenario": scenario,
        "seed": seed if scenario == "random" else None,
        "global_route_count": len(routes),
        "rank_route_counts": counts,
        "slowest_rank_route_count": max(counts),
        "ranks": ranks,
        "workspace": _workspace_model(batch),
        "network_payload_bytes": {
            "joint_payload_per_rank": batch * (LATENT + HIDDEN) * BF16_BYTES,
            "remote_reads_per_rank": (WORLD_SIZE - 1)
            * batch
            * (LATENT + HIDDEN)
            * BF16_BYTES,
        },
        "dag": _b1_dag() if batch == 1 else _batched_dag(),
    }


def model_contract() -> dict[str, Any]:
    """Return exact model, arithmetic-boundary, and checkpoint facts."""

    dense_flops = {
        "router_per_token": 2 * HIDDEN * EXPERTS,
        "routed_down_per_token": 2 * HIDDEN * LATENT,
        "shared_gate_up_per_token": 2 * HIDDEN * (2 * SHARED),
        "shared_down_per_token": 2 * SHARED * HIDDEN,
        "routed_up_per_token": 2 * LATENT * HIDDEN,
        "w13_per_route": 2 * LATENT * (2 * INTERMEDIATE),
        "w2_per_route": 2 * INTERMEDIATE * LATENT,
    }
    return {
        "dimensions": {
            "hidden": HIDDEN,
            "latent": LATENT,
            "experts": EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "top_k": TOP_K,
            "intermediate": INTERMEDIATE,
            "shared": SHARED,
            "world_size": WORLD_SIZE,
        },
        "bf16_weight_shapes": {
            name: list(shape) for name, shape in BF16_WEIGHT_SHAPES.items()
        },
        "expert_checkpoint_layouts": EXPERT_CHECKPOINT_LAYOUTS,
        "weight_bytes": {
            "dense_per_rank": _dense_weight_bytes(),
            "w13_per_expert": W13_WEIGHT_BYTES_PER_EXPERT,
            "w2_per_expert": W2_WEIGHT_BYTES_PER_EXPERT,
            "routed_expert_total": EXPERT_WEIGHT_BYTES,
        },
        "flops": dense_flops,
        "equations": {
            "router": "logits[t,e] = fp32_dot(bf16(hidden[t]), bf16(router[e]))",
            "selection": "ids = top16(sigmoid(logits)+bias); ties prefer lower expert id",
            "route_weight": "sigmoid(logits[ids]) / fp32_sum(sigmoid(logits[ids]))",
            "situ": "bf16((4*tanh(bf16(W13_gate*x)/4)*sigmoid(bf16(W13_gate*x))) * (25*tanh(bf16(W13_up*x)/25)))",
            "local_routed": "bf16(sum_topk_slot_order(fp32(bf16(W2*intermediate))*fp32(route_weight)))",
            "joint_reduce": "fp32 rank accumulation in the qualified Iris local-then-peer-id order, stored bf16",
            "final": "prefix + shared_reduced + W_up * RMSNorm(routed_reduced, eps=1e-5)",
        },
        "numerical_boundaries": [
            "router accumulation and top-k selection/normalization are FP32",
            "W13 projection output is rounded to BF16 before FP32 SiTU",
            "SiTU output is rounded to BF16 before W2",
            "each W2 route is rounded to BF16 before FP32 route weighting",
            "rank-local routes accumulate deterministically in ascending top-k slot order",
            "joint routed/shared reduction retains qualified Iris rank order and BF16 output",
        ],
    }


def artifact_contract() -> dict[str, Any]:
    """Return provenance and mutation-scope facts for this refreshed artifact."""

    return {
        "name": "kimi-k3-ep8-megamoe-cleansheet",
        "base_commit": REFRESHED_BASE_COMMIT,
        "refresh_date_utc": REFRESH_DATE_UTC,
        "original_design_commit": ORIGINAL_DESIGN_COMMIT,
        "historical_control_commit": HISTORICAL_CONTROL_COMMIT,
        "host_only": True,
        "production_code_changed": False,
        "production_megamoe_package_present": False,
    }


def self_check() -> None:
    """Fail if any structural or byte invariant is internally inconsistent."""

    if W13_WEIGHT_BYTES_PER_EXPERT != 11698176:
        raise AssertionError("W13 byte formula drifted from the linear EP8 storage ABI")
    if W2_WEIGHT_BYTES_PER_EXPERT != 5849088:
        raise AssertionError("W2 byte formula drifted from the linear EP8 storage ABI")
    if _dense_weight_bytes()["total"] != 148642816:
        raise AssertionError("dense weight byte model drifted")

    for batch in SUPPORTED_BATCHES:
        for scenario in ("balanced_unique", "hot_balanced", "rank_skew", "random"):
            routes = make_routes(batch, scenario)
            if len(routes) != batch * TOP_K:
                raise AssertionError("global route count mismatch")
            owners: list[int] = []
            for rank in range(WORLD_SIZE):
                local = rank_routes(routes, rank)
                schedule = stable_expert_schedule(routes, rank)
                counts, offsets = expert_offsets(schedule)
                owners.extend(route.ordinal for route in local)
                if sorted(route.ordinal for route in schedule) != sorted(
                    route.ordinal for route in local
                ):
                    raise AssertionError("stable schedule is not a route permutation")
                if offsets[-1] != len(local) or sum(counts) != len(local):
                    raise AssertionError("expert count/prefix mismatch")
                for expert in range(LOCAL_EXPERTS):
                    ordinals = [
                        route.ordinal
                        for route in schedule
                        if route.local_expert == expert
                    ]
                    if ordinals != sorted(ordinals):
                        raise AssertionError("within-expert stability was lost")
                for token in range(batch):
                    slots = combine_slots(routes, rank, token)
                    if slots != tuple(sorted(slots)):
                        raise AssertionError("combine order is not top-k-slot order")
            if sorted(owners) != list(range(batch * TOP_K)):
                raise AssertionError("a route is missing or multiply owned")
    b16_workspace = _workspace_model(16)
    if b16_workspace["symmetric_bytes"]["joint_producer_bf16"] != 344064:
        raise AssertionError("B16 symmetric payload size drifted")
    if b16_workspace["symmetric_bytes"]["ready_epoch_flags_u64"] != 21504:
        raise AssertionError("B16 ready-plane size drifted")
    if b16_workspace["symmetric_bytes"]["completion_epoch_flags_u64"] != 21504:
        raise AssertionError("B16 completion-plane size drifted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=int,
        action="append",
        choices=SUPPORTED_BATCHES,
        help="batch to emit; repeatable, defaults to 1, 8, and 16",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("balanced_unique", "hot_balanced", "rank_skew", "random"),
        help="routing case to emit; repeatable, defaults to every case",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="run invariants before emitting JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check:
        self_check()
    batches = tuple(dict.fromkeys(args.batch or SUPPORTED_BATCHES))
    scenarios = tuple(
        dict.fromkeys(
            args.scenario or ("balanced_unique", "hot_balanced", "rank_skew", "random")
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact_contract(),
        "host_only": True,
        "self_check": "passed" if args.check else "not_requested",
        "model_contract": model_contract(),
        "architecture": _architecture(),
        "cases": [
            build_oracle(batch, scenario, args.seed)
            for batch in batches
            for scenario in scenarios
        ],
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
