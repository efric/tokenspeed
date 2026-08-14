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

import random

import pytest
from tokenspeed_kernel_amd.ops.gfx950.moe.megamoe.schedule import (
    COMM_WORKERS,
    PROGRAMS,
    SHARED_COMM_WORKERS,
    WORKERS_PER_XCD,
    balanced_route_xcd_interval,
    communication_indices,
    expert_group_coordinates,
    expert_population,
    expert_routes_for_xcd,
    phase0_tasks,
    shared_down_tasks,
    topology_ticket_contributions,
    topology_ticket_gate_releases,
    w2_tasks,
    w13_tasks,
)


def _topology_capacity_valid(populations: list[int]) -> bool:
    """Mirror the static shared-worker quotas admitted by the raw kernel."""

    return (
        len(populations) == 8
        and sum(populations) == PROGRAMS
        and all(population >= 3 for population in populations[:6])
        and all(population >= 2 for population in populations[6:])
    )


def test_amd_phase0_and_shared_schedule_exact_coverage() -> None:
    phase0 = [task for pid in range(PROGRAMS) for task in phase0_tasks(pid)]
    assert sorted(phase0) == list(range(640))
    assert len(phase0) == len(set(phase0))

    shared = [
        stripe * 64 + task
        for pid in range(PROGRAMS)
        for stripe, task in shared_down_tasks(pid)
    ]
    assert sorted(shared) == list(range(14 * 64))
    assert len(shared) == len(set(shared))


def test_amd_phase0_schedule_uses_fixed_balanced_pairs() -> None:
    """Cover the exact routed/shared/router workgroup ownership map."""

    assert phase0_tasks(0) == (224, 336)
    assert phase0_tasks(111) == (335, 447)
    assert phase0_tasks(112) == (448, 544, 32, 144)
    assert phase0_tasks(191) == (527, 623, 111, 223)
    assert phase0_tasks(192) == (528, 624)
    assert phase0_tasks(207) == (543, 639)
    assert phase0_tasks(208) == (0, 112)
    assert phase0_tasks(239) == (31, 143)

    assert all(len(phase0_tasks(pid)) == 2 for pid in range(112))
    assert all(len(phase0_tasks(pid)) == 4 for pid in range(112, 192))
    assert all(len(phase0_tasks(pid)) == 2 for pid in range(192, PROGRAMS))

    routed = {
        task - 224
        for pid in range(112)
        for task in phase0_tasks(pid)
        if 224 <= task < 448
    }
    shared = {
        task - 448
        for pid in range(112, 208)
        for task in phase0_tasks(pid)
        if task >= 448
    }
    router = {
        task for pid in range(112, PROGRAMS) for task in phase0_tasks(pid) if task < 224
    }
    assert routed == set(range(224))
    assert shared == set(range(192))
    assert router == set(range(224))


def test_amd_topology_schedule_exact_coverage() -> None:
    generator = random.Random(0)
    for _ in range(100):
        populations = [16] * 8
        for _ in range(PROGRAMS - sum(populations)):
            candidates = [
                index for index, count in enumerate(populations) if count < 32
            ]
            populations[generator.choice(candidates)] += 1
        worker_xcc = [
            xcc for xcc, population in enumerate(populations) for _ in range(population)
        ]
        generator.shuffle(worker_xcc)

        comm = communication_indices(worker_xcc)
        assert {index for index in comm if index < COMM_WORKERS} == set(
            range(COMM_WORKERS)
        )
        assert sum(index < SHARED_COMM_WORKERS for index in comm) == 14

        for local_routes in range(1, 17):
            for route in range(local_routes):
                start, count = balanced_route_xcd_interval(route, local_routes)
                workers = expert_group_coordinates(worker_xcc, start, count)
                worker_count = len(workers)
                assert worker_count > 0
                w13 = [
                    task
                    for _, rank in workers
                    for task in w13_tasks(rank, worker_count)
                ]
                w2 = [
                    task for _, rank in workers for task in w2_tasks(rank, worker_count)
                ]
                assert sorted(w13) == list(range(384))
                assert len(w13) == len(set(w13))
                assert sorted(w2) == list(range(448))
                assert len(w2) == len(set(w2))


def test_amd_topology_ticket_gate_accepts_only_exact_30_per_xcd() -> None:
    """P=240 and the 32-CU/XCD cap make all eight rank-29 tickets exact."""

    exact = [WORKERS_PER_XCD] * 8
    for generation in (1, 2, 1_000_001):
        assert topology_ticket_contributions(exact, generation) == tuple(range(8))
        assert topology_ticket_gate_releases(exact, generation=generation)

    # Exhaust every population transfer admitted by one-workgroup/CU residency.
    # Any nonzero transfer creates one bucket below 30, so at most seven XCDs
    # can contribute the exact current-generation rank-29 ticket.
    for donor in range(8):
        for receiver in range(8):
            if donor == receiver:
                continue
            for transfer in (1, 2):
                populations = exact.copy()
                populations[donor] -= transfer
                populations[receiver] += transfer
                assert len(topology_ticket_contributions(populations)) == 7
                assert not topology_ticket_gate_releases(populations)


def test_amd_topology_ticket_gate_rejects_invalid_dense_xcc_ids() -> None:
    """A masked four-bit ID outside 0..7 cannot index or satisfy the gate."""

    populations = [30] * 7 + [29]
    assert sum(populations) + 1 == PROGRAMS
    assert not topology_ticket_gate_releases(
        populations,
        invalid_xcc_workers=1,
    )
    assert topology_ticket_contributions(populations) == tuple(range(7))


def test_amd_exact_ticket_coordinates_preserve_comm_and_expert_ownership() -> None:
    coordinates = [
        (xcd, local_rank) for local_rank in range(WORKERS_PER_XCD) for xcd in range(8)
    ]
    comm_indices = [local_rank * 8 + xcd for xcd, local_rank in coordinates]
    assert sorted(comm_indices) == list(range(PROGRAMS))
    assert sum(index < SHARED_COMM_WORKERS for index in comm_indices) == 14
    assert [expert_population(xcd) for xcd in range(8)] == [28] * 6 + [29] * 2

    for local_routes in range(1, 17):
        for route in range(local_routes):
            start, count = balanced_route_xcd_interval(route, local_routes)
            worker_count = sum(
                expert_population(xcd) for xcd in range(start, start + count)
            )
            w13 = [
                task
                for rank in range(worker_count)
                for task in w13_tasks(rank, worker_count)
            ]
            w2 = [
                task
                for rank in range(worker_count)
                for task in w2_tasks(rank, worker_count)
            ]
            assert sorted(w13) == list(range(384))
            assert len(w13) == len(set(w13))
            assert sorted(w2) == list(range(448))
            assert len(w2) == len(set(w2))


def test_amd_each_xcd_owns_at_most_two_routes() -> None:
    """The runtime inverse exactly covers every balanced route interval."""

    for local_routes in range(17):
        assigned: list[int] = []
        for xcd_ordinal in range(8):
            routes = expert_routes_for_xcd(xcd_ordinal, local_routes)
            assert len(routes) <= 2
            expected: list[int] = []
            for route in range(local_routes):
                start, count = balanced_route_xcd_interval(route, local_routes)
                if start <= xcd_ordinal < start + count:
                    expected.append(route)
            assert routes == tuple(expected)
            assigned.extend(routes)
        if local_routes <= 8:
            assert all(
                len(expert_routes_for_xcd(xcd_ordinal, local_routes))
                == int(local_routes > 0)
                for xcd_ordinal in range(8)
            )
        assert sorted(set(assigned)) == list(range(local_routes))


@pytest.mark.parametrize(
    "populations",
    (
        [2, 34, 34, 34, 34, 34, 34, 34],
        [34, 34, 34, 34, 34, 2, 34, 34],
        [34, 34, 34, 34, 34, 34, 1, 35],
        [34, 34, 34, 34, 34, 34, 35, 1],
    ),
)
def test_amd_topology_rejects_an_xcd_without_an_expert_worker(
    populations: list[int],
) -> None:
    """Eight observed XCDs and 240 workers alone are not sufficient."""

    assert len(populations) == 8
    assert sum(populations) == PROGRAMS
    assert not _topology_capacity_valid(populations)


def test_amd_topology_accepts_the_qualified_capacity_floor() -> None:
    populations = [3, 3, 3, 3, 3, 3, 111, 111]
    assert _topology_capacity_valid(populations)
