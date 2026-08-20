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

"""Pure-Python oracle for the gfx950 Kimi K3 MegaMoE static schedule."""

from __future__ import annotations

from collections.abc import Sequence

PROGRAMS = 240
XCDS = 8
SUBGROUPS = 8
SHARED_COMM_WORKERS = 14
ROUTED_COMM_WORKERS = 7
COMM_WORKERS = SHARED_COMM_WORKERS + ROUTED_COMM_WORKERS
WORKERS_PER_XCD = PROGRAMS // XCDS


def phase0_tasks(pid: int, programs: int = PROGRAMS) -> tuple[int, ...]:
    """Return phase-zero logical tasks owned by one persistent workgroup."""

    if not 0 <= pid < programs:
        raise ValueError("pid is outside the persistent grid")
    if programs != PROGRAMS:
        raise ValueError("phase-zero specialization requires P=240")
    if pid < 112:
        return (224 + pid, 224 + pid + 112)
    if pid < 208:
        shared_pair = pid - 112
        tasks = [448 + shared_pair, 448 + shared_pair + 96]
        if pid < 192:
            router_pair = pid - 80
            tasks.extend((router_pair, router_pair + 112))
        return tuple(tasks)
    router_pair = pid - 208
    return (router_pair, router_pair + 112)


def shared_down_tasks(pid: int) -> tuple[tuple[int, int], ...]:
    """Return ``(stripe, N8-task)`` pairs for a shared-down producer."""

    if not 0 <= pid < 224:
        return ()
    stripe, rank = divmod(pid, 16)
    return tuple((stripe, task) for task in range(rank, 64, 16))


def balanced_route_xcd_interval(route: int, local_routes: int) -> tuple[int, int]:
    """Return the deterministic XCD start/count for one compact local route."""

    if not 1 <= local_routes <= 16 or not 0 <= route < local_routes:
        raise ValueError("invalid compact route coordinate")
    if local_routes <= XCDS:
        base, extra = divmod(XCDS, local_routes)
        count = base + int(route < extra)
        start = route * base + min(route, extra)
        return start, count
    return route % XCDS, 1


def expert_routes_for_xcd(
    xcd_ordinal: int,
    local_routes: int,
) -> tuple[int, ...]:
    """Return the at-most-two compact routes owned by one XCD.

    This is the inverse of :func:`balanced_route_xcd_interval`.  For at most
    eight routes the balanced intervals partition the XCDs, so one XCD owns
    exactly one route.  Above eight routes, route ``i`` is assigned to XCD
    ``i % 8``, so an XCD owns its ordinal and, when present, ordinal plus eight.
    """

    if not 0 <= xcd_ordinal < XCDS:
        raise ValueError("invalid XCD ordinal")
    if not 0 <= local_routes <= 16:
        raise ValueError("invalid local route count")
    if local_routes == 0:
        return ()
    if local_routes <= XCDS:
        base, extra = divmod(XCDS, local_routes)
        long = base + 1
        long_xcds = extra * long
        if xcd_ordinal < long_xcds:
            return (xcd_ordinal // long,)
        return (extra + (xcd_ordinal - long_xcds) // base,)
    routes = [xcd_ordinal]
    if xcd_ordinal + XCDS < local_routes:
        routes.append(xcd_ordinal + XCDS)
    return tuple(routes)


def local_ranks(worker_xcc: Sequence[int]) -> tuple[list[int], dict[int, int]]:
    """Return program-ID-ordered XCD-local ranks and populations."""

    seen: dict[int, int] = {}
    ranks: list[int] = []
    for xcc in worker_xcc:
        ranks.append(seen.get(xcc, 0))
        seen[xcc] = seen.get(xcc, 0) + 1
    return ranks, seen


def communication_indices(worker_xcc: Sequence[int]) -> tuple[int, ...]:
    """Return the topology-ordered communication index for every workgroup."""

    observed = sorted(set(worker_xcc))
    ordinal = {xcc: index for index, xcc in enumerate(observed)}
    ranks, _ = local_ranks(worker_xcc)
    return tuple(
        ranks[pid] * len(observed) + ordinal[xcc] for pid, xcc in enumerate(worker_xcc)
    )


def shared_quota(xcd_ordinal: int, observed_xcds: int = XCDS) -> int:
    """Number of low local ranks reserved for shared communication."""

    return sum(
        local_rank * observed_xcds + xcd_ordinal < SHARED_COMM_WORKERS
        for local_rank in range(SHARED_COMM_WORKERS)
    )


def topology_ticket_contributions(
    populations: Sequence[int],
    generation: int = 1,
) -> tuple[int, ...]:
    """Return XCDs that contribute the exact rank-29 ticket this generation.

    The production counters are cumulative and a successful predecessor leaves
    every bucket at ``30 * (generation - 1)``.  This oracle deliberately models
    only the current launch population; an invalid launch has no successor.
    """

    if len(populations) != XCDS:
        raise ValueError("the ticket specialization requires eight dense XCDs")
    if generation <= 0:
        raise ValueError("generation must be positive")
    if any(population < 0 or population > 32 for population in populations):
        raise ValueError("an XCD population must be in [0, 32]")
    base = WORKERS_PER_XCD * (generation - 1)
    return tuple(
        xcd
        for xcd, population in enumerate(populations)
        if any(
            ticket // WORKERS_PER_XCD + 1 == generation
            and ticket % WORKERS_PER_XCD == WORKERS_PER_XCD - 1
            for ticket in range(base, base + population)
        )
    )


def topology_ticket_gate_releases(
    populations: Sequence[int],
    *,
    invalid_xcc_workers: int = 0,
    generation: int = 1,
) -> bool:
    """Return whether one P=240 launch releases the eight-XCD ticket gate."""

    if invalid_xcc_workers < 0:
        raise ValueError("invalid_xcc_workers must be nonnegative")
    if sum(populations) + invalid_xcc_workers != PROGRAMS:
        raise ValueError("the topology ticket proof requires P=240 arrivals")
    return len(topology_ticket_contributions(populations, generation)) == XCDS


def expert_population(xcd_ordinal: int) -> int:
    """Return the exact expert-worker count after the 14 shared reservations."""

    if not 0 <= xcd_ordinal < XCDS:
        raise ValueError("invalid XCD ordinal")
    return WORKERS_PER_XCD - shared_quota(xcd_ordinal)


def expert_group_coordinates(
    worker_xcc: Sequence[int],
    route_start: int,
    route_xcd_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(pid, expert-rank)`` for eligible workers in one XCD interval."""

    observed = sorted(set(worker_xcc))
    ordinal = {xcc: index for index, xcc in enumerate(observed)}
    ranks, populations = local_ranks(worker_xcc)
    expert_counts = {
        xcc: populations[xcc] - shared_quota(ordinal[xcc], len(observed))
        for xcc in observed
    }
    prefix: dict[int, int] = {}
    count = 0
    for xcc in observed:
        xcd_ordinal = ordinal[xcc]
        if route_start <= xcd_ordinal < route_start + route_xcd_count:
            prefix[xcc] = count
            count += expert_counts[xcc]
    result: list[tuple[int, int]] = []
    comm = communication_indices(worker_xcc)
    for pid, xcc in enumerate(worker_xcc):
        xcd_ordinal = ordinal[xcc]
        if (
            route_start <= xcd_ordinal < route_start + route_xcd_count
            and comm[pid] >= SHARED_COMM_WORKERS
        ):
            result.append(
                (
                    pid,
                    prefix[xcc] + ranks[pid] - shared_quota(xcd_ordinal, len(observed)),
                )
            )
    return tuple(result)


def w13_tasks(worker_rank: int, worker_count: int) -> tuple[int, ...]:
    """Return the two-cohort W13 N8 tasks assigned to one expert worker."""

    tasks: list[int] = []
    for cohort in range(2):
        tasks.extend(range(worker_rank + cohort * worker_count, 384, 2 * worker_count))
    return tuple(tasks)


def w2_tasks(worker_rank: int, worker_count: int) -> tuple[int, ...]:
    """Return W2 N8 tasks assigned to one expert worker."""

    return tuple(range(worker_rank, 448, worker_count))


__all__ = [
    "COMM_WORKERS",
    "PROGRAMS",
    "ROUTED_COMM_WORKERS",
    "SHARED_COMM_WORKERS",
    "SUBGROUPS",
    "WORKERS_PER_XCD",
    "XCDS",
    "balanced_route_xcd_interval",
    "communication_indices",
    "expert_group_coordinates",
    "expert_population",
    "expert_routes_for_xcd",
    "local_ranks",
    "phase0_tasks",
    "shared_down_tasks",
    "shared_quota",
    "topology_ticket_contributions",
    "topology_ticket_gate_releases",
    "w13_tasks",
    "w2_tasks",
]
