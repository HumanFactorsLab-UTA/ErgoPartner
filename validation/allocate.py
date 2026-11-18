"""Solve the frame allocation for a two-stratum validation design.

The design has to satisfy several things at once, and they are not equally
negotiable.  How many frames come from each (stratum, band) cell is the whole point
of the study, so those counts are exact.  Everything else -- task balance,
per-participant coverage, spreading frames over distinct recordings -- is something
we want as much of as the data will give, and demanding it exactly is what makes an
otherwise fine design report itself as infeasible.

So the hard counts are enforced as capacities and the rest are priced as costs.  A
task x cell-type count matrix is enumerated under capacity pruning with the cell-type
marginals fixed and the task marginals allowed to drift by a widening tolerance;
for each candidate matrix, participants are assigned by min-cost flow.  The search
is finite and always reports what it actually achieved, including where it had to
give ground.
"""

from __future__ import annotations

import logging
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

#: Charged for the 2nd, 3rd, ... frame drawn from one (participant, task) recording.
#: Dominates the tie-break jitter so the solver covers as many distinct recordings as
#: the data supports and doubles up only where a participant has no alternative.
PAIR_SPREAD_PENALTY = 1000

#: Random jitter that separates equally good allocations, so two seeds give genuinely
#: different but equally valid designs.
TIE_BREAK_RANGE = 64

MAX_MATRIX_CANDIDATES = 20000
MAX_MATRIX_ATTEMPTS = 400


class AllocationError(RuntimeError):
    """Raised when no allocation satisfies even the hard cell-type quotas."""


@dataclass
class Allocation:
    """A solved design: how many frames to draw from each (participant, task, cell)."""

    counts: dict[tuple[str, str, str], int]
    matrix: tuple[tuple[int, ...], ...]
    tasks: list[str]
    cell_types: list[str]
    task_totals: dict[str, int]
    task_target: dict[str, int]
    participant_totals: dict[str, int]
    recordings_used: int
    recordings_doubled: int
    notes: list[str] = field(default_factory=list)


class MinCostFlow:
    """Successive-shortest-path min-cost flow over the allocation network.

    Plain max-flow returns *a* feasible allocation, which in practice means a few
    participants supplying everything from one recording while other combinations go
    unused.  Costs let the design state a preference while capacities still enforce
    the quotas.

    Residual arcs carry negative costs, so shortest paths use SPFA rather than
    Dijkstra.  All forward costs must be non-negative and the network must be built
    completely before flowing: successive shortest paths from zero flow keep the
    residual graph free of negative cycles, and a negative cycle would make the
    predecessor chain cyclic.
    """

    def __init__(self, node_count: int) -> None:
        # edge = [target, residual capacity, cost, index of paired reverse edge]
        self._graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int, cost: int = 0) -> list[int]:
        forward = [target, capacity, cost, len(self._graph[target])]
        backward = [source, 0, -cost, len(self._graph[source])]
        self._graph[source].append(forward)
        self._graph[target].append(backward)
        return forward

    def flow(self, source: int, sink: int, limit: int) -> int:
        node_count = len(self._graph)
        pushed_total = 0
        while pushed_total < limit:
            distance = [sys.maxsize] * node_count
            previous: list[tuple[int, int] | None] = [None] * node_count
            queued = [False] * node_count
            distance[source] = 0
            queue = deque([source])
            queued[source] = True
            while queue:
                node = queue.popleft()
                queued[node] = False
                base = distance[node]
                for index, (target, capacity, cost, _) in enumerate(self._graph[node]):
                    if capacity > 0 and base + cost < distance[target]:
                        distance[target] = base + cost
                        previous[target] = (node, index)
                        if not queued[target]:
                            queue.append(target)
                            queued[target] = True
            if distance[sink] == sys.maxsize:
                break

            path: list[tuple[int, int]] = []
            node = sink
            while node != source:
                step = previous[node]
                if step is None or len(path) > node_count:
                    raise AllocationError(
                        "min-cost flow found a cyclic predecessor chain, which means the "
                        "network was built with a negative-cost cycle"
                    )
                path.append(step)
                node = step[0]

            bottleneck = limit - pushed_total
            for parent, index in path:
                bottleneck = min(bottleneck, self._graph[parent][index][1])
            for parent, index in path:
                edge = self._graph[parent][index]
                edge[1] -= bottleneck
                self._graph[edge[0]][edge[3]][1] += bottleneck
            pushed_total += bottleneck
        return pushed_total


def enumerate_matrices(
    row_target: list[int],
    row_tolerance: int,
    column_totals: list[int],
    cell_caps: list[list[int]],
    rng: random.Random,
    limit: int = MAX_MATRIX_CANDIDATES,
) -> tuple[list[tuple[tuple[int, ...], ...]], bool]:
    """Enumerate row x column matrices with exact column totals and near-target rows.

    Columns are the hard quotas (stratum x band); rows are tasks, allowed to deviate
    from target by up to ``row_tolerance``.  Candidates respect ``cell_caps`` -- what
    each cell could actually supply -- which prunes most hopeless branches before any
    flow runs.  Values are explored in randomised order so truncating at ``limit``
    does not bias toward lexicographically small matrices.
    """

    row_count, column_count = len(row_target), len(column_totals)
    matrices: list[tuple[tuple[int, ...], ...]] = []
    total_demand = sum(column_totals)

    def fill_row(row: int, remaining: list[int], placed: int, rows: list[tuple[int, ...]]) -> None:
        if len(matrices) >= limit:
            return
        if row == row_count:
            if all(value == 0 for value in remaining):
                matrices.append(tuple(rows))
            return

        outstanding = sum(remaining)
        # Every remaining row must land within tolerance of its target.
        lowest = sum(max(0, row_target[index] - row_tolerance) for index in range(row, row_count))
        highest = sum(row_target[index] + row_tolerance for index in range(row, row_count))
        if not (lowest <= outstanding <= highest):
            return

        row_cells: list[int] = []
        low_bound = max(0, row_target[row] - row_tolerance)
        high_bound = min(row_target[row] + row_tolerance, outstanding)

        def fill_cell(column: int, used: int, columns: list[int]) -> None:
            if len(matrices) >= limit:
                return
            if column == column_count:
                if low_bound <= used <= high_bound:
                    fill_row(row + 1, columns, placed + used, [*rows, tuple(row_cells)])
                return
            headroom = sum(
                min(cell_caps[row][index], columns[index]) for index in range(column, column_count)
            )
            if used + headroom < low_bound:
                return
            upper = min(columns[column], cell_caps[row][column], high_bound - used)
            options = list(range(upper + 1))
            rng.shuffle(options)
            for value in options:
                row_cells.append(value)
                columns[column] -= value
                fill_cell(column + 1, used + value, columns)
                columns[column] += value
                row_cells.pop()

        fill_cell(0, 0, remaining)

    if total_demand:
        fill_row(0, list(column_totals), 0, [])
    return matrices, len(matrices) >= limit


def _assign_participants(
    matrix: tuple[tuple[int, ...], ...],
    availability: dict[tuple[str, str, str], int],
    participants: list[str],
    tasks: list[str],
    cell_types: list[str],
    pair_cap: int,
    min_per_participant: int,
    max_per_participant: int,
    rng: random.Random,
) -> dict[tuple[str, str, str], int] | None:
    """Distribute a fixed task x cell-type matrix over participants, or ``None``.

    Network layout (all capacities integral, so the flow is integral too)::

        source -> participant                 min_per_participant units free,
                                              then (max - min) units at a dominating cost
        participant -> (participant, task)    pair_cap unit edges, rising cost
        (participant, task) -> (task, cell)   frames actually available
        (task, cell) -> sink                  matrix[task][cell]  (must saturate)

    The participant floor is priced rather than enforced: total flow is pinned to the
    matrix demand, so cost is minimised exactly when usage of the free below-floor
    edges is maximised.  If any allocation respects the floor, the cheapest one does
    -- and the result is verified explicitly regardless.
    """

    participant_count, task_count, cell_count = len(participants), len(tasks), len(cell_types)
    source = 0
    participant_base = 1
    pair_base = participant_base + participant_count
    cell_base = pair_base + participant_count * task_count
    sink = cell_base + task_count * cell_count
    network = MinCostFlow(sink + 1)

    # Priced above any achievable spread cost, so meeting the floor always outranks
    # spreading frames across recordings.
    above_floor_cost = (
        participant_count
        * max_per_participant
        * (pair_cap * PAIR_SPREAD_PENALTY + TIE_BREAK_RANGE)
        + 1
    )

    source_edges = []
    for index in range(participant_count):
        node = participant_base + index
        floor_edge = network.add_edge(source, node, min_per_participant, 0)
        extra_capacity = max(0, max_per_participant - min_per_participant)
        extra_edge = network.add_edge(source, node, extra_capacity, above_floor_cost)
        source_edges.append((floor_edge, min_per_participant, extra_edge, extra_capacity))

    for index in range(participant_count):
        for task in range(task_count):
            for unit in range(pair_cap):
                network.add_edge(
                    participant_base + index,
                    pair_base + index * task_count + task,
                    1,
                    unit * PAIR_SPREAD_PENALTY + rng.randrange(TIE_BREAK_RANGE),
                )

    cell_edges: list[tuple[int, int, int, list[int], int]] = []
    for index in range(participant_count):
        for task in range(task_count):
            for cell in range(cell_count):
                capacity = min(
                    availability.get((participants[index], tasks[task], cell_types[cell]), 0),
                    pair_cap,
                )
                if capacity <= 0:
                    continue
                edge = network.add_edge(
                    pair_base + index * task_count + task,
                    cell_base + task * cell_count + cell,
                    capacity,
                    rng.randrange(TIE_BREAK_RANGE),
                )
                cell_edges.append((index, task, cell, edge, capacity))

    demand = 0
    for task in range(task_count):
        for cell in range(cell_count):
            network.add_edge(cell_base + task * cell_count + cell, sink, matrix[task][cell])
            demand += matrix[task][cell]

    if network.flow(source, sink, demand) != demand:
        return None
    for floor_edge, floor_capacity, extra_edge, extra_capacity in source_edges:
        drawn = (floor_capacity - floor_edge[1]) + (extra_capacity - extra_edge[1])
        if drawn < min_per_participant:
            return None

    allocation: dict[tuple[str, str, str], int] = {}
    for index, task, cell, edge, capacity in cell_edges:
        used = capacity - edge[1]
        if used > 0:
            allocation[(participants[index], tasks[task], cell_types[cell])] = used
    return allocation


def describe_infeasibility(
    availability: dict[tuple[str, str, str], int],
    participants: list[str],
    tasks: list[str],
    cell_types: list[str],
    cell_quota: dict[str, int],
    total_frames: int,
    max_per_participant: int,
) -> str:
    """Explain why no design exists, in terms the operator can act on."""

    lines = ["No allocation satisfies the requested (stratum x band) quotas."]
    ceiling = sum(
        min(sum(value for key, value in availability.items() if key[0] == person), max_per_participant)
        for person in participants
    )
    if ceiling < total_frames:
        lines.append(
            f"  Cohort ceiling: {len(participants)} participant(s) can supply at most {ceiling} "
            f"frame(s) at {max_per_participant} each, short of the {total_frames} requested."
        )
    for cell in cell_types:
        reachable = sum(
            min(value, max_per_participant) for key, value in availability.items() if key[2] == cell
        )
        supply = sum(value for key, value in availability.items() if key[2] == cell)
        if reachable < cell_quota.get(cell, 0):
            lines.append(
                f"  {cell}: quota {cell_quota.get(cell, 0)}, only {reachable} reachable "
                f"frame(s) (raw supply {supply}) across {len(participants)} participants"
            )
    lines.append(
        "  Options: widen the stability window, move quota between strata, lower the "
        "per-participant floor, or reduce the total."
    )
    return "\n".join(lines)


def solve(
    availability: dict[tuple[str, str, str], int],
    participants: list[str],
    tasks: list[str],
    cell_types: list[str],
    cell_quota: dict[str, int],
    task_target: dict[str, int],
    min_per_participant: int,
    max_per_participant: int,
    pair_cap: int,
    seed: int,
    max_task_tolerance: int = 12,
) -> Allocation:
    """Find the best allocation meeting the cell quotas exactly.

    Task balance is relaxed one frame at a time until the design becomes solvable,
    so a task that genuinely cannot supply a cell -- seated screwdriving has no
    sustained high-risk posture, for instance -- costs exactly the deviation it has
    to, rather than failing the whole design.
    """

    rng = random.Random(seed)
    total_frames = sum(cell_quota[cell] for cell in cell_types)
    if min_per_participant * len(participants) > total_frames:
        raise AllocationError(
            f"{len(participants)} participants x {min_per_participant} minimum exceeds the "
            f"{total_frames}-frame budget"
        )

    cell_caps = [
        [
            sum(
                min(availability.get((person, task, cell), 0), pair_cap)
                for person in participants
            )
            for cell in cell_types
        ]
        for task in tasks
    ]
    row_target = [task_target[task] for task in tasks]
    column_totals = [cell_quota[cell] for cell in cell_types]
    notes: list[str] = []

    for tolerance in range(0, max_task_tolerance + 1):
        matrices, truncated = enumerate_matrices(
            row_target, tolerance, column_totals, cell_caps, rng
        )
        if not matrices:
            continue
        ideal = total_frames / max(1, len(tasks) * len(cell_types))

        def empty_but_available(matrix: tuple[tuple[int, ...], ...]) -> int:
            """Cells left at zero even though the data could have filled them.

            A task contributing nothing to a cell it has thousands of candidates for
            is a design wart a reader will ask about, and it is avoidable: the
            marginals do not care which cells carry the count.
            """

            return sum(
                1
                for row_index, row in enumerate(matrix)
                for column_index, value in enumerate(row)
                if value == 0 and cell_caps[row_index][column_index] > 0
            )

        matrices.sort(
            key=lambda matrix: (
                sum(abs(sum(row) - row_target[index]) for index, row in enumerate(matrix)),
                empty_but_available(matrix),
                sum((value - ideal) ** 2 for row in matrix for value in row),
                rng.random(),
            )
        )
        attempts = matrices[:MAX_MATRIX_ATTEMPTS]
        LOGGER.info(
            "Task tolerance +/-%d: trying %d of %d candidate matrices%s",
            tolerance,
            len(attempts),
            len(matrices),
            " (enumeration truncated)" if truncated else "",
        )
        for matrix in attempts:
            counts = _assign_participants(
                matrix,
                availability,
                participants,
                tasks,
                cell_types,
                pair_cap,
                min_per_participant,
                max_per_participant,
                rng,
            )
            if counts is None:
                continue
            task_totals = {task: sum(matrix[index]) for index, task in enumerate(tasks)}
            deviation = {
                task: task_totals[task] - task_target[task]
                for task in tasks
                if task_totals[task] != task_target[task]
            }
            if deviation:
                notes.append(
                    "Task balance relaxed to keep the design solvable: "
                    + ", ".join(f"{task} {value:+d}" for task, value in sorted(deviation.items()))
                )
            participant_totals: dict[str, int] = defaultdict(int)
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for (person, task, _cell), count in counts.items():
                participant_totals[person] += count
                pair_counts[(person, task)] += count
            return Allocation(
                counts=counts,
                matrix=matrix,
                tasks=list(tasks),
                cell_types=list(cell_types),
                task_totals=task_totals,
                task_target=dict(task_target),
                participant_totals=dict(participant_totals),
                recordings_used=len(pair_counts),
                recordings_doubled=sum(1 for value in pair_counts.values() if value > 1),
                notes=notes,
            )

    raise AllocationError(
        describe_infeasibility(
            availability, participants, tasks, cell_types, cell_quota, total_frames, max_per_participant
        )
    )
