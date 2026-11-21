"""Replay machinery behind `verify_movement` (private)."""

from __future__ import annotations

import copy
import itertools
import logging
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from pushquantum_validator.errors import ArchitectureError
from pushquantum_validator.spec import AXES, Zone, parse_spec

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction

Site = tuple[int, int]
MOVE_NAME = "move"
ALWAYS_ALLOWED = frozenset({"measure", "barrier"})
SPACING_PENALTY = 2.0
"""Time factor for a move with spacing warnings."""

logger = logging.getLogger("pushquantum_validator")


# ---------------------------------------------------------------------- zone geometry


def zone_contains(zone: Zone, site: Site) -> bool:
    x, y = site
    return zone.x[0] <= x <= zone.x[1] and zone.y[0] <= y <= zone.y[1]


def pair_slot(zone: Zone, line: int) -> int:
    """0 for the first site of a pair, 1 for the second."""
    return (line - (zone.x[0] if zone.pair_axis == "x" else zone.y[0])) % 2


def are_paired(zone: Zone, a: Site, b: Site) -> bool:
    if zone.pair_axis is None or not (zone_contains(zone, a) and zone_contains(zone, b)):
        return False
    axis = AXES.index(zone.pair_axis)
    low, high = sorted((a[axis], b[axis]))
    return a[1 - axis] == b[1 - axis] and high == low + 1 and pair_slot(zone, low) == 0


def physical(zone: Zone, line: int, axis: str) -> float:
    """Physical coordinate of grid line `line` along `axis`."""
    if axis != zone.pair_axis or zone.pair_distance is None:
        return line * zone.spacing
    start = zone.x[0] if axis == "x" else zone.y[0]
    pair, slot = divmod(line - start, 2)
    return start * zone.spacing + pair * 2 * zone.spacing + slot * zone.pair_distance


# ---------------------------------------------------------------------- move rules
# Copies of the template's checks (kept in sync by tests/test_copied_checks.py). `_as_lines` also
# rejects bools and non-sequences; `_check_lines` returns spacing warnings instead of raising.


def _is_proportional(source: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """Whether the gaps of `target` are those of `source` scaled by one positive factor."""
    if len(source) < 2:
        return True
    (s0, s1), (t0, t1) = source[:2], target[:2]
    return t1 > t0 and all((t - t0) * (s1 - s0) == (t1 - t0) * (s - s0) for s, t in zip(source, target, strict=True))


def _as_lines(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        msg = f"`{name}` must be a sequence of integers, got {values!r}."
        raise ValueError(msg)
    if not values:
        msg = f"`{name}` must not be empty."
        raise ValueError(msg)
    # `type(v) is int` is the fast path; the ABC check admits other integers (e.g. numpy's).
    if not all(type(v) is int or (isinstance(v, Integral) and not isinstance(v, bool)) for v in values):
        msg = f"`{name}` must contain integers, got {values!r}."
        raise ValueError(msg)
    result = tuple(int(v) for v in values)
    if len(set(result)) != len(result):
        msg = f"`{name}` must not contain duplicates, got {result!r}."
        raise ValueError(msg)
    return result


def _check_lines(start: tuple[int, ...], end: tuple[int, ...], name: str) -> list[str]:
    if len(end) > len(start):
        msg = f"`new_{name}` cannot have more lines than `{name}`, got {len(end)} and {len(start)}."
        raise ValueError(msg)
    contiguous = tuple(range(start[0], start[0] + len(start)))
    if start != contiguous:
        msg = f"Picked-up {name} {start} must be contiguous and ascending, e.g. {contiguous}."
        raise ValueError(msg)
    if any(b <= a for a, b in itertools.pairwise(end)):
        msg = f"Target {name} {end} must be ascending: the order of {name} is preserved."
        raise ValueError(msg)
    if len(end) == len(start) and not _is_proportional(start, end):
        return [f"Target {name} {end} are not evenly spaced (spacing not proportional to the contiguous {start})."]
    return []


@dataclass(frozen=True)
class ValidatedMove:
    """A move rebuilt from an instruction's raw params and re-checked; the template is not trusted."""

    rows: tuple[int, ...]
    cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]
    num_qubits: int
    spacing_warnings: tuple[str, ...] = ()

    @classmethod
    def from_instruction(cls, operation: Instruction) -> ValidatedMove:
        """Raises `ArchitectureError` if `operation` is not a valid move."""
        try:
            if operation.name != MOVE_NAME:
                msg = f"Expected a {MOVE_NAME!r} instruction, got {operation.name!r}."
                raise ValueError(msg)
            if len(operation.params) != 4:
                msg = f"A move needs 4 params (rows, cols, new_rows, new_cols), got {len(operation.params)}."
                raise ValueError(msg)
            if operation.num_clbits or operation.num_qubits < 1:
                msg = "A move must act on at least one qubit and no classical bits."
                raise ValueError(msg)
            rows, cols, new_rows, new_cols = (
                _as_lines(v, n) for v, n in zip(operation.params, ("rows", "cols", "new_rows", "new_cols"), strict=True)
            )
            spacing = _check_lines(rows, new_rows, "rows") + _check_lines(cols, new_cols, "cols")
        except ValueError as error:
            raise ArchitectureError(str(error)) from error
        return cls(rows, cols, new_rows, new_cols, operation.num_qubits, tuple(spacing))


def _remove_empty(
    lines: tuple[int, ...], targets: tuple[int, ...], used: set[int], name: str, spacing: list[str]
) -> tuple[int, ...]:
    """Drops the lines without atoms; the rest must match `targets` one-to-one."""
    if len(lines) == len(targets):
        return lines
    kept = tuple(line for line in lines if line in used)
    if len(kept) != len(targets):
        msg = (
            f"Moving {name} {lines} -> {targets}: after removing empty {name}, the occupied {name} "
            f"{kept} do not match the {len(targets)} target {name} one-to-one."
        )
        raise ArchitectureError(msg)
    if not _is_proportional(kept, targets):
        spacing.append(
            f"Moving {name} {lines} -> {targets}: the spacing of the occupied {name} {kept} does not "
            f"scale proportionally to {targets}."
        )
    return kept


def site_mapping(move: ValidatedMove, occupied: dict[Site, int], spacing: list[str]) -> dict[Site, Site]:
    """Every picked-up intersection mapped to its target; empty lines are dropped for a squeeze."""
    rows, cols = move.rows, move.cols
    if len(move.new_rows) < len(rows) or len(move.new_cols) < len(cols):
        selected = [(x, y) for x, y in occupied if x in cols and y in rows]
        rows = _remove_empty(rows, move.new_rows, {y for _, y in selected}, "rows", spacing)
        cols = _remove_empty(cols, move.new_cols, {x for x, _ in selected}, "cols", spacing)
    row_map = dict(zip(rows, move.new_rows, strict=True))
    col_map = dict(zip(cols, move.new_cols, strict=True))
    return {(x, y): (col_map[x], row_map[y]) for y in rows for x in cols}


# ---------------------------------------------------------------------- parallel moves
# Consecutive moves share a step if their line spans stay apart along x and y while moving, and no
# crossing of one move's rows with the other's columns passes over an atom not moved in the step.


@dataclass(frozen=True)
class AodLines:
    """The lines a move uses, each mapped to its target line."""

    rows: dict[int, int]
    cols: dict[int, int]

    @classmethod
    def of(cls, mapping: dict[Site, Site]) -> AodLines:
        return cls({y: y2 for (_, y), (_, y2) in mapping.items()}, {x: x2 for (x, _), (x2, _) in mapping.items()})


def _apart(a: dict[int, int], b: dict[int, int]) -> bool:
    (a_lo, a_hi), (b_lo, b_hi) = (min(a), max(a)), (min(b), max(b))
    return (a_hi < b_lo and a[a_hi] < b[b_lo]) or (b_hi < a_lo and b[b_hi] < a[a_lo])


def _path(start: Site, end: Site) -> list[Site]:
    """The grid sites a point passes moving in a straight line from `start` to `end`."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    steps = math.gcd(dx, dy)
    if steps == 0:
        return [start]
    return [(start[0] + i * dx // steps, start[1] + i * dy // steps) for i in range(steps + 1)]


def can_run_in_parallel(a: AodLines, b: AodLines, stationary: set[Site]) -> bool:
    if not (_apart(a.rows, b.rows) and _apart(a.cols, b.cols)):
        return False
    return not any(
        stationary.intersection(_path((x, y), (x2, y2)))
        for rows, cols in ((a.rows, b.cols), (b.rows, a.cols))
        for (y, y2), (x, x2) in itertools.product(rows.items(), cols.items())
    )


# ---------------------------------------------------------------------- replay


@dataclass(frozen=True)
class MoveRecord:
    index: int
    """Index of the move in `circuit.data`."""
    move: ValidatedMove
    qubits: tuple[int, ...]
    positions: dict[int, Site]
    """New site of every moved qubit."""
    spacing_warnings: tuple[str, ...] = ()
    distance: float = 0.0
    """Longest physical distance travelled by one atom."""
    step: int = 0
    """Parallel step of this move."""

    @property
    def time(self) -> float:
        return self.distance * SPACING_PENALTY if self.spacing_warnings else self.distance


def _validate_loading(loading: dict[int, Site], zone: Zone) -> dict[int, Site]:
    if set(loading) != set(range(len(loading))):
        msg = f"Initial loading qubit indices must be 0..{len(loading) - 1}, got {sorted(loading)}."
        raise ArchitectureError(msg, "loading")
    result: dict[int, Site] = {}
    for qubit, site in loading.items():
        if len(site) != 2 or not all(isinstance(c, Integral) and not isinstance(c, bool) for c in site):
            msg = f"Qubit {qubit}: site must be an integer (x, y) pair, got {site!r}."
            raise ArchitectureError(msg, "loading")
        result[qubit] = (int(site[0]), int(site[1]))
        if not zone_contains(zone, result[qubit]):
            msg = f"Qubit {qubit} is loaded at {result[qubit]}, outside the {zone.name!r} zone."
            raise ArchitectureError(msg, "loading")
    if len(set(result.values())) != len(result):
        msg = "Initial loading places more than one qubit on the same site."
        raise ArchitectureError(msg, "loading")
    return result


class ValidationArchitecture:
    """The validator's zones with a participant's initial loading; replays circuits on them."""

    def __init__(self, zones: dict[str, Zone], loading_zone: str, initial_loading: dict[int, Site]) -> None:
        self.zones = zones
        self.loading_zone = loading_zone
        self.initial_loading = _validate_loading(initial_loading, zones[loading_zone])
        self.zone_at: dict[Site, Zone] = {}
        for zone in reversed(zones.values()):  # the first zone containing a site wins
            sites = itertools.product(range(zone.x[0], zone.x[1] + 1), range(zone.y[0], zone.y[1] + 1))
            self.zone_at.update(dict.fromkeys(sites, zone))
        self.log_file: Path | None = None
        self.reset()

    @classmethod
    def from_loading(cls, loading: dict[int, Site], spec: dict | None = None) -> ValidationArchitecture:
        """The zones of `spec` (default: the validator's) with `loading` (qubit -> `(x, y)`)."""
        if not isinstance(loading, dict):
            msg = f"The initial loading must be a dict, got {type(loading).__name__}."
            raise ArchitectureError(msg)
        return cls(*parse_spec(spec), copy.deepcopy(loading))

    def reset(self) -> None:
        self.positions: dict[int, Site] = dict(self.initial_loading)
        self.occupant: dict[Site, int] = {site: qubit for qubit, site in self.positions.items()}

    def physical_position(self, site: Site) -> tuple[float, float]:
        zone = self.zone_at.get(site)
        if zone is None:
            msg = f"Site {site} is outside every zone."
            raise ArchitectureError(msg)
        return physical(zone, site[0], "x"), physical(zone, site[1], "y")

    def _pair_spacing(self, sites: Sequence[Site], side: str) -> list[str]:
        """Warnings for 3+ lines of a paired axis that mix first and second pair sites."""
        found = []
        for zone in self.zones.values():
            if zone.pair_axis is None:
                continue
            axis = AXES.index(zone.pair_axis)
            lines = sorted({site[axis] for site in sites if zone_contains(zone, site)})
            if len(lines) >= 3 and len({pair_slot(zone, line) for line in lines}) > 1:
                coords = [round(physical(zone, line, zone.pair_axis), 6) for line in lines]
                name = "cols" if zone.pair_axis == "x" else "rows"
                found.append(
                    f"{side.capitalize()} {name} {tuple(lines)} in the {zone.name!r} zone are not evenly spaced "
                    f"physically ({coords}): with 3 or more {name} in a paired zone, use the same site of "
                    "every pair (all first or all second sites)."
                )
        return found

    def _check_single_zone(self, sites: dict[int, Site], action: str) -> None:
        """Raises if the atoms at `sites` (qubit -> site) are in more than one zone."""
        by_zone: dict[str, list[int]] = {}
        for qubit, site in sorted(sites.items()):
            by_zone.setdefault(self.zone_at[site].name, []).append(qubit)
        if len(by_zone) > 1:
            where = "; ".join(f"{name!r}: qubits {qubits}" for name, qubits in by_zone.items())
            msg = f"Move {action} atoms in more than one zone ({where}); each end of a move must lie in one zone."
            raise ArchitectureError(msg)

    def apply_move(self, move: ValidatedMove, qubits: Sequence[int]) -> tuple[dict[Site, Site], tuple[str, ...]]:
        """Checks `move` on `qubits` against the current state and applies it.

        Returns the site mapping and the move's spacing warnings (never errors).
        """
        if len(qubits) != move.num_qubits or len(set(qubits)) != len(qubits):
            msg = f"Move acts on {move.num_qubits} distinct qubits, got {list(qubits)}."
            raise ArchitectureError(msg)
        spacing = list(move.spacing_warnings)
        mapping = site_mapping(move, self.occupant, spacing)
        for target in mapping.values():
            if target not in self.zone_at:
                msg = f"Move target {target} is outside every zone."
                raise ArchitectureError(msg)
        spacing += self._pair_spacing(list(mapping), "picked-up") + self._pair_spacing(list(mapping.values()), "target")

        listed = set(qubits)
        picked = {self.occupant[s]: s for s in mapping if s in self.occupant}
        if ghost := set(picked) - listed:
            qubit = min(ghost)
            msg = f"Qubit {qubit} at {picked[qubit]} lies on a selected row/column intersection and would be picked up."
            raise ArchitectureError(msg)
        if missing := listed - set(picked):
            where = {q: self.positions.get(q) for q in sorted(missing)}
            msg = f"Qubits {where} are not at any intersection of rows {move.rows} x cols {move.cols}."
            raise ArchitectureError(msg)
        self._check_single_zone(picked, "picks up")
        self._check_single_zone({q: mapping[s] for q, s in picked.items()}, "drops off")
        for qubit, source in picked.items():
            occupant = self.occupant.get(mapping[source])
            if occupant is not None and occupant not in picked:
                msg = f"Cannot deposit qubit {qubit} at {mapping[source]}: occupied by qubit {occupant}."
                raise ArchitectureError(msg)

        for source in picked.values():
            del self.occupant[source]
        for qubit, source in picked.items():
            self.positions[qubit] = mapping[source]
            self.occupant[mapping[source]] = qubit
        return mapping, tuple(spacing)

    def check_gate(self, name: str, qubits: Sequence[int]) -> None:
        """Raises unless every qubit is in a zone allowing `name` and a 2-qubit gate acts on a pair."""
        if name in ALWAYS_ALLOWED:
            return
        for qubit in qubits:
            site = self.positions[qubit]
            zone = self.zone_at.get(site)
            if zone is None or name not in zone.operations:
                allowed_in = sorted(z.name for z in self.zones.values() if name in z.operations) or "no zone"
                where = zone.name if zone else "no zone"
                msg = f"{name!r} on qubit {qubit} at {site} ({where}); allowed in: {allowed_in}."
                raise ArchitectureError(msg, "zone")
        if len(qubits) != 2:
            return
        a, b = self.positions[qubits[0]], self.positions[qubits[1]]
        zone = self.zone_at[a]
        if zone.pair_axis is not None and not are_paired(zone, a, b):
            msg = (
                f"{name!r} needs both atoms on one interaction pair of the {zone.name!r} zone "
                f"(sites 2k and 2k+1 along {zone.pair_axis}), but qubit {qubits[0]} is at {a} "
                f"and qubit {qubits[1]} at {b}."
            )
            raise ArchitectureError(msg, "pair")
        if zone.pair_axis is None and abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            msg = f"{name!r} needs adjacent atoms, but qubit {qubits[0]} is at {a} and qubit {qubits[1]} at {b}."
            raise ArchitectureError(msg, "pair")

    def run(self, circuit: QuantumCircuit, log_file: str | Path | None = None) -> list[MoveRecord]:
        """Replays `circuit` from the initial loading; writes spacing warnings, every violation and the
        verdict to `log_file` (default: a new temporary file), kept in `self.log_file` and on a raised error.

        An illegal instruction is skipped (an illegal move is not applied) and the replay goes on, so
        all violations are found. Raises the first as `ArchitectureError`, with all in `violations`."""
        if log_file is None:
            with tempfile.NamedTemporaryFile(prefix="pushquantum_validation_", suffix=".log", delete=False) as tmp:
                log_file = tmp.name
        self.log_file = Path(log_file)
        lines = [f"Replaying {len(circuit.data)} instructions on {circuit.num_qubits} qubits"]
        try:
            trace, violations = self._replay(circuit, lines)
            if violations:
                first = violations[0]
                more = (
                    f" (and {len(violations) - 1} more violation(s), see the replay log)" if len(violations) > 1 else ""
                )
                error = ArchitectureError(f"{first}{more}", first.kind)
                error.violations = tuple(violations)
                raise error
        except ArchitectureError as error:  # raised, so the caller reports it; only the replay log gets it here
            error.violations = error.violations or (error,)
            lines.append(f"ERROR: FAIL: {len(error.violations)} violation(s)")
            error.log_file = self.log_file
            raise
        else:
            spacing = sum(len(r.spacing_warnings) for r in trace)
            lines.append(f"PASS: {len(trace)} moves replayed, {spacing} spacing warning(s)")
            return trace
        finally:
            self.log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _replay(self, circuit: QuantumCircuit, log: list[str]) -> tuple[list[MoveRecord], list[ArchitectureError]]:
        if circuit.num_qubits != len(self.initial_loading):
            msg = f"Circuit has {circuit.num_qubits} qubits but {len(self.initial_loading)} are loaded."
            raise ArchitectureError(msg, "loading")
        self.reset()
        index_of = {bit: i for i, bit in enumerate(circuit.qubits)}
        trace: list[MoveRecord] = []
        violations: list[ArchitectureError] = []
        step = -1
        group: list[AodLines] = []  # moves of the current parallel step
        group_qubits: set[int] = set()
        group_start: dict[int, Site] = {}  # positions when the step started
        layer = _GateLayer()
        for index, instruction in enumerate(circuit.data):
            operation = instruction.operation
            qubits = tuple(index_of[q] for q in instruction.qubits)
            if operation.name == MOVE_NAME:
                self._close_layer(layer, violations, log)
                layer = _GateLayer()
            try:
                if operation.name != MOVE_NAME:
                    group = []
                    if len(qubits) == 2:
                        layer.add(index, qubits)
                    self.check_gate(operation.name, qubits)
                    if len(qubits) == 2:
                        layer.zones.add(self.zone_at[self.positions[qubits[0]]].name)
                    continue
                move = ValidatedMove.from_instruction(operation)
                before = {q: self.positions[q] for q in qubits}
                mapping, spacing = self.apply_move(move, qubits)
                after = {q: self.positions[q] for q in qubits}
                distance = max(
                    math.dist(self.physical_position(before[q]), self.physical_position(after[q])) for q in qubits
                )
                lines = AodLines.of(mapping)
                joins = bool(group) and not group_qubits & set(qubits)
                if joins:
                    moving = group_qubits | set(qubits)
                    stationary = {site for qubit, site in group_start.items() if qubit not in moving}
                    joins = all(can_run_in_parallel(lines, other, stationary) for other in group)
                if not joins:
                    step += 1
                    group, group_qubits, group_start = [], set(), {**self.positions, **before}
                group.append(lines)
                group_qubits.update(qubits)
                for message in spacing:
                    logger.warning("Instruction %d (move on %s): %s", index, list(qubits), message)
                    log.append(f"WARNING: Instruction {index} (move on {list(qubits)}): {message}")
                trace.append(MoveRecord(index, move, qubits, after, spacing, distance, step))
            except ArchitectureError as error:  # every check runs before any state changes: skip and go on
                violation = ArchitectureError(
                    f"Instruction {index} ({operation.name} on {list(qubits)}): {error}", error.kind
                )
                violations.append(violation)
                log.append(f"ERROR: {violation}")
                group = []
        self._close_layer(layer, violations, log)
        return trace, violations

    def _close_layer(self, layer: _GateLayer, violations: list[ArchitectureError], log: list[str]) -> None:
        """Adds an `idle` violation for every zone where the two-qubit gates of `layer` ran while other
        atoms sat there: the entangling pulse reaches every atom in the zone, so each must be in a gate."""
        for name in sorted(layer.zones):
            zone = self.zones[name]
            inside = (q for q, site in self.positions.items() if zone_contains(zone, site))
            idle = sorted(q for q in inside if q not in layer.qubits)
            if not idle:
                continue
            where = ", ".join(f"{q} at {self.positions[q]}" for q in idle)
            span = f"{layer.first}" if layer.first == layer.last else f"{layer.first}-{layer.last}"
            violation = ArchitectureError(
                f"Instructions {span} (two-qubit gates in the {name!r} zone): qubits {where} sit in that zone "
                "without taking part in a two-qubit gate; every atom in the zone must, or be moved out first.",
                "idle",
            )
            violations.append(violation)
            log.append(f"ERROR: {violation}")


@dataclass
class _GateLayer:
    """The two-qubit gates between two moves: the atoms stand still, so they share entangling pulses."""

    first: int = -1
    last: int = -1
    qubits: set[int] = field(default_factory=set)
    zones: set[str] = field(default_factory=set)
    """Zones where a legal two-qubit gate of the layer ran."""

    def add(self, index: int, qubits: Sequence[int]) -> None:
        if self.first < 0:
            self.first = index
        self.last = index
        self.qubits.update(qubits)
