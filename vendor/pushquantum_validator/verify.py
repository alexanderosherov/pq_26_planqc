"""Verification of a compiled circuit: `verify_movement` (replay on the architecture) and
`verify_equivalence` (QCEC, moves stripped) must both pass; `verify_compilation` runs a compiler first."""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import functools
import itertools
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mqt import qcec
from mqt.qcec.pyqcec import EquivalenceCriterion
from qiskit import QuantumCircuit, transpile

from pushquantum_validator._sandbox import DEFAULT_COMPILER, run_compiler_process
from pushquantum_validator._validation import MOVE_NAME, MoveRecord, Site, ValidationArchitecture
from pushquantum_validator.errors import (
    KINDS,
    ArchitectureError,
    CompilerOutputError,
    NotEquivalentError,
    VerificationError,
)
from pushquantum_validator.spec import load_trusted_spec

logger = logging.getLogger(__name__)

MAX_EQUIVALENCE_QUBITS = 19
"""QCEC runs only up to this many qubits; from 20 qubits it can run for hours (its timeouts are best-effort)."""

_IDENTITIES = frozenset({MOVE_NAME, "barrier"})

EQUIVALENCE_TIMEOUT = 300
"""Seconds for the exact (alternating) equivalence check, used when ZX + simulation is inconclusive."""
_INCONCLUSIVE = frozenset({EquivalenceCriterion.no_information, EquivalenceCriterion.probably_not_equivalent})


def strip_moves(circuit: QuantumCircuit) -> QuantumCircuit:
    """A copy of `circuit` without moves and barriers (both act as identity)."""
    stripped = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name not in _IDENTITIES:
            stripped._append(instruction)  # already a valid instruction of a circuit with the same bits
    return stripped


@dataclass(frozen=True)
class MovementResult:
    trace: list[MoveRecord]
    log_file: Path

    @functools.cached_property
    def _step_times(self) -> dict[int, float]:
        times: dict[int, float] = {}
        for record in self.trace:
            times[record.step] = max(times.get(record.step, 0.0), record.time)
        return times

    @property
    def spacing_warnings(self) -> int:
        return sum(len(record.spacing_warnings) for record in self.trace)

    @property
    def distance(self) -> float:
        """Sum over moves of the longest distance travelled by one atom."""
        return sum(record.distance for record in self.trace)

    @property
    def time(self) -> float:
        """Sum over parallel steps of their slowest move's time."""
        return sum(self._step_times.values())

    @property
    def steps(self) -> int:
        return len(self._step_times)


@dataclass(frozen=True)
class Metrics:
    num_moves: int
    move_steps: int
    move_distance: float
    move_time: float
    depth: int
    """Depth without moves and barriers."""
    num_operations: int
    """Operations without moves and barriers."""

    def __str__(self) -> str:
        return (
            f"{self.num_moves} moves in {self.move_steps} parallel steps, distance {self.move_distance:.3f}, "
            f"time {self.move_time:.3f}, depth {self.depth}, {self.num_operations} operations"
        )


def compute_metrics(compiled: QuantumCircuit, movement: MovementResult, gates: QuantumCircuit | None = None) -> Metrics:
    """`gates`: `strip_moves(compiled)`, if already computed."""
    gates = strip_moves(compiled) if gates is None else gates
    return Metrics(len(movement.trace), movement.steps, movement.distance, movement.time, gates.depth(), gates.size())


def verify_movement(
    candidate: QuantumCircuit,
    loading: dict[int, Site] | None = None,
    log_file: str | Path | None = None,
    spec: dict[str, Any] | None = None,
) -> MovementResult:
    """Replays `candidate` on the zones of `spec` (default: the validator's) from the initial
    `loading` (default: qubit `i` at `(i, 0)`). Raises `ArchitectureError` (with `log_file`)."""
    if loading is None:
        loading = {q: (q, 0) for q in range(candidate.num_qubits)}
    replay = ValidationArchitecture.from_loading(loading, spec)
    trace = replay.run(candidate, log_file)
    return MovementResult(trace, replay.log_file)


def verify_equivalence(
    candidate: QuantumCircuit, reference: QuantumCircuit, gates: QuantumCircuit | None = None
) -> qcec.EquivalenceCheckingResults | None:
    """QCEC on `candidate` without moves vs `reference`, final measurements removed. `gates`:
    `candidate` already stripped. Raises `NotEquivalentError`.

    ZX + simulation first (fast). The ZX checker cannot handle arbitrary rotation angles and may
    wrongly report `probably_not_equivalent`, so an inconclusive verdict is re-checked with the
    alternating decision-diagram checker (exact, slower, at most `EQUIVALENCE_TIMEOUT` seconds).

    Above `MAX_EQUIVALENCE_QUBITS` qubits QCEC is skipped and `None` is returned, after a cheap
    necessary check: both circuits must entangle the same qubit pairs (`interaction_pairs`)."""
    gates = strip_moves(candidate) if gates is None else gates
    if max(candidate.num_qubits, reference.num_qubits) > MAX_EQUIVALENCE_QUBITS:
        _check_interactions(gates, reference)
        return None
    circuits = gates.remove_final_measurements(inplace=False), reference.remove_final_measurements(inplace=False)
    off = {"run_alternating_checker": False, "run_construction_checker": False, "run_zx_checker": False}
    with _quiet_stderr():  # QCEC prints notes on inconclusive verdicts straight to stderr
        result = qcec.verify(*circuits, **{**off, "run_zx_checker": True}, run_simulation_checker=True)
        if result.equivalence in _INCONCLUSIVE:
            result = qcec.verify(
                *circuits,
                **{**off, "run_alternating_checker": True},
                run_simulation_checker=True,
                timeout=EQUIVALENCE_TIMEOUT,
            )
    if not result.considered_equivalent():
        msg = f"Circuits are not equivalent (QCEC: {result.equivalence})."
        raise NotEquivalentError(msg)
    return result


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Silences file descriptor 2, which C++ extensions write to directly."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def interaction_pairs(circuit: QuantumCircuit) -> set[frozenset[int]]:
    """Every pair of qubits acted on together by a gate (moves and barriers ignored)."""
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    return {
        frozenset(pair)
        for instruction in circuit.data
        if len(instruction.qubits) > 1 and instruction.operation.name not in _IDENTITIES
        for pair in itertools.combinations([index[q] for q in instruction.qubits], 2)
    }


def _check_interactions(gates: QuantumCircuit, reference: QuantumCircuit) -> None:
    if gates.num_qubits != reference.num_qubits:
        msg = f"Circuits have {gates.num_qubits} and {reference.num_qubits} qubits."
        raise NotEquivalentError(msg)
    # Required: the pairs that still interact once gates that cancel (e.g. `cx rz(0) cx`) are removed;
    # allowed: every pair the reference acts on. So a correct optimisation is not rejected.
    compiled, allowed = interaction_pairs(gates), interaction_pairs(reference)
    required = interaction_pairs(transpile(reference, basis_gates=["rx", "rz", "cz"], optimization_level=1))
    missing, extra = required - compiled, compiled - allowed
    if missing or extra:
        missing, extra = sorted(map(sorted, missing)), sorted(map(sorted, extra))
        msg = (
            f"Circuits are not equivalent: the compiled circuit does not entangle the same qubit pairs "
            f"({len(missing)} missing, e.g. {missing[:5]}; {len(extra)} extra, e.g. {extra[:5]})."
        )
        raise NotEquivalentError(msg)


def run_compiler(
    reference: QuantumCircuit,
    compiler: str = DEFAULT_COMPILER,
    spec: dict[str, Any] | None = None,
    timeout: float = 300,
    memory_mb: int = 4096,
) -> tuple[QuantumCircuit, dict[int, Site]]:
    """Runs `compiler` (`module:function` or `file.py:function`, default: the template's) on `reference`
    and `spec` (default: the validator's) in a separate process.

    Returns the compiled circuit, rebuilt by the validator from the child's JSON, and the initial
    loading. Raises `CompilerOutputError` if the compiler fails, times out or returns bad output.
    """
    spec = load_trusted_spec() if spec is None else copy.deepcopy(spec)
    return run_compiler_process(reference, spec, compiler, timeout, memory_mb)


@dataclass(frozen=True)
class CompilationResult:
    compiled: QuantumCircuit
    loading: dict[int, Site]
    movement: MovementResult
    equivalence: qcec.EquivalenceCheckingResults | None
    """`None` if not checked (more than `MAX_EQUIVALENCE_QUBITS` qubits)."""
    metrics: Metrics


def verify_compilation(
    reference: QuantumCircuit,
    compiler: str = DEFAULT_COMPILER,
    spec: dict[str, Any] | None = None,
    log_file: str | Path | None = None,
    timeout: float = 300,
) -> CompilationResult:
    """`run_compiler`, then both checks; raises the first `VerificationError`."""
    compiled, loading = run_compiler(reference, compiler, spec, timeout)
    movement = verify_movement(compiled, loading, log_file, spec)
    gates = strip_moves(compiled)
    equivalence = verify_equivalence(compiled, reference, gates)
    return CompilationResult(compiled, loading, movement, equivalence, compute_metrics(compiled, movement, gates))


BENCHMARK_DIR = Path(__file__).with_name("sanity_benchmarks")
METRIC_BENCHMARK_DIR = Path(__file__).with_name("metric_benchmarks")
NAIVE_BASELINE = Path(__file__).with_name("naive_baseline.json")
"""Results of the naive example compiler (src/pushquantum_validator/examples/naive.py) on every sanity
and metric benchmark, keyed by `benchmark_label`; regenerate with `uv run python tests/update_naive_baseline.py`."""


def sanity_benchmarks() -> list[Path]:
    """The bundled sanity benchmarks (`sample_circuit_NN.qasm`, already anonymised), sorted."""
    return sorted(BENCHMARK_DIR.glob("*.qasm"))


REDUCED_HEA_COUNT = 2
"""How many `hea` circuits (the first, by sorted path: 12 qubits p1 and p3) the reduced metric set keeps."""


def metric_benchmarks(reduced: bool = False) -> list[Path]:
    """The bundled metric benchmarks (any depth of folders), sorted by their path in `METRIC_BENCHMARK_DIR`.
    `reduced` keeps only the `ihva` circuits and the first `REDUCED_HEA_COUNT` `hea` circuits."""
    paths = sorted(METRIC_BENCHMARK_DIR.rglob("*.qasm"))
    if reduced:
        hea = [p for p in paths if p.parent.name == "hea"][:REDUCED_HEA_COUNT]
        paths = [p for p in paths if p.parent.name == "ihva" or p in hea]
    return paths


BENCHMARK_SETS = ("sanity", "metric")
"""The benchmark sets `--benchmarks` can pick, in the order `all` runs them."""


def benchmarks(name: str, reduced: bool = False) -> list[Path]:
    """The benchmarks of the set `name` (one of `BENCHMARK_SETS`); `reduced` applies to the metric set
    (see `metric_benchmarks`)."""
    return sanity_benchmarks() if name == "sanity" else metric_benchmarks(reduced)


def benchmark_label(path: Path) -> str | None:
    """The anonymous name of a bundled benchmark, used in logs and the baseline: the file name of a
    sanity benchmark (`sample_circuit_NN`), `metric_circuit_NN` (1-based, by sorted path) of a metric
    benchmark, `None` for any other file. Independent of what `metric_benchmarks()` returns."""
    path = path.resolve()
    if path.parent == BENCHMARK_DIR.resolve():
        return path.stem
    metric = [p.resolve() for p in sorted(METRIC_BENCHMARK_DIR.rglob("*.qasm"))]
    return f"metric_circuit_{metric.index(path) + 1:02d}" if path in metric else None


def baseline_record(result: CompilationResult) -> dict[str, Any]:
    """What `NAIVE_BASELINE` stores for one benchmark."""
    metrics = _rounded(result.metrics)
    equivalence = "not checked" if result.equivalence is None else str(result.equivalence.equivalence)
    return {"metrics": metrics, "spacing_warnings": result.movement.spacing_warnings, "equivalence": equivalence}


def _rounded(metrics: Metrics) -> dict[str, float]:
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in dataclasses.asdict(metrics).items()}


def naive_baseline() -> dict[str, dict[str, Any]]:
    """`NAIVE_BASELINE`: `benchmark_label` -> `baseline_record` of the naive compiler."""
    return json.loads(NAIVE_BASELINE.read_text(encoding="utf-8"))


def compare_metrics(metrics: Metrics, baseline: dict[str, float]) -> list[tuple[str, str, str]]:
    """`(IMPROVED | WORSENED, metric, change %)` for every metric that differs from `baseline`.
    Every metric is a cost, so lower is better."""
    changes = []
    for name, new in _rounded(metrics).items():
        old = baseline[name]
        if new == old:
            continue
        if not old:
            percent = f"+{new:g} (from 0)"
        elif abs(change := (new - old) / old * 100) < 0.05:
            percent = f"{'-' if change < 0 else '+'}<0.1%"  # e.g. 2 moves fewer out of 40,000
        else:
            percent = f"{change:+.1f}%"
        changes.append(("IMPROVED" if new < old else "WORSENED", name, percent))
    return changes


SCORE_DISTANCE_WEIGHT = 0.8
SCORE_DEPTH_WEIGHT = 0.2


def score_compiler(compiler: str = DEFAULT_COMPILER, timeout: float = 300, reduced: bool = False) -> float:
    """% improvement of `compiler` over the naive compiler on the metric benchmarks: the improvements of
    the summed move distance and of the summed depth, each `(naive - new) / naive * 100` (positive is
    better), weighted by `SCORE_DISTANCE_WEIGHT` and `SCORE_DEPTH_WEIGHT`. `reduced` scores only the
    reduced metric set (see `metric_benchmarks`). Raises the first `VerificationError` if any compilation
    is invalid, with a note naming its benchmark."""
    results: dict[str, Metrics] = {}
    for path in metric_benchmarks(reduced):
        label = benchmark_label(path) or path.stem
        reference = QuantumCircuit.from_qasm_file(str(path))
        try:
            results[label] = verify_compilation(reference, compiler, timeout=timeout).metrics
        except VerificationError as error:
            error.add_note(f"benchmark: {label}")
            raise
    return score_metrics(results)


def score_metrics(metrics: Mapping[str, Metrics]) -> float:
    """The score of `score_compiler` from already computed metrics (`benchmark_label` -> `Metrics`, every
    metric benchmark); logs it (also kept by `--summary`)."""
    naive = naive_baseline()
    distance = depth = naive_distance = naive_depth = 0.0
    for label, result in metrics.items():
        if label not in naive:
            msg = f"{label} is missing from {NAIVE_BASELINE}."
            raise KeyError(msg)
        distance += result.move_distance
        depth += result.depth
        naive_distance += naive[label]["metrics"]["move_distance"]
        naive_depth += naive[label]["metrics"]["depth"]
    distance_improvement = (naive_distance - distance) / naive_distance * 100
    depth_improvement = (naive_depth - depth) / naive_depth * 100
    score = SCORE_DISTANCE_WEIGHT * distance_improvement + SCORE_DEPTH_WEIGHT * depth_improvement
    # Logged as changes vs naive, like `compare_metrics`: every metric is a cost, so negative is better.
    logger.info(
        "SCORE: %+.2f%% vs naive (%g x move distance %.3f vs %.3f: %+.2f%%; %g x depth %d vs %d: %+.2f%%)",
        -score,
        SCORE_DISTANCE_WEIGHT,
        distance,
        naive_distance,
        -distance_improvement,
        SCORE_DEPTH_WEIGHT,
        depth,
        naive_depth,
        -depth_improvement,
        extra=_SUMMARY,
    )
    return score


def main(argv: list[str] | None = None) -> int:
    """CLI: compile reference circuits with `--compiler` (see `run_compiler`) and run both checks.
    Without a reference, the bundled benchmarks of `--benchmarks` are checked (default: all)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "reference", nargs="?", help="OpenQASM 2 file of the reference circuit (default: the bundled benchmarks)."
    )
    parser.add_argument(
        "--benchmarks",
        choices=[*BENCHMARK_SETS, "all"],
        default="all",
        help="Which bundled benchmarks to check without a reference (default: all).",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        help="OpenQASM 2 file of the candidate circuit (default: template.compile_circuit(reference, spec)). "
        "QASM cannot express `move`, so only equivalence is checked for it.",
    )
    parser.add_argument(
        "--log",
        help="Also write everything logged (results, metrics, spacing warnings) to this file.",
    )
    parser.add_argument(
        "--compiler",
        default=DEFAULT_COMPILER,
        help="module:function or file.py:function (default: the template's).",
    )
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per compilation (default: 300).")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only log each benchmark's name and its metric changes vs the naive compiler "
        "(or FAILED): the output for participants.",
    )
    parser.add_argument(
        "--reduced-metrics",
        action="store_true",
        help="Only run and score the ihva metric benchmarks and the first two hea ones (12 qubits, p1 and p3).",
    )
    args = parser.parse_args(argv)

    package_logger = logging.getLogger("pushquantum_validator")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers[0].setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    if args.log:
        handlers.append(logging.FileHandler(args.log, mode="w", encoding="utf-8"))
        detailed = "%(levelname)s %(message)s" if args.summary else "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handlers[1].setFormatter(logging.Formatter(detailed))
    previous_level = package_logger.level
    package_logger.setLevel(logging.INFO)
    for handler in handlers:
        if args.summary:
            handler.addFilter(lambda record: getattr(record, "summary", False))
        package_logger.addHandler(handler)
    try:
        if args.reference:
            return 0 if _check(Path(args.reference), args.candidate, args.compiler, args.timeout)[0] else 1
        sets = list(BENCHMARK_SETS) if args.benchmarks == "all" else [args.benchmarks]
        results = [_run_benchmarks(name, args.compiler, args.timeout, args.reduced_metrics) for name in sets]
        return 1 if any(results) else 0
    finally:
        for handler in handlers:
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(previous_level)


_SUMMARY = {"summary": True}
"""`extra` of the log records kept by `--summary`."""


def _run_benchmarks(name: str, compiler: str, timeout: float, reduced: bool = False) -> int:
    """Checks every benchmark of the set `name` (see `BENCHMARK_SETS`, `reduced` as in `benchmarks`);
    1 if any failed, else 0."""
    failed = []
    metrics: dict[str, Metrics] = {}
    paths = benchmarks(name, reduced)
    for path in paths:
        label = benchmark_label(path) or path.stem
        logger.info("== %s", label, extra=_SUMMARY)
        passed, result = _check(path, None, compiler, timeout)
        if passed and result is not None:
            metrics[label] = result
        else:
            failed.append(label)
    passed = len(paths) - len(failed)
    if failed:
        logger.error("%d/%d %s benchmarks passed; failed: %s", passed, len(paths), name, ", ".join(failed))
        if name == "metric":
            logger.error("SCORE: not computed, every metric benchmark must pass", extra=_SUMMARY)
        return 1
    logger.info("%d/%d %s benchmarks passed", passed, len(paths), name)
    if name == "metric":
        score_metrics(metrics)
    return 0


MAX_REPORTED_COUNT = 10
"""Violation counts above this are reported as `>10`."""


def describe_violations(violations: Sequence[ArchitectureError]) -> list[str]:
    """One `<count> <kind>` per kind of violation, e.g. `3 illegal moves` or `>10 gates outside their
    zone`, in the order the kinds first occur."""
    counts = Counter(violation.kind for violation in violations)
    descriptions = []
    for kind, count in counts.items():
        singular, plural = KINDS.get(kind, (kind, kind))
        shown = str(count) if count <= MAX_REPORTED_COUNT else f">{MAX_REPORTED_COUNT}"
        descriptions.append(f"{shown} {singular if count == 1 else plural}")
    return descriptions


def _check(
    reference_path: Path, candidate_path: str | None, compiler: str, timeout: float
) -> tuple[bool, Metrics | None]:
    """Logs the result of both checks on one reference circuit; whether both passed, and the metrics
    (`None` without a movement replay). The candidate is read from `candidate_path` (QASM, equivalence
    only) or compiled with `compiler`."""
    reference = QuantumCircuit.from_qasm_file(str(reference_path))
    if candidate_path:
        candidate = QuantumCircuit.from_qasm_file(candidate_path)
    else:
        try:
            candidate, loading = run_compiler(reference, compiler, timeout=timeout)
        except CompilerOutputError as error:
            logger.error("FAIL compiler output: %s", error)
            logger.error("FAILED: compiler error", extra=_SUMMARY)
            return False, None

    passed = True
    metrics: Metrics | None = None
    reasons: list[str] = []  # why the compilation failed, for the summary
    changes: list[tuple[str, str, str]] | None = None  # vs the naive baseline, if there is one
    gates = strip_moves(candidate)
    if candidate_path:
        logger.info("SKIP movement: QASM candidate has no moves")
    else:
        try:
            movement = verify_movement(candidate, loading)
            logger.info(
                "PASS movement: %d moves replayed, %d spacing warning(s); replay log: %s",
                len(movement.trace),
                movement.spacing_warnings,
                movement.log_file,
            )
            metrics = compute_metrics(candidate, movement, gates)
            logger.info("METRICS: %s", metrics)
            label = benchmark_label(reference_path)
            baseline = naive_baseline().get(label) if label else None
            if baseline is not None:
                changes = compare_metrics(metrics, baseline["metrics"])
        except ArchitectureError as error:
            logger.error("FAIL movement: %s; replay log: %s", error, error.log_file)
            reasons += describe_violations(error.violations or (error,))
            passed = False
    try:
        result = verify_equivalence(candidate, reference, gates)
        if result is None:
            logger.info(
                "SKIP equivalence: more than %d qubits (same entangled qubit pairs as the reference)",
                MAX_EQUIVALENCE_QUBITS,
            )
        else:
            logger.info("PASS equivalence: %s", result.equivalence)
    except NotEquivalentError as error:
        logger.error("FAIL equivalence: %s", error)
        reasons.append("not equivalent to the input circuit")
        passed = False

    # The comparison only means something for a valid compilation.
    if not passed:
        logger.error("FAILED: %s", "; ".join(reasons), extra=_SUMMARY)
    elif changes is not None:
        for status, name, percent in changes:
            logger.info("%s %s %s vs naive", status, name, percent, extra=_SUMMARY)
        if not changes:
            logger.info("UNCHANGED: every metric equals the naive baseline", extra=_SUMMARY)
    return passed, metrics


if __name__ == "__main__":
    sys.exit(main())
