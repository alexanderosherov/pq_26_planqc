"""Local scoring of the pasteable `compile_circuit_cell.py` with the official `pushquantum_validator`.

Mirrors the official pipeline as closely as possible:
- the cell is loaded with only the template's names in scope (as in the workspace notebook);
- `compile_circuit(reference, spec)` is called with the validator's bundled spec and default `time_budget`
  (unless `--budget` is given), on the reference loaded the way the validator loads it;
- the output goes through the same JSON round-trip as the official sandbox (`_compile_child._serialize`
  -> `_sandbox.parse_output`);
- then `verify_movement`, `verify_equivalence` (QCEC up to 19 qubits, interaction check above) and
  `compute_metrics`, and the score via `score_metrics` against `naive_baseline.json`.

Usage:
    python challenge/test_cell.py                    # all 36 metric benchmarks, official score
    python challenge/test_cell.py --budget 5 -j 8    # faster: 5 s search per circuit, 8 processes
    python challenge/test_cell.py -k ihva_stagger    # only benchmarks whose path contains a substring
    python challenge/test_cell.py --sanity           # also the sanity benchmarks (validity only)
    python challenge/test_cell.py --rzz              # use_rzz=True (rzz in the entangling zone)
    python challenge/test_cell.py --no-equiv         # skip QCEC (movement + interaction checks only)
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import logging
import os
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Integral
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CELL = Path(__file__).with_name("compile_circuit_cell.py")
sys.path.insert(0, str(ROOT / "vendor"))

import pushquantum_validator as pv  # noqa: E402
from pushquantum_validator import verify as pvv  # noqa: E402
from pushquantum_validator._sandbox import CHILD, parse_output  # noqa: E402
from pushquantum_validator.spec import load_trusted_spec  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Instruction  # noqa: E402

Site = tuple[int, int]
ArchitectureSpec = str | Path | dict[str, Any]


# --- template definitions (as in the workspace notebook `readme.py`) ------------------------------------


class Architecture:
    def __init__(self, initial_loading: dict[int, Site], spec: ArchitectureSpec | None = None) -> None:
        self.spec = spec
        self.initial_loading = dict(initial_loading)


def as_lines(values: int | Sequence[int], name: str) -> tuple[int, ...]:
    items = [values] if isinstance(values, Integral) else list(values)
    result = tuple(int(v) for v in items)
    if not result or len(set(result)) != len(result):
        msg = f"invalid `{name}`: {items!r}"
        raise ValueError(msg)
    return result


def check_lines(start: tuple[int, ...], end: tuple[int, ...], name: str) -> None:
    if len(end) > len(start):
        msg = f"`new_{name}` has more lines than `{name}`"
        raise ValueError(msg)
    if start != tuple(range(start[0], start[0] + len(start))):
        msg = f"picked-up {name} {start} not contiguous"
        raise ValueError(msg)
    if any(b <= a for a, b in itertools.pairwise(end)):
        msg = f"target {name} {end} not ascending"
        raise ValueError(msg)


class Move(Instruction):
    def __init__(self, rows, cols, new_rows, new_cols, num_qubits: int) -> None:
        names = ("rows", "cols", "new_rows", "new_cols")
        lines = [as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
        check_lines(lines[0], lines[2], "rows")
        check_lines(lines[1], lines[3], "cols")
        super().__init__("move", num_qubits, 0, lines)

    def validate_parameter(self, parameter: object) -> tuple[int, ...]:
        return parameter

    def _define(self) -> None:
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


def load_cell():
    """`compile_circuit` from the cell file, with only the template's names in scope."""
    scope = {
        "QuantumCircuit": QuantumCircuit,
        "ArchitectureSpec": ArchitectureSpec,
        "Architecture": Architecture,
        "Move": Move,
    }
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), scope)
    return scope["compile_circuit"]


def load_serializer():
    """`_serialize` of the official compile child (it is a script, so load it from its file)."""
    spec = importlib.util.spec_from_file_location("pq_compile_child", CHILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._serialize


# --- one benchmark (runs in a worker process) ------------------------------------------------------------


def run_one(path: str, budget: float | None, check_equiv: bool, extra: dict[str, Any]) -> dict[str, Any]:
    logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)  # spacing warnings: counted, not printed
    path = Path(path)
    label = pvv.benchmark_label(path) or path.stem
    row: dict[str, Any] = {"label": label, "name": str(path.relative_to(path.parents[1])), "ok": False}
    try:
        reference = QuantumCircuit.from_qasm_file(str(path))
        row["qubits"] = reference.num_qubits
        compile_circuit = load_cell()
        kwargs = dict(extra) if budget is None else {**extra, "time_budget": budget}
        start = time.monotonic()
        output = compile_circuit(reference, load_trusted_spec(), **kwargs)
        row["seconds"] = time.monotonic() - start
        compiled, loading = parse_output(load_serializer()(output))
        movement = pv.verify_movement(compiled, loading)
        gates = pv.strip_moves(compiled)
        if check_equiv:
            eq = pv.verify_equivalence(compiled, reference, gates)
            row["equiv"] = "pairs only (>19q)" if eq is None else str(eq.equivalence).rsplit(".", 1)[-1]
        else:
            row["equiv"] = "skipped"
        metrics = pv.compute_metrics(compiled, movement, gates)
        row.update(
            ok=True,
            metrics=metrics,
            spacing_warnings=movement.spacing_warnings,
        )
    except pv.VerificationError as error:
        row["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:  # compiler crash
        row["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"
    return row


# --- report ------------------------------------------------------------------------------------------


def pct(new: float, old: float) -> str:
    return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"


def report(rows: list[dict[str, Any]], naive: dict[str, Any]) -> None:
    rows = sorted(rows, key=lambda r: r["label"])
    header = f"{'label':18} {'benchmark':40} {'q':>3} {'ok':>4} {'moves':>6} {'dist':>9} {'vs naive':>9} "
    header += f"{'depth':>6} {'vs naive':>9} {'warn':>5} {'s':>5}  equiv"
    print("\n" + header + "\n" + "-" * len(header))
    for r in rows:
        m, base = r.get("metrics"), naive.get(r["label"], {}).get("metrics")
        if m is None:
            print(f"{r['label']:18} {r['name']:40} {r.get('qubits', '?'):>3} FAIL  {r['error'].splitlines()[0][:110]}")
            continue
        dist_vs = pct(m.move_distance, base["move_distance"]) if base else "n/a"
        depth_vs = pct(m.depth, base["depth"]) if base else "n/a"
        print(
            f"{r['label']:18} {r['name']:40} {r['qubits']:>3} {'yes':>4} {m.num_moves:>6} {m.move_distance:>9.1f} "
            f"{dist_vs:>9} {m.depth:>6} {depth_vs:>9} {r['spacing_warnings']:>5} {r['seconds']:>5.1f}  {r['equiv']}"
        )

    # per-family totals (hea / ihva / qaoa ...) over the metric benchmarks
    metric = [r for r in rows if r["label"].startswith("metric_circuit_")]
    families: dict[str, list[dict[str, Any]]] = {}
    for r in metric:
        families.setdefault(r["name"].split("/")[0], []).append(r)
    if families:
        total_naive = sum(naive[r["label"]]["metrics"]["move_distance"] for r in metric)
        print(f"\n{'family':8} {'n':>3} {'distance':>10} {'naive':>10} {'change':>8} {'share of naive':>15} {'depth':>7} {'naive':>7} {'change':>8}")
        for fam, rs in sorted(families.items()):
            good = [r for r in rs if r.get("metrics")]
            d = sum(r["metrics"].move_distance for r in good)
            nd = sum(naive[r["label"]]["metrics"]["move_distance"] for r in good)
            dp = sum(r["metrics"].depth for r in good)
            ndp = sum(naive[r["label"]]["metrics"]["depth"] for r in good)
            share = sum(naive[r["label"]]["metrics"]["move_distance"] for r in rs) / total_naive
            print(f"{fam:8} {len(rs):>3} {d:>10.1f} {nd:>10.1f} {pct(d, nd):>8} {share:>15.1%} {dp:>7} {ndp:>7} {pct(dp, ndp):>8}")

    failed = [r["label"] for r in rows if not r["ok"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    if not metric:
        return
    complete = len(metric) == len(pvv.metric_benchmarks())
    if any(not r["ok"] for r in metric):
        print("SCORE: not computed, every metric benchmark must pass")
        return
    score = pvv.score_metrics({r["label"]: r["metrics"] for r in metric})
    d = sum(r["metrics"].move_distance for r in metric)
    nd = sum(naive[r["label"]]["metrics"]["move_distance"] for r in metric)
    dp = sum(r["metrics"].depth for r in metric)
    ndp = sum(naive[r["label"]]["metrics"]["depth"] for r in metric)
    kind = "SCORE" if complete else f"PARTIAL SCORE ({len(metric)}/{len(pvv.metric_benchmarks())} metric benchmarks)"
    print(
        f"{kind}: {score:.2f}  (distance {d:.1f} vs {nd:.1f}: {pct(d, nd)}; depth {dp} vs {ndp}: {pct(dp, ndp)}; "
        f"score = {pvv.SCORE_DISTANCE_WEIGHT} * dist% + {pvv.SCORE_DEPTH_WEIGHT} * depth%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=float, default=None, help="time_budget per circuit (default: the cell's).")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    parser.add_argument("-k", "--filter", default="", help="only benchmarks whose path contains this substring")
    parser.add_argument("--sanity", action="store_true", help="also run the sanity benchmarks")
    parser.add_argument("--rzz", action="store_true", help="compile with use_rzz=True (native rzz gates)")
    parser.add_argument("--no-equiv", action="store_true", help="skip the equivalence check")
    args = parser.parse_args()

    paths = list(pvv.metric_benchmarks()) + (list(pvv.sanity_benchmarks()) if args.sanity else [])
    paths = [p for p in paths if args.filter in str(p)]
    if not paths:
        sys.exit(f"no benchmark matches {args.filter!r}")
    print(f"{len(paths)} benchmarks, {args.jobs} processes, time_budget={args.budget or 'default'}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, str(p), args.budget, not args.no_equiv, {"use_rzz": True} if args.rzz else {}): p for p in paths}
        for future in as_completed(futures):
            r = future.result()
            status = f"distance {r['metrics'].move_distance:.1f}" if r.get("metrics") else "FAIL"
            print(f"  {r['label']:18} {r['name']:40} {status}", flush=True)
            rows.append(r)
    report(rows, pvv.naive_baseline())


if __name__ == "__main__":
    main()
