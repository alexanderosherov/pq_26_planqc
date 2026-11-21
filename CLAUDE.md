# pq_26_planqc

planqc PushQuantum Hackathon 2026: compile circuits for a zoned neutral-atom device (storage zone with
`rx`/`rz`, entangling zone with `cz`/`rzz` on interaction pairs, AOD `Move` instructions between them).
Problem statement: `challenge/PROBLEM.md`. Score = 0.8 × % less move distance + 0.2 × % less depth than
the naive compiler, over 36 metric benchmarks.

## Layout

- `challenge/compile_circuit_cell.py` — the submission: one self-contained `def compile_circuit(...)`
  pasted into the `compile_circuit` cell of the aqora workspace notebook. It may only use the template's
  names (`QuantumCircuit`, `ArchitectureSpec`, `Move`, `Architecture`) and import marimo, qiskit, numpy,
  scipy (no network, no other packages). Options: `time_budget` (seconds, default 20), `use_rzz`.
- `challenge/test_cell.py` — local scoring with the official validator: loads the cell like the
  scorer, runs `vendor/pushquantum_validator` (movement replay, QCEC equivalence, metrics, score).
- `challenge/viewer.py` — the visualizer (below).
- `vendor/pushquantum_validator/` — official validator, metric/sanity benchmarks, naive baseline.
- `vendor/readme.py` — the workspace template notebook.

Use the project venv: `.venv/bin/python` (deps in `requirements.txt`).

## Visualizer: `challenge/viewer.py`

Compiles one circuit with `compile_circuit_cell.py`, runs the result through the official validator
(same JSON round-trip, `verify_movement`, `compute_metrics` as the scorer) and writes a self-contained
HTML page (no libraries, no network) to `challenge/viewer_out/<circuit>.html`.

```bash
.venv/bin/python challenge/viewer.py 12q_qaoa_p1                 # substring of a metric benchmark name
.venv/bin/python challenge/viewer.py 50q_ihva_tree_p3 --budget 20 # compile time budget (default 5 s)
.venv/bin/python challenge/viewer.py 12q_qaoa_p1 --rzz            # compile with use_rzz=True
.venv/bin/python challenge/viewer.py path/to/circuit.qasm         # any OpenQASM 2 file
open challenge/viewer_out/bus_stop_12q_qaoa_p1.html               # macOS; any browser works
```

Benchmark names (in `vendor/pushquantum_validator/metric_benchmarks/{hea,ihva,qaoa}/`):
`bus_stop_{12,20,50}q_hea_linear_p{1,3}`, `..._ihva_{stagger,tree,tree_warm}_p{1,3}`,
`..._qaoa{,_warm}_p{1,3}`. The first file whose name contains the substring is used.

If the validator rejects the compilation, the script exits with the error and writes no page.

### What the page shows

- Storage zone (bottom) and entangling zone (top) with every trap site; atoms numbered by qubit.
- **Moves**: atoms slide to their new sites (teal while carried) inside a dashed AOD pick-up box; the
  side panel lists picked/target rows × cols, qubits, the move's distance (validator's number), its
  parallel step, spacing warnings and the running distance total.
- **Gates**: rings on atoms for single-qubit gates (purple `rz`, blue `rx`, angles in the panel);
  `cz`/`rzz` pulses draw a line per interacting pair and highlight the entangling zone.
- Header: moves, total move distance, depth without moves, cz count, rounds, parallel move steps.
- "Move distance per round" bar chart and "Most expensive moves" list — click to jump there.

Controls: Play/Pause (Space), step (← →), Home/End, scrubber, speed 0.5×–16×, "moves only" playback.
The URL hash `#step=N` opens a given step (shareable / useful for screenshots).

### Typical use

Find where distance goes: open a page, check the per-round chart and the most expensive moves, jump
there and look at which atoms travel and why (far homes, pairs split across rows, single-atom moves
that could be merged into one rectangle move).

### Changing it

`viewer.py` = Python side (`build_trace` turns the compiled circuit + validator trace into JSON steps:
`move` / `cz` / `gates`) + `TEMPLATE` (the HTML/CSS/JS page, data injected at `__DATA__`). It depends on
`test_cell.py` for `load_cell`, `load_serializer`, `parse_output`, `pv`, `pvv`, `load_trusted_spec`; keep
those names exported when editing `test_cell.py`. Check a page headlessly with
`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=out.png "file://…/page.html#step=3"`.
