"""Step-through HTML viewer for a compilation of `compile_circuit_cell.py`.

Compiles one circuit with the cell, runs the result through the official validator in
`vendor/pushquantum_validator` (same serialisation, movement replay and metrics as the scorer), and
writes a self-contained HTML page (no libraries, no network) that animates every move and highlights
every gate and cz pulse. Move distances and parallel steps are the validator's own numbers.

Usage:
  python viewer.py 12q_qaoa_p1                       # substring of a metric benchmark filename
  python viewer.py path/to/circuit.qasm --budget 5   # any OpenQASM 2 file
  python viewer.py 12q_qaoa_p1 --rzz                 # compile with native rzz gates
Then open the printed .html file in a browser.
"""

import argparse
import inspect
import json
import logging
from pathlib import Path

from qiskit import QuantumCircuit

from test_cell import load_cell, load_serializer, parse_output, pv, pvv, load_trusted_spec


def pick_circuit(selector: str) -> tuple[str, QuantumCircuit]:
    path = Path(selector)
    if path.suffix == ".qasm" and path.exists():
        return path.stem, QuantumCircuit.from_qasm_file(str(path))
    paths = sorted(pvv.metric_benchmarks(), key=lambda p: p.name)
    matches = [p for p in paths if selector in p.name]
    if not matches:
        msg = f"No metric benchmark matches {selector!r}; options: {[p.stem for p in paths]}"
        raise SystemExit(msg)
    return matches[0].stem, QuantumCircuit.from_qasm_file(str(matches[0]))


def geometry(spec: dict):
    """Zones and a site -> physical position function from an architecture spec."""
    zones = []
    for name, z in spec["zones"].items():
        pairs = z.get("pairs") or {}
        zones.append(
            {
                "name": name,
                "x": z["x"],
                "y": z["y"],
                "ops": sorted(z.get("operations", ())),
                "spacing": float(z.get("spacing", 1.0)),
                "pair": float(pairs["distance"]) if pairs.get("axis") == "x" else None,
            }
        )

    def phys(site):
        x, y = site
        z = next(z for z in zones if z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1])
        if z["pair"] is not None:
            k, r = divmod(x - z["x"][0], 2)
            px = z["x"][0] * z["spacing"] + 2 * k * z["spacing"] + r * z["pair"]
        else:
            px = x * z["spacing"]
        return [round(px, 3), round(y * z["spacing"], 3)]

    return zones, phys


def build_trace(name, compiled, loading, movement, metrics, spec) -> dict:
    zones, phys = geometry(spec)
    sites, zone_info = [], []
    for z in zones:
        kind = "ent" if "cz" in z["ops"] else "storage"
        zone_info.append(
            {"name": z["name"], "kind": kind, "lo": phys((z["x"][0], z["y"][0])), "hi": phys((z["x"][1], z["y"][1])), "ops": z["ops"]}
        )
        for y in range(z["y"][0], z["y"][1] + 1):
            for x in range(z["x"][0], z["x"][1] + 1):
                sites.append([*phys((x, y)), kind == "ent"])

    records = {r.index: r for r in movement.trace}
    init = [phys(loading[q]) for q in range(compiled.num_qubits)]
    steps: list[dict] = []
    round_no, in_round, total = 0, False, 0.0
    for index, inst in enumerate(compiled.data):
        op = inst.operation.name
        qs = [compiled.find_bit(b).index for b in inst.qubits]
        if op == "move":
            if not in_round:
                round_no += 1
                in_round = True
            rec = records[index]
            total += rec.distance
            rows, cols, new_rows, new_cols = (list(v) for v in inst.operation.params)
            steps.append(
                {
                    "t": "move",
                    "q": qs,
                    "to": [phys(rec.positions[q]) for q in qs],
                    "d": round(rec.distance, 3),
                    "total": round(total, 3),
                    "round": round_no,
                    "par": rec.step,
                    "warn": len(rec.spacing_warnings),
                    "src": [rows, cols],
                    "dst": [new_rows, new_cols],
                }
            )
        elif op in ("cz", "rzz"):  # two-qubit gates in the entangling zone, drawn as one pulse
            if steps and steps[-1]["t"] == "cz":
                steps[-1]["pairs"].append(qs)
            else:
                steps.append({"t": "cz", "pairs": [qs], "round": round_no, "total": round(total, 3)})
        elif op in ("measure", "barrier"):
            continue
        else:
            in_round = False
            angle = float(inst.operation.params[0]) if inst.operation.params else 0.0
            gate = {"n": op, "q": qs[0] if len(qs) == 1 else qs, "a": round(angle, 4)}
            if steps and steps[-1]["t"] == "gates":
                steps[-1]["gates"].append(gate)
            else:
                steps.append({"t": "gates", "gates": [gate], "round": round_no, "total": round(total, 3)})
    ops = compiled.count_ops()
    return {
        "name": name,
        "n": compiled.num_qubits,
        "sites": sites,
        "zones": zone_info,
        "init": init,
        "steps": steps,
        "stats": {
            "valid": True,
            "errors": [],
            "moves": metrics.num_moves,
            "distance": round(metrics.move_distance, 2),
            "depth": metrics.depth,
            "cz": ops.get("cz", 0) + ops.get("rzz", 0),
            "gates1q": ops.get("rx", 0) + ops.get("rz", 0),
            "rounds": round_no,
            "parsteps": metrics.move_steps,
        },
    }


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compilation Viewer</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1d2330; --muted: #667085; --line: #d9dde5;
  --storage: #e8f0fb; --storage-edge: #9db7dd; --ent: #fdeee4; --ent-edge: #e6946a;
  --trap: #b8bfcc; --atom: #3d6fb6; --atom-ink: #ffffff; --moving: #0f9d8a; --beam-row: rgba(15,157,138,.16); --beam-col: rgba(15,157,138,.16);
  --rz: #7b61ff; --rx: #2aa3d8; --cz: #e2572b; --accent: #3d6fb6; --bar: #e2572b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12151b; --panel: #1b2029; --ink: #e6e9ef; --muted: #98a2b3; --line: #2c3340;
    --storage: #18263a; --storage-edge: #3f5f8c; --ent: #33221a; --ent-edge: #b8663e;
    --trap: #4a5263; --atom: #6c9be0; --atom-ink: #0e1117; --moving: #35c9b2; --beam-row: rgba(53,201,178,.17); --beam-col: rgba(53,201,178,.17);
    --rz: #a593ff; --rx: #5cc4f0; --cz: #ff7a4d; --accent: #6c9be0; --bar: #ff7a4d;
  }
}
:root[data-theme="dark"] {
  --bg: #12151b; --panel: #1b2029; --ink: #e6e9ef; --muted: #98a2b3; --line: #2c3340;
  --storage: #18263a; --storage-edge: #3f5f8c; --ent: #33221a; --ent-edge: #b8663e;
  --trap: #4a5263; --atom: #6c9be0; --atom-ink: #0e1117; --moving: #35c9b2; --beam-row: rgba(53,201,178,.17); --beam-col: rgba(53,201,178,.17);
  --rz: #a593ff; --rx: #5cc4f0; --cz: #ff7a4d; --accent: #6c9be0; --bar: #ff7a4d;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { padding: 16px 20px 8px; }
h1 { font-size: 18px; margin: 0 0 6px; }
.stats { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); }
.stats b { color: var(--ink); font-variant-numeric: tabular-nums; }
.bad { color: var(--cz); font-weight: 600; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 16px; padding: 8px 20px 20px; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; padding: 8px 16px 16px; } header { padding: 16px 16px 8px; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
#stage { width: 100%; height: auto; display: block; max-height: 78vh; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }
button, select { background: var(--panel); color: var(--ink); border: 1px solid var(--line);
  border-radius: 7px; padding: 5px 10px; font: inherit; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: var(--atom-ink); border-color: var(--accent); min-width: 64px; }
#scrub { flex: 1 1 200px; accent-color: var(--accent); }
.step-label { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 110px; text-align: right; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 8px; }
.side { display: flex; flex-direction: column; gap: 12px; }
#now .kind { font-weight: 600; }
#now .detail { color: var(--muted); word-break: break-word; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin-top: 6px; font-variant-numeric: tabular-nums; }
.kv span:nth-child(odd) { color: var(--muted); }
#top { list-style: none; margin: 0; padding: 0; max-height: 260px; overflow: auto; }
#top li { display: grid; grid-template-columns: 52px 1fr 54px; gap: 8px; align-items: center;
  padding: 3px 4px; border-radius: 6px; cursor: pointer; font-variant-numeric: tabular-nums; }
#top li:hover { background: var(--bg); }
#top .bar { height: 8px; border-radius: 4px; background: var(--bar); opacity: .8; }
#top .num { text-align: right; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted); font-size: 12px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
#costchart { width: 100%; height: 70px; display: block; cursor: pointer; }
.hint { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="stats" id="stats"></div>
</header>
<main>
  <section class="card">
    <svg id="stage" role="img" aria-label="Atom positions"></svg>
    <div class="controls">
      <button id="first" title="Start (Home)">⏮</button>
      <button id="back" title="Previous step (←)">◀</button>
      <button id="play" class="primary" title="Play / pause (Space)">Play</button>
      <button id="fwd" title="Next step (→)">▶</button>
      <button id="last" title="End (End)">⏭</button>
      <select id="speed" title="Speed">
        <option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option>
        <option value="4">4×</option><option value="8">8×</option><option value="16">16×</option>
      </select>
      <select id="filter" title="Play through">
        <option value="all">all steps</option><option value="move">moves only</option>
      </select>
      <label class="hint" title="Show the AOD row/column beams of a move (L)"><input id="beams" type="checkbox" checked> AOD lines</label>
      <input id="scrub" type="range" min="0" value="0">
      <span class="step-label" id="steplabel"></span>
    </div>
    <div class="legend" style="margin-top:8px">
      <span><span class="dot" style="background:var(--atom)"></span>atom</span>
      <span><span class="dot" style="background:var(--moving)"></span>moving</span>
      <span><span class="dot" style="background:var(--rz)"></span>rz</span>
      <span><span class="dot" style="background:var(--rx)"></span>rx</span>
      <span><span class="dot" style="background:var(--cz)"></span>cz pair</span>
      <span class="hint">Space play · ← → step · L AOD lines · dashed box = AOD pick-up</span>
    </div>
  </section>
  <aside class="side">
    <div class="card" id="now">
      <h2>Current step</h2>
      <div class="kind" id="kind"></div>
      <div class="detail" id="detail"></div>
      <div class="kv">
        <span>round</span><span id="round"></span>
        <span>this move</span><span id="thisd"></span>
        <span>distance so far</span><span id="sofar"></span>
      </div>
    </div>
    <div class="card">
      <h2>Move distance per round</h2>
      <svg id="costchart" preserveAspectRatio="none"></svg>
      <div class="hint">Click a bar to jump to that round.</div>
    </div>
    <div class="card">
      <h2>Most expensive moves</h2>
      <ul id="top"></ul>
    </div>
  </aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const S = D.steps;
const moveNo = new Map(); S.forEach((s, i) => { if (s.t === "move") moveNo.set(i, moveNo.size + 1); });
const PX = 46, R = 0.17, PAD = 0.7;
const xs = D.sites.map(s => s[0]), ys = D.sites.map(s => s[1]);
const minX = Math.min(...xs) - PAD, maxX = Math.max(...xs) + PAD;
const minY = Math.min(...ys) - PAD, maxY = Math.max(...ys) + PAD;
const sx = x => (x - minX) * PX, sy = y => (maxY - y) * PX;  // entangling zone (larger y) on top
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const stage = document.getElementById("stage");
stage.setAttribute("viewBox", `0 0 ${(maxX - minX) * PX} ${(maxY - minY) * PX}`);

// static layers
const gZones = el("g", {}, stage), gTraps = el("g", {}, stage), gAod = el("g", {}, stage),
      gLinks = el("g", {}, stage), gAtoms = el("g", {}, stage);
for (const z of D.zones) {
  // every zone box spans the full array width (paired entangling sites end at x = 8.3, storage at 9)
  const x0 = Math.min(...xs) - 0.45, x1 = Math.max(...xs) + 0.45;
  const y0 = Math.min(z.lo[1], z.hi[1]) - 0.45, y1 = Math.max(z.lo[1], z.hi[1]) + 0.45;
  const ent = z.kind === "ent";
  const r = el("rect", { x: sx(x0), y: sy(y1), width: (x1 - x0) * PX, height: (y1 - y0) * PX, rx: 10,
    fill: ent ? "var(--ent)" : "var(--storage)", stroke: ent ? "var(--ent-edge)" : "var(--storage-edge)",
    "stroke-width": 1.5 }, gZones);
  if (ent) r.id = "entzone";
  el("text", { x: sx(x0) + 8, y: sy(y0) - 6, fill: "var(--muted)", "font-size": 12 }, gZones).textContent =
    `${z.name} (${z.ops.join(", ")})`;
}
for (const [x, y] of D.sites) el("circle", { cx: sx(x), cy: sy(y), r: 3, fill: "var(--trap)" }, gTraps);

const atoms = [];
for (let q = 0; q < D.n; q++) {
  const g = el("g", {}, gAtoms);
  const ring = el("circle", { r: R * PX + 5, fill: "none", "stroke-width": 3, stroke: "transparent" }, g);
  const c = el("circle", { r: R * PX, fill: "var(--atom)" }, g);
  const t = el("text", { "text-anchor": "middle", "dominant-baseline": "central", "font-size": 10,
    "font-weight": 600, fill: "var(--atom-ink)" }, g);
  t.textContent = q;
  const title = el("title", {}, g); title.textContent = `q${q}`;
  atoms.push({ g, c, ring });
}

// precomputed positions before each step (per view)
function computeStates(list) {
  const out = [D.init.map(p => p.slice())];
  for (const s of list) {
    const next = out[out.length - 1].map(p => p.slice());
    if (s.t === "move") s.q.forEach((q, i) => { next[q] = s.to[i].slice(); });
    out.push(next);
  }
  return out;
}
const states = computeStates(S);

// header
document.getElementById("title").textContent = D.name;
const st = D.stats;
document.getElementById("stats").innerHTML = [
  `<span>valid <b class="${st.valid ? "" : "bad"}">${st.valid ? "yes" : "NO"}</b></span>`,
  `<span>qubits <b>${D.n}</b></span>`, `<span>moves <b>${st.moves}</b></span>`,
  `<span>move distance <b>${st.distance}</b></span>`, `<span>depth (no moves) <b>${st.depth}</b></span>`,
  `<span>cz <b>${st.cz}</b></span>`, `<span>1q gates <b>${st.gates1q}</b></span>`,
  `<span>rounds <b>${st.rounds}</b></span>`, `<span>parallel move steps <b>${st.parsteps}</b></span>`,
  ...(st.errors.length ? [`<span class="bad">${st.errors.join("; ")}</span>`] : []),
].join("");

// cost per round chart
const roundCost = new Map(), roundFirst = new Map();
S.forEach((s, i) => { if (s.t === "move") { roundCost.set(s.round, (roundCost.get(s.round) || 0) + s.d);
  if (!roundFirst.has(s.round)) roundFirst.set(s.round, i); } });
const chart = document.getElementById("costchart");
const rounds = [...roundCost.keys()].sort((a, b) => a - b);
const maxCost = Math.max(1e-9, ...roundCost.values());
chart.setAttribute("viewBox", `0 0 ${Math.max(1, rounds.length)} 100`);
const roundBars = new Map();
rounds.forEach((r, i) => {
  const h = 100 * roundCost.get(r) / maxCost;
  const b = el("rect", { x: i + 0.1, y: 100 - h, width: 0.8, height: h, fill: "var(--bar)", opacity: 0.55 }, chart);
  el("title", {}, b).textContent = `round ${r}: ${roundCost.get(r).toFixed(2)}`;
  b.addEventListener("click", () => go(roundFirst.get(r)));
  roundBars.set(r, b);
});

// top moves
const moveIdx = S.map((s, i) => i).filter(i => S[i].t === "move");
const topList = moveIdx.slice().sort((a, b) => S[b].d - S[a].d).slice(0, 15);
const maxD = topList.length ? S[topList[0]].d : 1;
const ul = document.getElementById("top");
for (const i of topList) {
  const li = document.createElement("li");
  li.innerHTML = `<span>#${moveIdx.indexOf(i) + 1}</span><span class="bar" style="width:${(100 * S[i].d / maxD).toFixed(0)}%"></span><span class="num">${S[i].d.toFixed(2)}</span>`;
  li.title = `${S[i].q.length} atoms, round ${S[i].round}`;
  li.addEventListener("click", () => go(i));
  ul.appendChild(li);
}

// rendering
let cur = 0, playing = false, anim = null;
const scrub = document.getElementById("scrub");
scrub.max = S.length;
function place(pos) { atoms.forEach((a, q) => a.g.setAttribute("transform", `translate(${sx(pos[q][0])},${sy(pos[q][1])})`)); }
function clearMarks() {
  atoms.forEach(a => { a.ring.setAttribute("stroke", "transparent"); a.c.setAttribute("fill", "var(--atom)"); });
  gLinks.replaceChildren(); gAod.replaceChildren();
  document.getElementById("entzone")?.setAttribute("stroke-width", 1.5);
  roundBars.forEach(b => b.setAttribute("opacity", 0.55));
}
function aodBox(pos, s) {  // dashed pick-up box + (optional) AOD row/column beams through the carried atoms
  const px = s.q.map(q => pos[q][0]), py = s.q.map(q => pos[q][1]);
  const x0 = Math.min(...px) - 0.35, x1 = Math.max(...px) + 0.35, y0 = Math.min(...py) - 0.35, y1 = Math.max(...py) + 0.35;
  gAod.replaceChildren();
  if (document.getElementById("beams").checked) {
    const W = (maxX - minX) * PX, H = (maxY - minY) * PX;
    // one band per axis, as wide as the box: spans every picked row / column
    el("rect", { x: 0, y: sy(y1), width: W, height: (y1 - y0) * PX, fill: "var(--beam-row)" }, gAod);
    el("rect", { x: sx(x0), y: 0, width: (x1 - x0) * PX, height: H, fill: "var(--beam-col)" }, gAod);
  }
  el("rect", { x: sx(x0), y: sy(y1), width: (x1 - x0) * PX, height: (y1 - y0) * PX, fill: "none",
    stroke: "var(--moving)", "stroke-width": 1.5, "stroke-dasharray": "5 4", rx: 6 }, gAod);
}
function redraw() { if (!playing) show(cur); }
function describe(i) {
  const k = document.getElementById("kind"), d = document.getElementById("detail");
  const rnd = document.getElementById("round"), th = document.getElementById("thisd"), so = document.getElementById("sofar");
  if (i >= S.length) { k.textContent = "End"; d.textContent = "All steps done."; rnd.textContent = "–"; th.textContent = "–";
    so.textContent = st.distance; return; }
  const s = S[i];
  rnd.textContent = s.round || "–"; so.textContent = (s.t === "move" ? s.total : s.total).toFixed(2);
  if (s.t === "move") {
    k.textContent = `Move #${moveNo.get(i)} · ${s.q.length} atom${s.q.length > 1 ? "s" : ""}`;
    d.textContent = `rows ${s.src[0].join(",")} × cols ${s.src[1].join(",")} → rows ${s.dst[0].join(",")} × cols ${s.dst[1].join(",")} · qubits ${s.q.join(", ")}`;
    th.textContent = `${s.d.toFixed(3)}${s.warn ? " · spacing warning" : ""}`;
  } else if (s.t === "cz") {
    k.textContent = `CZ pulse · ${s.pairs.length} pair${s.pairs.length > 1 ? "s" : ""}`;
    d.textContent = s.pairs.map(p => `(${p[0]},${p[1]})`).join(" ");
    th.textContent = "–";
  } else {
    k.textContent = `Single-qubit gates · ${s.gates.length}`;
    d.textContent = s.gates.slice(0, 40).map(g => `${g.n}(${g.a}) q${g.q}`).join(" · ") + (s.gates.length > 40 ? " …" : "");
    th.textContent = "–";
  }
}
function mark(i, pos) {
  clearMarks();
  if (i >= S.length) return;
  const s = S[i];
  if (s.round) roundBars.get(s.round)?.setAttribute("opacity", 1);
  if (s.t === "move") { s.q.forEach(q => atoms[q].c.setAttribute("fill", "var(--moving)")); aodBox(pos, s); }
  else if (s.t === "cz") {
    document.getElementById("entzone")?.setAttribute("stroke-width", 4);
    for (const [a, b] of s.pairs) {
      el("line", { x1: sx(pos[a][0]), y1: sy(pos[a][1]), x2: sx(pos[b][0]), y2: sy(pos[b][1]),
        stroke: "var(--cz)", "stroke-width": 6, "stroke-linecap": "round" }, gLinks);
      atoms[a].ring.setAttribute("stroke", "var(--cz)"); atoms[b].ring.setAttribute("stroke", "var(--cz)");
    }
  } else for (const g of s.gates) atoms[g.q].ring.setAttribute("stroke", g.n === "rz" ? "var(--rz)" : "var(--rx)");
}
function show(i) {
  cur = Math.max(0, Math.min(S.length, i));
  scrub.value = cur;
  history.replaceState(null, "", `#step=${cur}`);
  document.getElementById("steplabel").textContent = `step ${cur} / ${S.length}`;
  place(states[cur]); mark(cur, states[cur]); describe(cur);
}
function go(i) { stop(); show(i); }
const speed = () => parseFloat(document.getElementById("speed").value);
const movesOnly = () => document.getElementById("filter").value === "move";
function stepDuration(s) { return (s.t === "move" ? 700 : s.t === "cz" ? 450 : 300) / speed(); }
function advance() {
  if (!playing) return;
  if (cur >= S.length) { stop(); return; }
  if (movesOnly() && S[cur].t !== "move") { show(cur + 1); requestAnimationFrame(advance); return; }
  const s = S[cur], from = states[cur], to = states[cur + 1], dur = stepDuration(s), t0 = performance.now();
  mark(cur, from); describe(cur);
  const tick = now => {
    if (!playing) return;
    const f = Math.min(1, (now - t0) / dur), e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
    if (s.t === "move") {
      const pos = from.map((p, q) => [p[0] + (to[q][0] - p[0]) * e, p[1] + (to[q][1] - p[1]) * e]);
      place(pos); aodBox(pos, s);
    }
    if (f < 1) anim = requestAnimationFrame(tick);
    else { show(cur + 1); anim = requestAnimationFrame(advance); }
  };
  anim = requestAnimationFrame(tick);
}
function play() { if (cur >= S.length) show(0); playing = true; document.getElementById("play").textContent = "Pause"; advance(); }
function stop() { playing = false; if (anim) cancelAnimationFrame(anim); document.getElementById("play").textContent = "Play"; }
document.getElementById("play").onclick = () => (playing ? stop() : play());
document.getElementById("fwd").onclick = () => go(cur + 1);
document.getElementById("back").onclick = () => go(cur - 1);
document.getElementById("first").onclick = () => go(0);
document.getElementById("last").onclick = () => go(S.length);
scrub.oninput = () => go(parseInt(scrub.value, 10));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
  else if (e.key === "ArrowRight") go(cur + 1);
  else if (e.key === "ArrowLeft") go(cur - 1);
  else if (e.key === "Home") go(0);
  else if (e.key === "End") go(S.length);
  else if (e.key === "l" || e.key === "L") { const b = document.getElementById("beams"); b.checked = !b.checked; redraw(); }
});
document.getElementById("beams").onchange = redraw;
const hashStep = parseInt((location.hash.match(/step=(\d+)/) || [])[1], 10);
show(Number.isFinite(hashStep) ? hashStep : 0);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("circuit", help="substring of a metric benchmark filename, or a .qasm path")
    parser.add_argument("--budget", type=float, default=5.0, help="compile_circuit time budget in seconds (default 5)")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("viewer_out"))
    parser.add_argument("--rzz", action="store_true", help="compile with use_rzz=True (native rzz gates)")
    parser.add_argument("--cell", type=Path, default=None, help="compile_circuit file to use (default compile_circuit_cell.py)")
    args = parser.parse_args()
    logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)

    name, circuit = pick_circuit(args.circuit)
    spec = load_trusted_spec()
    compile_fn = load_cell(args.cell)
    params = inspect.signature(compile_fn).parameters
    options = {"use_rzz": True} if args.rzz else {}
    if "time_budget" in params:
        options["time_budget"] = args.budget
    output = compile_fn(circuit, spec, **options)
    compiled, loading = parse_output(load_serializer()(output))
    try:
        movement = pv.verify_movement(compiled, loading)
    except pv.VerificationError as error:
        raise SystemExit(f"{name}: INVALID compilation, not rendered: {error}") from error
    metrics = pv.compute_metrics(compiled, movement, pv.strip_moves(compiled))
    trace = build_trace(name, compiled, loading, movement, metrics, spec)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (f"{name}__{args.cell.stem}.html" if args.cell else f"{name}.html")
    data = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    print(f"{name}: valid movement | {metrics}")
    print(f"open {path.resolve()}")


if __name__ == "__main__":
    main()
