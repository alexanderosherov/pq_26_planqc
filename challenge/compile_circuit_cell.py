def compile_circuit(
    circuit: QuantumCircuit,
    architecture_spec: ArchitectureSpec,
    time_budget: float = 20.0,
    use_rzz: bool = False,
    reuse: bool = True,
    proxy_share: float = 0.8,
) -> tuple[QuantumCircuit, Architecture]:
    """Returns the compiled circuit and the `Architecture` (zones of `architecture_spec`, your initial loading).

    Self-contained: relies only on `QuantumCircuit`, `Move` and `Architecture` from the template.
    1. transpile to rx/rz/cz (or rx/rz/rzz with `use_rzz`) at optimization levels 0-3, keeping the candidate
       with the fewest entangling rounds (then the smallest depth); candidates whose transpilation left a
       qubit permutation (elided swaps) are rejected; final measurements are re-appended;
    2. rz gates are pushed through the diagonal two-qubit gates into the next rx block, so that qubits have
       fewer single-qubit gates between entangling rounds;
    3. cut into ASAP blocks of single-qubit gates and entangling rounds;
    4. fixed homes in the storage rows nearest the entangling zone (single-qubit gates run there);
    5. per round: one global slot assignment favouring a rigid row shift, transfers packed into as few valid
       AOD rectangle moves as possible (no ghosts, order preserved). With `reuse`, an atom with no
       single-qubit gate before its next entangling gate stays in the entangling zone (alone on its
       interaction pair, so later pulses act on it as identity); atoms with nothing left stay until the
       final measurement (allowed in any zone);
    6. layout search: a fast stateless cost first, then the exact sequential cost, both with early stop.
    """
    import itertools
    import json
    import math
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveBarriers
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    start_time = time.monotonic()

    # --- geometry -----------------------------------------------------------------------------------
    if isinstance(architecture_spec, dict):
        spec = architecture_spec
    else:
        spec = json.loads(Path(architecture_spec).read_text(encoding="utf-8"))
    zones = []
    for zname, z in spec["zones"].items():
        pairs_spec = z.get("pairs")
        zones.append(
            {
                "name": zname,
                "x": (int(z["x"][0]), int(z["x"][1])),
                "y": (int(z["y"][0]), int(z["y"][1])),
                "ops": set(z.get("operations", ())),
                "spacing": float(z.get("spacing", 1.0)),
                "pair_axis": pairs_spec["axis"] if pairs_spec else None,
                "pair_distance": float(pairs_spec["distance"]) if pairs_spec else 0.0,
            }
        )
    storage = next(z for z in zones if z["name"] == spec.get("loading_zone", "storage"))
    ent = next(z for z in zones if "cz" in z["ops"])
    if ent["pair_axis"] not in (None, "x") or ent["y"][0] <= storage["y"][1]:
        msg = "Expected horizontal interaction pairs and an entangling zone above the storage zone."
        raise NotImplementedError(msg)
    two_qubit = "rzz" if use_rzz and "rzz" in ent["ops"] else "cz"
    width = storage["x"][1] - storage["x"][0] + 1
    slots = [(x, y) for y in range(ent["y"][0], ent["y"][1] + 1) for x in range(ent["x"][0], ent["x"][1], 2)]
    slot_index = {s: i for i, s in enumerate(slots)}
    pos_cache = {}

    def pos(site):
        if site not in pos_cache:
            x, y = site
            z = next(z for z in zones if z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1])
            if z["pair_axis"] == "x":
                k, r = divmod(x - z["x"][0], 2)
                px = z["x"][0] * z["spacing"] + 2 * k * z["spacing"] + r * z["pair_distance"]
            else:
                px = x * z["spacing"]
            pos_cache[site] = (px, y * z["spacing"])
        return pos_cache[site]

    def dist(a, b):
        (ax, ay), (bx, by) = pos(a), pos(b)
        return math.hypot(ax - bx, ay - by)

    def in_ent(site):
        return ent["x"][0] <= site[0] <= ent["x"][1] and ent["y"][0] <= site[1] <= ent["y"][1]

    def slot_of(site):
        return (site[0] - (site[0] - ent["x"][0]) % 2, site[1])

    # --- native gates -------------------------------------------------------------------------------
    def defer_rz(nat):
        """Pushes every rz through the (diagonal) two-qubit gates up to the next other gate on its qubit;
        rz left at the end go back to just after the qubit's last single-qubit gate."""
        ops, pending, last_1q = [], {}, {}
        for inst in nat.data:
            name, qs = inst.operation.name, [nat.find_bit(b).index for b in inst.qubits]
            if name == "rz":
                pending[qs[0]] = pending.get(qs[0], 0.0) + float(inst.operation.params[0])
                continue
            if name != two_qubit:
                for q in qs:
                    if q in pending:
                        ops.append(("rz", q, pending.pop(q)))
            ops.append(inst)
            if len(qs) == 1 and name != two_qubit:
                last_1q[qs[0]] = len(ops)
        inserts = {}
        for q, angle in pending.items():
            inserts.setdefault(last_1q.get(q, 0), []).append(("rz", q, angle))
        out = nat.copy_empty_like()
        for i in range(len(ops) + 1):
            for _, q, angle in inserts.get(i, ()):
                out.rz(angle, q)
            if i < len(ops):
                op = ops[i]
                if isinstance(op, tuple):
                    out.rz(op[2], op[1])
                else:
                    out.append(op)
        return out

    def schedule(body):
        """ASAP steps: ("gates", [instruction index]) / ("round", [(a, b, instruction index)])."""
        qargs = [tuple(body.find_bit(b).index for b in inst.qubits) for inst in body.data]
        queues = {q: [] for q in range(body.num_qubits)}
        for i, qs in enumerate(qargs):
            name = body.data[i].operation.name
            if len(qs) > 2 or (len(qs) == 2 and name != two_qubit):
                msg = f"Unsupported operation after transpilation: {name}"
                raise ValueError(msg)
            for q in qs:
                queues[q].append(i)
        heads = dict.fromkeys(queues, 0)
        steps = []
        while True:
            gates = []
            for q, queue in queues.items():
                while heads[q] < len(queue) and len(qargs[queue[heads[q]]]) == 1:
                    gates.append(queue[heads[q]])
                    heads[q] += 1
            if gates:
                steps.append(("gates", sorted(gates)))
            front = Counter(queue[heads[q]] for q, queue in queues.items() if heads[q] < len(queue))
            if not front:
                break
            ready = [i for i, c in front.items() if c == 2]
            if not ready:
                msg = "Could not schedule the circuit."
                raise RuntimeError(msg)
            for i in ready:
                for q in qargs[i]:
                    heads[q] += 1
            steps.append(("round", [(*qargs[i], i) for i in sorted(ready)]))
        return steps

    def has_permutation(nat):
        layout = getattr(nat, "layout", None)
        if layout is None:
            return False
        perm = layout.routing_permutation()
        return list(perm) != list(range(len(perm)))

    best = None
    for level in (0, 1, 2, 3):
        nat = transpile(circuit, basis_gates=["rx", "rz", two_qubit], optimization_level=level, seed_transpiler=7)
        if has_permutation(nat):
            continue
        nat = PassManager([RemoveBarriers(), Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(nat)
        cand = nat.copy_empty_like()
        cand_measures, measured = [], set()
        for inst in nat.data:
            if inst.operation.name == "measure":
                cand_measures.append(inst)
                measured.update(inst.qubits)
            elif measured.intersection(inst.qubits):
                msg = "Mid-circuit measurements are not supported."
                raise NotImplementedError(msg)
            else:
                cand.append(inst)
        if reuse:
            cand = defer_rz(cand)
        cand_steps = schedule(cand)
        key = (sum(kind == "round" for kind, _ in cand_steps), cand.depth())
        if best is None or key < best[0]:
            best = (key, cand, cand_measures, cand_steps)
    _, body, measures, steps = best
    n = body.num_qubits
    qubit_of = [body.find_bit(inst.qubits[0]).index for inst in body.data]

    rounds = Counter(frozenset((a, b) for a, b, _ in s[1]) for s in steps if s[0] == "round")
    round_steps = [i for i, (kind, _) in enumerate(steps) if kind == "round"]

    # after round step i: must q go to storage (1q gate before its next round)? when is its next round?
    must_leave, next_round = {}, {}
    upcoming, upcoming_round = {}, {}
    for i in reversed(range(len(steps))):
        kind, items = steps[i]
        if kind == "gates":
            for idx in items:
                upcoming[qubit_of[idx]] = "gates"
            continue
        for a, b, _ in items:
            for q in (a, b):
                must_leave[i, q] = upcoming.get(q) == "gates"
                next_round[i, q] = upcoming_round.get(q)
                upcoming[q], upcoming_round[q] = "round", i
    partner_at = {(i, q): p for i in round_steps for a, b, _ in steps[i][1] for q, p in ((a, b), (b, a))}

    rows = [storage["y"][1] - i for i in range(max(1, math.ceil(n / width)))]
    max_shift = ent["y"][0] - min(rows)
    slot_lx = np.array([pos((x, y))[0] for x, y in slots])
    slot_rx = np.array([pos((x + 1, y))[0] for x, y in slots])
    slot_y = np.array([y for _, y in slots], dtype=float)

    # --- moves --------------------------------------------------------------------------------------
    def assign_slots(pairs, where, shift, free=None, row_weight=10.0):
        """Pairs -> (left, right) sites of distinct slots. Storage atoms aim `shift` rows up, atoms already
        in the entangling zone aim to stay. `free`: allowed slot indices (default all)."""
        cols = np.arange(len(slots)) if free is None else np.array(sorted(free))

        def atom_terms(qs):
            sites = [where[q] for q in qs]
            px = np.array([pos(s)[0] for s in sites])
            py = np.array([s[1] + (0 if in_ent(s) else shift) for s in sites], dtype=float)
            return px, py

        ax, ay = atom_terms([p[0] for p in pairs])
        bx, by = atom_terms([p[1] for p in pairs])
        sy, slx, srx = slot_y[cols], slot_lx[cols], slot_rx[cols]

        def atom_cost(px, py, tx):
            return row_weight * (sy[None, :] - py[:, None]) ** 2 + (tx[None, :] - px[:, None]) ** 2

        ab = atom_cost(ax, ay, slx) + atom_cost(bx, by, srx)
        ba = atom_cost(bx, by, slx) + atom_cost(ax, ay, srx)
        r_idx, c_idx = linear_sum_assignment(np.minimum(ab, ba))
        transfers = []
        for i, j in zip(r_idx, c_idx):
            p, (x, y) = pairs[i], slots[cols[j]]
            left, right = (p[0], p[1]) if ab[i, j] <= ba[i, j] else (p[1], p[0])
            transfers += [(left, where[left], (x, y)), (right, where[right], (x + 1, y))]
        return transfers

    def aod_consistent(group):
        for axis in (0, 1):
            mapping = {}
            for _, s, d in group:
                if mapping.setdefault(s[axis], d[axis]) != d[axis]:
                    return False
            keys = sorted(mapping)
            if any(mapping[u] >= mapping[v] for u, v in itertools.pairwise(keys)):
                return False
        return True

    def blocker(group, occ):
        moving = {q for q, _, _ in group}
        xs = [s[0] for _, s, _ in group]
        ys = [s[1] for _, s, _ in group]
        for y in range(min(ys), max(ys) + 1):
            for x in range(min(xs), max(xs) + 1):
                q = occ.get((x, y))
                if q is not None and q not in moving:
                    return ("ghost", (x, y))
        for _, _, d in group:
            q = occ.get(d)
            if q is not None and q not in moving:
                return ("target", d)
        return None

    def split_group(group, ghost):
        gx, gy = ghost
        parts = [
            [t for t in group if t[1][0] < gx],
            [t for t in group if t[1][0] > gx],
            [t for t in group if t[1][0] == gx and t[1][1] < gy],
            [t for t in group if t[1][0] == gx and t[1][1] > gy],
        ]
        return [p for p in parts if p]

    def pack_moves(transfers, occ):
        """Groups transfers into valid rectangle moves, in an order that never drops onto an occupied
        site; updates `occ`. Raises RuntimeError on cyclic transfers."""
        transfers = sorted((t for t in transfers if t[1] != t[2]), key=lambda t: (t[1][1], t[1][0]))
        groups = []
        for t in transfers:
            for g in groups:
                if aod_consistent([*g, t]):
                    g.append(t)
                    break
            else:
                groups.append([t])
        pending = sorted(groups, key=len, reverse=True)
        done = []
        while pending:
            reasons = []
            for i, g in enumerate(pending):
                reason = blocker(g, occ)
                if reason is None:
                    for q, s, _ in g:
                        del occ[s]
                    for q, _, d in g:
                        occ[d] = q
                    done.append(pending.pop(i))
                    break
                reasons.append(reason)
            else:
                for i, (g, reason) in enumerate(zip(pending, reasons)):
                    if len(g) > 1 and reason[0] == "ghost":
                        pending[i : i + 1] = split_group(g, reason[1])
                        break
                else:
                    multi = [i for i, g in enumerate(pending) if len(g) > 1]
                    if not multi:
                        msg = "Cyclic transfers are not supported."
                        raise RuntimeError(msg)
                    i = max(multi, key=lambda k: len(pending[k]))
                    pending[i : i + 1] = [[t] for t in pending[i]]
                pending.sort(key=len, reverse=True)
        return done

    def moves_cost(groups):
        return sum(max(dist(a, b) for _, a, b in g) for g in groups)

    # --- stateless round cost (fast proxy for the layout search) --------------------------------------
    def plan_round(pairs, home):
        best_plan = None
        for s in range(1, max_shift + 1):
            occ = {site: q for q, site in home.items()}
            inbound = assign_slots(pairs, home, s)
            moves_in = pack_moves(inbound, occ)
            moves_out = pack_moves([(q, d, src) for q, src, d in inbound], occ)
            cost = moves_cost(moves_in) + moves_cost(moves_out)
            if best_plan is None or cost < best_plan[1]:
                best_plan = (inbound, cost)
        return best_plan

    def proxy_cost(home):
        return sum(count * plan_round(list(pairs), home)[1] for pairs, count in rounds.items())

    # --- exact sequential simulation ----------------------------------------------------------------
    def simulate(home, stay, out=None):
        """Total move distance of the whole circuit from `home`; appends gates and moves to `out`."""
        where = dict(home)
        occ = {s: q for q, s in where.items()}
        total = 0.0

        def run(transfers):
            nonlocal total
            groups = pack_moves(transfers, occ)
            total += moves_cost(groups)
            for g in groups:
                for q, _, d in g:
                    where[q] = d
                if out is not None:
                    append_move(out, g)

        for i, (kind, items) in enumerate(steps):
            if kind == "gates":
                if out is not None:
                    for idx in items:
                        out.append(body.data[idx])
                continue
            pairs = [(a, b) for a, b, _ in items]
            in_round = {q for p in pairs for q in p}
            idle = [q for q, s in where.items() if in_ent(s) and q not in in_round]
            free = set(range(len(slots))) - {slot_index[slot_of(where[q])] for q in idle}
            if len(free) < len(pairs):  # not enough empty interaction pairs: idle atoms go home
                run([(q, where[q], home[q]) for q in idle])
                free = set(range(len(slots)))
            leaving = {q for q in in_round if must_leave[i, q]}
            best_plan = None
            for s in range(1, max_shift + 1):
                occ_try = dict(occ)
                transfers = assign_slots(pairs, where, s, free)
                try:
                    groups_in = pack_moves(transfers, occ_try)
                    target = {q: d for q, _, d in transfers}
                    groups_out = pack_moves([(q, target[q], home[q]) for q in leaving], occ_try)
                except RuntimeError:
                    continue
                cost = moves_cost(groups_in) + moves_cost(groups_out)
                if best_plan is None or cost < best_plan[1]:
                    best_plan = (transfers, cost)
            if best_plan is None:  # cyclic transfers between entangling sites: restart from storage
                run([(q, where[q], home[q]) for q, s in list(where.items()) if in_ent(s)])
                best_plan = plan_round(pairs, home)
            run(best_plan[0])
            if out is not None:
                for _, _, idx in items:
                    out.append(body.data[idx])
            # who goes back to storage
            go = set()
            for a, b in pairs:
                for q, p in ((a, b), (b, a)):
                    nxt = next_round[i, q]
                    if must_leave[i, q] or (not stay and nxt is not None):
                        go.add(q)
                # two atoms may only share a pair if they interact again in the very next round
                if a not in go and b not in go:
                    na, nb = next_round[i, a], next_round[i, b]
                    together = na is not None and na == nb and partner_at.get((na, a)) == b
                    following = next((r for r in round_steps if r > i), None)
                    if following is not None and not (together and na == following):
                        later = a if (na if na is not None else math.inf) >= (nb if nb is not None else math.inf) else b
                        go.add(later)
            run([(q, where[q], home[q]) for q in sorted(go)])
        return total

    # --- layout search ------------------------------------------------------------------------------
    orders = [list(range(n))]
    if n > 1:
        w = np.zeros((n, n))
        for pairs, count in rounds.items():
            for a, b in pairs:
                w[a, b] += count
                w[b, a] += count
        orders.append([int(q) for q in reverse_cuthill_mckee(csr_matrix(w), symmetric_mode=True)])
    candidates = []
    for order in orders:
        for snake in (False, True):
            cand = {}
            for idx, q in enumerate(order):
                r, c = divmod(idx, width)
                if snake and r % 2:
                    c = width - 1 - c
                cand[q] = (storage["x"][0] + c, rows[r])
            candidates.append(cand)

    def local_search(home, cost_fn, best_cost, deadline, patience):
        sites = [(x, y) for y in rows for x in range(storage["x"][0], storage["x"][1] + 1)]
        rng = np.random.default_rng(1)
        stale = 0
        while time.monotonic() < deadline and stale < patience:
            q = int(rng.integers(n))
            target = sites[int(rng.integers(len(sites)))]
            if target == home[q]:
                continue
            trial = dict(home)
            other = next((p for p, s in home.items() if s == target), None)
            if other is not None:
                trial[other] = home[q]
            trial[q] = target
            cost = cost_fn(trial)
            if cost < best_cost - 1e-9:
                stale = 0
            else:
                stale += 1
            if cost <= best_cost:
                home, best_cost = trial, cost
        return home, best_cost

    deadline = start_time + time_budget
    home, best_cost = min(((c, proxy_cost(c)) for c in candidates), key=lambda t: t[1])
    if rounds and n > 1:
        home, best_cost = local_search(
            home, proxy_cost, best_cost, start_time + proxy_share * time_budget, patience=60 + 4 * n
        )

    def exact(h):
        return min(simulate(h, stay) for stay in ({False, reuse}))

    home, best_cost = min(((h, exact(h)) for h in [home, *candidates]), key=lambda t: t[1])
    if rounds and n > 1:
        home, best_cost = local_search(home, exact, best_cost, deadline, patience=30 + 2 * n)

    # --- emit ---------------------------------------------------------------------------------------
    def append_move(out, group):
        xs = sorted({s[0] for _, s, _ in group})
        ys = sorted({s[1] for _, s, _ in group})
        cmap = {s[0]: d[0] for _, s, d in group}
        rmap = {s[1]: d[1] for _, s, d in group}
        move = Move(
            rows=tuple(range(ys[0], ys[-1] + 1)),
            cols=tuple(range(xs[0], xs[-1] + 1)),
            new_rows=tuple(rmap[y] for y in ys),
            new_cols=tuple(cmap[x] for x in xs),
            num_qubits=len(group),
        )
        out.append(move, [out.qubits[q] for q, _, _ in group])

    stay = reuse and simulate(home, True) <= simulate(home, False)
    compiled = body.copy_empty_like()
    simulate(home, stay, compiled)
    for inst in measures:
        compiled.append(inst)
    return compiled, Architecture(home, spec=architecture_spec)
