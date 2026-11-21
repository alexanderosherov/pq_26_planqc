"""Runs a participant's `compile_circuit` in its own process and writes its output as plain JSON.

Executed as a script (never imported by the validator), so the participant's code cannot reach the
validator. Everything it writes is treated as untrusted data by the parent.
Usage: python -I _compile_child.py <request.json> <result.json>
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import json
import sys
from numbers import Integral


def _load(target):
    """`module:function` or `path/to/file.py:function`."""
    location, _, attr = target.rpartition(":")
    if location.endswith(".py"):
        spec = importlib.util.spec_from_file_location("participant_compiler", location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    return getattr(module, attr)


def _plain(value):
    """JSON-ready copy of a move param, without coercion (bools and floats stay visible to the checks)."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Integral):  # e.g. numpy integers
        return int(value)
    if isinstance(value, (list, tuple, range)):
        return [_plain(v) for v in value]
    return repr(value)


def _serialize(output):
    if not (isinstance(output, tuple) and len(output) == 2):
        msg = f"compile_circuit must return (QuantumCircuit, Architecture), got {type(output).__name__}."
        raise TypeError(msg)
    compiled, architecture = output
    qubit_index = {bit: i for i, bit in enumerate(compiled.qubits)}
    clbit_index = {bit: i for i, bit in enumerate(compiled.clbits)}
    instructions = []
    for instruction in compiled.data:
        operation = instruction.operation
        if operation.name == "move":  # the custom AOD move: raw lines, re-checked by the validator
            params = [_plain(lines) for lines in operation.params]
        else:
            params = [float(p) for p in operation.params]
        instructions.append(
            [
                operation.name,
                [qubit_index[q] for q in instruction.qubits],
                [clbit_index[c] for c in instruction.clbits],
                params,
            ]
        )
    loading = [[int(q), int(site[0]), int(site[1])] for q, site in architecture.initial_loading.items()]
    return {
        "num_qubits": compiled.num_qubits,
        "num_clbits": compiled.num_clbits,
        "instructions": instructions,
        "loading": loading,
    }


def _limit(limits):
    """CPU seconds, address space and file size; set before any participant code runs (they cannot be raised)."""
    try:
        import resource
    except ImportError:  # not POSIX
        return
    for name, value in limits.items():
        # macOS refuses an address-space limit below the ~400 GiB every process reserves, and does not enforce it.
        if name == "RLIMIT_AS" and sys.platform == "darwin":
            continue
        # CPU: SIGXCPU at the soft limit (reported as a timeout), SIGKILL one second later.
        hard = value + 1 if name == "RLIMIT_CPU" else value
        resource.setrlimit(getattr(resource, name), (value, hard))


def main():
    request_path, result_path = sys.argv[1], sys.argv[2]
    with open(request_path, encoding="utf-8") as file:
        request = json.load(file)
    _limit(request["limits"])
    try:
        from qiskit import qpy

        (reference,) = qpy.load(io.BytesIO(base64.b64decode(request["reference"])))
        result = _serialize(_load(request["compiler"])(reference, request["spec"]))
    except BaseException as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
