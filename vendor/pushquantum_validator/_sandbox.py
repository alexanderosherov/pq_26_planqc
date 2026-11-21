"""Process boundary: the participant's compiler runs in a child process; only checked JSON comes back."""

from __future__ import annotations

import base64
import contextlib
import functools
import io
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Barrier, CircuitInstruction, Instruction
from qiskit.circuit.library import get_standard_gate_name_mapping

from pushquantum_validator._validation import MOVE_NAME
from pushquantum_validator.errors import CompilerOutputError

DEFAULT_COMPILER = "pushquantum.template:compile_circuit"
CHILD = Path(__file__).with_name("_compile_child.py")
MAX_OUTPUT_BYTES = 64 * 2**20
MAX_QUBITS = 10_000
GATES = {name: gate for name, gate in get_standard_gate_name_mapping().items() if name != "delay"}
NO_NETWORK = ("unshare", "-rn")
"""Command prefix for a child without network: new user and network namespaces (Linux only)."""


@functools.cache
def check_no_network() -> None:
    """Raises `OSError` unless `NO_NETWORK` works here (it needs unprivileged user namespaces)."""
    result = subprocess.run([*NO_NETWORK, "true"], capture_output=True, text=True, check=False)
    if result.returncode:
        msg = f"`{' '.join(NO_NETWORK)}` failed: {result.stderr.strip()}"
        raise OSError(msg)


class MoveInstruction(Instruction):
    """The validator's own `move` (identity); its raw params are re-checked by `ValidatedMove`."""

    def __init__(self, params: list[Any], num_qubits: int) -> None:
        super().__init__(MOVE_NAME, num_qubits, 0, params)

    def validate_parameter(self, parameter: object) -> object:
        return parameter

    def _define(self) -> None:
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


def _fail(msg: str) -> CompilerOutputError:
    return CompilerOutputError(f"compile_circuit output: {msg}")


def _int(value: object, low: int, high: int, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise _fail(f"{what} must be an integer in [{low}, {high}], got {value!r}.")
    return value


def _bits(values: object, size: int, what: str) -> list[int]:
    if not isinstance(values, list):
        raise _fail(f"{what} must be a list, got {values!r}.")
    bits = [_int(v, 0, size - 1, what) for v in values]
    if len(set(bits)) != len(bits):
        raise _fail(f"{what} must be distinct, got {bits}.")
    return bits


def _operation(name: str, params: list[Any], num_qubits: int, num_clbits: int) -> Instruction:
    if name == MOVE_NAME:
        return MoveInstruction([tuple(p) if isinstance(p, list) else p for p in params], num_qubits)
    if name == "barrier" and not params and not num_clbits:
        return Barrier(num_qubits)
    gate = GATES.get(name)
    if gate is None:
        raise _fail(f"unknown operation {name!r}.")
    if (gate.num_qubits, gate.num_clbits, len(gate.params)) != (num_qubits, num_clbits, len(params)):
        raise _fail(f"{name!r} on {num_qubits} qubits, {num_clbits} clbits with {len(params)} params.")
    if not all(isinstance(p, Real) and not isinstance(p, bool) and math.isfinite(p) for p in params):
        raise _fail(f"{name!r} params must be finite numbers, got {params!r}.")
    return type(gate)(*params)


def parse_output(data: object) -> tuple[QuantumCircuit, dict[int, tuple[int, int]]]:
    """Rebuilds the compiled circuit and the initial loading from the child's JSON, checking everything."""
    if not isinstance(data, dict):
        raise _fail("not a JSON object.")
    if "error" in data:
        raise CompilerOutputError(f"compile_circuit raised {str(data['error'])[:2000]}")
    num_qubits = _int(data.get("num_qubits"), 1, MAX_QUBITS, "num_qubits")
    num_clbits = _int(data.get("num_clbits"), 0, MAX_QUBITS, "num_clbits")
    circuit = QuantumCircuit(num_qubits, num_clbits)
    instructions = data.get("instructions")
    if not isinstance(instructions, list):
        raise _fail("instructions must be a list.")
    qubit_objects, clbit_objects = circuit.qubits, circuit.clbits
    for item in instructions:
        if not (isinstance(item, list) and len(item) == 4 and isinstance(item[0], str) and isinstance(item[3], list)):
            raise _fail(f"instruction must be [name, qubits, clbits, params], got {item!r:.200}.")
        name, qubits, clbits, params = item
        qubits, clbits = _bits(qubits, num_qubits, "qubits"), _bits(clbits, num_clbits, "clbits")
        operation = _operation(name, params, len(qubits), len(clbits))
        # Bits, arity and params are checked above, so skip `append`'s (slow) re-validation.
        circuit._append(
            CircuitInstruction(operation, [qubit_objects[q] for q in qubits], [clbit_objects[c] for c in clbits])
        )
    loading_data = data.get("loading")
    if not isinstance(loading_data, list):
        raise _fail("loading must be a list.")
    loading: dict[int, tuple[int, int]] = {}
    for entry in loading_data:
        if not (isinstance(entry, list) and len(entry) == 3):
            raise _fail(f"loading entry must be [qubit, x, y], got {entry!r:.200}.")
        qubit, x, y = (_int(v, 0, MAX_QUBITS, "loading") for v in entry)
        if qubit in loading:
            raise _fail(f"qubit {qubit} is loaded twice.")
        loading[qubit] = (x, y)
    return circuit, loading


def run_compiler_process(
    reference: QuantumCircuit,
    spec: dict[str, Any],
    compiler: str = DEFAULT_COMPILER,
    timeout: float = 300,
    memory_mb: int = 4096,
    network: bool = True,
) -> tuple[QuantumCircuit, dict[int, tuple[int, int]]]:
    """Runs `compiler` (`module:function` or `file.py:function`) on `reference` in a child process.

    The child gets a wall-clock `timeout`, the same CPU time, `memory_mb` of address space, a scratch
    working directory and a minimal environment. Only its JSON result is read; its exit code and
    output are ignored. Raises `CompilerOutputError` on any failure.

    With `network=False` the child runs under `NO_NETWORK`, so it cannot reach any host, not even
    localhost. Raises `OSError` before starting the compiler if that is not possible here.
    """
    if not network:
        check_no_network()  # an infrastructure failure, not the compiler's
    location, _, attr = compiler.rpartition(":")
    if location.endswith(".py"):  # the child runs in a scratch directory, so make the path absolute
        compiler = f"{Path(location).resolve()}:{attr}"
    buffer = io.BytesIO()
    qpy.dump(reference, buffer)
    with tempfile.TemporaryDirectory(prefix="pushquantum_compile_") as workdir:
        request, result = Path(workdir, "request.json"), Path(workdir, "result.json")
        request.write_text(
            json.dumps(
                {
                    "reference": base64.b64encode(buffer.getvalue()).decode(),
                    "spec": spec,
                    "compiler": compiler,
                    "limits": {
                        "RLIMIT_CPU": math.ceil(timeout),
                        "RLIMIT_AS": memory_mb * 2**20,
                        "RLIMIT_FSIZE": MAX_OUTPUT_BYTES,
                    },
                }
            ),
            encoding="utf-8",
        )
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": workdir,
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with open(Path(workdir, "output.log"), "wb") as output:
            process = subprocess.Popen(
                [*(() if network else NO_NETWORK), sys.executable, "-I", str(CHILD), str(request), str(result)],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                msg = f"compile_circuit timed out after {timeout:g} s."
                raise CompilerOutputError(msg) from None
            finally:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)  # the child and anything it started
                process.wait()
        if os.name == "posix" and process.returncode == -signal.SIGXCPU:
            # CPU time adds up over threads, so the CPU limit can trigger before the wall clock.
            msg = f"compile_circuit timed out after {timeout:g} s of CPU time."
            raise CompilerOutputError(msg)
        if not result.is_file():
            tail = Path(workdir, "output.log").read_bytes()[-2000:].decode(errors="replace").strip()
            msg = f"compile_circuit produced no result (exit code {process.returncode}). {tail}"
            raise CompilerOutputError(msg.strip())
        if result.stat().st_size > MAX_OUTPUT_BYTES:
            msg = "compile_circuit output is too large."
            raise CompilerOutputError(msg)
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            msg = f"compile_circuit output is not valid JSON: {error}"
            raise CompilerOutputError(msg) from None
    return parse_output(data)
