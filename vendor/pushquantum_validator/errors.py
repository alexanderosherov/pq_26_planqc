from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class VerificationError(Exception):
    """The compiled circuit failed a check."""


class ArchitectureError(VerificationError):
    """Illegal loading, move, gate zone or interaction.

    A replay does not stop at the first violation: the raised error is the first one, and
    `violations` holds every violation found (each with its own `kind`)."""

    log_file: Path | None = None
    kind: str = "move"
    """One of `KINDS`."""
    violations: tuple[ArchitectureError, ...] = ()

    def __init__(self, message: str, kind: str = "move") -> None:
        super().__init__(message)
        self.kind = kind


KINDS = {
    "move": ("illegal move", "illegal moves"),
    "zone": ("gate outside its zone", "gates outside their zone"),
    "pair": ("two-qubit gate off an interaction pair", "two-qubit gates off an interaction pair"),
    "loading": ("invalid initial loading", "invalid initial loadings"),
    "idle": (
        "two-qubit gate layer with idle atoms in its zone",
        "two-qubit gate layers with idle atoms in their zone",
    ),
}
"""`ArchitectureError.kind` -> (singular, plural) description."""


class CompilerOutputError(VerificationError):
    """`compile_circuit` did not return a valid `(QuantumCircuit, Architecture)`."""


class NotEquivalentError(VerificationError):
    """The compiled circuit is not equivalent to the reference."""
