"""pushquantum-validator: checks circuits compiled with the pushquantum template."""

from __future__ import annotations

from pushquantum_validator._validation import MoveRecord, ValidatedMove
from pushquantum_validator.errors import ArchitectureError, CompilerOutputError, NotEquivalentError, VerificationError
from pushquantum_validator.verify import (
    CompilationResult,
    Metrics,
    MovementResult,
    compute_metrics,
    run_compiler,
    score_compiler,
    strip_moves,
    verify_compilation,
    verify_equivalence,
    verify_movement,
)

__all__ = [
    "ArchitectureError",
    "CompilationResult",
    "CompilerOutputError",
    "Metrics",
    "MoveRecord",
    "MovementResult",
    "NotEquivalentError",
    "ValidatedMove",
    "VerificationError",
    "compute_metrics",
    "run_compiler",
    "score_compiler",
    "strip_moves",
    "verify_compilation",
    "verify_equivalence",
    "verify_movement",
]
