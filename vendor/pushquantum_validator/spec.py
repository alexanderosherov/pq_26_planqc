"""The validator's own architecture spec, parser and zone type (the template's may be edited)."""

from __future__ import annotations

import copy
import functools
import json
from dataclasses import dataclass
from importlib import resources
from numbers import Integral, Real
from typing import Any

from pushquantum_validator.errors import ArchitectureError

AXES = ("x", "y")


@dataclass(frozen=True)
class Zone:
    name: str
    x: tuple[int, int]
    y: tuple[int, int]
    operations: frozenset[str]
    spacing: float = 1.0
    pair_axis: str | None = None
    pair_distance: float | None = None


@functools.cache
def _trusted_text() -> str:
    return resources.files("pushquantum_validator").joinpath("architectures/default.json").read_text(encoding="utf-8")


def load_trusted_spec() -> dict[str, Any]:
    """The spec bundled with the validator (a fresh copy on every call)."""
    return json.loads(_trusted_text())


def _bounds(value: object, zone: str, axis: str) -> tuple[int, int]:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, Integral) and not isinstance(v, bool) for v in value)
        and 0 <= value[0] <= value[1]
    ):
        msg = f"Zone {zone!r}: {axis} must be [min, max] with 0 <= min <= max, got {value!r}."
        raise ArchitectureError(msg)
    return int(value[0]), int(value[1])


def _positive(value: object, what: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
        msg = f"{what} must be a positive number, got {value!r}."
        raise ArchitectureError(msg)
    return float(value)


def _parse_zone(name: str, z: object) -> Zone:
    if not isinstance(z, dict):
        msg = f"Zone {name!r} must be an object, got {z!r}."
        raise ArchitectureError(msg)
    operations = z.get("operations", [])
    if not isinstance(operations, list) or not all(isinstance(op, str) for op in operations):
        msg = f"Zone {name!r}: operations must be a list of gate names, got {operations!r}."
        raise ArchitectureError(msg)
    pairs = z.get("pairs")
    pair_axis = pair_distance = None
    if pairs is not None:
        if not isinstance(pairs, dict) or pairs.get("axis") not in AXES:
            msg = f"Zone {name!r}: pairs.axis must be 'x' or 'y', got {pairs!r}."
            raise ArchitectureError(msg)
        pair_axis = pairs["axis"]
        pair_distance = _positive(pairs.get("distance"), f"Zone {name!r}: pairs.distance")
    return Zone(
        name=name,
        x=_bounds(z.get("x"), name, "x"),
        y=_bounds(z.get("y"), name, "y"),
        operations=frozenset(operations),
        spacing=_positive(z.get("spacing", 1.0), f"Zone {name!r}: spacing"),
        pair_axis=pair_axis,
        pair_distance=pair_distance,
    )


def parse_spec(spec: dict[str, Any] | None = None) -> tuple[dict[str, Zone], str]:
    """`(zones, loading_zone)` of `spec` (default: the bundled one); raises `ArchitectureError`."""
    if spec is None:
        zones, loading_zone = _trusted_zones()
        return dict(zones), loading_zone
    return _parse(copy.deepcopy(spec))


@functools.cache
def _trusted_zones() -> tuple[dict[str, Zone], str]:
    return _parse(load_trusted_spec())


def _parse(spec: object) -> tuple[dict[str, Zone], str]:
    zones_data = spec.get("zones") if isinstance(spec, dict) else None
    if not isinstance(zones_data, dict) or not zones_data:
        msg = "Architecture must define at least one zone."
        raise ArchitectureError(msg)
    zones = {name: _parse_zone(name, z) for name, z in zones_data.items()}
    loading_zone = spec.get("loading_zone", "storage")
    if loading_zone not in zones:
        msg = f"Loading zone {loading_zone!r} is not one of the zones {sorted(zones)}."
        raise ArchitectureError(msg)
    return zones, loading_zone
