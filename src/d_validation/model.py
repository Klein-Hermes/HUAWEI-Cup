"""Small, dependency-light data models used by the independent validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Box:
    box_id: str
    service_id: str
    material_type: str
    mass_kg: float
    volume_m3: float
    first_batch: bool
    first_deadline_s: float | None
    expected_time_s: float | None
    priority: float


@dataclass(frozen=True)
class AircraftType:
    type_id: str
    max_mass_kg: float
    max_volume_m3: float
    energy_kwh: float
    reserve_ratio: float
    charge_time_s: float
    max_climb_mps: float
    max_descend_mps: float
    cruise_mps: float


@dataclass(frozen=True)
class Aircraft:
    aircraft_id: str
    type_id: str


@dataclass(frozen=True)
class Battery:
    battery_id: str
    type_id: str
    charge_time_s: float


@dataclass(frozen=True)
class RelayType:
    type_id: str
    energy_kwh: float
    reserve_ratio: float
    charge_time_s: float
    max_hover_height_m: float


@dataclass(frozen=True)
class Relay:
    relay_id: str
    type_id: str


@dataclass(frozen=True)
class RelayComponent:
    component_id: str
    type_id: str
    charge_time_s: float


@dataclass(frozen=True)
class Node:
    node_id: str
    x_m: float
    y_m: float
    ground_m: float


@dataclass
class Terrain:
    """Regular-grid terrain used only by the independent communication checker."""

    xs_m: list[float]
    ys_m: list[float]
    elevations_m: list[list[float]]

    def elevation(self, x_m: float, y_m: float) -> float:
        if not self.xs_m or not self.ys_m:
            return float("-inf")
        ix = min(range(len(self.xs_m)), key=lambda i: abs(self.xs_m[i] - x_m))
        iy = min(range(len(self.ys_m)), key=lambda i: abs(self.ys_m[i] - y_m))
        return float(self.elevations_m[iy][ix])


@dataclass
class Scenario:
    nodes: dict[str, Node]
    boxes: dict[str, Box]
    aircraft_types: dict[str, AircraftType]
    aircraft: dict[str, Aircraft]
    batteries: dict[str, Battery]
    relay_type: RelayType
    relays: dict[str, Relay]
    relay_components: dict[str, RelayComponent]
    communication: dict[str, float]
    terrain: Terrain
    services: list[str]
    inventory: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    name: str
    status: str = "PASS"
    checks: dict[str, str] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, code: str, message: str, **evidence: Any) -> None:
        self.status = "FAIL"
        self.violations.append({"code": code, "message": message, "evidence": evidence})

    def warn(self, code: str, message: str, **evidence: Any) -> None:
        self.warnings.append({"code": code, "message": message, "evidence": evidence})

    def mark(self, check: str, status: str) -> None:
        self.checks[check] = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "checks": self.checks,
            "violations": self.violations,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }
