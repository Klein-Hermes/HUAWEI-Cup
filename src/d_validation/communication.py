"""Independent communication and interval utilities."""

from __future__ import annotations

from math import log10, sqrt
from typing import Iterable

from .model import Terrain


def fspl_db(f_mhz: float, distance_km: float) -> float:
    if f_mhz <= 0 or distance_km <= 0:
        raise ValueError("频率和距离必须为正")
    return 32.45 + 20.0 * log10(f_mhz) + 20.0 * log10(distance_km)


def endpoint_distance_km(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / 1000.0


def line_of_sight(terrain: Terrain, a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Conservative grid-aware LOS check for a regular DEM fixture."""

    distance = sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    sample_count = max(2, int(distance / 30.0) + 1)
    for index in range(sample_count + 1):
        ratio = index / sample_count
        x = a[0] + ratio * (b[0] - a[0])
        y = a[1] + ratio * (b[1] - a[1])
        z_line = a[2] + ratio * (b[2] - a[2])
        if terrain.elevation(x, y) > z_line + 1e-9:
            return False
    return True


def _threshold(
    pt: float,
    gt: float,
    gr: float,
    lsys: float,
    psens: float,
    margin: float,
) -> float:
    return pt + gt + gr - lsys - (psens + margin)


def pair_threshold_db(pair: str, params: dict[str, float]) -> float:
    """Return the smaller directional budget for a communication pair."""

    psens = params["Psens"]
    margin = params["M"]
    lsys = params["Lsys"]
    if pair == "transport_gateway":
        forward = _threshold(params["Pt_transport"], params["G_transport"], params["G_gateway"], lsys, psens, margin)
        reverse = _threshold(params["Pt_gateway"], params["G_gateway"], params["G_transport"], lsys, psens, margin)
    elif pair == "transport_relay":
        forward = _threshold(params["Pt_transport"], params["G_transport"], params["G_relay_access"], lsys, psens, margin)
        reverse = _threshold(params["Pt_relay_access"], params["G_relay_access"], params["G_transport"], lsys, psens, margin)
    elif pair == "relay_gateway":
        forward = _threshold(params["Pt_relay_backhaul"], params["G_relay_backhaul"], params["G_gateway"], lsys, psens, margin)
        reverse = _threshold(params["Pt_gateway"], params["G_gateway"], params["G_relay_backhaul"], lsys, psens, margin)
    else:
        raise KeyError(f"未知通信链路类型: {pair}")
    return min(forward, reverse)


def link_available(
    pair: str,
    distance_km: float,
    blocked: bool,
    params: dict[str, float],
    tolerance_db: float = 1e-9,
) -> bool:
    loss = fspl_db(params["f"], distance_km)
    if blocked:
        loss += params["Lobs"]
    return loss <= pair_threshold_db(pair, params) + tolerance_db


def communication_state(direct: bool, access: bool, backhaul: bool) -> str:
    if direct:
        return "direct"
    if access and backhaul:
        return "relay"
    return "interrupted"


def interval_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance_s: float = 1e-9,
) -> tuple[float, float] | None:
    """Return the common usable interval of two half-open time windows."""

    start = max(float(first[0]), float(second[0]))
    end = min(float(first[1]), float(second[1]))
    return (start, end) if end > start + tolerance_s else None


def normalize_intervals(intervals: Iterable[dict]) -> list[dict]:
    normalized = []
    for item in intervals:
        normalized.append(
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "state": str(item["state"]),
                "relay_id": item.get("relay_id"),
            }
        )
    return sorted(normalized, key=lambda item: (item["start"], item["end"]))


def uncovered_intervals(
    intervals: Iterable[dict],
    start: float,
    end: float,
    tolerance_s: float = 1e-9,
) -> list[tuple[float, float]]:
    current = float(start)
    gaps: list[tuple[float, float]] = []
    for item in normalize_intervals(intervals):
        left = max(current, item["start"])
        if item["end"] <= current + tolerance_s:
            continue
        if left > current + tolerance_s:
            gaps.append((current, left))
        current = max(current, item["end"])
        if current >= end - tolerance_s:
            break
    if current < end - tolerance_s:
        gaps.append((current, end))
    return gaps
