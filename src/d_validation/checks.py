"""Independent constraint checks for the small D-problem fixtures."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable

from .communication import normalize_intervals, pair_threshold_db, uncovered_intervals
from .model import AuditReport, Scenario


EPS = 1e-8


def charge_time(soc: float, full_time_s: float) -> float:
    if soc < -EPS or soc > 1.0 + EPS:
        raise ValueError(f"SOC必须在[0,1]内: {soc}")
    soc = min(1.0, max(0.0, soc))
    if soc < 0.9:
        return full_time_s * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return full_time_s * 0.35 * (1.0 - soc) / 0.10


def _overlaps(intervals: Iterable[tuple[float, float]], tolerance: float = EPS) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    result = []
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1] - tolerance:
            result.append((previous, current))
    return result


def _max_concurrent(intervals: Iterable[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((float(start), 1))
        events.append((float(end), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _interval_coloring_count(intervals: Iterable[tuple[float, float]]) -> int:
    """Minimum number of identical resources for fixed intervals."""

    active: list[float] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals):
        reusable = [value for value in active if value <= start + EPS]
        if reusable:
            active.remove(min(reusable))
        active.append(end)
    return len(active)


def _events(sortie: dict[str, Any]) -> list[dict[str, float | str]]:
    events = sortie.get("node_events") or []
    return sorted(
        [
            {
                "node": str(event["node"]),
                "arrival": float(event["arrival"]),
                "departure": float(event.get("departure", event["arrival"])),
            }
            for event in events
        ],
        key=lambda event: (float(event["arrival"]), float(event["departure"])),
    )


def _sortie_end(sortie: dict[str, Any]) -> float:
    if sortie.get("end_time") is not None:
        return float(sortie["end_time"])
    events = _events(sortie)
    if not events:
        return float(sortie["start_time"])
    return float(events[-1]["departure"])


def schedule_fingerprint(schedule: dict[str, Any]) -> str:
    fields = []
    for sortie in sorted(schedule.get("sorties", []), key=lambda item: str(item.get("sortie_id"))):
        fields.append(
            {
                "sortie_id": sortie.get("sortie_id"),
                "uav_id": sortie.get("uav_id"),
                "uav_type": sortie.get("uav_type"),
                "route": sortie.get("route"),
                "box_ids": sorted(sortie.get("box_ids", [])),
                "start_time": sortie.get("start_time"),
                "relay_id": sortie.get("relay_id"),
                "communication_intervals": normalize_intervals(sortie.get("communication_intervals", [])),
            }
        )
    for task in sorted(schedule.get("relay_tasks", []), key=lambda item: str(item.get("task_id"))):
        fields.append(
            {
                "task_id": task.get("task_id"),
                "relay_id": task.get("relay_id"),
                "service_start": task.get("service_start"),
                "service_end": task.get("service_end"),
                "position": task.get("position"),
                "height_m": task.get("height_m"),
            }
        )
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_capacity(report: AuditReport, scenario: Scenario, sortie: dict[str, Any], boxes: list[Any]) -> None:
    aircraft_type = scenario.aircraft_types[sortie["uav_type"]]
    total_mass = sum(box.mass_kg for box in boxes)
    total_volume = sum(box.volume_m3 for box in boxes)
    if total_mass > aircraft_type.max_mass_kg + EPS:
        report.fail("MASS_EXCEEDED", "架次总质量超过机型上限", sortie_id=sortie.get("sortie_id"), total_mass=total_mass, limit=aircraft_type.max_mass_kg)
    if total_volume > aircraft_type.max_volume_m3 + EPS:
        report.fail("VOLUME_EXCEEDED", "架次总体积超过机型上限", sortie_id=sortie.get("sortie_id"), total_volume=total_volume, limit=aircraft_type.max_volume_m3)


def _check_load_trace(
    report: AuditReport,
    scenario: Scenario,
    sortie: dict[str, Any],
    route: list[str],
    boxes: list[Any],
) -> None:
    """Check that each leg carries the remaining indivisible box mass.

    ``leg_loads[i]`` is the mass loaded before leg ``route[i] -> route[i+1]``.
    Delivery is applied only after arriving at the next service node, so a
    malformed solution cannot pass merely by reporting a feasible total load.
    """

    sortie_id = sortie.get("sortie_id")
    raw_loads = sortie.get("leg_loads")
    expected_count = max(0, len(route) - 1)
    if not isinstance(raw_loads, (list, tuple)):
        report.fail("LEG_LOAD_MISSING", "架次缺少逐航段剩余载荷", sortie_id=sortie_id)
        return
    if len(raw_loads) != expected_count:
        report.fail("LEG_LOAD_COUNT", "逐航段载荷数量与路线航段数不一致", sortie_id=sortie_id, expected=expected_count, actual=len(raw_loads))
        return

    aircraft_type = scenario.aircraft_types.get(str(sortie.get("uav_type")))
    remaining = list(boxes)
    for leg_index, raw_value in enumerate(raw_loads):
        try:
            reported_mass = float(raw_value)
        except (TypeError, ValueError):
            report.fail("LEG_LOAD_INVALID", "逐航段载荷不是数值", sortie_id=sortie_id, leg_index=leg_index, value=raw_value)
            continue
        expected_mass = sum(box.mass_kg for box in remaining)
        expected_volume = sum(box.volume_m3 for box in remaining)
        if abs(reported_mass - expected_mass) > EPS:
            report.fail(
                "LOAD_TRACE",
                "逐航段载荷未按到达服务区后的不可拆分货箱更新",
                sortie_id=sortie_id,
                leg_index=leg_index,
                reported_mass=reported_mass,
                expected_mass=expected_mass,
                expected_volume=expected_volume,
            )
        if reported_mass < -EPS:
            report.fail("LEG_LOAD_NEGATIVE", "逐航段载荷不能为负", sortie_id=sortie_id, leg_index=leg_index, reported_mass=reported_mass)
        if aircraft_type is not None and reported_mass > aircraft_type.max_mass_kg + EPS:
            report.fail("LEG_MASS_EXCEEDED", "某航段载荷超过机型质量上限", sortie_id=sortie_id, leg_index=leg_index, reported_mass=reported_mass, limit=aircraft_type.max_mass_kg)
        if aircraft_type is not None and expected_volume > aircraft_type.max_volume_m3 + EPS:
            report.fail("LEG_VOLUME_EXCEEDED", "某航段剩余货箱总体积超过机型上限", sortie_id=sortie_id, leg_index=leg_index, expected_volume=expected_volume, limit=aircraft_type.max_volume_m3)

        next_node = route[leg_index + 1] if leg_index + 1 < len(route) else None
        if next_node and next_node != "O01":
            remaining = [box for box in remaining if box.service_id != next_node]


def _check_time_windows(report: AuditReport, scenario: Scenario, sortie: dict[str, Any]) -> None:
    delivery_times = {str(key): float(value) for key, value in (sortie.get("delivery_times") or {}).items()}
    for box_id in sortie.get("box_ids", []):
        if box_id not in scenario.boxes:
            continue
        box = scenario.boxes[box_id]
        if box_id not in delivery_times:
            report.fail("DELIVERY_TIME_MISSING", "货箱缺少送达时刻", sortie_id=sortie.get("sortie_id"), box_id=box_id)
            continue
        deadline = None
        if box.first_batch:
            deadline = box.first_deadline_s
        elif box.material_type == "医疗物资":
            deadline = box.expected_time_s
        if deadline is not None and delivery_times[box_id] > deadline + EPS:
            report.fail("TIME_WINDOW", "货箱送达时间超过硬时间窗", sortie_id=sortie.get("sortie_id"), box_id=box_id, delivery=delivery_times[box_id], deadline=deadline)


def _check_communication(report: AuditReport, schedule: dict[str, Any], sortie: dict[str, Any]) -> None:
    start = float(sortie["start_time"])
    end = _sortie_end(sortie)
    intervals = normalize_intervals(sortie.get("communication_intervals", []))
    gaps = uncovered_intervals(intervals, start, end)
    if gaps:
        report.fail("COMMUNICATION_GAP", "通信覆盖存在时间空档", sortie_id=sortie.get("sortie_id"), gaps=gaps)
    for item in intervals:
        if item["state"] == "interrupted" and item["end"] > item["start"] + EPS:
            report.fail("COMMUNICATION_INTERRUPTED", "架次存在明确的通信中断区间", sortie_id=sortie.get("sortie_id"), interval=item)
        if item["state"] == "relay" and not item.get("relay_id"):
            report.fail("RELAY_ID_MISSING", "中继状态缺少中继无人机编号", sortie_id=sortie.get("sortie_id"), interval=item)
    relay_tasks = {str(item.get("relay_id")): item for item in schedule.get("relay_tasks", [])}
    for item in intervals:
        if item["state"] != "relay":
            continue
        task = relay_tasks.get(str(item.get("relay_id")))
        if task is None:
            report.fail("RELAY_TASK_MISSING", "通信区间引用了不存在的中继任务", sortie_id=sortie.get("sortie_id"), relay_id=item.get("relay_id"))
            continue
        service_start = float(task["service_start"])
        service_end = float(task["service_end"])
        if item["start"] < service_start - EPS or item["end"] > service_end + EPS:
            report.fail("RELAY_SERVICE_COVERAGE", "中继服务时段未覆盖运输通信需求", sortie_id=sortie.get("sortie_id"), interval=item, service=(service_start, service_end))


def audit_q3(scenario: Scenario, schedule: dict[str, Any], name: str = "q3_core") -> AuditReport:
    report = AuditReport(name=name)
    all_box_ids: list[str] = []
    aircraft_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    battery_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    relay_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    component_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for sortie in schedule.get("sorties", []):
        required_fields = (
            "sortie_id", "uav_id", "uav_type", "battery_id", "route", "box_ids",
            "start_time", "node_arrival_times", "leg_loads", "leg_energy", "trip_energy",
            "return_soc", "communication_intervals", "relay_id",
        )
        for field in required_fields:
            if field not in sortie:
                report.fail("SCHEMA_FIELD_MISSING", "标准化运输架次缺少字段", sortie_id=sortie.get("sortie_id"), field=field)
        sortie_id = str(sortie.get("sortie_id"))
        uav_id = str(sortie.get("uav_id"))
        uav_type = str(sortie.get("uav_type"))
        battery_id = str(sortie.get("battery_id"))
        if uav_id not in scenario.aircraft:
            report.fail("UAV_UNKNOWN", "使用了未知运输无人机", sortie_id=sortie_id, uav_id=uav_id)
            continue
        if uav_type not in scenario.aircraft_types:
            report.fail("UAV_TYPE_UNKNOWN", "使用了未知机型", sortie_id=sortie_id, uav_type=uav_type)
            continue
        if scenario.aircraft[uav_id].type_id != uav_type:
            report.fail("UAV_TYPE_MISMATCH", "实体无人机与机型不匹配", sortie_id=sortie_id, uav_id=uav_id, uav_type=uav_type)
        if battery_id not in scenario.batteries:
            report.fail("BATTERY_UNKNOWN", "使用了未知电池", sortie_id=sortie_id, battery_id=battery_id)
            continue
        if scenario.batteries[battery_id].type_id != uav_type:
            report.fail("BATTERY_TYPE_MISMATCH", "电池与无人机机型不匹配", sortie_id=sortie_id, battery_id=battery_id, uav_type=uav_type)

        route = [str(node) for node in sortie.get("route", [])]
        arrival_times = sortie.get("node_arrival_times")
        if isinstance(arrival_times, (list, tuple)) and len(arrival_times) != len(route):
            report.fail("NODE_ARRIVAL_COUNT", "节点到达时刻数量与路线节点数不一致", sortie_id=sortie_id, expected=len(route), actual=len(arrival_times))
        if len(route) < 2 or route[0] != "O01" or route[-1] != "O01":
            report.fail("ROUTE_NOT_RETURNING", "路线未从 O01 出发并返回 O01", sortie_id=sortie_id, route=route)
        route_services = [node for node in route if node != "O01"]
        for node in route_services:
            if node not in scenario.services:
                report.fail("SERVICE_UNKNOWN", "路线包含未知服务区", sortie_id=sortie_id, service_id=node)

        box_ids = [str(box_id) for box_id in sortie.get("box_ids", [])]
        all_box_ids.extend(box_ids)
        boxes = []
        for box_id in box_ids:
            if box_id not in scenario.boxes:
                report.fail("BOX_UNKNOWN", "方案包含未知货箱", sortie_id=sortie_id, box_id=box_id)
            else:
                boxes.append(scenario.boxes[box_id])
                if scenario.boxes[box_id].service_id not in route_services:
                    report.fail("BOX_SERVICE_ROUTE", "货箱所属服务区不在架次路线中", sortie_id=sortie_id, box_id=box_id)
        _check_capacity(report, scenario, sortie, boxes)
        _check_load_trace(report, scenario, sortie, route, boxes)
        _check_time_windows(report, scenario, sortie)

        start = float(sortie.get("start_time", 0.0))
        end = _sortie_end(sortie)
        if end <= start + EPS:
            report.fail("TIME_INTERVAL_INVALID", "架次结束时间不晚于开始时间", sortie_id=sortie_id, start=start, end=end)
        aircraft_intervals[uav_id].append((start, end))

        leg_energy = [float(value) for value in sortie.get("leg_energy", [])]
        trip_energy = float(sortie.get("trip_energy", sum(leg_energy)))
        if abs(sum(leg_energy) - trip_energy) > EPS:
            report.fail("ENERGY_SUM", "架次总能耗与航段能耗之和不一致", sortie_id=sortie_id, trip_energy=trip_energy, leg_sum=sum(leg_energy))
        aircraft_type = scenario.aircraft_types[uav_type]
        return_soc = float(sortie.get("return_soc", 1.0 - trip_energy / aircraft_type.energy_kwh))
        expected_soc = 1.0 - trip_energy / aircraft_type.energy_kwh
        if abs(return_soc - expected_soc) > EPS:
            report.fail("SOC_RECOMPUTE", "返航 SOC 与独立重算不一致", sortie_id=sortie_id, reported=return_soc, expected=expected_soc)
        if return_soc < aircraft_type.reserve_ratio - EPS:
            report.fail("SOC_RESERVE", "返航 SOC 低于安全余量", sortie_id=sortie_id, return_soc=return_soc, reserve=aircraft_type.reserve_ratio)
        charge_end = end + charge_time(return_soc, scenario.batteries[battery_id].charge_time_s)
        battery_intervals[battery_id].append((start, charge_end))
        _check_communication(report, schedule, sortie)

    expected_boxes = set(scenario.boxes)
    actual_boxes = set(all_box_ids)
    duplicates = sorted(box_id for box_id in all_box_ids if all_box_ids.count(box_id) > 1)
    missing = sorted(expected_boxes - actual_boxes)
    if duplicates:
        report.fail("BOX_DUPLICATE", "货箱被重复分配", duplicates=sorted(set(duplicates)))
    if missing:
        report.fail("BOX_MISSING", "货箱未被分配", missing=missing)
    if len(all_box_ids) != len(actual_boxes):
        report.fail("BOX_ASSIGNMENT_COUNT", "货箱分配数量与唯一货箱数量不一致", total=len(all_box_ids), unique=len(actual_boxes))

    for uav_id, intervals in aircraft_intervals.items():
        for previous, current in _overlaps(intervals):
            report.fail("UAV_OVERLAP", "同一运输无人机任务区间重叠", uav_id=uav_id, previous=previous, current=current)
    for battery_id, intervals in battery_intervals.items():
        for previous, current in _overlaps(intervals):
            report.fail("BATTERY_OVERLAP", "同一电池任务或充电区间重叠", battery_id=battery_id, previous=previous, current=current)

    for task in schedule.get("relay_tasks", []):
        relay_id = str(task.get("relay_id"))
        component_id = str(task.get("component_id"))
        if relay_id not in scenario.relays:
            report.fail("RELAY_UNKNOWN", "中继任务引用了未知中继无人机", relay_id=relay_id)
        if component_id not in scenario.relay_components:
            report.fail("RELAY_COMPONENT_UNKNOWN", "中继任务引用了未知能源组件", component_id=component_id)
        start = float(task["start"])
        end = float(task["end"])
        service_start = float(task["service_start"])
        service_end = float(task["service_end"])
        if not (start <= service_start + EPS and service_end <= end + EPS):
            report.fail("RELAY_SERVICE_WINDOW", "中继服务时段超出中继任务区间", relay_id=relay_id)
        relay_intervals[relay_id].append((start, end))
        relay_energy = float(task.get("energy", 0.0))
        relay_soc = float(task.get("return_soc", 1.0 - relay_energy / scenario.relay_type.energy_kwh))
        expected_relay_soc = 1.0 - relay_energy / scenario.relay_type.energy_kwh
        if abs(relay_soc - expected_relay_soc) > EPS:
            report.fail("RELAY_SOC_RECOMPUTE", "中继返航 SOC 与独立重算不一致", relay_id=relay_id)
        if relay_soc < scenario.relay_type.reserve_ratio - EPS:
            report.fail("RELAY_SOC_RESERVE", "中继返航 SOC 低于安全余量", relay_id=relay_id, return_soc=relay_soc)
        component_intervals[component_id].append((start, end + charge_time(relay_soc, 1800.0)))
    for relay_id, intervals in relay_intervals.items():
        for previous, current in _overlaps(intervals):
            report.fail("RELAY_OVERLAP", "同一中继无人机任务区间重叠", relay_id=relay_id, previous=previous, current=current)
    for component_id, intervals in component_intervals.items():
        for previous, current in _overlaps(intervals):
            report.fail("RELAY_COMPONENT_OVERLAP", "同一中继能源组件区间重叠", component_id=component_id, previous=previous, current=current)

    report.metrics.update(
        {
            "box_count": len(actual_boxes),
            "sortie_count": len(schedule.get("sorties", [])),
            "relay_task_count": len(schedule.get("relay_tasks", [])),
            "objective_recomputed": [
                len(schedule.get("sorties", [])),
                max((_sortie_end(item) for item in schedule.get("sorties", [])), default=0.0),
                round(sum(float(item.get("trip_energy", sum(item.get("leg_energy", [])))) for item in schedule.get("sorties", [])), 8),
            ],
        }
    )
    report.warn("ENERGY_PHYSICS_BLOCKED", "题面与当前项目未提供 E_hor/E_up 具体公式；本轮仅审计预设能耗下的 SOC 与返航余量。")
    report.mark("non_energy_hard_constraints", "PASS" if not report.violations else "FAIL")
    report.mark("energy_physics", "BLOCKED")
    return report


def _group_resource_counts(scenario: Scenario, group: int, partition: dict[str, int], schedule: dict[str, Any]) -> dict[str, int]:
    sorties = [
        item
        for item in schedule.get("sorties", [])
        if any(partition.get(service) == group for service in item.get("route", []) if service != "O01")
    ]
    result: dict[str, int] = {}
    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    battery_by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in sorties:
        uav_type = str(item["uav_type"])
        interval = (float(item["start_time"]), _sortie_end(item))
        by_type[uav_type].append(interval)
        battery_id = str(item["battery_id"])
        battery = scenario.batteries[battery_id]
        trip_energy = float(item.get("trip_energy", sum(item.get("leg_energy", []))))
        soc = float(item.get("return_soc", 1.0 - trip_energy / scenario.aircraft_types[uav_type].energy_kwh))
        battery_by_type[battery.type_id].append((interval[0], interval[1] + charge_time(soc, battery.charge_time_s)))
    for type_id, intervals in by_type.items():
        result[f"{type_id}_uav"] = _max_concurrent(intervals)
    for type_id, intervals in battery_by_type.items():
        result[f"{type_id}_battery"] = _interval_coloring_count(intervals)

    relay_tasks = []
    for task in schedule.get("relay_tasks", []):
        task_groups = {partition.get(service) for service in task.get("services", [])}
        if group in task_groups:
            relay_tasks.append(task)
    if relay_tasks:
        result["R_relay"] = _max_concurrent((float(item["start"]), float(item["end"])) for item in relay_tasks)
        result["R_component"] = _interval_coloring_count(
            (
                float(item["start"]),
                float(item["end"]) + charge_time(float(item.get("return_soc", 1.0)), scenario.relay_type.charge_time_s),
            )
            for item in relay_tasks
        )
    return result


def _group_resource_ids(group: int, partition: dict[str, int], schedule: dict[str, Any]) -> dict[str, list[str]]:
    """Return resource identifiers used by one group under the fixed schedule."""

    result = {"uav": set(), "battery": set(), "relay": set(), "component": set()}
    for sortie in schedule.get("sorties", []):
        groups = {partition.get(service) for service in sortie.get("route", []) if service != "O01"}
        if group in groups:
            result["uav"].add(str(sortie.get("uav_id")))
            result["battery"].add(str(sortie.get("battery_id")))
    for task in schedule.get("relay_tasks", []):
        groups = {partition.get(service) for service in task.get("services", [])}
        if group in groups:
            result["relay"].add(str(task.get("relay_id")))
            result["component"].add(str(task.get("component_id")))
    return {key: sorted(value) for key, value in result.items()}


def _fixed_resource_counts(scenario: Scenario, resource_ids: dict[str, list[str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for resource_id in resource_ids["uav"]:
        if resource_id in scenario.aircraft:
            key = f"{scenario.aircraft[resource_id].type_id}_uav"
            result[key] = result.get(key, 0) + 1
    for resource_id in resource_ids["battery"]:
        if resource_id in scenario.batteries:
            key = f"{scenario.batteries[resource_id].type_id}_battery"
            result[key] = result.get(key, 0) + 1
    if resource_ids["relay"]:
        result["R_relay"] = len(resource_ids["relay"])
    if resource_ids["component"]:
        result["R_component"] = len(resource_ids["component"])
    return result


def audit_q4(scenario: Scenario, q3_schedule: dict[str, Any], partition: dict[str, Any], name: str = "q4_core") -> AuditReport:
    report = AuditReport(name=name)
    groups = int(partition.get("group_count", 0))
    service_to_group = {str(key): int(value) for key, value in partition.get("service_to_group", {}).items()}
    if groups not in {2, 3}:
        report.fail("GROUP_COUNT", "问题四任务组数量必须为2或3", group_count=groups)
    for service in scenario.services:
        if service not in service_to_group:
            report.fail("SERVICE_UNASSIGNED", "服务区没有任务组归属", service_id=service)
    if set(service_to_group.values()) != set(range(1, groups + 1)):
        report.fail("GROUP_NONEMPTY", "存在空任务组或任务组编号不连续", values=sorted(set(service_to_group.values())))

    for sortie in q3_schedule.get("sorties", []):
        route_services = [node for node in sortie.get("route", []) if node != "O01"]
        assigned = {service_to_group.get(service) for service in route_services}
        if None in assigned or len(assigned) > 1:
            report.fail("ROUTE_SPLIT", "同一架次涉及的服务区被分到不同任务组", sortie_id=sortie.get("sortie_id"), route_services=route_services, groups=sorted(assigned, key=str))

    expected_fingerprint = schedule_fingerprint(q3_schedule)
    supplied_fingerprint = partition.get("q3_schedule_fingerprint")
    if supplied_fingerprint != expected_fingerprint:
        report.fail("Q3_INHERITANCE", "问题四未继承问题三的固定调度方案", expected=expected_fingerprint, supplied=supplied_fingerprint)

    group_resources: dict[int, dict[str, int]] = {}
    group_resource_ids: dict[str, dict[str, list[str]]] = {}
    group_fixed_counts: dict[str, dict[str, int]] = {}
    for group in range(1, groups + 1):
        group_resources[group] = _group_resource_counts(scenario, group, service_to_group, q3_schedule)
        ids = _group_resource_ids(group, service_to_group, q3_schedule)
        group_resource_ids[str(group)] = ids
        group_fixed_counts[str(group)] = _fixed_resource_counts(scenario, ids)
    report.metrics["group_resource_counts_recomputed"] = group_resources
    report.metrics["group_resource_fixed_ids"] = group_resource_ids
    report.metrics["group_resource_counts_fixed"] = group_fixed_counts

    owners: dict[str, dict[str, set[int]]] = {"uav": defaultdict(set), "battery": defaultdict(set), "relay": defaultdict(set), "component": defaultdict(set)}
    for group_key, ids in group_resource_ids.items():
        group = int(group_key)
        for kind, values in ids.items():
            for resource_id in values:
                owners[kind][resource_id].add(group)
    for kind, resource_map in owners.items():
        for resource_id, owner_groups in resource_map.items():
            if len(owner_groups) > 1:
                report.fail("CROSS_GROUP_RESOURCE", "固定问题三方案中的资源被多个任务组实际使用", resource_type=kind, resource_id=resource_id, groups=sorted(owner_groups))
    report.metrics["resource_owners"] = {
        kind: {resource_id: sorted(groups_owned) for resource_id, groups_owned in resource_map.items()}
        for kind, resource_map in owners.items()
    }

    declared = partition.get("group_resource_counts") or {}
    for group, expected in group_resources.items():
        actual = {str(key): int(value) for key, value in declared.get(str(group), declared.get(group, {})).items()}
        for key, value in expected.items():
            if actual.get(key) != value:
                report.fail("RESOURCE_COUNT", "任务组资源数量与独立区间核算不一致", group=group, resource=key, expected=value, actual=actual.get(key))

    assignments = partition.get("group_resource_assignments") or {}
    for resource_type in ("uav", "battery", "relay", "component"):
        owner: dict[str, int] = {}
        for group_key, resources in assignments.items():
            group = int(group_key)
            for resource_id in resources.get(resource_type, []):
                if resource_id in owner and owner[resource_id] != group:
                    report.fail("CROSS_GROUP_RESOURCE", "同一资源被多个任务组共享", resource_type=resource_type, resource_id=resource_id, groups=[owner[resource_id], group])
                owner[resource_id] = group

    shortages = partition.get("resource_shortages") or {}
    recomputed_shortages: dict[str, dict[str, int]] = {}
    fixed_shortages: dict[str, dict[str, int]] = {}
    for group, counts in group_resources.items():
        recomputed: dict[str, int] = {}
        for key, required in counts.items():
            base_type = key.split("_")[0]
            inventory_key = "R_COMPONENT" if key == "R_component" else base_type
            recomputed[key] = max(0, int(required) - int(scenario.inventory.get(inventory_key, 0)))
        recomputed_shortages[str(group)] = recomputed
        supplied = shortages.get(str(group), shortages.get(group, {}))
        for key, value in recomputed.items():
            if int(supplied.get(key, -1)) != value:
                report.fail("RESOURCE_SHORTAGE", "资源缺口与独立核算不一致", group=group, resource=key, expected=value, actual=supplied.get(key))
        fixed_counts = group_fixed_counts[str(group)]
        fixed_shortages[str(group)] = {
            key: max(0, int(required) - int(scenario.inventory.get("R_COMPONENT" if key == "R_component" else key.split("_")[0], 0)))
            for key, required in fixed_counts.items()
        }
    report.metrics["resource_shortages_recomputed"] = recomputed_shortages
    report.metrics["resource_shortages_fixed"] = fixed_shortages
    report.mark("partition_hard_constraints", "PASS" if not report.violations else "FAIL")
    return report


def link_budget_checks(scenario: Scenario) -> dict[str, Any]:
    distance_km = 1.0
    expected_fspl = 32.45 + 20.0 * __import__("math").log10(2400.0)
    return {
        "fspl_1km_db": expected_fspl,
        "transport_gateway_threshold_db": pair_threshold_db("transport_gateway", scenario.communication),
        "transport_relay_threshold_db": pair_threshold_db("transport_relay", scenario.communication),
        "relay_gateway_threshold_db": pair_threshold_db("relay_gateway", scenario.communication),
        "distance_km": distance_km,
    }
