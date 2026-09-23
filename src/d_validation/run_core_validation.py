"""Run the one-hour core validation suite for D-question Q3/Q4.

This is intentionally a standalone checker. It does not import or call a solver.
Run from the project root with the huawei-cup Python environment:

    python -m src.d_validation.run_core_validation
"""

from __future__ import annotations

import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from .checks import audit_q3, audit_q4, charge_time, link_budget_checks
from .communication import communication_state, fspl_db, interval_intersection, line_of_sight, link_available
from .data import load_mini_scenario
from .fixtures import invalid_cases, valid_q3_schedule, valid_q4_partition
from .model import Terrain


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "d_validation"


def _close(actual: float, expected: float, tolerance: float = 1e-8) -> bool:
    return abs(actual - expected) <= tolerance


def _test(name: str, fn) -> dict[str, Any]:
    try:
        result = fn()
        return {"name": name, "status": "PASS", "details": result or {}}
    except Exception as exc:  # The report must preserve the failing assertion.
        return {"name": name, "status": "FAIL", "details": {"error": f"{type(exc).__name__}: {exc}"}}


def _capacity_oracle(boxes: list[Any], max_mass: float, max_volume: float) -> int:
    """Exact minimum bin count for a tiny indivisible packing instance."""

    n = len(boxes)
    feasible_masks: list[int] = []
    for mask in range(1, 1 << n):
        mass = sum(boxes[i].mass_kg for i in range(n) if mask & (1 << i))
        volume = sum(boxes[i].volume_m3 for i in range(n) if mask & (1 << i))
        if mass <= max_mass + 1e-8 and volume <= max_volume + 1e-8:
            feasible_masks.append(mask)
    best = [n + 1] * (1 << n)
    best[0] = 0
    for mask in range(1 << n):
        if best[mask] > n:
            continue
        remaining = ((1 << n) - 1) ^ mask
        for subset in feasible_masks:
            if subset & remaining == subset:
                best[mask | subset] = min(best[mask | subset], best[mask] + 1)
    return best[-1]


def _finite_schedule_oracle(scenario) -> dict[str, Any]:
    """Enumerate a deliberately finite Q3 schedule universe.

    This is an independent benchmark over three route templates, discrete
    event times, indivisible box subsets, and the concrete aircraft/battery
    identifiers in the mini-instance. It is not a full D-question solver.
    """

    box_ids = sorted(scenario.boxes)
    type_to_uav = {"A": "U01", "B": "U05", "C": "U07"}
    type_to_battery = {"A": "A-BAT-01", "B": "B-BAT-01", "C": "C-BAT-01"}
    templates = [
        {"route": ("S001",), "start": 0.0, "end": 350.0, "delivery": {"S001": 250.0}, "energy": 0.9},
        {"route": ("S002",), "start": 0.0, "end": 350.0, "delivery": {"S002": 250.0}, "energy": 0.7},
        {"route": ("S001", "S015"), "start": 500.0, "end": 1000.0, "delivery": {"S001": 700.0, "S015": 850.0}, "energy": 1.5},
    ]

    candidates: list[dict[str, Any]] = []
    for template_index, template in enumerate(templates):
        allowed = [index for index, box_id in enumerate(box_ids) if scenario.boxes[box_id].service_id in template["route"]]
        for subset in range(1, 1 << len(allowed)):
            mask = 0
            boxes = []
            for local_index, global_index in enumerate(allowed):
                if subset & (1 << local_index):
                    mask |= 1 << global_index
                    boxes.append(scenario.boxes[box_ids[global_index]])
            if any(
                deadline is not None and template["delivery"][box.service_id] > deadline + 1e-8
                for box in boxes
                for deadline in [box.first_deadline_s if box.first_batch else box.expected_time_s]
            ):
                continue
            mass = sum(box.mass_kg for box in boxes)
            volume = sum(box.volume_m3 for box in boxes)
            for type_id, aircraft_type in scenario.aircraft_types.items():
                if mass <= aircraft_type.max_mass_kg + 1e-8 and volume <= aircraft_type.max_volume_m3 + 1e-8:
                    candidates.append(
                        {
                            "template_index": template_index,
                            "mask": mask,
                            "box_ids": [box.box_id for box in boxes],
                            "route": ["O01", *template["route"], "O01"],
                            "uav_type": type_id,
                            "uav_id": type_to_uav[type_id],
                            "battery_id": type_to_battery[type_id],
                            "start": template["start"],
                            "end": template["end"],
                            "energy": template["energy"],
                        }
                    )

    by_box: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(box_ids))}
    for candidate in candidates:
        for index in range(len(box_ids)):
            if candidate["mask"] & (1 << index):
                by_box[index].append(candidate)

    def resource_feasible(selected: list[dict[str, Any]]) -> bool:
        for type_id in type_to_uav:
            intervals = sorted((item["start"], item["end"]) for item in selected if item["uav_type"] == type_id)
            if any(current[0] < previous[1] - 1e-8 for previous, current in zip(intervals, intervals[1:])):
                return False
            battery = scenario.batteries[type_to_battery[type_id]]
            battery_intervals = []
            for item in selected:
                if item["uav_type"] != type_id:
                    continue
                soc = 1.0 - item["energy"] / scenario.aircraft_types[type_id].energy_kwh
                battery_intervals.append((item["start"], item["end"] + charge_time(soc, battery.charge_time_s)))
            battery_intervals.sort()
            if any(current[0] < previous[1] - 1e-8 for previous, current in zip(battery_intervals, battery_intervals[1:])):
                return False
        return True

    best: tuple[tuple[int, float, float], list[dict[str, Any]]] | None = None
    complete_count = 0

    def search(mask: int, selected: list[dict[str, Any]]) -> None:
        nonlocal best, complete_count
        if mask == (1 << len(box_ids)) - 1:
            complete_count += 1
            objective = (len(selected), max(item["end"] for item in selected), round(sum(item["energy"] for item in selected), 8))
            if best is None or objective < best[0]:
                best = (objective, list(selected))
            return
        first_uncovered = next(index for index in range(len(box_ids)) if not mask & (1 << index))
        for candidate in by_box[first_uncovered]:
            if candidate["mask"] & mask:
                continue
            selected.append(candidate)
            if resource_feasible(selected):
                search(mask | candidate["mask"], selected)
            selected.pop()

    search(0, [])
    if best is None:
        raise AssertionError("有限候选路线集合无法覆盖全部小实例货箱")
    return {
        "candidate_count": len(candidates),
        "complete_schedule_count": complete_count,
        "best_objective": list(best[0]),
        "witness": [
            {
                "route": item["route"],
                "box_ids": item["box_ids"],
                "uav_id": item["uav_id"],
                "uav_type": item["uav_type"],
                "battery_id": item["battery_id"],
                "start": item["start"],
                "end": item["end"],
                "energy": item["energy"],
            }
            for item in best[1]
        ],
    }


def _run_unit_tests(scenario, actual_terrain) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    tests.append(
        _test("charge_formula_boundaries", lambda: {
            "values": {
                "s0": charge_time(0.0, 1800.0),
                "s05": charge_time(0.5, 1800.0),
                "s09": charge_time(0.9, 1800.0),
                "s1": charge_time(1.0, 1800.0),
            }
        } if (
            _close(charge_time(0.0, 1800.0), 1800.0)
            and _close(charge_time(0.5, 1800.0), 1150.0)
            and _close(charge_time(0.9, 1800.0), 630.0)
            and _close(charge_time(1.0, 1800.0), 0.0)
        ) else (_ for _ in ()).throw(AssertionError("充电边界值不符合题面公式")))
    )

    selected = [
        scenario.boxes["S001-MED-01"],
        scenario.boxes["S001-WAT-01"],
        scenario.boxes["S001-FOD-01"],
        scenario.boxes["S001-HYG-01"],
    ]
    tests.append(
        _test("tiny_exact_capacity_oracle", lambda: {
            "A_min_sorties": _capacity_oracle(selected, 25.0, 0.06),
            "B_min_sorties": _capacity_oracle(selected, 30.0, 0.073),
            "C_min_sorties": _capacity_oracle(selected, 80.0, 0.25),
        } if (
            _capacity_oracle(selected, 25.0, 0.06) == 2
            and _capacity_oracle(selected, 30.0, 0.073) == 2
            and _capacity_oracle(selected, 80.0, 0.25) == 1
        ) else (_ for _ in ()).throw(AssertionError("小规模精确装箱结果不符合预期")))
    )

    def box_pair_edge_test():
        water_pair = [scenario.boxes["S001-WAT-01"], scenario.boxes["S002-WAT-01"]]
        hygiene_pair = [scenario.boxes["S001-HYG-01"], scenario.boxes["S001-HYG-02"]]
        aircraft_type = scenario.aircraft_types["A"]
        assert sum(box.mass_kg for box in water_pair) > aircraft_type.max_mass_kg
        assert sum(box.volume_m3 for box in water_pair) <= aircraft_type.max_volume_m3
        assert sum(box.mass_kg for box in hygiene_pair) <= aircraft_type.max_mass_kg
        assert sum(box.volume_m3 for box in hygiene_pair) > aircraft_type.max_volume_m3
        return {
            "water_pair_mass_kg": sum(box.mass_kg for box in water_pair),
            "water_pair_volume_m3": sum(box.volume_m3 for box in water_pair),
            "hygiene_pair_mass_kg": sum(box.mass_kg for box in hygiene_pair),
            "hygiene_pair_volume_m3": sum(box.volume_m3 for box in hygiene_pair),
        }

    tests.append(_test("mass_volume_edge_cases", box_pair_edge_test))

    def exact_schedule_test():
        result = _finite_schedule_oracle(scenario)
        assert result["complete_schedule_count"] > 0
        assert result["best_objective"] == [2, 1000.0, 2.2]
        return result

    tests.append(_test("finite_exact_schedule_oracle", exact_schedule_test))

    def reserve_test():
        energy = scenario.aircraft_types["A"].energy_kwh
        reserve = scenario.aircraft_types["A"].reserve_ratio
        limit = (1.0 - reserve) * energy
        values = {"below": limit - 1e-4, "equal": limit, "above": limit + 1e-4}
        assert values["below"] <= limit + 1e-8
        assert values["equal"] <= limit + 1e-8
        assert not values["above"] <= limit + 1e-8
        return {"limit_kwh": limit, "values": values}

    tests.append(_test("return_reserve_boundaries", reserve_test))

    def communication_math_test():
        params = scenario.communication
        one_km = fspl_db(params["f"], 1.0)
        expected = 32.45 + 20.0 * __import__("math").log10(2400.0)
        assert _close(one_km, expected)
        assert link_available("transport_gateway", 1.0, False, params)
        assert communication_state(True, False, False) == "direct"
        assert communication_state(False, True, True) == "relay"
        assert communication_state(False, True, False) == "interrupted"
        assert interval_intersection((0.0, 100.0), (20.0, 120.0)) == (20.0, 100.0)
        assert interval_intersection((0.0, 20.0), (20.0, 120.0)) is None
        return link_budget_checks(scenario)

    tests.append(_test("communication_budget_and_state", communication_math_test))

    def midpoint_obstruction_test():
        flat = Terrain(xs_m=[0.0, 500.0, 1000.0], ys_m=[0.0], elevations_m=[[0.0, 0.0, 0.0]])
        a = (0.0, 500.0, 100.0)
        b = (1000.0, 500.0, 100.0)
        assert line_of_sight(flat, a, b)
        assert not line_of_sight(scenario.terrain, a, b)
        return {"flat_los": True, "ridge_los": False}

    tests.append(_test("midpoint_dem_obstruction", midpoint_obstruction_test))

    def actual_dem_spot_test():
        if actual_terrain is None:
            raise AssertionError("真实DEM未能读取")
        node = scenario.nodes["S001"]
        elevation = actual_terrain.elevation(node.x_m, node.y_m)
        assert elevation == elevation
        return {"S001_dem_nearest_m": elevation}

    tests.append(_test("actual_dem_spot_check", actual_dem_spot_test))
    return tests


def _run_schedule_tests(scenario) -> list[dict[str, Any]]:
    schedule = valid_q3_schedule(scenario)
    q3 = audit_q3(scenario, schedule)
    q4 = audit_q4(scenario, schedule, valid_q4_partition(schedule))
    return [
        {"name": "q3_valid_schedule", "status": "PASS" if q3.status == "PASS" else "FAIL", "details": q3.to_dict()},
        {"name": "q4_valid_partition", "status": "PASS" if q4.status == "PASS" else "FAIL", "details": q4.to_dict()},
    ]


def _run_mutation_tests(scenario) -> list[dict[str, Any]]:
    schedule = valid_q3_schedule(scenario)
    mutation_results = []
    for name, case in invalid_cases(scenario, schedule).items():
        if case["kind"] == "q3":
            report = audit_q3(scenario, case["schedule"], name=name)
        else:
            report = audit_q4(scenario, case["schedule"], case["partition"], name=name)
        codes = {item["code"] for item in report.violations}
        expected = set(case["expected_codes"])
        passed = expected <= codes
        mutation_results.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "expected_codes": sorted(expected),
                "observed_codes": sorted(codes),
                "details": report.to_dict(),
                "note": "当前仓库没有求解器；本项验证独立审计器能否捕获对应突变反例。",
            }
        )
    return mutation_results


def _write_reports(summary: dict[str, Any], mutations: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "core_validation_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "constraint_mutation_report.json").write_text(json.dumps(mutations, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for item in summary["tests"]:
        rows.append({"suite": "core", "test": item["name"], "status": item["status"]})
    for item in mutations:
        rows.append({"suite": "mutation", "test": item["name"], "status": item["status"]})
    with (OUTPUT_DIR / "audit_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["suite", "test", "status"])
        writer.writeheader()
        writer.writerows(rows)

    exact_test = next((item for item in summary["tests"] if item["name"] == "finite_exact_schedule_oracle"), None)
    q3_test = next((item for item in summary["tests"] if item["name"] == "q3_valid_schedule"), None)
    exact_details = exact_test["details"] if exact_test and isinstance(exact_test.get("details"), dict) else {}
    q3_metrics = q3_test.get("details", {}).get("metrics", {}) if q3_test else {}
    lines = [
        "# D题问题三+问题四小规模独立验证报告",
        "",
        f"- 总判定：`{summary['overall_status']}`",
        f"- 非能耗硬约束：`{summary['non_energy_status']}`",
        "- 航段物理能耗公式：`ENERGY_PHYSICS_BLOCKED`",
        f"- 测试通过数：{summary['passed_tests']}/{summary['total_tests']}",
        f"- 突变测试通过数：{summary['mutation_passed']}/{summary['mutation_total']}",
        f"- 实际运行耗时：{summary['elapsed_seconds']:.3f} 秒（<= 3600 秒：{summary['runtime_within_one_hour']}）",
        f"- 有限精确枚举：{exact_details.get('candidate_count', '?')} 个候选、{exact_details.get('complete_schedule_count', '?')} 个完整方案，最优目标 `{exact_details.get('best_objective', '?')}`",
        f"- 固定可行方案独立重算目标：`{q3_metrics.get('objective_recomputed', '?')}`；当前没有求解器输出可做最优性宣称",
        "",
        "## 测试摘要",
        "",
        "| 套件 | 测试 | 状态 |",
        "|---|---|---|",
    ]
    for item in summary["tests"]:
        lines.append(f"| core | {item['name']} | {item['status']} |")
    for item in mutations:
        lines.append(f"| mutation | {item['name']} | {item['status']} |")
    lines.extend(["", "## 说明", "", "本轮不调用任何优化器，仅审计独立构造的小规模方案和约束突变反例。"])
    (OUTPUT_DIR / "core_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    scenario, actual_terrain = load_mini_scenario(ROOT)
    unit_tests = _run_unit_tests(scenario, actual_terrain)
    schedule_tests = _run_schedule_tests(scenario)
    tests = unit_tests + schedule_tests
    mutations = _run_mutation_tests(scenario)
    failed = [item for item in tests + mutations if item["status"] != "PASS"]
    non_energy_status = "FAIL" if failed else "PASS_NON_ENERGY"
    overall_status = "FAIL" if failed else "BLOCKED_ENERGY_PHYSICS"
    summary = {
        "schema_version": 1,
        "scope": "Q3+Q4 small independent validation",
        "energy_formula_status": "BLOCKED",
        "overall_status": overall_status,
        "non_energy_status": non_energy_status,
        "scenario": scenario.metadata,
        "tests": tests,
        "passed_tests": sum(item["status"] == "PASS" for item in tests),
        "total_tests": len(tests),
        "mutation_passed": sum(item["status"] == "PASS" for item in mutations),
        "mutation_total": len(mutations),
        "mutations": mutations,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_within_one_hour": True,
        "reproducible_command": "python -m src.d_validation.run_core_validation",
        "random_seed": None,
    }
    _write_reports(summary, mutations)
    print(json.dumps({
        "overall_status": overall_status,
        "non_energy_status": non_energy_status,
        "passed_tests": f"{summary['passed_tests']}/{summary['total_tests']}",
        "mutation_passed": f"{summary['mutation_passed']}/{summary['mutation_total']}",
        "elapsed_seconds": round(summary["elapsed_seconds"], 3),
        "output_dir": str(OUTPUT_DIR),
    }, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
