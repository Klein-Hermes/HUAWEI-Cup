"""D 题能耗物理建模专项闭环验证。

本脚本对应能耗专项验证方案：

1. 在测试文件中独立重写最小航段几何、等效航程和能耗公式；
2. 对正常多点路线输出逐航段参考值、实现值和审计值；
3. 对真实 DEM 与合成平坦 DEM 各做一次端到端检查；
4. 构造返航余量必失败反例；
5. 检查载荷增加时能耗严格单调增加，并检查返航余量边界。

参考计算不调用 ``physics.leg_energy_transport`` 或 ``solver_small``，
避免“求解器和测试共同复用同一错误公式”。

用法（仓库根目录）：

    D:\\Anaconda\\envs\\huawei-cup\\python.exe -B src\\d_verify\\tests\\energy_physics_validation.py
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from d_verify import audit as audit_mod  # noqa: E402
from d_verify import params, physics  # noqa: E402
from d_verify.dem import RealDEM  # noqa: E402
from d_verify.plan import Schedule, TripPlan  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "D题_小规模验证"
G = 9.80665
J_PER_KWH = 3.6e6
DEPOT_AGL = 0.0
AREA_AGL = 30.0
CRUISE_CLEARANCE = 50.0
ABS_TOL = 1e-8


class FlatDEM:
    """平坦合成 DEM，用于隔离公式，不引入真实地形变化。"""

    def __init__(self, elevation: float = 600.0) -> None:
        self.base = float(elevation)

    def elevation(self, lat: float, lon: float) -> float:
        return self.base

    def max_elevation_along(self, p1, p2, sample_step_m: float = 15.0) -> float:
        return self.base

    def line_of_sight_clear(self, a, b, sample_step_m: float = 15.0) -> bool:
        return True


def haversine_ref(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """专项测试自己的水平大圆距离实现。"""
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def reference_leg(model, n1, n2, dem) -> dict:
    """独立参考航段几何，严格对应附录 2 的三段飞行口径。"""
    distance = haversine_ref(n1.lat, n1.lon, n2.lat, n2.lon)
    top = dem.max_elevation_along((n1.lat, n1.lon), (n2.lat, n2.lon))
    cruise = float(top) + CRUISE_CLEARANCE
    start_alt = n1.elevation + (DEPOT_AGL if n1.is_depot else AREA_AGL)
    end_alt = n2.elevation + (DEPOT_AGL if n2.is_depot else AREA_AGL)
    h_up = max(cruise - start_alt, 0.0)
    h_dn = max(cruise - end_alt, 0.0)
    return {
        "d_m": distance,
        "cruise_alt": cruise,
        "start_alt": start_alt,
        "end_alt": end_alt,
        "h_up": h_up,
        "h_dn": h_dn,
        "t_up": h_up / model.v_up,
        "t_cruise": distance / model.cruise_speed,
        "t_dn": h_dn / model.v_down,
    }


def equivalent_range_ref(model, q: float) -> float:
    if q < -ABS_TOL or q > model.max_payload + ABS_TOL:
        raise ValueError(f"payload out of range: {q}")
    q = min(max(q, 0.0), model.max_payload)
    return model.range_empty - (model.range_empty - model.range_full) * (
        q / model.max_payload
    ) ** 1.5


def reference_energy(model, geo: dict, q: float) -> float:
    e_horizontal = geo["d_m"] / equivalent_range_ref(model, q) * model.battery_energy
    e_up = (model.empty_mass + q) * G * geo["h_up"] / model.eta_up / J_PER_KWH
    return e_horizontal + e_up


def reference_trip(scn, model, seq: list[str], masses: list[float]) -> dict:
    nodes = [scn.depot] + [scn.areas[a] for a in seq] + [scn.depot]
    geos, rows = [], []
    q = sum(masses)
    total = 0.0
    for i in range(len(nodes) - 1):
        geo = reference_leg(model, nodes[i], nodes[i + 1], scn.dem)
        e = reference_energy(model, geo, q)
        geos.append(geo)
        rows.append({
            "from": nodes[i].id,
            "to": nodes[i + 1].id,
            "q": q,
            "d_m": geo["d_m"],
            "h_up": geo["h_up"],
            "h_dn": geo["h_dn"],
            "cruise_alt": geo["cruise_alt"],
            "e_horizontal": geo["d_m"] / equivalent_range_ref(model, q) * model.battery_energy,
            "e_up": e - geo["d_m"] / equivalent_range_ref(model, q) * model.battery_energy,
            "energy": e,
        })
        total += e
        if i < len(seq):
            q -= masses[i]
    return {"legs": rows, "total_energy": total}


def trip_plan(trip_id: str, model: str, seq: list[str], boxes: list[list[str]],
              uav: str, battery: str) -> TripPlan:
    return TripPlan(
        trip_id=trip_id,
        uav=uav,
        model=model,
        battery=battery,
        start_time=0.0,
        seq=list(seq),
        boxes_per_stop=[list(g) for g in boxes],
    )


def close(a: float, b: float, tol: float = ABS_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def run_check(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append({"name": name, "pass": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} | {detail}")


def normal_case(rows: list[dict], dem, label: str, segment_rows: list[dict]) -> None:
    scn = params.build_small_scenario(
        dem=dem,
        areas=["S001", "S015"],
        box_ids=["S001-WAT-01", "S015-WAT-01"],
        uavs=["U05"],
        battery_stock={"B": 2},
    )
    model = scn.transport_models["B"]
    seq = ["S001", "S015"]
    boxes = [["S001-WAT-01"], ["S015-WAT-01"]]
    masses = [scn.boxes_of(a)[0].mass for a in seq]
    ref = reference_trip(scn, model, seq, masses)
    tp = trip_plan("E-NORMAL", "B", seq, boxes, "U05", "B-1")
    sched = Schedule(name=f"energy-normal-{label}", trips=[tp])
    audited = audit_mod.audit(scn, sched, name=f"energy-normal-{label}")
    impl_rows = []
    nodes = [scn.depot] + [scn.areas[a] for a in seq] + [scn.depot]
    q = sum(masses)
    max_diff = 0.0
    for i, (n1, n2) in enumerate(zip(nodes, nodes[1:])):
        leg = physics.build_leg(model, n1, n2, dem)
        impl_energy = physics.leg_energy_transport(model, leg, q)
        expected = ref["legs"][i]
        max_diff = max(max_diff, abs(impl_energy - expected["energy"]))
        impl_rows.append({
            "case": label,
            "from": n1.id,
            "to": n2.id,
            "q": q,
            "reference_d_m": expected["d_m"],
            "implementation_d_m": leg.d_m,
            "reference_h_up": expected["h_up"],
            "implementation_h_up": leg.h_up,
            "reference_h_dn": expected["h_dn"],
            "implementation_h_dn": leg.h_dn,
            "reference_energy": expected["energy"],
            "implementation_energy": impl_energy,
        })
        q -= masses[i] if i < len(seq) else 0.0
    segment_rows.extend(impl_rows)
    audit_legs = audited.recomputed.get("trips", [{}])[0].get("legs", [])
    audit_diff = max(
        (abs(x["e"] - y["energy"]) for x, y in zip(audit_legs, ref["legs"])),
        default=0.0,
    )
    run_check(
        rows,
        f"{label}: 独立参考公式与 physics 逐航段一致",
        len(impl_rows) == 3 and max_diff < 1e-7,
        f"max_energy_diff={max_diff:.3e}",
    )
    run_check(
        rows,
        f"{label}: 独立审计器与参考能耗一致",
        audited.verdict == "PASS" and audit_diff < 1e-7,
        f"audit={audited.verdict}, max_energy_diff={audit_diff:.3e}, total_ref={ref['total_energy']:.6f}",
    )


def monotonic_case(rows: list[dict], dem) -> None:
    scn = params.build_small_scenario(dem=dem, areas=["S001"], box_ids=["S001-WAT-01"], uavs=["U01"], battery_stock={"A": 2})
    model = scn.transport_models["A"]
    node = scn.areas["S001"]
    nodes = [scn.depot, node, scn.depot]
    geos = [reference_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
    qs = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    ref_values, impl_values = [], []
    for q in qs:
        ref_values.append(reference_energy(model, geos[0], q) + reference_energy(model, geos[1], 0.0))
        impl_values.append(
            physics.leg_energy_transport(model, physics.build_leg(model, nodes[0], nodes[1], dem), q)
            + physics.leg_energy_transport(model, physics.build_leg(model, nodes[1], nodes[2], dem), 0.0)
        )
    ref_mono = all(a < b for a, b in zip(ref_values, ref_values[1:]))
    impl_mono = all(a < b for a, b in zip(impl_values, impl_values[1:]))
    max_diff = max(abs(a - b) for a, b in zip(ref_values, impl_values))
    run_check(rows, "载荷增加时独立参考能耗严格单调", ref_mono, f"E={','.join(f'{x:.6f}' for x in ref_values)}")
    run_check(rows, "physics 实现保持载荷能耗单调且匹配参考", impl_mono and max_diff < 1e-7, f"max_diff={max_diff:.3e}")


def reserve_boundary_case(rows: list[dict], dem) -> None:
    base = params.build_small_scenario(dem=dem, areas=["S001"], box_ids=["S001-WAT-01"], uavs=["U01"], battery_stock={"A": 2})
    model = base.transport_models["A"]
    limit = (1.0 - model.reserve_ratio) * model.battery_energy
    source = base.areas["S001"]
    selected = None
    for extra in [0.01 + 0.005 * i for i in range(30)]:
        far = replace(source, id="S999", lat=base.depot.lat + extra, lon=base.depot.lon + extra)
        nodes = [base.depot, far, base.depot]
        geos = [reference_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
        e0 = reference_energy(model, geos[0], 0.0) + reference_energy(model, geos[1], 0.0)
        eq = reference_energy(model, geos[0], model.max_payload) + reference_energy(model, geos[1], 0.0)
        if e0 < limit < eq:
            selected = (extra, far, e0, eq)
            break
    run_check(rows, "找到空载低于、满载高于返航余量的边界场景", selected is not None,
              "未找到交叉区间" if selected is None else f"extra={selected[0]:.3f}, E0={selected[2]:.6f}, EQ={selected[3]:.6f}, limit={limit:.6f}")
    if selected is None:
        return
    extra, far, _, _ = selected
    nodes = [base.depot, far, base.depot]
    geos = [reference_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
    lo, hi = 0.0, model.max_payload
    for _ in range(60):
        mid = (lo + hi) / 2.0
        e = reference_energy(model, geos[0], mid) + reference_energy(model, geos[1], 0.0)
        if e <= limit:
            lo = mid
        else:
            hi = mid
    q_star = lo
    e_star = reference_energy(model, geos[0], q_star) + reference_energy(model, geos[1], 0.0)
    run_check(rows, "返航余量等值点满足 E(q*)≈limit", abs(e_star - limit) < 1e-7,
              f"q*={q_star:.6f}, E(q*)={e_star:.9f}, limit={limit:.9f}")
    far_scn = params.build_small_scenario(dem=dem, areas=["S001"], box_ids=["S001-WAT-01"], uavs=["U01"], battery_stock={"A": 2})
    far_scn.areas = {"S999": far}
    far_box = replace(far_scn.boxes[0], id="S999-WAT-01", area="S999", mass=min(model.max_payload, q_star * 1.01))
    far_scn.boxes = [far_box]
    bad = trip_plan("E-RESERVE-FAIL", "A", ["S999"], [[far_box.id]], "U01", "A-1")
    audited = audit_mod.audit(far_scn, Schedule(name="energy-reserve-fail", trips=[bad]), name="energy-reserve-fail")
    run_check(rows, "略高于返航余量的样例被 checker 判 FAIL", audited.verdict == "FAIL" and audited.max_energy_violation > 0.0,
              f"verdict={audited.verdict}, energy_violation={audited.max_energy_violation:.6f}")


def main() -> int:
    started = time.time()
    checks: list[dict] = []
    segment_rows: list[dict] = []
    print("D 题能耗物理建模专项闭环验证")
    print("=" * 72)
    flat = FlatDEM(600.0)
    normal_case(checks, flat, "flat_synthetic", segment_rows)
    real_dem = RealDEM.load(params.DEM_MAT)
    normal_case(checks, real_dem, "real_dem", segment_rows)
    monotonic_case(checks, flat)
    reserve_boundary_case(checks, flat)
    passed = sum(1 for row in checks if row["pass"])
    failed = len(checks) - passed
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": "PASS" if failed == 0 else "FAIL",
        "summary": {"total": len(checks), "passed": passed, "failed": failed},
        "elapsed_s": round(time.time() - started, 3),
        "python": platform.python_version(),
        "reference_scope": "独立实现航段几何、等效航程、水平能耗、爬升能耗；不调用 physics 的能耗函数",
        "checks": checks,
        "segments": segment_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "能耗物理专项验证.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(segment_rows[0]) if segment_rows else []
    with (OUT_DIR / "能耗航段明细.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(segment_rows)
    md = [
        "# D 题能耗物理建模专项闭环验证",
        "",
        f"- 结论：**{payload['verdict']}**（{passed}/{len(checks)} 通过）",
        f"- 生成时间：{payload['generated_at']}",
        f"- 用时：{payload['elapsed_s']} s",
        "- 参考计算：测试文件内独立重写航段几何、等效航程、水平能耗和爬升能耗，不调用 `physics.leg_energy_transport`。",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    for row in checks:
        md.append(f"| {row['name']} | {'PASS' if row['pass'] else 'FAIL'} | {row['detail']} |")
    md += [
        "",
        "## 解释边界",
        "",
        "本专项验证证明当前代码与附录 2 统一口径在测试场景下逐航段一致，并能被独立审计器复核；它不证明水平能耗线性折算、爬升能耗按重力做功等未明写口径具有现实物理唯一性，也不替代完整 Q2/Q3/Q4 优化验证。",
    ]
    (OUT_DIR / "能耗物理专项验证.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("=" * 72)
    print(f"汇总：{passed}/{len(checks)} 通过，{failed} 失败，用时 {payload['elapsed_s']} s")
    print(f"结果：{OUT_DIR / '能耗物理专项验证.md'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
