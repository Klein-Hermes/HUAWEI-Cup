"""D 题小规模约束审计验证 —— 一体化运行器。

对应 ``D题_小规模约束审计验证方案.md``：

- 第 4 节基础公式与边界（质量/体积分离、充电公式边界、返航安全余量边界）；
- 第 5 节小规模实例（T1~T6 压缩版）；
- 第 6 节通信约束反例（地形遮挡、链路预算单位、直连/中继/中断优先级）；
- 第 7 节第 4 问任务分区（合法性 + 资源独立性两种口径）；
- 第 8 节约束突变测试（≥4 条，验证独立审计器仍能捕获违反）；
- 第 9/10.6 节 PASS/FAIL 汇总，写入 ``results/D题_小规模验证/``。

用法（仓库根目录）::

    python -B src/d_verify/tests/run_all.py

注意：本套件不使用任何外部优化求解器；小实例用自写精确枚举，并做一次
"独立暴力枚举 vs 求解器"的能耗对照。
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from d_verify import audit as audit_mod  # noqa: E402
from d_verify import comm, params, physics, solver_small  # noqa: E402
from d_verify.dem import RealDEM, SyntheticDEM  # noqa: E402
from d_verify.plan import RelayPlan, Schedule, TripPlan  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "D题_小规模验证"


# ---------------------------------------------------------------- 断言记录


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"group": group, "name": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {group} :: {name}" + (f" | {detail}" if detail else ""))
        return bool(ok)

    def check(self, group: str, name: str, cond: bool, detail: str = "") -> bool:
        return self.record(group, name, cond, detail)

    def close(self, group: str, name: str, value, expect, tol: float = 1e-6) -> bool:
        ok = abs(float(value) - float(expect)) <= tol
        return self.record(group, name, ok, f"实测 {value:.6f} / 期望 {expect}")

    def summary(self) -> dict:
        total = len(self.rows)
        passed = sum(1 for r in self.rows if r["pass"])
        return {"total": total, "passed": passed, "failed": total - passed}


REC = Recorder()


def _trip(tid: str, model: str, seq: list[str], boxes: list[list[str]],
          uav: str = "U01", battery: str | None = None, start: float = 0.0) -> TripPlan:
    return TripPlan(trip_id=tid, uav=uav, model=model,
                    battery=battery or f"{model}-1", start_time=start,
                    seq=list(seq), boxes_per_stop=[list(g) for g in boxes])


def _scn(**kw):
    """构造小实例（默认带真实 DEM）。"""
    return params.build_small_scenario(**kw)


# ---------------------------------------------------------------- 4.1 质量/体积


def test_41_mass_volume(G: str, dem) -> None:
    # 样例一：两个饮用水箱 28 kg / 0.054 m^3（A 型 25 kg / 0.06 m^3）→ 质量超限、体积合规
    sc = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01", "S001-WAT-02"],
              uavs=["U01"], battery_stock={"A": 2})
    sched = Schedule(name="mass-violation", trips=[
        _trip("T01", "A", ["S001"], [["S001-WAT-01", "S001-WAT-02"]])])
    res = audit_mod.audit(sc, sched, name="质量超限样例")
    REC.check(G, "两个饮用水箱：质量超限被捕获", res.verdict == "FAIL" and res.max_mass_violation > 2.9,
              f"mass_violation={res.max_mass_violation:.4f}")
    REC.check(G, "两个饮用水箱：体积本身合规", res.max_volume_violation < 1e-9,
              f"volume_violation={res.max_volume_violation:.6f}")

    # 样例二：两个生活卫生用品箱 12 kg / 0.070 m^3 → 质量合规、体积超限
    sc2 = _scn(dem=dem, areas=["S001"], box_ids=["S001-HYG-01", "S001-HYG-02"],
               uavs=["U01"], battery_stock={"A": 2})
    sched2 = Schedule(name="volume-violation", trips=[
        _trip("T01", "A", ["S001"], [["S001-HYG-01", "S001-HYG-02"]])])
    res2 = audit_mod.audit(sc2, sched2, name="体积超限样例")
    REC.check(G, "两个卫生用品箱：体积超限被捕获",
              res2.verdict == "FAIL" and abs(res2.max_volume_violation - 0.01) < 1e-6,
              f"volume_violation={res2.max_volume_violation:.6f}")
    REC.check(G, "两个卫生用品箱：质量本身合规", res2.max_mass_violation < 1e-9,
              f"mass_violation={res2.max_mass_violation:.6f}")


# ---------------------------------------------------------------- 4.2 充电公式


def test_42_charge(G: str) -> None:
    for s, exp in [(0.0, 1800.0), (0.5, 1150.0), (0.9, 630.0), (1.0, 0.0)]:
        REC.close(G, f"t_chg({s}) 边界", physics.charge_time(s, 1800.0), exp, tol=1e-6)
    a = physics.charge_time(0.8999999, 1800.0)
    b = physics.charge_time(0.9000001, 1800.0)
    REC.check(G, "s=0.9 两段公式连续", abs(a - b) < 1e-3, f"{a:.4f} vs {b:.4f}")
    REC.check(G, "充电时间非负", min(physics.charge_time(x / 100, 2400.0)
                                     for x in range(101)) >= 0.0)
    for bad in (-0.01, 1.01):
        try:
            physics.charge_time(bad, 1800.0)
            REC.check(G, f"拒绝非法 SOC {bad}", False, "未抛错")
        except ValueError:
            REC.check(G, f"拒绝非法 SOC {bad}", True)


# ---------------------------------------------------------------- 4.3 返航余量


def _find_over_limit_trip(sc, dem) -> tuple[TripPlan, float, float] | None:
    """（诊断用）扫描"服务区 x 机型 x 该区全部货箱"找出超出返航安全余量的自然组合。

    实测：真实服务区距离最远约 7 km，各机型往返能耗均未触及返航余量上限，
    因此边界与突变测试改用 :func:`_far_trip` 的远距虚拟服务区构造。
    """
    for area in sorted(sc.areas):
        boxes = sc.boxes_of(area)
        if not boxes:
            continue
        for mid, model in sc.transport_models.items():
            mass = sum(b.mass for b in boxes)
            volume = sum(b.volume for b in boxes)
            if mass > model.max_payload + 1e-9 or volume > model.volume + 1e-9:
                continue
            nodes = [sc.depot, sc.areas[area], sc.depot]
            legs = [physics.build_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
            q, energy = mass, 0.0
            for i, leg in enumerate(legs):
                energy += physics.leg_energy_transport(model, leg, q)
                q = 0.0
            limit = (1 - model.reserve_ratio) * model.battery_energy
            if energy > limit + 1e-9:
                uav = {"A": "U01", "B": "U05", "C": "U07"}.get(mid, "U01")
                tp = _trip("T01", mid, [area], [[b.id for b in boxes]], uav=uav)
                return tp, energy, limit
    return None


def _far_trip(dem, model_id: str = "A", extra_deg: float = 0.15,
              uav: str = "U01"):
    """构造"远距虚拟服务区"样例（S999）：用于返航安全余量的可行性/不可行边界测试。

    做法：把服务区坐标外推、货箱随服务区一起改名，保证审计器的"货箱-服务区一致性"
    检查仍然成立；这样可以在不修改物理参数的前提下得到必然超出返航余量的架次。
    """
    from dataclasses import replace as _replace

    sc = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01"], uavs=[uav],
              battery_stock={model_id: 2})
    src = sc.areas["S001"]
    far = _replace(src, id="S999", lat=sc.depot.lat + extra_deg,
                   lon=sc.depot.lon + extra_deg)
    box = _replace(sc.boxes[0], id="S999-WAT-01", area="S999")
    sc.areas = {"S999": far}
    sc.boxes = [box]
    tp = _trip("T01", model_id, ["S999"], [["S999-WAT-01"]], uav=uav)
    model = sc.transport_models[model_id]
    limit = (1 - model.reserve_ratio) * model.battery_energy
    return sc, tp, _energy_of(sc, dem, model_id, "S999", box.mass), limit


def _energy_of(sc, dem, model_id: str, area: str, q: float) -> float:
    """给定服务区与初始载荷的单点往返能耗（仅供测试构造使用）。"""
    model = sc.transport_models[model_id]
    nodes = [sc.depot, sc.areas[area], sc.depot]
    legs = [physics.build_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
    e, qq = 0.0, q
    for i, leg in enumerate(legs):
        e += physics.leg_energy_transport(model, leg, qq)
        qq = 0.0
    return e


def test_43_reserve(G: str, sc_full, dem) -> None:
    # 边界单调性：二分找 E(q*) = limit，验证略低可行 / 略高不可行
    model = sc_full.transport_models["A"]
    node = sc_full.areas["S015"]
    nodes = [sc_full.depot, node, sc_full.depot]
    legs = [physics.build_leg(model, nodes[i], nodes[i + 1], dem) for i in range(2)]
    limit = (1 - model.reserve_ratio) * model.battery_energy

    def energy_of(q: float) -> float:
        e, qq = 0.0, q
        for i, leg in enumerate(legs):
            e += physics.leg_energy_transport(model, leg, qq)
            qq = 0.0
        return e

    # 真实服务区距离近，A 型往返不会触限 → 记录该事实
    REC.check(G, "真实最近远区（S015）A 型满载不触限",
              energy_of(model.max_payload) <= limit + 1e-9,
              f"满载 {energy_of(model.max_payload):.6f} <= limit {limit:.6f}")

    # 边界三分支：在"远距虚拟服务区"上自适应选距离，使 E(0) < limit < E(Q)
    chosen = None
    for extra in (0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09, 0.10):
        sc_far, tp_far, e_at_box, limit_far = _far_trip(dem, "A", extra)
        e0 = _energy_of(sc_far, dem, "A", "S999", 0.0)
        e_full = _energy_of(sc_far, dem, "A", "S999", model.max_payload)
        if e0 < limit_far < e_full:
            chosen = (sc_far, tp_far, limit_far, e0, e_full, extra)
            break
    if chosen is None:
        REC.check(G, "找到跨越返航余量上限的载荷区间", False, "自适应扫描未命中，需调整 extra_deg 候选")
    else:
        sc_far, tp_far, limit_far, e0, e_full, extra = chosen
        REC.check(G, "找到跨越返航余量上限的载荷区间", True,
                  f"extra={extra}°：E(0)={e0:.4f} < limit={limit_far:.4f} < E(Q)={e_full:.4f}")

        def energy_far(q: float) -> float:
            return _energy_of(sc_far, dem, "A", "S999", q)

        lo, hi = 0.0, model.max_payload
        for _ in range(60):
            mid_q = (lo + hi) / 2
            if energy_far(mid_q) <= limit_far:
                lo = mid_q
            else:
                hi = mid_q
        q_star = lo
        REC.check(G, "等于上限的载荷点存在（E(q*)≈limit）",
                  abs(energy_far(q_star) - limit_far) < 1e-3,
                  f"E(q*)={energy_far(q_star):.6f} limit={limit_far:.6f}")
        REC.check(G, "略低于上限可行", energy_far(q_star * 0.99) <= limit_far + 1e-9,
                  f"E(0.99q*)={energy_far(q_star * 0.99):.6f}")
        REC.check(G, "略高于上限不可行", energy_far(min(q_star * 1.01, model.max_payload)) > limit_far,
                  f"E(1.01q*)={energy_far(min(q_star * 1.01, model.max_payload)):.6f}")

        # 方案级：同一"远距服务区"上的真实货箱方案必须被判 FAIL
        res_far = audit_mod.audit(sc_far, Schedule(name="reserve-far", trips=[tp_far]),
                                  name="远距返航余量")
        REC.check(G, "超出返航余量的远距方案被审计判 FAIL",
                  res_far.verdict == "FAIL" and res_far.max_energy_violation > 1e-6,
                  f"energy_violation={res_far.max_energy_violation:.4f}")

    # 正例：近服务区单箱方案必须 PASS
    sc_ok = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01"], uavs=["U01"],
                 battery_stock={"A": 2})
    ok_sched = Schedule(name="reserve-ok", trips=[_trip("T01", "A", ["S001"], [["S001-WAT-01"]])])
    res_ok = audit_mod.audit(sc_ok, ok_sched, name="返航余量合规")
    REC.check(G, "合规方案通过审计", res_ok.verdict == "PASS",
              f"violations={res_ok.hard_violation_count}")


# ---------------------------------------------------------------- 多点载荷递减


def test_multi_stop_payload(G: str, dem) -> None:
    sc = _scn(dem=dem, areas=["S001", "S015"], uavs=["U05"],
              box_ids=["S001-WAT-01", "S015-WAT-01"], battery_stock={"B": 2})
    tp = _trip("T01", "B", ["S001", "S015"], [["S001-WAT-01"], ["S015-WAT-01"]])
    sched = Schedule(name="multi-stop", trips=[tp])
    res = audit_mod.audit(sc, sched, name="多点载荷递减")
    REC.check(G, "多点架次通过审计", res.verdict == "PASS", f"violations={res.hard_violation_count}")
    legs = res.recomputed["trips"][0]["legs"] if res.recomputed.get("trips") else []
    if len(legs) == 3:
        qs = [leg["q"] for leg in legs]
        REC.check(G, "载荷按投递逐段递减且返航为 0",
                  abs(qs[0] - 28.0) < 1e-6 and abs(qs[1] - 14.0) < 1e-6 and abs(qs[2]) < 1e-6,
                  f"各段载荷={qs}")
    else:
        REC.check(G, "多点架次生成 3 个航段", False, f"实际 {len(legs)} 段")


# ---------------------------------------------------------------- 资源与时限


def test_resources_and_deadlines(G: str, dem) -> None:
    # 无人机时段重叠
    sc = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01", "S001-WAT-02"],
              uavs=["U01"], battery_stock={"A": 2})
    sched = Schedule(name="uav-overlap", trips=[
        _trip("T01", "A", ["S001"], [["S001-WAT-01"]], battery="A-1", start=0.0),
        _trip("T02", "A", ["S001"], [["S001-WAT-02"]], battery="A-2", start=10.0),
    ])
    res = audit_mod.audit(sc, sched, name="无人机时段重叠")
    REC.check(G, "同一无人机时段重叠被捕获",
              res.verdict == "FAIL" and res.resource_overlap_count >= 1,
              f"overlaps={res.resource_overlap_count}")

    # 电池未充满即复用
    sc2 = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01", "S001-WAT-02"],
               uavs=["U01", "U02"], battery_stock={"A": 1})
    sched2 = Schedule(name="battery-reuse", trips=[
        _trip("T01", "A", ["S001"], [["S001-WAT-01"]], uav="U01", battery="A-1", start=0.0),
        _trip("T02", "A", ["S001"], [["S001-WAT-02"]], uav="U02", battery="A-1", start=1.0),
    ])
    res2 = audit_mod.audit(sc2, sched2, name="电池未充满复用")
    REC.check(G, "电池未充满即复用被捕获",
              res2.verdict == "FAIL" and res2.resource_overlap_count >= 1,
              f"overlaps={res2.resource_overlap_count}")

    # 首批货箱超时（医疗物资）
    sc3 = _scn(dem=dem, areas=["S001"], box_ids=["S001-MED-01"], uavs=["U01"],
               battery_stock={"A": 2})
    sched3 = Schedule(name="deadline-violation", trips=[
        _trip("T01", "A", ["S001"], [["S001-MED-01"]], start=3600.0)])
    res3 = audit_mod.audit(sc3, sched3, name="首批货箱超时")
    REC.check(G, "首批/医疗时限超时被捕获",
              res3.verdict == "FAIL" and res3.max_time_violation > 1.0,
              f"time_violation={res3.max_time_violation:.2f}s")

    # 交付时刻按实际到达计算（多点架次：第二站交付晚于第一站）
    sc4 = _scn(dem=dem, areas=["S001", "S015"], uavs=["U05"],
               box_ids=["S001-WAT-01", "S015-WAT-01"], battery_stock={"B": 2})
    tp4 = _trip("T01", "B", ["S001", "S015"], [["S001-WAT-01"], ["S015-WAT-01"]])
    res4 = audit_mod.audit(sc4, Schedule(name="deliver-times", trips=[tp4]), name="交付时刻")
    dv = res4.recomputed["trips"][0]["deliver"] if res4.recomputed.get("trips") else {}
    REC.check(G, "逐箱交付时刻按到达各服务区计算", len(dv) == 2 and dv.get("S015", 0) > dv.get("S001", 0),
              f"deliver={dv}")


# ---------------------------------------------------------------- 6 通信反例


def test_comm_units_and_priority(G: str, sc_full) -> None:
    REC.close(G, "FSPL(2400MHz, 1km) 手算对照", comm.free_space_loss(2400.0, 1.0), 100.054, tol=1e-3)
    REC.close(G, "FSPL 距离=10km", comm.free_space_loss(2400.0, 10.0),
              100.054 + 20.0, tol=1e-3)
    REC.close(G, "直连双向门限", comm.bidirectional_threshold(sc_full.link, comm.DIRECT_LINK),
              122.0, tol=1e-6)
    REC.close(G, "中继接入双向门限", comm.bidirectional_threshold(sc_full.link, comm.ACCESS_LINK),
              116.0, tol=1e-6)
    REC.close(G, "中继回传双向门限", comm.bidirectional_threshold(sc_full.link, comm.BACKHAUL_LINK),
              126.0, tol=1e-6)

    # 直连 / 中继 / 中断 三态优先级（无地形遮盖，用距离控制）
    link = sc_full.link
    g01 = (sc_full.depot.lat, sc_full.depot.lon, sc_full.depot.elevation + link.gateway_agl)
    near = (sc_full.depot.lat + 0.001, sc_full.depot.lon, 200.0)
    st, rid = comm.transport_comm_state(link, None, near, g01, [])
    REC.check(G, "近距直连可用 -> 直连状态", st == comm.CommState.DIRECT, f"{st}")
    far = (sc_full.depot.lat + 0.1176, sc_full.depot.lon, 400.0)   # 约 13 km，直连超门限
    st2, _ = comm.transport_comm_state(link, None, far, g01, [])
    REC.check(G, "远距且无中继 -> 中断状态", st2 == comm.CommState.OUTAGE, f"{st2}")
    # 中继置于距 G01 约 8 km、距运输机约 5 km 处（接入 116 dB / 回传 126 dB 门限内）
    relay_pos = ("R01", (sc_full.depot.lat + 0.0724, sc_full.depot.lon, 600.0))
    st3, rid3 = comm.transport_comm_state(link, None, far, g01, [relay_pos])
    REC.check(G, "远距但有中继在线 -> 中继状态", st3 == comm.CommState.RELAYED, f"{st3}/{rid3}")


def _ridge_scenario(dem_base, mid_lat, mid_lon, sigma=0.010, height=600.0, **kw):
    """构造带山脊的合成 DEM 小实例（山脊位于 O01 与 S001 之间）。"""
    dem = SyntheticDEM(0.0, 0.0, 0.0, 0.0, base_elev=120.0)
    dem.add_ridge(mid_lat, mid_lon, sigma, height)
    return _scn(dem=dem, **kw), dem


def test_comm_continuity(G: str, sc_full) -> None:
    dep, tgt = sc_full.depot, sc_full.areas["S015"]      # 相距约 6.0 km
    mid_lat = (dep.lat + tgt.lat) / 2
    mid_lon = (dep.lon + tgt.lon) / 2
    sc, dem = _ridge_scenario(None, mid_lat, mid_lon, sigma=0.012, height=600.0,
                              areas=["S015"], box_ids=["S015-WAT-01"], uavs=["U01"],
                              battery_stock={"A": 2})
    cands = solver_small.build_candidates(sc, sc.boxes, single_area=True)
    if not cands:
        REC.check(G, "山脊场景可构造架次候选", False, "无候选")
        return
    cand = cands[0]
    tp = _trip("T01", cand.model, list(cand.seq), [list(g) for g in cand.stops_boxes])

    # 1) 直连可用性：起点（O01 上空）可用；服务区低空因遮挡 + 距离超门限而不可用
    g01 = (dep.lat, dep.lon, dep.elevation + sc.link.gateway_agl)
    p_start = (dep.lat, dep.lon, dep.elevation)
    p_low = (tgt.lat, tgt.lon, tgt.elevation + physics.AREA_WORK_ALT_AGL)
    ok_start = comm.link_available(sc.link, comm.DIRECT_LINK, dem, p_start, g01)
    ok_low = comm.link_available(sc.link, comm.DIRECT_LINK, dem, p_low, g01)
    REC.check(G, "航段起点（O01 上空）直连可用", ok_start, "起飞时刻无遮挡")
    REC.check(G, "服务区低空（30m AGL）直连不可用（遮挡 + 距离）", not ok_low,
              f"alt={p_low[2]:.1f}, cruise={cand.legs[0].cruise_alt:.1f}")

    # 2) 审计器连续检查必须发现空档；粗粒度（近似只查端点）空档数不增加
    sched = Schedule(name="ridge-outage", trips=[tp])
    res = audit_mod.audit(sc, sched, name="山脊遮挡", require_comm=True, comm_sample_dt=5.0)
    REC.check(G, "连续通信审计发现空档（FAIL）",
              res.verdict == "FAIL" and res.communication_gap_count > 0,
              f"gaps={res.communication_gap_count}")
    gaps_coarse = audit_mod._audit_communication(sc, sched, 1e9)
    REC.check(G, "粗粒度（近似只看两端）检查空档数不增加",
              gaps_coarse <= res.communication_gap_count,
              f"端点级 gaps={gaps_coarse} vs 细粒度 {res.communication_gap_count}")

    # 3) 中继覆盖：中继悬停于山脊上方，覆盖越完整空档越少
    hover_alt = dem.elevation(mid_lat, mid_lon) + 300.0

    def _relay_sched(tag: str, frac: float) -> Schedule:
        return Schedule(name=tag, trips=[tp], relays=[RelayPlan(
            relay_id="R1", relay_uav="R01", energy_unit="R-1", start_time=0.0,
            hover_lat=mid_lat, hover_lon=mid_lon,
            hover_elev=dem.elevation(mid_lat, mid_lon), hover_alt=hover_alt,
            service_start=-1.0, service_end=cand.duration * frac)])

    g_none = res.communication_gap_count
    g_full = audit_mod._audit_communication(sc, _relay_sched("relay-full", 1.0), 5.0)
    g_part = audit_mod._audit_communication(sc, _relay_sched("relay-partial", 0.5), 5.0)
    REC.check(G, "中继覆盖越完整通信空档越少（单调）", g_full <= g_part <= g_none,
              f"全覆盖={g_full} / 半程={g_part} / 无中继={g_none}")
    REC.check(G, "中继全程覆盖显著减少空档", g_full < g_none, f"{g_full} < {g_none}")


# ---------------------------------------------------------------- 7 分区


def test_partitions(G: str, dem) -> None:
    sc = _scn(dem=dem, areas=["S001", "S015"], uavs=["U01", "U02"],
              box_ids=["S001-WAT-01", "S015-WAT-01"], battery_stock={"A": 2})
    tp = _trip("T01", "A", ["S001", "S015"], [["S001-WAT-01"], ["S015-WAT-01"]],
               uav="U01", battery="A-1", start=0.0)

    # 7.1 非法分区：同架次涉及的两个服务区被划到不同任务组 → 必须 FAIL
    bad = Schedule(name="partition-bad", trips=[tp],
                   partitions={"2": {1: ["S001"], 2: ["S015"]}})
    res_bad = audit_mod.audit(sc, bad, name="非法分区")
    REC.check(G, "跨任务组的同一架次被捕获",
              res_bad.verdict == "FAIL" and res_bad.partition_violation_count >= 1,
              f"partition_violations={res_bad.partition_violation_count}")

    # 分区必须覆盖全部服务区、每组非空
    bad2 = Schedule(name="partition-bad2", trips=[tp],
                    partitions={"2": {1: ["S001", "S015"], 2: []},
                                "3": {1: ["S001"], 2: ["S015"], 3: ["S001"]}})
    res_bad2 = audit_mod.audit(sc, bad2, name="分区覆盖不完整")
    REC.check(G, "空任务组与重复服务区被捕获",
              res_bad2.verdict == "FAIL" and res_bad2.partition_violation_count >= 2,
              f"partition_violations={res_bad2.partition_violation_count}")

    # 7.2 资源独立性：同一架无人机跨两个任务组（时间不重叠）→ 独立执行必须各配 1 架
    tpa = _trip("T01", "A", ["S001"], [["S001-WAT-01"]], uav="U01", battery="A-1", start=0.0)
    tpb = _trip("T02", "A", ["S015"], [["S015-WAT-01"]], uav="U01", battery="A-2", start=2e5)
    sched_cross = Schedule(name="resource-independence", trips=[tpa, tpb],
                           partitions={"2": {1: ["S001"], 2: ["S015"]}})
    res_cross = audit_mod.audit(sc, sched_cross, name="资源独立核算")
    parts = res_cross.recomputed.get("partitions", {}).get("2", {})
    if parts:
        g1, g2 = parts.get(1, {}), parts.get(2, {})
        color_total = g1.get("uav_min_by_coloring", 0) + g2.get("uav_min_by_coloring", 0)
        global_ids = len({t.uav for t in sched_cross.trips})
        REC.check(G, "合法分区本身通过（无分区违反）", res_cross.partition_violation_count == 0,
                  f"partition_violations={res_cross.partition_violation_count}")
        REC.check(G, "资源独立性：只数 Q3 编号会低估（1 < 2）",
                  global_ids == 1 and color_total == 2,
                  f"Q3 用过的无人机编号数={global_ids}，独立执行着色需求={color_total}")
    else:
        REC.check(G, "分区资源核算输出", False, "未产生分区核算结果")


# ---------------------------------------------------------------- 5 小实例


def _brute_force_min_energy(scn, boxes, model_ids) -> float | None:
    """独立暴力枚举：最小总能耗（能耗只取决于分组与机型，与时间安排无关）。"""
    best = None

    def parts(items):
        if not items:
            return [[]]
        first, rest = items[0], items[1:]
        out = []
        for p in parts(rest):
            for i in range(len(p)):
                out.append(p[:i] + [[first] + p[i]] + p[i + 1:])
            out.append([[first]] + p)
        return out

    for part in parts(list(boxes)):
        for combo in itertools.product(model_ids, repeat=len(part)):
            total, ok = 0.0, True
            for block, mid in zip(part, combo):
                model = scn.transport_models[mid]
                mass = sum(b.mass for b in block)
                vol = sum(b.volume for b in block)
                if mass > model.max_payload + 1e-9 or vol > model.volume + 1e-9:
                    ok = False
                    break
                areas = sorted({b.area for b in block})
                nodes = [scn.depot] + [scn.areas[a] for a in areas] + [scn.depot]
                legs = [physics.build_leg(model, nodes[i], nodes[i + 1], scn.dem)
                        for i in range(len(nodes) - 1)]
                q, e = mass, 0.0
                for i, leg in enumerate(legs):
                    e += physics.leg_energy_transport(model, leg, q)
                    if i < len(areas):
                        q -= sum(b.mass for b in block if b.area == areas[i])
                limit = (1 - model.reserve_ratio) * model.battery_energy
                if e > limit + 1e-9:
                    ok = False
                    break
                total += e
            if ok and (best is None or total < best):
                best = total
    return best


def test_t1_exact(G: str, dem) -> None:
    box_ids = ["S001-MED-01", "S001-WAT-01", "S001-FOD-01"]
    sc = _scn(dem=dem, areas=["S001"], box_ids=box_ids, uavs=["U01", "U05", "U07"],
              battery_stock={"A": 2, "B": 2, "C": 2})
    sched = solver_small.solve_exact(sc, name="T1", single_area=True)
    if sched is None:
        REC.check(G, "T1 求解器找到可行解", False, "无解")
        return
    REC.check(G, "T1 求解器找到可行解", True,
              f"架次={len(sched.trips)} 能耗={sched.reported.get('total_energy'):.4f} kWh")
    res = audit_mod.audit(sc, sched, name="T1")
    REC.check(G, "T1 独立审计通过", res.verdict == "PASS",
              f"violations={res.hard_violation_count}")
    REC.close(G, "T1 目标重算误差", res.objective_recompute_error, 0.0, tol=1e-6)

    brute = _brute_force_min_energy(sc, sc.boxes, sorted(sc.transport_models))
    if brute is None:
        REC.check(G, "T1 暴力枚举找到可行解", False, "无解")
    else:
        REC.check(G, "T1 求解器能耗 = 暴力枚举最小能耗",
                  abs(sched.reported.get("total_energy", -1) - brute) < 1e-6,
                  f"求解器 {sched.reported.get('total_energy'):.6f} vs 暴力 {brute:.6f}")


def test_t3_charge_turnaround(G: str, dem) -> None:
    sc = _scn(dem=dem, areas=["S001", "S015"],
              box_ids=["S001-WAT-01", "S001-WAT-02", "S015-WAT-01"],
              uavs=["U05"], battery_stock={"B": 1})
    sched = solver_small.solve_exact(sc, name="T3")
    if sched is None:
        REC.check(G, "T3（1 机 1 电池）找到可行解", False, "无解")
        return
    res = audit_mod.audit(sc, sched, name="T3")
    REC.check(G, "T3 独立审计通过", res.verdict == "PASS", f"violations={res.hard_violation_count}")
    trips = sorted(res.recomputed.get("trips", []), key=lambda t: t["start"])
    REC.check(G, "T3 单电池架次串行（无重叠）", res.resource_overlap_count == 0,
              f"overlaps={res.resource_overlap_count}")
    if len(trips) >= 2:
        model = sc.transport_models["B"]
        chg = physics.charge_time(max(trips[0]["soc_end"], 0.0), sc.battery_charge_full["B"])
        gap = trips[1]["start"] - trips[0]["end"]
        # audit 导出的架次时刻保留到 1 ms；比较时给出 1 ms 数值容差，
        # 避免“实际只早 0.082 ms”被舍入误差误判为未计入充电周转。
        turnaround_tol = 1e-3
        REC.check(G, "T3 第二架次开始时刻已计入充电周转", gap >= chg - turnaround_tol,
                  f"间隔 {gap:.3f}s >= 充电 {chg:.3f}s（差值 {gap - chg:.6f}s）")


def test_small_instance_full(G: str, dem) -> None:
    sc = _scn(dem=dem)
    t0 = time.time()
    # 统一小实例（8 箱 / 3 服务区 / A·B·C 各 1 架）：每块保留 2 个候选、最多 3 架次
    sched = solver_small.solve_exact(sc, name="small-8box", max_trips=3,
                                     max_candidates_per_block=2)
    dt = time.time() - t0
    if sched is None:
        REC.check(G, "统一小实例（8 箱）找到可行解", False, "无解")
        return
    REC.check(G, "统一小实例（8 箱）找到可行解", True,
              f"架次={len(sched.trips)} 用时={dt:.1f}s（受限枚举：每块保留 2 个候选、最多 3 架次）")
    res = audit_mod.audit(sc, sched, name="small-8box")
    REC.check(G, "统一小实例独立审计通过", res.verdict == "PASS",
              f"violations={res.hard_violation_count}, 细节={res.details[:2]}")
    REC.check(G, "统一小实例货箱无遗漏/重复",
              res.box_missing_count == 0 and res.box_duplicate_count == 0,
              f"missing={res.box_missing_count}, duplicate={res.box_duplicate_count}")


# ---------------------------------------------------------------- 8 突变


def test_mutations(G: str, dem, sc_full) -> None:
    """约束突变测试（方案第 8 节）：M1 体积、M2 返航余量、M3 充电重叠、M4 时限、M5 通信端点。"""
    # M1：删除体积约束（A 型体积 0.06 m³；两个卫生用品箱合计 0.07 m³）
    sc = _scn(dem=dem, areas=["S001"], box_ids=["S001-HYG-01", "S001-HYG-02"],
              uavs=["U01"], battery_stock={"A": 2})
    normal = solver_small.solve_exact(sc, name="mut-volume-normal", single_area=True)
    relaxed = solver_small.solve_exact(sc, name="mut-volume-relaxed", single_area=True,
                                       relax=frozenset({"volume"}))
    REC.check(G, "M1 原模型有解且必须拆箱（架次数>=2）",
              normal is not None and len(normal.trips) >= 2,
              f"架次数={len(normal.trips) if normal else '无解'}")
    if normal is not None:
        res_n = audit_mod.audit(sc, normal, name="M1 原模型方案审计")
        REC.check(G, "M1 原模型方案满足体积约束（审计 PASS）",
                  res_n.verdict == "PASS" and res_n.max_volume_violation < 1e-9,
                  f"volume_violation={res_n.max_volume_violation:.6f}")
    REC.check(G, "M1 删除体积约束后模型合批（架次数=1）：输出确已变化",
              relaxed is not None and len(relaxed.trips) == 1,
              f"架次数={len(relaxed.trips) if relaxed else '无解'}")
    if relaxed is not None:
        res = audit_mod.audit(sc, relaxed, name="M1 突变方案审计")
        REC.check(G, "M1 独立审计仍捕获体积违反",
                  res.verdict == "FAIL" and res.max_volume_violation > 0.009,
                  f"volume_violation={res.max_volume_violation:.6f}")

    # M2：删除返航安全余量（远距虚拟服务区 S999 → 审计必须捕获）
    sc2, tp2, _, _ = _far_trip(dem, "A", 0.15)
    res2 = audit_mod.audit(sc2, Schedule(name="M2", trips=[tp2]), name="M2")
    REC.check(G, "M2 删除返航余量后审计仍捕获",
              res2.verdict == "FAIL" and res2.max_energy_violation > 1e-6,
              f"energy_violation={res2.max_energy_violation:.4f}")

    # M3：删除电池充电不重叠约束
    sc3 = _scn(dem=dem, areas=["S001"], box_ids=["S001-WAT-01", "S001-WAT-02"],
               uavs=["U01", "U02"], battery_stock={"A": 1})
    sched3 = Schedule(name="M3", trips=[
        _trip("T01", "A", ["S001"], [["S001-WAT-01"]], uav="U01", battery="A-1", start=0.0),
        _trip("T02", "A", ["S001"], [["S001-WAT-02"]], uav="U02", battery="A-1", start=5.0)])
    res3 = audit_mod.audit(sc3, sched3, name="M3")
    REC.check(G, "M3 删除充电约束后审计仍捕获",
              res3.verdict == "FAIL" and res3.resource_overlap_count >= 1,
              f"overlaps={res3.resource_overlap_count}")

    # M4：删除医疗/首批时限
    sc4 = _scn(dem=dem, areas=["S001"], box_ids=["S001-MED-01"], uavs=["U01"],
               battery_stock={"A": 2})
    sched4 = Schedule(name="M4", trips=[
        _trip("T01", "A", ["S001"], [["S001-MED-01"]], start=7200.0)])
    res4 = audit_mod.audit(sc4, sched4, name="M4")
    REC.check(G, "M4 删除时限约束后审计仍捕获",
              res4.verdict == "FAIL" and res4.max_time_violation > 0.0,
              f"time_violation={res4.max_time_violation:.1f}s")

    # M5：只检查通信端点（不检查中间时刻）
    dep, tgt5 = sc_full.depot, sc_full.areas["S015"]
    mid_lat, mid_lon = (dep.lat + tgt5.lat) / 2, (dep.lon + tgt5.lon) / 2
    sc5, dem5 = _ridge_scenario(None, mid_lat, mid_lon, sigma=0.012, height=600.0,
                                areas=["S015"], box_ids=["S015-WAT-01"], uavs=["U01"],
                                battery_stock={"A": 2})
    cands = solver_small.build_candidates(sc5, sc5.boxes, single_area=True)
    if cands:
        cand = cands[0]
        tp5 = _trip("T01", cand.model, list(cand.seq), [list(g) for g in cand.stops_boxes])
        sched5 = Schedule(name="M5", trips=[tp5])
        gaps_fine = audit_mod._audit_communication(sc5, sched5, 5.0)
        gaps_coarse = audit_mod._audit_communication(sc5, sched5, 1e9)
        REC.check(G, "M5 细粒度检查发现端点检查漏掉的空档",
                  gaps_fine > gaps_coarse,
                  f"细粒度 gaps={gaps_fine} > 端点 gaps={gaps_coarse}")
    else:
        REC.check(G, "M5 构造遮挡场景", False, "无候选")


# ---------------------------------------------------------------- 主流程


def main() -> int:
    t_start = time.time()
    print("=" * 78)
    print("D 题小规模约束审计验证（方案第 10 节核心版）")
    print("=" * 78)

    dem = RealDEM.load(params.DEM_MAT)
    sc_full = params.load_scenario(dem=dem)
    print(f"[信息] 真实 DEM {dem.rows}x{dem.cols}，15 服务区 / {len(sc_full.boxes)} 货箱 / "
          f"{len(sc_full.transport_uavs)} 架运输机 / {len(sc_full.relay_uavs)} 架中继机")

    stages = [
        ("4.1 质量与体积分离", lambda: test_41_mass_volume("4.1", dem)),
        ("4.2 充电公式边界", lambda: test_42_charge("4.2")),
        ("4.3 返航安全余量边界", lambda: test_43_reserve("4.3", sc_full, dem)),
        ("4.3b 多点载荷递减", lambda: test_multi_stop_payload("4.3b", dem)),
        ("2.2 时间/资源/时限", lambda: test_resources_and_deadlines("2.2", dem)),
        ("6.3 链路预算与状态优先级", lambda: test_comm_units_and_priority("6.3", sc_full)),
        ("6.1/6.2 连续通信与中继", lambda: test_comm_continuity("6.1", sc_full)),
        ("7 任务分区合法性", lambda: test_partitions("7", dem)),
        ("5/T1 精确枚举对照", lambda: test_t1_exact("5/T1", dem)),
        ("5/T3 充电周转", lambda: test_t3_charge_turnaround("5/T3", dem)),
        ("5/统一小实例", lambda: test_small_instance_full("5/small", dem)),
        ("8 约束突变", lambda: test_mutations("8", dem, sc_full)),
    ]
    for label, fn in stages:
        t0 = time.time()
        try:
            fn()
        except Exception:      # 测试自身异常也必须计入失败，不能静默
            REC.record(label, "测试执行异常", False, traceback.format_exc(limit=2).replace("\n", " | "))
        print(f"---- {label} 用时 {time.time() - t0:.1f}s ----")

    summary = REC.summary()
    print("=" * 78)
    print(f"汇总：{summary['passed']}/{summary['total']} 通过，"
          f"{summary['failed']} 失败，总用时 {time.time() - t_start:.1f}s")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "verdict": "PASS" if summary["failed"] == 0 else "FAIL",
        "rows": REC.rows,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    (OUT_DIR / "小规模验证结果.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# D 题小规模约束审计验证结果", "",
             f"- 生成时间：{payload['generated_at']}",
             f"- 结论：**{payload['verdict']}**（{summary['passed']}/{summary['total']} 通过）",
             f"- 总用时：{payload['elapsed_s']} s", "",
             "| 分组 | 检查项 | 结果 | 说明 |", "|---|---|---|---|"]
    for r in REC.rows:
        lines.append(f"| {r['group']} | {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |")
    (OUT_DIR / "小规模验证结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"结果已写入 {OUT_DIR}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
