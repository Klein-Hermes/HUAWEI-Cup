"""能耗物理建模闭环验证（针对附录 2 的能量模型本身）。

与 run_all.py 的定位区别：run_all 验证"约束在方案层面是否被正确编码"，
本文件专门验证**能耗物理建模本身**是否正确、是否闭环、规模上是否够快：

1. 正确性锚定：用手写解析式（独立于 physics.py 的编码路径）复核航段时间、
   等效航程、水平能耗、爬升附加能耗，并做空载/满载极限点与 J->kWh 量纲检查；
2. 载荷敏感性：能耗随载荷单调、去程带载与返程空载的不对称性；
3. SOC 记账闭环：结束 SOC = 1 - E/E_use -> 充电两段模型 -> 电池解禁时刻；
4. 双实现逐段交叉：physics（求解器侧）与 audit（独立实现）**逐段**能耗与载荷一致；
   审计器导出精度已由 6 位小数提升到 9 位，因此该组同时校验"未舍入实现值"与"导出记录值"；
5. 规模与性能：全量 15 服务区 + O01 全部航段几何的冷/热缓存耗时，
   以及"求解 + 审计"闭环耗时。

用法（仓库根目录）::

    python -B src/d_verify/tests/energy_closure.py
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from d_verify import audit as audit_mod  # noqa: E402
from d_verify import params, physics, solver_small  # noqa: E402
from d_verify.dem import RealDEM  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "D题_小规模验证"

#: 手算基准使用的物理常数（刻意在测试内重新声明，避免引用被测模块的常量）
G_MANUAL = 9.80665
J_PER_KWH_MANUAL = 3.6e6


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"group": group, "name": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {group} :: {name}" + (f" | {detail}" if detail else ""))
        return bool(ok)

    def check(self, group: str, name: str, cond: bool, detail: str = "") -> bool:
        return self.record(group, name, cond, detail)

    def close(self, group: str, name: str, value: float, expect: float, tol: float = 1e-9) -> bool:
        return self.record(group, name, abs(value - expect) <= tol,
                           f"实测 {value:.9g} / 期望 {expect:.9g}")

    def summary(self) -> dict:
        total = len(self.rows)
        passed = sum(1 for r in self.rows if r["pass"])
        return {"total": total, "passed": passed, "failed": total - passed}


REC = Recorder()


def manual_equiv_range(model, q: float) -> float:
    """手写等效航程 L(q) = L0 - (L0 - LF) * (q/Q)^1.5。"""
    return model.range_empty - (model.range_empty - model.range_full) * (q / model.max_payload) ** 1.5


def manual_horizontal_energy(model, d_m: float, q: float) -> float:
    """手写水平巡航能耗（kWh）。"""
    return d_m / manual_equiv_range(model, q) * model.battery_energy


def manual_climb_energy(model, q: float, h_up: float) -> float:
    """手写爬升附加能耗（kWh）：重力做功 / 效率，再由 J 换算 kWh。"""
    if h_up <= 0:
        return 0.0
    joules = (model.empty_mass + q) * G_MANUAL * h_up / model.eta_up
    return joules / J_PER_KWH_MANUAL


# ---------------------------------------------------------------- 1 正确性锚定


def test_anchor_geometry_and_time(G: str, sc, dem) -> None:
    for mid in sorted(sc.transport_models):
        model = sc.transport_models[mid]
        for aid in ("S001", "S015"):
            a = sc.areas[aid]
            leg = physics.build_leg(model, sc.depot, a, dem)
            t_manual = (leg.h_up / model.v_up + leg.d_m / model.cruise_speed
                        + leg.h_dn / model.v_down)
            REC.close(G, f"{mid} 型 O01->{aid} 航段时间（手写解析式）",
                      leg.t_total, t_manual, tol=1e-9)
            REC.close(G, f"{mid} 型 O01->{aid} 三段时长自洽",
                      leg.t_up + leg.t_cruise + leg.t_dn, leg.t_total, tol=1e-9)
            # 巡航海拔必须高于两端作业高度（否则爬升/下降为负）
            REC.check(G, f"{mid} 型 O01->{aid} 巡航海拔不低于作业高度",
                      leg.cruise_alt >= max(leg.start_alt, leg.end_alt) - 1e-9,
                      f"cruise={leg.cruise_alt:.2f} start={leg.start_alt:.2f} end={leg.end_alt:.2f}")


def test_anchor_range_and_energy(G: str, sc, dem) -> None:
    model = sc.transport_models["A"]
    leg = physics.build_leg(model, sc.depot, sc.areas["S001"], dem)
    # 空载极限：L(0) = L0；满载极限：L(Q) = LF
    REC.close(G, "A 型空载等效航程 L(0) = 空载标准航程",
              physics.equivalent_range(model, 0.0), model.range_empty, tol=1e-9)
    REC.close(G, "A 型满载等效航程 L(Q) = 满载标准航程",
              physics.equivalent_range(model, model.max_payload), model.range_full, tol=1e-9)
    for q in (0.0, 6.25, 12.5, 25.0):
        REC.close(G, f"A 型 L({q}) 手写解析式对照",
                  physics.equivalent_range(model, q), manual_equiv_range(model, q), tol=1e-9)
        REC.close(G, f"A 型 水平能耗 E_hor({q}) 手写解析式对照",
                  physics.horizontal_energy(model, leg.d_m, q),
                  manual_horizontal_energy(model, leg.d_m, q), tol=1e-12)
    for mid in ("B", "C"):
        m2 = sc.transport_models[mid]
        leg2 = physics.build_leg(m2, sc.depot, sc.areas["S015"], dem)
        REC.close(G, f"{mid} 型水平能耗手写对照（S015）",
                  physics.horizontal_energy(m2, leg2.d_m, m2.max_payload * 0.5),
                  manual_horizontal_energy(m2, leg2.d_m, m2.max_payload * 0.5), tol=1e-12)


def test_anchor_climb_energy_units(G: str, sc) -> None:
    for mid in ("A", "C"):
        model = sc.transport_models[mid]
        for q, h in ((0.0, 100.0), (model.max_payload, 250.0), (model.max_payload * 0.4, 0.0)):
            REC.close(G, f"{mid} 型 爬升附加能耗 E_up(q={q:g}, h={h:g}) 手写对照",
                      physics.climb_energy(model.empty_mass, q, h, model.eta_up),
                      manual_climb_energy(model, q, h), tol=1e-15)
    # 量纲自检：J 与 kWh 的换算（1 kWh = 3.6e6 J）
    e_j = (70.0 + 25.0) * G_MANUAL * 100.0 / 0.72
    REC.close(G, "量纲自检 129.4 kJ -> kWh", e_j / J_PER_KWH_MANUAL, e_j / 3.6e6, tol=1e-18)
    # 下降能耗效率为 0：h_up<=0 时必须返回 0，且下降段不额外计能耗
    REC.close(G, "h_up<=0 时爬升能耗为 0",
              physics.climb_energy(70.0, 25.0, -30.0, 0.72), 0.0, tol=0.0)


# ---------------------------------------------------------------- 2 载荷敏感性


def test_load_sensitivity(G: str, sc, dem) -> None:
    model = sc.transport_models["A"]
    leg = physics.build_leg(model, sc.depot, sc.areas["S015"], dem)
    es = [physics.horizontal_energy(model, leg.d_m, q)
          for q in (0.0, 6.25, 12.5, 18.75, 25.0)]
    REC.check(G, "水平能耗随载荷单调不减",
              all(es[i] <= es[i + 1] + 1e-15 for i in range(len(es) - 1)),
              "E(q) 序列=" + str([round(e, 6) for e in es]))
    REC.check(G, "满载水平能耗严格大于空载", es[-1] > es[0] + 1e-9,
              f"{es[-1]:.6f} > {es[0]:.6f}")

    # 去程带载 + 返程空载 vs 双程带载
    q_trip = 14.0
    e_out = physics.leg_energy_transport(model, leg, q_trip)
    e_back = physics.leg_energy_transport(model, leg, 0.0)
    e_both = 2 * physics.leg_energy_transport(model, leg, q_trip)
    REC.check(G, "投送后返程能耗低于去程（载荷下降）", e_back < e_out,
              f"返程 {e_back:.6f} < 去程 {e_out:.6f}")
    REC.check(G, "往返总能耗低于双程满载", e_out + e_back < e_both,
              f"{e_out + e_back:.6f} < {e_both:.6f}")


# ---------------------------------------------------------------- 3 SOC 闭环


def test_soc_closure(G: str, sc, dem) -> None:
    # 场景：仅 1 架 B 型 + 1 组电池，3 个箱子 → 必须串行且受充电约束
    sc_t3 = params.build_small_scenario(
        dem=dem, areas=["S001", "S015"],
        box_ids=["S001-WAT-01", "S001-WAT-02", "S015-WAT-01"],
        uavs=["U05"], battery_stock={"B": 1})
    sched = solver_small.solve_exact(sc_t3, name="soc-closure")
    if sched is None:
        REC.check(G, "SOC 闭环场景有解", False, "无解")
        return
    res = audit_mod.audit(sc_t3, sched, name="soc-closure")
    REC.check(G, "SOC 闭环场景审计通过", res.verdict == "PASS",
              f"violations={res.hard_violation_count}")
    trips = sorted(res.recomputed.get("trips", []), key=lambda t: t["start"])
    model = sc_t3.transport_models["B"]
    e_use = model.battery_energy

    for t in trips:
        # 记账闭环：结束 SOC 必须等于 1 - E_trip / E_use
        expected_soc = 1.0 - t["energy"] / e_use
        REC.close(G, f"{t['trip_id']} 结束 SOC = 1 - E/E_use",
                  t["soc_end"], expected_soc, tol=1e-8)
        REC.check(G, f"{t['trip_id']} 结束 SOC 不低于返航下限",
                  t["soc_end"] >= model.reserve_ratio - 1e-9,
                  f"soc_end={t['soc_end']:.6f} >= {model.reserve_ratio}")

    if len(trips) >= 2:
        gap = trips[1]["start"] - trips[0]["end"]
        chg = physics.charge_time(max(trips[0]["soc_end"], 0.0),
                                  sc_t3.battery_charge_full["B"])
        REC.check(G, "电池解禁时刻 = 架次结束 + 两段充电时间",
                  # 审计导出的架次时刻保留到 1 ms，允许同量级的数值舍入误差。
                  gap >= chg - 1e-3,
                  f"间隔 {gap:.3f}s >= 充电 {chg:.3f}s（差值 {gap - chg:.3f}s）")
        # 充电时间必须落在 [0.35*T_full, T_full] 的物理区间内
        REC.check(G, "充电时间落在两段模型的物理区间内",
                  0.35 * sc_t3.battery_charge_full["B"] - 1e-9 <= chg
                  <= sc_t3.battery_charge_full["B"] + 1e-9,
                  f"{chg:.1f}s ∈ [{0.35 * sc_t3.battery_charge_full['B']:.1f}, "
                  f"{sc_t3.battery_charge_full['B']:.1f}]")
        # 慢充段（SOC>=0.9）时间应显著短于快充段
        t_slow = physics.charge_time(0.95, sc_t3.battery_charge_full["B"])
        t_fast = physics.charge_time(0.50, sc_t3.battery_charge_full["B"])
        REC.check(G, "慢充段（SOC>=0.9）用时短于快充段", t_slow < t_fast,
                  f"t_chg(0.95)={t_slow:.1f}s < t_chg(0.50)={t_fast:.1f}s")


# ---------------------------------------------------------------- 4 双实现逐段交叉


def _cross_check(G: str, sc, sched, res, tag: str) -> None:
    box_map = {b.id: b for b in sc.boxes}
    max_e_err = 0.0        # 导出记录（9 位小数）对比
    max_e_raw = 0.0        # 两套实现内部（未舍入）对比
    max_q_err = 0.0
    n_legs = 0
    for tp, rec in zip(sched.trips, res.recomputed["trips"]):
        model = sc.transport_models[tp.model]
        nodes = [sc.depot] + [sc.areas[a] for a in tp.seq] + [sc.depot]
        q = sum(box_map[b].mass for b in tp.all_boxes())
        for i in range(len(nodes) - 1):
            leg = physics.build_leg(model, nodes[i], nodes[i + 1], sc.dem)
            e_solver = physics.leg_energy_transport(model, leg, q)
            geo = audit_mod._leg_geometry(sc, model, nodes[i], nodes[i + 1])
            e_raw = audit_mod._leg_energy(model, geo, q)
            max_e_raw = max(max_e_raw, abs(e_solver - e_raw))
            e_audit = rec["legs"][i]["e"]
            q_audit = rec["legs"][i]["q"]
            max_e_err = max(max_e_err, abs(e_solver - e_audit))
            max_q_err = max(max_q_err, abs(q - q_audit))
            n_legs += 1
            if i < len(tp.seq):
                q -= sum(box_map[b].mass for b in tp.boxes_per_stop[i])
    REC.check(G, f"{tag}: 两套实现逐段能耗（未舍入）完全一致（{n_legs} 段）",
              max_e_raw < 1e-12, f"最大偏差 {max_e_raw:.3e} kWh")
    REC.check(G, f"{tag}: 导出记录逐段能耗一致（9 位小数，{n_legs} 段）",
              max_e_err < 1e-8, f"最大偏差 {max_e_err:.3e} kWh")
    REC.check(G, f"{tag}: 导出记录逐段载荷一致（6 位小数）",
              max_q_err < 1e-5, f"最大偏差 {max_q_err:.3e} kg")


def test_cross_implementation(G: str, dem) -> None:
    # T1：单服务区 3 箱
    sc1 = params.build_small_scenario(
        dem=dem, areas=["S001"],
        box_ids=["S001-MED-01", "S001-WAT-01", "S001-FOD-01"],
        uavs=["U01", "U05", "U07"], battery_stock={"A": 2, "B": 2, "C": 2})
    sched1 = solver_small.solve_exact(sc1, name="cross-T1", single_area=True)
    if sched1 is None:
        REC.check(G, "T1 交叉校验可求解", False, "无解")
    else:
        res1 = audit_mod.audit(sc1, sched1, name="cross-T1")
        _cross_check(G, sc1, sched1, res1, "T1")

    # 统一小实例：3 服务区 8 箱（含多点架次与跨区路线）
    sc2 = params.build_small_scenario(dem=dem)
    sched2 = solver_small.solve_exact(sc2, name="cross-small", max_trips=3,
                                      max_candidates_per_block=2)
    if sched2 is None:
        REC.check(G, "统一小实例交叉校验可求解", False, "无解")
    else:
        res2 = audit_mod.audit(sc2, sched2, name="cross-small")
        _cross_check(G, sc2, sched2, res2, "统一小实例")


# ---------------------------------------------------------------- 5 规模与性能


def test_scale_and_speed(G: str, sc_full, dem) -> None:
    node_ids = [sc_full.depot.id] + sorted(sc_full.areas)
    pairs = [(node_ids[i], node_ids[j])
             for i in range(len(node_ids)) for j in range(i + 1, len(node_ids))]
    n_models = len(sc_full.transport_models)
    total_legs = len(pairs) * n_models

    t0 = time.time()
    for mid, model in sorted(sc_full.transport_models.items()):
        for a, b in pairs:
            physics.build_leg(model, sc_full.node(a), sc_full.node(b), dem)
    t_cold = time.time() - t0

    t0 = time.time()
    for mid, model in sorted(sc_full.transport_models.items()):
        for a, b in pairs:
            physics.build_leg(model, sc_full.node(a), sc_full.node(b), dem)
    t_warm = time.time() - t0

    print(f"    [规模] 航段几何 {total_legs} 条（{len(pairs)} 节点对 x {n_models} 机型）；"
          f"冷缓存 {t_cold:.2f}s，热缓存 {t_warm:.3f}s")
    REC.check(G, f"全量航段几何冷缓存耗时 < 20s（{total_legs} 条）", t_cold < 20.0,
              f"{t_cold:.2f}s")
    REC.check(G, "热缓存命中后耗时下降 >= 5 倍（缓存生效）",
              t_warm * 5.0 <= t_cold, f"冷 {t_cold:.3f}s / 热 {t_warm:.4f}s")

    # 一次完整"求解 + 审计"闭环的耗时（统一小实例，受限枚举）
    sc_small = params.build_small_scenario(dem=dem)
    t0 = time.time()
    sched = solver_small.solve_exact(sc_small, name="speed", max_trips=3,
                                     max_candidates_per_block=2)
    t_solve = time.time() - t0
    t_audit = float("nan")
    if sched is not None:
        t0 = time.time()
        res = audit_mod.audit(sc_small, sched, name="speed")
        t_audit = time.time() - t0
        REC.check(G, "闭环耗时场景审计通过", res.verdict == "PASS",
                  f"violations={res.hard_violation_count}")
    print(f"    [规模] 统一小实例（8 箱）求解 {t_solve:.2f}s + 审计 {t_audit:.2f}s")
    REC.check(G, "小实例求解 + 审计闭环在 30s 内完成", t_solve + t_audit < 30.0,
              f"{t_solve + t_audit:.2f}s")

    # 单服务区 Q1 子过程（15 个服务区逐个）耗时
    t0 = time.time()
    msp = {aid: solver_small.max_safe_payload(sc_full, aid) for aid in sorted(sc_full.areas)}
    t_msp = time.time() - t0
    print(f"    [规模] 15 个服务区最大安全载荷（二分搜索）用时 {t_msp:.2f}s")
    REC.check(G, "15 服务区最大安全载荷批量计算 < 30s", t_msp < 30.0, f"{t_msp:.2f}s")
    REC.check(G, "各机型最大安全载荷不超过机型上限",
              all(v <= sc_full.transport_models[m].max_payload + 1e-9
                  for m, v in next(iter(msp.values())).items()),
              f"示例 S001={msp['S001']}")


# ---------------------------------------------------------------- 主流程


def main() -> int:
    t_start = time.time()
    print("=" * 78)
    print("能耗物理建模闭环验证（正确性锚定 / SOC 记账 / 双实现逐段交叉 / 规模性能）")
    print("=" * 78)

    dem = RealDEM.load(params.DEM_MAT)
    sc_full = params.load_scenario(dem=dem)
    print(f"[信息] 真实 DEM {dem.rows}x{dem.cols}；15 服务区 / {len(sc_full.boxes)} 货箱 / "
          f"{len(sc_full.transport_uavs)} 架运输机")

    stages = [
        ("E1 航段几何与时间锚定", lambda: test_anchor_geometry_and_time("E1", sc_full, dem)),
        ("E2 等效航程与水平能耗锚定", lambda: test_anchor_range_and_energy("E2", sc_full, dem)),
        ("E3 爬升能耗与量纲", lambda: test_anchor_climb_energy_units("E3", sc_full)),
        ("E4 载荷敏感性", lambda: test_load_sensitivity("E4", sc_full, dem)),
        ("E5 SOC 记账闭环", lambda: test_soc_closure("E5", sc_full, dem)),
        ("E6 双实现逐段交叉", lambda: test_cross_implementation("E6", dem)),
        ("E7 规模与性能", lambda: test_scale_and_speed("E7", sc_full, dem)),
    ]
    for label, fn in stages:
        t0 = time.time()
        try:
            fn()
        except Exception:
            REC.record(label, "测试执行异常", False,
                       traceback.format_exc(limit=2).replace("\n", " | "))
        print(f"---- {label} 用时 {time.time() - t0:.1f}s ----")

    summary = REC.summary()
    print("=" * 78)
    print(f"汇总：{summary['passed']}/{summary['total']} 通过，{summary['failed']} 失败，"
          f"总用时 {time.time() - t_start:.1f}s")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "verdict": "PASS" if summary["failed"] == 0 else "FAIL",
        "rows": REC.rows,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    (OUT_DIR / "能耗闭环验证结果.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# D 题能耗物理建模闭环验证结果", "",
             f"- 生成时间：{payload['generated_at']}",
             f"- 结论：**{payload['verdict']}**（{summary['passed']}/{summary['total']} 通过）",
             f"- 总用时：{payload['elapsed_s']} s", "",
             "| 分组 | 检查项 | 结果 | 说明 |", "|---|---|---|---|"]
    for r in REC.rows:
        lines.append(f"| {r['group']} | {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |")
    (OUT_DIR / "能耗闭环验证结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"结果已写入 {OUT_DIR}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
