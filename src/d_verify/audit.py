"""独立可行性审计器。

设计原则（方案第 3 节）：

- 只读取"输入数据 + 方案表"，**不复用求解器的约束判断函数**；
- 本模块自带一套独立实现的物理计算（航段几何、能耗、SOC、充电、链路），
  只共享 :mod:`params` 的原始数据与 :func:`params.haversine_m` 这一几何工具；
- 任一硬约束违反 => ``FAIL``，与求解器是否返回 "optimal" 无关；
- 审计结果字段与方案第 3 节的清单一一对应。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .params import Scenario, haversine_m
from .plan import RelayPlan, Schedule, TripPlan

G = 9.80665
J_PER_KWH = 3.6e6
AREA_AGL = 30.0
CRUISE_CLEARANCE = 50.0
FREQ_MHZ = 2400.0
L_SYS = 3.0
L_OBS = 10.0
P_THRESHOLD = -98.0 + 8.0        # P_sens + M = -90 dBm

#: 端点设备参数（Pt dBm, G dBi），与附件“通信链路参数.xlsx”一致
TX = {
    "transport": (20.0, 3.0),
    "relay_access": (20.0, 6.0),
    "relay_backhaul": (19.0, 8.0),
    "gateway": (27.0, 12.0),
}


# ---------------------------------------------------------------- 独立物理重算


def _work_alt(node) -> float:
    return node.elevation + (0.0 if node.is_depot else AREA_AGL)


def _leg_geometry(scn: Scenario, model, n1, n2) -> dict:
    """独立实现的航段几何与飞行时间。"""
    d = haversine_m(n1.lat, n1.lon, n2.lat, n2.lon)
    if scn.dem is not None:
        top = scn.dem.max_elevation_along((n1.lat, n1.lon), (n2.lat, n2.lon))
        if not math.isfinite(top):
            top = max(n1.elevation, n2.elevation)
    else:
        top = max(n1.elevation, n2.elevation)
    cruise = top + CRUISE_CLEARANCE
    a1, a2 = _work_alt(n1), _work_alt(n2)
    h_up = max(cruise - a1, 0.0)
    h_dn = max(cruise - a2, 0.0)
    vc = model.cruise_speed
    t = h_up / model.v_up + d / vc + h_dn / model.v_down
    return {"d": d, "h_up": h_up, "h_dn": h_dn, "cruise": cruise,
            "start_alt": a1, "end_alt": a2, "t_total": t,
            "t_up": h_up / model.v_up, "t_cruise": d / vc, "t_dn": h_dn / model.v_down}


def _equiv_range(model, q: float) -> float:
    qq = min(max(q, 0.0), model.max_payload)
    return model.range_empty - (model.range_empty - model.range_full) * (qq / model.max_payload) ** 1.5


def _leg_energy(model, geo: dict, q: float) -> float:
    e_hor = geo["d"] / _equiv_range(model, q) * model.battery_energy
    e_up = 0.0 if geo["h_up"] <= 0 else (model.empty_mass + q) * G * geo["h_up"] / model.eta_up / J_PER_KWH
    return e_hor + e_up


def _charge_time(soc: float, t_full: float) -> float:
    if soc < 0.90:
        return t_full * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return t_full * 0.35 * (1.0 - soc) / 0.10


def _fspl(d_km: float) -> float:
    if d_km <= 0:
        return 0.0
    return 32.45 + 20 * math.log10(FREQ_MHZ) + 20 * math.log10(d_km)


def _link_threshold(role_a: str, role_b: str) -> float:
    def one(tx: str, rx: str) -> float:
        pt, gt = TX[tx]
        _, gr = TX[rx]
        return pt + gt + gr - L_SYS - P_THRESHOLD
    return min(one(role_a, role_b), one(role_b, role_a))


def _pos3d(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    return (lat, lon, alt)


def _dist_km(a, b) -> float:
    import math as _m
    mean_lat = _m.radians((a[0] + b[0]) / 2)
    dx = (b[1] - a[1]) * 111320.0 * _m.cos(mean_lat)
    dy = (b[0] - a[0]) * 110574.0
    dz = b[2] - a[2]
    return _m.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0


def _link_ok(scn: Scenario, role_a: str, role_b: str, pa, pb) -> bool:
    blocked = False
    if scn.dem is not None:
        blocked = not scn.dem.line_of_sight_clear(pa, pb)
    l_path = _fspl(_dist_km(pa, pb)) + (L_OBS if blocked else 0.0)
    return l_path <= _link_threshold(role_a, role_b) + 1e-9


# ---------------------------------------------------------------- 审计结果


@dataclass
class AuditResult:
    name: str = ""
    hard_violation_count: int = 0
    max_mass_violation: float = 0.0
    max_volume_violation: float = 0.0
    max_energy_violation: float = 0.0
    max_time_violation: float = 0.0
    communication_gap_count: int = 0
    resource_overlap_count: int = 0
    box_duplicate_count: int = 0
    box_missing_count: int = 0
    objective_recompute_error: float = 0.0
    partition_violation_count: int = 0
    details: list[str] = field(default_factory=list)
    recomputed: dict = field(default_factory=dict)
    verdict: str = "PASS"

    def fail(self, msg: str, count: int = 1) -> None:
        self.hard_violation_count += count
        self.details.append(msg)

    def finalize(self) -> "AuditResult":
        if self.hard_violation_count > 0:
            self.verdict = "FAIL"
        return self

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "hard_violation_count": self.hard_violation_count,
            "max_mass_violation": round(self.max_mass_violation, 9),
            "max_volume_violation": round(self.max_volume_violation, 9),
            "max_energy_violation": round(self.max_energy_violation, 9),
            "max_time_violation": round(self.max_time_violation, 9),
            "communication_gap_count": self.communication_gap_count,
            "resource_overlap_count": self.resource_overlap_count,
            "box_duplicate_count": self.box_duplicate_count,
            "box_missing_count": self.box_missing_count,
            "objective_recompute_error": round(self.objective_recompute_error, 9),
            "partition_violation_count": self.partition_violation_count,
            "recomputed": {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in self.recomputed.items()},
            "details": self.details[:40],
        }


# ---------------------------------------------------------------- 主审计


def audit(scn: Scenario, sched: Schedule, *, name: str = "", require_comm: bool = False,
          comm_sample_dt: float = 5.0, check_objective: bool = True) -> AuditResult:
    """对一个方案做独立可行性审计。"""
    res = AuditResult(name=name or sched.name)
    box_map = {b.id: b for b in scn.boxes}

    # ---------- 1. 货箱分配：恰好一次 ----------
    seen: dict[str, int] = {}
    for tp in sched.trips:
        for bid in tp.all_boxes():
            seen[bid] = seen.get(bid, 0) + 1
    dup = sum(1 for bid, c in seen.items() if c > 1)
    missing = [b.id for b in scn.boxes if b.id not in seen]
    unknown = [bid for bid in seen if bid not in box_map]
    res.box_duplicate_count = dup
    res.box_missing_count = len(missing)
    if dup:
        res.fail(f"有 {dup} 个货箱被重复分配：{[bid for bid, c in seen.items() if c > 1][:10]}")
    if missing:
        res.fail(f"有 {len(missing)} 个货箱未分配：{missing[:10]}")
    if unknown:
        res.fail(f"方案中出现未知货箱编号：{unknown[:10]}")

    # ---------- 2. 逐架次独立重算 ----------
    uav_intervals: dict[str, list[tuple[float, float, str]]] = {}
    battery_intervals: dict[str, list[tuple[float, float, str]]] = {}
    makespan = 0.0
    total_energy = 0.0
    lateness = 0.0
    trip_recomputed = []

    for tp in sched.trips:
        model = scn.transport_models[tp.model]
        # 服务区顺序与箱子归属的一致性
        stop_boxes = [list(g) for g in tp.boxes_per_stop]
        if len(stop_boxes) != len(tp.seq):
            res.fail(f"{tp.trip_id}: 站序与投递分组长度不一致")
            continue
        for area, grp in zip(tp.seq, stop_boxes):
            for bid in grp:
                if bid in box_map and box_map[bid].area != area:
                    res.fail(f"{tp.trip_id}: 货箱 {bid} 被投递到错误服务区 {area}")

        mass = sum(box_map[b].mass for b in tp.all_boxes() if b in box_map)
        volume = sum(box_map[b].volume for b in tp.all_boxes() if b in box_map)
        # --- 质量与体积分别检查 ---
        m_viol = max(0.0, mass - model.max_payload)
        v_viol = max(0.0, volume - model.volume)
        res.max_mass_violation = max(res.max_mass_violation, m_viol)
        res.max_volume_violation = max(res.max_volume_violation, v_viol)
        if m_viol > 1e-9:
            res.fail(f"{tp.trip_id}: 载质量超限 {m_viol:.4f} kg（{mass:.4f} > {model.max_payload}）")
        if v_viol > 1e-9:
            res.fail(f"{tp.trip_id}: 装载体积超限 {v_viol:.6f} m^3（{volume:.6f} > {model.volume}）")

        # --- 航段：载荷递减 + 能耗 ---
        nodes = [scn.depot] + [scn.areas[a] for a in tp.seq] + [scn.depot]
        q = mass
        energy = 0.0
        flight = 0.0
        legs = []
        for i in range(len(nodes) - 1):
            geo = _leg_geometry(scn, model, nodes[i], nodes[i + 1])
            e = _leg_energy(model, geo, q)
            legs.append((geo, q, e))
            energy += e
            flight += geo["t_total"]
            if i < len(tp.seq):
                q -= sum(box_map[b].mass for b in stop_boxes[i] if b in box_map)
        limit = (1 - model.reserve_ratio) * model.battery_energy
        e_viol = max(0.0, energy - limit)
        res.max_energy_violation = max(res.max_energy_violation, e_viol)
        if e_viol > 1e-9:
            res.fail(f"{tp.trip_id}: 超出返航安全余量 {e_viol:.6f} kWh（{energy:.6f} > {limit:.6f}）")
        if abs(q) > 1e-6:
            res.fail(f"{tp.trip_id}: 返航载荷不为 0（{q:.6f} kg），货箱质量与实际投递不一致")

        # --- 时间：准备 + 飞行 + 交接 ---
        prep = model.ground_prep + model.load_per_box * len(tp.all_boxes())
        handover = [model.handover_base + model.handover_per_box * len(g) for g in stop_boxes]
        t = tp.start_time + prep
        deliver_times = {}
        for i, (geo, _, _) in enumerate(legs):
            t += geo["t_total"]
            if i < len(tp.seq):
                t += handover[i]
                deliver_times[tp.seq[i]] = t
                for bid in stop_boxes[i]:
                    b = box_map.get(bid)
                    if b is None:
                        continue
                    if b.is_first_batch and b.first_deadline is not None and t > b.first_deadline + 1e-9:
                        res.max_time_violation = max(res.max_time_violation, t - b.first_deadline)
                        res.fail(f"{tp.trip_id}: 首批货箱 {bid} 超时 {t - b.first_deadline:.2f} s")
                    if (not b.is_first_batch) and t > b.expected_time + 1e-9:
                        lateness += t - b.expected_time
        end = t
        makespan = max(makespan, end)
        total_energy += energy

        # --- 资源占用区间 ---
        uav_intervals.setdefault(tp.uav, []).append((tp.start_time, end, tp.trip_id))
        soc_end = 1.0 - energy / model.battery_energy
        if soc_end < model.reserve_ratio - 1e-9:
            res.fail(f"{tp.trip_id}: 结束 SOC {soc_end:.4f} 低于返航电量下限 {model.reserve_ratio}")
        chg = _charge_time(max(soc_end, 0.0), scn.battery_charge_full[tp.model])
        battery_intervals.setdefault(tp.battery, []).append(
            (tp.start_time, end, tp.trip_id))
        battery_intervals[tp.battery].append((end, end + chg, f"{tp.trip_id}-充电"))

        trip_recomputed.append({
            "trip_id": tp.trip_id, "model": tp.model, "uav": tp.uav, "battery": tp.battery,
            "mass": round(mass, 4), "volume": round(volume, 6),
            "energy": round(energy, 9), "energy_limit": round(limit, 6),
            # 资源充电解禁时间可能在毫秒以下存在差异；只保留 3 位小数会让
            # “结束时刻 + 充电时间”在独立复算时出现假性早启。保存 9 位，
            # 让导出审计记录本身也能闭环，而不是只依赖内存中的未舍入区间。
            "start": round(tp.start_time, 9), "end": round(end, 9),
            "soc_end": round(soc_end, 9),
            "legs": [{"from": n1.id, "to": n2.id, "q": round(qq, 6), "e": round(ee, 9)}
                     for (n1, n2), (_, qq, ee) in zip(zip(nodes, nodes[1:]), legs)],
            "deliver": {k: round(v, 9) for k, v in deliver_times.items()},
        })

    # ---------- 3. 资源区间重叠 ----------
    for uid, ivs in uav_intervals.items():
        overlap = _count_overlaps(ivs)
        if overlap:
            res.resource_overlap_count += overlap
            res.fail(f"无人机 {uid} 的任务时段重叠 {overlap} 处")
    for bid, ivs in battery_intervals.items():
        overlap = _count_overlaps(ivs)
        if overlap:
            res.resource_overlap_count += overlap
            res.fail(f"电池 {bid} 的任务/充电区间重叠 {overlap} 处")

    # ---------- 4. 连续通信（Q3） ----------
    if require_comm:
        gaps = _audit_communication(scn, sched, comm_sample_dt)
        res.communication_gap_count = gaps
        if gaps:
            res.fail(f"存在 {gaps} 个通信中断采样点")

    # ---------- 5. Q4 分区 ----------
    part_summary = {}
    if sched.partitions:
        res.partition_violation_count, part_summary = _audit_partitions(
            scn, sched, res, trip_recomputed)

    # ---------- 6. 目标重算与比对 ----------
    res.recomputed = {
        "makespan": makespan,
        "total_energy": total_energy,
        "n_trips": len(sched.trips),
        "lateness": lateness,
        "trips": trip_recomputed,
    }
    if part_summary:
        res.recomputed["partitions"] = part_summary
    if check_objective and sched.reported:
        err = _objective_error(sched, res.recomputed)
        res.objective_recompute_error = err
        if err > 1e-4 * max(1.0, abs(sched.reported.get("objective", 1.0))):
            res.fail(f"目标重算不一致：误差 {err:.6f}（报告值 {sched.reported.get('objective')}）")
    return res.finalize()


def _count_overlaps(intervals: list[tuple[float, float, str]]) -> int:
    ivs = sorted(intervals, key=lambda x: (x[0], x[1]))
    bad = 0
    for i in range(len(ivs)):
        for j in range(i + 1, len(ivs)):
            if ivs[j][0] < ivs[i][1] - 1e-9:
                bad += 1
            else:
                break
    return bad


def _objective_error(sched: Schedule, recomputed: dict) -> float:
    """用独立重算的分量与求解器报告的加权目标比较。"""
    rep = sched.reported
    w = {"makespan": 1.0, "energy": 1000.0, "trips": 300.0, "lateness": 1.0}
    obj = (w["makespan"] * recomputed["makespan"] + w["energy"] * recomputed["total_energy"]
           + w["trips"] * recomputed["n_trips"] + w["lateness"] * recomputed["lateness"])
    return abs(obj - rep.get("objective", 0.0))


# ---------------------------------------------------------------- 通信审计


def _audit_communication(scn: Scenario, sched: Schedule, dt: float) -> int:
    """对每个架次全过程（爬升/巡航/下降/投送）采样，统计通信中断采样点数。

    投送阶段按服务区 30 m 作业高度悬停处理；地面准备阶段不参与判定。
    """
    gaps = 0
    gpos = (scn.depot.lat, scn.depot.lon, scn.depot.elevation + 20.0)
    for tp in sched.trips:
        model = scn.transport_models[tp.model]
        nodes = [scn.depot] + [scn.areas[a] for a in tp.seq] + [scn.depot]
        geos = [_leg_geometry(scn, model, nodes[i], nodes[i + 1])
                for i in range(len(nodes) - 1)]
        prep = model.ground_prep + model.load_per_box * len(tp.all_boxes())
        t = tp.start_time + prep
        for i, geo in enumerate(geos):
            seg_start = t
            steps = max(int(math.ceil(geo["t_total"] / dt)), 1)
            for k in range(steps + 1):
                tau = min(k * dt, geo["t_total"])
                pos = _position_on_leg(nodes[i], nodes[i + 1], geo, tau)
                if not _comm_ok_at(scn, sched, pos, gpos, seg_start + tau):
                    gaps += 1
            t = seg_start + geo["t_total"]
            if i < len(tp.seq):
                hold = model.handover_base + model.handover_per_box * len(tp.boxes_per_stop[i])
                pos = (nodes[i + 1].lat, nodes[i + 1].lon,
                       nodes[i + 1].elevation + AREA_AGL)
                steps = max(int(math.ceil(hold / dt)), 1)
                for k in range(steps + 1):
                    tau = min(k * dt, hold)
                    if not _comm_ok_at(scn, sched, pos, gpos, t + tau):
                        gaps += 1
                t += hold
    return gaps


def _position_on_leg(n1, n2, geo: dict, t: float):
    """航段内 t 秒的位置（独立实现）。"""
    t = max(0.0, min(t, geo["t_total"]))
    if t <= geo["t_up"] and geo["t_up"] > 0:
        frac = t / geo["t_up"]
        return (n1.lat, n1.lon, geo["start_alt"] + (geo["cruise"] - geo["start_alt"]) * frac)
    t -= geo["t_up"]
    if t <= geo["t_cruise"]:
        frac = t / geo["t_cruise"] if geo["t_cruise"] > 0 else 1.0
        return (n1.lat + (n2.lat - n1.lat) * frac, n1.lon + (n2.lon - n1.lon) * frac, geo["cruise"])
    t -= geo["t_cruise"]
    frac = min(t / geo["t_dn"], 1.0) if geo["t_dn"] > 0 else 1.0
    return (n1.lat + (n2.lat - n1.lat) * frac, n1.lon + (n2.lon - n1.lon) * frac,
            geo["cruise"] + (geo["end_alt"] - geo["cruise"]) * frac)


def _relay_positions(scn: Scenario, sched: Schedule, t: float) -> list[tuple[str, tuple]]:
    out = []
    for rp in sched.relays:
        if rp.service_start - 1e-9 <= t <= rp.service_end + 1e-9:
            out.append((rp.relay_uav, _pos3d(rp.hover_lat, rp.hover_lon, rp.hover_alt)))
    return out


def _comm_ok_at(scn: Scenario, sched: Schedule, pos, gpos, t: float) -> bool:
    """独立实现的通信状态判定：直连优先，其次中继（两段须同刻可用）。"""
    if _link_ok(scn, "transport", "gateway", pos, gpos):
        return True
    for _, rpos in _relay_positions(scn, sched, t):
        if (_link_ok(scn, "transport", "relay_access", pos, rpos)
                and _link_ok(scn, "relay_backhaul", "gateway", rpos, gpos)):
            return True
    return False


# ---------------------------------------------------------------- 分区审计


def _audit_partitions(scn: Scenario, sched: Schedule, res: AuditResult,
                      trips_info: list[dict]) -> tuple[int, dict]:
    """Q4：分区合法性与资源独立性（方案 7.1 / 7.2）。

    资源核算给出两种口径：

    1. ``*_fixed_count``：固定 Q3 方案中的资源编号，统计各任务组出现过多少个编号；
    2. ``*_min_by_coloring``：固定 Q3 时间表，对组内架次区间做着色得到的**最少**资源数。

    方案 7.2 明确：若论文声称"各组独立执行所需最少资源"，必须采用口径 2，
    不能只统计 Q3 中出现过多少个资源编号。
    """
    n_bad = 0
    area_ids = set(scn.areas)
    info = {t["trip_id"]: t for t in trips_info}
    summary_all: dict = {}

    for k, groups in sched.partitions.items():
        members = [a for g in groups.values() for a in g]
        if sorted(members) != sorted(area_ids):
            res.fail(f"K={k}: 分区未覆盖全部服务区或存在重复：{sorted(members)}")
            n_bad += 1
        if any(len(g) == 0 for g in groups.values()):
            res.fail(f"K={k}: 存在空任务组")
            n_bad += 1
        area_group = {a: gid for gid, g in groups.items() for a in g}

        bad_trip = False
        for tp in sched.trips:
            gids = {area_group.get(a) for a in tp.seq}
            if len(gids) > 1:
                res.fail(f"K={k}: 架次 {tp.trip_id} 跨任务组 {gids}（服务区 {tp.seq} 必须同组）")
                n_bad += 1
                bad_trip = True
        if bad_trip:      # 分区非法时资源核算无意义
            continue

        fixed_uav = {gid: set() for gid in groups}
        fixed_bat = {gid: set() for gid in groups}
        uav_by_model: dict = {gid: {} for gid in groups}
        bat_by_model: dict = {gid: {} for gid in groups}
        n_trips = {gid: 0 for gid in groups}
        for tp in sched.trips:
            gid = area_group.get(tp.seq[0])
            d = info.get(tp.trip_id)
            if d is None:
                continue
            chg = _charge_time(max(d["soc_end"], 0.0), scn.battery_charge_full[tp.model])
            fixed_uav[gid].add(tp.uav)
            fixed_bat[gid].add(tp.battery)
            uav_by_model[gid].setdefault(tp.model, []).append((d["start"], d["end"]))
            bat_by_model[gid].setdefault(tp.model, []).append((d["start"], d["end"] + chg))
            n_trips[gid] += 1

        summary = {}
        for gid in groups:
            summary[gid] = {
                "areas": sorted(groups[gid]),
                "n_trips": n_trips[gid],
                "uav_fixed_count": len(fixed_uav[gid]),
                "uav_min_by_coloring": sum(_max_concurrency(v) for v in uav_by_model[gid].values()),
                "battery_fixed_count": len(fixed_bat[gid]),
                "battery_min_by_coloring": sum(_max_concurrency(v) for v in bat_by_model[gid].values()),
            }
        summary_all[k] = summary
    return n_bad, summary_all


def _max_concurrency(intervals: list[tuple[float, float]]) -> int:
    """区间集合的最大同时占用数（= 覆盖全部区间所需的最少资源数）。"""
    events: list[tuple[float, int]] = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best
