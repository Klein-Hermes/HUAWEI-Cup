"""小规模精确枚举求解器（Q1 单点往返 / Q2 多点多架次）。

设计要点：

- 求解器只对小实例工作（方案 10.2 的统一小实例与 T1~T4 实例）；
- 枚举对象：货箱划分 -> 每块的站序排列 -> 机型 -> 资源槽位（无人机 x 电池）-> 执行顺序；
- 目标为加权标量（:data:`WEIGHTS`），所有分量同时记录在 ``Schedule.reported``，
  供审计器独立重算比对；
- ``relax`` 参数用于**约束突变测试**：故意删除某条约束的编码，检验审计器能否捕获。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

from . import comm
from .params import Box, Node, Scenario, haversine_m
from .physics import (build_leg, charge_time, leg_energy_transport,
                      relay_leg_energy, work_altitude)
from .plan import RelayPlan, Schedule, TripPlan

#: 目标权重（加权标量；单位：makespan s、能耗 kWh、架次 个、超时 s）
WEIGHTS = {
    "makespan": 1.0,
    "energy": 1000.0,
    "trips": 300.0,
    "lateness": 1.0,
}

#: 小实例允许的最大架次数（超过则该划分不可能比已有更优/超出资源）
MAX_TRIPS = 5


# ---------------------------------------------------------------- 货箱划分


def enumerate_partitions(items: list) -> list[list[list]]:
    """枚举集合的全部划分（Bell 数）。"""
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    out = []
    for p in enumerate_partitions(rest):
        for i in range(len(p)):
            out.append(p[:i] + [[first] + p[i]] + p[i + 1:])
        out.append([[first]] + p)
    return out


# ---------------------------------------------------------------- 架次候选


class TripCandidate:
    """一个可行的架次候选（不含资源与时刻）。"""

    __slots__ = ("boxes", "model", "seq", "stops_boxes", "energy", "flight_time",
                 "prep", "handover", "duration", "legs")

    def __init__(self, boxes, model, seq, stops_boxes, energy, flight_time, prep,
                 handover, duration, legs):
        self.boxes = boxes
        self.model = model
        self.seq = seq
        self.stops_boxes = stops_boxes
        self.energy = energy
        self.flight_time = flight_time
        self.prep = prep
        self.handover = handover
        self.duration = duration
        self.legs = legs

    # 交付时刻（相对架次开始）按累计飞行/交接计算
    def arrival_offsets(self) -> list[float]:
        offs, t = [], self.prep
        for i, leg in enumerate(self.legs):
            t += leg.t_total
            offs.append(t)
            if i < len(self.handover) - 1 or i < len(self.legs) - 1:
                t += self.handover[i]
        return offs


def build_candidates(scn: Scenario, boxes: list[Box], relax: frozenset = frozenset(),
                     single_area: bool = False) -> list[TripCandidate]:
    """为给定货箱集合构造所有满足（未被 relax 删除的）硬约束的架次候选。"""
    areas = sorted({b.area for b in boxes})
    if single_area and len(areas) > 1:
        return []
    area_seqs = [areas] if single_area else [list(p) for p in itertools.permutations(areas)]
    out: list[TripCandidate] = []
    total_mass = sum(b.mass for b in boxes)
    total_volume = sum(b.volume for b in boxes)

    for mid, model in scn.transport_models.items():
        # --- 质量与体积必须分别检查（方案 4.1） ---
        if total_mass > model.max_payload + 1e-9:
            continue
        if "volume" not in relax and total_volume > model.volume + 1e-9:
            continue
        for seq in area_seqs:
            stops_boxes = [[b.id for b in boxes if b.area == a] for a in seq]
            nodes = [scn.depot] + [scn.areas[a] for a in seq] + [scn.depot]
            legs, energy, flight = [], 0.0, 0.0
            q = total_mass
            ok = True
            for i in range(len(nodes) - 1):
                leg = build_leg(model, nodes[i], nodes[i + 1], scn.dem)
                legs.append(leg)
                e = leg_energy_transport(model, leg, q)
                if e > model.battery_energy:       # 单段都飞不完，必然不可行
                    ok = False
                    break
                energy += e
                flight += leg.t_total
                if i < len(seq):                    # 投送后剩余载荷下降
                    q -= sum(b.mass for b in boxes if b.area == seq[i])
            if not ok:
                continue
            limit = (1 - model.reserve_ratio) * model.battery_energy
            if "reserve" not in relax and energy > limit + 1e-9:
                continue
            if "reserve" in relax and energy > model.battery_energy + 1e-9:
                continue
            n_boxes = len(boxes)
            prep = model.ground_prep + model.load_per_box * n_boxes
            handover = [model.handover_base + model.handover_per_box * len(g)
                        for g in stops_boxes]
            duration = prep + flight + sum(handover)
            out.append(TripCandidate(
                boxes=[b.id for b in boxes], model=mid, seq=list(seq),
                stops_boxes=stops_boxes, energy=energy, flight_time=flight,
                prep=prep, handover=handover, duration=duration, legs=legs,
            ))
    return out


# ---------------------------------------------------------------- 时间表


class BatteryState:
    """一块电池的可用时刻（含充电周转）。"""

    def __init__(self, bid: str, model: str):
        self.id = bid
        self.model = model
        self.free_at = 0.0

    def ready(self, start: float) -> bool:
        return start >= self.free_at - 1e-9


class UavState:
    def __init__(self, uid: str, model: str):
        self.id = uid
        self.model = model
        self.free_at = 0.0


def schedule_trips(scn: Scenario, plans: list[TripPlan], relax: frozenset = frozenset()
                   ) -> tuple[list[tuple[TripPlan, float, float, float]], list[str]]:
    """给一组架次分配开始时刻（按给定顺序串行化到各自的无人机与电池上）。

    返回 ``[(trip, start, end, soc_end), ...]`` 与违规说明列表。
    时间规则：同一无人机的架次不得重叠；同一电池的架次之间必须完成充电；
    电池任务结束 SOC 由实际能耗决定。
    """
    violations: list[str] = []
    uav_free: dict[str, float] = {}
    battery_free: dict[str, float] = {}
    battery_soc: dict[str, float] = {}
    result = []
    for tp in plans:
        model = scn.transport_models[tp.model]
        cand_energy = tp.__dict__.get("_energy")
        uav_ready = uav_free.get(tp.uav, 0.0)
        bat_ready = battery_free.get(tp.battery, 0.0)
        start = max(uav_ready, bat_ready)
        end = start + tp.__dict__.get("_duration", 0.0)
        soc_end = 1.0 - cand_energy / model.battery_energy
        if "charge_overlap" in relax:
            battery_free[tp.battery] = end          # 突变：电池不充电即可复用
            soc_end_used = 1.0
        else:
            t_chg = charge_time(soc_end, scn.battery_charge_full[tp.model])
            battery_free[tp.battery] = end + t_chg
            soc_end_used = soc_end
        uav_free[tp.uav] = end
        if soc_end_used < model.reserve_ratio - 1e-9:
            violations.append(f"{tp.trip_id}: 结束 SOC {soc_end_used:.3f} 低于返航下限")
        battery_soc[tp.battery] = soc_end_used
        result.append((tp, start, end, soc_end_used))
    return result, violations


# ---------------------------------------------------------------- 主求解流程


def solve_exact(scn: Scenario, *, name: str = "small", relax: frozenset = frozenset(),
                single_area: bool = False, require_comm: bool = False,
                max_trips: int = MAX_TRIPS,
                max_candidates_per_block: int | None = None) -> Schedule | None:
    """对小实例做精确枚举，返回加权目标最优的可行方案。

    ``relax`` 可为以下标记（约束突变测试）：``volume``、``reserve``、
    ``charge_overlap``、``deadline``、``comm_midpoint``、``relay_sync``。

    ``max_candidates_per_block=None`` 时为严格穷举；给出正整数时，对每个货箱块仅按
    (能耗, 时长) 保留前 N 个候选，用于压缩较大实例（须在报告中声明为"受限枚举"）。
    """
    boxes = list(scn.boxes)
    best: tuple[float, Schedule] | None = None
    restricted = max_candidates_per_block is not None
    partitions = enumerate_partitions(boxes)
    for part in partitions:
        if len(part) > max_trips:
            continue
        # 逐块构造架次候选；任一块没有候选则整个划分不可行
        per_block: list[list[TripCandidate]] = []
        ok = True
        for block in part:
            cands = build_candidates(scn, block, relax=relax, single_area=single_area)
            if not cands:
                ok = False
                break
            if max_candidates_per_block is not None and len(cands) > max_candidates_per_block:
                cands = sorted(cands, key=lambda c: (c.energy, c.duration))[:max_candidates_per_block]
            per_block.append(cands)
        if not ok:
            continue

        for combo in itertools.product(*per_block):
            trips = _assign_resources(scn, combo, relax=relax)
            if trips is None:
                continue
            sched, plans = trips
            if require_comm:
                if not _comm_feasible(scn, plans, relax=relax):
                    continue
            if "deadline" not in relax and not _deadlines_ok(scn, plans, sched):
                continue
            obj = _objective(scn, plans, sched)
            if best is None or obj < best[0]:
                best = (obj, sched)
    if best is None:
        return None
    sched = best[1]
    sched.reported = objective_components(scn, sched.trips)
    sched.reported["objective"] = best[0]
    sched.relax = relax
    return sched


def _assign_resources(scn: Scenario, combo: tuple[TripCandidate, ...],
                      relax: frozenset):
    """为架次组合分配无人机、电池与时刻（按机型枚举槽位）。"""
    # 每个架次的候选槽位：同机型的（无人机, 电池）组合
    slots_per_trip = []
    for cand in combo:
        uavs = scn.uavs_of_model(cand.model)
        bats = [f"{cand.model}-{i+1}" for i in range(scn.battery_stock[cand.model])]
        if not uavs or not bats:
            return None
        slots_per_trip.append([(u, b) for u in uavs for b in bats])

    best = None
    # 架次执行顺序（并行/串行由资源自由时刻决定）
    for order in itertools.permutations(range(len(combo))):
        real_cands = [combo[i] for i in order]
        for slot_choice in itertools.product(*[slots_per_trip[i] for i in order]):
            uav_free: dict[str, float] = {}
            bat_free: dict[str, float] = {}
            rows = []
            feasible = True
            for cand, (uav, bat) in zip(real_cands, slot_choice):
                model = scn.transport_models[cand.model]
                start = max(uav_free.get(uav, 0.0), bat_free.get(bat, 0.0))
                end = start + cand.duration
                soc_end = 1.0 - cand.energy / model.battery_energy
                if soc_end < model.reserve_ratio - 1e-9 and "reserve" not in relax:
                    feasible = False
                    break
                uav_free[uav] = end
                if "charge_overlap" in relax:
                    bat_free[bat] = end
                else:
                    bat_free[bat] = end + charge_time(soc_end, scn.battery_charge_full[cand.model])
                rows.append((cand, uav, bat, start, end, soc_end))
            if not feasible:
                continue
            key = max(r[4] for r in rows)
            total_energy = sum(r[0].energy for r in rows)
            obj = (WEIGHTS["makespan"] * key + WEIGHTS["energy"] * total_energy
                   + WEIGHTS["trips"] * len(rows))
            if best is None or obj < best[0]:
                best = (obj, rows)
    if best is None:
        return None

    plans, sched_trips = [], []
    for idx, (cand, uav, bat, start, end, soc_end) in enumerate(best[1]):
        tp = TripPlan(
            trip_id=f"T{idx+1:02d}", uav=uav, model=cand.model, battery=bat,
            start_time=start, seq=list(cand.seq), boxes_per_stop=[list(g) for g in cand.stops_boxes],
        )
        tp.__dict__["_candidate"] = cand
        plans.append(tp)
        sched_trips.append(tp)
    sched = Schedule(name="pending", trips=sched_trips)
    return sched, plans


def _deadlines_ok(scn: Scenario, plans: list[TripPlan], sched: Schedule) -> bool:
    """医疗物资与首批保障货箱必须满足截止时间。"""
    box_map = {b.id: b for b in scn.boxes}
    for tp in sched.trips:
        cand = tp.__dict__["_candidate"]
        offs = cand.arrival_offsets()
        for i, area in enumerate(tp.seq):
            deliver = tp.start_time + offs[i] + cand.handover[i]
            for bid in tp.boxes_per_stop[i]:
                b = box_map[bid]
                if b.is_first_batch and b.first_deadline is not None:
                    if deliver > b.first_deadline + 1e-9:
                        return False
    return True


def _comm_feasible(scn: Scenario, plans: list[TripPlan], relax: frozenset) -> bool:
    """架次全程通信保障检查（Q3）。"""
    from .comm_audit import check_trip_communication
    for tp in plans:
        cand = tp.__dict__["_candidate"]
        res = check_trip_communication(scn, tp, cand, relax=relax)
        if not res["feasible"]:
            return False
    return True


def objective_components(scn: Scenario, trips: list[TripPlan]) -> dict:
    """目标分量（求解器侧）。"""
    makespan = 0.0
    energy = 0.0
    late = 0.0
    box_map = {b.id: b for b in scn.boxes}
    for tp in trips:
        cand = tp.__dict__["_candidate"]
        offs = cand.arrival_offsets()
        makespan = max(makespan, tp.start_time + cand.duration)
        energy += cand.energy
        for i in range(len(tp.seq)):
            deliver = tp.start_time + offs[i] + cand.handover[i]
            for bid in tp.boxes_per_stop[i]:
                b = box_map[bid]
                if not b.is_first_batch and deliver > b.expected_time + 1e-9:
                    late += deliver - b.expected_time
    return {
        "makespan": makespan,
        "total_energy": energy,
        "n_trips": len(trips),
        "lateness": late,
        "objective": (WEIGHTS["makespan"] * makespan + WEIGHTS["energy"] * energy
                      + WEIGHTS["trips"] * len(trips) + WEIGHTS["lateness"] * late),
    }


def _objective(scn: Scenario, plans: list[TripPlan], sched: Schedule) -> float:
    comp = objective_components(scn, sched.trips)
    return comp["objective"]


# ---------------------------------------------------------------- Q1 专用


def q1_batch_plan(scn: Scenario, area: str, travel_reserve: float | None = None
                  ) -> dict:
    """问题 1：单服务区直接往返的货箱组批（按架次数 -> 能耗 -> 时间排序）。"""
    boxes = scn.boxes_of(area)
    results = []
    for part in enumerate_partitions(boxes):
        cands_per_block = [build_candidates(scn, blk, single_area=True) for blk in part]
        if any(not c for c in cands_per_block):
            continue
        best_combo = None
        for combo in itertools.product(*cands_per_block):
            energy = sum(c.energy for c in combo)
            time_ = sum(c.duration for c in combo)
            key = (len(combo), energy, time_)
            if best_combo is None or key < best_combo[0]:
                best_combo = (key, combo)
        if best_combo:
            results.append(best_combo)
    if not results:
        return {"area": area, "feasible": False}
    results.sort(key=lambda r: (r[0][0], r[0][1], r[0][2]))
    key, combo = results[0]
    hits = []
    for r in results[:5]:
        hits.append((r[0][0], r[0][1], r[0][2]))
    return {
        "area": area, "feasible": True,
        "batches": [(c.model, c.boxes, round(c.energy, 4)) for c in combo],
        "n_trips": key[0], "total_energy": key[1], "total_time": key[2],
        "top5_counterfactuals": hits,
    }


def max_safe_payload(scn: Scenario, area: str) -> dict:
    """问题 1：各机型在给定服务区往返的最大安全载荷（二分搜索）。"""
    out = {}
    node = scn.areas[area]
    for mid, model in scn.transport_models.items():
        lo, hi = 0.0, model.max_payload
        best = 0.0
        for _ in range(40):
            mid_q = (lo + hi) / 2
            nodes = [scn.depot, node, scn.depot]
            energy = 0.0
            q = mid_q
            for i in range(2):
                leg = build_leg(model, nodes[i], nodes[i + 1], scn.dem)
                energy += leg_energy_transport(model, leg, q)
                q = 0.0
            limit = (1 - model.reserve_ratio) * model.battery_energy
            if energy <= limit + 1e-12:
                best = mid_q
                lo = mid_q
            else:
                hi = mid_q
        out[mid] = round(best, 4)
    return out


# ---------------------------------------------------------------- Q4


def interval_coloring_min_resources(intervals: list[tuple[float, float]]) -> int:
    """区间着色的最少资源数（等价于最大同时占用数）。"""
    if not intervals:
        return 0
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best
