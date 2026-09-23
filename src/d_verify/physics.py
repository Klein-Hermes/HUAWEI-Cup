"""飞行时间、能耗、充电与航段几何（附录 2 的统一口径）。

附录 2 明确给出：

- 航段 = 两节点水平直线；计划巡航海拔 = 航段所经 DEM 像元最高地面高程 + 50 m；
- 每航段按爬升 / 巡航 / 下降三段计算；O01 作业高度 = 地面海拔，服务区作业高度 = 地面海拔 + 30 m；
  连续访问多个服务区时，每次投送后均从 30 m 作业高度重新爬升；
- ``L_g(q) = L_g0 - (L_g0 - L_gF) * (q / Q_g) ** 1.5``（0 <= q <= Q_g）；
- ``t_gij = h+ / v_up + d / v_c + h- / v_down``；
- ``E_gij(q) = E_hor(q) + E_up(q)``；下降能耗效率取 0，不单独计算下降附加能耗；
- 返航安全余量：``sum(leg energy) <= (1 - rho_g) * E_use``；
- 充电两段等效模型。

!!! 口径假设（题目未明写，必须与论文口径一致）!!!

1. 水平巡航能耗按等效航程线性折算：``E_hor = (d / L_g(q)) * E_use``。
   即"该载荷下的标准航程"对应"单组电池可用能量"，飞 d 米按比例消耗能量。
2. 爬升附加能耗按重力做功/效率计算：``E_up = (m_empty + q) * g * h+ / eta_up``，
   其中 ``m_empty`` 为含电池空载总质量（中继机取计划起飞总质量）。
3. 若某节点作业高度已高于巡航海拔（例如服务区地面海拔高于航段最高 DEM 像元 + 50 m），
   该段爬升高度取 0，不做负爬升。

以上三条是本验证套件统一采用的口径，审计器与求解器共用同一物理常量，
但各自独立组合（审计器不调用求解器的约束判断函数）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import Node, RelayModel, Scenario, TransportModel, haversine_m

G = 9.80665
J_PER_KWH = 3.6e6

#: O01 作业高度相对地面的偏移（m）
DEPOT_WORK_ALT_AGL = 0.0
#: 服务区作业高度相对地面的偏移（m，附录 2 规定 30 m）
AREA_WORK_ALT_AGL = 30.0
#: 计划巡航海拔相对航段最高 DEM 像元的余量（m，附录 2 规定 50 m）
CRUISE_CLEARANCE = 50.0


# ------------------------------------------------------------ 等效航程 / 能耗


def equivalent_range(model: TransportModel, q: float) -> float:
    """机型等效航程 L_g(q)（m）。q 超出 [0, Q] 时抛错。"""
    if q < -1e-9:
        raise ValueError(f"载荷不能为负：{q}")
    if q > model.max_payload + 1e-9:
        raise ValueError(f"载荷 {q} 超过机型 {model.id} 最大载货质量 {model.max_payload}")
    q = min(max(q, 0.0), model.max_payload)
    return model.range_empty - (model.range_empty - model.range_full) * (q / model.max_payload) ** 1.5


def horizontal_energy(model: TransportModel, d_m: float, q: float) -> float:
    """水平巡航能耗（kWh）：按等效航程线性折算。"""
    return d_m / equivalent_range(model, q) * model.battery_energy


def climb_energy(empty_mass: float, q: float, h_up: float, eta_up: float) -> float:
    """爬升附加能耗（kWh）。h_up <= 0 时返回 0（下降能耗效率为 0，不单独计算）。"""
    if h_up <= 0:
        return 0.0
    if eta_up <= 0:
        raise ValueError("爬升能耗效率必须为正")
    return (empty_mass + q) * G * h_up / eta_up / J_PER_KWH


def charge_time(soc: float, t_full: float) -> float:
    """两阶段等效充电模型（附录 2）：SOC s -> 100% 所需时间（s）。"""
    if soc < -1e-9 or soc > 1 + 1e-9:
        raise ValueError(f"SOC 必须落在 [0, 1]：{soc}")
    s = min(max(soc, 0.0), 1.0)
    if s < 0.90:
        return t_full * (0.65 * (0.90 - s) / 0.90 + 0.35)
    return t_full * 0.35 * (1.0 - s) / 0.10


# ---------------------------------------------------------------- 航段几何


@dataclass(frozen=True)
class Leg:
    """一个已定几何的航段（不含载荷信息）。"""

    from_id: str
    to_id: str
    d_m: float            # 水平巡航距离
    h_up: float           # 爬升高度
    h_dn: float           # 下降高度
    cruise_alt: float     # 计划巡航海拔
    start_alt: float      # 起点作业高度
    end_alt: float         # 终点作业高度
    t_up: float
    t_cruise: float
    t_dn: float

    @property
    def t_total(self) -> float:
        return self.t_up + self.t_cruise + self.t_dn


def work_altitude(node: Node) -> float:
    """节点作业高度：O01 取地面海拔，服务区取地面海拔 + 30 m。"""
    agl = DEPOT_WORK_ALT_AGL if node.is_depot else AREA_WORK_ALT_AGL
    return node.elevation + agl


def cruise_altitude(dem, n1: Node, n2: Node) -> float:
    """航段计划巡航海拔 = 航段所经 DEM 像元最高地面高程 + 50 m。"""
    if dem is None:
        # 无 DEM 时退化为两节点地面海拔的较高者（仅用于纯公式测试）
        return max(n1.elevation, n2.elevation) + CRUISE_CLEARANCE
    top = dem.max_elevation_along((n1.lat, n1.lon), (n2.lat, n2.lon))
    if not math.isfinite(top):
        top = max(n1.elevation, n2.elevation)
    return float(top) + CRUISE_CLEARANCE


#: 航段几何缓存：键包含 DEM、节点身份/坐标与机型。
#:
#: 测试和参数扫描可能在同一进程中复用节点编号、但替换节点坐标（例如
#: 构造远距虚拟服务区）。只使用节点编号会把旧坐标的几何结果错误复用到
#: 新坐标上，进而污染能耗和返航余量边界判断。
_LEG_CACHE: dict = {}


def build_leg(model: TransportModel | RelayModel, n1: Node, n2: Node, dem) -> Leg:
    """构造航段几何与三段飞行时间（带缓存，便于小规模精确枚举）。"""
    def node_key(node: Node) -> tuple:
        return (node.id, node.lat, node.lon, node.elevation, node.is_depot)

    key = (id(dem), node_key(n1), node_key(n2), model.id)
    hit = _LEG_CACHE.get(key)
    if hit is not None:
        return hit
    d = haversine_m(n1.lat, n1.lon, n2.lat, n2.lon)
    ca = cruise_altitude(dem, n1, n2)
    a1, a2 = work_altitude(n1), work_altitude(n2)
    h_up = max(ca - a1, 0.0)
    h_dn = max(ca - a2, 0.0)
    leg = Leg(
        from_id=n1.id, to_id=n2.id, d_m=d, h_up=h_up, h_dn=h_dn,
        cruise_alt=ca, start_alt=a1, end_alt=a2,
        t_up=h_up / model.v_up, t_cruise=d / model.cruise_speed, t_dn=h_dn / model.v_down,
    )
    _LEG_CACHE[key] = leg
    return leg


def build_route_legs(model: TransportModel | RelayModel, seq: list[Node], dem) -> list[Leg]:
    """按节点序列构造连续航段。"""
    return [build_leg(model, seq[i], seq[i + 1], dem) for i in range(len(seq) - 1)]


def leg_energy_transport(model: TransportModel, leg: Leg, q: float) -> float:
    """运输航段能耗（kWh）：水平巡航 + 爬升附加。"""
    return horizontal_energy(model, leg.d_m, q) + climb_energy(model.empty_mass, q, leg.h_up, model.eta_up)


def position_at(leg: Leg, n1: Node, n2: Node, t: float) -> tuple[float, float, float]:
    """航段内 t 秒（0 <= t <= t_total）时无人机的三维位置 (lat, lon, alt)。

    爬升 / 巡航 / 下降三段线性插值；t 超过航段时长时按终点返回（便于容差处理）。
    """
    t = max(0.0, min(t, leg.t_total))
    if t <= leg.t_up and leg.t_up > 0:
        frac = t / leg.t_up
        alt = leg.start_alt + (leg.cruise_alt - leg.start_alt) * frac
        lat = n1.lat
        lon = n1.lon
        return lat, lon, alt
    t2 = t - leg.t_up
    if t2 <= leg.t_cruise:
        frac = t2 / leg.t_cruise if leg.t_cruise > 0 else 1.0
        lat = n1.lat + (n2.lat - n1.lat) * frac
        lon = n1.lon + (n2.lon - n1.lon) * frac
        return lat, lon, leg.cruise_alt
    t3 = t2 - leg.t_cruise
    frac = t3 / leg.t_dn if leg.t_dn > 0 else 1.0
    frac = min(frac, 1.0)
    alt = leg.cruise_alt + (leg.end_alt - leg.cruise_alt) * frac
    lat = n1.lat + (n2.lat - n1.lat) * frac
    lon = n1.lon + (n2.lon - n1.lon) * frac
    return lat, lon, alt


# ---------------------------------------------------------------- 中继能耗


def relay_leg_energy(model: RelayModel, leg: Leg) -> float:
    """中继航段能耗（kWh）：巡航功率 x 巡航时间 + 爬升附加能耗（质量取计划起飞总质量）。"""
    e_cruise = model.cruise_power * leg.t_cruise / 3600.0
    e_up = climb_energy(model.takeoff_mass, 0.0, leg.h_up, model.eta_up)
    return e_cruise + e_up


def relay_service_energy(model: RelayModel, service_seconds: float) -> float:
    """中继通信服务能耗（kWh）：（悬停功率 + 通信附加功率）x 服务时长。"""
    return (model.hover_power + model.comm_power) * service_seconds / 3600.0


# ---------------------------------------------------------------- 目标分量


@dataclass
class TripSummary:
    """架次指标汇总（求解器与审计器各自独立生成，用于交叉比对）。"""

    trip_id: str
    model_id: str
    seq: list[str]
    leg_payload: list[float]
    leg_energy: list[float]
    total_energy: float
    flight_time: float
    total_time: float
    energy_limit: float
    feasible: bool


def summarize_trip(scenario: Scenario, trip_id: str, model: TransportModel,
                   seq_nodes: list[Node], box_mass_per_stop: list[float],
                   *, prep_time: float | None = None,
                   handover_time: float | None = None,
                   n_boxes_per_stop: list[int] | None = None) -> TripSummary:
    """按给定路线与每站投递质量计算架次指标（含时限与交接时间的简化版）。"""
    legs = build_route_legs(model, seq_nodes, scenario.dem)
    payload = sum(box_mass_per_stop)
    leg_payload, leg_energy = [], []
    q = payload
    for i, leg in enumerate(legs):
        leg_payload.append(q)
        leg_energy.append(leg_energy_transport(model, leg, q))
        if i < len(box_mass_per_stop):       # 投送后载荷下降
            q -= box_mass_per_stop[i]
    total_energy = sum(leg_energy)
    flight_time = sum(l.t_total for l in legs)
    if prep_time is None:
        prep_time = model.ground_prep + model.load_per_box * sum(n_boxes_per_stop or [])
    if handover_time is None:
        handover_time = sum(
            model.handover_base + model.handover_per_box * nb
            for nb in (n_boxes_per_stop or [0] * len(seq_nodes))
        )
    limit = (1 - model.reserve_ratio) * model.battery_energy
    return TripSummary(
        trip_id=trip_id, model_id=model.id, seq=[n.id for n in seq_nodes],
        leg_payload=leg_payload, leg_energy=leg_energy, total_energy=total_energy,
        flight_time=flight_time, total_time=flight_time + prep_time + handover_time,
        energy_limit=limit, feasible=total_energy <= limit + 1e-9,
    )
