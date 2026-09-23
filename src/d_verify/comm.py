"""通信链路计算与服务状态判定（附录 3）。

规则要点：

- 门限 ``P_th = P_sens + M``；方向允许损耗 ``L_max(a->b) = Pt_a + Gt_a + Gr_b - L_sys - P_th``；
  双向链路取两个方向较小值；
- ``L_FSPL = 32.45 + 20*log10(f/MHz) + 20*log10(D/km)``；
- ``L_path = L_FSPL + L_obs * b``，b 为视线被地形遮挡的 0/1 变量；
- 直连优先级最高，其次中继（接入链路与回传链路必须"同一时刻同时可用"），否则中断；
- 一架运输无人机在任一时刻只能由 G01 或一架中继无人机保障。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .params import LinkParams

#: 通信端点角色
TRANSPORT = "transport"
RELAY_ACCESS = "relay_access"
RELAY_BACKHAUL = "relay_backhaul"
GATEWAY = "gateway"


class CommState(str, Enum):
    DIRECT = "直连"
    RELAYED = "中继"
    OUTAGE = "中断"


@dataclass(frozen=True)
class LinkKind:
    """一种链路的两端角色。"""

    a: str
    b: str


DIRECT_LINK = LinkKind(TRANSPORT, GATEWAY)
ACCESS_LINK = LinkKind(TRANSPORT, RELAY_ACCESS)
BACKHAUL_LINK = LinkKind(RELAY_BACKHAUL, GATEWAY)


def free_space_loss(freq_mhz: float, d_km: float) -> float:
    """自由空间传播损耗（dB）。f 以 MHz、D 以 km 计。"""
    if d_km <= 0:
        return 0.0
    return 32.45 + 20 * math.log10(freq_mhz) + 20 * math.log10(d_km)


def tx_params(link: LinkParams, role: str) -> tuple[float, float]:
    """端点角色的 (发射功率 dBm, 天线增益 dBi)。"""
    if role == TRANSPORT:
        return link.tx_transport
    if role == RELAY_ACCESS:
        return link.tx_relay_access
    if role == RELAY_BACKHAUL:
        return link.tx_relay_backhaul
    if role == GATEWAY:
        return link.tx_gateway
    raise KeyError(role)


def max_allowable_loss(link: LinkParams, tx_role: str, rx_role: str) -> float:
    """方向 tx_role -> rx_role 的最大允许总传播损耗（dB）。"""
    pt, gt = tx_params(link, tx_role)
    _, gr = tx_params(link, rx_role)
    return pt + gt + gr - link.l_sys - link.p_threshold


def bidirectional_threshold(link: LinkParams, kind: LinkKind) -> float:
    """双向链路门限 = min(L_max(a->b), L_max(b->a))。"""
    return min(max_allowable_loss(link, kind.a, kind.b),
               max_allowable_loss(link, kind.b, kind.a))


def terrain_blocked(dem, p1: tuple[float, float, float],
                    p2: tuple[float, float, float]) -> bool:
    """地形遮挡变量 b_ij(t)：视线被地形阻挡返回 True。"""
    if dem is None:
        return False
    return not dem.line_of_sight_clear(p1, p2)


def link_available(link: LinkParams, kind: LinkKind, dem,
                   p_a: tuple[float, float, float],
                   p_b: tuple[float, float, float]) -> bool:
    """单链路可用性 A_ij(t)。p 为 (lat, lon, alt_m)。"""
    d_km = _distance_km(p_a, p_b)
    blocked = terrain_blocked(dem, p_a, p_b)
    l_path = free_space_loss(link.freq_mhz, d_km) + (link.l_obs if blocked else 0.0)
    return l_path <= bidirectional_threshold(link, kind) + 1e-9


def transport_comm_state(link: LinkParams, dem,
                         p_transport: tuple[float, float, float],
                         p_gateway: tuple[float, float, float],
                         relays: list[tuple[str, tuple[float, float, float]]],
                         ) -> tuple[CommState, str | None]:
    """判定运输无人机在给定时刻的通信状态。

    返回 ``(状态, 使用的中继编号或 None)``。直连优先；否则任一中继的接入与回传
    链路同时可用即记为中继状态；都不满足为中断。
    """
    if link_available(link, DIRECT_LINK, dem, p_transport, p_gateway):
        return CommState.DIRECT, None
    for rid, p_relay in relays:
        if (link_available(link, ACCESS_LINK, dem, p_transport, p_relay)
                and link_available(link, BACKHAUL_LINK, dem, p_relay, p_gateway)):
            return CommState.RELAYED, rid
    return CommState.OUTAGE, None


def _distance_km(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """两个三维端点之间的直线距离（km）。"""
    lat1, lon1, alt1 = p1
    lat2, lon2, alt2 = p2
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110574.0
    dh = alt2 - alt1
    return math.sqrt(dx * dx + dy * dy + dh * dh) / 1000.0
