"""求解器侧的连续通信可行性检查（Q3 约束的编码实现）。

与审计器的通信检查相互独立：本模块复用 :mod:`physics` 的航段实现，
审计器（:mod:`audit`）自带一套独立实现，用于交叉验证。
"""

from __future__ import annotations

from .comm import (ACCESS_LINK, BACKHAUL_LINK, DIRECT_LINK, link_available)
from .params import Scenario
from .physics import position_at
from .plan import TripPlan


def check_trip_communication(scn: Scenario, trip: TripPlan, candidate,
                             relax: frozenset = frozenset(), dt: float = 10.0,
                             relays: list[tuple[str, tuple[float, float, float]]] | None = None
                             ) -> dict:
    """检查架次全过程（爬升/巡航/下降/投送）通信是否连续。

    ``relax`` 含 ``comm_midpoint`` 时只检查航段两端（突变：漏检中间时刻）；
    含 ``relay_sync`` 时中继只要求两段链路"各自"可用，不要求同刻可用。
    """
    gpos = (scn.depot.lat, scn.depot.lon, scn.depot.elevation + scn.link.gateway_agl)
    nodes = [scn.depot] + [scn.areas[a] for a in trip.seq] + [scn.depot]
    legs = candidate.legs
    gaps = 0
    samples = 0
    relay_list = relays or []
    for i, leg in enumerate(legs):
        if "comm_midpoint" in relax:
            taus = [0.0, leg.t_total]
        else:
            n = max(int(leg.t_total / dt), 1)
            taus = [leg.t_total * k / n for k in range(n + 1)]
        for tau in taus:
            pos = position_at(leg, nodes[i], nodes[i + 1], tau)
            samples += 1
            if not _comm_ok(scn, pos, gpos, relay_list, relax):
                gaps += 1
    return {"feasible": gaps == 0, "gaps": gaps, "samples": samples}


def _comm_ok(scn: Scenario, pos, gpos, relays, relax: frozenset) -> bool:
    if link_available(scn.link, DIRECT_LINK, scn.dem, pos, gpos):
        return True
    if "relay_sync" in relax:
        for _, rpos in relays:
            if (link_available(scn.link, ACCESS_LINK, scn.dem, pos, rpos)
                    or link_available(scn.link, BACKHAUL_LINK, scn.dem, rpos, gpos)):
                return True
        return False
    for _, rpos in relays:
        if (link_available(scn.link, ACCESS_LINK, scn.dem, pos, rpos)
                and link_available(scn.link, BACKHAUL_LINK, scn.dem, rpos, gpos)):
            return True
    return False
