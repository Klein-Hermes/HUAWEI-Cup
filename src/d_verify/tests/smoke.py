"""快速自检：参数加载 / 附录 2 公式 / 附录 3 链路预算 / 真实 DEM 航段 / Q1 小规模子过程。

用法（在仓库根目录）：``python -B src/d_verify/tests/smoke.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from d_verify import comm, params, physics, solver_small  # noqa: E402
from d_verify.dem import RealDEM  # noqa: E402


def main() -> int:
    sc = params.load_scenario()
    print("[1] 机型:", {k: (v.max_payload, v.volume, v.battery_energy)
                      for k, v in sc.transport_models.items()})
    print("[2] 机队:", sc.transport_uavs)
    print("[3] 电池:", sc.battery_stock, sc.battery_charge_full,
          "| 中继:", sc.relay_uavs, sc.relay_energy_stock, sc.relay_charge_full)
    print("[4] P_th:", sc.link.p_threshold, "| f:", sc.link.freq_mhz,
          "| G01(Pt,G):", sc.link.tx_gateway, "| 网关天线离地:", sc.link.gateway_agl)

    ok = True
    for s, exp in [(0.0, 1800.0), (0.5, 1150.0), (0.9, 630.0), (1.0, 0.0)]:
        got = physics.charge_time(s, 1800.0)
        flag = abs(got - exp) < 1e-6
        ok &= flag
        print(f"[5] t_chg({s}) = {got:.4f} 期望 {exp} -> {'OK' if flag else 'MISMATCH'}")

    print("[6] 双向门限 直连:", comm.bidirectional_threshold(sc.link, comm.DIRECT_LINK),
          "接入:", comm.bidirectional_threshold(sc.link, comm.ACCESS_LINK),
          "回传:", comm.bidirectional_threshold(sc.link, comm.BACKHAUL_LINK))

    dem = RealDEM.load(params.DEM_MAT)
    sc.dem = dem
    n1, n2 = sc.depot, sc.areas["S001"]
    print("[7] O01->S001 水平距离(m):",
          round(params.haversine_m(n1.lat, n1.lon, n2.lat, n2.lon), 1))
    print("[8] 航段最高 DEM(m):",
          round(dem.max_elevation_along((n1.lat, n1.lon), (n2.lat, n2.lon)), 2))
    m = sc.transport_models["A"]
    leg = physics.build_leg(m, n1, n2, dem)
    print("[9] A 型 O01->S001: d=%.1f h+=%.1f h-=%.1f cruise=%.1f t=%.1fs"
          % (leg.d_m, leg.h_up, leg.h_dn, leg.cruise_alt, leg.t_total))
    print("    q=0 能耗=%.6f kWh | q=25 能耗=%.6f kWh"
          % (physics.leg_energy_transport(m, leg, 0.0),
             physics.leg_energy_transport(m, leg, 25.0)))

    small = params.build_small_scenario(dem=dem)
    print("[10] 小实例: 服务区", sorted(small.areas), "货箱", len(small.boxes),
          "无人机", small.transport_uavs, "电池", small.battery_stock)

    # Q1 子过程：A/B/C 型在 S001 的最大安全载荷（单点往返）
    msp = solver_small.max_safe_payload(small, "S001")
    print("[11] S001 最大安全载荷(kg):", msp)
    over = {k: v for k, v in msp.items() if v > small.transport_models[k].max_payload + 1e-9}
    if over:
        print("    !! 超过机型最大载货质量:", over)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
