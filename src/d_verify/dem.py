"""高程模型：真实 30 m DEM（.mat）与合成 DEM 的统一接口。

接口约定（附录 2）：

- 航段巡航海拔 = 该航段水平直线所经过 DEM 像元的"最高地面高程" + 50 m；
- 因此需要 ``max_elevation_along``：沿水平直线采样所有经过的像元并取最大高程。

真实 DEM：``镇龙乡及周边30米DEM.mat``（1309×1486，EPSG:4326，nodata=-32767），
``transform`` 给出左上角角点坐标 + 像元尺寸，``latitude``/``longitude`` 为像元中心坐标。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

NODATA = -32767.0


class ElevationModel:
    """高程模型基类：经纬度 -> 地面高程（m）。"""

    def elevation(self, lat: float, lon: float) -> float:  # pragma: no cover - 抽象
        raise NotImplementedError

    def max_elevation_along(self, p1: tuple[float, float], p2: tuple[float, float],
                            sample_step_m: float = 15.0) -> float:
        """水平直线航段所经像元的最高地面高程。

        ``p1``/``p2`` 为 ``(lat, lon)``。沿途按 ``sample_step_m`` 步长采样，
        步长默认取 30 m 像元的一半，保证不会跨过整个像元。
        """
        lat1, lon1 = p1
        lat2, lon2 = p2
        cache = getattr(self, "_max_cache", None)
        if cache is None:
            cache = self._max_cache = {}
        key = (round(lat1, 9), round(lon1, 9), round(lat2, 9), round(lon2, 9), sample_step_m)
        if key in cache:
            return cache[key]
        dist = _approx_distance_m(lat1, lon1, lat2, lon2)
        n = max(int(math.ceil(dist / sample_step_m)), 1)
        best = -np.inf
        for k in range(n + 1):
            t = k / n
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            e = self.elevation(lat, lon)
            if e is not None and np.isfinite(e):
                best = max(best, e)
        cache[key] = float(best)
        return float(best)

    def line_of_sight_clear(self, a: tuple[float, float, float],
                            b: tuple[float, float, float],
                            sample_step_m: float = 15.0) -> bool:
        """判断三维端点 a、b 之间的视线是否无地形遮挡（True = 无遮挡）。

        端点格式 ``(lat, lon, alt_m)``。沿水平投影采样，检查视线高度是否低于地面高程。
        """
        lat1, lon1, alt1 = a
        lat2, lon2, alt2 = b
        cache = getattr(self, "_los_cache", None)
        if cache is None:
            cache = self._los_cache = {}
        key = (round(lat1, 9), round(lon1, 9), round(alt1, 6),
               round(lat2, 9), round(lon2, 9), round(alt2, 6), sample_step_m)
        if key in cache:
            return cache[key]
        dist = _approx_distance_m(lat1, lon1, lat2, lon2)
        n = max(int(math.ceil(dist / sample_step_m)), 1)
        for k in range(1, n):  # 端点本身不参与遮挡判定
            t = k / n
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            alt = alt1 + (alt2 - alt1) * t
            terrain = self.elevation(lat, lon)
            if terrain is not None and np.isfinite(terrain) and alt < terrain:
                cache[key] = False
                return False
        cache[key] = True
        return True


class RealDEM(ElevationModel):
    """真实 30 m DEM。"""

    def __init__(self, dem: np.ndarray, origin_lon: float, origin_lat: float,
                 res: float):
        self.dem = dem.astype(np.float64)
        self.origin_lon = origin_lon   # 左上角角点经度
        self.origin_lat = origin_lat   # 左上角角点纬度
        self.res = res                 # 像元尺寸（度）
        self.rows, self.cols = dem.shape

    @classmethod
    def load(cls, path: Path) -> "RealDEM":
        import scipy.io as sio

        m = sio.loadmat(str(path))
        dem = m["dem"]
        t = m["transform"].ravel()
        # transform = [res_lon, 0, origin_lon, 0, -res_lat, origin_lat]
        return cls(dem, origin_lon=float(t[2]), origin_lat=float(t[5]), res=float(t[0]))

    # -- 索引换算 -------------------------------------------------------
    def _indices(self, lat: float, lon: float) -> tuple[float, float]:
        col = (lon - self.origin_lon) / self.res - 0.5
        row = (self.origin_lat - lat) / self.res - 0.5
        return row, col

    def in_bounds(self, lat: float, lon: float) -> bool:
        row, col = self._indices(lat, lon)
        return 0 <= row <= self.rows - 1 and 0 <= col <= self.cols - 1

    def elevation(self, lat: float, lon: float) -> float:
        """最近邻取高程（与“所经过 DEM 像元”的口径一致）。"""
        row, col = self._indices(lat, lon)
        i = int(round(row))
        j = int(round(col))
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            return float("nan")
        # nodata 像元：向四周搜索最近的有效像元
        v = self.dem[i, j]
        if v == NODATA or not np.isfinite(v):
            for rad in range(1, 6):
                i0, i1 = max(0, i - rad), min(self.rows, i + rad + 1)
                j0, j1 = max(0, j - rad), min(self.cols, j + rad + 1)
                patch = self.dem[i0:i1, j0:j1]
                ok = patch[(patch != NODATA) & np.isfinite(patch)]
                if ok.size:
                    return float(ok.mean())
            return float("nan")
        return float(v)


class SyntheticDEM(ElevationModel):
    """合成 DEM：一个矩形网格，可选在指定经纬度注入遮挡山脊。

    用于方案 6.1 的"中间时刻被地形遮挡、端点可用"反例：把山脊放在航段中段。
    """

    def __init__(self, lat0: float, lon0: float, dlat: float, dlon: float,
                 base_elev: float = 100.0):
        self.lat0, self.lon0 = lat0, lon0
        self.dlat, self.dlon = dlat, dlon
        self.base = base_elev
        self.features: list[tuple[float, float, float, float]] = []  # (lat, lon, sigma_deg, height)

    def add_ridge(self, lat: float, lon: float, sigma_deg: float, height: float) -> None:
        """在 (lat, lon) 处注入一个高斯山脊，附加高度 height（m）。"""
        self.features.append((lat, lon, sigma_deg, height))

    def elevation(self, lat: float, lon: float) -> float:
        e = self.base
        for (flat, flon, sigma, height) in self.features:
            d2 = ((lat - flat) ** 2 + (lon - flon) ** 2) / (sigma ** 2)
            e += height * math.exp(-d2)
        return e


def _approx_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """小范围等距近似（用于采样步数估计）。"""
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110574.0
    return math.hypot(dx, dy)
