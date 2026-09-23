"""方案表结构：求解器输出、审计器输入的统一数据契约。

方案表只包含"决策变量"，所有派生量（载荷、能耗、SOC、交付时刻、通信状态）
都由求解器和审计器**各自独立**重新计算，以便交叉比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TripPlan:
    """一个运输架次（决策变量部分）。"""

    trip_id: str
    uav: str                       # 无人机编号
    model: str                     # 机型编号
    battery: str                   # 共享电池编号
    start_time: float              # 开始工位准备时刻（s）
    seq: list[str]                 # 服务区访问顺序（不含 O01）
    boxes_per_stop: list[list[str]]  # 每个服务区投递的货箱编号

    def all_boxes(self) -> list[str]:
        return [b for group in self.boxes_per_stop for b in group]


@dataclass
class RelayPlan:
    """一个中继架次（决策变量部分）。"""

    relay_id: str
    relay_uav: str
    energy_unit: str
    start_time: float
    hover_lat: float
    hover_lon: float
    hover_elev: float              # 悬停点地面海拔（m）
    hover_alt: float               # 悬停飞行海拔（m，离地高度 = hover_alt - hover_elev）
    service_start: float
    service_end: float

    @property
    def hover_agl(self) -> float:
        return self.hover_alt - self.hover_elev


@dataclass
class Schedule:
    """一个完整方案（可含 Q3 中继安排与 Q4 分区）。"""

    name: str
    trips: list[TripPlan] = field(default_factory=list)
    relays: list[RelayPlan] = field(default_factory=list)
    #: Q4 分区方案：{"2": {组号: [服务区...]}, "3": {...}}
    partitions: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    #: 求解器自报的目标分量（用于与审计器独立重算比对）
    reported: dict = field(default_factory=dict)
    #: 求解器使用的约束放宽标记（突变测试用），审计器不读取该字段做判断
    relax: frozenset = frozenset()

    def trip(self, trip_id: str) -> TripPlan:
        for t in self.trips:
            if t.trip_id == trip_id:
                return t
        raise KeyError(trip_id)
