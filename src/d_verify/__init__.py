"""D 题小规模约束审计验证套件（对应 ``D题_小规模约束审计验证方案.md``）。

模块划分：

- :mod:`params`      —— 附件真实参数加载与统一小实例构造；
- :mod:`dem`         —— 真实 30 m DEM 与合成 DEM；
- :mod:`physics`     —— 附录 2 的飞行/能耗/充电公式；
- :mod:`comm`        —— 附录 3 的链路预算与通信状态判定；
- :mod:`plan`        —— 方案表数据契约；
- :mod:`solver_small`—— 小规模精确枚举求解器（含 Q1 专用子过程）；
- :mod:`audit`       —— 独立可行性审计器（自带独立物理重算）。
"""

__all__ = ["params", "dem", "physics", "comm", "plan", "solver_small", "audit", "comm_audit"]
