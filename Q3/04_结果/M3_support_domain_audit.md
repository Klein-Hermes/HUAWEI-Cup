# Q3 M3 支持域审计

**生成时间（UTC）：** 2026-09-24T15:26:03+00:00  
**候选域来源：** `Q2 B1 observed 1,176-point grid; matches the existing M0 search domain`  
**M3 主分析口径：** `exact_observed_only; hull and distance are diagnostics`  

- 实际观测点：1,176；候选点：1,176。
- 候选中精确匹配观测组合：1,176；凸包内：1,176；凸包内非观测插值候选：0；凸包外：0。
- 当前允许 M1 搜索的候选点：1,176。凸包和标准化对数距离仅作诊断，不自动开放插值。
- 凸包在标准化 $(\log N_B,\log D_B)$ 空间中构造；每轴使用观测支持的总体标准差标准化。凸包顶点数：4。
- 最近点距离保存在 `m3_q3_support_grid.csv`，不设未经验证的距离阈值。

当前主支路沿用实测离散域。若未来开放插值，需另提供候选网格、插值验证证据和预先冻结的接受规则；凸包内本身不足以放行。

复现命令：`python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m3`。
