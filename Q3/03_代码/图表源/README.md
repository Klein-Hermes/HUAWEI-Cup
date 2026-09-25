# Q3 补充图集绘图源

本目录保留 P1–P4 的共享绘图核心和论文版入口。脚本已纳入 Q3，并使用项目内 Q1/Q3 输入；不读取 `zwj/`。原图集输出继续保存在 `Q3/04_结果/figures/supplementary_20260925/`，重绘时写入单独的 `_rebuild_supplementary_full/` 或 `_rebuild_supplementary_paper/` 暂存目录，不覆盖当前图集。

从项目根目录运行：

```powershell
python Q3/03_代码/图表源/plot_q3_supplementary_figures.py
python Q3/03_代码/图表源/论文版/plot_q3_supplementary_figures_paper.py
```

脚本输入来自 Q1/Q3 的现有 CSV 和代码；图表样式检查还调用本机安装的数学建模可视化工具。论文版的排版决定和建议图注见 `Q3/04_结果/figures/supplementary_20260925/论文版排版决定与建议图注.md`。当前交付图的哈希仍由图集的 `supplementary_manifest.json` 记录。
