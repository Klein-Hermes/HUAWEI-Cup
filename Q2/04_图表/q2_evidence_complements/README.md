# Q2 补充图：未覆盖证据视角

本组补充三个现有图表尚未直接展示的视角，不增加新模型结论：

1. result_q2_b3_track_level_errors：将 B3 总计 4,000 个插值点拆到 8 条 Pythia 轨迹展示，避免把检查点数当作独立实验数。
2. raw_q2_b10_extrapolation_support_map：显示 B10 估算点相对 B1 经验 N/D 支持点的位置。它展示外推域，不是外部验证。
3. result_q2_b6_b7_conditional_equal_loss：展示报告中的 B6/B7 半合成等 Loss 条件曲线；B6 的 Q=0.7 点为拟合插值。

每图导出 PDF、SVG、PNG 和灰度 PNG。合同、布局 QA、哈希及复现命令见 figure_contract.csv、figure_qa.json、manifest.json。

复现命令（项目根目录）：

    D:\\Anaconda\\python.exe src/q2_evidence_complement_figures.py

本组只消费已冻结结构化结果与可见审计材料；不读取隐藏文本，不拼接 A/B/半合成来源为真实联合样本。
