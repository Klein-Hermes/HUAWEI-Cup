# Q2-B2 Cerebras 半合成轨迹压力测试规则

## 裁定

流程顺序正确：先核对 Cerebras 原始事实与 B2 的数据性质，再冻结 Pythia 校准规则，编写生成器和压力测试，最后只有在 B1 冻结证据通过后才生成正式版本。本项目的经典 B1 基线已有冻结报告、参数、拟合预测与可复现清单；生成器会逐项核验输入哈希、Gate 0 快照和 B1 输出哈希。

此处的“正式运行”仅指正式生成**版本化半合成压力测试产物**。它不把新轨迹改称 Cerebras 原始实验，也不据此声称独立外部验证。

## Cerebras 原始信息与本地附件差异

1. Cerebras 作者论文报告七个模型规模：111M、256M、590M、1.3B、2.7B、6.7B、13B；使用 Pile，并以约 20 tokens/parameter 进行 compute-optimal 训练。
2. 论文 Table 1 给出的训练 token 数分别为 2.2B、5.1B、11.8B、26.3B、53.0B、133.2B、257.1B。论文报告的 Pile test xent 分别为 2.608、2.349、2.181、1.997、1.834、1.704、1.572 nats/token。
3. 赛题附件中的 `cerebras_training_log.csv` 被数据说明及 `source_manifest.json` 标为半合成，不是原始 Cerebras 日志。它有 7×147 行，D 范围约 14–2050B，七个模型都使用相同的 D 检查点序列，且 GPU days、step time、gradient norm 全缺失；因此不得把这些字段补 0，也不得将该 D 轨迹描述为实际 Cerebras 训练计划。
4. 本地 `scaling_baseline.csv` 的七条 Cerebras 记录把终点统一写为 2,050B tokens，损失列为 2.95、2.70、2.50、2.35、2.22、2.08、2.00；这与上面的逐模型论文数据不符。它不参与新轨迹校准，正式生成清单记录其 SHA-256 和排除理由，原文件保持不变。
5. 本生成器按论文逐模型训练终点和 Pile test xent 锚定，不读取、不覆盖旧半合成 B2 的 `cerebras_training_log.csv`。论文引用：<https://arxiv.org/html/2304.03208v1>（arXiv:2304.03208v1）。

## 固定的 Pythia 校准规则

冻结的 B1 拟合为：

\[
f_{P}(N,D)=E+A N^{-\alpha}+B D^{-\beta}.
\]

对每个 Cerebras 模型规模 (N_i)，令 (D_i^{end}) 为论文报告的终点 token 数、(L_i^{C}) 为论文报告的 Pile test xent。先计算冻结 Pythia 曲线相对终点的改变量，再把改变量平移到 Cerebras 公开终点：

\[
L_i^{syn}(D)=L_i^C+s\{f_P(N_i,D)-f_P(N_i,D_i^{end})\}+\delta_i+\epsilon_i(D).
\]

- 中央场景固定 (s=1,\delta_i=0,\epsilon_i=0)，保证终点严格等于公开 Cerebras 锚点。
- 压力情景分别加入整体族偏移 ±0.15 loss、轨迹改变量系数 (s=0.75/1.25)，或标准差 0.02、ρ=0.70 的 AR(1) 路径噪声。噪声以终点为零，保持校准锚点。
- 每个模型生成 147 个检查点，训练进度为其公开终点的 10%–100%，按对数间距采样；这些是合成检查点，不伪造实际 step、batch、GPU 时长或 FLOPs 记录。
- 指标只在非终点检查点计算；所有情景的终点位置都从得分中排除。中央、斜率和 AR(1) 情景的终点匹配论文锚点；两个族偏移情景的终点分别保留 −0.15、+0.15 偏移，所以“终点位置”不等于“匹配公开锚点”。轨迹表和锚点审计表分别记录这两个状态。
- B1 的 `val_loss` 与论文的 Cerebras Pile test xent 是否同一评测口径，现有字段无法证明。压力得分标记为诊断值，不能包装成验证 RMSE 或独立泛化证据。

## 冻结与再现门禁

生成器 `src/f_q2_b2_semisynthetic.py` 仅在以下证据全部通过时运行：

- Q2 基线报告声明经典参考支路已冻结；
- 当前 Pythia B1 文件哈希与 B1 运行清单、Gate 0 审计、冻结快照一致；
- 冻结参数、B1 拟合预测和运行报告哈希与清单一致；
- Gate 0 状态为 `PASS`，拟合范围为 `B1 only`。

唯一命令：

```powershell
"D:\Anaconda\python.exe" src/f_q2_b2_semisynthetic.py --seed 20260924
```

生成到 `results/q2_b2_semisynthetic/v1/`。目录已存在时脚本拒绝覆盖。自动测试：

```powershell
"D:\Anaconda\python.exe" -m unittest src.tests.test_f_q2_b2_semisynthetic -v
```
