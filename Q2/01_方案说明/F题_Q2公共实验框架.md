# F 题 Q2 公共实验框架

## 判定与范围

任务五可以独立推进：它交付的是跨模型共享的数据、切分、评分、Bootstrap 与结果目录接口，不需要等 Q1 的最终输出才能搭建。真实数据演练使用已有 B1 经典支路。

这不等于完整 Q2 已完成。当前 B 组训练表可运行 (M_0=f(N,D))，但没有可直接联接的领域配比 \(\boldsymbol p\)；Q1.1 的正式质量分数 \(Q_{final}\) 也尚未冻结。框架保留 (M_1=f(N,D,\boldsymbol p)) 与 (M_2=f(N,D,\boldsymbol p,Q)) 的同一调用接口，并硬性拒绝候选 Q 进入 M2。**M1 不依赖最终 Q；当前不能运行 M1 有两项原因：B 组缺少逐样本 p，且尚无 Q2 的 M1 拟合器。**框架不生成或假定 p、Q。

## 当前实现

主接口位于 `src/q2_experiment_framework.py`：

- `load_q2_data(source_specs)`：用每个来源的列映射，把 N、D、Loss、Q、领域配比和分组键整理成统一表，并记录输入 SHA-256 和 Q 状态。
- `select_complete_cases(dataset, models)`：显式构造多个模型可比较的共同完整样本集，不做隐式填补。
- `make_cv_splits(...)` / `save_cv_splits(...)`：固定种子按独立组生成 K 折，保存 row ID 清单；同一数据行只能由一折留出，同一训练轨迹不能跨训练/验证折。
- `evaluate_model(...)`：调用外部拟合器，统一计算 RMSE、MAE 和 \(R^2\)，返回可追溯预测行。
- `bootstrap_eval(...)`：对留出预测按整组重抽样，不默认逐行 IID；至少需要两个独立组。
- `run_experiment(...)`：复用已保存切分，消费端再次检查切分类型与每折训练/测试组互斥，并生成 `config.json`、`metrics.json`、`predictions.csv`、`bootstrap.csv` 和 `plots/` 目录。
- `model_feature_columns(...)`：`M0` 使用 N、D；`M1` 增加所有 p 分量；`M2` 再增加 Q。M2 只接受每个来源明确标记为 `final` 且提供冻结版本号的数据。

共同切分绑定到具体的 row ID 集合与分组键。只有输入行集合完全相同的模型比较才能复用同一个 `q2_cv_splits.json`；数据来源或完整样本集不同，就另存一份 split manifest，不能把不对应的行 ID 硬套到另一批数据。

## B1 真实输入演练

`src/q2_framework_b1_demo.py` 将既有经典 Scaling Law 作为 M0 拟合器，在 B1 的 8 条经 Gate 0 确认的 Pythia 轨迹上做 5 折整轨迹留出。它会把每折留出的预测拼成 OOF 表，再以轨迹为 cluster 对 OOF 指标做 2,000 次 Bootstrap。该演练验证公共框架贯通真实 B1；不取代 `results/q2_scaling_baseline/v1/` 的全量经典基线结果，也不能证明 p 或 Q 的效果。

从项目根目录唯一复现命令：

```powershell
D:\Anaconda\python.exe src/q2_framework_b1_demo.py --splits 5 --bootstrap-resamples 2000
```

结果写入 `results/q2_public_framework/`：

```text
q2_cv_splits.json
复现清单.json
B1/
  config.json
  metrics.json
  predictions.csv
  bootstrap.csv
  plots/
```

## B2–B5 验证结论

现有留出评分已经按来源完成，不能把这些来源等同看待：B2 是 Cerebras 半合成轨迹，B3 是 Pythia 插值轨迹，两者可用于迁移/插值压力检查，不是独立真实外部验证；B4/B5 虽有跨族或文献观测，但当前没有统一 tokenizer、评测集和 Loss 协议证据，因此只分源报告，不计算合并分数。2026-09-24 的逐来源审计见 `Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md`、`loss_protocol_matrix.csv` 和 `loss_protocol_evidence_sources.md`：B4/B5 对 B1 均判为 `NOT_COMPARABLE`，B5 只保留补齐原文定位后的来源内趋势候选；B4 七条 Cerebras-GPT 记录和 B5 GPT-3 来源标签还存在明确的来源/数据核对事项。只有新增并核验逐行来源、原始评测协议、Loss 定义和训练状态后，才有依据升级可比性结论；当前不需要重跑现有验证来弥补元数据缺口。

## 后续模型接入

有逐样本 p 时，即使 Q 尚未冻结，也可用 `Q2Dataset` 和共同 splits 调用 M1；以下示例中的 `fit_m1` 需先按 Q2 主方案实现：

```python
common = select_complete_cases(dataset, ["M0", "M1"])
splits = make_cv_splits(common, n_splits=5, seed=20260924, group_col="group")
save_cv_splits(splits, "results/q2_public_framework/q2_cv_splits.json")
run_experiment(common, experiment="p_only", model="M1", fit_predict=fit_m1,
               splits=splits, output_dir="results/q2_public_framework/p_only")
```

有逐样本 p 后，即可在不加载 Q 的情况下，对同一共同完整样本集比较 M0 与 M1，并按训练轨迹分组生成、复用相同的 splits。待最终 Q 冻结后，再加载 `q_status="final"`、`q_version="<冻结版本标识>"` 的来源，构造 M0/M1/M2 的共同完整样本集并让三种模型复用一份 splits。候选 Q 可用于接口调试的非最终本地副本，但 M2 正式调用会被拦截；不能把 `candidate` 或版本号缺失的 Q 写入最终结果。

## 当前 p-only 运行门禁（审计完成，2026-09-24）

配比可行性审计已完成，结果见 [`q2_p_feasibility_audit.md`](../03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md) 及本目录下同版本审计文件。**当前真实 B1–B5 的 p-only 数据 Gate 为 `FAIL`：已核验并与 Loss 逐行/逐轨迹联接的数值 p 向量为 0。** B1 的八条 Pythia 规模轨迹使用共同数据顺序政策，不能算成八个独立配方；B2 为半合成，B3 为插值；B4 缺少逐行 source/运行键，B5 的论文引用仍未与具体版本、训练配方和 Loss 行建立连接。官方资料和部分论文提供了有条件的来源重建线索，但还没有项目 17 域 p 向量或 checkpoint 级 (p(D))。审计未拟合模型，也未在 p 缺失时计算 VIF/相关性。

Q1.3 的 p-only 结果来自 A4/A5 配比—Loss 数据，尚无经审计的键把这些配方映射到 B1 的具体模型/训练轨迹/checkpoint；不能按参数规模或行序拼接，也不能直接把 Q1.3 拟合器当成 Q2 的 M1。公共框架仍只有 B1/M0 适配器，没有 Q2 的 `fit_m1`。B4/B5 Loss 绝对可比性是外部验证门禁，不是 M1 的先决条件；但这不改变本次 p Gate 失败的结论。只有可信 p 数据链、可识别的配比变化及 `fit_m1` 均到位后，M1 才能启动。

启动 M1 前至少需要：

1. 每条训练 Loss 记录对应的实际训练配方 p，使用题目与 Q1 冻结接口规定的域列；
2. 可追溯且唯一的训练运行/模型/检查点键，以及经核验的 p 到 B 行映射和覆盖率；
3. p 的非负性、有限性、组成约束及单位/域顺序审计；
4. 检查 p 在独立训练轨迹/模型族间是否有足够变化，且不是 N、D 的完全替身；若 p 恒定或与规模完全混杂，B1 不支持识别 p 效应；
5. 按 Q2 主方案冻结并实现 M1 拟合器，包括 ILR 零分量处理与正则化选择规则；不得用 Q1.3 的目标结构代替；
6. M0/M1 共用的完整样本集和整轨迹分组切分，模型选择只看训练折，验证轨迹不回流。

这些输入到位后，M1 可先于 Q_final 运行；只有涉及 Q 效应、p×Q 或质量—规模替代分析的 M2 才必须等待冻结的 Q 及版本号。若仍没有可信的逐样本 p，正确结论是“Q2 p-only 数据受限”，而不是用 Q1.3 的聚合结果代替训练配方。

## 当前边界

- P1 仅覆盖 B1/M0 的共享切分、预测、指标、组级 Bootstrap 和文件写出。
- `p_only` 与 `p_plus_Q` 目录及接口已预留，暂不运行；当前既缺可追溯的逐样本 p 联接，也缺 Q2 的 `fit_m1`。M1 不需要冻结 Q；M2 才需要 `Q_final` 和冻结版本号。
- `plots/` 由公共结构预建；正式 Q2 模型图应在模型、样本和验证合同冻结后按图表规范生成，不以空目录代替出图门禁。
- B1 的 8 个独立轨迹较少，Bootstrap 区间只作稳定性提示。B1 五折结果是轨迹留出预测，不能与既有全样本拟合指标混为一谈。
