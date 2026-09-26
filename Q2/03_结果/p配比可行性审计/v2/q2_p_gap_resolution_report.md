# Q2 G1：p–Loss 缺口处理与恢复步骤

> 审计版本：v2；日期：2026-09-26。本审计完成数据恢复检查和门禁判定，不伪造 B 侧 p，也不拟合不具备识别条件的联合模型。

## 结论

**B 侧检查点实测累计组成 `p(D)` 与 Loss 的联接数仍为 0，G1 的 B 侧联合模型继续标记 `NOT_ESTIMATED`。** 同时已新增一条有来源标记的静态配方候选：B1 的 1,176/1,176 行经 `N_params_B → model_repo`、`steps → checkpoint_index.step` 唯一匹配本地 Pythia 检查点索引，并附上 Pile 官方 22 项 `Weight` 表聚合成的 18 部分候选向量。它是同一个静态 corpus-weight 候选复制到各 checkpoint 行，**不等于这些检查点实际累计看到的 token 组成，也不是经 B1 运行配置验证的 token recipe**；因此不计入实测 p–Loss 联接或独立 recipe 数，也不能估计 M1 系数。A 侧有 512 条 RegMix 配方和域级 Loss 可按共同 `index` 精确配对；它是固定实验尺度下的 p-only 证据，不等于 B 侧真实模型标度律的训练配方数据。

Pile 官方组成表可映射至项目 17 个 p 域的列权重合计 **75.02%**；未纳入这 17 个域的另 5 项共 **24.98%**。这些全 22 项权重只用于构造 `p_recipe_candidate`，并完整保留为 18 部分总和 100%；它们不是 Pythia 每个检查点实际累计训练 Token 数。因此不得把候选值当作 `p(D)` 或已验证训练配置，也不得将遗漏的五项置零或把 17 项单独重归一化。

## 已执行的恢复步骤

| 步骤 | 执行动作 | 结果 |
|---|---|---|
| 1. 冻结联接键与观测单位 | 检查 A 的 `index`、B1 的 `N_params_B + steps` 与 Pythia 检查点索引 | A 可按 `index` 对齐；B1 的 1,176 行唯一映射到八个基础模型及检查点 |
| 2. 重建配方候选 | 核对 Pythia 共同数据顺序资料和 Pile 22 项来源权重，五项余量汇入 `other` | 建立一个 18 部分 `p_recipe_candidate`；其中 17 域覆盖 75.02%，`other` 为 24.98%；候选的权重单位不等于 checkpoint token 暴露 |
| 3. 严格联接组成与 Loss | A 用共同 `index` 检查重复、未匹配；B1 候选用 model_repo+step 键，另查实测 `p(D)` 字段 | A：512 对；B1 静态候选：1,176 行；B1–B5 实测 `p(D)`：0 行 |
| 4. 检验可识别性输入 | 检查是否存在独立真实 recipe 变化，以及是否可控制 N、D、模型族 | B1 静态候选只有一个恒定向量，无配方对比；真实 p(D) 矩阵不存在，不能检验 M1 秩/混杂，也不计算 VIF 或系数 |
| 5. 决定是否拟合 | 应用 M1 输入门禁 | 不拟合 B 侧模型，状态 `NOT_ESTIMATED` |
| 6. 固化恢复合同 | 生成逐行所需字段与重开门禁条件 | 见 `q2_p_required_input_contract.csv` |

## 数据边界和目标变量

B1 当前有 1,176 行 Pythia 检查点日志、8 条规模轨迹。通过 Gate0 的精确参数量—模型仓库映射和本地 `pythia_checkpoint_index.csv` 的 model_repo+step，1,176 行全部唯一匹配。官方 Pythia 仓库确认基础模型使用 The Pile 且各规模使用同一数据顺序；因此把 Pile 22 项 `Weight` 映射成静态候选并关联到 B1 行。**该候选目前只表示 Pile 公布的组件权重；项目没有 B1 run 配置中的权重单位/分母证明，也没有每检查点按来源累计 Token 计数。** 它不叫实测 p。B2 有 1,029 行且来源清单标注为 semi-synthetic；B3 有 4000 行插值点；B4 有 57 行且缺精确运行键；B5 有 44 行且只有文献线索，未连到精确版本及完整配方。B2/B3 不计作真实独立 recipe；B4/B5 不能按 family 或引用字符串直接补 p。

本次恢复合同把目标定义为**检查点累计暴露组成** `p(D)`：按来源统计截至检查点模型实际见到的 Token 数。固定配置 `p_recipe` 另行存档，不能冒充 `p(D)`。来源 taxonomy 要完整覆盖项目 17 域，并增加显式 `other` 余项；即组成空间有 18 个部分，保留 `p_01..p_17` 与 `p_other`，以 `other` 为基准使用 17 个对数比坐标。这样能让总量闭合为 1，又不会把缺失的五项分配给任一已知域。若论文坚持原始 17 部分定义，则必须明确研究总体仅限该 17 域并逐条证明来源排除/过滤规则，否则不能把它叫完整配比。

## 重新开启 B 侧 M1 的必要条件

1. 为每个候选模型冻结 `model_family`、`model_name`、`model_version`、`recipe_id`、`run_id`、`checkpoint_id` 和 step；Loss 同时记录验证集、Tokenizer 与评测协议版本。
2. 对每个拟合检查点取得按来源标记的累计 Token 暴露计数；分母须等于 checkpoint 的累计训练 Token 数 D。另保留静态配方版本和其抽样权重作 provenance，不用它覆盖实测累计组成。
3. 把每个来源唯一映射到项目 17 域之一或 `other`；输出各项数量、份额及与 D 的核对差值，18 部分总和应为 1。来源定义、语料版本、Tokenizer、重复/重采样计数规则都要随表留档。
4. 按 `run_id + checkpoint_id` 严格联接 Loss 与 p；先发布重复、未匹配、排除数和每个排除原因。严禁靠模型族、年份、论文引用或规模近似配对。
5. 在真实独立 recipe/run 上，先冻结组成零值处理规则，再检查组成对比设计矩阵的有效秩、剩余自由度，以及与 `log N`、`log D`、模型族和数据来源的混杂；按 recipe/run 分组留出验证。B1 当前只有一个恒定的静态候选向量，八种模型规模和 1,176 个重复候选行不构成 1,176 个独立 recipe；若取得共同顺序的 `p(D)`，还需单独证明它能与 D 区分。
6. 只有上述门禁通过后，才拟合冻结 M0 与扩展 M1 并比较分组留出表现。否则保留 `NOT_ESTIMATED`，并按可识别性条件收缩论文表述。

## 当前可报告的证据

- A 侧精确 `index` 联接行数：512；配方含 17 个域，Loss 含 13 个域级目标。原始 A 配方列和范围为 0.996–1.003，有 47 行与 1 的绝对偏差大于 0.001；因此键配对完整不等于所有原始比例向量严格闭合。Q1.3 报告已并列 raw 与闭合变体，沿用其口径作受限的 A 侧证据。
- B1 静态 `p_recipe_candidate` 关联：**1,176 行、一个恒定候选向量**；只用于 provenance 候选，不作为 verified p。
- B1–B5 中通过校验的检查点累计组成 `p(D)`–Loss 联接：**0 行**。
- Pile 22 项来源表映射覆盖：**17/22 项，权重合计 75.02%**；未覆盖 **5/22 项，24.98%**。
- B 侧 p 模型：**未拟合**；不报告系数、VIF、显著性或不确定性区间。

## 复现

在项目根目录运行：`python src/q2_p_gap_resolution_audit.py`。脚本只使用 Python 标准库，输出本目录下的 JSON 摘要、A 联接覆盖、B 源门禁、B1 `p_recipe_candidate` 关联表、来源交叉表和输入合同。关键数据、Pythia 检查点索引、Gate0 映射及审计脚本的 SHA-256 列于 `q2_p_gap_resolution_summary.json`。

## 主要来源

- 项目原始数据：`中文题目/F题/real_attachments/A_data_value/regmix_tables/` 与 `B_scaling_laws/`。
- Pythia 官方仓库与论文：[仓库](https://github.com/EleutherAI/pythia)，[论文](https://arxiv.org/abs/2304.01373)。官方共同数据顺序声明不等于本地已有逐检查点来源 Token 计数。
- Pile 官方组成表：[EleutherAI/the-pile](https://github.com/EleutherAI/the-pile)。其 `Weight` 仅用于生成静态 `p_recipe_candidate` 和覆盖核对；没有据此声称检查点 token 暴露。
