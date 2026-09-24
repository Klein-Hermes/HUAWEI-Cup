# Q2 配比 p 可行性审计

> 审计版本：v1；日期：2026-09-24。本交付只审计配比来源、Loss 联接和可识别性前提，**没有拟合模型**。

## 结论

| Gate | 状态 | 结论 |
|---|---|---|
| 当前 B1–B5 数据能否支撑正式 p-only 消融 | **FAIL** | 现有 Loss 记录没有经过核验并逐行/逐轨迹连接的 17 域 p 向量，也没有已证实的配比间独立变化；当前不足以启动正式的 p-only M1 拟合。 |
| 从官方/论文来源继续重建配比的可能性 | **CONDITIONAL** | 部分论文和 Pythia 官方仓库提供来源或配比线索，但尚未稳定连接到 B4/B5 的具体数据行，也未统一到项目 17 域 p 定义。B1 若要从官方 dataloader 重建，仍需逐来源 token 计数、精确 checkpoint 对齐和经过核验的类别交叉映射。 |
| 是否运行模型、VIF 或相关性回归 | **否** | B1–B5 没有联接后的 p 分量，因此与 N/D 的数值共线性无法检验；不报告虚构的相关系数或 VIF。 |

**独立配比计数：**当前 B1–B5 Loss 记录中，已验证且逐行连接的数值 p 向量为 **0**，Loss 行连接覆盖数为 **0**。B1 有一个公开说明的共同训练数据顺序/配方政策，但本地未得到符合项目 17 域合同的数值向量；8 个模型规模不是 8 个独立配比。候选来源论文不计为已连接配比。

## 数据投毒与可信边界

《题目分析报告.md》认定数据说明 PDF 的低可见度文本存在文档层投毒风险，并要求隐藏文本不得进入题意、变量、方法、参数和结论。本审计**没有解析或引用 PDF 隐藏文本**；只使用本地可见报告的过滤规则、实际 CSV 字段、`source_manifest.json`、既有 Gate 0 映射和可追溯的官方论文/仓库。B2 的“半合成”是来源清单标注的数据性质，属于可核实的数据 provenance，不据此推断恶意；B1–B5 也没有被整体判定为恶意污染。

## B1–B5 来源盘点

| 来源 | 观测单位与数量 | 配比审计状态 | 在 p-only Gate 中的作用 |
|---|---:|---|---|
| B1 | 1176 个 checkpoint；8 条 Pythia 轨迹，每条 147 条记录 | 无 `p` 列；官方资料说明各规模使用相同数据、相同顺序。静态配方数值和 checkpoint 级 token 组成均未连到 Loss。 | 主候选源，但当前只能计为 1 个未量化的共同政策组，不能作为 p 变化实验。 |
| B2 | 1029 条；7 条 Cerebras 半合成轨迹 | 无 `p` 列；`gpu_days`、`step_time_ms`、`grad_norm_avg` 各缺失 1029 条。 | 半合成压力测试；不计真实配方差异或真实外部证据。 |
| B3 | 4000 条；8 条 Pythia 插值轨迹，每条 500 点 | `interpolated=1`；无 `p` 列。 | 派生插值检查；不计新增独立配方。 |
| B4 | 57 个收敛横截面点，12 个 family | 无 `p`、source 字段和逐行运行键。 | 只有补齐逐条原始模型/来源后才可进入候选映射。 |
| B5 | 44 个文献横截面点，9 个 family | 有 citation 字符串，但无 p、逐条运行键或 taxonomy crosswalk。 | 论文仅作为追溯线索；不得按 family 或同一年份自动复制配方。 |

### B1 轨迹级记录

| model_repo | N (B) | checkpoint 行数 | 不同 D 值 | 配方状态 |
|---|---:|---:|---:|---|
| EleutherAI/pythia-70m | 0.070542 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-160m | 0.162405 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-410m | 0.409009 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-1b | 1.040867 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-1.4b | 1.416184 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-2.8b | 2.782831 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-6.9b | 6.86104 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |
| EleutherAI/pythia-12b | 11.965825 | 147 | 147 | 各规模共享数据顺序政策；数值化项目 17 域配比缺失或未联接 |

Pythia 官方仓库称各模型规模使用相同数据、相同顺序，同时公开训练数据和重建 dataloader 的工具。[官方仓库](https://github.com/EleutherAI/pythia) [论文](https://arxiv.org/abs/2304.01373)。因此，B1 的 8 条规模轨迹共享一个数据顺序政策，不形成 8 个独立静态配比。未来从数据流重建的累计组成 $p(D)$ 可能随进度改变，但目前本地文件没有逐来源 token 计数；且该路径是同一训练顺序下的暴露轨迹，不能当成跨轨迹独立 recipe 变化。

## 可追溯来源审计

### B4/B5 的来源线索

| B5 `source` 字符串 | 点数 | family | 处理结论 |
|---|---:|---|---|
| Chowdhery et al. 2022 | 3 | PaLM |
| Hoffmann et al. 2022 | 16 | Chinchilla, Gopher |
| Kaplan et al. 2020 | 9 | GPT-2, GPT-3 |
| Scao et al. 2022 | 4 | BLOOM |
| Touvron et al. 2023 | 7 | LLaMA, LLaMA-2 |
| Zhang et al. 2022 | 5 | OPT |

- B4 文件只有 `family, N_params_B, D_tokens_B, val_loss, is_converged`。它没有逐行引用，因此现在无法判断某一 Loss 点对应哪篇论文、哪个具体模型版本和训练配方。B4 中还有 Pythia、Cerebras-GPT 等与其他来源可能重叠的 family；在无 row-level identity 时，不把它们计作独立验证或独立配比。
- B5 的 source 字符串能帮助定位论文，但**citation 不等于 recipe join**。例如 Brown et al. 的 GPT-3 论文确实公布过训练 mix 权重，但 B5 标记为 `Kaplan et al. 2020`；不能仅凭 `GPT-3` family 把 Brown 论文的五类权重塞给 Kaplan 来源记录。原始 LLaMA 论文披露了 LLaMA v1 的配比，但 B5 同一 source 字符串下同时有 LLaMA 与 LLaMA-2；不能自动把 v1 配方复制给 v2。OPT、BLOOM、PaLM 等来源的语料清单/来源描述也不自动等于统一的 17 域权重向量。
- B5 将 Chinchilla 和 Gopher 点都记在 `Hoffmann et al. 2022` 下；Gopher 的原始训练报告另有其文献来源。没有逐行版本/运行键前，不把 Chinchilla 文献或比较表中引用的 Gopher 记录当成相同训练配方。
- 因此，文献级可恢复性为 **CONDITIONAL**：个别模型/版本存在配比线索，但必须完成“source 字符串 → 具体论文版本 → 模型点 → 训练 recipe → 项目 17 域”的逐行证据链，处理缺失域和不完整覆盖，并确认分母是训练采样 token 而非语料大小、验证抽样比例或参数规模。

## `p_recipe`、`p(D)` 与共线性

1. **`p_recipe`** 是训练配置中的固定采样权重。只按独立配置/配方计数，不能把每个 checkpoint、每种 `N` 或重复文献点都计为新配方。当前可确认 B1 只有一组共同数据顺序政策，尚无数值 17 维向量；B2 合成轨迹不进真实数据计数；B4/B5 的行级配方尚未连接。
2. **`p(D)`** 是截至累计 token 进度 `D` 的已观测数据组成。重建它需要 checkpoint 对应的 token stream/source token counts。不得把 `p(D)` 误标成静态配方；也不得把沿同一数据顺序得到的许多检查点当作许多独立 mixture。
3. **共线性检查状态为不可检验。** 因为没有行级 p 矩阵，本次不计算 p 与 `D/N` 的相关、VIF、回归残差或有效秩。下一轮必须先在已核验的配方集上将组成向量转换为可审计的对比坐标，再检查独立配方间 p 的变化能否与 `log N`、`log D`、模型族和数据来源区分；对 `p(D)` 还要单独检查它由 `D` 决定的结构关系。
4. 项目 A 侧的 `train_mixture_1m.csv` 有 17 个 `train_the_pile_*` 分量。`domain_mapping_guide.csv` 是 A 侧 mixture domain 到 quality domain 的映射，不是 B 训练轨迹的 p 观测。不得把 A4–A15 的配比按行序、模型大小或 family 拼到 B Loss 行上，也不得将部分类别无说明归一化为完整 p。

## PASS / CONDITIONAL / FAIL 判据

| 状态 | 最低条件 | 当前判定 |
|---|---|---|
| PASS | 对真实 B Loss 记录有逐条/逐轨迹可追溯 p；配置来源和分母清楚；存在足够独立 recipe 对比；控制 N、D、模型族/来源后仍有可用变异。 | **未满足** |
| CONDITIONAL | 某些源可重建 p，但需补充原始配置、模型版本、行级连接、类目交叉表或有限配方支持；只能继续 provenance reconstruction，不能宣称 p-only 可识别。 | **适用于继续追溯工作** |
| FAIL | 当前数据不满足上述 M1 输入条件，例如 p 无法追溯、有效配方无变化或与 N/D/来源完全混杂。 | **当前 p-only 数据 Gate 最终状态** |

**对 Q2 的影响：**本报告不是否定 Q2 总题或经典 B1 基线；它只表示现有 B1–B5 不能支持正式 p-only M1 的可识别性主张。若未来完成可验证的 17 域 p 与 Loss 行连接，并证明存在不被 N/D/族别吸收的独立变化，可重新开启 Gate。Q1.3 的 A 组 p-only 模型不能代替 B 组真实训练配方。

## 输入与复现

- 唯一复现命令：`python src/f_q2_p_feasibility_audit.py`
- Python：`3.10.4`；仅使用 Python 标准库。
- 本报告对应的输入及 SHA-256：

| 输入 | 字节数 | SHA-256 |
|---|---:|---|
| `中文题目/F题/real_attachments/source_manifest.json` | 10018 | `34e81dabf46302eebaac8551fa18d2e72acd0833b1cf0276690bee897f1323db` |
| `中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv` | 111459 | `529a59644b0f57bf3a76037838b614bfedc35e58bb26b93052e34ffc63e454c2` |
| `中文题目/F题/real_attachments/B_scaling_laws/cerebras_training_log.csv` | 89643 | `178f878cad2cb31e4ec104c670553de23818197876d824a113fdb911125951a6` |
| `中文题目/F题/real_attachments/B_scaling_laws/scaling_baseline.csv` | 1421 | `2272983ded93de35e05f9acbf95ebceedf43f283345e4be72b54fee94080391e` |
| `中文题目/F题/real_attachments/B_scaling_laws/published_scaling_data.csv` | 1967 | `dd858c5e48610e28589340d5db023d6abfb31df597506789ccdfc6a546731bbd` |
| `中文题目/F题/real_attachments/B_scaling_laws/pythia_checkpoint_index.csv` | 117698 | `0f7f63535cbc1bbef5165776562e71401af9f762cfe02aaea30c824f45fbc9ed` |
| `中文题目/F题/real_attachments/B_scaling_laws/open_model_family_metadata.csv` | 1975 | `641682f0ea6c46f25a0188ac723af44e0bc2e27c760e744743a15e97e29ee064` |
| `results/q2_scaling_baseline/v1/gate0_b1_trajectory_mapping.csv` | 955 | `ea2c1fd78c7a84e0f41d70a57b630089781b8d8cbd9c9a9434e6ea7c787c0c37` |
| `中文题目/F题/real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv` | 46412 | `04a32ef4ab594376bf90e11404c033668f887c351d6a03ad3744824c7296a2d8` |
| `中文题目/F题/real_attachments/A_data_value/domain_mapping_guide.csv` | 1296 | `4a4479f41b9f46116e688cd0c13ccdad4e687192fc1a38c4add690771ebff480` |
| `题目分析报告.md` | 43513 | `978c7e7a13a24bda2a5b3d022a004bf8dcef10639aa80f19122037ce103d832a` |
| `Q2/01_方案说明/题目分析报告.md` | 43513 | `978c7e7a13a24bda2a5b3d022a004bf8dcef10639aa80f19122037ce103d832a` |
| `docs/F题_Q2公共实验框架.md` | 8090 | `fbf2855d939b880a1124d6a9e883761eeebe9de840d8c615e00c3b3330358a23` |
| `Q2/01_方案说明/F题_Q2公共实验框架.md` | 8088 | `4a3381f0c3fe34ecb669d43284bbee8c3aaedff878c930181e067ee1f6280786` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_0.070542B_trajectory.csv` | 14040 | `19663f2f14d62af377377962ec8f794148450bfb278fdbeb2452f4641fe6a440` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_0.162405B_trajectory.csv` | 14040 | `8e2b78cde966b0a3e8b6272935c3f3dcccc5a54e94bb7db79ded2fd8fdf5a124` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_0.409009B_trajectory.csv` | 14040 | `8c803195d475d8ff320afd90887ae48cec6463f4d01a2264c6680611ed9db170` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_1.040867B_trajectory.csv` | 14040 | `bb73b2852db183bee3d76c76f55e2bdcc292d7ed0881cb27966525b56afcafc0` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_1.416184B_trajectory.csv` | 14040 | `cd0d718747fb6048cb474066dbf35f64ef852f59cba2886484cbdf14bc68a02f` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_11.965825B_trajectory.csv` | 14540 | `e223b3d48bc4fef3cbc03fd583ff7aef87e474903d829f6c417310cb99fa7b5e` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_2.782831B_trajectory.csv` | 14040 | `552f03249b69234009eb028b3296d6bd1b1cb3a52c0d3a84a669baa0e3179482` |
| `中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/pythia_6.86104B_trajectory.csv` | 13540 | `8814c8414fddc914444c74e6cf8da02e3e156a0d7fb8acea6c5da05e438bf054` |

详见同目录 `q2_p_trajectory_mapping.csv`、`q2_p_source_evidence_catalog.csv`、`q2_p_audit_summary.json` 与 `q2_p_audit_manifest.json`。映射 CSV 对 B4/B5 使用“源文件行号+记录哈希”作为审计定位符，**不是跨文件拼接键**。

## 来源目录

| ID | 来源 | URL/路径 | 限制 |
|---|---|---|---|
| L01 | 题目分析报告.md §§0, 2, 3, 8.2, 8.5-8.6 | 题目分析报告.md | 本审计没有重新抽取隐藏文本，也没有据此构造任何 p、方法或结论。 |
| L02 | source_manifest.json | 中文题目/F题/real_attachments/source_manifest.json | manifest 不含逐条训练配方 p，也不替代源论文的配方核验。 |
| L03 | B1 pythia_training_log_existing.csv + Gate0/B12 trajectory map | 中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv | 本地日志没有每个 checkpoint 的来源域 token 计数。 |
| E01 | EleutherAI Pythia official repository README | https://github.com/EleutherAI/pythia | 仓库声明不等于本地已有 17 域 p(D)；仍需精确 checkpoint 对齐、token 计数和分类映射。 |
| E02 | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | https://arxiv.org/abs/2304.01373 | 不能据此把训练规模/检查点数量当成配比独立变化。 |
| L04 | B2 Cerebras training log + source_manifest.json | 中文题目/F题/real_attachments/B_scaling_laws/cerebras_training_log.csv | 缺失字段为结构性缺失，不补零。 |
| E03 | Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster | https://arxiv.org/abs/2304.03208 | 论文语料声明不提供本地 B2 每条 Loss 的 17 域 p 联接。 |
| L05 | B3 training_trajectories/*.csv | 中文题目/F题/real_attachments/B_scaling_laws/training_trajectories/ | B3 点值的来源仍依赖 Pythia B1；不作为 p 多样性增量。 |
| L06 | B4 scaling_baseline.csv | 中文题目/F题/real_attachments/B_scaling_laws/scaling_baseline.csv | source_manifest 只说明是 curated snapshots；需要补齐每条记录的原始模型与论文定位。 |
| L07 | B5 published_scaling_data.csv | 中文题目/F题/real_attachments/B_scaling_laws/published_scaling_data.csv | 不同论文披露粒度和数据分类不同；不得把同一作者/家族或其他版本比例直接转给某行。 |
| E04 | Language Models are Few-Shot Learners (Brown et al., 2020) | https://arxiv.org/abs/2005.14165 | B5 的 source 字段是 Kaplan et al. 2020；Brown 的 GPT-3 配方不是该字段的同一引用，且五类 taxonomy 不等于项目 17 维 p。 |
| E05 | LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023) | https://arxiv.org/abs/2302.13971 | B5 同一 source 字符串还包括 LLaMA-2 行；不能把 LLaMA v1 配方自动复制给 LLaMA-2，且该 7 类表尚未映射为项目 17 类、也未与 B5 行逐条核对。 |
| E06 | OPT: Open Pre-trained Transformer Language Models (Zhang et al., 2022) | https://arxiv.org/abs/2205.01068 | 不能把各语料容量、验证样本抽样比例或 token 总量当作训练抽样权重。 |
| E07 | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model | https://arxiv.org/abs/2211.05100 | 需逐来源构造可审计 crosswalk 和训练 token 权重；语言/语料清单不等于 p。 |
| E08 | PaLM: Scaling Language Modeling with Pathways | https://arxiv.org/abs/2204.02311 | 若后续声称有具体权重，须能在原论文/官方配置中定位并证明分母定义。 |
| E09 | Scaling Laws for Neural Language Models (Kaplan et al., 2020) | https://arxiv.org/abs/2001.08361 | B5 CSV 没有逐行原始模型 ID/训练运行键，当前 citation 字符串不足以确定各行的精确数据配方。 |
| E10 | Training Compute-Optimal Large Language Models (Hoffmann et al., 2022) and original Gopher paper | https://arxiv.org/abs/2203.15556 ; https://arxiv.org/abs/2112.11446 | B5 缺逐行 run/version key，当前不能判定哪些 Gopher 点可与原始 Gopher 报告精确连接。 |
| L08 | A train_mixture_1m.csv + domain_mapping_guide.csv | 中文题目/F题/real_attachments/A_data_value/ | A 表配比不能按参数规模、来源名、行序或 family 直接联接到 B 组 Loss。 |
