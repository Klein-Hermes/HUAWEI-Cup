# Q3 本轮结果包 rev2 复现清单

## 复现顺序

1. 在仓库根目录确认 Q1/Q2 上游源文件存在且哈希与各任务 manifest 相符。
2. 按需重跑函数形式与整轨迹留一分析：`D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_trajectory_form_sensitivity.py`。完整环境、输入/输出哈希与 8 折结果见 `Q3/04_结果/轨迹留一与函数形式敏感性/reproduction_manifest.json`。本分析复用 Q2 当前 M0 权威留一拟合；只重新拟合两种受限形式。
3. 按需重跑其余任务原始命令（T1/M3/M4、T2、T3、T4、预算覆盖/有限网格切换）；完整命令和环境见相应目录内的任务级 `reproduction_manifest.json`。本次冻结包组装没有重采 Bootstrap 或重算 T2/T3/T4。
4. 重新组装汇总表和新版本快照：
   `python Q3/03_代码/assemble_q3_scoped_release.py --release-dir Q3/冻结包_20260925_模型形式敏感性_独立复核`；已有冻结目录不覆盖。
5. 完成独立 P2 复核后，将回执写入该快照的 `QA/P2_independent_release_receipt.md`，并运行 `python Q3/03_代码/assemble_q3_scoped_release.py --release-dir Q3/冻结包_20260925_模型形式敏感性_独立复核 --refresh-release-manifest` 更新最终包哈希。
6. 核对本文件列出的源任务 manifest、包内表格/图件与 `release_manifest.json` 哈希。
7. 总表组装器 `Q3/03_代码/assemble_q3_scoped_release.py` SHA-256：`16cae58d3aab1309bd6ce0a5d1e3e25e9965faed21538fe6cbf12978d69b5a5f`。

## 本次任务级 manifest 校验

| Manifest | 已核验文件 | 缺失 | 哈希不符 | 状态 |
|---|---:|---:|---:|---|
| `Q3/04_结果/Q3_fixedQ_results_manifest.json` | 23 | 0 | 0 | PASS |
| `Q3/04_结果/M3_M4成本独立审计/reproduction_manifest.json` | 28 | 0 | 0 | PASS |
| `Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/reproduction_manifest.json` | 18 | 0 | 0 | PASS |
| `Q3/04_结果/Q3_T3_稳健性与配置切换/reproduction_manifest.json` | 27 | 0 | 0 | PASS |
| `Q3/04_结果/q3_t4_conditional_beta_manifest.json` | 7 | 0 | 0 | PASS |
| `Q3/04_结果/预算饱和与有限网格切换/reproduction_manifest.json` | 13 | 0 | 0 | PASS |
| `Q3/04_结果/轨迹留一与函数形式敏感性/reproduction_manifest.json` | 16 | 0 | 0 | PASS |

## 环境与随机性

- T2/T3/T4 任务级 manifest 记录 Python 3.10.20、项目 `mmdet` 环境和所用 NumPy/Matplotlib/Pandas/Pillow 版本；具体以各 manifest 为准。
- T2 为确定性有限网格枚举；其任务 manifest 的 `random_seed` 为 `null`，并注明复用冻结的 1,000 行 Q2 轨迹 Bootstrap 参数表，不新增抽样。边际收益区间复用既有配对 Bootstrap 估计；T3 也复用既有轨迹 Bootstrap 与配对抽样，总表合并过程不重采样。
- T1、M3/M4、T4 和轨迹留一/模型形式敏感性分析的重跑命令、输入输出哈希均保留在各自任务级 manifest。
- 汇总表由当前源 CSV 的预算×上下文键连接生成，不使用正文摘录的数字作为输入。输出形状：15 行 × {table_cols} 列。
- 冻结快照需要仓库内 Q1/Q2 上游输入才能从原始任务重跑；本包提供的复制品足以审阅与做包内一致性核对。

## 自身哈希说明

本文件和 `figure_manifest.csv`、`table_manifest.csv`、`release_manifest.json` 的文件哈希不在自身内容中自引用；`release_manifest.json` 记录其余所有冻结文件的哈希并自排除。
