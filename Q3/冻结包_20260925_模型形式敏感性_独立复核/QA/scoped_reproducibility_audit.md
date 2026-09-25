# Q3 本轮结果包限定范围可复现性核对

**状态：** PASS（限已有结果快照及复现信息；从 Q1/Q2 原始源头重跑仍依赖项目上游文件）

- 源任务 manifest 哈希核对：132 项通过，缺失 0 项，哈希不符 0 项。
- 复现命令、Python 环境、随机性/Bootstrap 口径见 `reproduction_manifest.md` 与任务级 `reproduction_manifest.json`。
- 新增函数形式敏感性 manifest 声明的输入、代码和 7 项主要输出由上项哈希核对；P1 回执与独立 P2 回执分别保留在对应 QA 目录。
- 图件 SHA-256、格式、PNG 像素尺寸/DPI、矢量画布尺寸见 `figure_manifest.csv`；数据表行数/列数/哈希见 `table_manifest.csv`。
- `release_manifest.json` 记录冻结包内每个文件的路径、SHA-256 和字节数，自身按约定排除以避免递归哈希。
- T2/T3/T4 复用已有生成结果；新增预算分析是固定 M0 的确定性有限网格枚举，不重拟合模型、不做连续阈值推断。

如需从任务原始输入重跑，按总复现清单中的命令执行，并使用任务级 manifest 中的上游 Q1/Q2 路径与哈希。
