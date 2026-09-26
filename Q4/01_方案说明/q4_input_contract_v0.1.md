# Q4 输入合同 v0.1

> AI 辅助说明：本合同由 OpenAI Codex 桌面版 26.917.71314（build 10954；prod；2026-09-25 更新器核验为 up_to_date）辅助起草，开发机构 OpenAI。模型选择器显示名称及模型版本/发布日期按用户指示暂空。它是可审阅草案；用户已确认采用 `q4_cohort_decision.md` 中的暂定样本和指标口径，但不得原样作为竞赛核心分析结论。AI 输入、输出处理与核验记录见 `../05_审计交付/AI工具使用记录.md`。

## 1. 观测单位和主键

### C1–C8 模型观测

| 概念 | 合同字段 | 定义与规则 |
|---|---|---|
| 来源记录 | `source_record_id` | 来源文件相对路径 + 原始行号/JSON 路径；同名行不合并 |
| 模型名 | `model_name_raw` | 保留来源原文；规范化副本仅用于候选匹配 |
| 模型版本 | `model_version_id` | 只有来源提供 revision/SHA 或可核实的具体版本时填写；否则为缺失并标 `version_unresolved`，不从名字猜版本 |
| 模型族 | `family_id` | 来自明确的 `Base model`、来源元数据或队伍确认的映射；无证据则缺失，不自动按字符串分族 |
| 评测运行 | `run_id` | C8 模型目录 + 文件名时间戳 + 文件 SHA-256；区分同一模型的多次评测 |
| 逐任务观测 | `observation_id` | `run_id + task_key + task_version + metric_key`（保留过滤器后缀，避免将来多个过滤器时主键冲突） |

C1 中 79 个模型名各有两条且内容不同，故 C1 行是观测单位；`Model` 不是足够的去重主键。C1 与 C2 顺序不同；核对十二个共同基础字段时要求文本字段精确一致，数值字段绝对差不超过 `5e-13`。4,576 行全部形成唯一的一对一整行匹配，最大数值差为 `7.11e-15`。仅按该完整行规则对齐增强字段；不得按行号或只按模型名合并。C1 的分数保持为主值，C2 仅补充元数据候选。自动生成的哈希行 ID 仅用于追溯，不冒充外部版本 ID。

## 2. 时间字段

| 字段 | 来源 | 含义 | 主分析使用条件 |
|---|---|---|---|
| `publication_date` | C4 Epoch `Publication date`；C2 中的 Epoch 日期只作候选 | 公布/发表日期，可能并非权重首次可用日 | 需确认对应模型版本，保留来源和匹配状态 |
| `submission_date` | C1/C2 `Submission Date` | Leaderboard 记录日期 | 可作为排行榜进入时间的候选；不称为模型发布日 |
| `eval_file_date` | C8 文件名 | 该 JSON 文件时间戳 | 用于排序运行和回退审计 |
| `eval_timestamp` | C8 JSON `date` | 评测器记录时间 | 转换为 UTC 和北京时间两列；原始 Unix 值保留 |
| `year` | C3 `Year` | 时序表年份 | 必须连同 C3 `Source` 使用；历史来源不与 C8 主评测直接拼接 |

时间预测的原点由目标样本中最后一个有效且可比的观测决定，不预设为竞赛当前日期。C8 文件时间的已核实范围是 2024-06-16 至 2025-03-14；C1 `Submission Date` 是 2024-06-08 至 2025-03-13。

## 3. 数值、单位与方向

| 字段 | 统一单位/方向 | 缺失处理 |
|---|---|---|
| 参数量 `parameter_count_B` | 十亿参数（B）；优先取 C1 `#Params (B)` | 原值缺失、0 值单独标记；不从模型名解析补值 |
| 训练算力 `training_compute_FLOP` | FLOP；C4 `Training compute (FLOP)` | 仅精确/人工确认版本匹配后使用；上下界另列，不取中点冒充点值 |
| 微调算力 `finetune_compute_FLOP` | FLOP；C4 `Finetune compute (FLOP)` | 与预训练算力分列；未知仍为空 |
| C1 六维分数 | 保留 C1 原值与百分数口径 | 不跨列填补 |
| C8 分数 | 保留原分数及原指标名；另存方向和标准误 | `N/A`、缺项、解析错或无效任务均为缺失状态，不是 0 |
| C8 样本量 | `n_original`、`n_effective` 个样本 | 原始/有效样本量分别记录 |

不同任务、指标定义或任务版本的分数不直接求均值。解析键保留 C8 原始 `metric_key`（包括 `,none` 等后缀），并另存过滤器 `metric_filter` 与 `metric_name`。先从原始键拆过滤器，再识别指标名结尾的 `_stderr`；方向从 `higher_is_better[task_key][metric_name]` 读取。标准误是基础指标的配对不确定性，不是分数；无基础分数时单独标为未配对。空白指标名仍留在原始长表，但标记为非指标元数据，不进入任务覆盖率或综合分。缺少方向时该分数不能进入统一能力指标。

## 4. 来源登记与字段优先级

| 题面编号 | 项目内来源 | 在 Q4 的用途 | 优先级/限制 |
|---|---|---|---|
| C1 | `leaderboard_cleaned.csv` | 主六维排行榜及模型类型、参数量、提交日、Hub License | 主排行榜；79 组重复模型名需逐行保留 |
| C2 | `leaderboard_enhanced.csv` | Epoch 发布信息补充 | 增强列来自匹配；当前标为候选，未经核对不用于严格版本匹配 |
| C3 | `leaderboard_extended_timeseries.csv` | Leaderboard 年份时序 | `Source=Historical (papers/reports)` 的 26 行隔离为敏感性/背景数据 |
| C4 | `epoch_all_ai_models.csv` | 发布日、组织、参数/算力、开放权重等元数据 | 与 C1 的 `Model` 精确名称交集为 0；只有经别名/版本人工核实后才能合并。否则 FLOPs、发布日和开放权重均记未知；notes 是来源说明，不解析为数值 |
| C5/C6 | `loss_benchmark_bridge.csv` / `loss_benchmark_bridge_expanded.csv` | Loss–Benchmark 桥接 | 按 `Loss_Comparability` 分层；C6 75 行中仅 7 行含可直接比较的训练数据量 D |
| C7 | `model_architecture_metadata.csv` | 架构与上下文元数据 | 名称不必与排行榜一致，匹配状态单列 |
| C8 | `detailed_results/*/results_*.json` | 必用逐任务指标 | 先解析、保留运行，再按版本/任务/指标聚合 |

使用项目中的 `数据说明(无隐藏字段版本）.pdf` 作字段字典；原始附件只读。PDF 可见字段、`source_manifest.json`、文件实际字段和脚本运行结果冲突时，先记录差异，不静默选择。

## 5. 版本匹配状态

匹配表必须保留原始名、候选规范名、来源、匹配依据和行数：

- `exact_model_id_unique_record`：名称精确且 C1 中只有一条记录。这仅是名称/行唯一候选，不构成版本证据；
- `exact_name_multiple_records`：名称精确但有多个记录/版本，不能进严格版本样本；
- `alias_candidate_unconfirmed`：规范化或别名候选，待人工确认；
- `manual_version_confirmed`：队伍记录确认依据与确认人/日期后才能进入严格样本；
- `semantic_candidate`：按组织、系列或架构推断的候选，仅用于检查；
- `unmatched`、`missing_source`、`parse_failed`：保持明确缺失状态。

另设 `version_match_status`，取 `revision_exact`、`manual_version_confirmed`、`source_version_missing` 或 `revision_conflict`。只有 `revision_exact`（两侧 revision/SHA 可比较且一致）或已记录证据的 `manual_version_confirmed` 才属于严格版本确认。没有精确版本证据时，不以“相似名称”替代版本匹配。C8 的 `config.model_sha`/`config.model_revision`、C1 `Model`、C1 重复行和 C4 名称字段分别保留以支持审计。当前 C1 不提供 revision/SHA，故仅凭现有 C1 不能自动达到 `revision_exact`。

## 6. 类型、权重与样本冻结门

模型类型先保存 C1 原始 `Type`，派生类型只做显式映射：`pretrained`、`continuously_pretrained`、`domain_finetuned`、`chat_posttrained`、`merge`、`multimodal`、`other`。不得把这些类型全并成一组。用户已确认 `q4_cohort_decision.md` 中的暂定模型族、开放权重范围、任务主指标、权重和主分析样本；正式冻结仍需完成 AI 披露字段。

开放权重来源保留三态（`Yes`/`No`/`unknown`），另保留 C1 license 原值；C2 Epoch 字段尚有模糊匹配风险。未知状态不补成开放或封闭。覆盖表须在任何样本过滤前后都列出模型数、版本数、任务数和缺失交集。

## 7. 评价口径与预测边界

1. C8 先按 `task_key × task_version × metric × direction` 形成可比单元；基准总分与逐任务分分别保存。
2. 如需综合分，先按 `q4_cohort_decision.md` 中已确认的暂定指标选择、方向统一、量纲变换和权重合同；不得平均不同原始任务分数。
3. 置信区间/预测区间必须传播样本、桥接和模型不确定性，报告有效样本量与数据截止日。
4. 滚动回测只能使用每个预测时点之前已可见的数据。若可用同质时间点太少，结论降为描述性趋势或停止预测，不以 C3 混合年份分数补足样本。
5. “规模贡献”与“非规模变化”是模型条件下的描述性分解。横截面/观测数据不能单独支持因果措辞。

## 8. AI 辅助与人工责任

2026 年竞赛规则要求披露数据分析使用的 AI 工具、版本/模型、公司/开发者和版本发布日期；AI 辅助代码需在程序前部作相同标注；按要求在支撑材料说明提供给 AI 的输入材料及其输出处理方式。现有工作项执行如下：

1. 每份 Q4 程序文件顶部放置工具披露注释：OpenAI Codex 桌面版 26.917.71314（build 10954；prod；2026-09-25 更新器核验为 up_to_date）、开发机构 OpenAI；模型选择器名称及模型版本/发布日期按用户指示暂空，提交前补齐。
2. Q4 数据分析结果文档首页或末页放同一披露。`../05_审计交付/AI工具使用记录.md` 列出输入范围、辅助环节、输出处理和复核情况；模型字段按用户指示暂空。
3. 队伍复算关键数字、审阅代码与输出，并用自己的表述撰写论文。AI 不替代队伍的核心建模判断或创新。

依据：竞赛组委会 AI 工具使用规则，https://www.cmathc.org.cn/cpmcm/news/643.html。

## 9. 本版本尚未冻结的口径

- 严格/扩展开源范围及 license 分类表；
- `pretrained`、`continuously_pretrained`、`chat_posttrained` 等是否分别估计；
- IFEval 与其余任务的主指标；任务族构成和权重；
- 时间轴选模型公布日或 Leaderboard 提交日；
- C8 最新解析回退记录是否进入主样本；
- 前沿预测的 12/24 个月起算点及可接受的最少滚动回测窗口数。

上述未决项是 `q4_cohort_decision.md` 的人工决策输入，确认前不进入最终 Q4 聚合、分解或前沿结论。当前暂定运行选择已由用户确认并另行记录；其中 License 仅新增字段覆盖审计，仍没有许可证条款及研究/复现权限的分类结论，不能据此扩大“Open Weights=Yes”的含义。
