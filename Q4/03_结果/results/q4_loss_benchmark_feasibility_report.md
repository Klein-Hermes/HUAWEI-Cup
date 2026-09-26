# Q4 Loss–Benchmark 预拟合可行性门控报告

> 门控仅做来源、字段、重复、量尺与支持范围审计；拟合执行数 = 0。合同：`Q4/01_方案说明/q4_model_contract_v0.6.md`。

## 判定

预拟合状态：`BLOCKED_NO_FIT`。现阶段不得拟合 Loss–Benchmark 映射。

阻断理由：

- Q2 `loss_protocol_matrix.csv` 对 B1 的 Loss 单位、评测集/切分、tokenizer 与预处理均标为 UNKNOWN；因此虽有 C5/C6 的“高可比”文字标签，仍不能独立核验实际协议与单位。
- C5/C6 含有 `Model` 模型名称和 `Loss_Source` 来源字段；模型名称、原始来源与数值字段已纳入完全重复键，并在逐行审计表中保留。表内没有 revision/SHA 或排行榜提交/评测运行 ID，因此无法把名称对应到确切历史版本与评测运行。六个 `LB_*` 字段按 Open LLM Leaderboard 官方说明映射到 IFEval、BBH、MATH、GPQA、MuSR、MMLU-Pro，门控核对了 `LB_Average` 与六项算术均值的一致性。
- 高可比记录的模型名称为 Pythia 各规模模型；C5/C6 的 7 条高可比内容记录跨表逐行重复。按模型名、来源及数值去重后有 7 个名称、1 个来源家族提示，但没有经核验的模型 revision 或评测运行，不能将名称数当作独立版本数，也不能验证跨家族迁移。官方 Pythia 说明 8 个主模型规模各训练约 299.893B tokens，与桥接表 D 相符；这不能确认本地 Loss 的评测切分或具体版本。
- Q2 B1 每个参数规模的终点观测与 M0 终点拟合值已列为候选 Loss 目标；Q4 尚未指定需要转换的未来 `(N,D)` 目标网格，因此不能把训练样本范围冒充预测目标支持。
- Open LLM Leaderboard 的任务定义及高分方向见 [官方任务与评测说明](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about)；Pythia 训练 token 数与模型版本信息见 [EleutherAI Pythia 官方仓库](https://github.com/EleutherAI/pythia)。这些公开说明用于核验字段语义，不补出附件中缺失的行级版本和 Loss protocol。

## 门控产物

- `Q4/03_结果/results/q4_loss_benchmark_feasibility.csv`：按 C5、C6 与跨表去重合并表给出状态、配对数、支持范围与阻断原因。
- `Q4/03_结果/results/q4_loss_bridge_pair_audit.csv`：逐行审计表；源表行号、可比性标签、数值字段、跨表重复键和身份/单位限制均保留。
- `Q4/03_结果/results/q4_loss_bridge_target_audit.csv`：Q2 B1/M0 每个规模终点的候选 Loss 值及其相对 C5/C6 高可比范围位置。它们不是已批准的未来转换目标，也不证明单位/协议一致。

## 证据限制

本门控没有读入或改写 Q1–Q3 原始数据；只读复核 Q2 的冻结 B1 与 M0 结果文件。没有识别到已批准的 Q1/Q3 直接 Loss-to-Benchmark 转换目标。继续桥接前须由队伍明确未来目标网格及 Benchmark 定义，并从来源补足 Loss 单位/评测协议证据；取得证据后重做门控。
