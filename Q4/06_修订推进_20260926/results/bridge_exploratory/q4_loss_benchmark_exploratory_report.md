# Q4 Loss–Benchmark 单家族探索映射

**状态：** `ELIGIBLE_LIMITED_EXPLORATORY_NOT_CALIBRATED`。本结果补充旧版 `BLOCKED_NO_FIT` 门控，但不覆盖该历史判定，也不支持跨家族、跨协议或未来能力的校准。

## 证据与样本

- 主样本为 C5/C6 去重后的 7 个 Pythia 参数规模，来自 1 个模型家族；每条 Loss 均匹配 B1 最终 `steps=143000`、`D=299.893B` 的实测终点。考虑 B1 展示的 `val_loss`（3–4 位小数）和 `PPL`（2 位小数）的舍入区间后，逐行的 `exp(val_loss)` 与 `PPL` 区间相容。
- C5/C6 的七条 `LB_Average` 和逐项分数与 C1 对应排行榜行一致；C1 `Average` 等于六个 Benchmark 字段的算术平均。目标是 Open LLM Leaderboard v2 六项归一化分数的平均。
- 对照本地详细评测 JSON 的版本诊断中，5/7 行六项都与 C1 精确相符；差异模型为 `EleutherAI/pythia-1b, EleutherAI/pythia-6.9b`。差异字段保存在逐行 crosswalk，原始 C1/C5/C6 分数保留为拟合目标，不用 JSON 替换。
- C1 精确 Model 均唯一，但其 `#Params(B)` 与 B1/桥接表参数量有差异；绝对差与相对差逐行列于 crosswalk。这不改变主回归使用 C1 分数的规则，但表示名称连接不能证明 leaderboard checkpoint 与训练日志参数元数据完全一致。
- Loss 支持范围：`[2.0933, 2.5978]` nats/token。Pythia 单家族样本不能识别跨家族迁移，也不能分离模型规模与 Loss 的相关变化。

## 线性探索关系

**LB_Average = 7.671652 -0.881664 × Val_Loss**

Pearson `r=-0.397`，`R²=0.158`，样本内 RMSE `0.334`。该关系较弱，且不能解释为降低 Loss 会因果提高 Benchmark。

按模型尺寸逐一留出，LOSO MAE 为 `0.478`，留出训练集均值基线 MAE 为 `0.376`；LOSO RMSE 为 `0.553`，均值基线 RMSE 为 `0.424`。线性映射**未优于**留一均值基线。七个留一残差见拟合表；它们是单一家族敏感性，不是跨家族验证或校准误差。

版本差异敏感性：仅保留评测详情 JSON 六项均与 C1 精确相符的 5 行时，斜率为 `-0.762177`、R² `0.197`；若仅作替代敏感性、用所选详情 JSON 的六项均值替代两条不一致行的 C1 `Average`，7 行斜率为 `-0.791646`、R² `0.119`。主结果始终保留 C1 值；该敏感性说明分数版本差异会改变关系，不能用来裁定哪一版本“正确”。各方案的样本内及留一基线诊断见 sensitivity CSV。

## Q2 候选目标支持

Q2 B1 观察终点和 M0 拟合终点共 16 个候选目标：13 个落在 Pythia Loss 范围内，3 个超出范围。支持内分数仅作条件点映射；标记为模型名称/规模重叠的记录是样本内演示。超出范围的记录留空，不外推。表内不提供预测区间或桥接校准误差。

## 使用限制

- 本结果不替换 C1 直接分数的 Task 3 前沿情景，不向其他模型家族迁移。
- Open LLM 详细结果 JSON 提供 revision、task hash、Harness commit、Transformers 版本与运行日期；C1 清洗表没有同一行的 revision/run ID。局部 MATH 差异显示，不能把“名称相同”升级为每个记录的严格 checkpoint 证明。
- 没有足够独立家族估计跨家族不确定性；因此不报告置信区间、校准区间或 `SUPPORTED_FOR_CALIBRATION`。

## 复现

从仓库根目录运行：`python Q4/06_修订推进_20260926/scripts/q4_loss_benchmark_exploratory_mapping.py`。输入/输出/代码/合同 SHA-256 见 manifest；逐行来源、参数量和版本差异见 crosswalk；LOSO 误差见 mapping fit；版本敏感性见 sensitivity；Q2 目标支持见 target support。
