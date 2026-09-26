# Q4 历史问题收口

本目录把 Q4 v0.6 冻结包、Q4/06 修订与 Q4/07 G4 数值结果合并成当前可引用口径。它不覆盖历史快照。

- [`q4_current_status.md`](q4_current_status.md)：当前结论、数值预测、分层桥接、两种估计对象下的规模/非规模份额边界和许可口径。C1 全窗最低限度描述性占比另见 [`Q4/09`](../09_C1全窗占比补充_20260926/README.md)。
- `scripts/q4_evidence_gap_reconciliation.py`：重算 C5/C6 分层回归/留出指标与 Open Weights 许可证字段覆盖。
- `results/q4_bridge_tier_validation.csv`：按来源和可比等级的汇总指标。
- `results/q4_bridge_holdout_folds.csv`：留一尺寸/留一发布者各折误差。
- `results/q4_open_weight_license_coverage.csv`：C2 Open Weights=Yes 模型的原始 License 字段计数。
- `results/q4_evidence_gap_reconciliation_manifest.json`：输入、脚本和结果 SHA-256。

复现命令（从仓库根目录）：

```powershell
python Q4/08_历史遗留问题收口_20260926/scripts/q4_evidence_gap_reconciliation.py
```
