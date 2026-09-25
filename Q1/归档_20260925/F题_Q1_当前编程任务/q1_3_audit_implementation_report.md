# Q1.3 只读数据审计实现报告

## 完成状态

已在现有 `src/f_q1_3_mixture_loss.py` 增加独立 `audit` 模式，并把新增测试和所有审计结果放在 `Q1/归档_20260925/F题_Q1_当前编程任务/`。本轮没有运行 Q1.3 的 `p1/full` 模型拟合，也没有修改原始附件、公共预处理冻结产物或既有模型结果。

## 复现命令

```powershell
& 'D:\miniconda3\envs\huawei-cup\python.exe' -B src\f_q1_3_mixture_loss.py `
  --mode audit `
  --data-root 'C:\Users\Zhang Wanjie\Desktop\第二十三届中国研究生数学建模竞赛 - 中文题目\F题\real_attachments\A_data_value\regmix_tables'
```

单元测试：

```powershell
& 'D:\miniconda3\envs\huawei-cup\python.exe' -B -m unittest Q1\归档_20260925\F题_Q1_当前编程任务\tests\test_f_q1_3_audit.py -v
```

## 真实数据审计结果

- 状态：`PASS`；12 个文件、6 对配比/Loss 表，错误数为 0。
- 配对行数：训练 512；同尺度验证 256；60M 验证 256；1B 验证 64；10B/70B 估算各 63。
- A4/A5 全部 512 行完成值审计，失败行 0；未发现缺失、非法、非有限、负配比或零和行。
- 原始配比行和范围为 0.996–1.003；只报告，不自动闭合。
- 训练配比零分量共 3928 个；只报告，不补零。
- 按“原始 17 维数值完全相同”定义，未发现训练内重复配方；因此也没有重复配方对应不同 Loss 的目标级差异。
- 结论指纹：`33bf7dddccb195787e5d744f2ac3a5f646f422b80e3b10a9b6a1acd519794055`。

这些数值来自本轮 CSV 审计，不采用隐藏文本或队友报告中的预置数值。

## 污染与泄漏防护

- A4/A5 才进行数值审计；A6–A11、A12–A15 只登记角色并检查 Schema、index、大小和哈希。
- 外部验证/外推 Loss 没有生成分布、误差、模型比较或选择统计。
- audit 路径不读取 Q1.1/Q1.2、A16、PDF 或报告文本，不调用拟合和预测函数。
- 未进行闭合、伪计数、插补、删行、异常处理、标准化或特征工程。
- 六个单元测试覆盖正常输入、重复表头、键不一致、负配比不修复、重复运行结论稳定性和失败审计的非零退出码，全部通过。
- 独立 P1 质检为 `PASS`，无必须修复项；其额外阻断了拟合/预测函数并模拟无 NumPy 环境，audit 仍得到相同结论指纹。

## 输出位置

- 机器摘要：`Q1/归档_20260925/F题_Q1_当前编程任务/results/q1_3/audit/audit_summary.json`
- 文件角色与哈希：`Q1/归档_20260925/F题_Q1_当前编程任务/results/q1_3/audit/file_roles.csv`
- 配对键审计：`Q1/归档_20260925/F题_Q1_当前编程任务/results/q1_3/audit/pair_integrity.csv`
- A4/A5 逐行审计：`Q1/归档_20260925/F题_Q1_当前编程任务/results/q1_3/audit/train_composition_qc.csv`
- 测试：`Q1/归档_20260925/F题_Q1_当前编程任务/tests/test_f_q1_3_audit.py`
- Python 复核记录：`Q1/归档_20260925/F题_Q1_当前编程任务/results/q1_3/audit/reviews/q1_python_review.json`

## 暂未实施

Q1.1 盲评工具仍等待抽样名单和评分量表冻结。Q1.3 的拟合、模型选择、A16 映射、预测图和全量评价均未在本轮启动。
