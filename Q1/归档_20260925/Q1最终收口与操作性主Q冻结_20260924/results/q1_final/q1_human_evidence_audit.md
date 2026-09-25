# Q1.1 候选分与人工证据核查

## 候选分核验

- 原始候选文件：`results/q1_1/v1/sample_scores.csv.gz`
- 归档镜像：`Q1/03_结果/Q1.1/v1/sample_scores.csv.gz`
- 版本：`q1-1-quality-model-v1`
- 行数：272,505
- 列数：13
- 联合主键：`dataset + source_domain + id + sub_path`
- 主键重复：0
- q_equal 缺失：0
- q_huber 缺失：0
- SHA-256：`52843766cf9d4d0edb18d85e2a59296e906a08dda17c7f2b628cd4798bcb7276`
- 两处文件哈希一致：是

本任务未修改该文件，q_equal 与 q_huber 原结果哈希核验通过。

Q1.1 manifest：`Q1/03_结果/Q1.1/v1/manifest.json`，SHA-256 为 `ae5a72f147e80180ba023af3bb217dd6684d6af5bdcc894e75cbe1bd13e1ee44`。其中记录 272,505 行候选分与 Q1.2 逐样本一致性最大绝对误差均为 0。

补充差异：归档目录中的 `Q1/03_结果/Q1.1/v1/q1_1_results.md` 当前 SHA-256 为 `824dad95aa87049a328a71dede987d852e65a417559c41f8a846bb1edf28a849`，与原 manifest 记录的 `a15134d61d4707f2ee0ef507be1ef47aa0a9c52c77972f629d27f438c48a2051` 不同；根目录 `results/q1_1/v1/q1_1_results.md` 仍匹配原 manifest。数值候选 CSV 未发生这一差异，本任务不覆盖任一报告。

## 盲评抽样框

- 冻结映射：`Q1/03_结果/Q1.1/v1/review_packet/restricted_id_mapping.csv`
- 行数/唯一 blind_id：793/793
- SHA-256：`d7c5267c14c57c09a6b6ea7a026ff7d9b6abb413d3dd291baf2866fb57662ebb`
- 抽样层文件：`sampling_frame.csv`
- 抽样层数：42
- SHA-256：`f238dd06133a2be1085e121739e0ed59e4a94ebc4c0d1334d3ad83bc30085bd8`

793 是冻结抽样框规模，不等于已完成 793 条人工评分。

## 当前收到的人工评分文件

- 文件：`C:/Users/Zhang Wanjie/Desktop/reviewer1_中文翻译版_zwj.xlsx`
- SHA-256：`0e0860323cb2881672c66a90dca89435ea2d776dbbe43bfeb3e7477024a7238a`
- 数据行：30；blind_id 唯一：30；五项评分完整：150/150；评分均为 1–5 整数。
- 30 个 blind_id 全部属于冻结的 793 条映射。
- 工作簿内部记录：created `2026-09-24 04:01:48`，modified `2026-09-24 04:21:55`；这些属性只作来源记录，不单独用于证明流程先后。

第二位评审：

- 文件：`C:/Users/Zhang Wanjie/Desktop/reviewer2_中文翻译版.xlsx`
- SHA-256：`e4ab3f73023b6d3f60ed56d5f3bc91388cddae96484fc510511f0379612f6e6b`
- 数据行：30；blind_id 唯一：30；五项评分完整：150/150；评分均为 1–5 整数。
- 工作簿内部记录：created `2026-09-24 11:21:54`，modified `2026-09-24 11:57:24`；这些属性只作来源记录。

双表联合校验：

- blind_id 集合相同：是；行序相同：是；逐行文本相同：是。
- 由于行序相同，任务说明所述“不同随机顺序”协议未满足，`review_order_protocol_deviation = true`。
- 整体质量二次加权 Cohen Kappa：`0.765146358067`，与说明中的 0.765146 一致。
- 两位评审整体质量 Spearman：`0.805995228596`，与说明中的 0.805995 一致。
- 这两项是对原表的只读点值复核；未做 bootstrap 或新推断。

仍未找到：

1. 对应这批 30 条的 `sampling_manifest_private.csv`，或等价的 24+6 分组清单；
2. `human_consensus.csv`；
3. `human_check_metrics.csv`；
4. `diagnostic_cases.csv`；
5. 产生候选相关 0.566692 的程序输出或摘要。

仓库中的 `Q1/03_结果/Q1.1/review_intake/v1/rater_1_form.csv` 与 `rater_2_form.csv` 是 793 行空白接收表，不能替代实际评分。

## 可接受表述与限制

按任务说明记录：小样本双人核查未能明显区分 q_equal 与 q_huber；两表顺序协议存在偏差；评审使用中文机译辅助文本；两位评审是否互不沟通独立评分当前未核实。

当前已独立核验 Kappa 与评审间 Spearman；由于 24+6 分组及候选相关输出不齐，尚不能独立核验两个候选均为 0.566692 的 representative 相关性。不得表述为“人工盲评选出了唯一最优 Q”，也不得把先前的 AI 代理评分与这批人工评审混用。
