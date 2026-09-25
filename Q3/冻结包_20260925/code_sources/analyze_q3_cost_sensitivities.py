#!/usr/bin/env python3
"""Compute Q3 cost-only Q curves and descriptive M0 scenario shifts.

The script uses the frozen Q1 quality interface, question-provided g(Q)
formulae, the already archived Q2 B1 N/D support, and the existing Q3 M0
frontier. It does not need new Q2 observations and does not infer Q/p effects.

Run from the repository root:
    python Q3/03_代码/analyze_q3_cost_sensitivities.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Q1_FREEZE = Path("Q1/03_结果/Q1_最终收口/q1_final_freeze.json")
Q1_DOMAIN = Path("Q1/03_结果/Q1_最终收口/q1_final_domain_quality.csv")
Q1_MACRO = Path("Q1/03_结果/Q1.1/v1/a1_macro_summary.csv")
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
Q_COST_CSV = Path("Q3/04_结果/Q_cost_increment_envelope.csv")
SHARE_CSV = Path("Q3/04_结果/M0_budget_context_cost_shares.csv")
SHIFT_CSV = Path("Q3/04_结果/M0_budget_context_descriptive_shifts.csv")
REPORT_MD = Path("Q3/04_结果/Q3_cost_sensitivities_report.md")
MANIFEST_JSON = Path("Q3/04_结果/Q3_cost_sensitivities_manifest.json")

EXPECTED_Q_VERSION = "Q1-q_huber-v1"
EXPECTED_OPERATIONAL_Q = "q_huber"
G_FUNCTIONS = {
    "exponential": lambda q: 1e7 * math.exp(6.0 * q),
    "power4": lambda q: 5e9 * q**4,
    "logarithmic": lambda q: 2e9 * math.log1p(10.0 * q),
}
Q_GRID = [0.50 + 0.05 * index for index in range(11)]
TRAINING_COEFFICIENT = 6
ETA = Fraction(1, 5000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: Path) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: Path) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(relative: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {relative}")
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_frozen_q0() -> tuple[float, float, dict[str, str]]:
    freeze = read_json(Q1_FREEZE)
    if (
        freeze.get("status") != "FROZEN_WITH_LIMITATIONS"
        or freeze.get("q_version") != EXPECTED_Q_VERSION
        or freeze.get("operational_q") != EXPECTED_OPERATIONAL_Q
    ):
        raise ValueError("Q1 interface is not the expected frozen q_huber version")
    domain_rows = read_csv(Q1_DOMAIN)
    a1_rows = [row for row in domain_rows if row.get("dataset") == "a1"]
    if len(a1_rows) != 7 or any(row.get("q_version") != EXPECTED_Q_VERSION for row in a1_rows):
        raise ValueError("Expected seven A1 domain scores from the frozen Q1 version")
    q0 = sum(float(row["q_operational"]) for row in a1_rows) / 7
    macro_rows = read_csv(Q1_MACRO)
    qhuber = next((row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_huber"), None)
    qequal = next((row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_equal"), None)
    if qhuber is None or qequal is None or abs(q0 - float(qhuber["macro_domain_score"])) > 1e-10:
        raise ValueError("Frozen Q1 domain scores do not match the A1 macro summary")
    return q0, float(qequal["macro_domain_score"]), {
        "q_version": EXPECTED_Q_VERSION,
        "operational_q": EXPECTED_OPERATIONAL_Q,
        "q1_status": freeze["status"],
    }


def make_quality_cost_rows(q0: float, q2_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    d_unique_b = sorted({float(row["D_tokens_B"]) for row in q2_rows})
    if len(q2_rows) != 1176 or len(d_unique_b) != 147:
        raise ValueError("Expected the existing B1 support of 1,176 rows and 147 shared D values")
    d_reference = {
        "min": int(Fraction(str(d_unique_b[0])) * 10**9),
        "median": int(Fraction(str(d_unique_b[len(d_unique_b) // 2])) * 10**9),
        "max": int(Fraction(str(d_unique_b[-1])) * 10**9),
    }
    q_values = [q0] + [round(value, 2) for value in Q_GRID if value > q0]
    if any(q < q0 or q > 1 for q in q_values):
        raise ValueError("Q sensitivity grid must remain within [Q0, 1]")
    rows = []
    for function_name, function in G_FUNCTIONS.items():
        base_cost = function(q0)
        for q in q_values:
            delta_per_token = max(0.0, function(q) - base_cost)
            row: dict[str, Any] = {
                "g_function": function_name,
                "Q0_q_huber": q0,
                "Q_target": q,
                "g_Q_flops_per_token": function(q),
                "g_Q0_flops_per_token": base_cost,
                "incremental_flops_per_token": delta_per_token,
            }
            for d_label, d_tokens in d_reference.items():
                row[f"D_B1_{d_label}_tokens"] = d_tokens
                row[f"C_Q_at_D_B1_{d_label}_flops"] = delta_per_token * d_tokens
            rows.append(row)
    return rows, {key: float(value) for key, value in d_reference.items()}


def make_cost_share_rows(frontier_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if len(frontier_rows) != 15:
        raise ValueError(f"Expected 15 existing M0 frontier scenarios; got {len(frontier_rows)}")
    rows = []
    keys = set()
    for item in frontier_rows:
        budget = int(item["budget_flops"])
        context = int(item["context_length_tokens"])
        key = (budget, context)
        if key in keys:
            raise ValueError(f"Duplicate M0 frontier scenario: {key}")
        keys.add(key)
        if item["q1_q_version"] != EXPECTED_Q_VERSION or float(item["C_Q_flops"]) != 0:
            raise ValueError("Existing M0 frontier does not use frozen Q0 with C_Q=0")
        n_abs = int(item["selected_N_parameters"])
        d_abs = int(item["selected_D_tokens"])
        nd = n_abs * d_abs
        train = Fraction(TRAINING_COEFFICIENT * nd)
        attention = ETA * nd * context
        total = train + attention
        if total > budget:
            raise ValueError(f"Existing selected configuration exceeds budget: {key}")
        if abs(float(train) - float(item["C_train_flops"])) > 1e-6 * max(1.0, float(train)):
            raise ValueError(f"Training cost mismatch in existing frontier: {key}")
        if abs(float(attention) - float(item["C_attn_flops"])) > 1e-12 * max(1.0, float(attention)):
            raise ValueError(f"Attention cost mismatch in existing frontier: {key}")
        rows.append(
            {
                "budget_flops": budget,
                "context_length_tokens": context,
                "selected_run_id": item["selected_run_id"],
                "selected_N_B": float(item["selected_N_B"]),
                "selected_D_B": float(item["selected_D_B"]),
                "m0_predicted_val_loss": float(item["m0_predicted_val_loss"]),
                "feasible_grid_points": int(item["b1_feasible_grid_points"]),
                "C_train_flops": int(train),
                "C_Q_flops": 0,
                "C_attn_flops": float(attention),
                "C_total_flops": float(total),
                "budget_utilization": float(total / budget),
                "train_cost_share": float(train / total),
                "quality_cost_share": 0.0,
                "attention_cost_share": float(attention / total),
                "at_b1_support_upper_bound": item["selected_at_support_max_N_and_D"].lower() == "true",
                "interpretation": "descriptive fixed-Q B1/M0 scenario; no formal structural-transition claim",
            }
        )
    return rows


def make_shift_rows(share_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["budget_flops"], row["context_length_tokens"]): row for row in share_rows}
    budgets = sorted({row["budget_flops"] for row in share_rows})
    contexts = sorted({row["context_length_tokens"] for row in share_rows})
    pairs = []
    for budget in budgets:
        for left_context, right_context in zip(contexts, contexts[1:]):
            pairs.append(("context_at_fixed_budget", budget, left_context, budget, right_context))
    for context in contexts:
        for left_budget, right_budget in zip(budgets, budgets[1:]):
            pairs.append(("budget_at_fixed_context", left_budget, context, right_budget, context))

    rows = []
    for axis, left_budget, left_context, right_budget, right_context in pairs:
        left = by_key[(left_budget, left_context)]
        right = by_key[(right_budget, right_context)]
        left_n, right_n = left["selected_N_B"], right["selected_N_B"]
        left_d, right_d = left["selected_D_B"], right["selected_D_B"]
        left_loss, right_loss = left["m0_predicted_val_loss"], right["m0_predicted_val_loss"]
        share_tvd = 0.5 * sum(
            abs(right[key] - left[key])
            for key in ("train_cost_share", "quality_cost_share", "attention_cost_share")
        )
        rows.append(
            {
                "comparison_axis": axis,
                "from_budget_flops": left_budget,
                "from_context_tokens": left_context,
                "to_budget_flops": right_budget,
                "to_context_tokens": right_context,
                "from_run_id": left["selected_run_id"],
                "to_run_id": right["selected_run_id"],
                "N_B_from": left_n,
                "N_B_to": right_n,
                "N_B_absolute_change": right_n - left_n,
                "N_B_relative_change": (right_n / left_n - 1.0) if left_n else None,
                "D_B_from": left_d,
                "D_B_to": right_d,
                "D_B_absolute_change": right_d - left_d,
                "D_B_relative_change": (right_d / left_d - 1.0) if left_d else None,
                "m0_loss_from": left_loss,
                "m0_loss_to": right_loss,
                "m0_loss_change": right_loss - left_loss,
                "train_share_from": left["train_cost_share"],
                "train_share_to": right["train_cost_share"],
                "attention_share_from": left["attention_cost_share"],
                "attention_share_to": right["attention_cost_share"],
                "cost_share_total_variation": share_tvd,
                "configuration_changed": (left_n != right_n or left_d != right_d),
                "support_bound_at_to_scenario": right["at_b1_support_upper_bound"],
                "interpretation": "descriptive pairwise change only; not inferential structural transition",
            }
        )
    return rows


def main() -> None:
    q0, q_equal, q_metadata = get_frozen_q0()
    q2_rows = read_csv(Q2_GRID)
    frontier_rows = read_csv(Q3_FRONTIER)
    q_cost_rows, d_reference = make_quality_cost_rows(q0, q2_rows)
    share_rows = make_cost_share_rows(frontier_rows)
    shift_rows = make_shift_rows(share_rows)
    write_csv(Q_COST_CSV, q_cost_rows)
    write_csv(SHARE_CSV, share_rows)
    write_csv(SHIFT_CSV, shift_rows)

    max_d = int(d_reference["max"])
    by_cost = {
        (row["g_function"], round(float(row["Q_target"]), 8)): float(row["C_Q_at_D_B1_max_flops"])
        for row in q_cost_rows
    }
    q_cost_table = [
        "| 质量成本形式 | Q | 增量质量成本（B1 最大 D） FLOPs |",
        "|---|---:|---:|",
    ]
    for q in (0.50, 0.60, 0.80, 1.00):
        for function_name in G_FUNCTIONS:
            value = by_cost[(function_name, round(q, 8))]
            q_cost_table.append(f"| {function_name} | {q:.2f} | {value:.4e} |")

    changes = [row for row in shift_rows if row["configuration_changed"]]
    ctx_pairs = [row for row in shift_rows if row["comparison_axis"] == "context_at_fixed_budget"]
    budget_pairs = [row for row in shift_rows if row["comparison_axis"] == "budget_at_fixed_context"]
    stable_selections = sum(not row["configuration_changed"] for row in shift_rows)
    report = [
        "# Q3 质量成本曲线与预算/上下文描述性敏感性",
        "",
        "## 数据依赖核验",
        "",
        f"本分析直接读取已冻结的 Q1 主质量接口（`{q_metadata['q_version']}`）、题面成本公式、已归档的 Q2 B1 1,176 点网格（仅用于 D 支持范围参考）以及现有 Q3 15 情景前沿。**无需新增 Q2 数据、重新拟合 Q2 或估计 p/Q 响应。**",
        f"Q1 主基线 $Q_0={q0:.10f}$，95% 区间已在 Q3 基线报告中归档；`q_equal` 对照值为 {q_equal:.10f}。新增质量曲线仅计算公式给出的 $C_Q$，不声称 Q 提升带来多少 Loss 改善。",
        "",
        "## 质量提升的算力成本曲线",
        "",
        f"按题面三种 $g(Q)$ 和 $C_Q=D[g(Q)-g(Q_0)]_+$ 计算，Q 网格取冻结 $Q_0$ 及 0.50–1.00、步长 0.05 的场景点。D 参考为现有 B1 实测支持中的最小/中位/最大值：{d_reference['min']:,} / {d_reference['median']:,} / {d_reference['max']:,} tokens。所有单位均为绝对 Token 数与 FLOPs。",
        "",
        f"下表展示最大实测 D（{max_d:,} tokens）下的成本；完整 Q 网格和三档 D 参考见 `Q_cost_increment_envelope.csv`。",
        "",
        *q_cost_table,
        "",
        "这些数字可以回答某个质量目标的成本规模，但没有 Q→Loss 响应，不能据此选出最优 Q。",
        "",
        "## 现有 15 情景的成本构成与配置变化",
        "",
        f"在固定 $Q=Q_0$ 的现有前沿中，$C_Q=0$；训练与注意力成本占实际成本的份额见 `M0_budget_context_cost_shares.csv`。共比较了 {len(shift_rows)} 对相邻预算或上下文情景，其中 {len(changes)} 对 N/D 配置发生变化、{stable_selections} 对配置保持不变。逐对 N/D、Loss、成本份额变化见 `M0_budget_context_descriptive_shifts.csv`。",
        f"固定预算比较上下文长度共 {len(ctx_pairs)} 对；固定上下文比较相邻预算共 {len(budget_pairs)} 对。该比较用于描述资源配置如何随情景档位变化，不使用 bootstrap/KKT 判据，不称为正式结构转移。最高预算结果仍到达 B1 支持网格上界。",
        "",
        "## 解释边界",
        "",
        "1. 输入是已存在的冻结 Q1/Q2/Q3 文件；此分析未引入新观测或新拟合。",
        "2. 质量成本是题面代理函数的算术结果，不验证题面系数的经验适配性，也不估计质量收益。",
        "3. 场景变化来自离散预算与上下文档位，当前 Q2 M0 是 B1 内样本拟合；变化表不作因果解释或显著性推断。",
        "4. Q/p 联合优化仍需 Q2 提供可追溯的配比数据与可识别的 Q/p→Loss 响应。",
        "",
        "复现命令：`python Q3/03_代码/analyze_q3_cost_sensitivities.py`。",
        "",
    ]
    (ROOT / REPORT_MD).write_text("\n".join(report), encoding="utf-8")

    script_path = Path(__file__).resolve()
    inputs = [Q1_FREEZE, Q1_DOMAIN, Q1_MACRO, Q2_GRID, Q3_FRONTIER]
    outputs = [Q_COST_CSV, SHARE_CSV, SHIFT_CSV, REPORT_MD]
    manifest = {
        "artifact": "Q3 cost-only quality uplift curves and descriptive budget/context shifts",
        "new_q2_data_required": False,
        "dependencies": {
            "q1": "frozen q_huber Q0 interface",
            "q2": "existing archived B1 grid for observed D support references; no new observations or fit",
            "q3": "existing fixed-Q M0 15-scenario frontier",
            "formulas": "question-provided g(Q) and compute-cost equations",
        },
        "claim_boundary": "cost-only Q scenarios plus descriptive B1/M0 config changes; no Q/p response, full Q3 optimum, or formal structural-transition claim",
        "quality_grid": [q0] + [round(value, 2) for value in Q_GRID if value > q0],
        "D_reference_tokens": d_reference,
        "q1_metadata": q_metadata,
        "python_version": platform.python_version(),
        "script": {"path": script_path.relative_to(ROOT).as_posix(), "sha256": sha256(script_path)},
        "inputs": {path.as_posix(): sha256(ROOT / path) for path in inputs},
        "outputs": {path.as_posix(): sha256(ROOT / path) for path in outputs},
    }
    (ROOT / MANIFEST_JSON).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(q_cost_rows)} quality-cost rows, {len(share_rows)} scenario cost-share rows, and {len(shift_rows)} pairwise descriptive comparisons.")
    print(f"No new Q2 data used; configuration changed in {len(changes)}/{len(shift_rows)} adjacent comparisons.")


if __name__ == "__main__":
    main()
