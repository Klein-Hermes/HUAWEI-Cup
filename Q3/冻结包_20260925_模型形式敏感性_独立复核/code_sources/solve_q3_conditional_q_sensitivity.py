#!/usr/bin/env python3
"""Build a conditional Q-only optimizer with break-even response thresholds.

Conditional response assumption (not estimated from Q2):
    L_cond(N,D,Q | beta_Q) = L_M0(N,D) - beta_Q * (Q - Q0), beta_Q >= 0.

For each B1-supported N/D point, Q is allowed to rise continuously from the
frozen Q0 up to the maximum level affordable under each g(Q) cost function.
The script reports the exact lower-envelope regimes in beta_Q rather than
inventing a response coefficient. The domain mix p is excluded; no p vector is assigned.

Run from the repository root:
    python Q3/03_代码/solve_q3_conditional_q_sensitivity.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_q3_cost_sensitivities as cost_module


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
INTERVALS_CSV = Path("Q3/04_结果/Q_conditional_Q_optimality_intervals.csv")
BREAK_EVEN_CSV = Path("Q3/04_结果/Q_conditional_Q_break_even_summary.csv")
REPORT_MD = Path("Q3/04_结果/Q3_conditional_Q_optimization_report.md")
MANIFEST_JSON = Path("Q3/04_结果/Q3_conditional_Q_optimization_manifest.json")
ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
PREDICTION_TOLERANCE = 1e-8


def write_csv(relative: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {relative}")
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def g_inverse(function_name: str, value: float) -> float:
    if function_name == "exponential":
        return math.log(value / 1e7) / 6.0
    if function_name == "power4":
        return (value / 5e9) ** 0.25
    if function_name == "logarithmic":
        return (math.exp(value / 2e9) - 1.0) / 10.0
    raise ValueError(f"Unknown g(Q) form: {function_name}")


def max_affordable_q(function_name: str, q0: float, d_tokens: int, remaining_flops: float) -> float:
    g = cost_module.G_FUNCTIONS[function_name]
    if remaining_flops <= 0:
        return q0
    q1_cost = d_tokens * max(0.0, g(1.0) - g(q0))
    if q1_cost <= remaining_flops:
        return 1.0
    target_g = g(q0) + remaining_flops / d_tokens
    q_high = min(1.0, max(q0, g_inverse(function_name, target_g)))
    # Guard the inverse against one-ULP budget overshoot.
    if d_tokens * max(0.0, g(q_high) - g(q0)) <= remaining_flops:
        return q_high
    low, high = q0, q_high
    for _ in range(80):
        middle = (low + high) / 2.0
        if d_tokens * max(0.0, g(middle) - g(q0)) <= remaining_flops:
            low = middle
        else:
            high = middle
    return low


def lower_envelope(lines: list[dict[str, Any]]) -> list[tuple[float, float, dict[str, Any]]]:
    """Return (beta_start, beta_end, line) segments for beta >= 0."""
    # For equal slopes, only the lowest M0 loss can ever appear on the envelope.
    by_slope: dict[float, dict[str, Any]] = {}
    for line in lines:
        slope = float(line["slope"])
        old = by_slope.get(slope)
        if old is None or (line["intercept"], line["total_cost"], line["run_id"]) < (
            old["intercept"], old["total_cost"], old["run_id"]
        ):
            by_slope[slope] = line

    ordered = sorted(by_slope.values(), key=lambda line: line["slope"], reverse=True)
    hull: list[dict[str, Any]] = []
    starts: list[float] = []
    for line in ordered:
        start = -math.inf
        while hull:
            previous = hull[-1]
            intersection = (line["intercept"] - previous["intercept"]) / (
                previous["slope"] - line["slope"]
            )
            if intersection <= starts[-1]:
                hull.pop()
                starts.pop()
            else:
                start = intersection
                break
        if not hull:
            start = -math.inf
        hull.append(line)
        starts.append(start)

    segments = []
    for index, line in enumerate(hull):
        start = max(0.0, starts[index])
        end = starts[index + 1] if index + 1 < len(hull) else math.inf
        if end > start and end > 0:
            segments.append((start, end, line))
    if not segments:
        raise ValueError("Conditional objective lower envelope is empty")
    return segments


def main() -> None:
    q0, q_equal, q_metadata = cost_module.get_frozen_q0()
    grid_rows = cost_module.read_csv(Q2_GRID)
    params_file = cost_module.read_json(Q2_PARAMS)
    frontier_rows = cost_module.read_csv(Q3_FRONTIER)
    if params_file.get("fit_scope") != "B1 only" or len(grid_rows) != 1176 or len(frontier_rows) != 15:
        raise ValueError("Expected the frozen B1-only M0 fit, 1,176 grid points, and 15 Q3 scenarios")
    parameters = {key: float(value) for key, value in params_file["parameters"].items()}

    points = []
    seen = set()
    for row in grid_rows:
        run_id = row["run_id"]
        if run_id in seen:
            raise ValueError(f"Duplicate B1 run_id: {run_id}")
        seen.add(run_id)
        n_b = float(row["N_params_B"])
        d_b = float(row["D_tokens_B"])
        supplied = float(row["predicted_val_loss"])
        predicted = (
            parameters["E"]
            + parameters["A"] * n_b ** (-parameters["alpha"])
            + parameters["B"] * d_b ** (-parameters["beta"])
        )
        if abs(predicted - supplied) > PREDICTION_TOLERANCE:
            raise ValueError(f"M0 prediction mismatch for run_id={run_id}")
        points.append(
            {
                "run_id": run_id,
                "n_b": n_b,
                "d_b": d_b,
                "n_abs": int(Fraction(row["N_params_B"]) * 10**9),
                "d_abs": int(Fraction(row["D_tokens_B"]) * 10**9),
                "loss": predicted,
            }
        )

    interval_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario in frontier_rows:
        budget = int(scenario["budget_flops"])
        context = int(scenario["context_length_tokens"])
        baseline_run = scenario["selected_run_id"]
        base_feasible = []
        base_costs: dict[str, Fraction] = {}
        for point in points:
            nd = point["n_abs"] * point["d_abs"]
            base_cost = Fraction(TRAINING_COEFFICIENT * nd) + ETA * nd * context
            if base_cost <= budget:
                base_feasible.append(point)
                base_costs[point["run_id"]] = base_cost
        if len(base_feasible) != int(scenario["b1_feasible_grid_points"]):
            raise ValueError(f"Base feasibility count mismatch: C={budget}, L={context}")
        baseline = min(
            base_feasible,
            key=lambda point: (
                point["loss"], base_costs[point["run_id"]], point["n_b"], point["d_b"], point["run_id"]
            ),
        )
        if baseline["run_id"] != baseline_run:
            raise ValueError(f"Conditional model baseline disagrees with fixed-Q frontier: C={budget}, L={context}")
        baseline_loss = baseline["loss"]

        for function_name, function in cost_module.G_FUNCTIONS.items():
            lines = []
            # At beta_Q=0, spending no quality-uplift cost is the deterministic tie-break.
            base_nd = baseline["n_abs"] * baseline["d_abs"]
            base_train = Fraction(TRAINING_COEFFICIENT * base_nd)
            base_attention = ETA * base_nd * context
            lines.append(
                {
                    "slope": 0.0,
                    "intercept": baseline_loss,
                    "delta_q": 0.0,
                    "q": q0,
                    "point": baseline,
                    "train_cost": base_train,
                    "attention_cost": base_attention,
                    "quality_cost": 0.0,
                    "total_cost": base_train + base_attention,
                    "run_id": baseline["run_id"],
                    "is_q0_baseline": True,
                }
            )
            for point in base_feasible:
                nd = point["n_abs"] * point["d_abs"]
                train_cost = Fraction(TRAINING_COEFFICIENT * nd)
                attention_cost = ETA * nd * context
                base_cost = train_cost + attention_cost
                remaining = float(Fraction(budget) - base_cost)
                q_max = max_affordable_q(function_name, q0, point["d_abs"], remaining)
                delta_q = max(0.0, q_max - q0)
                quality_cost = point["d_abs"] * max(0.0, function(q_max) - function(q0))
                total_cost = float(base_cost) + quality_cost
                if total_cost > budget + max(1.0, budget) * 1e-12:
                    raise ValueError("Conditional configuration exceeds its FLOPs budget")
                lines.append(
                    {
                        "slope": -delta_q,
                        "intercept": point["loss"],
                        "delta_q": delta_q,
                        "q": q_max,
                        "point": point,
                        "train_cost": train_cost,
                        "attention_cost": attention_cost,
                        "quality_cost": quality_cost,
                        "total_cost": total_cost,
                        "run_id": point["run_id"],
                        "is_q0_baseline": False,
                    }
                )

            segments = lower_envelope(lines)
            for segment_index, (beta_start, beta_end, line) in enumerate(segments, start=1):
                point = line["point"]
                interval_rows.append(
                    {
                        "budget_flops": budget,
                        "context_length_tokens": context,
                        "g_function": function_name,
                        "beta_Q_segment": segment_index,
                        "beta_Q_lower": beta_start,
                        "beta_Q_lower_inclusive": not (beta_start == 0 and line["delta_q"] > 1e-12),
                        "beta_Q_upper": "Infinity" if math.isinf(beta_end) else beta_end,
                        "beta_Q_upper_exclusive": True,
                        "beta_Q_units": "validation-loss units per one Q-score unit",
                        "selected_run_id": line["run_id"],
                        "selected_N_B": point["n_b"],
                        "selected_D_B": point["d_b"],
                        "selected_Q": line["q"],
                        "Q_increment_from_Q0": line["delta_q"],
                        "M0_base_loss": line["intercept"],
                        "conditional_loss_at_segment_start": line["intercept"] - beta_start * line["delta_q"],
                        "C_train_flops": int(line["train_cost"]),
                        "C_Q_flops": line["quality_cost"],
                        "C_attn_flops": float(line["attention_cost"]),
                        "C_total_flops": float(line["total_cost"]),
                        "budget_utilization": float(line["total_cost"] / budget),
                        "is_fixed_Q0_baseline": line["is_q0_baseline"],
                        "p_setting": "p excluded from model and decision variables; no p vector assigned",
                        "conditional_assumption": "L_cond = L_M0(N,D) - beta_Q*(Q-Q0), beta_Q >= 0",
                    }
                )

            positive_q_lines = [line for line in lines if line["delta_q"] > 1e-12]
            if positive_q_lines:
                first = min(
                    positive_q_lines,
                    key=lambda line: (
                        max(0.0, (line["intercept"] - baseline_loss) / line["delta_q"]),
                        line["intercept"],
                        line["total_cost"],
                        line["run_id"],
                    ),
                )
                beta_threshold = max(0.0, (first["intercept"] - baseline_loss) / first["delta_q"])
                first_line = first
            else:
                beta_threshold = None
                first_line = None
            envelope_positive = [line for _, _, line in segments if line["delta_q"] > 1e-12]
            first_reallocation = next(
                ((start, line) for start, _, line in segments if line["run_id"] != baseline["run_id"]),
                None,
            )
            summary_rows.append(
                {
                    "budget_flops": budget,
                    "context_length_tokens": context,
                    "g_function": function_name,
                    "q0_reference": q0,
                    "q_equal_sensitivity_reference": q_equal,
                    "fixed_Q0_baseline_run_id": baseline["run_id"],
                    "fixed_Q0_baseline_loss": baseline_loss,
                    "beta_Q_first_positive_Q": beta_threshold,
                    "first_positive_Q_run_id": first_line["run_id"] if first_line else None,
                    "first_positive_Q_N_B": first_line["point"]["n_b"] if first_line else None,
                    "first_positive_Q_D_B": first_line["point"]["d_b"] if first_line else None,
                    "first_positive_Q_level": first_line["q"] if first_line else None,
                    "beta_Q_first_ND_reallocation": first_reallocation[0] if first_reallocation else None,
                    "first_reallocated_run_id": first_reallocation[1]["run_id"] if first_reallocation else None,
                    "first_reallocated_N_B": first_reallocation[1]["point"]["n_b"] if first_reallocation else None,
                    "first_reallocated_D_B": first_reallocation[1]["point"]["d_b"] if first_reallocation else None,
                    "first_reallocated_Q": first_reallocation[1]["q"] if first_reallocation else None,
                    "conditional_envelope_segments": len(segments),
                    "positive_Q_segments_on_envelope": len(envelope_positive),
                    "B1_feasible_ND_points": len(base_feasible),
                    "model_scope": "conditional Q-only response; p excluded; B1/M0 support grid",
                }
            )

    write_csv(INTERVALS_CSV, interval_rows)
    write_csv(BREAK_EVEN_CSV, summary_rows)

    report_lines = [
        "# Q3 条件质量优化：响应系数临界值",
        "",
        "## 模型定义与识别边界",
        "",
        f"正式基线为冻结 Q1 `{q_metadata['q_version']}`，$Q_0={q0:.10f}$。在现有 Q2 B1/M0 损失面和 1,176 个 B1 实测 N/D 点上，构造一个显式假设的 Q-only 响应：",
        "",
        r"$$L_{\mathrm{cond}}(N,D,Q\mid\beta_Q)=\widehat L_{M0}(N,D)-\beta_Q(Q-Q_0),\qquad \beta_Q\ge0.$$",
        "",
        "这里 $\\beta_Q$ 是未知的 Loss 改善系数，不从现有数据估计；p 不进入模型，也不作为决策变量，本分支未设置配比向量。对每个 N/D 点，在题面预算约束下连续求可负担的最大 Q，再比较各方案的条件目标值。报告给出 $\\beta_Q$ 的最优配置区间和 N/D 首次重新分配的临界值，不挑选一个未经数据支持的系数。",
        "",
        "因此这是**参数化条件优化**，不是经 Q2 识别的质量响应，也不是完整 Q/p 联合最优。Q2 新数据不是本计算的前置条件；实际计算只复用冻结 Q1、既有 Q2 M0/B1 网格、已有 Q3 预算/上下文情景及题面成本函数。",
        "",
        "## N/D 重新分配的系数阈值",
        "",
        "基线最优 N/D 点通常留有预算余量，因此只要 $\\beta_Q>0$，就可能先在同一 N/D 点上作一个很小的 Q 提升；这个零阈值本身不表示 N/D 资源重新分配。下表报告更有解释力的阈值：N/D 首次改选为另一网格点所需的 $\\beta_Q$。若显示“无重新分配”，表示对所有 $\\beta_Q\\ge0$，该情景的条件最优仍保持在同一 N/D 点（Q 可随预算余量提高）。",
        "在 $\\beta_Q=0$ 时，按不额外花费质量成本的规则保留固定-Q0基线；正响应系数下先可能在同一 N/D 点使用剩余预算提升 Q。有限断点处相邻配置目标值相同；区间文件将断点分配给新配置。",
        "",
        "| 上下文 | 预算 FLOPs | $g(Q)$ | N/D 改选临界 $\\beta_Q$ | 改选配置 (N_B, D_B, Q) | 条件包络段数 |",
        "|---:|---:|---|---:|---|---:|",
    ]
    for row in summary_rows:
        threshold = "无重新分配" if row["beta_Q_first_ND_reallocation"] is None else f"{row['beta_Q_first_ND_reallocation']:.6g}"
        config = "—" if row["first_reallocated_run_id"] is None else (
            f"({row['first_reallocated_N_B']:.6g}, {row['first_reallocated_D_B']:.6g}, {row['first_reallocated_Q']:.6g})"
        )
        report_lines.append(
            f"| {row['context_length_tokens']:,} | {row['budget_flops']:.0e} | {row['g_function']} | {threshold} | {config} | {row['conditional_envelope_segments']} |"
        )
    report_lines.extend(
        [
            "",
            "完整的 $\\beta_Q$ 分段最优方案见 `Q_conditional_Q_optimality_intervals.csv`；首次发生 N/D 重新分配的摘要见 `Q_conditional_Q_break_even_summary.csv`。区间端点来自可行 B1 网格与本条件目标函数的线性下包络。",
            "",
            "## 解释边界",
            "",
            "1. 线性 Q→Loss 响应是条件假设；当前 Q2 M0 不含 Q 项，故 $\\beta_Q$ 未被识别。",
            "2. p 不进入模型，也不作为决策变量；本分支未设置配比向量，因此不回答领域配比优化。",
            "3. N/D 仅搜索 B1 的 8×147 实测网格；不插值、不外推。",
            "4. 该模型给出“如果响应系数达到某值，配置何时变化”的条件结论；不得把阈值或条件配置写成实测最优质量。",
            "5. 正式 Q/p 优化仍需可追溯的配比数据与独立质量响应证据。",
            "",
            "复现命令：`python Q3/03_代码/solve_q3_conditional_q_sensitivity.py`。",
            "",
        ]
    )
    (ROOT / REPORT_MD).write_text("\n".join(report_lines), encoding="utf-8")

    script_path = Path(__file__).resolve()
    inputs = [Q2_GRID, Q2_PARAMS, Q3_FRONTIER]
    # Q1 files are direct inputs through the shared verified baseline routine.
    q1_inputs = [cost_module.Q1_FREEZE, cost_module.Q1_DOMAIN, cost_module.Q1_MACRO]
    outputs = [INTERVALS_CSV, BREAK_EVEN_CSV, REPORT_MD]
    manifest = {
        "artifact": "Q3 conditional Q-only optimizer with response-coefficient thresholds",
        "conditional_response": "L_cond = L_M0(N,D) - beta_Q*(Q-Q0), beta_Q >= 0",
        "beta_Q_source": "unknown conditional parameter; not estimated from Q2",
        "p_setting": "p excluded from model and decision variables; no p vector assigned",
        "new_q2_data_required": False,
        "scope": "Q2 B1/M0 support grid; existing budgets and contexts; three question-provided g(Q) cost forms",
        "claim_boundary": "conditional sensitivity only; not a data-identified Q response or full Q/p optimum",
        "q1_metadata": q_metadata,
        "python_version": platform.python_version(),
        "script": {"path": script_path.relative_to(ROOT).as_posix(), "sha256": cost_module.sha256(script_path)},
        "shared_cost_formula_script": {
            "path": Path(cost_module.__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": cost_module.sha256(Path(cost_module.__file__).resolve()),
        },
        "inputs": {
            path.as_posix(): cost_module.sha256(ROOT / path)
            for path in q1_inputs + inputs
        },
        "outputs": {path.as_posix(): cost_module.sha256(ROOT / path) for path in outputs},
    }
    (ROOT / MANIFEST_JSON).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_rows)} conditional scenario thresholds and {len(interval_rows)} lower-envelope segments.")
    print("All response coefficients remain explicit unknowns; no Q/p effect was inferred from Q2.")


if __name__ == "__main__":
    main()
