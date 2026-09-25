#!/usr/bin/env python3
"""Propagate Q1 baseline uncertainty through Q3 cost and conditional thresholds.

Q0 branches are the frozen q_huber point, its A1 macro 95% interval endpoints,
and the separately frozen q_equal sensitivity interface. This changes only the
baseline in the question-provided cost functions; it does not estimate Q->Loss.

Run from the repository root:
    python Q3/03_代码/analyze_q3_q0_baseline_sensitivity.py
"""

from __future__ import annotations

import argparse
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
import solve_q3_conditional_q_sensitivity as conditional_module


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
Q3_EXISTING_BETA = Path("Q3/04_结果/Q_conditional_Q_break_even_summary.csv")
Q1_MACRO = Path("Q1/03_结果/Q1.1/v1/a1_macro_summary.csv")
OUT_COST = Path("Q3/04_结果/Q3_Q0_baseline_sensitivity_cost.csv")
OUT_BETA = Path("Q3/04_结果/Q3_Q0_baseline_sensitivity_beta_thresholds.csv")
OUT_REPORT = Path("Q3/04_结果/Q3_Q0_baseline_sensitivity_report.md")
OUT_MANIFEST = Path("Q3/04_结果/Q3_Q0_baseline_sensitivity_manifest.json")
COMMON_TARGETS = [round(0.50 + 0.05 * i, 2) for i in range(11)]
TRAINING_COEFFICIENT = 6
ETA = Fraction(1, 5000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(relative: Path) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(relative: Path) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_output_dir(parser: argparse.ArgumentParser, smoke: bool, supplied: str | None) -> Path:
    if not smoke:
        if supplied:
            parser.error("--output-dir is only allowed with --smoke")
        return ROOT
    if not supplied:
        parser.error("--smoke requires --output-dir so partial outputs cannot overwrite full results")
    raw = Path(supplied)
    resolved = (raw if raw.is_absolute() else ROOT / raw).resolve()
    project_root = ROOT.resolve()
    official_results = (ROOT / "Q3" / "04_结果").resolve()
    if (
        not resolved.is_relative_to(project_root)
        or resolved == project_root
        or resolved.is_relative_to(official_results)
        or official_results.is_relative_to(resolved)
    ):
        parser.error("smoke --output-dir must be a project-local scratch directory isolated from final results")
    return resolved


def q0_branches(macro_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qhuber = next((row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_huber"), None)
    qequal = next((row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_equal"), None)
    if qhuber is None or qequal is None:
        raise ValueError("A1 macro summary must contain q_huber and q_equal")
    point = float(qhuber["macro_domain_score"])
    low = float(qhuber["ci_low"])
    high = float(qhuber["ci_high"])
    equal = float(qequal["macro_domain_score"])
    if not (0 < low <= point <= high <= 1 and 0 < equal < 1):
        raise ValueError("Q1 Q0 values or the q_huber interval are outside (0,1]")
    branches = [
        {"branch": "q_huber_point", "q0": point, "basis": "frozen operational Q1 main interface"},
        {"branch": "q_huber_ci95_low", "q0": low, "basis": "lower endpoint of A1 seven-domain macro 95% interval"},
        {"branch": "q_huber_ci95_high", "q0": high, "basis": "upper endpoint of A1 seven-domain macro 95% interval"},
        {"branch": "q_equal_sensitivity", "q0": equal, "basis": "frozen Q1 sensitivity interface; not an uncertainty endpoint"},
    ]
    metadata = {
        "q_version": "Q1-q_huber-v1",
        "q_huber_point": point,
        "q_huber_ci95_low": low,
        "q_huber_ci95_high": high,
        "q_equal_sensitivity": equal,
        "q_huber_ci_repetitions": int(qhuber["bootstrap_repetitions"]),
    }
    return branches, metadata


def quality_cost_rows(branches: list[dict[str, Any]], q2_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    d_values = sorted({float(row["D_tokens_B"]) for row in q2_rows})
    if len(q2_rows) != 1176 or len(d_values) != 147:
        raise ValueError("Expected 1,176 B1 grid points and 147 unique D values")
    d_refs = {
        "min": int(Fraction(str(d_values[0])) * 10**9),
        "median": int(Fraction(str(d_values[len(d_values) // 2])) * 10**9),
        "max": int(Fraction(str(d_values[-1])) * 10**9),
    }
    rows: list[dict[str, Any]] = []
    for branch in branches:
        q0 = branch["q0"]
        targets = sorted({q0, *[q for q in COMMON_TARGETS if q >= q0]})
        for g_name, g in cost_module.G_FUNCTIONS.items():
            g0 = g(q0)
            for target in targets:
                delta_per_token = max(0.0, g(target) - g0)
                row: dict[str, Any] = {
                    "q0_branch": branch["branch"],
                    "q0_basis": branch["basis"],
                    "q0": q0,
                    "g_function": g_name,
                    "Q_target": target,
                    "incremental_cost_flops_per_token": delta_per_token,
                }
                for label, d_tokens in d_refs.items():
                    row[f"D_B1_{label}_tokens"] = d_tokens
                    row[f"C_Q_at_D_B1_{label}_flops"] = delta_per_token * d_tokens
                rows.append(row)
    return rows


def build_points(grid_rows: list[dict[str, str]], params_file: dict[str, Any]) -> list[dict[str, Any]]:
    if params_file.get("fit_scope") != "B1 only" or len(grid_rows) != 1176:
        raise ValueError("Expected frozen B1-only M0 and 1,176 observed grid points")
    parameters = {key: float(value) for key, value in params_file["parameters"].items()}
    points = []
    seen = set()
    for row in grid_rows:
        run_id = row["run_id"]
        if run_id in seen:
            raise ValueError(f"Duplicate run_id: {run_id}")
        seen.add(run_id)
        n_b, d_b = float(row["N_params_B"]), float(row["D_tokens_B"])
        pred = (
            parameters["E"] + parameters["A"] * n_b ** (-parameters["alpha"])
            + parameters["B"] * d_b ** (-parameters["beta"])
        )
        if abs(pred - float(row["predicted_val_loss"])) > 1e-8:
            raise ValueError(f"Frozen M0 prediction mismatch for {run_id}")
        points.append({
            "run_id": run_id,
            "n_b": n_b,
            "d_b": d_b,
            "n_abs": int(Fraction(row["N_params_B"]) * 10**9),
            "d_abs": int(Fraction(row["D_tokens_B"]) * 10**9),
            "loss": pred,
        })
    return points


def first_reallocation_threshold(
    q0: float,
    g_name: str,
    scenario: dict[str, str],
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    budget = int(scenario["budget_flops"])
    context = int(scenario["context_length_tokens"])
    feasible = []
    base_costs: dict[str, Fraction] = {}
    for point in points:
        nd = point["n_abs"] * point["d_abs"]
        base_cost = Fraction(TRAINING_COEFFICIENT * nd) + ETA * nd * context
        if base_cost <= budget:
            feasible.append(point)
            base_costs[point["run_id"]] = base_cost
    if len(feasible) != int(scenario["b1_feasible_grid_points"]):
        raise ValueError(f"Feasible-grid count mismatch for C={budget}, context={context}")
    baseline = min(
        feasible,
        key=lambda point: (
            point["loss"], base_costs[point["run_id"]], point["n_b"], point["d_b"], point["run_id"]
        ),
    )
    if baseline["run_id"] != scenario["selected_run_id"]:
        raise ValueError(f"Fixed-Q baseline mismatch for C={budget}, context={context}")
    lines = [{
        "slope": 0.0,
        "intercept": baseline["loss"],
        "delta_q": 0.0,
        "q": q0,
        "point": baseline,
        "total_cost": float(base_costs[baseline["run_id"]]),
        "run_id": baseline["run_id"],
    }]
    g = cost_module.G_FUNCTIONS[g_name]
    for point in feasible:
        nd = point["n_abs"] * point["d_abs"]
        base_cost = Fraction(TRAINING_COEFFICIENT * nd) + ETA * nd * context
        remaining = float(Fraction(budget) - base_cost)
        q_max = conditional_module.max_affordable_q(g_name, q0, point["d_abs"], remaining)
        delta_q = max(0.0, q_max - q0)
        q_cost = point["d_abs"] * max(0.0, g(q_max) - g(q0))
        total_cost = float(base_cost) + q_cost
        if total_cost > budget + max(1.0, budget) * 1e-12:
            raise ValueError("Q0 sensitivity candidate exceeds its FLOPs budget")
        lines.append({
            "slope": -delta_q,
            "intercept": point["loss"],
            "delta_q": delta_q,
            "q": q_max,
            "point": point,
            "total_cost": total_cost,
            "run_id": point["run_id"],
        })
    segments = conditional_module.lower_envelope(lines)
    first = next(((start, line) for start, _, line in segments if line["run_id"] != baseline["run_id"]), None)
    return {
        "beta_Q_first_ND_reallocation": first[0] if first else None,
        "first_reallocated_run_id": first[1]["run_id"] if first else None,
        "first_reallocated_N_B": first[1]["point"]["n_b"] if first else None,
        "first_reallocated_D_B": first[1]["point"]["d_b"] if first else None,
        "first_reallocated_Q": first[1]["q"] if first else None,
        "conditional_envelope_segments": len(segments),
        "baseline_run_id": baseline["run_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small real-input P1 slice")
    parser.add_argument("--output-dir", help="Optional output directory relative to project root")
    args = parser.parse_args()
    output_dir = validate_output_dir(parser, args.smoke, args.output_dir)

    q1_freeze = read_json(cost_module.Q1_FREEZE)
    if q1_freeze.get("q_version") != "Q1-q_huber-v1" or q1_freeze.get("status") != "FROZEN_WITH_LIMITATIONS":
        raise ValueError("Expected the frozen Q1-q_huber-v1 interface")
    branches, qmeta = q0_branches(read_csv(Q1_MACRO))
    q0_main, q_equal, _ = cost_module.get_frozen_q0()
    if abs(q0_main - qmeta["q_huber_point"]) > 1e-10 or abs(q_equal - qmeta["q_equal_sensitivity"]) > 1e-10:
        raise ValueError("Q0 branches disagree with the shared frozen Q1 interface")

    grid_rows = read_csv(Q2_GRID)
    params_file = read_json(Q2_PARAMS)
    frontier_rows = read_csv(Q3_FRONTIER)
    existing_beta_rows = read_csv(Q3_EXISTING_BETA)
    points = build_points(grid_rows, params_file)

    cost_rows = quality_cost_rows(branches, grid_rows)
    threshold_rows: list[dict[str, Any]] = []
    scenarios = frontier_rows
    g_functions = list(cost_module.G_FUNCTIONS)
    active_branches = branches
    if args.smoke:
        scenarios = frontier_rows[:1]
        g_functions = g_functions[:1]

    main_q0_beta = {
        (int(row["budget_flops"]), int(row["context_length_tokens"]), row["g_function"]): row
        for row in existing_beta_rows
    }
    for scenario in scenarios:
        budget = int(scenario["budget_flops"])
        context = int(scenario["context_length_tokens"])
        for g_name in g_functions:
            for branch in active_branches:
                result = first_reallocation_threshold(branch["q0"], g_name, scenario, points)
                row = {
                    "q0_branch": branch["branch"],
                    "q0_basis": branch["basis"],
                    "q0": branch["q0"],
                    "budget_flops": budget,
                    "context_length_tokens": context,
                    "g_function": g_name,
                    "beta_Q_units": "validation-loss units per one Q-score unit; beta_Q is unknown",
                    **result,
                    "claim_boundary": "conditional linear Q-only response; not estimated from Q2; p excluded",
                }
                if branch["branch"] == "q_huber_point":
                    old = main_q0_beta.get((budget, context, g_name))
                    if old is None:
                        raise ValueError("Existing Q3 beta report lacks matching baseline scenario")
                    old_threshold = old["beta_Q_first_ND_reallocation"]
                    new_threshold = result["beta_Q_first_ND_reallocation"]
                    if old_threshold in (None, ""):
                        if new_threshold is not None:
                            raise ValueError("Central Q0 threshold no longer matches the existing Q3 analysis")
                    elif new_threshold is None or abs(float(old_threshold) - new_threshold) > 1e-8:
                        raise ValueError(
                            f"Central Q0 threshold mismatch for C={budget}, context={context}, g={g_name}: "
                            f"new={new_threshold}, existing={old_threshold}"
                        )
                    old_run_id = old["first_reallocated_run_id"] or None
                    if result["first_reallocated_run_id"] != old_run_id:
                        raise ValueError(
                            f"Central Q0 reallocation run_id mismatch for C={budget}, context={context}, g={g_name}: "
                            f"new={result['first_reallocated_run_id']}, existing={old_run_id}, "
                            f"threshold={result['beta_Q_first_ND_reallocation']}"
                        )
                threshold_rows.append(row)

    cost_path = output_dir / OUT_COST
    beta_path = output_dir / OUT_BETA
    report_path = output_dir / OUT_REPORT
    manifest_path = output_dir / OUT_MANIFEST
    reproduction_command = "python Q3/03_代码/analyze_q3_q0_baseline_sensitivity.py"
    if args.smoke:
        reproduction_command += f' --smoke --output-dir "{args.output_dir}"'
    write_csv(cost_path, cost_rows)
    write_csv(beta_path, threshold_rows)

    qcost_at_one = []
    for branch in branches:
        relevant = [row for row in cost_rows if row["q0_branch"] == branch["branch"] and row["Q_target"] == 1.0]
        for row in relevant:
            qcost_at_one.append(row)
    beta_index = {
        (row["q0_branch"], row["g_function"], int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in threshold_rows
    }
    shared_main_scenarios = [row for row in threshold_rows if row["q0_branch"] == "q_huber_point"]
    report_lines = [
        "# Q3 的 Q0 基线敏感性：质量成本与条件重分配阈值",
        "",
        "## 分析口径",
        "",
        f"以 Q1 A1 七域宏平均 `q_huber` 点值 {qmeta['q_huber_point']:.10f} 为主基线；同时传播其 95% 区间端点 [{qmeta['q_huber_ci95_low']:.10f}, {qmeta['q_huber_ci95_high']:.10f}]，并将冻结的 `q_equal` {qmeta['q_equal_sensitivity']:.10f} 作为单独接口敏感性分支。`q_equal` 是替代口径，不是 `q_huber` 的置信区间端点。",
        "",
        "第一部分按题面三种 g(Q) 公式重算 Q 提升成本；第二部分在原有条件假设 L_cond=L_M0−β_Q(Q−Q0)、β_Q≥0 下重算 N/D 首次重分配阈值。β_Q 保持未知，Q0 敏感性不提供 Q→Loss 的经验估计。",
        "",
        "## Q=1 时的成本敏感性（B1 最大 D 支持点）",
        "",
        "| Q0 分支 | g(Q) | 增量成本 FLOPs |",
        "|---|---|---:|",
    ]
    for row in qcost_at_one:
        if row["D_B1_max_tokens"] == 299893000000:
            report_lines.append(
                f"| {row['q0_branch']} ({row['q0']:.6f}) | {row['g_function']} | {row['C_Q_at_D_B1_max_flops']:.6g} |"
            )
    if shared_main_scenarios:
        report_lines.extend([
            "",
            "## Q0 变化对条件 N/D 重分配阈值的影响",
            "",
            f"下表比较 q_huber 点值、区间端点和 q_equal 分支，在本次实际计算的 {len(scenarios)} 个预算×上下文情景和所列 g(Q) 下的首次 N/D 重分配 β_Q 阈值范围；它不是对 β_Q 的估计。",
            "",
            "| Q0 分支 | g(Q) | 首次 N/D 重分配 β_Q 最小值 | 最大值 | 无重分配情景数 |",
            "|---|---|---:|---:|---:|",
        ])
        for branch in branches:
            for g_name in g_functions if not args.smoke else list(cost_module.G_FUNCTIONS):
                subset = [
                    row for row in threshold_rows
                    if row["q0_branch"] == branch["branch"] and row["g_function"] == g_name
                ]
                finite = [float(row["beta_Q_first_ND_reallocation"]) for row in subset if row["beta_Q_first_ND_reallocation"] is not None]
                if not subset:
                    continue
                min_text = f"{min(finite):.6g}" if finite else "—"
                max_text = f"{max(finite):.6g}" if finite else "—"
                none_count = sum(row["beta_Q_first_ND_reallocation"] is None for row in subset)
                report_lines.append(f"| {branch['branch']} | {g_name} | {min_text} | {max_text} | {none_count} |")
    report_lines.extend([
        "",
        "## 解释边界",
        "",
        "`q_huber` 区间来自 Q1 A1 七域宏平均 Bootstrap；它描述该质量接口的 A1 汇总不确定性，不是 B1/Pythia 训练配方混合质量的区间。`q_equal` 作为另一冻结接口的敏感性比较，不应与抽样区间混为一谈。",
        "所有阈值仍依赖未识别的线性 β_Q 条件模型、Q2 M0 和 B1 支持网格；不得把它们写成 Q2 估计出的质量收益或经验 Q 最优。p 未进入目标函数。",
        "",
        f"成本明细见 `{OUT_COST.name}`；条件阈值明细见 `{OUT_BETA.name}`。",
        "",
        f"复现命令：`{reproduction_command}`。",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    inputs = [
        cost_module.Q1_FREEZE, cost_module.Q1_DOMAIN, Q1_MACRO,
        Q2_GRID, Q2_PARAMS, Q3_FRONTIER, Q3_EXISTING_BETA,
    ]
    outputs = [cost_path, beta_path, report_path]
    manifest = {
        "artifact": "Q3 Q0 baseline sensitivity for quality cost and conditional Q-only reallocation thresholds",
        "q0_branches": branches,
        "q0_metadata": qmeta,
        "conditional_response": "L_cond = L_M0(N,D) - beta_Q*(Q-Q0), beta_Q >= 0; beta_Q remains unknown",
        "new_q2_data_required": False,
        "no_empirical_q_response": True,
        "p_setting": "excluded from model and decisions; no p vector assigned",
        "bootstrap_repetitions_for_q_huber_ci": qmeta["q_huber_ci_repetitions"],
        "python_version": platform.python_version(),
        "reproduction_command": reproduction_command,
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {path.as_posix(): sha256(ROOT / path) for path in inputs},
        "outputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs},
        "claim_boundary": "Q1 A1 baseline sensitivity only; Q1 interval is not B1 mixture uncertainty; beta_Q conditional thresholds are not empirical Q effects",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cost_rows)} Q-cost rows and {len(threshold_rows)} conditional threshold rows to {output_dir}")
    print("Q1 q_huber interval and q_equal are separate baseline branches; beta_Q remains unknown.")


if __name__ == "__main__":
    main()
