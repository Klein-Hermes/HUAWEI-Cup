#!/usr/bin/env python3
"""Run Q3 quality-response transition scenarios from the supplied B6/B8 tables.

This is a conditional, source-separated sensitivity analysis. It holds the
frozen B1 M0 coefficients and observed N/D grid fixed, then applies the B6
semi-synthetic Q exponent or separate B8 calibrated/extrapolated exponents.
It does not estimate a real joint p/Q effect and does not assign a p vector.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import analyze_q3_cost_sensitivities as cost_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("Q3/04_结果/Qp条件转移敏感性_20260926")
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q2_B6_FIT = Path("Q2/03_结果/完整回答补充/v1/b6_b7_q_model_supplement.csv")
Q2_B6_ROWS = Path("中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv")
Q2_B8_ROWS = Path("中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
SEED = 260926
BOOTSTRAPS = 1000
TRAINING_COEFFICIENT = 6
ETA = Fraction(1, 5000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows and path.name != "q3_qp_adjacent_transitions.csv":
        raise ValueError(f"Refusing to write an empty table: {path}")
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        empty_transition_fields = [
            "q_response_source", "transition_type", "left_budget_flops", "left_context_tokens",
            "right_budget_flops", "right_context_tokens", "N_changed", "D_changed", "Q_changed",
            "left_N_B", "right_N_B", "left_D_B", "right_D_B", "left_Q", "right_Q",
            "left_predicted_loss", "right_predicted_loss",
        ]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_transition_fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_log_slopes(rows: list[dict[str, str]], q0: float) -> list[float]:
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = (row["N_params_B"], row["D_tokens_B"])
        q, loss = float(row["Q_score"]), float(row["val_loss"])
        if q <= 0 or loss <= 0:
            raise ValueError(f"Log slope requires positive Q and loss: {row}")
        groups.setdefault(key, []).append((np.log(q / q0), np.log(loss)))
    slopes = []
    for observations in groups.values():
        if len(observations) < 2 or len({x for x, _ in observations}) < 2:
            continue
        x, y = np.asarray(observations, dtype=float).T
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    if not slopes:
        raise ValueError("No matched N,D groups have enough Q levels")
    return slopes


def slope_summary(dataset: str, rows: list[dict[str, str]], q0: float,
                  rng: np.random.Generator) -> dict[str, Any]:
    slopes = grouped_log_slopes(rows, q0)
    draw = np.asarray([np.median(rng.choice(slopes, size=len(slopes), replace=True))
                       for _ in range(BOOTSTRAPS)])
    median_slope = float(np.median(slopes))
    return {
        "dataset": dataset,
        "rows": len(rows),
        "matched_ND_groups": len(slopes),
        "median_log_loss_slope_per_log_Q": median_slope,
        "kappa_from_matched_slope": -median_slope,
        "kappa_cluster_bootstrap_95_low": -float(np.quantile(draw, 0.975)),
        "kappa_cluster_bootstrap_95_high": -float(np.quantile(draw, 0.025)),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_unit": "matched (N,D) cell",
    }


def max_affordable_q(name: str, q0: float, d_tokens: int, remaining: float) -> float:
    if remaining <= 0:
        return q0
    g = cost_module.G_FUNCTIONS[name]
    if d_tokens * max(0.0, g(1.0) - g(q0)) <= remaining:
        return 1.0
    low, high = q0, 1.0
    for _ in range(90):
        mid = (low + high) / 2
        if d_tokens * max(0.0, g(mid) - g(q0)) <= remaining:
            low = mid
        else:
            high = mid
    return low


def main() -> None:
    global OUT, BOOTSTRAPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one real budget/context with fewer bootstrap draws")
    args = parser.parse_args()
    if args.smoke:
        OUT = OUT / "_p1_smoke"
        BOOTSTRAPS = 100
    q0, _, q_meta = cost_module.get_frozen_q0()
    params_file = read_json(Q2_PARAMS)
    if params_file.get("fit_scope") != "B1 only":
        raise ValueError("Expected the frozen B1-only M0 parameters")
    p = {k: float(v) for k, v in params_file["parameters"].items()}
    b1 = read_csv(Q2_GRID)
    frontier = read_csv(Q3_FRONTIER)
    if args.smoke:
        frontier = frontier[:1]
    expected_scenarios = 1 if args.smoke else 15
    if len(b1) != 1176 or len(frontier) != expected_scenarios:
        raise ValueError(f"Expected 1,176 B1 points and {expected_scenarios} frontier scenarios; got {len(b1)}, {len(frontier)}")
    if len({r["run_id"] for r in b1}) != len(b1):
        raise ValueError("B1 run_id is not unique")

    b6_rows = read_csv(Q2_B6_ROWS)
    b8_all = read_csv(Q2_B8_ROWS)
    b8_cal = [r for r in b8_all if r["data_type"] == "calibrated"]
    b8_extra = [r for r in b8_all if r["data_type"] == "extrapolated"]
    b6_fit = [r for r in read_csv(Q2_B6_FIT) if r["dataset"] == "B6" and r["model"] == "M_Q_NDQ"]
    if len(b6_rows) != 360 or len(b6_fit) != 1 or len(b8_cal) != 984 or len(b8_extra) != 720:
        raise ValueError("B6/B8 row counts or B6 model calibration do not match the audited inputs")

    rng = np.random.default_rng(SEED)
    response_rows = [
        slope_summary("B6_semi_synthetic", b6_rows, q0, rng),
        slope_summary("B8_calibrated", b8_cal, q0, rng),
        slope_summary("B8_extrapolated", b8_extra, q0, rng),
    ]
    kappa_b6 = float(b6_fit[0]["kappa_Q_exponent"])
    # Keep the already fitted B6 exponent as the favorable calibration. For B8,
    # convert within-(N,D) log-loss slopes into the corresponding Q exponent.
    response_rows[0]["kappa_model_fit_used_for_optimization"] = kappa_b6
    response_rows[0]["optimization_basis"] = "precomputed B6 M_Q_NDQ; semi-synthetic"
    for row in response_rows[1:]:
        row["kappa_model_fit_used_for_optimization"] = row["kappa_from_matched_slope"]
        row["optimization_basis"] = "median matched-(N,D) log-loss/log-Q slope; B8 source kept separate"

    sources = [
        ("B6_semi_synthetic", kappa_b6),
        ("B8_calibrated", float(response_rows[1]["kappa_model_fit_used_for_optimization"])),
        ("B8_extrapolated", float(response_rows[2]["kappa_model_fit_used_for_optimization"])),
    ]
    points = []
    for row in b1:
        n_b, d_b = float(row["N_params_B"]), float(row["D_tokens_B"])
        n_abs = int(Fraction(row["N_params_B"]) * 10**9)
        d_abs = int(Fraction(row["D_tokens_B"]) * 10**9)
        base_loss = p["E"] + p["A"] * n_b ** (-p["alpha"]) + p["B"] * d_b ** (-p["beta"])
        if abs(base_loss - float(row["predicted_val_loss"])) > 2e-8:
            raise ValueError(f"Frozen M0 prediction mismatch for {row['run_id']}")
        points.append({"run_id": row["run_id"], "n_b": n_b, "d_b": d_b,
                       "n_abs": n_abs, "d_abs": d_abs, "base_loss": base_loss})

    scenario_rows = []
    for source, kappa in sources:
        if not np.isfinite(kappa):
            raise ValueError(f"Non-finite Q exponent for {source}")
        for sc in frontier:
            budget = int(sc["budget_flops"])
            context = int(sc["context_length_tokens"])
            candidates = []
            for point in points:
                nd = point["n_abs"] * point["d_abs"]
                train = TRAINING_COEFFICIENT * nd
                attention = float(ETA * nd * context)
                base_cost = train + attention
                if base_cost > budget:
                    continue
                q_max = max_affordable_q("exponential", q0, point["d_abs"], budget - base_cost)
                q = q_max if kappa > 0 else q0
                q_cost = point["d_abs"] * max(0.0, cost_module.G_FUNCTIONS["exponential"](q)
                                                - cost_module.G_FUNCTIONS["exponential"](q0))
                total = base_cost + q_cost
                factor = (q / q0) ** (-kappa)
                loss = p["E"] + p["A"] * point["n_b"] ** (-p["alpha"])
                loss += p["B"] * point["d_b"] ** (-p["beta"]) * factor
                if total > budget + max(1.0, budget) * 1e-12:
                    raise ValueError("Selected point exceeds its FLOPs budget")
                candidates.append((loss, total, point["n_b"], point["d_b"], point["run_id"],
                                   q, train, q_cost, attention, point))
            if len(candidates) != int(sc["b1_feasible_grid_points"]):
                raise ValueError("Feasible candidate count disagrees with the frozen M0 frontier")
            best = min(candidates, key=lambda x: x[:5])
            base = next((x for x in candidates if x[4] == sc["selected_run_id"]), None)
            if base is None:
                # For B8 response coefficients <= 0, Q=Q0 and the selected run
                # must reproduce the frozen M0 selection exactly.
                raise ValueError("Frozen M0 selected run is missing from feasible candidates")
            scenario_rows.append({
                "q_response_source": source,
                "kappa_Q_exponent": kappa,
                "budget_flops": budget,
                "context_length_tokens": context,
                "selected_run_id": best[4],
                "selected_N_B": best[2],
                "selected_D_B": best[3],
                "selected_Q": best[5],
                "predicted_loss": best[0],
                "m0_run_id": sc["selected_run_id"],
                "m0_N_B": float(sc["selected_N_B"]),
                "m0_D_B": float(sc["selected_D_B"]),
                "m0_predicted_loss": float(sc["m0_predicted_val_loss"]),
                "delta_N_B_vs_M0": best[2] - float(sc["selected_N_B"]),
                "delta_D_B_vs_M0": best[3] - float(sc["selected_D_B"]),
                "delta_Q_vs_Q0": best[5] - q0,
                "delta_loss_vs_M0": best[0] - float(sc["m0_predicted_val_loss"]),
                "C_train_flops": int(best[6]),
                "C_Q_flops": float(best[7]),
                "C_attn_flops": best[8],
                "C_total_flops": best[1],
                "budget_utilization": best[1] / budget,
                "feasible_B1_grid_points": len(candidates),
                "N_D_changed_vs_M0": best[4] != sc["selected_run_id"],
                "at_Q0_boundary": abs(best[5] - q0) < 1e-10,
                "interpretation": "conditional source-specific scenario; not empirical joint p/Q optimum",
            })

    # Enumerate adjacent budget/context comparisons separately by source.
    transition_rows = []
    by_key = {(r["q_response_source"], r["budget_flops"], r["context_length_tokens"]): r
              for r in scenario_rows}
    budgets = sorted({int(r["budget_flops"]) for r in scenario_rows})
    contexts = sorted({int(r["context_length_tokens"]) for r in scenario_rows})
    for source, _ in sources:
        pairs = []
        for context in contexts:
            for left, right in zip(budgets, budgets[1:]):
                pairs.append(("adjacent_budget", left, context, right, context))
        for budget in budgets:
            for left, right in zip(contexts, contexts[1:]):
                pairs.append(("adjacent_context", budget, left, budget, right))
        for kind, lb, lc, rb, rc in pairs:
            a = by_key[(source, lb, lc)]
            b = by_key[(source, rb, rc)]
            transition_rows.append({
                "q_response_source": source,
                "transition_type": kind,
                "left_budget_flops": lb,
                "left_context_tokens": lc,
                "right_budget_flops": rb,
                "right_context_tokens": rc,
                "N_changed": a["selected_run_id"] != b["selected_run_id"] and a["selected_N_B"] != b["selected_N_B"],
                "D_changed": a["selected_D_B"] != b["selected_D_B"],
                "Q_changed": abs(a["selected_Q"] - b["selected_Q"]) > 1e-10,
                "left_N_B": a["selected_N_B"], "right_N_B": b["selected_N_B"],
                "left_D_B": a["selected_D_B"], "right_D_B": b["selected_D_B"],
                "left_Q": a["selected_Q"], "right_Q": b["selected_Q"],
                "left_predicted_loss": a["predicted_loss"],
                "right_predicted_loss": b["predicted_loss"],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "q_response_exponent_estimates.csv", response_rows)
    write_csv(OUT / "q3_qp_conditional_scenarios.csv", scenario_rows)
    write_csv(OUT / "q3_qp_adjacent_transitions.csv", transition_rows)

    summaries = []
    for source, kappa in sources:
        rows = [r for r in scenario_rows if r["q_response_source"] == source]
        trans = [r for r in transition_rows if r["q_response_source"] == source]
        summaries.append({
            "source": source, "kappa": kappa,
            "scenarios": len(rows),
            "scenarios_Q_above_Q0": sum(not r["at_Q0_boundary"] for r in rows),
            "scenarios_ND_reselected_vs_M0": sum(r["N_D_changed_vs_M0"] for r in rows),
            "adjacent_transitions_with_N_change": sum(r["N_changed"] for r in trans),
            "adjacent_transitions_with_D_change": sum(r["D_changed"] for r in trans),
            "adjacent_transitions_with_Q_change": sum(r["Q_changed"] for r in trans),
            "min_Q": min(r["selected_Q"] for r in rows),
            "max_Q": max(r["selected_Q"] for r in rows),
            "min_delta_loss_vs_M0": min(r["delta_loss_vs_M0"] for r in rows),
            "max_delta_loss_vs_M0": max(r["delta_loss_vs_M0"] for r in rows),
        })

    report_path = ROOT / OUT / "report.md"
    lines = [
        "# Q/p 条件响应下的 Q3 资源转移敏感性",
        "",
        "日期：2026-09-26。范围：基于本地附件的条件情景计算，不是新的模型训练实验。",
        "",
        "## 计算口径",
        "",
        f"- 冻结 B1 M0 参数与 1,176 个实测 (N,D) 网格；Q1 q_huber 参考值 Q0={q0:.10f}。",
        "- 目标函数：L=E+A N^(-alpha)+B D^(-beta)(Q/Q0)^(-kappa)，N/D 系数固定为 B1 M0。",
        "- 成本：沿用题面指数型 g(Q)，总成本含训练、质量与注意力三项；Q 搜索区间 [Q0,1]。",
        "- B6 使用已报告的半合成 M_Q_NDQ 指数；B8 calibrated/extrapolated 分开用匹配 (N,D) 单元的 log(Loss)-log(Q/Q0) 中位斜率换算指数。",
        "- B8 系数不与 B6 合并。B6/B7 不作独立重复，且本分析固定 p，不将跨来源 p 系数接入 B1。",
        "",
        "## Q 响应估计审计",
        "",
        "| 来源 | 匹配单元 | 中位 log Loss/log Q 斜率 | kappa（优化使用） | cluster bootstrap 95% 区间（描述性） |",
        "|---|---:|---:|---:|---:|",
    ]
    for r, (_, used_kappa) in zip(response_rows, sources):
        lines.append(f"| {r['dataset']} | {r['matched_ND_groups']} | {r['median_log_loss_slope_per_log_Q']:.6g} | {used_kappa:.6g} | [{r['kappa_cluster_bootstrap_95_low']:.6g}, {r['kappa_cluster_bootstrap_95_high']:.6g}] |")
    lines += ["", "注：B6 优化用已拟合 M_Q_NDQ 的 kappa；其表列匹配斜率/区间是独立的描述性核对。B8 bootstrap 以固定 (N,D) 单元为重抽样单位。", "",
              "## 转移情景摘要", "",
              "| Q 响应来源 | 情景数 | Q>Q0 情景 | 相对 M0 重选 N/D | 相邻情景 N 改变 | D 改变 | Q 改变 | Q 范围 | ΔLoss 相对 M0 范围 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['source']} | {s['scenarios']} | {s['scenarios_Q_above_Q0']} | {s['scenarios_ND_reselected_vs_M0']} | {s['adjacent_transitions_with_N_change']} | {s['adjacent_transitions_with_D_change']} | {s['adjacent_transitions_with_Q_change']} | [{s['min_Q']:.4g}, {s['max_Q']:.4g}] | [{s['min_delta_loss_vs_M0']:.5g}, {s['max_delta_loss_vs_M0']:.5g}] |")
    lines += ["", "## 结论与边界", "",
              "1. 该计算把现有半合成/校准质量响应映射到 B1 的 M0 支持网格，回答‘若采用该来源的 Q 响应，预算配置会怎样变化’。",
              "2. 若 B6 与 B8 的指数符号或配置不同，资源转移方向不具来源稳健性；应报告情景包络，不能声称唯一的经验最优。",
              "3. p 固定为 B1 的共同数据政策条件，但其 17 维数值未观测；A4/A5 的 p 系数不并入 B1。",
              "4. 因而本结果推进了转移情景分析，但没有识别真实训练中的独立 p、Q 边际效应，也没有替代受控训练实验。",
              "5. 逐情景结果见 `q3_qp_conditional_scenarios.csv`；22 对相邻情景/来源结果见 `q3_qp_adjacent_transitions.csv`；指数与区间见 `q_response_exponent_estimates.csv`。",
              "", "复现命令：`D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/solve_q3_qp_conditioned_transition.py`。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    inputs = [Q2_GRID, Q2_PARAMS, Q2_B6_FIT, Q2_B6_ROWS, Q2_B8_ROWS, Q3_FRONTIER,
              Path("Q1/03_结果/Q1_最终收口/q1_final_freeze.json"),
              Path("Q1/03_结果/Q1_最终收口/q1_final_domain_quality.csv"),
              Path("Q1/03_结果/Q1.1/v1/a1_macro_summary.csv")]
    outputs = [OUT / "q_response_exponent_estimates.csv", OUT / "q3_qp_conditional_scenarios.csv",
               OUT / "q3_qp_adjacent_transitions.csv", OUT / "report.md"]
    manifest = {
        "artifact": "Q3 source-separated Q/p conditional transition sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/solve_q3_qp_conditioned_transition.py",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAPS,
        "quality_reference": {**q_meta, "Q0": q0},
        "scope": "conditional Q scenario; fixed p; fixed B1 M0 coefficients; no empirical joint p/Q identification",
        "source_calibrations": {name: kappa for name, kappa in sources},
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "output_sha256": {str(path): sha256(path) for path in outputs},
        "row_counts": {"B1_grid": len(b1), "frontier_scenarios": len(frontier),
                       "Q_response_estimates": len(response_rows), "conditional_scenarios": len(scenario_rows),
                       "adjacent_transitions": len(transition_rows)},
        "checks": {"B1_M0_prediction_reconciled": True, "budget_feasibility_checked": True,
                   "B8_sources_kept_separate": True, "p_vector_not_fabricated": True},
    }
    (ROOT / OUT / "reproduction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "response": response_rows, "summary": summaries,
                      "rows": manifest["row_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
