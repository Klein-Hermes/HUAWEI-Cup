#!/usr/bin/env python3
"""Audit discrete Q3 allocation shifts under paired Q2 trajectory bootstrap.

Scope: fixed Q=Q0, the frozen B1/M0 loss surface, and its 1,176 observed
N/D points. This is a scenario-transition audit, not a full Q/p optimizer,
causal claim, or validation outside the B1 support grid.

Run from the repository root:
    python Q3/03_代码/assess_q3_m0_discrete_transitions.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q2_BOOTSTRAP = Path("Q2/03_结果/经典ScalingLaw基线/v1/cluster_bootstrap_estimates.csv")
Q2_BOOTSTRAP_META = Path("Q2/03_结果/经典ScalingLaw基线/v1/cluster_bootstrap_intervals.json")
Q2_TRAJECTORIES = Path("Q2/03_结果/经典ScalingLaw基线/v1/gate0_b1_trajectory_mapping.csv")
Q3_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
OUT_PAIRS = Path("Q3/04_结果/M0_discrete_transition_pairs.csv")
OUT_DRAWS = Path("Q3/04_结果/M0_discrete_transition_bootstrap_draws.csv")
OUT_REPORT = Path("Q3/04_结果/M0_discrete_transition_report.md")
OUT_MANIFEST = Path("Q3/04_结果/M0_discrete_transition_manifest.json")
ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
EXPECTED_REPLICATES = 1000


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


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fraction_quantile(values: list[Fraction], q: Fraction) -> Fraction:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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


def read_parameters(parameters: dict[str, Any]) -> dict[str, float]:
    result = {name: float(parameters[name]) for name in ("E", "A", "alpha", "B", "beta")}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("M0 parameters must be finite")
    if not (result["E"] >= 0 and all(result[name] > 0 for name in ("A", "alpha", "B", "beta"))):
        raise ValueError(f"M0 parameters violate the frozen model contract: {result}")
    return result


def predict_loss(point: dict[str, Any], parameters: dict[str, float]) -> float:
    return (
        parameters["E"]
        + parameters["A"] * point["n_b"] ** (-parameters["alpha"])
        + parameters["B"] * point["d_b"] ** (-parameters["beta"])
    )


def cost_components(point: dict[str, Any], context: int) -> tuple[Fraction, Fraction, Fraction]:
    nd = point["n_abs"] * point["d_abs"]
    training = Fraction(TRAINING_COEFFICIENT * nd)
    attention = ETA * nd * context
    return training, attention, training + attention


def build_pair_specs(frontier_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scenarios = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in frontier_rows
    }
    budgets = sorted({budget for budget, _ in scenarios})
    contexts = sorted({context for _, context in scenarios})
    pairs: list[dict[str, Any]] = []
    for context in contexts:
        for left_budget, right_budget in zip(budgets, budgets[1:]):
            pairs.append({
                "pair_type": "adjacent_budget",
                "left_budget": left_budget,
                "right_budget": right_budget,
                "left_context": context,
                "right_context": context,
            })
    for budget in budgets:
        for left_context, right_context in zip(contexts, contexts[1:]):
            pairs.append({
                "pair_type": "adjacent_context",
                "left_budget": budget,
                "right_budget": budget,
                "left_context": left_context,
                "right_context": right_context,
            })
    for index, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"T{index:02d}"
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small real-input P1 slice")
    parser.add_argument("--output-dir", help="Optional output directory relative to project root")
    args = parser.parse_args()
    output_dir = validate_output_dir(parser, args.smoke, args.output_dir)

    grid_rows = read_csv(Q2_GRID)
    params_file = read_json(Q2_PARAMS)
    bootstrap_rows = read_csv(Q2_BOOTSTRAP)
    bootstrap_meta = read_json(Q2_BOOTSTRAP_META)
    trajectory_rows = read_csv(Q2_TRAJECTORIES)
    frontier_rows = read_csv(Q3_FRONTIER)
    if params_file.get("fit_scope") != "B1 only":
        raise ValueError("Expected the frozen B1-only Q2 M0 fit")
    if len(grid_rows) != 1176 or len(frontier_rows) != 15:
        raise ValueError("Expected 1,176 observed B1 points and 15 Q3 budget/context scenarios")
    if (
        len(bootstrap_rows) != EXPECTED_REPLICATES
        or bootstrap_meta.get("requested_replicates") != EXPECTED_REPLICATES
        or bootstrap_meta.get("successful_replicates") != EXPECTED_REPLICATES
        or bootstrap_meta.get("failed_replicates") != 0
        or bootstrap_meta.get("cluster_count") != 8
        or len(trajectory_rows) != 8
    ):
        raise ValueError("Expected 1,000 successful trajectory-cluster fits over 8 B1 trajectories")

    points: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for row in grid_rows:
        run_id = row["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate B1 run_id: {run_id}")
        run_ids.add(run_id)
        n_b = float(row["N_params_B"])
        d_b = float(row["D_tokens_B"])
        if not (math.isfinite(n_b) and math.isfinite(d_b) and n_b > 0 and d_b > 0):
            raise ValueError(f"Invalid N/D for run_id={run_id}")
        points.append({
            "run_id": run_id,
            "n_b": n_b,
            "d_b": d_b,
            "n_abs": int(Fraction(row["N_params_B"]) * 10**9),
            "d_abs": int(Fraction(row["D_tokens_B"]) * 10**9),
        })

    valid_clusters = {row["model_repo"] for row in trajectory_rows}
    if len(valid_clusters) != 8:
        raise ValueError("B1 trajectory mapping must contain 8 unique model_repo clusters")
    for fit in bootstrap_rows:
        sampled = fit["sampled_cluster_sequence"].split(";")
        if len(sampled) != 8 or not set(sampled).issubset(valid_clusters):
            raise ValueError(f"Invalid cluster sample in bootstrap replicate {fit['replicate']}")

    scenarios: dict[tuple[int, int], dict[str, Any]] = {}
    nominal_parameters = read_parameters(params_file["parameters"])
    for row in frontier_rows:
        budget = int(row["budget_flops"])
        context = int(row["context_length_tokens"])
        feasible = []
        for point in points:
            training, attention, total = cost_components(point, context)
            if total <= budget:
                feasible.append((point, training, attention, total))
        if len(feasible) != int(row["b1_feasible_grid_points"]):
            raise ValueError(f"Feasible-grid count mismatch for C={budget}, context={context}")
        nominal_best = min(
            feasible,
            key=lambda item: (
                predict_loss(item[0], nominal_parameters), item[3], item[0]["n_b"],
                item[0]["d_b"], item[0]["run_id"],
            ),
        )
        if nominal_best[0]["run_id"] != row["selected_run_id"]:
            raise ValueError(f"Nominal M0 choice disagrees with Q3 frontier for C={budget}, context={context}")
        scenarios[(budget, context)] = {"feasible": feasible, "frontier": row}

    pair_specs = build_pair_specs(frontier_rows)
    fits = bootstrap_rows
    if args.smoke:
        fits = bootstrap_rows[:20]
        pair_specs = pair_specs[:1]
    pair_csv = output_dir / OUT_PAIRS
    draw_csv = output_dir / OUT_DRAWS
    report_md = output_dir / OUT_REPORT
    manifest_json = output_dir / OUT_MANIFEST
    reproduction_command = "python Q3/03_代码/assess_q3_m0_discrete_transitions.py"
    if args.smoke:
        reproduction_command += f' --smoke --output-dir "{args.output_dir}"'

    draw_rows: list[dict[str, Any]] = []
    pair_groups: dict[str, list[dict[str, Any]]] = {pair["pair_id"]: [] for pair in pair_specs}
    scenarios_to_select = {
        (pair["left_budget"], pair["left_context"])
        for pair in pair_specs
    } | {
        (pair["right_budget"], pair["right_context"])
        for pair in pair_specs
    }

    for fit in fits:
        replicate = int(fit["replicate"])
        parameters = read_parameters(fit)
        selected: dict[tuple[int, int], dict[str, Any]] = {}
        for key in scenarios_to_select:
            candidate_rows = scenarios[key]["feasible"]
            best = min(
                candidate_rows,
                key=lambda item: (
                    predict_loss(item[0], parameters), item[3], item[0]["n_b"],
                    item[0]["d_b"], item[0]["run_id"],
                ),
            )
            selected[key] = {
                "point": best[0],
                "training": best[1],
                "attention": best[2],
                "total": best[3],
                "loss": predict_loss(best[0], parameters),
            }

        for pair in pair_specs:
            left_key = (pair["left_budget"], pair["left_context"])
            right_key = (pair["right_budget"], pair["right_context"])
            left, right = selected[left_key], selected[right_key]
            lp, rp = left["point"], right["point"]
            left_share = left["training"] / left["total"]
            right_share = right["training"] / right["total"]
            delta_share = right_share - left_share
            tv_share = abs(delta_share)  # two cost components sum to one
            row = {
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "bootstrap_replicate": replicate,
                "left_budget_flops": pair["left_budget"],
                "right_budget_flops": pair["right_budget"],
                "left_context_tokens": pair["left_context"],
                "right_context_tokens": pair["right_context"],
                "left_run_id": lp["run_id"],
                "right_run_id": rp["run_id"],
                "configuration_changed": lp["run_id"] != rp["run_id"],
                "left_N_B": lp["n_b"],
                "right_N_B": rp["n_b"],
                "delta_log_N": math.log(rp["n_b"] / lp["n_b"]),
                "left_D_B": lp["d_b"],
                "right_D_B": rp["d_b"],
                "delta_log_D": math.log(rp["d_b"] / lp["d_b"]),
                "left_predicted_loss": left["loss"],
                "right_predicted_loss": right["loss"],
                "delta_predicted_loss": right["loss"] - left["loss"],
                "left_training_cost_share": float(left_share),
                "right_training_cost_share": float(right_share),
                "delta_training_cost_share": float(delta_share),
                "total_variation_cost_share": float(tv_share),
                "left_budget_utilization": float(left["total"] / pair["left_budget"]),
                "right_budget_utilization": float(right["total"] / pair["right_budget"]),
            }
            draw_rows.append(row)
            pair_groups[pair["pair_id"]].append({**row, "_tv_exact": tv_share})

    summary_rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        rows = pair_groups[pair["pair_id"]]
        change_frequency = sum(bool(row["configuration_changed"]) for row in rows) / len(rows)
        delta_log_n = [float(row["delta_log_N"]) for row in rows]
        delta_log_d = [float(row["delta_log_D"]) for row in rows]
        delta_train_share = [float(row["delta_training_cost_share"]) for row in rows]
        tv_exact = [row["_tv_exact"] for row in rows]
        tv_p05 = fraction_quantile(tv_exact, Fraction(5, 100))
        cost_share_change_frequency = sum(value > 0 for value in tv_exact) / len(tv_exact)
        n_ci = (quantile(delta_log_n, 0.025), quantile(delta_log_n, 0.975))
        d_ci = (quantile(delta_log_d, 0.025), quantile(delta_log_d, 0.975))
        share_ci = (quantile(delta_train_share, 0.025), quantile(delta_train_share, 0.975))
        robust_allocation_change = change_frequency >= 0.95
        robust_cost_mix_change = cost_share_change_frequency >= 0.95
        if robust_allocation_change and robust_cost_mix_change:
            interpretation = "Bootstrap稳健的N/D配置与成本构成改变"
        elif robust_allocation_change:
            interpretation = "Bootstrap稳健的N/D配置改变"
        elif robust_cost_mix_change:
            interpretation = "成本构成改变；未达N/D配置稳健切换门槛"
        else:
            interpretation = "未见Bootstrap稳健的配置或成本构成改变"
        summary_rows.append({
            "pair_id": pair["pair_id"],
            "pair_type": pair["pair_type"],
            "left_budget_flops": pair["left_budget"],
            "right_budget_flops": pair["right_budget"],
            "left_context_tokens": pair["left_context"],
            "right_context_tokens": pair["right_context"],
            "paired_bootstrap_replicates": len(rows),
            "configuration_change_frequency": change_frequency,
            "delta_log_N_p2_5": n_ci[0],
            "delta_log_N_median": quantile(delta_log_n, 0.5),
            "delta_log_N_p97_5": n_ci[1],
            "delta_log_D_p2_5": d_ci[0],
            "delta_log_D_median": quantile(delta_log_d, 0.5),
            "delta_log_D_p97_5": d_ci[1],
            "delta_training_share_p2_5": share_ci[0],
            "delta_training_share_median": quantile(delta_train_share, 0.5),
            "delta_training_share_p97_5": share_ci[1],
            "cost_share_change_frequency": cost_share_change_frequency,
            "cost_share_TV_p05_exact": float(tv_p05),
            "cost_share_TV_median": quantile([float(value) for value in tv_exact], 0.5),
            "robust_ND_allocation_change": robust_allocation_change,
            "robust_cost_mix_change": robust_cost_mix_change,
            "interpretation": interpretation,
        })

    if args.smoke:
        scope = "P1 smoke slice: one adjacent-budget pair, 20 actual Q2 trajectory-bootstrap fits"
    else:
        scope = "full: 22 adjacent budget/context pairs, 1,000 actual Q2 trajectory-bootstrap fits"
    write_csv(pair_csv, summary_rows)
    write_csv(draw_csv, draw_rows)

    report_lines = [
        "# Q3 固定 Q 的 M0 离散资源转移审计",
        "",
        f"**计算范围：** {scope}。",
        "",
        "## 判据与方法",
        "",
        f"固定 Q=Q0，不估计 Q→Loss，也不纳入 p。对预算相邻档与上下文相邻档分别组成 {len(pair_specs)} 对情景；每对使用同一个 Q2 B1 轨迹 Bootstrap 复本进行配对比较，共计 {len(fits)} 个复本/情景对。每个复本都在满足题面成本约束的 B1 实测 1,176 点子集中重新选择 M0 预测 Loss 最低配置。",
        "",
        "将 N/D 网格点发生改变的 Bootstrap 频率至少 95% 定义为‘Bootstrap 稳健配置改变’。训练与注意力成本份额的总变差按精确有理数计算；至少 95% 配对复本出现非零总变差时，定义为‘Bootstrap 稳健成本构成改变’。同时报告总变差第 5 百分位数作为分布摘要，但不以它单独决定该标签。成本构成变化与 N/D 重新分配分开报告。这里不使用连续优化器的 KKT 活跃集判据，因为当前支路是有限候选网格枚举。",
        "",
        "## 逐对结果",
        "",
        "| 对比 | 左情景 | 右情景 | N/D配置改变频率 | Δlog N 中位数 [95%区间] | Δlog D 中位数 [95%区间] | 成本份额 TV 第5百分位数 | 判定摘要 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        left = f"{int(row['left_budget_flops']):.0e} FLOPs / {row['left_context_tokens']:,} tok"
        right = f"{int(row['right_budget_flops']):.0e} FLOPs / {row['right_context_tokens']:,} tok"
        n_text = f"{row['delta_log_N_median']:.5g} [{row['delta_log_N_p2_5']:.5g}, {row['delta_log_N_p97_5']:.5g}]"
        d_text = f"{row['delta_log_D_median']:.5g} [{row['delta_log_D_p2_5']:.5g}, {row['delta_log_D_p97_5']:.5g}]"
        report_lines.append(
            f"| {row['pair_id']} ({row['pair_type']}) | {left} | {right} | "
            f"{row['configuration_change_frequency']:.1%} | {n_text} | {d_text} | "
            f"{row['cost_share_TV_p05_exact']:.6g} | {row['interpretation']} |"
        )
    robust_count = sum(bool(row["robust_ND_allocation_change"]) for row in summary_rows)
    robust_mix_count = sum(bool(row["robust_cost_mix_change"]) for row in summary_rows)
    report_lines.extend([
        "",
        "## 解释边界",
        "",
        f"在本次分析的 {len(summary_rows)} 对情景中，{robust_count} 对达到 N/D 配置改变频率 ≥95% 的门槛，{robust_mix_count} 对达到成本份额非零变化频率 ≥95% 的门槛。配对 Bootstrap 复本数为 {len(fits)}；正式全量运行使用 1,000 次复本，但独立轨迹簇只有 8 条，故分位数用于稳定性提示。",
        "该结果是给定冻结 M0 形式和 B1 离散支持域下的情景敏感性结论，不代表因果效应、连续预算阈值或完整 Q3 的结构转移。最高预算若选择 B1 支持网格上界，结果仍受网格边界截断。Q、p 效应以及支持域外行为不在本分析范围。",
        "",
        f"配对逐复本结果见 `{OUT_DRAWS.name}`；汇总及可复现哈希见 `{OUT_PAIRS.name}` 与 `{OUT_MANIFEST.name}`。",
        "",
        f"复现命令：`{reproduction_command}`。",
        "",
    ])
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    input_paths = [Q2_GRID, Q2_PARAMS, Q2_BOOTSTRAP, Q2_BOOTSTRAP_META, Q2_TRAJECTORIES, Q3_FRONTIER]
    output_paths = [pair_csv, draw_csv, report_md]
    manifest = {
        "artifact": "Q3 fixed-Q M0 discrete scenario transition audit",
        "scope": scope,
        "fixed_q": "Q=Q0; no p or Q response in the objective",
        "pairing": "same Q2 trajectory-bootstrap replicate compared across adjacent scenario pairs",
        "decision_rule": {
            "robust_allocation_change": "selected N/D grid point differs in at least 95% of paired bootstrap replicates",
            "robust_cost_mix_change": "exact cost-share total variation is nonzero in at least 95% of paired bootstrap replicates",
            "cost_share_TV_p05": "reported descriptively; not used alone for classification",
            "continuous_KKT": "not used; finite grid enumeration",
        },
        "bootstrap_cluster_count": bootstrap_meta["cluster_count"],
        "bootstrap_replicates": len(fits),
        "python_version": platform.python_version(),
        "reproduction_command": reproduction_command,
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {path.as_posix(): sha256(ROOT / path) for path in input_paths},
        "outputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in output_paths},
        "claim_boundary": "discrete scenario sensitivity within frozen Q2 B1/M0 and observed support grid; not full Q3 optimum, causality, or external validation",
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_rows)} pair summaries and {len(draw_rows)} paired bootstrap rows to {output_dir}")
    print(f"Robust N/D configuration changes: {robust_count}/{len(summary_rows)}")


if __name__ == "__main__":
    main()
