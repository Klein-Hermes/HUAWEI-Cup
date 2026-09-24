#!/usr/bin/env python3
"""Propagate Q2's trajectory-cluster bootstrap fits through the Q3 M0 grid.

This is a sensitivity analysis of the restricted, fixed-Q B1/M0 reference
frontier. It does not add Q/p effects, independent validation, or a full Q3
joint optimum.

Run from the repository root:
    python Q3/03_代码/assess_q3_m0_cluster_bootstrap.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_BOOTSTRAP = Path("Q2/03_结果/经典ScalingLaw基线/v1/cluster_bootstrap_estimates.csv")
Q2_BOOTSTRAP_SUMMARY = Path("Q2/03_结果/经典ScalingLaw基线/v1/cluster_bootstrap_intervals.json")
Q2_TRAJECTORY_MAPPING = Path("Q2/03_结果/经典ScalingLaw基线/v1/gate0_b1_trajectory_mapping.csv")
BASELINE_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
SUMMARY_CSV = Path("Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv")
FREQUENCY_CSV = Path("Q3/04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv")
REPORT_MD = Path("Q3/04_结果/M0_ND_cluster_bootstrap_report.md")
MANIFEST_JSON = Path("Q3/04_结果/M0_cluster_bootstrap_sensitivity_manifest.json")
BOOTSTRAP_REPLICATES = 1000
ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(relative: Path) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(relative: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {relative}")
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def main() -> None:
    q2_grid_rows = read_csv(Q2_GRID)
    bootstrap_rows = read_csv(Q2_BOOTSTRAP)
    bootstrap_meta = json.loads((ROOT / Q2_BOOTSTRAP_SUMMARY).read_text(encoding="utf-8"))
    trajectory_rows = read_csv(Q2_TRAJECTORY_MAPPING)
    frontier_rows = read_csv(BASELINE_FRONTIER)
    if len(q2_grid_rows) != 1176:
        raise ValueError(f"Expected 1,176 B1 grid rows; got {len(q2_grid_rows)}")
    if (
        len(bootstrap_rows) != BOOTSTRAP_REPLICATES
        or bootstrap_meta.get("requested_replicates") != BOOTSTRAP_REPLICATES
        or bootstrap_meta.get("successful_replicates") != BOOTSTRAP_REPLICATES
        or bootstrap_meta.get("failed_replicates") != 0
        or bootstrap_meta.get("cluster_count") != 8
    ):
        raise ValueError("Expected all 1,000 successful Q2 cluster-bootstrap fits")
    if len(trajectory_rows) != 8:
        raise ValueError(f"Expected 8 B12-confirmed B1 trajectories; got {len(trajectory_rows)}")
    if len(frontier_rows) != 15:
        raise ValueError(f"Expected 15 baseline scenarios; got {len(frontier_rows)}")

    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    min_observed_loss = math.inf
    for row in q2_grid_rows:
        run_id = row["run_id"]
        if run_id in seen:
            raise ValueError(f"Duplicate B1 run_id: {run_id}")
        seen.add(run_id)
        n_b = float(row["N_params_B"])
        d_b = float(row["D_tokens_B"])
        min_observed_loss = min(min_observed_loss, float(row["observed_val_loss"]))
        if n_b <= 0 or d_b <= 0:
            raise ValueError(f"Nonpositive grid value for run_id={run_id}")
        n_abs = int(Fraction(row["N_params_B"]) * 10**9)
        d_abs = int(Fraction(row["D_tokens_B"]) * 10**9)
        points.append({"run_id": run_id, "n_b": n_b, "d_b": d_b, "n_abs": n_abs, "d_abs": d_abs})

    valid_clusters = {row["model_repo"] for row in trajectory_rows}
    if len(valid_clusters) != 8:
        raise ValueError("B12 trajectory mapping does not contain 8 unique model_repo clusters")
    bootstrap_ids = [int(row["replicate"]) for row in bootstrap_rows]
    if sorted(bootstrap_ids) != list(range(1, BOOTSTRAP_REPLICATES + 1)):
        raise ValueError("Bootstrap replicate IDs must be unique and span 1..1000")
    for fit in bootstrap_rows:
        sampled_clusters = fit["sampled_cluster_sequence"].split(";")
        if len(sampled_clusters) != 8 or not set(sampled_clusters).issubset(valid_clusters):
            raise ValueError(f"Invalid 8-trajectory sample in bootstrap replicate {fit['replicate']}")
        parameters = {name: float(fit[name]) for name in ("E", "A", "alpha", "B", "beta")}
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError(f"Nonfinite fit parameter in bootstrap replicate {fit['replicate']}")
        if not (0 <= parameters["E"] < min_observed_loss) or not all(
            parameters[name] > 0 for name in ("A", "alpha", "B", "beta")
        ):
            raise ValueError(f"Bootstrap fit violates Q2 parameter bounds in replicate {fit['replicate']}")
        sse = float(fit["sse"])
        if not math.isfinite(sse) or sse < 0:
            raise ValueError(f"Invalid SSE in bootstrap replicate {fit['replicate']}")

    scenarios = []
    for row in frontier_rows:
        context = int(row["context_length_tokens"])
        budget = int(row["budget_flops"])
        baseline_run = row["selected_run_id"]
        feasible = []
        for point in points:
            nd = point["n_abs"] * point["d_abs"]
            train = Fraction(TRAINING_COEFFICIENT * nd)
            attention = ETA * nd * context
            total = train + attention
            if total <= budget:
                feasible.append((point, total))
        if len(feasible) != int(row["b1_feasible_grid_points"]):
            raise ValueError(f"Feasibility count mismatch for budget={budget}, context={context}")
        scenarios.append({"context": context, "budget": budget, "baseline_run": baseline_run, "feasible": feasible})

    scenario_results: dict[tuple[int, int], list[dict[str, Any]]] = {
        (item["budget"], item["context"]): [] for item in scenarios
    }
    for fit in bootstrap_rows:
        replicate = int(fit["replicate"])
        parameters = {name: float(fit[name]) for name in ("E", "A", "alpha", "B", "beta")}
        for scenario in scenarios:
            candidates = []
            for point, total_cost in scenario["feasible"]:
                prediction = (
                    parameters["E"]
                    + parameters["A"] * point["n_b"] ** (-parameters["alpha"])
                    + parameters["B"] * point["d_b"] ** (-parameters["beta"])
                )
                candidates.append((prediction, total_cost, point))
            prediction, total_cost, selected = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2]["n_b"], item[2]["d_b"], item[2]["run_id"]),
            )
            scenario_results[(scenario["budget"], scenario["context"])].append(
                {
                    "replicate": replicate,
                    "run_id": selected["run_id"],
                    "n_b": selected["n_b"],
                    "d_b": selected["d_b"],
                    "prediction": prediction,
                    "cost": float(total_cost),
                }
            )

    summary_rows = []
    frequency_rows = []
    for scenario in scenarios:
        key = (scenario["budget"], scenario["context"])
        draws = scenario_results[key]
        counts = Counter(row["run_id"] for row in draws)
        top_run, top_count = counts.most_common(1)[0]
        selected_n = [row["n_b"] for row in draws]
        selected_d = [row["d_b"] for row in draws]
        losses = [row["prediction"] for row in draws]
        baseline_count = counts.get(scenario["baseline_run"], 0)
        summary_rows.append(
            {
                "budget_flops": scenario["budget"],
                "context_length_tokens": scenario["context"],
                "bootstrap_replicates": len(draws),
                "baseline_selected_run_id": scenario["baseline_run"],
                "baseline_selection_frequency": baseline_count / len(draws),
                "modal_selected_run_id": top_run,
                "modal_selection_frequency": top_count / len(draws),
                "unique_selected_grid_points": len(counts),
                "selected_N_B_p2_5": percentile(selected_n, 0.025),
                "selected_N_B_median": percentile(selected_n, 0.5),
                "selected_N_B_p97_5": percentile(selected_n, 0.975),
                "selected_D_B_p2_5": percentile(selected_d, 0.025),
                "selected_D_B_median": percentile(selected_d, 0.5),
                "selected_D_B_p97_5": percentile(selected_d, 0.975),
                "predicted_loss_p2_5": percentile(losses, 0.025),
                "predicted_loss_median": percentile(losses, 0.5),
                "predicted_loss_p97_5": percentile(losses, 0.975),
                "feasible_grid_points": len(scenario["feasible"]),
                "interpretation": "trajectory-cluster bootstrap sensitivity; fixed-Q B1/M0 support grid only",
            }
        )
        for run_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            selected_point = next(point for point in points if point["run_id"] == run_id)
            frequency_rows.append(
                {
                    "budget_flops": scenario["budget"],
                    "context_length_tokens": scenario["context"],
                    "run_id": run_id,
                    "N_B": selected_point["n_b"],
                    "D_B": selected_point["d_b"],
                    "selection_count": count,
                    "selection_frequency": count / len(draws),
                    "baseline_selected": run_id == scenario["baseline_run"],
                }
            )

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(FREQUENCY_CSV, frequency_rows)

    baseline_frequency_values = [row["baseline_selection_frequency"] for row in summary_rows]
    report_lines = [
        "# Q3 M0 参考前沿的轨迹 Bootstrap 敏感性",
        "",
        "## 方法与范围",
        "",
        f"将 Q2 已完成的 {BOOTSTRAP_REPLICATES} 次 B1 轨迹级 cluster Bootstrap 拟合参数，逐次代入 Q3 固定 $Q=Q_0$ 的 15 个预算×上下文情景；每次都在对应的可行 B1 实测网格点中重新最小化 M0 预测 Loss。成本和可行点集合保持不变。",
        "",
        "本分析传播的是 Q2 M0 参数的轨迹重抽样敏感性，不是独立外部验证、完整 Q3 联合最优或 Q/p 效应估计。Bootstrap 仅基于 8 条 Pythia 轨迹，区间应视作有限样本稳定性提示。",
        "",
        "## 15 个情景摘要",
        "",
        "| 上下文 | 预算 FLOPs | 原始最优点在 Bootstrap 中重选比例 | 众数配置比例 | 不同选中网格点数 | N_B 中位数 [2.5%,97.5%] | D_B 中位数 [2.5%,97.5%] | 预测 Loss 中位数 [2.5%,97.5%] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| {L:,} | {C:.0e} | {base:.1%} | {modal:.1%} | {unique} | {nmed} [{nlo}, {nhi}] | {dmed} [{dlo}, {dhi}] | {lmed} [{llo}, {lhi}] |".format(
                L=row["context_length_tokens"], C=row["budget_flops"],
                base=row["baseline_selection_frequency"], modal=row["modal_selection_frequency"],
                unique=row["unique_selected_grid_points"],
                nmed=fmt(row["selected_N_B_median"]), nlo=fmt(row["selected_N_B_p2_5"]), nhi=fmt(row["selected_N_B_p97_5"]),
                dmed=fmt(row["selected_D_B_median"]), dlo=fmt(row["selected_D_B_p2_5"]), dhi=fmt(row["selected_D_B_p97_5"]),
                lmed=fmt(row["predicted_loss_median"]), llo=fmt(row["predicted_loss_p2_5"]), lhi=fmt(row["predicted_loss_p97_5"]),
            )
        )
    report_lines.extend(
        [
            "",
            "## 解读边界",
            "",
            f"- 15 个情景中，原始 M0 拟合选中配置的 Bootstrap 重选比例范围为 {min(baseline_frequency_values):.1%}–{max(baseline_frequency_values):.1%}。这只说明在既定函数形式和 B1 八轨迹重抽样下的选择稳定性。",
            "- Bootstrap 参数不确定性不覆盖 M0 函数形式偏差、B1 与目标训练方案的迁移误差、N/D 网格之外的行为，也不补出 Q/p 响应。",
            "- 最高预算档仍受 B1 支持网格上界约束；较窄的 Bootstrap 区间不能解除该边界。",
            "- 逐配置频率见 `M0_ND_cluster_bootstrap_selection_frequency.csv`；机器摘要见 `M0_ND_cluster_bootstrap_frontier.csv`。",
            "",
            f"复现命令：`python Q3/03_代码/assess_q3_m0_cluster_bootstrap.py`。输入复现信息见 `{MANIFEST_JSON.as_posix()}`。",
            "",
        ]
    )
    (ROOT / REPORT_MD).write_text("\n".join(report_lines), encoding="utf-8")

    input_paths = [Q2_GRID, Q2_BOOTSTRAP, Q2_BOOTSTRAP_SUMMARY, Q2_TRAJECTORY_MAPPING, BASELINE_FRONTIER]
    script_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    script_path = ROOT / script_rel
    outputs = [SUMMARY_CSV, FREQUENCY_CSV, REPORT_MD]
    manifest = {
        "artifact": "Q3 M0 B1 trajectory-cluster bootstrap frontier sensitivity",
        "method": "Propagate the existing Q2 eight-trajectory cluster-bootstrap fit coefficients through each fixed-Q feasible B1 grid scenario; reselect the minimum predicted M0 loss for each replicate.",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "claim_boundary": "parameter-resampling sensitivity within B1/M0; not independent validation, not Q/p response, not full Q3 joint optimum",
        "python_version": platform.python_version(),
        "script": {"path": script_rel, "sha256": sha256(script_path)},
        "inputs": {path.as_posix(): sha256(ROOT / path) for path in input_paths},
        "outputs": {path.as_posix(): sha256(ROOT / path) for path in outputs},
    }
    (ROOT / MANIFEST_JSON).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_rows)} scenario summaries and {len(frequency_rows)} nonzero selection-frequency rows.")
    print(f"Baseline selection frequency range: {min(baseline_frequency_values):.1%}–{max(baseline_frequency_values):.1%}.")


if __name__ == "__main__":
    main()
