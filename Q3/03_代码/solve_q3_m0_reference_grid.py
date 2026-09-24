#!/usr/bin/env python3
"""Build Q3's fixed-Q, B1/M0 support-grid reference frontier.

Run from the repository root:
    python Q3/03_代码/solve_q3_m0_reference_grid.py

This is a restricted N/D reference analysis. It does not optimize Q or p and
does not claim a full Q3 joint optimum.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

Q1_FREEZE_REL = Path("Q1/03_结果/Q1_最终收口/q1_final_freeze.json")
Q1_SAMPLE_REL = Path("Q1/03_结果/Q1_最终收口/q1_final_quality.csv")
Q1_DOMAIN_REL = Path("Q1/03_结果/Q1_最终收口/q1_final_domain_quality.csv")
Q1_MACRO_REL = Path("Q1/03_结果/Q1.1/v1/a1_macro_summary.csv")
Q2_GRID_REL = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
Q2_PARAMS_REL = Path("Q2/03_结果/经典ScalingLaw基线/v1/fit_parameters.json")
Q2_MAPPING_REL = Path("Q2/03_结果/经典ScalingLaw基线/v1/gate0_b1_trajectory_mapping.csv")
C7_REL = Path(
    "中文题目/F题/real_attachments/C_efficiency_evolution/"
    "model_architecture_metadata.csv"
)

OUTPUT_REL = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
SUMMARY_REL = Path("Q3/04_结果/M0_ND_reference_frontier_summary.json")
REPORT_REL = Path("Q3/04_结果/M0_ND_reference_frontier_report.md")
MANIFEST_REL = Path("Q3/04_结果/M0_reference_reproduction_manifest.json")

ETA = Fraction(1, 5000)  # 2e-4 FLOPs per parameter-token-position proxy
TRAINING_COEFFICIENT = 6
BUDGETS_FLOPS = (10**19, 10**22, 10**24)
EXPECTED_Q_VERSION = "Q1-q_huber-v1"
EXPECTED_OPERATIONAL_Q = "q_huber"
PREDICTION_TOLERANCE = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_q1_interface() -> tuple[float, float, float, dict[str, str]]:
    freeze_path = ROOT / Q1_FREEZE_REL
    sample_path = ROOT / Q1_SAMPLE_REL
    domain_path = ROOT / Q1_DOMAIN_REL
    macro_path = ROOT / Q1_MACRO_REL
    freeze = read_json(freeze_path)

    require(
        freeze.get("status") == "FROZEN_WITH_LIMITATIONS",
        "Q1 freeze status does not match the expected frozen interface state",
    )
    require(
        freeze.get("operational_q") == EXPECTED_OPERATIONAL_Q
        and freeze.get("q_version") == EXPECTED_Q_VERSION,
        "Q1 operational Q/version differs from the frozen Q3 baseline contract",
    )

    expected_outputs = freeze.get("outputs", {})
    sample_record = expected_outputs.get("q1_final_quality.csv", {})
    domain_record = expected_outputs.get("q1_final_domain_quality.csv", {})
    require(sample_record.get("sha256") == sha256_file(sample_path), "Q1 sample interface hash mismatch")
    require(domain_record.get("sha256") == sha256_file(domain_path), "Q1 domain interface hash mismatch")

    domain_rows = read_csv(domain_path)
    a1_rows = [row for row in domain_rows if row.get("dataset") == "a1"]
    require(len(a1_rows) == 7, f"Expected seven A1 quality domains, found {len(a1_rows)}")
    require(
        all(row.get("q_version") == EXPECTED_Q_VERSION for row in a1_rows),
        "A1 domain rows do not use the frozen Q1 operational version",
    )
    q0 = sum(float(row["q_operational"]) for row in a1_rows) / len(a1_rows)

    macro_rows = read_csv(macro_path)
    q_huber_macro = next(
        (row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_huber"),
        None,
    )
    q_equal_macro = next(
        (row for row in macro_rows if row.get("dataset") == "a1" and row.get("candidate") == "q_equal"),
        None,
    )
    require(q_huber_macro is not None and q_equal_macro is not None, "A1 macro summary is missing a Q candidate")
    macro_value = float(q_huber_macro["macro_domain_score"])
    require(abs(q0 - macro_value) <= 1e-10, "Q1 domain-interface mean does not match the A1 q_huber macro summary")
    q0_ci_low = float(q_huber_macro["ci_low"])
    q0_ci_high = float(q_huber_macro["ci_high"])
    q_equal_value = float(q_equal_macro["macro_domain_score"])

    input_hashes = {
        rel(freeze_path): sha256_file(freeze_path),
        rel(sample_path): sha256_file(sample_path),
        rel(domain_path): sha256_file(domain_path),
        rel(macro_path): sha256_file(macro_path),
    }
    return q0, q0_ci_low, q0_ci_high, {
        "q_version": EXPECTED_Q_VERSION,
        "operational_q": EXPECTED_OPERATIONAL_Q,
        "q0_reference": f"{q0:.15g}",
        "q0_equal_sensitivity": f"{q_equal_value:.15g}",
        "q0_ci_low": f"{q0_ci_low:.15g}",
        "q0_ci_high": f"{q0_ci_high:.15g}",
        "q1_status": freeze["status"],
        "q1_interface_note": "A1 seven-domain macro reference; not a measured B1/Pythia mixture quality.",
        "input_hashes": input_hashes,
    }


def verify_b1_grid() -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    grid_path = ROOT / Q2_GRID_REL
    params_path = ROOT / Q2_PARAMS_REL
    mapping_path = ROOT / Q2_MAPPING_REL
    params_file = read_json(params_path)
    require(params_file.get("fit_scope") == "B1 only", "Q2 fit parameter file is not the B1-only M0 fit")
    params = {key: float(value) for key, value in params_file["parameters"].items()}

    grid_rows = read_csv(grid_path)
    require(len(grid_rows) == 1176, f"Expected 1,176 B1 fit rows; found {len(grid_rows)}")
    required = {"run_id", "N_params_B", "D_tokens_B", "observed_val_loss", "predicted_val_loss"}
    require(required.issubset(grid_rows[0].keys()), "B1 prediction table is missing required columns")

    points: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    d_by_n: dict[float, set[float]] = {}
    for row in grid_rows:
        n_b = float(row["N_params_B"])
        d_b = float(row["D_tokens_B"])
        key = (n_b, d_b)
        require(key not in seen, f"Duplicate B1 N/D point found: {key}")
        seen.add(key)
        d_by_n.setdefault(n_b, set()).add(d_b)
        observed = float(row["observed_val_loss"])
        supplied_prediction = float(row["predicted_val_loss"])
        computed_prediction = (
            params["E"]
            + params["A"] * n_b ** (-params["alpha"])
            + params["B"] * d_b ** (-params["beta"])
        )
        require(
            abs(computed_prediction - supplied_prediction) <= PREDICTION_TOLERANCE,
            f"M0 prediction does not reproduce for run_id={row['run_id']}",
        )
        require(n_b > 0 and d_b > 0, f"Nonpositive B1 N/D value at run_id={row['run_id']}")
        points.append(
            {
                "run_id": row["run_id"],
                "N_params_B": n_b,
                "D_tokens_B": d_b,
                "observed_val_loss": observed,
                "predicted_val_loss": computed_prediction,
            }
        )

    n_values = sorted(d_by_n)
    common_d = set.intersection(*(values for values in d_by_n.values()))
    require(len(n_values) == 8, f"Expected eight B1 parameter sizes; found {len(n_values)}")
    require(len(common_d) == 147 and all(values == common_d for values in d_by_n.values()), "B1 is not the expected 8x147 common grid")
    require(len(seen) == len(n_values) * len(common_d), "B1 N/D support grid is incomplete")

    mapping_rows = read_csv(mapping_path)
    require(len(mapping_rows) == 8, f"Expected eight B12 trajectory mappings; found {len(mapping_rows)}")
    for row in mapping_rows:
        require(int(row["B1_rows"]) == 147, "B12 mapping has a trajectory with a non-147 B1 row count")
        require(int(row["matched_steps"]) == 147, "B12 mapping has a trajectory with unmatched B1 steps")
        require(row["missing_steps"] == "[]", "B12 mapping reports missing steps")

    hashes = {
        rel(grid_path): sha256_file(grid_path),
        rel(params_path): sha256_file(params_path),
        rel(mapping_path): sha256_file(mapping_path),
    }
    return points, params, {
        **hashes,
        "row_count": str(len(points)),
        "N_unique": str(len(n_values)),
        "D_unique_common": str(len(common_d)),
        "N_min_B": f"{min(n_values):.15g}",
        "N_max_B": f"{max(n_values):.15g}",
        "D_min_B": f"{min(common_d):.15g}",
        "D_max_B": f"{max(common_d):.15g}",
    }


def read_context_lengths() -> tuple[list[int], dict[str, str]]:
    path = ROOT / C7_REL
    rows = read_csv(path)
    require(rows, "C7 context metadata is empty")
    require("max_position_embeddings" in rows[0], "C7 max_position_embeddings column is missing")
    lengths = sorted({int(row["max_position_embeddings"]) for row in rows})
    require(all(length > 0 for length in lengths), "C7 contains a nonpositive context length")
    return lengths, {rel(path): sha256_file(path)}


def exact_compute_cost(point: dict[str, Any], context_length: int) -> tuple[Fraction, Fraction, Fraction]:
    n_abs = int(Fraction(str(point["N_params_B"])) * 10**9)
    d_abs = int(Fraction(str(point["D_tokens_B"])) * 10**9)
    require(n_abs > 0 and d_abs > 0, "N or D did not convert to a positive integer")
    nd = n_abs * d_abs
    training = Fraction(TRAINING_COEFFICIENT * nd)
    attention = ETA * nd * context_length
    return training, attention, training + attention


def solve_scenarios(
    points: list[dict[str, Any]],
    contexts: list[int],
    q0: float,
    q0_ci_low: float,
    q0_ci_high: float,
    q_metadata: dict[str, Any],
    b1_metadata: dict[str, str],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for context in contexts:
        point_costs = [(point, *exact_compute_cost(point, context)) for point in points]
        for budget in BUDGETS_FLOPS:
            feasible = [entry for entry in point_costs if entry[3] <= budget]
            require(feasible, f"No B1 support-grid point is feasible for C={budget}, L={context}")
            selected = min(
                feasible,
                key=lambda entry: (
                    entry[0]["predicted_val_loss"],
                    entry[3],
                    entry[0]["N_params_B"],
                    entry[0]["D_tokens_B"],
                    entry[0]["run_id"],
                ),
            )
            point, training, attention, total = selected
            n_abs = int(Fraction(str(point["N_params_B"])) * 10**9)
            d_abs = int(Fraction(str(point["D_tokens_B"])) * 10**9)
            outputs.append(
                {
                    "budget_flops": budget,
                    "context_length_tokens": context,
                    "critical_context_length_tokens": 30000,
                    "q1_q_version": EXPECTED_Q_VERSION,
                    "q0_reference_a1_macro": q0,
                    "q0_reference_ci95_low": q0_ci_low,
                    "q0_reference_ci95_high": q0_ci_high,
                    "q_equal_sensitivity_macro": float(q_metadata["q0_equal_sensitivity"]),
                    "q_setting": "Q=Q0; quality uplift disabled; C_Q=0",
                    "p_setting": "not optimized; M0 has no p response",
                    "g_function": "not active because Q=Q0",
                    "b1_feasible_grid_points": len(feasible),
                    "b1_total_grid_points": len(points),
                    "selected_run_id": point["run_id"],
                    "selected_N_B": point["N_params_B"],
                    "selected_D_B": point["D_tokens_B"],
                    "selected_N_parameters": n_abs,
                    "selected_D_tokens": d_abs,
                    "m0_predicted_val_loss": point["predicted_val_loss"],
                    "b1_observed_val_loss_at_selected_point": point["observed_val_loss"],
                    "C_train_flops": int(training),
                    "C_Q_flops": 0,
                    "C_attn_flops": float(attention),
                    "C_total_flops": float(total),
                    "budget_utilization": float(total / budget),
                    "budget_slack_flops": float(Fraction(budget) - total),
                    "all_support_points_feasible": len(feasible) == len(points),
                    "selected_at_support_max_N_and_D": (
                        point["N_params_B"] == max(item["N_params_B"] for item in points)
                        and point["D_tokens_B"] == max(item["D_tokens_B"] for item in points)
                    ),
                    "interpretation": "B1/M0 support-grid reference; in-sample fit; not full Q3 optimum",
                }
            )
    return outputs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    output_rows: list[dict[str, Any]],
    q0_metadata: dict[str, Any],
    b1_metadata: dict[str, str],
    c7_lengths: list[int],
) -> str:
    lines = [
        "# Q3 固定 Q 的 M0 支持网格参考前沿",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        "**范围：** Q1 冻结的 q_huber 主接口；固定 $Q=Q_0$，不优化 p；仅在 Q2 B1 实际观测的 N/D 网格上按 Q2 M0 预测 Loss 筛选。  ",
        "**解释：** 这是 B1/M0 的受限参考配置，不是完整 Q3 联合最优、独立外部验证或结构转移结论。",
        "",
        "## 冻结基线与支持域",
        "",
        f"- Q1 版本：`{q0_metadata['q_version']}`；主 Q=`{q0_metadata['operational_q']}`；A1 七域等权宏平均 $Q_0={float(q0_metadata['q0_reference']):.10f}$，95% 区间 [{float(q0_metadata['q0_ci_low']):.6f}, {float(q0_metadata['q0_ci_high']):.6f}]。这是 Q3 成本参考基线，不是 B1 配方质量的直接测量。",
        f"- B1 网格：{b1_metadata['N_unique']} 个 N 值 × {b1_metadata['D_unique_common']} 个共同 D 检查点；$N_B$ 范围 [{b1_metadata['N_min_B']}, {b1_metadata['N_max_B']}], $D_B$ 范围 [{b1_metadata['D_min_B']}, {b1_metadata['D_max_B']}]; 候选点 {b1_metadata['row_count']}。",
        f"- C7 上下文情景：{', '.join(f'{length:,}' for length in c7_lengths)} tokens。",
        "- 成本为 $C_{train}=6ND$、$C_Q=0$、$C_{attn}=\\eta ND L_{ctx}$，其中 N 和 D 从十亿单位转为绝对参数数与 Token 数；预算为 $10^{19},10^{22},10^{24}$ FLOPs。",
        "- 每个情景从满足预算的实测 B1 网格点中选择 M0 预测验证 Loss 最低者；同时列出该点的 B1 观测验证 Loss 作参照。",
        "",
        "## 结果（15 个预算 × 上下文情景）",
        "",
        "| $L_{ctx}$ | 预算 FLOPs | 可行网格点 | $N_B$ | $D_B$ | M0 预测 Loss | B1 观测 Loss | 成本占用 | 预算利用率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['context_length_tokens']:,} | $10^{{{len(str(row['budget_flops'])) - 1}}}$ | "
            f"{row['b1_feasible_grid_points']}/{row['b1_total_grid_points']} | "
            f"{row['selected_N_B']:.6g} | {row['selected_D_B']:.6g} | "
            f"{row['m0_predicted_val_loss']:.6f} | {row['b1_observed_val_loss_at_selected_point']:.6f} | "
            f"{row['C_total_flops']:.4g} | {row['budget_utilization']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "1. 此支路固定质量在 q_huber 基线，所以三种 $g(Q)$ 形式不影响本次选择；它们要等质量变化能影响 Loss 时才进入完整优化。",
            "2. Q2 M0 不含 p 或 Q 项，当前 p Gate 仍为 FAIL；表格不包含配比效应、质量边际效应或 p/Q 联合最优。",
            "3. Q2 M0 是在同一 B1 网格上拟合得到的，表中最优选择使用其样本内预测；B1 只有 8 个独立轨迹簇，不能把 1,176 个点解释成 1,176 个独立实验。",
            "4. 若一个预算下 1,176/1,176 个点均可行，该结果受 B1 网格上界限制；不能据此断言预算已经充分利用或更高 N/D 的边际收益为零。",
            "5. 不对本支路作正式结构转移判断；当前只归档预算与上下文变化下的参考配置。",
            "",
            "## 输入来源",
            "",
            f"- Q1 冻结基线与域接口：`{Q1_FREEZE_REL.as_posix()}`、`{Q1_DOMAIN_REL.as_posix()}`。",
            f"- Q2 B1/M0 逐点预测和参数：`{Q2_GRID_REL.as_posix()}`、`{Q2_PARAMS_REL.as_posix()}`。",
            f"- C7 上下文：`{C7_REL.as_posix()}`。",
            "- 复现命令：`python Q3/03_代码/solve_q3_m0_reference_grid.py`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    input_paths = [
        ROOT / Q1_FREEZE_REL,
        ROOT / Q1_SAMPLE_REL,
        ROOT / Q1_DOMAIN_REL,
        ROOT / Q1_MACRO_REL,
        ROOT / Q2_GRID_REL,
        ROOT / Q2_PARAMS_REL,
        ROOT / Q2_MAPPING_REL,
        ROOT / C7_REL,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input files missing:\n- " + "\n- ".join(missing))

    q0, q0_ci_low, q0_ci_high, q_metadata = verify_q1_interface()
    points, params, b1_metadata = verify_b1_grid()
    contexts, c7_hashes = read_context_lengths()
    outputs = solve_scenarios(
        points,
        contexts,
        q0,
        q0_ci_low,
        q0_ci_high,
        q_metadata,
        b1_metadata,
    )

    output_path = ROOT / OUTPUT_REL
    summary_path = ROOT / SUMMARY_REL
    report_path = ROOT / REPORT_REL
    manifest_path = ROOT / MANIFEST_REL
    write_csv(output_path, outputs)

    input_hashes = {
        **q_metadata["input_hashes"],
        **{key: value for key, value in b1_metadata.items() if key.endswith(".csv") or key.endswith(".json")},
        **c7_hashes,
    }
    summary = {
        "artifact": "Q3 fixed-Q Q2-M0 B1 support-grid reference frontier",
        "scope": {
            "q_version": q_metadata["q_version"],
            "q0_reference_a1_macro": q0,
            "q0_reference_ci95": [q0_ci_low, q0_ci_high],
            "q_equal_sensitivity_macro": float(q_metadata["q0_equal_sensitivity"]),
            "quality_setting": "Q=Q0; C_Q=0",
            "p_setting": "not optimized; M0 has no p response",
            "objective": "minimize Q2 M0 predicted validation loss over feasible observed B1 N/D points",
            "claim_boundary": "B1 in-sample support-grid reference; not full Q3 joint optimum or structural-transition claim",
        },
        "support": {
            "grid_points": len(points),
            "N_unique": int(b1_metadata["N_unique"]),
            "D_unique_common": int(b1_metadata["D_unique_common"]),
            "N_B_range": [float(b1_metadata["N_min_B"]), float(b1_metadata["N_max_B"])],
            "D_B_range": [float(b1_metadata["D_min_B"]), float(b1_metadata["D_max_B"])],
            "context_lengths": contexts,
            "budgets_flops": list(BUDGETS_FLOPS),
        },
        "m0_parameters": params,
        "results": outputs,
        "input_sha256": input_hashes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = render_report(outputs, q_metadata, b1_metadata, contexts)
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "artifact": "Q3 M0 support-grid reference reproducibility manifest",
        "command": "python Q3/03_代码/solve_q3_m0_reference_grid.py",
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "script": {"path": rel(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "inputs": input_hashes,
        "parameters": {
            "eta": float(ETA),
            "training_cost_coefficient": TRAINING_COEFFICIENT,
            "budgets_flops": list(BUDGETS_FLOPS),
            "objective": "Q2 M0 predicted validation loss",
            "quality_setting": "Q=Q0 so C_Q=0",
        },
        "outputs": {
            rel(output_path): sha256_file(output_path),
            rel(summary_path): sha256_file(summary_path),
            rel(report_path): sha256_file(report_path),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Q1 baseline: {EXPECTED_Q_VERSION}; Q0={q0:.10f}")
    print(f"B1 support grid: {len(points)} points; {len(contexts)} contexts")
    print(f"Generated {len(outputs)} budget/context reference configurations")
    for row in outputs:
        print(
            f"C={row['budget_flops']:.0e}, L={row['context_length_tokens']}: "
            f"N_B={row['selected_N_B']:.6g}, D_B={row['selected_D_B']:.6g}, "
            f"Lhat={row['m0_predicted_val_loss']:.6f}, "
            f"feasible={row['b1_feasible_grid_points']}/{row['b1_total_grid_points']}"
        )
    print(f"Wrote {rel(output_path)}")
    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(report_path)}")
    print(f"Wrote {rel(manifest_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
