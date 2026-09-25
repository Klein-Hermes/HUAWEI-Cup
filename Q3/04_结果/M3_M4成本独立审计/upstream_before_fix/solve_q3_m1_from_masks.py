#!/usr/bin/env python3
"""Run M1 only after M3 support and M4 budget masks have been produced.

Run after build_q3_m3_m4_mask_pipeline.py:
    python Q3/03_代码/solve_q3_m1_from_masks.py

M1 minimizes the frozen B1/M0 predicted validation loss over rows where both
M3 search support and M4 budget feasibility are true. It does not define the
support domain or recalculate feasibility as an independent eligibility rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Q2_GRID = Path("Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv")
M0_FRONTIER = Path("Q3/04_结果/M0_ND_reference_frontier.csv")
M3_CANDIDATES = Path("Q3/04_结果/m3_candidate_grid.csv")
M3_SUPPORT = Path("Q3/04_结果/m3_q3_support_grid.csv")
M4_MASK = Path("Q3/04_结果/m3_budget_feasible_mask.csv")
M3_MANIFEST = Path("Q3/04_结果/M3_support_manifest.json")
M4_MANIFEST = Path("Q3/04_结果/M4_budget_manifest.json")

OUT_BOUNDARY = Path("Q3/04_结果/m1_optimum_boundary_audit.csv")
OUT_REPORT = Path("Q3/04_结果/M1_masked_optimization_report.md")
OUT_MANIFEST = Path("Q3/04_结果/M1_masked_optimization_manifest.json")

ETA = Fraction(1, 5000)
TRAINING_COEFFICIENT = 6
UNIT_SCALE = 10**9
FLOAT_TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty output: {relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_cost_components(n_b: str, d_b: str, context: int) -> tuple[Fraction, Fraction, Fraction]:
    n_abs = int(Fraction(n_b) * UNIT_SCALE)
    d_abs = int(Fraction(d_b) * UNIT_SCALE)
    nd = n_abs * d_abs
    return Fraction(TRAINING_COEFFICIENT * nd), Fraction(0), ETA * nd * context


def load_interfaces() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[tuple[int, int, str], dict[str, str]],
]:
    candidates = read_csv(ROOT / M3_CANDIDATES)
    support_rows = read_csv(ROOT / M3_SUPPORT)
    mask_rows = read_csv(ROOT / M4_MASK)
    require(candidates and support_rows and mask_rows, "M3/M4 mask files must be nonempty")
    candidate_by_id = {row["grid_id"]: row for row in candidates}
    support_by_id = {row["grid_id"]: row for row in support_rows}
    require(len(candidate_by_id) == len(candidates), "M3 candidate grid_id is not unique")
    require(len(support_by_id) == len(support_rows), "M3 support grid_id is not unique")
    require(set(candidate_by_id) == set(support_by_id), "Candidate and M3 support grid IDs differ")

    masks: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in mask_rows:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]), row["grid_id"])
        require(key not in masks, f"Duplicate M4 mask row: {key}")
        require(row["grid_id"] in candidate_by_id, f"M4 mask references unknown grid_id={row['grid_id']}")
        candidate = candidate_by_id[row["grid_id"]]
        require(
            Fraction(row["N_params_B"]) == Fraction(candidate["N_params_B"])
            and Fraction(row["D_tokens_B"]) == Fraction(candidate["D_tokens_B"]),
            f"M4 coordinates differ from candidate grid for grid_id={row['grid_id']}",
        )
        masks[key] = row
    for grid_id, candidate in candidate_by_id.items():
        support = support_by_id[grid_id]
        require(
            Fraction(support["N_params_B"]) == Fraction(candidate["N_params_B"])
            and Fraction(support["D_tokens_B"]) == Fraction(candidate["D_tokens_B"]),
            f"M3 coordinates differ from candidate grid for grid_id={grid_id}",
        )
    scenario_keys = {(budget, context) for budget, context, _ in masks}
    require(scenario_keys, "M4 mask contains no scenarios")
    expected_keys = {
        (budget, context, grid_id)
        for budget, context in scenario_keys
        for grid_id in candidate_by_id
    }
    require(set(masks) == expected_keys, "M4 mask does not cover every candidate in every scenario")
    return candidates, support_by_id, masks


def solve_m1(
    candidates: list[dict[str, str]],
    support_by_id: dict[str, dict[str, str]],
    masks: dict[tuple[int, int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    candidate_by_id = {row["grid_id"]: row for row in candidates}
    n_values = [float(row["N_params_B"]) for row in candidates]
    d_values = [float(row["D_tokens_B"]) for row in candidates]
    results = []
    scenarios = sorted({(budget, context) for budget, context, _ in masks}, key=lambda item: (item[1], item[0]))

    for budget, context in scenarios:
        eligible: list[tuple[dict[str, str], dict[str, str], Fraction, float]] = []
        budget_feasible_count = 0
        for candidate in candidates:
            grid_id = candidate["grid_id"]
            support = support_by_id[grid_id]
            mask = masks[(budget, context, grid_id)]
            m3_allowed = support["m3_search_allowed"].lower() == "true"
            budget_feasible = mask["budget_feasible"].lower() == "true"
            # M3 and M4 remain separate upstream responsibilities. M1 forms
            # their intersection here, immediately before optimization.
            final_feasible = m3_allowed and budget_feasible

            train_cost, quality_cost, attention_cost = exact_cost_components(
                candidate["N_params_B"], candidate["D_tokens_B"], context
            )
            computed_cost = train_cost + quality_cost + attention_cost
            cost_feasible = computed_cost <= budget
            require(
                cost_feasible == budget_feasible,
                f"M4 budget flag disagrees with exact cost for {grid_id}, C={budget}, L={context}",
            )
            require(int(mask["C_train_flops"]) == int(train_cost), f"M4 training cost mismatch for {grid_id}")
            require(int(mask["C_Q_flops"]) == int(quality_cost), f"M4 quality cost mismatch for {grid_id}")
            require(
                math.isclose(float(mask["C_attn_flops"]), float(attention_cost), rel_tol=FLOAT_TOL, abs_tol=1.0),
                f"M4 attention cost mismatch for {grid_id}",
            )
            require(
                math.isclose(float(mask["C_total_flops"]), float(computed_cost), rel_tol=FLOAT_TOL, abs_tol=1.0),
                f"M4 total cost mismatch for {grid_id}",
            )
            if budget_feasible:
                budget_feasible_count += 1
            if not final_feasible:
                continue

            require(candidate["observed_exact"].lower() == "true", f"M1 eligible point is not observed exact: {grid_id}")
            require(candidate["m0_predicted_val_loss"] != "", f"M1 eligible point has no M0 prediction: {grid_id}")
            loss = float(candidate["m0_predicted_val_loss"])
            require(math.isfinite(loss), f"Non-finite M0 loss for grid_id={grid_id}")
            eligible.append((candidate, mask, computed_cost, loss))

        require(eligible, f"M1 has no eligible point for C={budget}, L={context}")
        selected, mask, total, loss = min(
            eligible,
            key=lambda item: (
                item[3],
                item[2],
                float(item[0]["N_params_B"]),
                float(item[0]["D_tokens_B"]),
                item[0]["source_run_id"],
            ),
        )
        support = support_by_id[selected["grid_id"]]
        results.append(
            {
                "scenario_id": mask["scenario_id"],
                "budget_flops": budget,
                "context_length_tokens": context,
                "m1_selected_grid_id": selected["grid_id"],
                "m1_selected_run_id": selected["source_run_id"],
                "selected_N_B": float(selected["N_params_B"]),
                "selected_D_B": float(selected["D_tokens_B"]),
                "m0_predicted_val_loss": loss,
                "observed_val_loss": selected["observed_val_loss"],
                "m3_search_allowed": support["m3_search_allowed"],
                "budget_feasible": mask["budget_feasible"],
                "final_feasible": final_feasible,
                "budget_feasible_candidate_count": budget_feasible_count,
                "m1_eligible_candidate_count": len(eligible),
                "candidate_count": len(candidates),
                "hit_N_min": math.isclose(float(selected["N_params_B"]), min(n_values), rel_tol=0, abs_tol=FLOAT_TOL),
                "hit_N_max": math.isclose(float(selected["N_params_B"]), max(n_values), rel_tol=0, abs_tol=FLOAT_TOL),
                "hit_D_min": math.isclose(float(selected["D_tokens_B"]), min(d_values), rel_tol=0, abs_tol=FLOAT_TOL),
                "hit_D_max": math.isclose(float(selected["D_tokens_B"]), max(d_values), rel_tol=0, abs_tol=FLOAT_TOL),
                "inside_convex_hull": support["inside_convex_hull"],
                "interpolation_flag": support["interpolation_candidate"],
                "extrapolation_flag": support["extrapolation_flag"],
                "nearest_observed_distance_standardized_log": support["nearest_observed_distance_standardized_log"],
                "C_train_flops": mask["C_train_flops"],
                "C_Q_flops": mask["C_Q_flops"],
                "C_attn_flops": mask["C_attn_flops"],
                "C_total_flops": float(total),
                "budget_utilization": float(total / budget),
                "budget_slack_flops": float(Fraction(budget) - total),
            }
        )
    return results


def check_existing_frontier(results: list[dict[str, Any]], default_grid: bool) -> dict[str, Any]:
    if not default_grid:
        return {
            "performed": False,
            "reason": "a custom candidate grid was supplied; its search domain differs from the frozen M0 frontier",
        }
    existing = read_csv(ROOT / M0_FRONTIER)
    reference = {
        (int(row["budget_flops"]), int(row["context_length_tokens"])): row
        for row in existing
    }
    require(len(reference) == 15, "Frozen M0 frontier must contain 15 unique scenarios")
    compared = []
    for row in results:
        key = (int(row["budget_flops"]), int(row["context_length_tokens"]))
        old = reference.get(key)
        require(old is not None, f"Frozen M0 frontier is missing scenario {key}")
        same_choice = (
            row["m1_selected_run_id"] == old["selected_run_id"]
            and math.isclose(row["selected_N_B"], float(old["selected_N_B"]), rel_tol=0, abs_tol=FLOAT_TOL)
            and math.isclose(row["selected_D_B"], float(old["selected_D_B"]), rel_tol=0, abs_tol=FLOAT_TOL)
        )
        same_count = row["budget_feasible_candidate_count"] == int(old["b1_feasible_grid_points"])
        require(same_choice and same_count, f"Masked M1 does not reproduce existing M0 scenario {key}")
        compared.append({"budget_flops": key[0], "context_length_tokens": key[1], "selection_match": True, "feasible_count_match": True})
    return {"performed": True, "scenario_count": len(compared), "all_match": True, "scenarios": compared}


def render_report(results: list[dict[str, Any]], parity: dict[str, Any]) -> str:
    lines = [
        "# M1：读取 M3/M4 掩码后的配置优化",
        "",
        f"**生成时间（UTC）：** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        "**优化范围：** 只选择 `m3_search_allowed=true` 且 `budget_feasible=true` 的候选。当前固定 Q=Q0、p 不进入模型，目标为冻结 B1/M0 预测验证 Loss。",
        "**职责边界：** M1 消费 M3/M4 的资格标记，不定义插值支持域，也不替 M4 重新判预算。",
        "",
        (
            f"与既有 M0 前沿核对：**{parity['scenario_count']}/{parity['scenario_count']} 个情景的选择和预算可行数完全一致**。"
            if parity.get("all_match")
            else f"与既有 M0 前沿核对：未执行（{parity.get('reason', '不适用')}）。"
        ),
        "",
        "| 上下文 | 预算 FLOPs | M1 可选点 | 最优 run_id | $N_B$ | $D_B$ | 触边维度 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        hits = [
            name for name, flag in (
                ("N_min", row["hit_N_min"]),
                ("N_max", row["hit_N_max"]),
                ("D_min", row["hit_D_min"]),
                ("D_max", row["hit_D_max"]),
            ) if flag
        ]
        lines.append(
            f"| {row['context_length_tokens']:,} | {row['budget_flops']:.0e} | "
            f"{row['m1_eligible_candidate_count']:,}/{row['candidate_count']:,} | "
            f"{row['m1_selected_run_id']} | {row['selected_N_B']:.6g} | {row['selected_D_B']:.6g} | "
            f"{', '.join(hits) if hits else '无'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "1. 当前候选网格为 B1 实测的 1,176 个 N/D 组合，故本次没有凸包内非观测候选；最优点均在实际离散支持域中。",
            "2. $10^{24}$ FLOPs 情景中所有支持点均可行，且最优配置触及支持域上界；结果受当前观测范围约束，不外推到更大的 N/D。",
            "3. 这是固定 Q 的 M0 参考优化，不代表完整 Q/p 联合最优，也不构成独立外部验证。",
            "",
            "复现命令：`python Q3/03_代码/solve_q3_m1_from_masks.py`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    inputs = [
        ROOT / M0_FRONTIER,
        ROOT / M3_CANDIDATES,
        ROOT / M3_SUPPORT,
        ROOT / M4_MASK,
        ROOT / M3_MANIFEST,
        ROOT / M4_MANIFEST,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run the M3/M4 mask builder first; missing:\n- " + "\n- ".join(missing))

    candidates, support_by_id, masks = load_interfaces()
    results = solve_m1(candidates, support_by_id, masks)
    m3_manifest = json.loads((ROOT / M3_MANIFEST).read_text(encoding="utf-8"))
    m4_manifest = json.loads((ROOT / M4_MANIFEST).read_text(encoding="utf-8"))
    m3_source = m3_manifest.get("scope", {}).get("candidate_grid_source", "")
    m4_source = m4_manifest.get("scope", {}).get("candidate_grid_source", "")
    default_grid = (
        m3_source == "Q2 B1 observed 1,176-point grid; matches the existing M0 search domain"
        and m4_source == m3_source
    )
    parity = check_existing_frontier(results, default_grid)
    write_csv(ROOT / OUT_BOUNDARY, results)
    (ROOT / OUT_REPORT).write_text(render_report(results, parity), encoding="utf-8")

    input_hashes = {relative(path): sha256_file(path) for path in inputs}
    outputs = [OUT_BOUNDARY, OUT_REPORT]
    manifest = {
        "artifact": "Q3 M1 masked optimization and boundary audit",
        "command": "python Q3/03_代码/solve_q3_m1_from_masks.py",
        "script": {"path": relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "python_version": platform.python_version(),
        "inputs_sha256": input_hashes,
        "outputs_sha256": {relative(ROOT / path): sha256_file(ROOT / path) for path in outputs},
        "objective": "minimize frozen B1/M0 predicted validation loss",
        "eligibility": "M3 m3_search_allowed AND M4 budget_feasible",
        "parity_with_existing_m0_frontier": parity,
        "claim_boundary": "fixed-Q B1/M0 support-grid reference; no interpolation or Q/p response inferred",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ROOT / OUT_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M1 eligible scenarios: {len(results)}")
    if parity.get("all_match"):
        print("M1 selections and M4 feasible counts match the existing frontier in all 15 scenarios")
    for path in outputs + [OUT_MANIFEST]:
        print(f"Wrote {path.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
