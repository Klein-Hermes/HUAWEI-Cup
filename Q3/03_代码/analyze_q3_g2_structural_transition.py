"""Recompute the finite-support budget-limitation transition evidence for Q3 G2.

Only Python's standard library is required. Costs are evaluated with exact
fractions from the decimal values recorded in the source CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Q3" / "04_结果" / "G2_结构性转移判定_20260926"
P1_OUTPUT = ROOT / "tmp" / "g2_structural_transition_p1_20260926"
INPUTS = {
    "bootstrap_parameters": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "cluster_bootstrap_estimates.csv",
    "b1_candidate_grid": ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b1_fitted_predictions.csv",
    "m0_reference_frontier": ROOT / "Q3" / "04_结果" / "M0_ND_reference_frontier.csv",
    "m0_bootstrap_frontier": ROOT / "Q3" / "04_结果" / "M0_ND_cluster_bootstrap_frontier.csv",
    "m0_transition_draws": ROOT / "Q3" / "04_结果" / "M0_discrete_transition_bootstrap_draws.csv",
    "support_saturation": ROOT / "Q3" / "04_结果" / "预算饱和与有限网格切换" / "q3_budget_saturation_by_context.csv",
}
BUDGETS = (10**19, 10**22, 10**24)
CONTEXTS = (2048, 4096, 8192, 32768, 131072)
BOOTSTRAP_LIMIT = 1000
DECISION_THRESHOLD = Decimal("0.95")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc


def parse_bool(value: str, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} is not a boolean: {value!r}")


def decimal(value: str, label: str) -> Decimal:
    try:
        result = Decimal(value)
    except Exception as exc:  # Decimal raises several conversion exceptions.
        raise ValueError(f"{label} is not a decimal: {value!r}") from exc
    require(result.is_finite(), f"{label} must be finite")
    return result


def exact_integer_count(value: str, label: str) -> int:
    """Convert a B1 value in billions to an exact count of units."""
    units = Fraction(Decimal(value)) * 10**9
    require(units.denominator == 1, f"{label} does not map to an integer count: {value}")
    return units.numerator


def exact_cost(n_parameters: int, d_tokens: int, context: int) -> int:
    """C=6ND+2e-4 NDL, using exact arithmetic and actual counts."""
    result = Fraction(6 * n_parameters * d_tokens) + Fraction(
        n_parameters * d_tokens * context, 5000
    )
    require(result.denominator == 1, f"cost is unexpectedly non-integral: {result}")
    return result.numerator


def rate_text(numerator: int, denominator: int) -> str:
    return f"{Decimal(numerator) / Decimal(denominator):.6f}"


def run(output_dir: Path, p1: bool) -> dict[str, Any]:
    for name, path in INPUTS.items():
        require(path.is_file(), f"missing input {name}: {path}")

    grid = read_csv(INPUTS["b1_candidate_grid"])
    require(len(grid) == 1176, f"expected 1176 B1 candidates, got {len(grid)}")
    candidates: list[dict[str, Any]] = []
    run_id_seen: set[int] = set()
    for row in grid:
        run_id = parse_int(row["run_id"], "B1 run_id")
        require(run_id not in run_id_seen, f"duplicate B1 run_id {run_id}")
        run_id_seen.add(run_id)
        candidates.append(
            {
                "run_id": run_id,
                "n_b": decimal(row["N_params_B"], "N_params_B"),
                "d_b": decimal(row["D_tokens_B"], "D_tokens_B"),
                "n": exact_integer_count(row["N_params_B"], "N_params_B"),
                "d": exact_integer_count(row["D_tokens_B"], "D_tokens_B"),
            }
        )
    max_n = max(row["n"] for row in candidates)
    max_d = max(row["d"] for row in candidates)
    joint_max = [
        row for row in candidates if row["n"] == max_n and row["d"] == max_d
    ]
    require(len(joint_max) == 1, "the B1 support must contain a unique joint maximum (N,D)")
    optimum = joint_max[0]

    parameters = read_csv(INPUTS["bootstrap_parameters"])
    require(len(parameters) == BOOTSTRAP_LIMIT, f"expected 1000 parameter draws, got {len(parameters)}")
    parameter_by_rep: dict[int, dict[str, Any]] = {}
    minima = {name: Decimal("Infinity") for name in ("A", "alpha", "B", "beta")}
    for row in parameters:
        rep = parse_int(row["replicate"], "parameter replicate")
        require(rep not in parameter_by_rep, f"duplicate parameter replicate {rep}")
        values = {name: decimal(row[name], name) for name in minima}
        for name, value in values.items():
            require(value > 0, f"non-positive {name} in parameter replicate {rep}")
            minima[name] = min(minima[name], value)
        parameter_by_rep[rep] = values
    require(sorted(parameter_by_rep) == list(range(1, BOOTSTRAP_LIMIT + 1)),
            "parameter replicate ids must be exactly 1..1000")
    replicate_limit = 100 if p1 else BOOTSTRAP_LIMIT
    selected_replicates = set(range(1, replicate_limit + 1))
    selected_contexts = (CONTEXTS[0],) if p1 else CONTEXTS

    saturation_rows = {
        parse_int(row["context_length_tokens"], "context length"): row
        for row in read_csv(INPUTS["support_saturation"])
    }
    require(set(CONTEXTS).issubset(saturation_rows), "support saturation table lacks a C7 context")

    reference_rows = read_csv(INPUTS["m0_reference_frontier"])
    reference: dict[tuple[int, int], dict[str, str]] = {}
    for row in reference_rows:
        key = (parse_int(row["context_length_tokens"], "reference context"),
               parse_int(row["budget_flops"], "reference budget"))
        require(key not in reference, f"duplicate reference scenario {key}")
        reference[key] = row
    frontier_rows = read_csv(INPUTS["m0_bootstrap_frontier"])
    frontier: dict[tuple[int, int], dict[str, str]] = {}
    for row in frontier_rows:
        key = (parse_int(row["context_length_tokens"], "bootstrap context"),
               parse_int(row["budget_flops"], "bootstrap budget"))
        require(key not in frontier, f"duplicate bootstrap frontier scenario {key}")
        frontier[key] = row

    for context in CONTEXTS:
        costs = [exact_cost(row["n"], row["d"], context) for row in candidates]
        csat = max(costs)
        support = saturation_rows[context]
        recorded = parse_int(support["C_sat_support_grid_flops"], "recorded C_sat")
        require(recorded == csat, f"C_sat mismatch at context={context}: recomputed={csat}, recorded={recorded}")
        require(parse_int(support["candidate_count"], "candidate count") == len(candidates),
                f"candidate count mismatch at context={context}")
        for budget in BUDGETS:
            ref_key = (context, budget)
            require(ref_key in reference and ref_key in frontier, f"missing scenario {ref_key}")
            ref = reference[ref_key]
            summary = frontier[ref_key]
            require(parse_int(summary["bootstrap_replicates"], "frontier bootstrap count") == BOOTSTRAP_LIMIT,
                    f"frontier bootstrap count mismatch for {ref_key}")
            require(parse_int(summary["unique_selected_grid_points"], "unique selected points") == 1,
                    f"selected point is not unique across bootstrap for {ref_key}")
            require(decimal(summary["baseline_selection_frequency"], "baseline frequency") == 1,
                    f"baseline selection frequency is not 1 for {ref_key}")
            require(decimal(summary["modal_selection_frequency"], "modal frequency") == 1,
                    f"modal frequency is not 1 for {ref_key}")
            require(parse_int(ref["selected_run_id"], "reference selected run") ==
                    parse_int(summary["baseline_selected_run_id"], "bootstrap baseline run"),
                    f"reference and bootstrap baseline disagree for {ref_key}")

    transition_groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(INPUTS["m0_transition_draws"]):
        if row["pair_type"] != "adjacent_budget":
            continue
        context = parse_int(row["left_context_tokens"], "left context")
        right_context = parse_int(row["right_context_tokens"], "right context")
        require(context == right_context, f"adjacent budget pair changes context in {row['pair_id']}")
        key = (context, parse_int(row["left_budget_flops"], "left budget"),
               parse_int(row["right_budget_flops"], "right budget"))
        transition_groups[key].append(row)

    summary_output: list[dict[str, Any]] = []
    for context in selected_contexts:
        csat = max(exact_cost(row["n"], row["d"], context) for row in candidates)
        for left_budget, right_budget in zip(BUDGETS, BUDGETS[1:]):
            edge = (context, left_budget, right_budget)
            draws = transition_groups.get(edge, [])
            require(len(draws) == BOOTSTRAP_LIMIT, f"expected 1000 paired selection draws for {edge}, got {len(draws)}")
            draw_by_rep: dict[int, dict[str, str]] = {}
            for row in draws:
                rep = parse_int(row["bootstrap_replicate"], "selection bootstrap replicate")
                require(rep not in draw_by_rep, f"duplicate selection draw {edge}, replicate={rep}")
                draw_by_rep[rep] = row
            require(sorted(draw_by_rep) == list(range(1, BOOTSTRAP_LIMIT + 1)),
                    f"selection bootstrap ids must be 1..1000 for {edge}")
            selected_draws = [draw_by_rep[rep] for rep in sorted(selected_replicates)]
            require(len(selected_draws) == replicate_limit, f"wrong P1/full slice size for {edge}")

            left_state = int(csat > left_budget)
            right_state = int(csat > right_budget)
            state_switches = sum(left_state != right_state for _ in selected_replicates)
            config_changes = 0
            cost_share_changes = 0
            for row in selected_draws:
                config_changes += int(parse_bool(row["configuration_changed"], "configuration_changed"))
                cost_share_changes += int(decimal(row["total_variation_cost_share"], "total variation") > 0)
                left_run = parse_int(row["left_run_id"], "left run")
                right_run = parse_int(row["right_run_id"], "right run")
                for budget, draw_run in ((left_budget, left_run), (right_budget, right_run)):
                    modal = parse_int(frontier[(context, budget)]["modal_selected_run_id"], "modal selected run")
                    require(draw_run == modal, f"raw draw and bootstrap frontier disagree for {edge}, rep={row['bootstrap_replicate']}")

            right_ref = reference[(context, right_budget)]
            right_frontier = frontier[(context, right_budget)]
            right_boundary_hit = parse_bool(
                right_ref["selected_at_support_max_N_and_D"], "selected_at_support_max_N_and_D"
            )
            if right_budget == BUDGETS[-1] and right_state == 0:
                require(parse_int(right_ref["selected_run_id"], "high-budget selected run") == optimum["run_id"],
                        f"high-budget point is not the joint support maximum for context={context}")
                require(right_boundary_hit, f"high-budget boundary consistency check failed for context={context}")
                require(parse_int(right_frontier["modal_selected_run_id"], "high-budget modal run") == optimum["run_id"],
                        f"high-budget bootstrap mode is not the joint support maximum for context={context}")

            state_transition = Decimal(state_switches) / Decimal(replicate_limit) >= DECISION_THRESHOLD
            config_rate = Decimal(config_changes) / Decimal(replicate_limit)
            if state_transition:
                label = "有限网格预算限制状态转换"
            elif config_rate >= DECISION_THRESHOLD:
                label = "仅离散配置切换；预算限制状态未转换"
            else:
                label = "未达到预先设定的结构性转移门槛"
            summary_output.append(
                {
                    "pair_id": draws[0]["pair_id"],
                    "context_length_tokens": context,
                    "left_budget_flops": left_budget,
                    "right_budget_flops": right_budget,
                    "C_sat_support_grid_flops": csat,
                    "left_budget_limited_state": left_state,
                    "right_budget_limited_state": right_state,
                    "bootstrap_replicates_used": replicate_limit,
                    "budget_state_switches": state_switches,
                    "budget_state_switch_rate": rate_text(state_switches, replicate_limit),
                    "configuration_changes": config_changes,
                    "configuration_change_rate": rate_text(config_changes, replicate_limit),
                    "cost_share_changes": cost_share_changes,
                    "cost_share_change_rate": rate_text(cost_share_changes, replicate_limit),
                    "right_endpoint_selected_at_joint_support_max": right_boundary_hit,
                    "right_endpoint_selected_run_id": right_ref["selected_run_id"],
                    "right_endpoint_bootstrap_unique_grid_points": right_frontier["unique_selected_grid_points"],
                    "decision_threshold": str(DECISION_THRESHOLD),
                    "classification": label,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "g2_transition_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_output[0].keys()))
        writer.writeheader()
        writer.writerows(summary_output)

    report_path = output_dir / "report.md"
    positive_line = ", ".join(f"{name}={minima[name]:.6f}" for name in ("A", "alpha", "B", "beta"))
    transitions = [row for row in summary_output if "状态转换" in row["classification"]]
    config_only = [row for row in summary_output if row["classification"].startswith("仅离散")]
    result_scope = (
        f"该 P1 试运行切片覆盖上下文 {selected_contexts[0]}、前 {replicate_limit} 个 Bootstrap 复本和两段预算区间；"
        if p1 else "完整复算覆盖题面五个上下文、全部 1,000 个参数 Bootstrap 复本和十段相邻预算比较。"
    )
    report_lines = [
        "# Q3-G2 有限网格预算限制状态判定",
        "",
        f"计算模式：{'P1 试运行' if p1 else '完整复算'}；参数 Bootstrap 复本数：{replicate_limit}；上下文：{', '.join(map(str, selected_contexts))}。",
        "",
        "## 判据",
        "",
        "对每个参数 Bootstrap 复本，先在固定 Q=Q0 的 M0/B1 损失模型下确定 1,176 点有限支持域中的无预算最优候选，再定义预算限制状态为 1{该候选成本 > 预算}。配对状态转换率达到 95% 时，判为有限网格预算限制状态转换。预算后命中联合最大支持点只作一致性校验；单独的 N/D 配置改变不计为该转移。",
        "",
        "B1 的 N_params_B、D_tokens_B 以十亿为单位用于缩放律损失；成本按实际参数数和 Token 数计算。精确成本为 C=6ND+NDL/5000 FLOPs。无预算最优点唯一性由所有 1,000 组中的 A、alpha、B、beta 正值及支持域唯一联合最大点保证。参数正值最小值：" + positive_line + ".",
        "",
        "## 定量判定",
        "",
        "| 上下文 L | 预算区间 (FLOPs) | C_sat (FLOPs) | 状态 1→0 复本率 | N/D 配置改变率 | 成本份额改变率 | 判定 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_output:
        report_lines.append(
            f"| {row['context_length_tokens']} | {row['left_budget_flops']:.3g}→{row['right_budget_flops']:.3g} | "
            f"{row['C_sat_support_grid_flops']} | {row['budget_state_switches']}/{row['bootstrap_replicates_used']} "
            f"({row['budget_state_switch_rate']}) | {row['configuration_changes']}/{row['bootstrap_replicates_used']} "
            f"({row['configuration_change_rate']}) | {row['cost_share_changes']}/{row['bootstrap_replicates_used']} "
            f"({row['cost_share_change_rate']}) | {row['classification']} |"
        )
    report_lines += [
        "",
        "## 结论范围",
        "",
        result_scope + f"其中 {len(transitions)} 个相邻预算比较达到有限网格预算限制状态转换门槛；{len(config_only)} 个比较仅出现稳健 N/D 配置切换而状态不变。转换若发生在中到高预算之间，表示预算从排除有限支持域无预算最优点转为不再限制该点。高预算联合最大点命中是与判据一致的校验结果，不作为额外独立证据。",
        "",
        "1000/1000 表示冻结的参数 Bootstrap 复本集合中的稳定性，不是 1,000 个独立训练实验；该比例不覆盖模型形式误差或支持域外行为。",
        "",
        "本结论只适用于固定 Q=Q0、B1 共同训练配比政策、冻结 M0 损失与 1,176 个已观测候选点。它不识别支持域之外的规模，也不构成完整 Q/p 联合最优模型的经验判定。B6 与 B8 的 Q 响应方向冲突且 Q 映射尚未验证，因此不能据此给出统一的 Q/p 结构转移结论。",
        "",
        "成本份额变化率只按已有固定-Q M0 配对表中的 total_variation_cost_share 计算；Q=Q0 时质量成本项为零，不从该结果推断未知 Q 下的成本结构。",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    source_hashes = {name: sha256(path) for name, path in INPUTS.items()}
    outputs = {summary_path.name: sha256(summary_path), report_path.name: sha256(report_path)}
    manifest = {
        "artifact": "Q3-G2 finite-support budget-limitation-state transition analysis",
        "mode": "p1_smoke" if p1 else "full",
        "scope": "fixed_Q0_M0_B1_policy_1176_point_observed_support",
        "formula": "loss=E+A*n^(-alpha)+B*d^(-beta), with n=N_params_B and d=D_tokens_B in billions; cost=6*N*D+N*D*L/5000 using actual counts",
        "decision_threshold": str(DECISION_THRESHOLD),
        "bootstrap_replicates_used": replicate_limit,
        "bootstrap_replicates_available": BOOTSTRAP_LIMIT,
        "contexts_used": list(selected_contexts),
        "budgets_flops": list(BUDGETS),
        "candidate_count": len(candidates),
        "unique_joint_max_support_point": {
            "run_id": optimum["run_id"],
            "N_params_B": str(optimum["n_b"]),
            "D_tokens_B": str(optimum["d_b"]),
        },
        "bootstrap_positive_parameter_minima": {k: str(v) for k, v in minima.items()},
        "input_sha256": source_hashes,
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": outputs,
        "python": platform.python_version(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", action="store_true", help="run one-context, 100-replicate smoke slice")
    parser.add_argument("--output-dir", type=Path, help="write outputs to a scoped directory")
    args = parser.parse_args()
    out = args.output_dir or (P1_OUTPUT if args.p1 else DEFAULT_OUTPUT)
    manifest = run(out.resolve(), args.p1)
    print(json.dumps({
        "mode": manifest["mode"],
        "output_dir": str(out.resolve()),
        "contexts_used": manifest["contexts_used"],
        "bootstrap_replicates_used": manifest["bootstrap_replicates_used"],
        "manifest_sha256": sha256(out.resolve() / "manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
